"""MissForest-style time series imputer.

This class formalizes the quick prototype shown in the scratch notebook where an
XGBModel is trained on a time series containing missing values by:

1. Replacing missing target values with a large out-of-domain *indicator* value.
2. Supplying a `sample_weight` TimeSeries with weight 0 for originally-missing
   timestamps (so the model effectively ignores the sentinel values) and 1 otherwise.
3. Running `historical_forecasts()` (with `retrain=False`) once to obtain one-step
   ahead forecasts over the history.
4. Using the model's predictions only at the timestamps that were originally
   missing, and keeping original observed values elsewhere.

This is *not* a faithful implementation of the original MissForest iterative
random forest imputation algorithm; rather it is a light-weight, model-agnostic
utility for single-pass imputation that mimics that workflow using any Darts
global forecasting model supporting the `sample_weight` argument in `fit()`.

Limitations / assumptions:
* Currently supports univariate TimeSeries only.
* Missing values at the very beginning (before the model's minimum lags allow
  forecasting) remain un-imputed (stay NaN) unless `fallback_strategy` is used.
* No iterative refinement (single pass only).

Future extensions could include: iterative refinement, multivariate handling,
and support for alternative fallback strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Sequence, List

import numpy as np
import pandas as pd
import warnings

from darts import TimeSeries

try:  # Optional import; users may pass in a different model.
	from darts.models import XGBModel
except Exception:  # pragma: no cover - optional dependency not present
	XGBModel = None  # type: ignore


class MissForestImputer:
	"""Single-pass model-based imputer for univariate TimeSeries.

	Parameters
	----------
	model : optional
		A Darts forecasting model instance. If not supplied, an ``XGBModel``
		will be created using the provided ``lags`` and ``forecast_horizon``
		(as its output_chunk_length). Requires xgboost optional dependency.
	lags : int, optional
		Number of past lags passed to the default model. If not provided when creating the
		default model (``model`` is None), it will default to ``lookback`` (or 24 if both are omitted).
		indicator_value : float, default -1e6
		Value used to temporarily fill missing targets before model training.
		Should be far outside the normal data range.
	forecast_horizon : int, default 1
		Horizon used when generating historical forecasts. Values > 1 will
		result in multi-step-ahead predictions being available for imputation.
	fallback_strategy : {None, 'ffill', 'zero', 'mean'}, default None
		Strategy applied to any leading NaNs that cannot be imputed because the
		model lacks sufficient history. If ``None`` they remain NaN.
		lookback : int or None, default None
		Number of padding steps prepended before the original series. If provided and ``lags`` is
		omitted for the default model, ``lags`` will inherit this value. If both ``lags`` and
		``lookback`` are omitted, a fallback of 24 lags and 24 lookback (padding) is used.
	"""

	def __init__(
		self,
		model: Optional[object] = None,
		*,
		lags: Optional[int] = None,
		indicator_value: float = -1e6,
		forecast_horizon: int = 1,
		fallback_strategy: Optional[str] = None,
		n_rounds: int = 1,
		convergence_tol: float = 0.0,
		lookback: Optional[int] = None,
		lookahead: int = 0,
	) -> None:

		if model is None:
			# Derive lags from lookback if not explicitly given
			if lags is None:
				if lookback is not None:
					lags = lookback
				else:
					lags = 24  # sensible default
			if XGBModel is None:
				raise ImportError(
					"Default XGBModel is unavailable (xgboost not installed). "
					"Install optional dependencies or pass a custom model instance."
				)
			model = XGBModel(lags=lags, output_chunk_length=forecast_horizon)
		if fallback_strategy not in {None, "ffill", "zero", "mean"}:
			raise ValueError("fallback_strategy must be one of {None, 'ffill', 'zero', 'mean'}")

		self.model = model
		self.indicator_value = indicator_value
		self.forecast_horizon = forecast_horizon
		self.fallback_strategy = fallback_strategy
		self._is_fitted = False
		self._training_nan_mask: Optional[np.ndarray] = None  # original mask only
		self._training_index = None
		self.n_rounds = int(n_rounds)
		if self.n_rounds < 1:
			raise ValueError("n_rounds must be >= 1")
		self._final_filled_values: Optional[np.ndarray] = None
		self._imputed_series: Optional[TimeSeries] = None
		self.convergence_tol = float(convergence_tol)
		# Finalize lookback: if still None, use lags (from model if available) else 0
		if lookback is None:
			if lags is not None:
				self.lookback = int(lags)
			else:
				model_lags = getattr(self.model, "lags", 0)
				self.lookback = int(model_lags) if isinstance(model_lags, int) else 0
		else:
			self.lookback = int(lookback)
		self.lookahead = int(lookahead)
		if self.lookback < 0 or self.lookahead < 0:
			raise ValueError("lookback and lookahead must be >= 0")

	def fit(self, series: TimeSeries) -> "MissForestImputer":
		"""Fits the underlying forecasting model ignoring missing values.

		Missing values are replaced by the sentinel and assigned weight 0.
		"""
		if series.width > 1:
			raise ValueError("MissForestImputer currently supports univariate series only.")

        # Base values and masks on original series
		values_orig = series.values(copy=False).squeeze()
		orig_nan_mask = np.isnan(values_orig)
		self._training_nan_mask = orig_nan_mask.copy()
		self._training_index = series.time_index

		# Derive frequency
		if series.time_index.freq is not None:
			freq = series.time_index.freq
		else:
			freq = series.time_index[1] - series.time_index[0]

		total_len = self.lookback + len(series) + self.lookahead
		full_start = series.time_index[0] - self.lookback * freq if self.lookback > 0 else series.time_index[0]
		full_index = pd.date_range(start=full_start, periods=total_len, freq=freq)

		# Initialize full values with indicator
		full_values = np.full(shape=(total_len,), fill_value=self.indicator_value, dtype=float)
		# Insert original segment
		orig_start = self.lookback
		orig_end = orig_start + len(series)
		full_values[orig_start:orig_end] = values_orig

		# Working filled array (start identical to full padded array)
		filled_values = full_values.copy()
		# Ensure original missing points inside window are set to indicator
		if orig_nan_mask.any():
			missing_local_idx = np.where(orig_nan_mask)[0] + orig_start
			filled_values[missing_local_idx] = self.indicator_value

		# Sample weights: 0 for padded zones, 1 for observed original non-missing, 0 for original missing initially
		weights_full = np.zeros_like(full_values)
		obs_mask_orig = (~orig_nan_mask)
		weights_full[orig_start:orig_end][obs_mask_orig] = 1.0
		prev_imputed = None

		# Helper mask for original window indices in full array corresponding to missing
		orig_missing_indices_full = np.where(orig_nan_mask)[0] + orig_start

		for round_idx in range(self.n_rounds):
			# Always re-assert padding zones remain indicator (avoid accidental overwrite)
			if self.lookback > 0:
				filled_values[:orig_start] = self.indicator_value
			if self.lookahead > 0:
				filled_values[orig_end:] = self.indicator_value
			filled_ts = TimeSeries.from_times_and_values(full_index, filled_values)
			weight_ts = TimeSeries.from_times_and_values(full_index, weights_full)
			try:
				self.model.fit(filled_ts, sample_weight=weight_ts)
			except TypeError as e:
				raise TypeError(
					"Provided model does not accept 'sample_weight' in fit(); pass a compatible Darts model."
				) from e
			# One-step forecasts for refinement of currently missing positions
			hf = self.model.historical_forecasts(
				series=filled_ts,
				retrain=False,
				forecast_horizon=1,
				verbose=False,
				last_points_only=True,
			)
			pred_pd = hf.to_series()
			# Only consider predictions for original missing timestamps
			orig_missing_times = full_index[orig_missing_indices_full]
			update_index = pred_pd.index.intersection(orig_missing_times)
			if len(update_index) > 0:
				pos_full = full_index.get_indexer(update_index)
				filled_values[pos_full] = pred_pd.loc[update_index].values
				# Include these in training from next round
				weights_full[pos_full] = 1.0

			# Convergence check (only if tolerance > 0 and at least one update)
			if self.convergence_tol > 0 and len(update_index) > 0:
				current_imputed = filled_values[orig_missing_indices_full]
				if prev_imputed is not None:
					# Use relative mean absolute change
					diff = np.nanmean(np.abs(current_imputed - prev_imputed))
					baseline = max(np.nanmean(np.abs(prev_imputed)), 1e-12)
					rel_change = diff / baseline
					if rel_change <= self.convergence_tol:
						break
				prev_imputed = current_imputed.copy()

		self._final_filled_values = filled_values
		# Build and store imputed TimeSeries (crop to original window)
		imputed_segment = filled_values[orig_start:orig_end]
		orig_series = series.to_series(copy=True)
		# Replace only originally missing points
		missing_pos = np.where(orig_nan_mask)[0]
		if missing_pos.size > 0:
			orig_series.iloc[missing_pos] = imputed_segment[missing_pos]
		# Any indicator values persisting become NaN
		persist_mask = orig_series.iloc[missing_pos] == self.indicator_value
		if persist_mask.any():
			idxs = orig_series.index[missing_pos][persist_mask]
			orig_series.loc[idxs] = np.nan
		self._imputed_series = TimeSeries.from_series(orig_series)
		self._imputed_series = TimeSeries.from_series(orig_series)

		self._is_fitted = True
		return self

	def transform(self, series: TimeSeries) -> TimeSeries:
		"""Returns the imputed series produced in fit()."""
		if not self._is_fitted:
			raise RuntimeError("Call fit() before transform().")
		if not series.time_index.equals(self._training_index):
			raise ValueError("Series passed to transform must be the same (or aligned) as during fit().")
		if self._imputed_series is None:
			raise RuntimeError("Internal error: imputed series missing.")
		result = self._imputed_series
		if self.fallback_strategy is not None:
			pd_series = result.to_series(copy=True)
			if self.fallback_strategy == "ffill":
				pd_series = pd_series.ffill()
			elif self.fallback_strategy == "zero":
				pd_series = pd_series.fillna(0.0)
			elif self.fallback_strategy == "mean":
				pd_series = pd_series.fillna(pd_series.mean())
			result = TimeSeries.from_series(pd_series)
		return result

	def fit_transform(self, series: TimeSeries) -> TimeSeries:
		"""Fit and return final imputed series (no extra recomputation)."""
		self.fit(series)
		return self.transform(series)

