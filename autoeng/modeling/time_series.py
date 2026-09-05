"""
Time-series forecasting: reframed as supervised regression on lag/rolling/
calendar features, evaluated with expanding-window (TimeSeriesSplit) cross-
validation instead of ordinary K-fold — a random split would let a model
train on rows that come chronologically after the ones it's validated on,
which is exactly the temporal-leakage failure mode the leakage detector
also checks for defensively. This is the prevention side of that check.

Three things this module does that a naive lag-feature setup doesn't:

1. SEASONAL PERIOD DETECTION. The seasonal lag to use is discovered from
   the data's autocorrelation function, not hard-coded to 7 or 12. A
   weekly series gets lag 52, a daily one gets 7, an aperiodic one gets
   none and simply skips seasonal features.

2. DIFFERENCING FOR NON-STATIONARY SERIES. This is the fix for the single
   biggest weakness found in testing: on both a random walk and real CO2
   data, the naive "predict the previous value" baseline beat every ML
   model. That happens because a strongly trending series has almost all
   its variance in the LEVEL, so a model fitted on levels spends its
   capacity re-deriving "tomorrow ≈ today" and the extra features only add
   noise. Modelling the first DIFFERENCE instead makes the target
   stationary, turns the naive forecast into the trivial "predict zero
   change", and leaves the model to learn only the part that's actually
   learnable. Predictions are reconstructed back onto the original scale
   (ŷ_t = y_{t-1} + Δ̂_t) before scoring, so every number stays directly
   comparable to the baselines.

3. FAIR BASELINES. Naive/seasonal-naive/moving-average are evaluated
   one-step-ahead on the same folds, at the same information level the ML
   models get.

NOT done, deliberately: STL trend/seasonal components as features. Fitting
STL on the whole series and using its components at time t leaks future
information into the past — the decomposition at every point is computed
using the entire series. Doing it correctly requires re-fitting the
decomposition inside each training fold, which this architecture (lag frame
built once, before CV) doesn't currently support. Shipping the leaky version
would have "improved" the scores by cheating.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.features.pipeline_builder import build_preprocessing_pipeline
from autoeng.modeling.model_zoo import SCALE_SENSITIVE_MODELS, get_regression_models
from autoeng.modeling.search import TREE_LIKE_MODELS, ModelResult

BASE_LAGS = (1, 2, 3)
DEFAULT_ROLLING_WINDOWS = (3, 7)
# Lag-1 autocorrelation at or above this means the level series is dominated by
# its own trend (random-walk-like) and should be modelled in differences.
NON_STATIONARY_AUTOCORR = 0.9
MIN_SEASONAL_PERIOD = 2
ACF_PEAK_THRESHOLD = 0.2


def detect_seasonal_period(y: pd.Series, max_period: int | None = None) -> tuple[int | None, str]:
    """
    Find the dominant seasonal period from the autocorrelation function of the
    DIFFERENCED series (differencing first strips the trend, which would
    otherwise dominate the ACF and mask any real seasonality).

    Returns (period, reason); period is None when nothing periodic stands out.
    """
    values = pd.to_numeric(y, errors="coerce").dropna()
    n = len(values)
    if n < 20:
        return None, "series too short for seasonality detection"

    max_period = max_period or min(n // 3, 400)
    if max_period < MIN_SEASONAL_PERIOD:
        return None, "series too short for a meaningful seasonal period"

    try:
        from statsmodels.tsa.stattools import acf
        differenced = values.diff().dropna()
        if differenced.std() == 0:
            return None, "differenced series is constant"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlations = acf(differenced, nlags=max_period, fft=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"ACF computation failed ({type(exc).__name__})"

    # Ignore lags 0 and 1 — lag 1 on a differenced series reflects the
    # differencing itself, not seasonality.
    candidate_lags = np.arange(MIN_SEASONAL_PERIOD, len(correlations))
    if len(candidate_lags) == 0:
        return None, "no candidate lags available"
    peak_lag = int(candidate_lags[np.argmax(correlations[MIN_SEASONAL_PERIOD:])])
    peak_value = float(correlations[peak_lag])

    if peak_value < ACF_PEAK_THRESHOLD:
        return None, f"strongest ACF peak (lag {peak_lag}, {peak_value:.2f}) below {ACF_PEAK_THRESHOLD} — treating as aperiodic"
    return peak_lag, f"ACF of the differenced series peaks at lag {peak_lag} ({peak_value:.2f})"


def should_difference(y: pd.Series) -> tuple[bool, str]:
    """Decide whether to model the level or the first difference."""
    values = pd.to_numeric(y, errors="coerce").dropna()
    if len(values) < 10 or values.std() == 0:
        return False, "series too short or constant; modelling levels"

    a, b = values.iloc[:-1].to_numpy(), values.iloc[1:].to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return False, "no variation; modelling levels"
    autocorr = float(np.corrcoef(a, b)[0, 1])

    if autocorr >= NON_STATIONARY_AUTOCORR:
        return True, (
            f"lag-1 autocorrelation {autocorr:.3f} >= {NON_STATIONARY_AUTOCORR}: the series is "
            "dominated by its own level (random-walk-like), so the model is fitted on first "
            "differences and predictions are reconstructed as previous value + predicted change"
        )
    return False, f"lag-1 autocorrelation {autocorr:.3f} is low enough to model levels directly"


def build_lag_feature_frame(
    df: pd.DataFrame, target_column: str, time_column: str,
    exogenous_columns: list[str], seasonal_period: int | None = None,
    lags=BASE_LAGS, rolling_windows=DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Sort by time, then add causal (past-only) features of the target: plain
    lags, rolling statistics, first differences, and — when a seasonal period
    was detected — the seasonal lag and seasonal difference.

    Every rolling/aggregate feature is computed on a series shifted by one, so
    no row ever sees its own value. Rows without enough history are dropped;
    there is no honest way to fill them, because that data genuinely would not
    exist at prediction time.
    """
    ordered = df.sort_values(time_column).reset_index(drop=True)
    target = ordered[target_column]
    out = ordered[[time_column, target_column] + [c for c in exogenous_columns if c in ordered.columns]].copy()

    all_lags = list(lags)
    if seasonal_period and seasonal_period not in all_lags:
        all_lags.append(seasonal_period)

    for lag in all_lags:
        out[f"target_lag_{lag}"] = target.shift(lag)

    shifted = target.shift(1)  # never include the current row in its own statistics
    for window in rolling_windows:
        out[f"target_rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"target_rolling_std_{window}"] = shifted.rolling(window).std()

    # Momentum: how the series was already moving, using only past values.
    out["target_diff_1"] = shifted.diff(1)
    out["target_diff_2"] = shifted.diff(2)

    if seasonal_period:
        out[f"target_rolling_mean_{seasonal_period}"] = shifted.rolling(seasonal_period).mean()
        out["target_seasonal_diff"] = shifted - target.shift(seasonal_period + 1)

    max_history = max(max(all_lags), max(rolling_windows) + 1, (seasonal_period or 0) + 2)
    return out.iloc[max_history:].reset_index(drop=True)


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TimeSeriesSetup:
    seasonal_period: int | None
    seasonality_reason: str
    differenced: bool
    differencing_reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _accumulate(store: dict[str, list[float]], y_true, y_pred) -> None:
    store["r2"].append(r2_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0)
    store["neg_rmse"].append(-float(np.sqrt(mean_squared_error(y_true, y_pred))))
    store["neg_mae"].append(-float(mean_absolute_error(y_true, y_pred)))


def _evaluate_baselines(y: np.ndarray, seasonal_period: int, cv: TimeSeriesSplit) -> list[BaselineResult]:
    """
    One-step-ahead baselines using the TRUE previous actual(s) at each test
    point — the same information level the lag-feature ML models get (their
    `target_lag_1` etc. are also the true prior actuals, not the model's own
    prior forecasts). Holding a single value constant across an entire test
    fold would be a multi-step-ahead forecast and *not* a fair comparison
    against 1-step-ahead ML models — it would make naive baselines look far
    worse than they really are and overstate the ML models' advantage.
    """
    naive_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}
    seasonal_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}
    ma_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}

    for _, test_idx in cv.split(y):
        y_test = y[test_idx]

        naive_pred = np.array([y[i - 1] for i in test_idx])
        _accumulate(naive_scores, y_test, naive_pred)

        seasonal_pred = np.array([y[i - seasonal_period] if i - seasonal_period >= 0 else y[i - 1] for i in test_idx])
        _accumulate(seasonal_scores, y_test, seasonal_pred)

        ma_pred = np.array([y[max(0, i - 7):i].mean() if i > 0 else y[0] for i in test_idx])
        _accumulate(ma_scores, y_test, ma_pred)

    return [
        BaselineResult("naive_last_value", {k: float(np.mean(v)) for k, v in naive_scores.items()}),
        BaselineResult("seasonal_naive", {k: float(np.mean(v)) for k, v in seasonal_scores.items()}),
        BaselineResult("moving_average_7", {k: float(np.mean(v)) for k, v in ma_scores.items()}),
    ]


def run_time_series_search(
    df: pd.DataFrame, target_column: str, time_column: str, roles: FeatureRoleAssignment,
    cv_folds: int = 5, seasonal_period: int | None = None,
) -> tuple[list[ModelResult], list[BaselineResult], TimeSeriesSetup]:
    ordered_target = df.sort_values(time_column)[target_column]
    detected_period, period_reason = (
        (seasonal_period, "seasonal period supplied by caller") if seasonal_period
        else detect_seasonal_period(ordered_target)
    )
    difference, difference_reason = should_difference(ordered_target)
    setup = TimeSeriesSetup(
        seasonal_period=detected_period, seasonality_reason=period_reason,
        differenced=difference, differencing_reason=difference_reason,
    )

    exogenous = [c for c in roles.feature_columns if c not in (target_column, time_column)]
    lag_frame = build_lag_feature_frame(df, target_column, time_column, exogenous, detected_period)

    y_level = lag_frame[target_column].to_numpy(dtype=float)
    previous_level = lag_frame["target_lag_1"].to_numpy(dtype=float)
    # What the model is fitted on: either the level itself, or the change since
    # the previous observation. Scoring always happens on the level.
    y_fit = (y_level - previous_level) if difference else y_level

    X = lag_frame.drop(columns=[target_column, time_column])

    from autoeng.common.roles import assign_feature_roles
    from autoeng.profiling.profiler import profile_dataset
    sub_profile = profile_dataset(X)
    sub_roles = assign_feature_roles(sub_profile, target_column=None, time_column=None)

    cv = TimeSeriesSplit(n_splits=cv_folds)
    models = get_regression_models()
    results: list[ModelResult] = []

    for name, factory in models.items():
        try:
            cap_outliers = name not in TREE_LIKE_MODELS
            pre = build_preprocessing_pipeline(sub_roles, problem_kind="regression",
                                                cap_outliers=cap_outliers, use_interactions=False)
            steps = list(pre.steps)
            if name in SCALE_SENSITIVE_MODELS:
                steps.append(("scale", StandardScaler()))
            steps.append(("model", factory()))
            pipe = Pipeline(steps)

            fold_metrics = {"r2": [], "neg_rmse": [], "neg_mae": []}
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_idx, test_idx in cv.split(X):
                    model = clone(pipe)
                    model.fit(X.iloc[train_idx], y_fit[train_idx])
                    raw_preds = model.predict(X.iloc[test_idx])
                    # Reconstruct onto the original scale so every model and
                    # baseline is scored against the same quantity.
                    preds = (previous_level[test_idx] + raw_preds) if difference else raw_preds
                    _accumulate(fold_metrics, y_level[test_idx], preds)
            elapsed = time.time() - t0
            results.append(ModelResult(
                name=name, status="ok",
                metrics={k: float(np.mean(v)) for k, v in fold_metrics.items()},
                fit_time_seconds=elapsed,
            ))
        except Exception as e:  # noqa: BLE001
            results.append(ModelResult(name=name, status="failed", error=f"{type(e).__name__}: {e}"))

    baselines = _evaluate_baselines(y_level, seasonal_period=detected_period or 7, cv=cv)
    return results, baselines, setup
