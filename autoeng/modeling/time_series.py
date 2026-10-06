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
   none and simply skips seasonal features. (Until T2-2 a weekly series got
   3: see `detect_seasonal_period`.)

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

3. FAIR BASELINES. Naive/seasonal-naive/moving-average are evaluated on the
   same folds, at the same horizon and information level the ML models get.

4. THE REAL HORIZON (T2-2). `horizon` sets how far ahead the forecast is
   made; every target feature and every baseline sees the series only up to
   the forecast origin (see `build_lag_feature_frame`). The old module scored
   everything one step ahead whatever the use.

5. FOLD-AWARE STL (T2-2). STL fitted on the whole series leaks: its value at
   every point depends on the entire series, which is why it was refused for
   so long. `_seasonal_index` fits it on one training fold's levels and turns
   it into a per-phase index that a future row can look up, since its phase is
   known in advance. It is offered to the three best models as `<name>+stl`
   and kept only where the same cross-validation says it helps.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
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
from autoeng.modeling.search import ModelResult

BASE_LAGS = (1, 2, 3)
DEFAULT_ROLLING_WINDOWS = (3, 7)
# Lag-1 autocorrelation at or above this means the level series is dominated by
# its own trend (random-walk-like) and should be modelled in differences.
NON_STATIONARY_AUTOCORR = 0.9
MIN_SEASONAL_PERIOD = 2
ACF_PEAK_THRESHOLD = 0.2
#: Fold-aware STL is offered to this many of the best models (see run_time_series_search).
STL_CANDIDATES = 3


#: Calendar periods, in observations, for each recognised sampling interval
#: (median spacing in days). With a datetime index these are the only
#: candidates: a free search on a series whose interval is known mostly finds
#: spurious peaks, and every true period in the measurement below was one of these.
CALENDAR_PERIODS = {1 / 24: (24, 168), 1: (7, 30, 365), 7: (52,), 30: (12,), 91: (4,)}
#: A spacing within this log-distance of a key (about +-40%) counts as that interval.
CALENDAR_TOLERANCE = 0.35
#: A real cycle's ACF peaks at P and troughs near P/2; a random walk's only decays.
ACF_PEAK_CONTRAST = 0.2


def _seasonal_index(levels: np.ndarray, period: int) -> np.ndarray:
    """STL's seasonal component averaged by phase — one value per position in the cycle.

    Called with ONE training fold's levels at a time. The whole-series version this
    module refused leaks, because STL's value at every point depends on the entire
    series; a per-phase index fitted on the training fold and looked up by a future
    row's phase (known in advance) does not.
    """
    from statsmodels.tsa.seasonal import STL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seasonal = STL(levels, period=period, robust=True).fit().seasonal
    phases = np.arange(len(levels)) % period
    return np.array([seasonal[phases == k].mean() for k in range(period)])


def _acf(values: np.ndarray, nlags: int) -> np.ndarray:
    from statsmodels.tsa.stattools import acf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return acf(values, nlags=nlags, fft=True)


def _detrend(values: np.ndarray, window: int) -> np.ndarray:
    """Subtract a centred moving average. Unlike first differencing — a high-pass
    filter that scales a period-P cycle by 2*sin(pi/P), 0.12 at P=52 — this leaves
    every period shorter than the window intact."""
    s = pd.Series(values)
    trend = s.rolling(window, center=True, min_periods=max(2, window // 2)).mean()
    return (s - trend).dropna().to_numpy()


def _calendar_candidates(times) -> tuple[tuple[int, ...], str] | None:
    parsed = pd.to_datetime(pd.Series(times), errors="coerce").dropna()
    if len(parsed) < 3:
        return None
    spacing = parsed.diff().median().total_seconds() / 86400
    if not np.isfinite(spacing) or spacing <= 0:
        return None
    key = min(CALENDAR_PERIODS, key=lambda k: abs(np.log(spacing / k)))
    if abs(np.log(spacing / key)) > CALENDAR_TOLERANCE:
        return None
    label = {1 / 24: "hourly", 1: "daily", 7: "weekly", 30: "monthly", 91: "quarterly"}[key]
    return CALENDAR_PERIODS[key], label


def _period_strength(values: np.ndarray, period: int) -> float | None:
    """ACF at `period` on the series detrended over two cycles, or None if that
    lag is not a real peak (local maximum, >= ACF_PEAK_THRESHOLD, and at least
    ACF_PEAK_CONTRAST above the ACF near half the period)."""
    if 2 * period + 2 >= len(values):
        return None
    resid = _detrend(values, 2 * period + 1)
    if len(resid) < 2 * period + 2 or resid.std() == 0:
        return None
    c = _acf(resid, min(2 * period + 2, len(resid) - 1))
    slack = max(1, period // 20)
    lo, hi = max(MIN_SEASONAL_PERIOD, period - slack), min(len(c) - 2, period + slack)
    if lo > hi:
        return None
    peak = lo + int(np.argmax(c[lo:hi + 1]))
    half = max(1, round(period / 2))
    trough = float(np.min(c[max(1, half - slack): half + slack + 1]))
    if c[peak] < c[peak - 1] or c[peak] < c[peak + 1]:
        return None
    if c[peak] < ACF_PEAK_THRESHOLD or c[peak] - trough < ACF_PEAK_CONTRAST:
        return None
    return float(c[peak])


def _free_search(values: np.ndarray, max_period: int) -> tuple[int | None, str]:
    """No recognised interval: any lag may be the period, so demand more —
    a local ACF peak on the detrended series that ALSO peaks near twice its lag."""
    window = max(3, min(len(values) // 4, 2 * max_period + 1))
    resid = _detrend(values, window)
    if len(resid) < MIN_SEASONAL_PERIOD + 4 or resid.std() == 0:
        return None, "detrended series is constant or too short"
    c = _acf(resid, min(max_period, len(resid) - 2))
    peaks = [lag for lag in range(MIN_SEASONAL_PERIOD, len(c) - 1)
             if c[lag] > c[lag - 1] and c[lag] >= c[lag + 1] and c[lag] >= ACF_PEAK_THRESHOLD]
    supported = []
    for lag in peaks:
        double, slack = 2 * lag, max(1, lag // 10)
        if double + slack >= len(c):
            continue  # a harmonic that was not computed cannot support anything
        if float(np.max(c[max(1, double - slack): double + slack + 1])) < ACF_PEAK_THRESHOLD / 2:
            continue
        # The same peak-to-trough test as the calendar path. Without it, harmonic
        # support alone gave 8 of 181 aperiodic series (mostly random walks) a
        # period; with it, 2 — while finding 156 of 180 true periods, not 136.
        half, narrow = max(1, round(lag / 2)), max(1, lag // 20)
        if c[lag] - float(np.min(c[max(1, half - narrow): half + narrow + 1])) < ACF_PEAK_CONTRAST:
            continue
        supported.append(lag)
    if not supported:
        strongest = max(peaks, key=lambda lag: c[lag]) if peaks else None
        return None, (f"treating as aperiodic: no ACF peak repeats at twice its lag with a trough at half "
                      f"of it (strongest unsupported peak: lag {strongest})"
                      if strongest else "no ACF peak above the threshold — treating as aperiodic")
    lag = max(supported, key=lambda p: c[p])
    return lag, f"ACF of the detrended series peaks at lag {lag} ({c[lag]:.2f}), with a peak near lag {2 * lag}"


def detect_seasonal_period(y: pd.Series, max_period: int | None = None,
                           times=None) -> tuple[int | None, str]:
    """
    Find the dominant seasonal period. Returns (period, reason); period is None
    when nothing periodic stands out.

    The first version read the ACF of the first DIFFERENCE and took its global
    maximum. Differencing is a high-pass filter, so it shrank the annual cycle of
    weekly CO2 to an eighth while short-lag noise passed through, and the
    detector reported period 3. Measured over 31 dated series with known periods
    (real weekly CO2 plus synthetic daily/weekly/monthly/hourly cycles, random
    walks and white noise, three seeds each): that detector got 17 right; this
    one gets 31. On undated series it found 156 of 180 true periods against
    52, while giving about 1% of aperiodic ones a period either way (2 of 181
    against 1) — a first sample of 30 had suggested none, and a larger one said
    otherwise.

    With a datetime index (`times`), only the calendar periods for the sampling
    interval are checked, each on the series detrended over two of its cycles.
    Without one, `_free_search` checks every lag but requires harmonic support.
    """
    values = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 20:
        return None, "series too short for seasonality detection"
    max_period = max_period or min(n // 3, 400)
    if max_period < MIN_SEASONAL_PERIOD:
        return None, "series too short for a meaningful seasonal period"

    try:
        calendar = _calendar_candidates(times) if times is not None else None
        if calendar is not None:
            candidates, interval = calendar
            strengths = {p: s for p in candidates if (s := _period_strength(values, p)) is not None}
            if not strengths:
                checked = ", ".join(str(p) for p in candidates if 2 * p + 2 < n) or "none fit the series"
                return None, f"{interval} series: no calendar cycle ({checked}) shows a real ACF peak — aperiodic"
            period = max(strengths, key=strengths.get)
            return period, (f"{interval} series: ACF of the detrended series peaks at the calendar period "
                            f"{period} ({strengths[period]:.2f})")
        return _free_search(values, max_period)
    except Exception as exc:  # noqa: BLE001
        return None, f"ACF computation failed ({type(exc).__name__})"


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


def seasonal_lag(seasonal_period: int, horizon: int) -> int:
    """The most recent same-season observation known `horizon` steps before the target."""
    return seasonal_period * int(np.ceil(horizon / seasonal_period))


def build_lag_feature_frame(
    df: pd.DataFrame, target_column: str, time_column: str,
    exogenous_columns: list[str], seasonal_period: int | None = None,
    lags=BASE_LAGS, rolling_windows=DEFAULT_ROLLING_WINDOWS, horizon: int = 1,
) -> pd.DataFrame:
    """
    Sort by time, then add causal features of the target that are known
    `horizon` steps before it: plain lags, rolling statistics, differences, and
    — when a seasonal period was detected — the seasonal lag and difference.

    A forecast made at time T for T+h can use the target up to T and no later,
    so every target feature is computed on the series shifted by `horizon`
    (T2-2's direct strategy): lag k becomes lag h+k-1, and the seasonal lag is
    the latest same-season value at least h back. At h=1 this is exactly the
    one-step frame. `target_lag_{horizon}` is the level at the forecast origin,
    which differencing reconstructs from. Exogenous columns are taken at the
    target's own time, i.e. treated as known in advance (a planned promotion);
    a covariate that is not known ahead would need lagging too.

    Rows without enough history are dropped; there is no honest way to fill
    them, because that data genuinely would not exist at prediction time.
    """
    ordered = df.sort_values(time_column).reset_index(drop=True)
    target = ordered[target_column]
    out = ordered[[time_column, target_column] + [c for c in exogenous_columns if c in ordered.columns]].copy()

    shifts = [horizon + lag - 1 for lag in lags]
    if seasonal_period:
        shifts.append(seasonal_lag(seasonal_period, horizon))
    shifts = sorted(set(shifts))
    for shift in shifts:
        out[f"target_lag_{shift}"] = target.shift(shift)

    shifted = target.shift(horizon)  # nothing later than the forecast origin
    for window in rolling_windows:
        out[f"target_rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"target_rolling_std_{window}"] = shifted.rolling(window).std()

    # Momentum: how the series was already moving at the forecast origin.
    out["target_diff_1"] = shifted.diff(1)
    out["target_diff_2"] = shifted.diff(2)

    if seasonal_period:
        out[f"target_rolling_mean_{seasonal_period}"] = shifted.rolling(seasonal_period).mean()
        out["target_seasonal_diff"] = shifted - target.shift(seasonal_period + horizon)

    max_history = max(max(shifts), max(rolling_windows) + horizon, (seasonal_period or 0) + horizon + 1)
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
    horizon: int = 1
    stl: str = "not evaluated"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _accumulate(store: dict[str, list[float]], y_true, y_pred) -> None:
    store["r2"].append(r2_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0)
    store["neg_rmse"].append(-float(np.sqrt(mean_squared_error(y_true, y_pred))))
    store["neg_mae"].append(-float(mean_absolute_error(y_true, y_pred)))


def _evaluate_baselines(y: np.ndarray, seasonal_period: int, cv: TimeSeriesSplit,
                        horizon: int = 1) -> list[BaselineResult]:
    """
    Baselines at the same horizon and information level as the ML models: each
    test point is forecast from the TRUE actuals up to `horizon` steps before
    it — the forecast origin — exactly as the lag features see them. At h=1
    that is the previous actual. Holding one value constant across an entire
    test fold would be a longer-horizon forecast than the models are making,
    and would flatter every model against it (invariant 4).
    """
    naive_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}
    seasonal_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}
    ma_scores = {"r2": [], "neg_rmse": [], "neg_mae": []}
    same_season = seasonal_lag(seasonal_period, horizon)

    for _, test_idx in cv.split(y):
        y_test = y[test_idx]
        origin = np.maximum(test_idx - horizon, 0)

        naive_pred = y[origin]
        _accumulate(naive_scores, y_test, naive_pred)

        seasonal_pred = np.array([y[i - same_season] if i - same_season >= 0 else y[max(i - horizon, 0)]
                                  for i in test_idx])
        _accumulate(seasonal_scores, y_test, seasonal_pred)

        ma_pred = np.array([y[max(0, o - 6):o + 1].mean() for o in origin])
        _accumulate(ma_scores, y_test, ma_pred)

    return [
        BaselineResult("naive_last_value", {k: float(np.mean(v)) for k, v in naive_scores.items()}),
        BaselineResult("seasonal_naive", {k: float(np.mean(v)) for k, v in seasonal_scores.items()}),
        BaselineResult("moving_average_7", {k: float(np.mean(v)) for k, v in ma_scores.items()}),
    ]


@dataclass
class LagDesign:
    """Everything the search and the final forecaster must build identically."""
    lag_frame: pd.DataFrame
    X: pd.DataFrame
    roles: FeatureRoleAssignment
    exogenous: list[str]
    y_level: np.ndarray
    origin_level: np.ndarray
    y_fit: np.ndarray
    offset: int  # rows dropped at the start for lack of history


def lag_design(df: pd.DataFrame, target_column: str, time_column: str, roles: FeatureRoleAssignment,
               seasonal_period: int | None, horizon: int, difference: bool) -> LagDesign:
    """The lag frame, its feature matrix and roles, and the target as fitted.

    Shared by `run_time_series_search` and `autoeng.modeling.forecaster`, so the
    model that is served is built by exactly the code that was cross-validated.
    """
    from autoeng.common.roles import assign_feature_roles
    from autoeng.profiling.profiler import profile_dataset

    exogenous = [c for c in roles.feature_columns if c not in (target_column, time_column)]
    lag_frame = build_lag_feature_frame(df, target_column, time_column, exogenous, seasonal_period,
                                        horizon=horizon)
    y_level = lag_frame[target_column].to_numpy(dtype=float)
    origin_level = lag_frame[f"target_lag_{horizon}"].to_numpy(dtype=float)
    # What the model is fitted on: either the level itself, or the change since
    # the forecast origin. Scoring always happens on the level.
    y_fit = (y_level - origin_level) if difference else y_level
    X = lag_frame.drop(columns=[target_column, time_column])
    sub_roles = assign_feature_roles(profile_dataset(X), target_column=None, time_column=None)
    return LagDesign(lag_frame=lag_frame, X=X, roles=sub_roles, exogenous=exogenous, y_level=y_level,
                     origin_level=origin_level, y_fit=y_fit, offset=len(df) - len(lag_frame))


def make_forecast_pipeline(name: str, factory, roles_for_x: FeatureRoleAssignment) -> Pipeline:
    # Never cap lag features, for any model. The cap is fitted on the training
    # fold, and on a trending series the test fold's lags sit above that range
    # by construction — so the cap clips exactly the most recent information.
    # Measured on real weekly CO2 at a 13-week horizon: ridge 0.814 capped
    # against 0.933 uncapped, huber 0.717 against 0.944; no effect either way
    # on a stationary series.
    pre = build_preprocessing_pipeline(roles_for_x, problem_kind="regression",
                                        cap_outliers=False, use_interactions=False)
    steps = list(pre.steps)
    if name in SCALE_SENSITIVE_MODELS:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", factory()))
    return Pipeline(steps)


def run_time_series_search(
    df: pd.DataFrame, target_column: str, time_column: str, roles: FeatureRoleAssignment,
    cv_folds: int = 5, seasonal_period: int | None = None, horizon: int = 1,
) -> tuple[list[ModelResult], list[BaselineResult], TimeSeriesSetup]:
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    ordered = df.sort_values(time_column)
    ordered_target = ordered[target_column]
    detected_period, period_reason = (
        (seasonal_period, "seasonal period supplied by caller") if seasonal_period
        else detect_seasonal_period(ordered_target, times=ordered[time_column])
    )
    difference, difference_reason = should_difference(ordered_target)
    setup = TimeSeriesSetup(
        seasonal_period=detected_period, seasonality_reason=period_reason,
        differenced=difference, differencing_reason=difference_reason, horizon=horizon,
    )

    design = lag_design(df, target_column, time_column, roles, detected_period, horizon, difference)
    lag_frame, X, sub_roles = design.lag_frame, design.X, design.roles
    y_level, previous_level, y_fit = design.y_level, design.origin_level, design.y_fit

    cv = TimeSeriesSplit(n_splits=cv_folds)
    models = get_regression_models()
    results: list[ModelResult] = []
    make_pipeline = make_forecast_pipeline

    def score(name: str, pipe: Pipeline, stl_period: int | None = None) -> ModelResult:
        fold_metrics = {"r2": [], "neg_rmse": [], "neg_mae": []}
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    if stl_period:
                        # Fitted on THIS training fold's levels only; a test row gets
                        # the index for its phase, which is known in advance.
                        index = _seasonal_index(y_level[train_idx], stl_period)
                        X_train = X_train.assign(stl_seasonal=index[phase[train_idx] % stl_period])
                        X_test = X_test.assign(stl_seasonal=index[phase[test_idx] % stl_period])
                    model = clone(pipe)
                    model.fit(X_train, y_fit[train_idx])
                    raw_preds = model.predict(X_test)
                    # Reconstruct onto the original scale so every model and
                    # baseline is scored against the same quantity.
                    preds = (previous_level[test_idx] + raw_preds) if difference else raw_preds
                    _accumulate(fold_metrics, y_level[test_idx], preds)
        except Exception as e:  # noqa: BLE001
            return ModelResult(name=name, status="failed", error=f"{type(e).__name__}: {e}")
        return ModelResult(name=name, status="ok", metrics={k: float(np.mean(v)) for k, v in fold_metrics.items()},
                           fit_time_seconds=time.time() - t0)

    for name, factory in models.items():
        results.append(score(name, make_pipeline(name, factory, sub_roles)))

    # Fold-aware STL (T2-2), offered to the best models and kept only where the
    # same cross-validation says it helps: measured, it lifted a daily series with
    # two cycles at every horizon, but cost the best model on real weekly CO2 and
    # on a cycle riding a random walk, so it cannot be a blanket feature.
    # Position in the lag frame. Every expanding-window training fold starts at
    # position 0, and `_seasonal_index` numbers phases from its own first row, so
    # the lookup must use the same origin (an offset here shifts every phase).
    phase = np.arange(len(lag_frame))
    min_train = min(len(train_idx) for train_idx, _ in cv.split(X))
    if not detected_period:
        setup.stl = "not offered: no seasonal period"
    elif min_train < 2 * detected_period + 1:
        setup.stl = (f"not offered: the first training fold has {min_train} rows, and STL needs two "
                     f"full cycles ({2 * detected_period + 1})")
    else:
        stl_roles = replace(sub_roles, numeric_columns=sub_roles.numeric_columns + ["stl_seasonal"])
        best = sorted((r for r in results if r.status == "ok"), key=lambda r: r.metrics["r2"],
                      reverse=True)[:STL_CANDIDATES]
        for base in best:
            variant = score(f"{base.name}+stl", make_pipeline(base.name, models[base.name], stl_roles),
                            stl_period=detected_period)
            results.append(variant)
        kept = [f"{r.name}" for r in results if r.name.endswith("+stl") and r.status == "ok"
                and r.metrics["r2"] > next(b.metrics["r2"] for b in best if f"{b.name}+stl" == r.name)]
        setup.stl = (f"offered to {', '.join(b.name for b in best)} with period {detected_period}; "
                     + (f"improved cross-validated r2 for {', '.join(kept)}" if kept else "improved none of them"))

    baselines = _evaluate_baselines(y_level, seasonal_period=detected_period or 7, cv=cv, horizon=horizon)
    return results, baselines, setup
