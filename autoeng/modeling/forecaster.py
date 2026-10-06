"""
A deployable forecaster: the winner of a time-series run, plus everything it
needs to forecast again later.

Time-series runs used to persist nothing, because "the model" is not one
estimator: a forecast needs the recent history its lag features read, the
horizon and sampling interval they were built for, whether the target was
differenced, the fold-aware STL index if a `+stl` variant won — or no estimator
at all, when a classical baseline beat every model. A `Forecaster` holds all of
it, and builds each forecast's features with `build_lag_feature_frame`, the same
function the search cross-validated, so a served forecast is the offline one.

Three commitments, each pinned by a test:

- **A winning baseline is deployed as itself.** If seasonal-naive beat every
  model, seasonal-naive is what serves, with the same formula it was scored by.
- **Nothing is invented.** A covariate known in advance must be supplied for the
  target time or the forecast is refused, and an observation that skips a step
  is refused rather than silently misaligning every lag (invariant 7a).
- **History is append-only.** Observations extend it in time order; the oldest
  rows are trimmed only once far more than any feature needs is held, and the
  count trimmed is kept so fold-aware STL phases stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.modeling.model_zoo import get_regression_models
from autoeng.modeling.time_series import (
    BASE_LAGS, DEFAULT_ROLLING_WINDOWS, TimeSeriesSetup, _seasonal_index, build_lag_feature_frame, lag_design,
    make_forecast_pipeline, seasonal_lag,
)

BASELINES = ("naive_last_value", "seasonal_naive", "moving_average_7")
#: Rows of history kept beyond what the features need, before the front is trimmed.
HISTORY_SLACK = 2_000


class ForecastInputError(ValueError):
    """A forecast or observation that cannot be served honestly; the message says why."""


@dataclass
class Forecaster:
    name: str
    kind: str  # "model" or "baseline"
    target_column: str
    time_column: str
    exogenous_columns: list[str]
    horizon: int
    seasonal_period: int | None
    baseline_period: int
    differenced: bool
    frequency: str | None
    spacing: Any
    history: pd.DataFrame
    offset: int
    rows_trimmed: int = 0
    pipeline: Any = None
    stl_index: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    # -- time ---------------------------------------------------------------

    @property
    def last_time(self):
        return self.history[self.time_column].iloc[-1]

    def step(self, time, k: int = 1):
        """The time `k` sampling steps after `time`."""
        if self.frequency:
            return time + k * pd.tseries.frequencies.to_offset(self.frequency)
        return time + k * self.spacing

    @property
    def next_observation_time(self):
        return self.step(self.last_time)

    @property
    def target_time(self):
        """What a forecast made now is FOR: `horizon` steps past the latest observation."""
        return self.step(self.last_time, self.horizon)

    def _parse_time(self, value):
        if isinstance(self.last_time, pd.Timestamp):
            return pd.Timestamp(value)
        return type(self.last_time)(value)

    # -- history ------------------------------------------------------------

    def observe(self, rows: list[dict[str, Any]], strict: bool = True) -> list[Any]:
        """Append actual observations, each exactly one step after the last.

        Returns their times. A gap or an out-of-order row is refused: lag features
        count steps, so a skipped week would silently shift every one of them.
        `strict=False` (internal replay of HISTORICAL rows only) requires just that
        time moves forward — the row-counting rule the search itself trained on,
        so a historical gap is replayed exactly as it was learned.
        """
        appended = []
        for row in rows:
            if self.time_column not in row or row.get(self.target_column) is None:
                raise ForecastInputError(
                    f"An observation needs '{self.time_column}' and '{self.target_column}'; got {sorted(row)}.")
            when = self._parse_time(row[self.time_column])
            expected = self.next_observation_time
            if (when != expected) if strict else (when <= self.last_time):
                raise ForecastInputError(
                    f"Expected the observation for {expected}, the step after {self.last_time}; got {when}. "
                    f"Observations must arrive in order without gaps — a skipped step would misalign every lag.")
            try:
                value = float(row[self.target_column])
            except (TypeError, ValueError) as exc:
                raise ForecastInputError(f"'{self.target_column}' must be numeric; got {row[self.target_column]!r}.") \
                    from exc
            record = {self.time_column: when, self.target_column: value}
            record.update({c: row.get(c) for c in self.exogenous_columns})
            self.history = pd.concat([self.history, pd.DataFrame([record])], ignore_index=True)
            appended.append(when)
        self._trim()
        return appended

    def _trim(self) -> None:
        keep = self.offset + 2 * self.horizon + HISTORY_SLACK
        if len(self.history) > keep:
            dropped = len(self.history) - keep
            self.history = self.history.iloc[dropped:].reset_index(drop=True)
            self.rows_trimmed += dropped

    # -- forecasting ----------------------------------------------------------

    def forecast(self, exogenous: dict[str, Any] | None = None) -> dict[str, Any]:
        """The forecast for `target_time`, from the history as it stands."""
        exogenous = exogenous or {}
        missing = [c for c in self.exogenous_columns if exogenous.get(c) is None]
        if missing and self.kind == "model":
            raise ForecastInputError(
                f"The model reads {missing} at the target time {self.target_time}; supply "
                f"{'it' if len(missing) == 1 else 'them'}. A covariate is never filled in.")
        y = self.history[self.target_column].to_numpy(dtype=float)
        n = len(y)
        if self.kind == "baseline":
            value = self._baseline(y)
        else:
            value = self._model(exogenous)
        return {
            "target_time": self.target_time, "origin_time": self.last_time, "horizon": self.horizon,
            "forecast": float(value), "model": self.name, "kind": self.kind,
            "observations_held": int(n),
        }

    def _baseline(self, y: np.ndarray) -> float:
        # The same formulas `_evaluate_baselines` scored, with the forecast origin
        # at the latest observation.
        if self.name == "naive_last_value":
            return float(y[-1])
        if self.name == "moving_average_7":
            return float(y[-7:].mean())
        back = seasonal_lag(self.baseline_period, self.horizon) - self.horizon
        return float(y[-1 - back]) if back < len(y) else float(y[-1])

    def _model(self, exogenous: dict[str, Any]) -> float:
        future = []
        for k in range(1, self.horizon + 1):
            row = {self.time_column: self.step(self.last_time, k), self.target_column: np.nan}
            row.update({c: (exogenous.get(c) if k == self.horizon else np.nan) for c in self.exogenous_columns})
            future.append(row)
        extended = pd.concat([self.history, pd.DataFrame(future)], ignore_index=True)
        lag = build_lag_feature_frame(extended, self.target_column, self.time_column, self.exogenous_columns,
                                      self.seasonal_period, BASE_LAGS, DEFAULT_ROLLING_WINDOWS, horizon=self.horizon)
        row = lag.iloc[[-1]]
        X = row.drop(columns=[self.target_column, self.time_column])
        if self.stl_index is not None:
            # Position in the training lag frame: every row ever held, minus the
            # rows the lag frame drops for lack of history.
            position = self.rows_trimmed + len(extended) - 1 - self.offset
            X = X.assign(stl_seasonal=self.stl_index[position % len(self.stl_index)])
        raw = float(self.pipeline.predict(X)[0])
        return float(row[f"target_lag_{self.horizon}"].iloc[0]) + raw if self.differenced else raw


def walk_forward_baseline(df: pd.DataFrame, target_column: str, time_column: str, roles: FeatureRoleAssignment,
                          setup: TimeSeriesSetup, winner: str, cv_folds: int = 5,
                          window: int | None = None) -> dict[str, float]:
    """The deployed forecaster's own walk-forward error, as monitoring will see it.

    Fit on everything before the last `window` rows, then forecast and observe
    through them one step at a time — the same procedure serving follows. Two
    yardsticks were measured and rejected first:

    - the search's mean over expanding folds: its first folds train on a fraction
      of the data, and the deployed model's live MAE ran at 0.44x it, so a
      doubled error read as healthy;
    - the last fold alone (~40 forecasts on a 250-week series): live/baseline MAE
      spread 0.65-2.60 with no drift at all, because forecast error varies across
      the seasonal cycle and 40 weeks cover less than one. 16 of 60 unshifted
      windows were flagged even with a noise margin.

    So the window spans TWO seasonal cycles (or the last fold if that is longer),
    capped at half the series so the forecaster still fits on most of it. On a
    400-week series that flagged 0 of 60 unshifted windows and caught a tripled
    volatility in 18 of 30. On a short series the cap bites and the baseline turns
    pessimistic (detection 8 of 30 at 250 weeks); `reliability` says so.

    Returns MAE and RMSE with standard errors counted in roughly independent
    errors (consecutive h-step forecasts share h-1 shocks).
    """
    ordered = df.sort_values(time_column).reset_index(drop=True)
    if pd.api.types.is_object_dtype(ordered[time_column]):
        ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce", format="mixed")
    n_lag = len(ordered) - lag_design(ordered, target_column, time_column, roles, setup.seasonal_period,
                                      setup.horizon, setup.differenced).offset
    cycles = 2 * setup.seasonal_period if setup.seasonal_period else 0
    wanted = window or max(n_lag // (cv_folds + 1), cycles, setup.horizon + 3)
    test = min(wanted, max(n_lag // 2, setup.horizon + 3))
    cut = len(ordered) - test
    f = fit_forecaster(ordered.iloc[:cut], target_column, time_column, roles, setup, winner)
    errors = []
    for i in range(cut, len(ordered)):
        target_position = i - 1 + setup.horizon
        if target_position < len(ordered):
            exogenous = {c: ordered[c].iloc[target_position] for c in f.exogenous_columns}
            errors.append(float(ordered[target_column].iloc[target_position]) - f.forecast(exogenous)["forecast"])
        f.observe([ordered.iloc[i].to_dict()], strict=False)
    e = np.asarray(errors)
    n_eff = max(len(e) / setup.horizon, 2.0)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    short = bool(cycles and test < cycles)
    return {
        "mae": float(np.mean(np.abs(e))), "rmse": rmse,
        "mae_se": float(np.std(np.abs(e), ddof=1) / np.sqrt(n_eff)),
        "rmse_se": float(np.std(e ** 2, ddof=1) / (2.0 * max(rmse, 1e-12) * np.sqrt(n_eff))),
        "n_forecasts": int(len(e)), "window": int(test), "fit_rows": int(cut),
        "reliability": (
            f"The baseline covers {test} steps, less than two seasonal cycles ({cycles}): the series is too "
            f"short to measure forecast error across the cycle while leaving enough to fit on. Expect weak "
            f"detection of degradation here, not false alarms." if short else
            f"The baseline covers {test} steps ({test / cycles:.1f} seasonal cycles), fitted on {cut} rows."
            if cycles else f"The baseline covers {test} steps, fitted on {cut} rows."),
    }


def _frequency(times: pd.Series) -> tuple[str | None, Any]:
    if pd.api.types.is_datetime64_any_dtype(times):
        try:
            frequency = pd.infer_freq(times.iloc[-min(len(times), 500):])
        except (TypeError, ValueError):
            frequency = None
        return frequency, times.diff().median()
    return None, times.diff().median()


def fit_forecaster(df: pd.DataFrame, target_column: str, time_column: str, roles: FeatureRoleAssignment,
                   setup: TimeSeriesSetup, winner: str) -> Forecaster:
    """Refit the search's winner on every row and package it for serving.

    `winner` is a model name from the zoo, a `<model>+stl` variant, or one of
    `BASELINES`. The design is rebuilt by `lag_design`, the code the search used.
    """
    ordered = df.sort_values(time_column).reset_index(drop=True)
    if pd.api.types.is_object_dtype(ordered[time_column]):
        ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce", format="mixed")
    design = lag_design(ordered, target_column, time_column, roles, setup.seasonal_period, setup.horizon,
                        setup.differenced)
    frequency, spacing = _frequency(ordered[time_column])
    history = ordered[[time_column, target_column] + design.exogenous].reset_index(drop=True)
    forecaster = Forecaster(
        name=winner, kind="baseline" if winner in BASELINES else "model", target_column=target_column,
        time_column=time_column, exogenous_columns=design.exogenous, horizon=setup.horizon,
        seasonal_period=setup.seasonal_period, baseline_period=setup.seasonal_period or 7,
        differenced=setup.differenced, frequency=frequency, spacing=spacing, history=history,
        offset=design.offset,
    )
    if forecaster.kind == "baseline":
        forecaster.notes.append(f"'{winner}' beat every model under the same cross-validation, so it is "
                                f"deployed as itself: no estimator, the formula it was scored by.")
        return forecaster

    base = winner[:-4] if winner.endswith("+stl") else winner
    X, roles_for_x = design.X, design.roles
    if winner.endswith("+stl"):
        from dataclasses import replace

        forecaster.stl_index = _seasonal_index(design.y_level, setup.seasonal_period)
        X = X.assign(stl_seasonal=forecaster.stl_index[np.arange(len(X)) % setup.seasonal_period])
        roles_for_x = replace(roles_for_x, numeric_columns=roles_for_x.numeric_columns + ["stl_seasonal"])
    pipeline = make_forecast_pipeline(base, get_regression_models()[base], roles_for_x)
    pipeline.fit(X, design.y_fit)
    forecaster.pipeline = pipeline
    forecaster._trim()
    return forecaster
