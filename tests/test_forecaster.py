"""
Deployable forecasters: the served forecast must be the offline one.

A time-series run used to persist nothing. A `Forecaster` now carries the
winner — an estimator, a `+stl` variant, or a baseline formula — with the
history and design its features need. The test that matters: fit on the series
up to T, forecast T+h, and get exactly what the same pipeline predicts for that
row of the FULL series' lag frame (whose features, by construction, see nothing
after T). Then observe the next actuals one by one and require the same at every
new origin.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.forecaster import BASELINES, ForecastInputError, fit_forecaster
from autoeng.modeling.time_series import TimeSeriesSetup, lag_design, seasonal_lag
from autoeng.profiling.profiler import profile_dataset


def _series(n=260, seed=0, promo=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-06", periods=n, freq="W-MON"),
        "sales": 100 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 52) + np.cumsum(rng.normal(0, 1, n)),
    })
    if promo:
        frame["promo"] = rng.integers(0, 2, n).astype(float)
        frame["sales"] += 5 * frame["promo"]
    return frame


def _setup(horizon=4, period=52, differenced=True):
    return TimeSeriesSetup(seasonal_period=period, seasonality_reason="test", differenced=differenced,
                           differencing_reason="test", horizon=horizon)


def _roles(frame):
    return assign_feature_roles(profile_dataset(frame), target_column="sales", time_column="date")


def _offline(forecaster, full: pd.DataFrame, target_time) -> float:
    """What the forecaster's own pipeline predicts for `target_time` from the full series."""
    design = lag_design(full, "sales", "date", _roles(full), forecaster.seasonal_period, forecaster.horizon,
                        forecaster.differenced)
    j = int(np.flatnonzero(design.lag_frame["date"].to_numpy() == np.datetime64(target_time))[0])
    X = design.X.iloc[[j]]
    if forecaster.stl_index is not None:
        X = X.assign(stl_seasonal=forecaster.stl_index[j % len(forecaster.stl_index)])
    raw = float(forecaster.pipeline.predict(X)[0])
    return design.origin_level[j] + raw if forecaster.differenced else raw


class TestServedEqualsOffline:
    @pytest.mark.parametrize("winner", ["ridge", "ridge+stl"])
    def test_every_origin_reproduces_the_offline_forecast(self, winner):
        full = _series()
        cut = 200
        forecaster = fit_forecaster(full.iloc[:cut], "sales", "date", _roles(full.iloc[:cut]), _setup(), winner)
        for k in range(6):
            served = forecaster.forecast()
            assert served["target_time"] == full["date"].iloc[cut - 1 + k + 4]
            assert served["forecast"] == pytest.approx(_offline(forecaster, full, served["target_time"]), abs=1e-9)
            next_row = full.iloc[cut + k]
            forecaster.observe([{"date": next_row["date"], "sales": next_row["sales"]}])

    def test_a_known_in_advance_covariate_is_read_at_the_target_time(self):
        full = _series(promo=True)
        cut = 200
        forecaster = fit_forecaster(full.iloc[:cut], "sales", "date", _roles(full.iloc[:cut]), _setup(), "ridge")
        target = forecaster.target_time
        promo_then = float(full.loc[full["date"] == target, "promo"].iloc[0])
        served = forecaster.forecast({"promo": promo_then})["forecast"]
        assert served == pytest.approx(_offline(forecaster, full, target), abs=1e-9)
        other = forecaster.forecast({"promo": 1.0 - promo_then})["forecast"]
        assert other != pytest.approx(served), "the covariate must reach the model"

    def test_it_survives_a_round_trip_through_joblib(self, tmp_path):
        full = _series()
        forecaster = fit_forecaster(full, "sales", "date", _roles(full), _setup(), "ridge+stl")
        joblib.dump(forecaster, tmp_path / "model.joblib")
        reloaded = joblib.load(tmp_path / "model.joblib")
        assert reloaded.forecast() == forecaster.forecast()


class TestBaselinesDeployAsThemselves:
    @pytest.mark.parametrize("name", BASELINES)
    def test_the_formula_it_was_scored_by(self, name):
        full = _series()
        horizon = 4
        forecaster = fit_forecaster(full, "sales", "date", _roles(full), _setup(horizon=horizon), name)
        assert forecaster.kind == "baseline" and forecaster.pipeline is None
        y = full["sales"].to_numpy()
        expected = {
            "naive_last_value": y[-1],
            "moving_average_7": y[-7:].mean(),
            "seasonal_naive": y[len(y) - 1 + horizon - seasonal_lag(52, horizon)],
        }[name]
        assert forecaster.forecast()["forecast"] == pytest.approx(expected)


class TestNothingIsInvented:
    def test_a_missing_covariate_is_refused(self):
        full = _series(promo=True)
        forecaster = fit_forecaster(full, "sales", "date", _roles(full), _setup(), "ridge")
        with pytest.raises(ForecastInputError, match="promo"):
            forecaster.forecast()

    def test_an_observation_that_skips_a_step_is_refused(self):
        full = _series()
        forecaster = fit_forecaster(full.iloc[:200], "sales", "date", _roles(full.iloc[:200]), _setup(), "ridge")
        skipped = full.iloc[201]
        with pytest.raises(ForecastInputError, match="without gaps"):
            forecaster.observe([{"date": skipped["date"], "sales": skipped["sales"]}])

    def test_an_observation_without_a_value_is_refused(self):
        full = _series()
        forecaster = fit_forecaster(full.iloc[:200], "sales", "date", _roles(full.iloc[:200]), _setup(), "ridge")
        with pytest.raises(ForecastInputError):
            forecaster.observe([{"date": full["date"].iloc[200]}])


def test_history_is_trimmed_without_losing_the_stl_phase():
    """Thousands of observations later the oldest rows are dropped — and the count
    is kept, or every fold-aware STL phase would shift by the rows trimmed."""
    from autoeng.modeling import forecaster as module

    full = _series(n=400)
    cut = 200
    original = module.HISTORY_SLACK
    module.HISTORY_SLACK = 10  # force trimming within a short test
    try:
        f = fit_forecaster(full.iloc[:cut], "sales", "date", _roles(full.iloc[:cut]), _setup(), "ridge+stl")
        for i in range(cut, cut + 150):
            f.observe([{"date": full["date"].iloc[i], "sales": full["sales"].iloc[i]}])
        assert f.rows_trimmed > 0
        served = f.forecast()
        assert served["forecast"] == pytest.approx(_offline(f, full, served["target_time"]), abs=1e-9)
    finally:
        module.HISTORY_SLACK = original


def test_the_pipeline_persists_the_search_winner(tmp_path):
    """Through the real search and the pipeline's own persistence helper."""
    from autoeng.modeling.time_series import run_time_series_search
    from autoeng.pipeline import _persist_forecaster
    from autoeng.registry.model_store import load_model

    frame = _series(n=300)
    roles = _roles(frame)
    results, baselines, setup = run_time_series_search(frame, "sales", "date", roles, cv_folds=3, horizon=4)
    winner = max([r for r in results if r.status == "ok"] + list(baselines), key=lambda r: r.metrics["r2"]).name
    artifact = _persist_forecaster(frame, "sales", "date", roles, setup, winner,
                                   model_dir=tmp_path / "model", dataset_path="series.csv")
    assert artifact["status"] == "saved", artifact
    loaded = load_model(artifact["model_path"])
    assert loaded.schema["problem_type"] == "time_series_forecasting"
    assert loaded.schema["model"]["name"] == winner and loaded.estimator.name == winner
    baseline = loaded.schema["baseline_metrics"]
    assert set(baseline) == {"mae", "rmse", "mae_se", "rmse_se"}, "no R²: it measures the window, not the forecast"
    assert "walk-forward" in loaded.schema["baseline_source"]
    assert loaded.estimator.forecast()["target_time"] == frame["date"].iloc[-1] + 4 * pd.Timedelta(weeks=1)


def test_the_baseline_covers_two_seasonal_cycles_and_says_when_it_cannot():
    from autoeng.modeling.forecaster import walk_forward_baseline

    long = _series(n=420)
    base = walk_forward_baseline(long, "sales", "date", _roles(long), _setup(), "ridge")
    assert base["window"] >= 104 and "seasonal cycles" in base["reliability"]
    short = _series(n=160)
    base = walk_forward_baseline(short, "sales", "date", _roles(short), _setup(), "ridge")
    assert base["window"] < 104 and "too short" in base["reliability"]
    assert base["fit_rows"] >= len(short) // 2, "the forecaster must still fit on most of the series"
