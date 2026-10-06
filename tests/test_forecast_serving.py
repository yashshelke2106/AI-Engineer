"""
Serving a forecaster over HTTP, and judging its forecasts as the actuals arrive.

/observe appends actual values; /forecast returns the forecast `horizon` steps
past the latest one. Every forecast is logged with its target time, and when the
actual for that time is observed it is attached as the outcome — so concept
drift reads forecast error with no labelling step a caller could forget.
Observations persist in the prediction log and are replayed on restart.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.forecaster import fit_forecaster, walk_forward_baseline
from autoeng.modeling.time_series import TimeSeriesSetup
from autoeng.monitoring.report import run_drift_report
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import load_training_schema, save_forecaster
from autoeng.serving.app import create_app
from autoeng.serving.store import PredictionStore

H, P, CUT = 4, 52, 220


def _series(n=330, seed=0, promo=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    frame = pd.DataFrame({
        "date": pd.date_range("2019-01-07", periods=n, freq="W-MON"),
        "sales": 100 + 0.3 * t + 8 * np.sin(2 * np.pi * t / P) + np.cumsum(rng.normal(0, 1, n)),
    })
    if promo:
        frame["promo"] = rng.integers(0, 2, n).astype(float)
        frame["sales"] += 5 * frame["promo"]
    return frame


def _deploy(tmp_path, frame, winner="ridge"):
    train = frame.iloc[:CUT]
    roles = assign_feature_roles(profile_dataset(train), target_column="sales", time_column="date")
    setup = TimeSeriesSetup(seasonal_period=P, seasonality_reason="test", differenced=True,
                            differencing_reason="test", horizon=H)
    forecaster = fit_forecaster(train, "sales", "date", roles, setup, winner)
    baseline = walk_forward_baseline(train, "sales", "date", roles, setup, winner)
    saved = save_forecaster(forecaster, tmp_path / "model", cv_metrics=baseline, setup=setup.as_dict())
    return saved, tmp_path / "log.db"


def _walk(client, frame, start, steps, promo=False):
    served = []
    for i in range(start, start + steps):
        body = {}
        if promo:
            target = frame["date"].iloc[i - 1 + H] if i - 1 + H < len(frame) else None
            body = {"exogenous": {"promo": float(frame.loc[frame["date"] == target, "promo"].iloc[0])}}
        response = client.post("/forecast", json=body)
        assert response.status_code == 200, response.json()
        served.append(response.json())
        observed = client.post("/observe", json={"observations": [
            {"date": str(frame["date"].iloc[i]), "sales": float(frame["sales"].iloc[i])}]})
        assert observed.status_code == 200, observed.json()
    return served


class TestTheContract:
    def test_model_publishes_horizon_and_the_next_times(self, tmp_path):
        frame = _series()
        saved, log = _deploy(tmp_path, frame)
        contract = TestClient(create_app(saved.model_dir, store_path=log)).get("/model").json()
        assert contract["problem_type"] == "time_series_forecasting" and contract["horizon"] == H
        assert contract["next_observation_time"] == str(frame["date"].iloc[CUT])
        assert contract["next_forecast_is_for"] == str(frame["date"].iloc[CUT - 1 + H])
        assert contract["baseline_metrics"]["mae_se"] > 0

    def test_served_forecasts_are_the_artifacts_own(self, tmp_path):
        frame = _series()
        saved, log = _deploy(tmp_path, frame)
        reference = joblib.load(saved.model_path)
        served = _walk(TestClient(create_app(saved.model_dir, store_path=log)), frame, CUT, 10)
        for i, response in enumerate(served):
            assert response["forecast"] == pytest.approx(reference.forecast()["forecast"], abs=1e-9)
            assert response["target_time"] == str(frame["date"].iloc[CUT + i - 1 + H])
            reference.observe([{"date": frame["date"].iloc[CUT + i], "sales": frame["sales"].iloc[CUT + i]}])


class TestForecastsAreJudged:
    def test_each_forecast_gets_its_actual_when_that_time_is_observed(self, tmp_path):
        frame = _series()
        saved, log = _deploy(tmp_path, frame)
        served = _walk(TestClient(create_app(saved.model_dir, store_path=log)), frame, CUT, 12)
        labelled = PredictionStore(log).labelled_frame()
        assert len(labelled) == 12 - H + 1, "a forecast matures once its target time has been observed"
        truth = {str(t): v for t, v in zip(frame["date"], frame["sales"])}  # str(Timestamp), as logged
        for _, row in labelled.iterrows():
            assert row["actual"] == pytest.approx(truth[row["date"]])
        assert {r["request_id"] for r in served} >= set(labelled["request_id"])

    def test_the_drift_report_reads_forecast_error(self, tmp_path):
        frame = _series(n=330)
        saved, log = _deploy(tmp_path, frame)
        _walk(TestClient(create_app(saved.model_dir, store_path=log)), frame, CUT, 60)
        report = run_drift_report(PredictionStore(log), load_training_schema(saved.schema_path))
        assert report.data is None and any("Input drift is not measured for a forecaster" in n for n in report.notes)
        assert report.concept is not None and report.concept.severity.value != "unknown"
        assert {"mae", "rmse"} <= set(report.concept.observed)
        assert "noise margin" in report.concept.summary


class TestRestarts:
    def test_a_restarted_server_stands_where_it_stood(self, tmp_path):
        frame = _series()
        saved, log = _deploy(tmp_path, frame)
        first = TestClient(create_app(saved.model_dir, store_path=log))
        _walk(first, frame, CUT, 7)
        before = first.get("/model").json()
        second = TestClient(create_app(saved.model_dir, store_path=log))
        assert second.get("/health").json()["observations_replayed_at_start"] == 7
        assert second.get("/model").json()["next_forecast_is_for"] == before["next_forecast_is_for"]
        assert second.post("/forecast", json={}).json()["forecast"] == pytest.approx(
            first.post("/forecast", json={}).json()["forecast"])


class TestRefusals:
    def test_an_observation_that_skips_a_step_is_a_422(self, tmp_path):
        frame = _series()
        saved, log = _deploy(tmp_path, frame)
        client = TestClient(create_app(saved.model_dir, store_path=log))
        response = client.post("/observe", json={"observations": [
            {"date": str(frame["date"].iloc[CUT + 1]), "sales": 1.0}]})
        assert response.status_code == 422 and "without gaps" in response.json()["detail"]["message"]

    def test_a_forecast_missing_its_covariate_is_a_422(self, tmp_path):
        frame = _series(promo=True)
        saved, log = _deploy(tmp_path, frame)
        client = TestClient(create_app(saved.model_dir, store_path=log))
        assert client.get("/model").json()["required_covariates_at_target_time"] == ["promo"]
        response = client.post("/forecast", json={})
        assert response.status_code == 422 and "promo" in response.json()["detail"]["message"]
        assert client.post("/forecast", json={"exogenous": {"promo": 1.0}}).status_code == 200
