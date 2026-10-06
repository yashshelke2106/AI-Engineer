"""
Serving a forecaster: observations in, forecasts out, every forecast judged later.

    POST /observe    actual values of the series, each the step after the last
    POST /forecast   the forecast `horizon` steps past the latest observation
    GET  /model      the contract: horizon, sampling, required covariates, next times
    GET  /health

Every forecast is logged with its target time. When the actual value for that
time arrives through /observe, it is attached as the forecast's outcome — so the
existing concept-drift and retraining machinery reads forecast error exactly as
it reads any regression error, with no separate labelling step a caller could
forget.

Observations are written to the prediction log as they arrive and replayed when
the server starts, so a restart stands exactly where it stood. 422, never a
default: a forecast missing a known-in-advance covariate, or an observation that
skips a step, is refused with the reason (invariant 7a).
"""
from __future__ import annotations

import threading
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from autoeng.modeling.forecaster import ForecastInputError
from autoeng.serving.store import PredictionStore, new_request_id


class ObserveRequest(BaseModel):
    observations: list[dict[str, Any]] = Field(
        ..., description="Actual values in time order, each keyed by the time and target columns "
                         "(and any covariates). Each must be exactly one step after the last.")


class ForecastRequest(BaseModel):
    exogenous: dict[str, Any] = Field(
        default_factory=dict, description="Known-in-advance covariates at the target time, if the model uses any.")


def _contract(forecaster, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_type": "time_series_forecasting",
        "model": schema.get("model"),
        "trained_at": schema.get("created_utc"),
        "target": forecaster.target_column,
        "time_column": forecaster.time_column,
        "horizon": forecaster.horizon,
        "frequency": forecaster.frequency,
        "required_covariates_at_target_time": list(forecaster.exogenous_columns) if forecaster.kind == "model" else [],
        "latest_observation": str(forecaster.last_time),
        "next_observation_time": str(forecaster.next_observation_time),
        "next_forecast_is_for": str(forecaster.target_time),
        "baseline_metrics": schema.get("baseline_metrics"),
        "notes": list(forecaster.notes),
        "library_versions": schema.get("library_versions"),
    }


def add_forecasting_routes(app: FastAPI, forecaster, schema: dict[str, Any], store: PredictionStore | None,
                           version: str) -> None:
    lock = threading.Lock()  # one history, mutated by /observe and read by /forecast
    replayed = 0
    if store is not None:
        for payload in store.observations():
            when = forecaster._parse_time(payload[forecaster.time_column])
            if when <= forecaster.last_time:
                continue  # already in the history the model was trained or last saved on
            forecaster.observe([payload])
            replayed += 1
    app.state.forecaster = forecaster
    app.state.observations_replayed = replayed

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "degraded" if app.state.version_warnings else "ok",
            "model_dir": app.state.model_dir,
            "model": forecaster.name, "kind": forecaster.kind,
            "problem_type": "time_series_forecasting",
            "version_warnings": app.state.version_warnings,
            "model_version": version,
            "latest_observation": str(forecaster.last_time),
            "observations_replayed_at_start": replayed,
            "prediction_log": (store.counts() | {"path": str(store.path)}) if store else None,
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        return _contract(forecaster, schema)

    @app.post("/observe")
    def observe(request: ObserveRequest) -> dict[str, Any]:
        matured = 0
        with lock:
            try:
                times = forecaster.observe(request.observations)
            except ForecastInputError as e:
                raise HTTPException(status_code=422, detail={"message": str(e)}) from e
            if store is not None:
                for row, when in zip(request.observations, times):
                    store.log_observation(row | {forecaster.time_column: when})
                    # Every forecast made FOR this time now has its answer.
                    for request_id in store.request_ids_where(forecaster.time_column, when):
                        store.record_outcome(request_id, float(row[forecaster.target_column]), source="observe")
                        matured += 1
            return {"appended": len(times), "latest_observation": str(forecaster.last_time),
                    "next_observation_time": str(forecaster.next_observation_time),
                    "forecasts_matured": matured}

    @app.post("/forecast")
    def forecast(request: ForecastRequest) -> dict[str, Any]:
        with lock:
            try:
                result = forecaster.forecast(request.exogenous)
            except ForecastInputError as e:
                raise HTTPException(status_code=422, detail={"message": str(e)}) from e
        request_id = new_request_id()
        payload = {forecaster.time_column: result["target_time"], "origin": result["origin_time"],
                   **{c: request.exogenous.get(c) for c in forecaster.exogenous_columns}}
        warnings: list[str] = []
        if store is not None:
            try:
                store.log_prediction(payload=payload, prediction=result["forecast"], request_id=request_id,
                                     decision_rule=f"{forecaster.horizon}-step forecast", model_version=version,
                                     model_name=forecaster.name)
            except Exception as e:  # noqa: BLE001 - never deny the forecast; never hide the hole either
                warnings.append(f"Forecast was not logged ({type(e).__name__}: {e}).")
        return {**{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in result.items()},
                "request_id": request_id, "warnings": warnings}
