"""
FastAPI serving layer over a persisted run.

Loads a model directory written by `autoeng/registry/model_store.py` — the
joblib estimator plus `training_schema.json` — and exposes it. Everything
interesting happens in `validation.py` (the contract) and `predictor.py` (the
decision threshold); this module is the HTTP surface over them.

    uvicorn autoeng.serving.app:app          # reads AUTOENG_MODEL_DIR
    python -m autoeng.cli serve runs/models/my_run

Endpoints:

    GET  /health         is a model loaded, and which
    GET  /model          the full serving contract: columns, order, dtypes,
                         target labels, decision threshold, library versions
    POST /predict        one row
    POST /predict/batch  many rows

**422, never a default.** A payload missing a feature is rejected with the
column named. The alternative — imputing it — returns a confident prediction
built on a value nobody supplied, and nothing in the response would say so.
That is the single design commitment this layer exists to keep.

Version skew is surfaced, not hidden: the model store compares the library
versions an artifact was fit under against the ones running now, and `/health`
reports the mismatch rather than letting a silently-different estimator serve
traffic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autoeng.registry.champion import read_champion
from autoeng.registry.model_store import MODEL_FILENAME, SCHEMA_FILENAME, load_model
from autoeng.serving.predictor import predict_frame, threshold_from_schema
from autoeng.serving.store import PredictionStore, UnknownRequestError, new_request_id
from autoeng.serving.validation import validate_payload

MODEL_DIR_ENV = "AUTOENG_MODEL_DIR"
STORE_PATH_ENV = "AUTOENG_PREDICTION_LOG"
DEFAULT_STORE_FILENAME = "predictions.db"


class PredictRequest(BaseModel):
    row: dict[str, Any] = Field(..., description="One record, keyed by training column name.")
    allow_unknown: bool = Field(
        False, description="Ignore columns the model was not trained on instead of rejecting them.",
    )


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., description="Records, each keyed by training column name.")
    allow_unknown: bool = False


class OutcomeRequest(BaseModel):
    request_id: str = Field(..., description="The request_id returned by /predict.")
    actual: Any = Field(..., description="The ground truth that eventually arrived.")
    source: str | None = Field(None, description="Where this label came from, for auditing.")


class BatchOutcomeRequest(BaseModel):
    outcomes: list[OutcomeRequest]


def _model_version(schema: dict[str, Any]) -> str:
    """
    A stable identifier for the artifact serving traffic.

    Predictions are logged against it so T1-5 can compare a champion's
    behaviour with a challenger's, and so a drift alarm can be attributed to
    the model that actually produced the predictions rather than to whatever is
    deployed at the moment someone looks.
    """
    model = schema.get("model") or {}
    return f"{model.get('name', 'unknown')}@{schema.get('created_utc', 'unknown')}"


def _serving_contract(schema: dict[str, Any]) -> dict[str, Any]:
    """What a caller needs in order to construct a valid request."""
    threshold, objective = threshold_from_schema(schema)
    columns = schema.get("columns") or {}
    return {
        "problem_type": schema.get("problem_type"),
        "model": schema.get("model"),
        "trained_at": schema.get("created_utc"),
        "n_training_rows": schema.get("n_training_rows"),
        "target": {
            "column": (schema.get("target") or {}).get("column"),
            "class_labels": (schema.get("target") or {}).get("class_labels"),
        },
        "decision_threshold": {"value": threshold, "objective": objective},
        # Order is part of the contract; returned as a list, not a mapping.
        "feature_columns": schema.get("feature_columns"),
        "features": [
            {
                "name": name,
                "dtype": (columns.get(name) or {}).get("dtype"),
                "semantic_type": (columns.get(name) or {}).get("semantic_type"),
                "nullable_in_training": (columns.get(name) or {}).get("nullable"),
                "required": True,
            }
            for name in (schema.get("feature_columns") or [])
        ],
        "library_versions": schema.get("library_versions"),
    }


def create_app(
    model_dir: str | Path | None = None,
    store_path: str | Path | None = None,
    log_predictions: bool = True,
) -> FastAPI:
    """
    Build an app bound to one model directory.

    A factory rather than a module-level singleton so tests can point at a
    temporary directory without touching the environment.
    """
    resolved = Path(model_dir or os.environ.get(MODEL_DIR_ENV, "")).expanduser()
    # A models root holding a CHAMPION.json pointer is followed to whichever
    # model the gate last promoted. That is what makes "a rejected challenger
    # stays out of production" true rather than merely recorded: serving reads
    # the pointer, and only a promotion moves it.
    if not (resolved / MODEL_FILENAME).is_file():
        champion = read_champion(resolved)
        if champion and champion.get("model_dir"):
            resolved = Path(champion["model_dir"])
    app = FastAPI(
        title="Autonomous ML Engineer — serving",
        description="Scores rows through a persisted run under its training contract.",
        version="1.0",
    )

    if not resolved or not (resolved / MODEL_FILENAME).is_file():
        raise FileNotFoundError(
            f"No model at {resolved!s}: expected {MODEL_FILENAME} and {SCHEMA_FILENAME}. "
            f"Point {MODEL_DIR_ENV} at a directory written by a pipeline run "
            f"(runs/models/<run_name>)."
        )

    loaded = load_model(resolved / MODEL_FILENAME, resolved / SCHEMA_FILENAME)
    estimator, schema = loaded.estimator, loaded.schema or {}
    app.state.estimator = estimator
    app.state.schema = schema
    app.state.model_dir = str(resolved)
    app.state.version_warnings = loaded.warnings

    # Logging is ON by default. A serving API that keeps no record of what it
    # predicted cannot be measured for drift later, and the moment to start
    # collecting is the first request — not the day someone wants the data.
    store = None
    if log_predictions:
        store_path = Path(store_path or os.environ.get(STORE_PATH_ENV) or
                          (resolved / DEFAULT_STORE_FILENAME))
        store = PredictionStore(store_path)
    app.state.store = store
    version = _model_version(schema)

    def _score(rows: list[dict[str, Any]], allow_unknown: bool) -> dict[str, Any]:
        result = validate_payload(rows, schema, allow_unknown=allow_unknown)
        if not result.ok:
            # 422, with every offending column named. Never a filled-in default.
            raise HTTPException(status_code=422, detail={
                "message": "Payload does not match the training schema.",
                "errors": result.error_dicts(),
                "expected_features": schema.get("feature_columns"),
            })
        batch = predict_frame(estimator, result.frame, schema)
        payload = batch.as_dict()
        payload["warnings"] = result.warnings

        for row, prediction in zip(rows, payload["predictions"]):
            request_id = new_request_id()
            prediction["request_id"] = request_id
            if store is None:
                continue
            try:
                store.log_prediction(
                    payload=row, prediction=prediction["prediction"], request_id=request_id,
                    probability=prediction["probability"], threshold=prediction["threshold"],
                    decision_rule=prediction["decision_rule"],
                    model_version=version, model_name=(schema.get("model") or {}).get("name"),
                )
            except Exception as e:  # noqa: BLE001
                # A logging failure must not deny a caller their prediction —
                # but it must not pass unnoticed either, or the drift baseline
                # silently develops holes.
                payload["warnings"].append(f"Prediction was not logged ({type(e).__name__}: {e}).")
        return payload

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "degraded" if app.state.version_warnings else "ok",
            "model_dir": app.state.model_dir,
            "model": (schema.get("model") or {}).get("name"),
            "problem_type": schema.get("problem_type"),
            "schema_version": schema.get("schema_version"),
            "n_features": len(schema.get("feature_columns") or []),
            # Surfaced rather than swallowed: an estimator unpickled under a
            # different scikit-learn minor can score differently with no error.
            "version_warnings": app.state.version_warnings,
            "model_version": version,
            "prediction_log": (store.counts() | {"path": str(store.path)}) if store else None,
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        return _serving_contract(schema)

    @app.post("/outcomes")
    def record_outcome(request: OutcomeRequest) -> dict[str, Any]:
        """Attach ground truth to a served prediction, joined on request_id."""
        if store is None:
            raise HTTPException(status_code=409, detail={
                "message": "Prediction logging is disabled, so there is nothing to attach to.",
            })
        try:
            store.record_outcome(request.request_id, request.actual, source=request.source)
        except UnknownRequestError as e:
            # 404, not a silent accept: a label with nothing to join to shows up
            # much later as an evaluation set that is quietly too small.
            raise HTTPException(status_code=404, detail={"message": str(e)}) from e
        return {"request_id": request.request_id, "recorded": True, "counts": store.counts()}

    @app.post("/outcomes/batch")
    def record_outcomes(request: BatchOutcomeRequest) -> dict[str, Any]:
        if store is None:
            raise HTTPException(status_code=409, detail={
                "message": "Prediction logging is disabled, so there is nothing to attach to.",
            })
        recorded, unknown = [], []
        for outcome in request.outcomes:
            try:
                store.record_outcome(outcome.request_id, outcome.actual, source=outcome.source)
                recorded.append(outcome.request_id)
            except UnknownRequestError:
                unknown.append(outcome.request_id)
        return {"recorded": recorded, "unknown_request_ids": unknown, "counts": store.counts()}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        payload = _score([request.row], request.allow_unknown)
        single = payload["predictions"][0]
        single["notes"] = payload["notes"]
        single["warnings"] = payload["warnings"]
        return single

    @app.post("/predict/batch")
    def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
        return _score(request.rows, request.allow_unknown)

    return app


def _default_app() -> FastAPI | None:
    """Module-level app for `uvicorn autoeng.serving.app:app`, absent until
    AUTOENG_MODEL_DIR points somewhere real."""
    try:
        return create_app()
    except FileNotFoundError:
        return None


app = _default_app()
