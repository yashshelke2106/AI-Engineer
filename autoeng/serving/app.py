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

from autoeng.registry.model_store import MODEL_FILENAME, SCHEMA_FILENAME, load_model
from autoeng.serving.predictor import predict_frame, threshold_from_schema
from autoeng.serving.validation import validate_payload

MODEL_DIR_ENV = "AUTOENG_MODEL_DIR"


class PredictRequest(BaseModel):
    row: dict[str, Any] = Field(..., description="One record, keyed by training column name.")
    allow_unknown: bool = Field(
        False, description="Ignore columns the model was not trained on instead of rejecting them.",
    )


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., description="Records, each keyed by training column name.")
    allow_unknown: bool = False


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


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    """
    Build an app bound to one model directory.

    A factory rather than a module-level singleton so tests can point at a
    temporary directory without touching the environment.
    """
    resolved = Path(model_dir or os.environ.get(MODEL_DIR_ENV, "")).expanduser()
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
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        return _serving_contract(schema)

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
