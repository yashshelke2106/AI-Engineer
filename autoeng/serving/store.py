"""
The prediction and outcome log.

This is the step most implementations skip, and skipping it is how a demo ends
up "auto-retraining" forever on the original static file. Without stored
predictions there is no drift baseline — a drift detector with nothing logged
can only compare a training set to itself, which looks like it works and
detects nothing. Without ground truth arriving later there is no concept-drift
measurement and no retraining data.

Two tables, both append-only:

    predictions   what was served, when, by which model, and on what input
    outcomes      what actually happened, arriving minutes or months later

They are separate because they arrive at different times. Joining them is
`labelled_frame()`, and that single function is what T1-3 measures concept
drift against and what T1-4 retrains on — one definition of "a labelled
evaluation window", not two that drift apart.

## Why append-only

A prediction is a record of what the system did at a moment. Rewriting it
destroys the only evidence of what was actually served, which is exactly what
you need when someone asks why a decision was made. So a duplicate
`request_id` is an error rather than an overwrite.

Outcomes are append-only for a subtler reason: labels get revised. A chargeback
is reversed, a diagnosis is corrected. Overwriting erases the fact that the
label changed — which is itself a signal, and occasionally the explanation for
a model that appears to have degraded. Both rows are kept and the join takes
the most recent per request.

## Why the raw payload

The stored payload is the JSON that arrived, not the transformed design
matrix. Drift has to be measured in the space data arrives in, and a stored
matrix stops being comparable the moment the pipeline changes — precisely when
you most want to look at it. It also means the log can be replayed into a
retrained pipeline, which a matrix could not be.

SQLite keeps this a zero-external-services setup, consistent with the MLflow
store. WAL mode is on so a reader (drift check) does not block the writer
(serving).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id     TEXT PRIMARY KEY,
    predicted_at   TEXT NOT NULL,
    model_version  TEXT,
    model_name     TEXT,
    payload_json   TEXT NOT NULL,
    prediction     TEXT,
    probability    REAL,
    threshold      REAL,
    decision_rule  TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_time    ON predictions (predicted_at);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions (model_version);

-- Append-only: a revised label adds a row, it does not replace one. The join
-- takes MAX(rowid) per request, so the newest wins without losing the history.
CREATE TABLE IF NOT EXISTS outcomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT NOT NULL REFERENCES predictions (request_id),
    outcome_at   TEXT NOT NULL,
    actual_json  TEXT NOT NULL,
    source       TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_request ON outcomes (request_id);
"""


class UnknownRequestError(KeyError):
    """An outcome arrived for a request_id that was never served.

    Rejected rather than accepted, because a label with nothing to join to is
    invisible: it shows up much later as an evaluation window that is quietly
    smaller than the number of labels collected.
    """


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def new_request_id() -> str:
    return uuid.uuid4().hex


class PredictionStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL so a drift check reading the log does not block serving.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writing -----------------------------------------------------------

    def log_prediction(
        self,
        payload: dict[str, Any],
        prediction: Any,
        *,
        request_id: str | None = None,
        probability: float | None = None,
        threshold: float | None = None,
        decision_rule: str | None = None,
        model_version: str | None = None,
        model_name: str | None = None,
        predicted_at: datetime | str | None = None,
    ) -> str:
        """Record one served prediction. Returns the request_id to attach an outcome to."""
        request_id = request_id or new_request_id()
        when = predicted_at.isoformat() if isinstance(predicted_at, datetime) else (predicted_at or _now())
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO predictions (request_id, predicted_at, model_version, model_name,"
                    " payload_json, prediction, probability, threshold, decision_rule)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (request_id, when, model_version, model_name,
                     json.dumps({k: _jsonable(v) for k, v in payload.items()}),
                     json.dumps(_jsonable(prediction)),
                     None if probability is None else float(probability),
                     None if threshold is None else float(threshold),
                     decision_rule),
                )
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f"request_id '{request_id}' has already been logged. The prediction log is "
                f"append-only: overwriting it would destroy the record of what was actually "
                f"served."
            ) from e
        return request_id

    def record_outcome(self, request_id: str, actual: Any, source: str | None = None) -> None:
        """Attach ground truth to a served prediction. Revisions add a row."""
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)
            ).fetchone()
            if exists is None:
                raise UnknownRequestError(
                    f"No prediction was served for request_id '{request_id}', so there is "
                    f"nothing for this outcome to join to."
                )
            conn.execute(
                "INSERT INTO outcomes (request_id, outcome_at, actual_json, source)"
                " VALUES (?, ?, ?, ?)",
                (request_id, _now(), json.dumps(_jsonable(actual)), source),
            )

    # -- reading -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
                "outcomes": conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0],
                "labelled": conn.execute(
                    "SELECT COUNT(DISTINCT request_id) FROM outcomes"
                ).fetchone()[0],
            }

    def _query(self, sql: str, params: list[Any]) -> pd.DataFrame:
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        return pd.DataFrame(rows)

    @staticmethod
    def _expand_payloads(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Widen the stored JSON back into feature columns.

        Rows are unioned rather than intersected: payload shape can change
        across a deploy, and a column absent from older rows becomes NaN rather
        than truncating the frame to whatever the oldest row happened to carry.
        """
        if frame.empty:
            return frame
        payloads = [json.loads(p) for p in frame["payload_json"]]
        expanded = pd.DataFrame(payloads, index=frame.index)
        # Never let a feature name shadow a log column.
        clashes = set(expanded.columns) & set(frame.columns)
        if clashes:
            expanded = expanded.rename(columns={c: f"feature_{c}" for c in clashes})
        return pd.concat([frame, expanded], axis=1)

    def prediction_frame(
        self, since: datetime | None = None, model_version: str | None = None,
    ) -> pd.DataFrame:
        """
        Every served prediction, labelled or not.

        This is what DATA drift reads: it compares incoming feature
        distributions against the training reference distributions captured in
        T0-1, and needs no ground truth at all.
        """
        sql = "SELECT * FROM predictions WHERE 1=1"
        params: list[Any] = []
        if since is not None:
            sql += " AND predicted_at >= ?"
            params.append(since.isoformat())
        if model_version is not None:
            sql += " AND model_version = ?"
            params.append(model_version)
        sql += " ORDER BY predicted_at"

        frame = self._query(sql, params)
        if frame.empty:
            return frame
        frame["prediction"] = [json.loads(v) if v is not None else None for v in frame["prediction"]]
        return self._expand_payloads(frame)

    def labelled_frame(
        self, since: datetime | None = None, model_version: str | None = None,
    ) -> pd.DataFrame:
        """
        Predictions joined to the ground truth that arrived for them.

        **The one function T1-3 and T1-4 both consume.** Concept drift is
        rolling performance over this frame; retraining is a re-fit over it
        plus the original training data. Defining it once is what keeps those
        two from disagreeing about what a labelled window is.

        Only the most recent outcome per request is used — see the note on
        append-only outcomes in the module docstring.
        """
        sql = (
            "SELECT p.*, o.actual_json, o.outcome_at, o.source AS outcome_source"
            " FROM predictions p"
            " JOIN outcomes o ON o.request_id = p.request_id"
            " JOIN (SELECT request_id, MAX(id) AS latest FROM outcomes GROUP BY request_id) m"
            "   ON m.latest = o.id"
            " WHERE 1=1"
        )
        params: list[Any] = []
        if since is not None:
            sql += " AND p.predicted_at >= ?"
            params.append(since.isoformat())
        if model_version is not None:
            sql += " AND p.model_version = ?"
            params.append(model_version)
        sql += " ORDER BY p.predicted_at"

        frame = self._query(sql, params)
        if frame.empty:
            return frame
        frame["prediction"] = [json.loads(v) if v is not None else None for v in frame["prediction"]]
        frame["actual"] = [json.loads(v) for v in frame["actual_json"]]
        frame = frame.drop(columns=["actual_json"])
        return self._expand_payloads(frame)
