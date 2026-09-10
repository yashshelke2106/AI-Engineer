"""
T1-2 at the HTTP boundary: predict, wait, label, join.

`tests/test_prediction_store.py` covers the store itself. This file covers the
loop as a caller actually experiences it — because the store is only useful if
the API hands back a `request_id` the caller can quote later. Without that
there is no way to attach ground truth to anything, and the whole tier
downstream is built on a log that can never be labelled.

The end-to-end assertion is the one that matters: rows served through
`/predict` and labelled through `/outcomes` come back out of
`labelled_frame()` as a frame that could be scored or retrained on.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autoeng.serving.app import create_app
from autoeng.serving.store import PredictionStore
from tests.test_serving import _persist, _row


@pytest.fixture
def served_with_log(clean_classification_df, tmp_path):
    estimator, X, _ = _persist(clean_classification_df, "churned", tmp_path)
    client = TestClient(create_app(tmp_path / "model", store_path=tmp_path / "log.db"))
    return client, estimator, X, PredictionStore(tmp_path / "log.db")


class TestTheLoop:
    def test_predict_then_label_then_join(self, served_with_log):
        client, _, X, store = served_with_log

        served = []
        for i in range(6):
            body = client.post("/predict", json={"row": _row(X, i)}).json()
            assert body["request_id"], "a caller cannot attach an outcome without this"
            served.append(body["request_id"])

        for request_id, actual in zip(served, [1, 0, 1, 1, 0, 0]):
            response = client.post("/outcomes", json={"request_id": request_id, "actual": actual})
            assert response.status_code == 200, response.text

        frame = store.labelled_frame()
        assert len(frame) == 6
        assert set(frame["request_id"]) == set(served)
        # Feature columns come back too, so this frame can be retrained on
        # directly — the property T1-4 depends on.
        for column in X.columns:
            assert column in frame.columns, column
        assert frame["actual"].tolist() == [1, 0, 1, 1, 0, 0]

    def test_batch_predictions_each_get_their_own_request_id(self, served_with_log):
        client, _, X, store = served_with_log
        rows = [_row(X, i) for i in range(4)]
        body = client.post("/predict/batch", json={"rows": rows}).json()

        ids = [p["request_id"] for p in body["predictions"]]
        assert len(set(ids)) == 4, "ids must be unique or outcomes attach to the wrong row"
        assert store.counts()["predictions"] == 4

    def test_the_logged_payload_is_what_the_caller_sent(self, served_with_log):
        client, _, X, store = served_with_log
        sent = _row(X, 0)
        client.post("/predict", json={"row": sent})

        logged = store.prediction_frame().iloc[0]
        for key, value in sent.items():
            if value is None:
                continue
            assert logged[key] == pytest.approx(value) if isinstance(value, float) else logged[key] == value

    def test_rejected_payloads_are_not_logged(self, served_with_log):
        """A 422 never reached the model, so logging it would put rows in the
        drift baseline that were never scored."""
        client, _, X, store = served_with_log
        row = _row(X, 0)
        del row["income"]
        assert client.post("/predict", json={"row": row}).status_code == 422
        assert store.counts()["predictions"] == 0


class TestOutcomeErrors:
    def test_outcome_for_an_unserved_request_is_404(self, served_with_log):
        client, _, _, _ = served_with_log
        response = client.post("/outcomes", json={"request_id": "nope", "actual": 1})
        assert response.status_code == 404
        assert "nope" in str(response.json()["detail"])

    def test_batch_outcomes_report_which_ids_were_unknown(self, served_with_log):
        """Partial success is reported rather than failing the whole batch —
        one stale id should not discard a thousand good labels."""
        client, _, X, _ = served_with_log
        request_id = client.post("/predict", json={"row": _row(X, 0)}).json()["request_id"]

        body = client.post("/outcomes/batch", json={"outcomes": [
            {"request_id": request_id, "actual": 1},
            {"request_id": "stale", "actual": 0},
        ]}).json()
        assert body["recorded"] == [request_id]
        assert body["unknown_request_ids"] == ["stale"]


class TestLoggingIsOptionalButOn:
    def test_health_reports_the_log(self, served_with_log):
        client, _, X, _ = served_with_log
        client.post("/predict", json={"row": _row(X, 0)})
        log = client.get("/health").json()["prediction_log"]
        assert log["predictions"] == 1
        assert log["path"].endswith("log.db")

    def test_logging_can_be_disabled_and_outcomes_then_409(self, clean_classification_df, tmp_path):
        _persist(clean_classification_df, "churned", tmp_path)
        client = TestClient(create_app(tmp_path / "model", log_predictions=False))

        body = client.post("/predict", json={"row": {"age": 40, "income": 5e4, "city": "Pune"}})
        assert body.status_code == 200
        assert client.get("/health").json()["prediction_log"] is None
        # Attaching an outcome must fail loudly rather than appear to succeed.
        assert client.post("/outcomes", json={"request_id": "x", "actual": 1}).status_code == 409

    def test_logging_defaults_on_beside_the_model(self, clean_classification_df, tmp_path):
        """The moment to start collecting is the first request, not the day
        someone wants the data — so the default must not be off."""
        _persist(clean_classification_df, "churned", tmp_path)
        client = TestClient(create_app(tmp_path / "model"))
        client.post("/predict", json={"row": {"age": 40, "income": 5e4, "city": "Pune"}})
        assert (tmp_path / "model" / "predictions.db").is_file()
