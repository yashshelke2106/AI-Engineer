"""
T1-2 — the prediction and outcome store.

This is the item that decides whether the rest of the roadmap is real. Drift
detection with no stored predictions can only compare a training set to
itself; it will look like it works and detect nothing. Retraining with no
arriving ground truth re-fits forever on the original static file and calls it
a lifecycle. Both failures are invisible from the outside — dashboards, green
checks, no data.

So the "done when" is a join: predictions and outcomes must come back as one
labelled evaluation frame through one function. That function is what T1-3
measures concept drift against and what T1-4 retrains on, so its shape is
pinned here rather than discovered twice.

Two properties this file is deliberate about:

  - **The raw payload is stored, not the transformed matrix.** Drift has to be
    measured in the space data arrives in. A stored design matrix stops being
    comparable the moment the pipeline changes, which is precisely when you
    most want to look at it.
  - **The log is append-only.** A prediction is a record of what was served at
    a moment; rewriting it destroys the only evidence of what the model
    actually did. Outcomes arrive later as separate rows, and a corrected
    label adds a row rather than overwriting one.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from autoeng.serving.store import PredictionStore, UnknownRequestError


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "predictions.db")


def _log(store, request_id, *, payload=None, prediction=1, probability=0.8, model="m1", when=None):
    return store.log_prediction(
        request_id=request_id,
        payload=payload or {"age": 40, "income": 50_000.0, "city": "Pune"},
        prediction=prediction,
        probability=probability,
        threshold=0.28,
        decision_rule="predict_proba >= 0.28",
        model_version=model,
        model_name="logistic_regression",
        predicted_at=when,
    )


class TestTheJoin:
    """The ROADMAP's 'done when': one function, one labelled frame."""

    def test_predictions_and_outcomes_join_into_a_labelled_frame(self, store):
        _log(store, "r1", prediction=1, probability=0.91)
        _log(store, "r2", prediction=0, probability=0.10)
        store.record_outcome("r1", actual=1)
        store.record_outcome("r2", actual=1)

        frame = store.labelled_frame()

        assert len(frame) == 2
        assert set(frame["request_id"]) == {"r1", "r2"}
        # Everything T1-3 and T1-4 need, in one place: the raw features, what
        # was predicted, and what actually happened.
        for column in ("request_id", "predicted_at", "model_version",
                       "prediction", "probability", "actual", "outcome_at"):
            assert column in frame.columns, column
        # Raw feature columns are expanded, so the frame can be fed straight
        # back into a pipeline for retraining.
        for column in ("age", "income", "city"):
            assert column in frame.columns, column

        row = frame.set_index("request_id").loc["r2"]
        assert row["prediction"] == 0 and row["actual"] == 1, "a miss must survive the join"

    def test_unlabelled_predictions_are_excluded_from_the_labelled_frame(self, store):
        _log(store, "labelled")
        _log(store, "still_waiting")
        store.record_outcome("labelled", actual=0)

        assert set(store.labelled_frame()["request_id"]) == {"labelled"}
        # ...but are still visible for DATA drift, which needs no labels at all.
        assert set(store.prediction_frame()["request_id"]) == {"labelled", "still_waiting"}

    def test_frames_are_empty_not_broken_when_nothing_is_logged(self, store):
        assert store.labelled_frame().empty
        assert store.prediction_frame().empty
        assert store.counts() == {"predictions": 0, "outcomes": 0, "labelled": 0}


class TestRawPayload:
    def test_the_payload_is_stored_exactly_as_received(self, store):
        payload = {"age": 40, "income": 50_000.0, "city": "Pune", "note": None}
        _log(store, "r1", payload=payload)
        stored = store.prediction_frame().iloc[0]
        assert json.loads(stored["payload_json"]) == payload

    def test_expanded_columns_survive_a_null(self, store):
        _log(store, "r1", payload={"age": None, "income": 1.0, "city": "Pune"})
        store.record_outcome("r1", actual=1)
        frame = store.labelled_frame()
        assert pd.isna(frame.iloc[0]["age"])
        assert frame.iloc[0]["income"] == 1.0

    def test_rows_with_different_keys_do_not_corrupt_each_other(self, store):
        """Payload shape can change across a deploy. The store keeps whatever
        arrived rather than forcing an old schema onto new rows."""
        _log(store, "old", payload={"age": 1})
        _log(store, "new", payload={"age": 2, "added_later": "x"})
        frame = store.prediction_frame().set_index("request_id")
        assert frame.loc["new", "added_later"] == "x"
        assert pd.isna(frame.loc["old", "added_later"])


class TestAppendOnly:
    def test_a_duplicate_request_id_is_rejected(self, store):
        _log(store, "r1")
        with pytest.raises(ValueError, match="already"):
            _log(store, "r1")

    def test_a_corrected_outcome_adds_a_row_and_the_latest_wins(self, store):
        """Labels get revised. Overwriting would erase the evidence that they
        were, so both are kept and the join takes the most recent."""
        _log(store, "r1")
        store.record_outcome("r1", actual=0)
        store.record_outcome("r1", actual=1)

        assert store.counts()["outcomes"] == 2, "both outcomes must be retained"
        frame = store.labelled_frame()
        assert len(frame) == 1, "the join must not duplicate the prediction"
        assert frame.iloc[0]["actual"] == 1

    def test_an_outcome_for_an_unknown_request_is_rejected(self, store):
        """Silently accepting it would create a label with nothing to join to,
        which shows up much later as a quietly shrinking evaluation set."""
        with pytest.raises(UnknownRequestError, match="never_served"):
            store.record_outcome("never_served", actual=1)


class TestFiltering:
    """T1-3 measures drift over a window; T1-5 compares champion to
    challenger. Both need to slice the log."""

    def test_since_filters_by_prediction_time(self, store):
        now = datetime.now(timezone.utc)
        _log(store, "old", when=now - timedelta(days=10))
        _log(store, "recent", when=now - timedelta(hours=1))
        store.record_outcome("old", actual=1)
        store.record_outcome("recent", actual=1)

        recent = store.labelled_frame(since=now - timedelta(days=1))
        assert set(recent["request_id"]) == {"recent"}

    def test_model_version_filters_champion_from_challenger(self, store):
        _log(store, "a", model="champion")
        _log(store, "b", model="challenger")
        store.record_outcome("a", actual=1)
        store.record_outcome("b", actual=1)

        assert set(store.labelled_frame(model_version="challenger")["request_id"]) == {"b"}


class TestDurability:
    def test_the_log_survives_reopening(self, tmp_path):
        path = tmp_path / "predictions.db"
        first = PredictionStore(path)
        _log(first, "r1")
        first.record_outcome("r1", actual=1)

        reopened = PredictionStore(path)
        assert reopened.counts() == {"predictions": 1, "outcomes": 1, "labelled": 1}
        assert reopened.labelled_frame().iloc[0]["actual"] == 1
