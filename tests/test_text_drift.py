"""
Drift on text columns — which, until this, were not compared at all.

`check_data_drift` skipped any text reference with a note, so a model that reads
the words (T2-3) had no drift signal on them, and the docs claimed length and
word count were watched when nothing was. A text column now has three numeric
views measured like any numeric feature: length, word count, and the share of
each document's words the training corpus did not know.

Measured before shipping, on real 20-newsgroups posts (reference: 980
baseball/medicine posts; windows of 300):
  - the leave-one-out reference is honest: training median unknown share 0.090,
    genuinely new same-topic posts 0.088;
  - no drift: 0/40 windows flag unknown words, 2/40 reports not ok (the FDR rate);
  - 30% of posts from four other newsgroups: 40/40 reports raised to investigate;
  - 100% from other newsgroups: 40/40 flagged. Length and word count flagged
    none of these windows — before this change, a topic switch was invisible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.common.text import build_vocabulary, coverage
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.monitoring.drift import check_data_drift
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import save_model

DATA = Path(__file__).resolve().parents[1] / "data"
SWAPS = {"invoice": "receipt", "subscription": "membership", "account": "profile", "refund": "rebate",
         "charged": "debited", "export": "backup", "feature": "function", "mobile": "tablet", "login": "signin",
         "usage": "volume", "report": "digest", "limit": "quota", "discount": "voucher", "renewal": "rollover"}


class TestTheVocabulary:
    def test_leave_one_out_scores_a_training_document_as_if_it_were_new(self):
        docs = ["apple berry", "apple berry", "apple cherry", "date elder"]
        vocabulary, loo = build_vocabulary(docs)
        assert vocabulary == ["apple", "berry"]  # seen in at least two documents
        # 'berry' is in two documents, so seen from either one it is in only ONE other.
        np.testing.assert_allclose(loo, [0.5, 0.5, 0.5, 0.0])
        # A genuinely new document sees all four, so 'berry' (in two of them) is known.
        assert coverage(["apple berry"], vocabulary)[0] == 1.0

    def test_a_document_without_words_has_no_coverage(self):
        _, loo = build_vocabulary(["", None, "apple pie", "apple tart"])
        assert np.isnan(loo[0]) and np.isnan(loo[1])
        assert np.isnan(coverage([""], ["apple"])[0])


@pytest.fixture(scope="module")
def ticket_schema(tmp_path_factory):
    train = pd.read_csv(DATA / "synthetic_text.csv")
    profile = profile_dataset(train)
    roles = assign_feature_roles(profile, target_column="escalated")
    X, y = train[roles.feature_columns], train["escalated"]
    model = _build_pipeline_for_model("logistic_regression", get_classification_models(2)["logistic_regression"],
                                      roles, "classification").fit(X, y)
    saved = save_model(model, X, y, profile, roles, problem_type="binary_classification",
                       model_name="logistic_regression", output_dir=tmp_path_factory.mktemp("model"),
                       write_mlflow_model=False)
    return json.loads(Path(saved.schema_path).read_text(encoding="utf-8")), train


def _views(report, column="ticket_text"):
    return {f.column.split("(")[1].rstrip(")"): f for f in report.features if f.column.startswith(column)}


class TestTextColumnsAreMeasured:
    def test_the_reference_carries_a_vocabulary_and_the_unknown_share(self, ticket_schema):
        schema, _ = ticket_schema
        ref = schema["columns"]["ticket_text"]["reference"]
        assert ref["kind"] == "text" and len(ref["vocabulary"]) > 20
        assert {"length", "word_count", "unknown_share"} <= set(ref)

    def test_an_unshifted_window_moves_none_of_the_views(self, ticket_schema):
        schema, train = ticket_schema
        report = check_data_drift(train.sample(300, random_state=1), schema)
        views = _views(report)
        assert set(views) == {"length", "word count", "unknown words"}
        assert all(v.severity.value == "ok" for v in views.values())

    def test_new_words_of_the_same_length_are_seen(self, ticket_schema):
        schema, train = ticket_schema
        window = train.sample(300, random_state=2).copy()
        reworded = window.index[: len(window) * 3 // 10]
        window.loc[reworded, "ticket_text"] = window.loc[reworded, "ticket_text"].map(
            lambda t: " ".join(SWAPS.get(w, w) for w in t.split()))
        views = _views(check_data_drift(window, schema))
        assert views["unknown words"].severity.value != "ok"
        assert views["length"].severity.value == "ok", "the words changed, not their length"

    def test_the_column_weighs_what_it_weighs_not_three_times_that(self, ticket_schema):
        schema, train = ticket_schema
        report = check_data_drift(train.sample(300, random_state=3), schema,
                                  importances={"ticket_text": 0.6, "channel": 0.3, "response_hours": 0.1})
        assert sum(v.importance for v in _views(report).values()) == pytest.approx(0.6)

    def test_an_artifact_from_before_vocabulary_tracking_says_what_it_cannot_see(self, ticket_schema):
        schema, train = ticket_schema
        old = json.loads(json.dumps(schema))
        for key in ("vocabulary", "unknown_share"):
            old["columns"]["ticket_text"]["reference"].pop(key)
        report = check_data_drift(train.sample(300, random_state=4), old)
        assert set(_views(report)) == {"length", "word count"}
        assert any("predates vocabulary tracking" in n for n in report.notes)


def test_a_column_constant_in_training_is_measured_not_skipped():
    """A point mass is a distribution: departing from it is drift. It used to be
    reported as unmeasurable, which hid an all-known vocabulary's unknown share."""
    rng = np.random.default_rng(5)
    reference = {"kind": "numeric", "n_observed": 500, "n_effective": 500.0, "n_effective_mean": 500.0,
                 "quantiles": {f"{q / 10:.2f}": 0.0 for q in range(11)},
                 "cdf": {f"{q / 10:.2f}": 1.0 for q in range(11)}, "mean": 0.0, "std": 0.0}
    schema = {"feature_columns": ["x"], "columns": {"x": {"reference": reference}}}
    steady = check_data_drift(pd.DataFrame({"x": np.zeros(300)}), schema)
    assert steady.features and steady.features[0].severity.value == "ok"
    moved = check_data_drift(pd.DataFrame({"x": np.where(rng.random(300) < 0.2, 0.1, 0.0)}), schema)
    assert moved.features[0].severity.value != "ok"
