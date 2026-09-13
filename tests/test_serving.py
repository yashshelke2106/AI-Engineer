"""
T1-1 — the serving API.

Two "done when" assertions from the ROADMAP anchor this file:

  1. A row scored through the API exactly matches the same row scored
     in-process. If they diverge, the report describes one model and
     production runs another.
  2. A request missing one feature returns 422 naming it.

The second is the design commitment, and it is worth being explicit about why
it is a test rather than a convention. The pipeline contains an imputer. When a
payload arrives without a feature, filling it with the training median makes
the request succeed and returns a confident, plausible, entirely fabricated
prediction — with nothing in the response saying a value was invented. It is
the most comfortable wrong behaviour available, which is exactly why it needs
an assertion holding it shut.

The third thread here is the decision threshold. T0-2 selected one out-of-fold
and stored it in the schema; `estimator.predict()` ignores it and uses 0.5. A
serving layer that calls `predict()` silently deploys a different operating
point from the one the report measured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.model_zoo import get_classification_models, get_regression_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.modeling.threshold import out_of_fold_probabilities, select_threshold
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import MODEL_FILENAME, SCHEMA_FILENAME, save_model
from autoeng.serving.app import create_app


def _row(frame: pd.DataFrame, i: int) -> dict:
    """One record as JSON would carry it — NaN is not valid JSON."""
    return {k: (None if pd.isna(v) else v) for k, v in frame.iloc[i].to_dict().items()}


def _persist(df, target, tmp_path, model_name="decision_tree", kind="classification",
             with_threshold=True):
    """Train and persist a model the way run_pipeline does, then serve it."""
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column=target)
    X, y = df[roles.feature_columns], df[target]
    factories = (get_classification_models(n_classes=int(y.nunique()))
                 if kind == "classification" else get_regression_models())
    estimator = _build_pipeline_for_model(model_name, factories[model_name], roles, kind)

    threshold = None
    if with_threshold and kind == "classification" and y.nunique() == 2:
        proba = out_of_fold_probabilities(estimator, X, y, cv_folds=3)
        threshold = select_threshold(y, proba, objective="f1").as_dict()

    estimator.fit(X, y)
    problem_type = "binary_classification" if kind == "classification" else "regression"
    saved = save_model(
        estimator, X, y, profile, roles, problem_type=problem_type,
        model_name=model_name, output_dir=tmp_path / "model",
        decision_threshold=threshold, write_mlflow_model=False,
    )
    return estimator, X, saved


@pytest.fixture
def served(clean_classification_df, tmp_path):
    estimator, X, saved = _persist(clean_classification_df, "churned", tmp_path)
    client = TestClient(create_app(tmp_path / "model"))
    return client, estimator, X, saved


class TestScoringMatchesInProcess:
    """ROADMAP done-when #1."""

    def test_api_row_matches_the_same_row_scored_in_process(self, served):
        client, estimator, X, _ = served
        for i in range(5):
            response = client.post("/predict", json={"row": _row(X, i)})
            assert response.status_code == 200, response.text

            in_process = float(estimator.predict_proba(X.iloc[[i]])[:, 1][0])
            assert response.json()["probability"] == pytest.approx(in_process, abs=1e-12), (
                "the served probability must be the in-process probability, exactly"
            )

    def test_batch_matches_row_by_row(self, served):
        client, estimator, X, _ = served
        rows = [_row(X, i) for i in range(8)]
        body = client.post("/predict/batch", json={"rows": rows}).json()
        expected = estimator.predict_proba(X.iloc[:8])[:, 1]
        got = [p["probability"] for p in body["predictions"]]
        assert np.allclose(got, expected, atol=1e-12)

    def test_column_order_is_taken_from_the_schema_not_the_payload(self, served):
        """A frame built from dict keys inherits insertion order. If serving
        trusted that, a caller sending fields in a different order would score
        the wrong columns against each other — silently, with no error."""
        client, estimator, X, _ = served
        row = _row(X, 0)
        shuffled = dict(reversed(list(row.items())))
        assert list(shuffled) != list(row), "fixture must actually reorder the keys"

        straight = client.post("/predict", json={"row": row}).json()
        reordered = client.post("/predict", json={"row": shuffled}).json()
        assert reordered["probability"] == pytest.approx(straight["probability"], abs=1e-12)


class TestStrictValidation:
    """ROADMAP done-when #2, and the rest of the contract."""

    def test_missing_feature_returns_422_naming_it(self, served):
        client, _, X, _ = served
        row = _row(X, 0)
        dropped = X.columns[0]
        del row[dropped]

        response = client.post("/predict", json={"row": row})
        assert response.status_code == 422
        detail = response.json()["detail"]
        errors = detail["errors"]
        assert any(e["kind"] == "missing_feature" and e["column"] == dropped for e in errors), errors
        # The name has to appear, not just a count — "a field is missing" sends
        # the caller looking through every field.
        assert dropped in str(detail)

    def test_missing_feature_is_never_quietly_imputed(self, served):
        """The failure this endpoint exists to prevent: a 200 with a
        fabricated value behind it."""
        client, _, X, _ = served
        row = _row(X, 0)
        del row["income"]
        response = client.post("/predict", json={"row": row})
        assert response.status_code != 200
        assert "prediction" not in response.json()

    def test_explicit_null_is_accepted_and_warned_about(self, served):
        """A null is ordinary missing data the pipeline's imputer handles —
        rejecting it would make the API stricter than the model. It differs
        from an absent column, which is a contract violation."""
        client, _, X, _ = served
        row = _row(X, 0)
        row["income"] = None
        response = client.post("/predict", json={"row": row})
        assert response.status_code == 200, response.text
        assert any("income" in w for w in response.json()["warnings"])

    def test_unknown_column_is_rejected_by_default(self, served):
        client, _, X, _ = served
        row = _row(X, 0)
        row["a_field_from_the_future"] = 1
        response = client.post("/predict", json={"row": row})
        assert response.status_code == 422
        assert "a_field_from_the_future" in str(response.json()["detail"])

    def test_unknown_column_can_be_ignored_explicitly(self, served):
        client, _, X, _ = served
        row = _row(X, 0)
        row["a_field_from_the_future"] = 1
        response = client.post("/predict", json={"row": row, "allow_unknown": True})
        assert response.status_code == 200
        assert any("a_field_from_the_future" in w for w in response.json()["warnings"])

    def test_uncoercible_value_is_an_error_not_a_silent_nan(self, served):
        """pd.to_numeric(errors="coerce") would turn "N/A" into NaN, which the
        imputer replaces with a median — the invented-value problem again,
        through a different door."""
        client, _, X, _ = served
        row = _row(X, 0)
        row["income"] = "not-a-number"
        response = client.post("/predict", json={"row": row})
        assert response.status_code == 422
        assert any(e["kind"] == "uncoercible_value" for e in response.json()["detail"]["errors"])

    def test_empty_batch_is_rejected(self, served):
        client, _, _, _ = served
        assert client.post("/predict/batch", json={"rows": []}).status_code == 422


class TestThresholdIsApplied:
    def test_labels_come_from_the_stored_threshold_not_from_predict(
        self, imbalanced_classification_df, tmp_path,
    ):
        """The integration T0-2 exists for. On a 3.9%-positive dataset the
        tuned threshold sits far below 0.5, so rows between the two get
        opposite labels from `predict()` and from the served model. The served
        one is the operating point the report measured."""
        # gradient_boosting rather than a bare tree: an unpruned decision tree
        # on rare positives emits hard 0/1 probabilities, so its F1-optimal cut
        # lands at 1.0 and there is no gap between the tuned threshold and the
        # default for this test to inspect.
        estimator, X, _ = _persist(imbalanced_classification_df, "is_fraud", tmp_path,
                                   model_name="gradient_boosting")
        client = TestClient(create_app(tmp_path / "model"))

        contract = client.get("/model").json()
        threshold = contract["decision_threshold"]["value"]
        assert threshold is not None and threshold < 0.5, (
            "fixture must produce a tuned threshold below the default for this to bite"
        )

        proba = estimator.predict_proba(X)[:, 1]
        between = np.where((proba >= threshold) & (proba < 0.5))[0]
        assert len(between) > 0, "fixture must have rows between the two thresholds"

        i = int(between[0])
        body = client.post("/predict", json={"row": _row(X, i)}).json()

        assert body["prediction"] == 1, "served label must use the stored threshold"
        assert estimator.predict(X.iloc[[i]])[0] == 0, "predict() disagrees — that is the point"
        assert body["threshold"] == pytest.approx(threshold)
        assert "predict_proba >=" in body["decision_rule"]

    def test_regression_reports_no_threshold_rather_than_inventing_one(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
        df["y"] = 3 * df["a"] + rng.normal(0, 0.2, 120)
        estimator, X, _ = _persist(df, "y", tmp_path, model_name="random_forest", kind="regression")
        client = TestClient(create_app(tmp_path / "model"))

        body = client.post("/predict", json={"row": _row(X, 0)}).json()
        assert body["threshold"] is None
        assert body["decision_rule"] == "estimator.predict"
        assert body["prediction"] == pytest.approx(float(estimator.predict(X.iloc[[0]])[0]), abs=1e-12)


class TestIntrospection:
    def test_the_root_sends_a_browser_to_the_docs(self, served):
        """Opening the server's address used to show a bare 404."""
        client, _, _, _ = served
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/docs"
        assert client.get("/docs").status_code == 200

    def test_model_endpoint_publishes_the_full_contract(self, served):
        client, _, X, _ = served
        contract = client.get("/model").json()
        assert contract["feature_columns"] == list(X.columns), "order is part of the contract"
        assert {f["name"] for f in contract["features"]} == set(X.columns)
        assert all(f["required"] for f in contract["features"])
        assert contract["target"]["column"] == "churned"
        assert contract["library_versions"]

    def test_health_reports_the_loaded_model(self, served):
        client, _, X, _ = served
        health = client.get("/health").json()
        assert health["status"] in ("ok", "degraded")
        assert health["n_features"] == len(X.columns)
        assert health["model"] == "decision_tree"

    def test_missing_model_dir_fails_loudly_at_startup(self, tmp_path):
        """Better to refuse to start than to serve 500s per request."""
        with pytest.raises(FileNotFoundError) as e:
            create_app(tmp_path / "nothing_here")
        assert MODEL_FILENAME in str(e.value)
        assert SCHEMA_FILENAME in str(e.value)
