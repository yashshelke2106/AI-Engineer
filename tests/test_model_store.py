"""
T0-1 — the persisted model must be the model that was evaluated.

`run_pipeline` fits a winner, scores it on a held-out split, explains it, and
writes a report full of numbers about it. Until T0-1 that estimator then went
out of scope and was garbage-collected, so every number in the report
described an artifact nobody had. Persisting it is only worth something if
what comes back off disk behaves *identically* to what went in — so these
tests assert equality to 1e-9, not similarity. A model that round-trips
"approximately" is a model whose report is approximately true.

Both shapes the pipeline can hand the store are covered:

  - the ordinary winner, an sklearn `Pipeline`;
  - the stacked ensemble, a bare `StackingClassifier` holding whole pipelines
    as nested base estimators.

Those nest differently enough that one round-tripping proves nothing about the
other, which is why the ROADMAP calls the stack out by name.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.ensemble import build_stacked_ensemble
from autoeng.modeling.model_zoo import get_classification_models, get_regression_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.pipeline import _classification_metric, _regression_metric_from_r2
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import load_model, save_model

# The round-trip is a pickle of the same fitted object, so the honest bar is
# exact equality. 1e-9 is the ROADMAP's stated tolerance and leaves room for
# nothing but float formatting.
TOLERANCE = 1e-9


def _split(df: pd.DataFrame, target: str, stratify: bool):
    """Mirror run_pipeline: roles first, then slice X with feature_columns."""
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column=target)
    X, y = df[roles.feature_columns], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if stratify else None,
    )
    return profile, roles, X_train, X_test, y_train, y_test


class TestRoundTripReproducesMetrics:
    """Reload the saved artifact, score the same held-out rows, demand the
    same numbers. This is the ROADMAP's "done when" for T0-1."""

    def test_pipeline_round_trips_exactly(self, clean_classification_df, tmp_path):
        profile, roles, X_train, X_test, y_train, y_test = _split(
            clean_classification_df, "churned", stratify=True,
        )
        factories = get_classification_models(n_classes=2)
        estimator = _build_pipeline_for_model(
            "random_forest", factories["random_forest"], roles, "classification",
        )
        estimator.fit(X_train, y_train)
        assert isinstance(estimator, Pipeline), "fixture must exercise the Pipeline shape"

        in_run = _classification_metric(estimator, X_test, y_test, n_classes=2)

        saved = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type="binary_classification", model_name="random_forest",
            output_dir=tmp_path,
        )
        reloaded = load_model(saved.model_path, saved.schema_path).estimator
        after_reload = _classification_metric(reloaded, X_test, y_test, n_classes=2)

        assert set(after_reload) == set(in_run)
        for name, value in in_run.items():
            assert after_reload[name] == pytest.approx(value, abs=TOLERANCE), (
                f"{name} moved across the save/load boundary: "
                f"{value!r} -> {after_reload[name]!r}"
            )
        # Metrics can coincide while individual rows disagree, so pin the
        # predictions themselves too.
        assert np.array_equal(reloaded.predict(X_test), estimator.predict(X_test))
        assert np.array_equal(reloaded.predict_proba(X_test), estimator.predict_proba(X_test))

    def test_stacking_classifier_round_trips_exactly(self, clean_classification_df, tmp_path):
        """The stack is a StackingClassifier, not a Pipeline — the store must
        take an estimator, not assume `.named_steps`."""
        profile, roles, X_train, X_test, y_train, y_test = _split(
            clean_classification_df, "churned", stratify=True,
        )
        factories = get_classification_models(n_classes=2)
        estimator = build_stacked_ensemble(
            ["decision_tree", "logistic_regression"], factories, roles, "classification",
        )
        estimator.fit(X_train, y_train)
        assert not isinstance(estimator, Pipeline), "fixture must exercise the non-Pipeline shape"

        in_run = _classification_metric(estimator, X_test, y_test, n_classes=2)

        saved = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type="binary_classification", model_name="stacked_ensemble",
            output_dir=tmp_path,
        )
        reloaded = load_model(saved.model_path, saved.schema_path).estimator
        after_reload = _classification_metric(reloaded, X_test, y_test, n_classes=2)

        for name, value in in_run.items():
            assert after_reload[name] == pytest.approx(value, abs=TOLERANCE), (
                f"{name} moved across the save/load boundary for the stacked ensemble"
            )
        assert np.array_equal(reloaded.predict(X_test), estimator.predict(X_test))

    def test_regression_pipeline_round_trips_exactly(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "area": rng.normal(1500, 400, n).round(1),
            "rooms": rng.integers(1, 6, n),
            "district": rng.choice(["north", "south", "east"], n),
        })
        df["price"] = 300 * df["area"] + 20_000 * df["rooms"] + rng.normal(0, 5_000, n)

        profile, roles, X_train, X_test, y_train, y_test = _split(df, "price", stratify=False)
        factories = get_regression_models()
        estimator = _build_pipeline_for_model(
            "random_forest", factories["random_forest"], roles, "regression",
        )
        estimator.fit(X_train, y_train)

        in_run = _regression_metric_from_r2(estimator, X_test, y_test)
        saved = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type="regression", model_name="random_forest", output_dir=tmp_path,
        )
        reloaded = load_model(saved.model_path, saved.schema_path).estimator
        after_reload = _regression_metric_from_r2(reloaded, X_test, y_test)

        for name, value in in_run.items():
            assert after_reload[name] == pytest.approx(value, abs=TOLERANCE), f"{name} drifted"


class TestMlflowExport:
    """
    The MLflow export is written with `save_model` (no active run needed) and
    reports failure instead of raising, so that a broken export cannot take
    down a run that already spent minutes searching models. That design makes
    it fail *quietly*, which is precisely why it needs an assertion.

    The concrete trap: MLflow's default sklearn serialization format is skops,
    which refuses to serialize any non-sklearn class. Every pipeline here
    embeds this project's own transformers, so the default silently produced
    no model at all.
    """

    def test_export_survives_the_projects_custom_transformers(self, clean_classification_df, tmp_path):
        profile, roles, X_train, X_test, y_train, _ = _split(
            clean_classification_df, "churned", stratify=True,
        )
        factories = get_classification_models(n_classes=2)
        estimator = _build_pipeline_for_model(
            "decision_tree", factories["decision_tree"], roles, "classification",
        )
        estimator.fit(X_train, y_train)
        assert any(
            type(step).__module__.startswith("autoeng")
            for _, step in estimator.steps
        ), "fixture must contain a custom transformer for this test to mean anything"

        saved = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type="binary_classification", model_name="decision_tree",
            output_dir=tmp_path,
        )
        assert saved.mlflow_model_dir is not None, f"export was skipped: {saved.warnings}"
        assert saved.warnings == []

        model_dir = Path(saved.mlflow_model_dir)
        assert (model_dir / "MLmodel").is_file(), "no MLmodel metadata written"
        # The ROADMAP asks for a signature and an input example specifically —
        # they are what T1-1 validates serving payloads against.
        mlmodel = (model_dir / "MLmodel").read_text()
        assert "signature" in mlmodel
        assert "input_example" in mlmodel

        import mlflow.sklearn
        via_mlflow = mlflow.sklearn.load_model(str(model_dir))
        assert np.array_equal(via_mlflow.predict(X_test), estimator.predict(X_test))


class TestTrainingSchema:
    """The schema is what T1-1 validates payloads against and what T1-3 diffs
    drift against. Both are downstream, so the shape is pinned here."""

    @pytest.fixture
    def saved(self, clean_classification_df, tmp_path):
        profile, roles, X_train, _, y_train, _ = _split(
            clean_classification_df, "churned", stratify=True,
        )
        factories = get_classification_models(n_classes=2)
        estimator = _build_pipeline_for_model(
            "decision_tree", factories["decision_tree"], roles, "classification",
        )
        estimator.fit(X_train, y_train)
        result = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type="binary_classification", model_name="decision_tree",
            output_dir=tmp_path,
        )
        return load_model(result.model_path, result.schema_path).schema, X_train

    def test_records_serving_contract(self, saved):
        schema, X_train = saved
        # Column ORDER is part of the contract: a positional mismatch at serving
        # time silently scores the wrong columns rather than raising.
        assert schema["feature_columns"] == list(X_train.columns)
        assert "customer_id" not in schema["feature_columns"], "identifier must not be a serving input"
        assert schema["target"]["column"] == "churned"
        assert schema["target"]["class_labels"] == [0, 1]
        # An estimator unpickled under a different sklearn minor version is a
        # silent correctness risk, so the version it was fit under is recorded.
        assert "scikit-learn" in schema["library_versions"]

    def test_captures_numeric_reference_distribution(self, saved):
        schema, X_train = saved
        ref = schema["columns"]["income"]["reference"]
        assert ref["kind"] == "numeric"
        quantiles = [v for _, v in sorted(ref["quantiles"].items(), key=lambda kv: float(kv[0]))]
        assert quantiles == sorted(quantiles), "quantiles must be non-decreasing to serve as PSI bin edges"
        assert ref["min"] == pytest.approx(float(X_train["income"].min()))
        assert ref["max"] == pytest.approx(float(X_train["income"].max()))

    def test_captures_categorical_reference_distribution(self, saved):
        schema, X_train = saved
        ref = schema["columns"]["city"]["reference"]
        assert ref["kind"] == "categorical"
        assert set(ref["frequencies"]) == set(X_train["city"].unique())
        # Frequencies, not counts — drift compares distributions across windows
        # of different sizes.
        assert sum(ref["frequencies"].values()) == pytest.approx(1.0)
        expected = X_train["city"].value_counts(normalize=True)
        for category, freq in ref["frequencies"].items():
            assert freq == pytest.approx(float(expected[category]))
