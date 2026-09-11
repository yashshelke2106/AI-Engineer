"""
T1-3 composed: all three checks over a real prediction log.

`test_drift.py` covers the measures. This covers the verdict, and the verdict
has one property worth defending in a test: **it is not the maximum of the
three checks.**

Data drift alarming on its own means the inputs moved. That may or may not
matter, and treating it as a failure is exactly the over-claim this module was
built to avoid. Concept drift alarming means the model is measurably worse,
which always matters. So concept drift dominates when it has the labels to
speak, and the distribution checks are leading indicators that inform rather
than decide.

The other property: UNKNOWN never collapses into OK. A window with no labels
and a window full of good ones look the same on a dashboard if you conflate
them, and they mean opposite things.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.common.roles import assign_feature_roles
from autoeng.monitoring.drift import DriftSeverity
from autoeng.monitoring.report import run_drift_report
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import build_training_schema
from autoeng.serving.store import PredictionStore


class _Estimator:
    classes_ = np.array([0, 1])


def _training(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": rng.lognormal(4.0, 0.8, n).round(2),
        "tenure": rng.normal(500, 120, n).round(1),
        "noise": rng.normal(0, 1, n).round(3),
        "y": rng.integers(0, 2, n),
    })


@pytest.fixture
def schema():
    df = _training()
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y")
    built = build_training_schema(
        _Estimator(), df[roles.feature_columns], df["y"], profile, roles,
        problem_type="binary_classification", model_name="test",
    )
    built["feature_importances"] = {"amount": 0.75, "tenure": 0.23, "noise": 0.02}
    built["baseline_metrics"] = {"precision": 0.80, "recall": 0.78, "f1": 0.79}
    built["baseline_source"] = "out-of-fold operating point on the training partition"
    return built


def _serve(store, n, *, seed, amount_mu=4.0, correct=True, label=True, version="v1"):
    rng = np.random.default_rng(seed)
    for i in range(n):
        payload = {
            "amount": float(rng.lognormal(amount_mu, 0.8)),
            "tenure": float(rng.normal(500, 120)),
            "noise": float(rng.normal(0, 1)),
        }
        prediction = int(rng.integers(0, 2))
        request_id = store.log_prediction(
            payload=payload, prediction=prediction, probability=float(rng.beta(2, 5)),
            threshold=0.3, decision_rule="predict_proba >= 0.3", model_version=version,
            model_name="test",
        )
        if label:
            # `correct` controls whether the model is still right; that is the
            # difference between concept drift and none.
            actual = prediction if (correct and rng.uniform() < 0.8) else int(1 - prediction)
            store.record_outcome(request_id, actual=actual)


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "log.db")


class TestTheVerdict:
    def test_a_healthy_stream_reads_ok(self, store, schema):
        _serve(store, 300, seed=1)
        report = run_drift_report(store, schema)
        assert report.severity == DriftSeverity.OK, report.summary
        assert report.data.severity == DriftSeverity.OK

    def test_input_drift_alone_does_not_claim_degradation(self, store, schema):
        """The distinction the module exists for: inputs moved, performance
        did not. That is a leading indicator, and the summary has to say so
        rather than announce a broken model."""
        _serve(store, 300, seed=2, amount_mu=5.4)
        report = run_drift_report(store, schema)

        assert report.data.severity == DriftSeverity.ALARM
        assert report.severity == DriftSeverity.ALARM
        assert "not been shown to drop" in report.summary, report.summary

    def test_concept_drift_dominates_the_verdict(self, store, schema):
        """Performance has collapsed while the inputs look normal — the model's
        relationship to the world changed, not its population. That is the
        worst case and the summary must name it."""
        _serve(store, 300, seed=3, correct=False)
        report = run_drift_report(store, schema)

        assert report.concept.severity == DriftSeverity.ALARM
        assert report.severity == DriftSeverity.ALARM
        assert "performance has dropped" in report.summary.lower()
        assert "relationship itself has changed" in report.summary

    def test_unlabelled_window_is_unknown_not_ok(self, store, schema):
        _serve(store, 300, seed=4, label=False)
        report = run_drift_report(store, schema)
        assert report.concept.severity == DriftSeverity.UNKNOWN
        assert report.severity != DriftSeverity.OK
        assert "labels are still arriving" in report.summary

    def test_an_empty_window_is_unknown(self, store, schema):
        report = run_drift_report(store, schema)
        assert report.severity == DriftSeverity.UNKNOWN
        assert report.window_rows == 0


class TestHonestGaps:
    def test_a_window_spanning_two_model_versions_is_called_out(self, store, schema):
        """Drift measured across a deploy boundary mixes two models' behaviour
        and attributes it to neither."""
        _serve(store, 150, seed=5, version="v1")
        _serve(store, 150, seed=6, version="v2")
        report = run_drift_report(store, schema)
        assert any("model versions" in n for n in report.notes), report.notes

    def test_isolating_one_version_removes_the_warning(self, store, schema):
        _serve(store, 150, seed=7, version="v1")
        _serve(store, 150, seed=8, version="v2")
        report = run_drift_report(store, schema, model_version="v2")
        assert not any("model versions" in n for n in report.notes)
        assert report.model_version == "v2"

    def test_a_missing_baseline_is_reported_not_assumed(self, store, schema):
        _serve(store, 300, seed=9)
        del schema["baseline_metrics"]
        report = run_drift_report(store, schema)
        assert report.concept is None
        assert any("no performance baseline" in n.lower() for n in report.notes), report.notes


class TestRendering:
    def test_markdown_shows_raw_and_weighted_side_by_side(self, store, schema):
        _serve(store, 300, seed=10, amount_mu=5.4)
        markdown = run_drift_report(store, schema).as_markdown()
        assert "| Feature | PSI | Noise floor | Importance | Weighted | Severity |" in markdown
        assert "Drift is not degradation" in markdown

    def test_report_is_json_serialisable(self, store, schema):
        import json
        _serve(store, 300, seed=11)
        json.dumps(run_drift_report(store, schema).as_dict())
