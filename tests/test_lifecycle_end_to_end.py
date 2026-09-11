"""
T1-4 and T1-5 composed: alarm, challenger, gate, and the answer afterwards.

The ROADMAP's two remaining "done when" clauses meet here:

  - a drift alarm produces a complete challenger run, logged to MLflow as a
    child of the champion;
  - an intentionally degraded challenger is rejected and stays out, and
    `ask <run_id> "why did you reject the latest model"` returns the actual
    comparison figures.

The second clause is why the gate logs an interval rather than a sentence. A
system that can say "rejected" but not "rejected because the 95% CI on the
paired difference was [-0.08, -0.02] over 400 rows" is asking to be trusted
rather than showing its work — and it is the showing that makes the rejection
auditable a year later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.explain.qa import answer_question
from autoeng.lifecycle.gate import GateVerdict, evaluate_gate
from autoeng.lifecycle.retrain import RetrainTrigger, build_retraining_frame, should_retrain
from autoeng.monitoring.drift import DriftSeverity
from autoeng.monitoring.report import DriftReport
from autoeng.serving.store import PredictionStore
from autoeng.tracking.mlflow_tracker import log_pipeline_run


def _minimal_run_payload(**overrides):
    payload = {
        "source_path": "data.csv",
        "ingestion_report": {"file_format": "csv", "detected_encoding": "utf-8", "warnings": []},
        "profile_summary": {"n_rows": 100, "n_cols": 3, "columns": {}},
        "problem_decision": {"chosen": {"problem_type": "binary_classification",
                                         "target_column": "y", "time_column": None,
                                         "reasoning": []},
                              "confidence": 0.9, "alternatives": []},
        "structural_cleaning_report": {"actions": []},
        "role_assignment": {},
        "pre_training_leakage": {"flags": []},
        "leaderboard": {"problem_kind": "classification", "primary_metric": "roc_auc",
                         "results": [{"name": "random_forest", "status": "ok",
                                      "metrics": {"roc_auc": 0.8}, "fit_time_seconds": 1.0,
                                      "error": None, "evaluation_stage": "full"}]},
        "hpo_results": [],
        "post_training_leakage": {"flags": []},
        "explanation": {"winner_name": "random_forest", "winner_score": 0.8,
                         "runner_up_name": None, "runner_up_score": None, "margin": None,
                         "hpo_improvement": None, "feature_importances": [],
                         "importance_method": "n/a", "narrative": "test"},
        "final_report_text": "# test",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "log.db")


def _serve_and_label(store, n, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        request_id = store.log_prediction(
            payload={"amount": float(rng.lognormal(4, 0.8)), "tenure": float(rng.normal(500, 120))},
            prediction=int(rng.integers(0, 2)), probability=float(rng.beta(2, 5)),
            model_version="champion", model_name="rf",
        )
        store.record_outcome(request_id, actual=int(rng.integers(0, 2)))


class TestAlarmToChallenger:
    def test_an_alarm_with_labels_produces_a_retraining_frame(self, tmp_path, store):
        rng = np.random.default_rng(0)
        original = pd.DataFrame({
            "amount": rng.lognormal(4, 0.8, 300).round(2),
            "tenure": rng.normal(500, 120, 300).round(1),
            "y": rng.integers(0, 2, 300),
        })
        path = tmp_path / "original.csv"
        original.to_csv(path, index=False)
        _serve_and_label(store, 120)

        schema = {"feature_columns": ["amount", "tenure"], "target": {"column": "y"},
                  "problem_type": "binary_classification", "feature_roles": {}}
        decision = should_retrain(
            DriftReport(model_version="champion", window_rows=120,
                        severity=DriftSeverity.ALARM, summary="shifted"),
            store,
        )
        assert decision.triggered and decision.trigger == RetrainTrigger.DRIFT_ALARM

        frame, report = build_retraining_frame(path, store, schema)
        assert report["n_new_rows"] == 120
        assert len(frame) == 420
        assert frame["y"].notna().all()


class TestRejectionIsAnswerable:
    """The ROADMAP's 'free win': the question answers itself from the logged
    numbers, with no new retrieval code."""

    def test_a_degraded_challenger_is_rejected_and_the_reason_is_retrievable(self, tmp_path):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 400)
        champion = np.where(rng.uniform(size=400) < 0.88, y, 1 - y)
        challenger = np.where(rng.uniform(size=400) < 0.55, y, 1 - y)

        decision = evaluate_gate(y, champion, challenger, metric="f1", n_bootstrap=400)
        assert decision.verdict == GateVerdict.REJECTED
        assert not decision.promote

        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        run_id = log_pipeline_run(
            tracking_uri=tracking_uri, run_name="challenger",
            promotion_decision=decision.as_dict(), **_minimal_run_payload(),
        )

        answer = answer_question(tracking_uri, run_id, "why did you reject the latest model?")
        # Real figures, not a stored adjective.
        assert "rejected" in answer.lower()
        assert "CI" in answer
        assert f"{decision.comparison.champion_score:.4f}" in answer
        assert f"{decision.comparison.challenger_score:.4f}" in answer
        assert "400" in answer, "the sample size the decision rests on must be visible"

    def test_an_inconclusive_gate_explains_the_noise_rather_than_claiming_a_verdict(self, tmp_path):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 300)
        champion = np.where(rng.uniform(size=300) < 0.800, y, 1 - y)
        challenger = np.where(rng.uniform(size=300) < 0.805, y, 1 - y)

        decision = evaluate_gate(y, champion, challenger, metric="f1", n_bootstrap=400)
        assert decision.verdict == GateVerdict.INCONCLUSIVE

        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        run_id = log_pipeline_run(
            tracking_uri=tracking_uri, run_name="challenger",
            promotion_decision=decision.as_dict(), **_minimal_run_payload(),
        )
        answer = answer_question(tracking_uri, run_id, "should we promote the challenger?")
        assert "spans zero" in answer
        assert "not won" in answer

    def test_a_run_with_no_gate_says_so_rather_than_inventing_one(self, tmp_path):
        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        run_id = log_pipeline_run(tracking_uri=tracking_uri, run_name="original",
                                  **_minimal_run_payload())
        answer = answer_question(tracking_uri, run_id, "why did you reject the latest model?")
        assert "no champion-challenger comparison" in answer.lower()


class TestChallengerIsAChildRun:
    def test_the_parent_run_id_is_recorded(self, tmp_path):
        """A challenger floating loose in the experiment list is one nobody can
        trace back to the model it was meant to replace."""
        import mlflow

        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        champion_id = log_pipeline_run(tracking_uri=tracking_uri, run_name="champion",
                                       **_minimal_run_payload())
        challenger_id = log_pipeline_run(
            tracking_uri=tracking_uri, run_name="challenger",
            parent_run_id=champion_id, **_minimal_run_payload(),
        )

        mlflow.set_tracking_uri(tracking_uri)
        tags = mlflow.tracking.MlflowClient().get_run(challenger_id).data.tags
        assert tags["mlflow.parentRunId"] == champion_id
        assert tags["run_role"] == "challenger"
