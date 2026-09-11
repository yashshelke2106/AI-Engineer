"""
MLflow-backed experiment history.

Every decision the pipeline makes — which target/problem type it inferred
and why, what cleaning actions it took, what leakage it flagged, the full
leaderboard (not just the winner), what HPO changed, and the final
explanation — gets logged as params/metrics/artifacts on one MLflow run.
This is what makes the conversational Q&A layer (autoeng/explain/qa.py)
honest: it answers "why did you reject model X" by reading this run's
actual logged leaderboard, not by re-deriving or guessing an answer.

Tracking URI defaults to a local `mlruns/` directory under the project so
the whole thing works with zero external services — MLflow's file store is
enough to get real run history, comparison, and querying.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

DEFAULT_EXPERIMENT = "autonomous_ml_engineer"


def _log_dict_artifact(payload: dict[str, Any], filename: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_pipeline_run(
    tracking_uri: str,
    run_name: str,
    source_path: str,
    ingestion_report: dict[str, Any],
    profile_summary: dict[str, Any],
    problem_decision: dict[str, Any],
    structural_cleaning_report: dict[str, Any],
    role_assignment: dict[str, Any],
    pre_training_leakage: dict[str, Any],
    leaderboard: dict[str, Any],
    hpo_results: list[dict[str, Any]],
    post_training_leakage: dict[str, Any],
    explanation: dict[str, Any],
    final_report_text: str,
    model_artifact: dict[str, Any] | None = None,
    decision_threshold: dict[str, Any] | None = None,
    group_decision: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        if parent_run_id:
            # MLflow renders a run with this tag as nested under its parent, so
            # a challenger sits under the champion it was produced to replace
            # rather than floating loose in the experiment list.
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.set_tag("champion_run_id", parent_run_id)
            mlflow.set_tag("run_role", "challenger")
        mlflow.log_params({
            "source_path": source_path,
            "problem_type": problem_decision["chosen"]["problem_type"],
            "target_column": problem_decision["chosen"]["target_column"],
            "time_column": problem_decision["chosen"]["time_column"],
            "problem_detection_confidence": problem_decision["confidence"],
            "n_rows": profile_summary["n_rows"],
            "n_cols": profile_summary["n_cols"],
            "primary_metric": leaderboard["primary_metric"],
            "winner_model": explanation["winner_name"],
        })

        mlflow.log_metrics({
            "winner_score": explanation["winner_score"] if explanation["winner_score"] is not None else float("nan"),
            "runner_up_score": explanation["runner_up_score"] or float("nan"),
            "margin_over_runner_up": explanation["margin"] if explanation["margin"] is not None else float("nan"),
            "hpo_improvement": explanation["hpo_improvement"] or 0.0,
            "n_critical_leakage_flags": sum(
                1 for f in pre_training_leakage["flags"] + post_training_leakage["flags"]
                if f["severity"] == "critical"
            ),
            "n_models_evaluated": len(leaderboard["results"]),
            "n_models_succeeded": sum(1 for r in leaderboard["results"] if r["status"] == "ok"),
        })

        mlflow.set_tags({
            "has_critical_leakage": any(
                f["severity"] == "critical" for f in pre_training_leakage["flags"] + post_training_leakage["flags"]
            ),
        })

        _log_dict_artifact(ingestion_report, "ingestion_report.json")
        _log_dict_artifact(profile_summary, "data_profile.json")
        _log_dict_artifact(problem_decision, "problem_type_decision.json")
        _log_dict_artifact(structural_cleaning_report, "structural_cleaning_report.json")
        _log_dict_artifact(role_assignment, "feature_role_assignment.json")
        _log_dict_artifact(pre_training_leakage, "pre_training_leakage_report.json")
        _log_dict_artifact(leaderboard, "model_leaderboard.json")
        _log_dict_artifact({"hpo_results": hpo_results}, "hpo_results.json")
        _log_dict_artifact(post_training_leakage, "post_training_leakage_report.json")
        _log_dict_artifact(explanation, "model_explanation.json")

        if group_decision:
            _log_dict_artifact(group_decision, "group_decision.json")
            mlflow.set_tag("group_column", group_decision.get("column") or "none")

        if decision_threshold:
            _log_dict_artifact(decision_threshold, "decision_threshold.json")
            selected = decision_threshold.get("selected") or {}
            held_out = (decision_threshold.get("held_out") or {}).get("at_selected_threshold") or {}
            # Logged as metrics, not just an artifact, so runs can be compared
            # and sorted on the operating point rather than only on ROC-AUC —
            # which is the whole point of T0-2.
            mlflow.log_metrics({
                k: float(v) for k, v in {
                    "decision_threshold": selected.get("threshold"),
                    "held_out_precision": held_out.get("precision"),
                    "held_out_recall": held_out.get("recall"),
                    "held_out_f1": held_out.get("f1"),
                }.items() if v is not None
            })

        if model_artifact:
            _log_dict_artifact(model_artifact, "model_artifact.json")
            # The model directory was written to disk right after fit (see
            # autoeng/registry/model_store.py, which uses mlflow.sklearn.
            # save_model precisely so it needs no active run). Copying it in
            # here is what turns it into a `runs:/<run_id>/model` URI that
            # mlflow.sklearn.load_model can resolve.
            mlflow_model_dir = model_artifact.get("mlflow_model_dir")
            if mlflow_model_dir and Path(mlflow_model_dir).is_dir():
                mlflow.log_artifacts(mlflow_model_dir, artifact_path="model")
                mlflow.set_tag("model_uri", f"runs:/{run.info.run_id}/model")

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "human_readable_report.md"
            report_path.write_text(final_report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

        return run.info.run_id


def list_runs(tracking_uri: str) -> list[dict[str, Any]]:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "start_time": r.info.start_time,
            "params": r.data.params,
            "metrics": r.data.metrics,
            "tags": r.data.tags,
        }
        for r in runs
    ]


def get_run_artifact(tracking_uri: str, run_id: str, artifact_name: str) -> dict[str, Any] | str | None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_path = client.download_artifacts(run_id, artifact_name, tmp)
        except Exception:
            return None
        text = Path(local_path).read_text(encoding="utf-8")
        if artifact_name.endswith(".json"):
            return json.loads(text)
        return text
