"""
End-to-end orchestration: point this at a raw dataset file with zero prior
knowledge of its schema, and it runs the full core loop —

  ingest -> profile -> detect problem type -> clean -> engineer features ->
  scan for leakage -> search 20+ models -> tune the top candidates ->
  explain the winner -> re-scan for leakage post-training -> log everything
  to MLflow -> write a human-readable report.

For classification/regression, a held-out test split is carved off BEFORE
model search even begins, so nothing about it can influence model
selection, cleaning statistics, or feature engineering — model search and
HPO only ever see the training partition; the held-out set is touched
exactly once, for final evaluation. Time-series forecasting uses its own
chronological (never shuffled) cross-validation instead, per
autoeng/modeling/time_series.py. Clustering has no target to hold out
against and is evaluated with internal validity indices on the full data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from autoeng.cleaning.structural import clean_structural
from autoeng.common.roles import assign_feature_roles
from autoeng.detection.problem_detector import ProblemType, decision_from_override, detect_problem_type
from autoeng.explain.explainer import explain_winner
from autoeng.ingestion.loader import load_raw_dataset
from autoeng.leakage.detector import (
    check_temporal_split, check_train_test_row_overlap, merge_reports,
    scan_post_training, scan_pre_training,
)
from autoeng.modeling.clustering_search import run_clustering_search, rank_clustering_results, select_k
from autoeng.modeling.ensemble import STACK_MODEL_NAME, evaluate_stacked_ensemble
from autoeng.modeling.hpo import optimize_top_candidates
from autoeng.modeling.model_zoo import get_classification_models, get_regression_models
from autoeng.modeling.search import (
    CLASSIFICATION_SCORING, REGRESSION_SCORING, _build_pipeline_for_model,
    run_classification_search, run_regression_search,
)
from autoeng.modeling.time_series import run_time_series_search
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import save_model
from autoeng.reporting.report_generator import generate_report
from autoeng.tracking.mlflow_tracker import log_pipeline_run

HOLDOUT_FRACTION = 0.2
RANDOM_STATE = 42


@dataclass
class PipelineRunResult:
    run_id: str | None
    problem_type: str
    target_column: str | None
    report_text: str
    report_path: str
    leaderboard: dict[str, Any] | None = None
    held_out_metrics: dict[str, float] | None = None
    model_artifact: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _persist_final_model(
    estimator, X_train, y_train, profile, roles, *,
    problem_type: str, model_name: str | None, selection_source: str,
    dataset_path: str, model_dir: Path,
) -> dict[str, Any]:
    """
    Save the fitted winner plus its training schema, and return a JSON-able
    record of what happened for the report and MLflow.

    A failure here is *reported*, not raised. Raising would throw away a
    completed leaderboard, HPO sweep and explanation — minutes of work — over a
    serialization problem. Swallowing it silently would be worse: the report
    would go on citing an artifact that does not exist. So whichever of the two
    actually happened is what the report says.
    """
    try:
        saved = save_model(
            estimator, X_train, y_train, profile, roles,
            problem_type=problem_type, model_name=model_name,
            output_dir=model_dir, selection_source=selection_source,
            dataset_path=dataset_path,
        )
        return saved.as_dict()
    except Exception as e:  # noqa: BLE001 - see docstring
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def _regression_metric_from_r2(pipeline, X_test, y_test) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    preds = pipeline.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }


def _select_final_model(ranked, hpo_outcomes, stack_result, primary_metric: str):
    """
    Pick the single best option across three sources, all scored on the same
    cross-validation: the untuned leaderboard winner, any tuned candidate, and
    the stacked ensemble. The stack has to earn its place like everything else
    — it is not assumed to be better just because it's fancier.

    Returns (name, params, source, cv_score, hpo_improvement).
    """
    options: list[tuple[str, dict, str, float, float | None]] = []
    if ranked:
        best = ranked[0]
        options.append((best.name, {}, "leaderboard (untuned)",
                        best.metrics.get(primary_metric, float("-inf")), None))
    for outcome in hpo_outcomes or []:
        options.append((outcome.model_name, outcome.best_params, "hyperparameter tuning",
                        outcome.best_score, outcome.improvement))
    if stack_result is not None and stack_result.status == "ok":
        options.append((stack_result.name, {}, "stacked ensemble",
                        stack_result.metrics.get(primary_metric, float("-inf")), None))

    if not options:
        return None, {}, "none", float("nan"), None
    return max(options, key=lambda o: o[3])


def _classification_metric(pipeline, X_test, y_test, n_classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    preds = pipeline.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, preds))}
    metrics["f1_macro"] = float(f1_score(y_test, preds, average="macro"))
    try:
        if n_classes == 2:
            if hasattr(pipeline, "predict_proba"):
                scores = pipeline.predict_proba(X_test)[:, 1]
            else:
                scores = pipeline.decision_function(X_test)
            metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
        else:
            if hasattr(pipeline, "predict_proba"):
                scores = pipeline.predict_proba(X_test)
                metrics["roc_auc"] = float(roc_auc_score(y_test, scores, multi_class="ovr"))
    except Exception:
        pass
    return metrics


def run_pipeline(
    dataset_path: str,
    output_dir: str = "./runs",
    cv_folds: int = 5,
    hpo_trials: int = 20,
    hpo_top_n: int = 3,
    mlflow_tracking_uri: str | None = None,
    run_name: str | None = None,
    target_override: str | None = None,
    problem_type_override: str | None = None,
) -> PipelineRunResult:
    dataset_path = str(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # MLflow 3.x deprecated the plain filesystem tracking backend in favor of a
    # DB-backed store; SQLite keeps this a zero-external-services setup while
    # avoiding that "maintenance mode" exception.
    mlflow_tracking_uri = mlflow_tracking_uri or f"sqlite:///{(out_dir / 'mlflow.db').resolve()}"
    run_name = run_name or Path(dataset_path).stem

    df, ingestion_report = load_raw_dataset(dataset_path)
    profile = profile_dataset(df)
    if target_override is not None or problem_type_override is not None:
        decision = decision_from_override(df, profile, target_override, problem_type_override)
        target_source = "supplied by caller"
    else:
        decision = detect_problem_type(df, profile)
        target_source = "auto-detected"
    clean_df, structural_report = clean_structural(df, profile)
    # profile indices may reference columns dropped by structural cleaning (e.g.
    # constant columns) — re-profile the cleaned frame so downstream role
    # assignment only ever sees columns that actually still exist.
    profile = profile_dataset(clean_df)

    chosen = decision.chosen
    target_column, time_column = chosen.target_column, chosen.time_column
    roles = assign_feature_roles(profile, target_column=target_column, time_column=time_column)

    leaderboard_dict = None
    hpo_results_list: list[dict[str, Any]] = []
    explanation_dict = None
    held_out_metrics = None
    # Only the supervised branches fit a single final estimator worth
    # persisting; time series may select a classical baseline (no fitted
    # object) and clustering has no model to serve.
    model_artifact: dict[str, Any] | None = None
    model_dir = out_dir / "models" / run_name
    pre_leak_dict = {"flags": []}
    post_leak_dict = {"flags": []}
    clustering_summary = None
    ts_baselines = None
    ts_setup_dict = None

    if chosen.problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        y_full = clean_df[target_column]
        X_full = clean_df[roles.feature_columns]
        n_classes = y_full.nunique()

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE, stratify=y_full,
        )
        pre_leak = scan_pre_training(clean_df.loc[X_train.index], profile, target_column, roles.feature_columns)
        overlap = check_train_test_row_overlap(clean_df.loc[X_train.index], clean_df.loc[X_test.index])
        pre_leak = merge_reports(pre_leak, overlap)
        pre_leak_dict = pre_leak.as_dict()

        leaderboard = run_classification_search(X_train, y_train, roles, n_classes=n_classes, cv_folds=cv_folds)
        leaderboard_dict = leaderboard.as_dict()
        ranked = leaderboard.ranked()
        winner_name = ranked[0].name if ranked else None

        factories = get_classification_models(n_classes=n_classes)
        hpo_outcomes = optimize_top_candidates(
            leaderboard.results, factories, X_train, y_train, roles, "classification",
            leaderboard.primary_metric, top_n=hpo_top_n, n_trials=hpo_trials,
        )
        hpo_results_list = [o.as_dict() for o in hpo_outcomes]

        stack_result = evaluate_stacked_ensemble(
            leaderboard.results, factories, X_train, y_train, roles, "classification",
            leaderboard.primary_metric, CLASSIFICATION_SCORING if n_classes == 2
            else {"accuracy": "accuracy", "f1_macro": "f1_macro"}, cv_folds=cv_folds,
        )
        if stack_result is not None:
            leaderboard.results.append(stack_result)
            leaderboard_dict = leaderboard.as_dict()

        final_name, final_params, final_source, _, hpo_improvement = _select_final_model(
            ranked, hpo_outcomes, stack_result, leaderboard.primary_metric,
        )
        if final_name == STACK_MODEL_NAME:
            final_pipeline = stack_result.pipeline
        else:
            final_pipeline = _build_pipeline_for_model(final_name, factories[final_name], roles, "classification")
            if final_params:
                final_pipeline.named_steps["model"].set_params(**final_params)
        final_pipeline.fit(X_train, y_train)
        model_artifact = _persist_final_model(
            final_pipeline, X_train, y_train, profile, roles,
            problem_type=chosen.problem_type.value, model_name=final_name,
            selection_source=final_source, dataset_path=dataset_path, model_dir=model_dir,
        )

        held_out_metrics = _classification_metric(final_pipeline, X_test, y_test, n_classes)
        explanation = explain_winner(
            leaderboard.results, final_name, final_pipeline, X_test, y_test,
            leaderboard.primary_metric, hpo_improvement=hpo_improvement,
            selection_source=final_source,
        )
        explanation_dict = explanation.as_dict()

        importances_dict = dict(explanation.feature_importances)
        post_leak = scan_post_training(leaderboard.primary_metric, held_out_metrics.get(leaderboard.primary_metric, float("nan")), importances_dict)
        post_leak_dict = post_leak.as_dict()

    elif chosen.problem_type == ProblemType.REGRESSION:
        y_full = clean_df[target_column]
        X_full = clean_df[roles.feature_columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE,
        )
        pre_leak = scan_pre_training(clean_df.loc[X_train.index], profile, target_column, roles.feature_columns)
        overlap = check_train_test_row_overlap(clean_df.loc[X_train.index], clean_df.loc[X_test.index])
        pre_leak = merge_reports(pre_leak, overlap)
        pre_leak_dict = pre_leak.as_dict()

        leaderboard = run_regression_search(X_train, y_train, roles, cv_folds=cv_folds)
        leaderboard_dict = leaderboard.as_dict()
        ranked = leaderboard.ranked()
        winner_name = ranked[0].name if ranked else None

        factories = get_regression_models()
        hpo_outcomes = optimize_top_candidates(
            leaderboard.results, factories, X_train, y_train, roles, "regression",
            leaderboard.primary_metric, top_n=hpo_top_n, n_trials=hpo_trials,
        )
        hpo_results_list = [o.as_dict() for o in hpo_outcomes]

        stack_result = evaluate_stacked_ensemble(
            leaderboard.results, factories, X_train, y_train, roles, "regression",
            leaderboard.primary_metric, REGRESSION_SCORING, cv_folds=cv_folds,
        )
        if stack_result is not None:
            leaderboard.results.append(stack_result)
            leaderboard_dict = leaderboard.as_dict()

        final_name, final_params, final_source, _, hpo_improvement = _select_final_model(
            ranked, hpo_outcomes, stack_result, leaderboard.primary_metric,
        )
        if final_name == STACK_MODEL_NAME:
            final_pipeline = stack_result.pipeline
        else:
            final_pipeline = _build_pipeline_for_model(final_name, factories[final_name], roles, "regression")
            if final_params:
                final_pipeline.named_steps["model"].set_params(**final_params)
        final_pipeline.fit(X_train, y_train)
        model_artifact = _persist_final_model(
            final_pipeline, X_train, y_train, profile, roles,
            problem_type=chosen.problem_type.value, model_name=final_name,
            selection_source=final_source, dataset_path=dataset_path, model_dir=model_dir,
        )

        held_out_metrics = _regression_metric_from_r2(final_pipeline, X_test, y_test)
        explanation = explain_winner(
            leaderboard.results, final_name, final_pipeline, X_test, y_test,
            leaderboard.primary_metric, hpo_improvement=hpo_improvement,
            selection_source=final_source,
        )
        explanation_dict = explanation.as_dict()

        importances_dict = dict(explanation.feature_importances)
        post_leak = scan_post_training("r2", held_out_metrics.get("r2", float("nan")), importances_dict)
        post_leak_dict = post_leak.as_dict()

    elif chosen.problem_type == ProblemType.TIME_SERIES_FORECASTING:
        results, baselines, ts_setup = run_time_series_search(
            clean_df, target_column, time_column, roles, cv_folds=cv_folds,
        )
        ts_baselines = [b.as_dict() for b in baselines]
        ts_setup_dict = ts_setup.as_dict()
        leaderboard_dict = {"problem_kind": "regression", "primary_metric": "r2",
                             "results": [r.as_dict() for r in results]}
        ok_ml = sorted([r for r in results if r.status == "ok"], key=lambda r: r.metrics.get("r2", float("-inf")), reverse=True)
        best_ml = ok_ml[0] if ok_ml else None
        best_baseline = max(baselines, key=lambda b: b.metrics["r2"]) if baselines else None

        if best_baseline and (not best_ml or best_baseline.metrics["r2"] > best_ml.metrics["r2"]):
            explanation_dict = {
                "winner_name": best_baseline.name, "winner_score": best_baseline.metrics["r2"],
                "runner_up_name": best_ml.name if best_ml else None,
                "runner_up_score": best_ml.metrics.get("r2") if best_ml else None,
                "margin": (best_baseline.metrics["r2"] - best_ml.metrics["r2"]) if best_ml else None,
                "cv_fold_std": None, "margin_within_noise": None, "hpo_improvement": None,
                "feature_importances": [], "importance_method": "n/a (classical baseline, no learned features)",
                "narrative": (
                    f"The classical baseline '{best_baseline.name}' (r2={best_baseline.metrics['r2']:.4f}) "
                    f"outperformed every tested ML model on lag-feature regression"
                    + (f" (best: {best_ml.name}, r2={best_ml.metrics['r2']:.4f})" if best_ml else "")
                    + ". Recommend using the baseline forecast rather than a fitted model for this series."
                ),
            }
            held_out_metrics = {"r2": best_baseline.metrics["r2"], "rmse": -best_baseline.metrics["neg_rmse"]}
        elif best_ml:
            held_out_metrics = {"r2": best_ml.metrics["r2"], "rmse": -best_ml.metrics["neg_rmse"]}
            explanation_dict = {
                "winner_name": best_ml.name, "winner_score": best_ml.metrics["r2"],
                "runner_up_name": best_baseline.name if best_baseline else None,
                "runner_up_score": best_baseline.metrics["r2"] if best_baseline else None,
                "margin": (best_ml.metrics["r2"] - best_baseline.metrics["r2"]) if best_baseline else None,
                "cv_fold_std": None, "margin_within_noise": None, "hpo_improvement": None,
                "feature_importances": [], "importance_method": "n/a (see leaderboard; SHAP omitted for time series lag models in this build)",
                "narrative": f"Selected lag-feature model: {best_ml.name} (r2={best_ml.metrics['r2']:.4f} under expanding-window CV).",
            }

        temporal_check = check_temporal_split(
            clean_df.sort_values(time_column).iloc[: int(len(clean_df) * 0.8)],
            clean_df.sort_values(time_column).iloc[int(len(clean_df) * 0.8):],
            time_column,
        )
        pre_leak_dict = temporal_check.as_dict()

    elif chosen.problem_type == ProblemType.CLUSTERING:
        X_full = clean_df[roles.feature_columns]
        results, best_k, k_scan = run_clustering_search(X_full, roles)
        ranked = rank_clustering_results(results)
        clustering_summary = {
            "best_k": best_k, "k_scan_silhouette": k_scan,
            "ranked": [r.as_dict() for r in ranked],
        }
        if ranked:
            top = ranked[0]
            explanation_dict = {
                "winner_name": top.name, "winner_score": top.metrics["silhouette"],
                "runner_up_name": ranked[1].name if len(ranked) > 1 else None,
                "runner_up_score": ranked[1].metrics["silhouette"] if len(ranked) > 1 else None,
                "margin": (top.metrics["silhouette"] - ranked[1].metrics["silhouette"]) if len(ranked) > 1 else None,
                "cv_fold_std": None, "margin_within_noise": None, "hpo_improvement": None,
                "feature_importances": [], "importance_method": "n/a (unsupervised)",
                "narrative": (
                    f"No usable target column was found, so the dataset was treated as unsupervised. "
                    f"Best clustering: {top.name} with k={top.n_clusters_found} (silhouette={top.metrics['silhouette']:.3f}, "
                    f"chosen via a k=2..10 silhouette sweep with KMeans)."
                ),
            }
            held_out_metrics = {"silhouette": top.metrics["silhouette"]}

    profile_summary = profile.as_dict()
    role_dict = {
        "numeric_columns": roles.numeric_columns, "categorical_columns": roles.categorical_columns,
        "low_card_categorical_columns": roles.low_card_categorical_columns,
        "high_card_categorical_columns": roles.high_card_categorical_columns,
        "datetime_columns": roles.datetime_columns, "text_columns": roles.text_columns,
        "excluded_columns": roles.excluded_columns,
    }

    report_text = generate_report(
        source_path=dataset_path, ingestion_report=ingestion_report.as_dict(), profile_summary=profile_summary,
        problem_decision=decision.as_dict(), structural_report=structural_report.as_dict(),
        role_assignment=role_dict, leaderboard=leaderboard_dict, hpo_results=hpo_results_list,
        pre_training_leakage=pre_leak_dict, post_training_leakage=post_leak_dict,
        explanation=explanation_dict, held_out_metrics=held_out_metrics,
        clustering_summary=clustering_summary, time_series_baselines=ts_baselines,
        target_source=target_source, time_series_setup=ts_setup_dict,
        model_artifact=model_artifact,
    )

    report_path = out_dir / f"{run_name}_report.md"
    report_path.write_text(report_text)

    run_id = None
    try:
        run_id = log_pipeline_run(
            tracking_uri=mlflow_tracking_uri, run_name=run_name, source_path=dataset_path,
            ingestion_report=ingestion_report.as_dict(), profile_summary=profile_summary,
            problem_decision=decision.as_dict(), structural_cleaning_report=structural_report.as_dict(),
            role_assignment=role_dict, pre_training_leakage=pre_leak_dict,
            leaderboard=leaderboard_dict or {"problem_kind": "n/a", "primary_metric": "n/a", "results": []},
            hpo_results=hpo_results_list, post_training_leakage=post_leak_dict,
            explanation=explanation_dict or {"winner_name": None, "winner_score": None, "runner_up_name": None,
                                              "runner_up_score": None, "margin": None, "hpo_improvement": None,
                                              "feature_importances": [], "importance_method": "n/a", "narrative": ""},
            final_report_text=report_text,
            model_artifact=model_artifact,
        )
    except Exception as e:  # noqa: BLE001 - tracking must never take down the run
        (out_dir / f"{run_name}_mlflow_error.txt").write_text(f"{type(e).__name__}: {e}")

    return PipelineRunResult(
        run_id=run_id, problem_type=chosen.problem_type.value, target_column=target_column,
        report_text=report_text, report_path=str(report_path), leaderboard=leaderboard_dict,
        held_out_metrics=held_out_metrics, model_artifact=model_artifact,
        extra={"mlflow_tracking_uri": mlflow_tracking_uri, "explanation": explanation_dict},
    )
