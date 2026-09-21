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

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from autoeng.cleaning.structural import clean_structural
from autoeng.common.roles import assign_feature_roles
from autoeng.detection.group_detector import detect_group_column, group_values
from autoeng.detection.problem_detector import ProblemType, decision_from_override, detect_problem_type
from autoeng.explain.explainer import explain_winner
from autoeng.ingestion.loader import load_raw_dataset
from autoeng.leakage.detector import (
    check_group_overlap, check_temporal_split, check_train_test_row_overlap, merge_reports,
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
from autoeng.modeling.calibration import apply_calibration, expected_calibration_error, select_calibration
from autoeng.modeling.threshold import (
    DEFAULT_OBJECTIVE, DEFAULT_PRECISION_FLOOR, DEFAULT_THRESHOLD,
    binary_indicator, operating_point, out_of_fold_probabilities, select_threshold,
)
from autoeng.modeling.time_series import run_time_series_search
from autoeng.profiling.profiler import profile_dataset
from autoeng.monitoring.drift import raw_column_importances
from autoeng.registry.model_store import freeze_holdout, save_model, update_training_schema
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
    decision_threshold: dict[str, Any] | None = None
    group_decision: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _select_operating_point(
    estimator, X_train, y_train, *, cv_folds: int, objective: str,
    precision_floor: float, cost_false_negative: float, cost_false_positive: float, groups=None,
) -> dict[str, Any] | None:
    """
    Choose the decision threshold from out-of-fold predictions on the TRAINING
    partition (CLAUDE.md #5 — never the held-out split).

    Called before the final fit, on a clone, so nothing about the held-out rows
    can reach the choice. Returns None when thresholding does not apply:
    multiclass targets are a different problem, and several models in the zoo
    (RidgeClassifier, LinearSVC) expose `decision_function` rather than
    calibrated probabilities, so there is no 0-1 scale to cut.
    """
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        proba = out_of_fold_probabilities(estimator, X_train, y_train, cv_folds=cv_folds, groups=groups)
        choice = select_threshold(
            y_train, proba, objective=objective, precision_floor=precision_floor,
            cost_false_negative=cost_false_negative, cost_false_positive=cost_false_positive,
            groups=groups,
        )
        result = choice.as_dict()
        # Chosen from the same out-of-fold probabilities, so the calibration is
        # never fitted on the held-out rows either (T2-1).
        indicator, _ = binary_indicator(y_train, choice.positive_label)
        calibration = select_calibration(indicator, proba, groups).as_dict()
        result["calibration"] = calibration
        if calibration["method"] == "platt":
            result["calibrated_threshold"] = float(apply_calibration([choice.threshold], calibration)[0])
        return result
    except Exception as e:  # noqa: BLE001 - falls back to the 0.5 default, reported below
        return {
            "threshold": DEFAULT_THRESHOLD, "objective": objective, "metrics": {},
            "default_metrics": {}, "n_candidates": 0, "curve": [],
            "reasoning": f"Threshold selection failed ({type(e).__name__}: {e}); "
                         f"kept the {DEFAULT_THRESHOLD} default.",
        }


def _held_out_operating_point(pipeline, X_test, y_test, threshold: float,
                              calibration: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    What the chosen threshold actually does on data nothing has touched.

    Reported alongside the 0.5 default deliberately: the point of T0-2 is the
    gap between them, and showing only the tuned figure would repeat the
    original sin in the other direction.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(X_test)[:, 1]
    classes = list(getattr(pipeline, "classes_", []))
    # Column 1 is classes_[1]. Comparing against the literal 1 would score zero
    # true positives at every threshold on a string target.
    y, _ = binary_indicator(y_test, classes[1] if len(classes) == 2 else None)
    result = {
        "at_selected_threshold": operating_point(y, proba, threshold),
        "at_default_threshold": operating_point(y, proba, DEFAULT_THRESHOLD),
    }
    if calibration and calibration.get("method") == "platt" and len(np.unique(y)) == 2:
        from sklearn.metrics import brier_score_loss

        calibrated = apply_calibration(proba, calibration)
        result["calibration"] = {
            "brier_raw": float(brier_score_loss(y, proba)), "brier_calibrated": float(brier_score_loss(y, calibrated)),
            "ece_raw": expected_calibration_error(y, proba), "ece_calibrated": expected_calibration_error(y, calibrated),
            "n_rows": int(len(y)),
        }
    return result


def _persist_final_model(
    estimator, X_train, y_train, profile, roles, *,
    problem_type: str, model_name: str | None, selection_source: str,
    dataset_path: str, model_dir: Path,
    decision_threshold: dict[str, Any] | None = None,
    groups=None,
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
            # Without this the artifact predicts at 0.5 while the report quotes
            # a tuned operating point — the two must not come apart.
            decision_threshold=decision_threshold,
            # Sizes the drift references in entities rather than rows.
            groups=groups,
        )
        return saved.as_dict()
    except Exception as e:  # noqa: BLE001 - see docstring
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}


def _freeze_holdout_into_artifact(model_artifact, X_test, y_test, target_column, clean_df, group_column):
    """
    Persist this run's held-out rows beside the model.

    A later challenger is compared with this model on exactly these rows (T1-5),
    and a retrain excludes them from its training data so the comparison is not
    rigged. The group column rides along when there is one, so an entity-aware
    comparison stays possible. Failure is reported on the artifact rather than
    raised: the model itself is already safely on disk.
    """
    if not model_artifact or model_artifact.get("status") != "saved":
        return model_artifact
    try:
        extra = (clean_df.loc[X_test.index, [group_column]]
                 if group_column and group_column in clean_df.columns else None)
        record = freeze_holdout(model_artifact["model_dir"], X_test, y_test, target_column,
                                extra_columns=extra)
        model_artifact = dict(model_artifact)
        model_artifact["holdout"] = record
    except Exception as e:  # noqa: BLE001 - see docstring
        model_artifact = dict(model_artifact)
        model_artifact.setdefault("warnings", []).append(
            f"Holdout was not frozen ({type(e).__name__}: {e}); a later gate can only "
            f"compare on the forward window."
        )
    return model_artifact


def _holdout_split(X, y, problem_kind: str, groups=None):
    """
    Carve off the held-out test split.

    With groups, whole entities move together — otherwise the final evaluation
    is scored on customers the model already trained on, which is the same leak
    grouped CV removes from the leaderboard, surviving into the one number the
    report leads with. Returns (X_train, X_test, y_train, y_test, groups_train).
    """
    if groups is None:
        stratify = y if problem_kind == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE, stratify=stratify,
        )
        return X_train, X_test, y_train, y_test, None

    groups = np.asarray(groups)
    splitter = (
        StratifiedGroupKFold(n_splits=int(round(1 / HOLDOUT_FRACTION)), shuffle=True, random_state=RANDOM_STATE)
        if problem_kind == "classification"
        else GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE)
    )
    train_idx, test_idx = next(iter(splitter.split(X, y, groups=groups)))
    return (X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
            groups[train_idx])


def _enrich_schema_after_evaluation(
    model_artifact: dict[str, Any] | None, explanation: dict[str, Any] | None,
    held_out_metrics: dict[str, float] | None, threshold_choice: dict[str, Any] | None,
    feature_columns: list[str],
) -> dict[str, Any] | None:
    """
    Write what only the explain and evaluation stages know back into the saved
    schema: feature importances (for T1-3's drift weighting) and a performance
    baseline (for T1-3's concept drift and T1-5's champion comparison).

    Importances are folded onto RAW input columns first. SHAP explains the
    transformed matrix, so a categorical arrives as `city_Pune`, `city_Delhi`;
    drift is measured on `city`. Storing the transformed names would make the
    model's most-used categorical look unused and discount its drift to zero.

    The baseline prefers the out-of-fold operating point over the held-out one:
    on a rare-positive dataset the holdout can contain a handful of positives,
    and cross-validated estimates over the whole training partition are the
    steadier reference to detect a real drop against.
    """
    if not model_artifact or model_artifact.get("status") != "saved":
        return model_artifact

    updates: dict[str, Any] = {}
    importances = (explanation or {}).get("feature_importances") or []
    if importances:
        updates["feature_importances"] = raw_column_importances(importances, feature_columns)
        updates["importance_method"] = (explanation or {}).get("importance_method")

    baseline = dict((threshold_choice or {}).get("metrics") or {})
    baseline_source = "out-of-fold operating point on the training partition"
    if not baseline and held_out_metrics:
        baseline, baseline_source = dict(held_out_metrics), "held-out test split"
    if baseline:
        updates["baseline_metrics"] = baseline
        updates["baseline_source"] = baseline_source

    if not updates:
        return model_artifact
    try:
        update_training_schema(model_artifact["schema_path"], updates)
        model_artifact = dict(model_artifact)
        model_artifact["enriched_with"] = sorted(updates)
    except Exception as e:  # noqa: BLE001 - the model itself is already safely on disk
        model_artifact = dict(model_artifact)
        model_artifact.setdefault("warnings", []).append(
            f"Schema enrichment failed ({type(e).__name__}: {e}); drift weighting will "
            f"fall back to uniform."
        )
    return model_artifact


def _regression_metric_from_r2(pipeline, X_test, y_test) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    preds = pipeline.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }


class NoViableModelError(RuntimeError):
    """
    Every candidate failed, so there is no model to fit.

    Raised instead of letting `factories[None]` produce a bare `KeyError: None`,
    which discards the only useful information available: each candidate
    recorded its own exception on the leaderboard, and those errors ARE the
    explanation. A total search failure is rare, but it is exactly when the
    reason matters most.
    """

    @classmethod
    def from_leaderboard(cls, leaderboard) -> "NoViableModelError":
        reasons = [
            f"  - {r.name} ({r.status}): {r.error}"
            for r in leaderboard.results if r.error
        ]
        joined = '\n'.join(reasons[:10])
        detail = ('\n' + joined) if reasons else (
            " No candidates were evaluated at all — the model zoo produced nothing "
            "to try, which usually means the feature set was empty after role assignment."
        )
        return cls(
            f"No model could be fitted: all {len(leaderboard.results)} candidates failed, "
            f"and neither hyperparameter tuning nor the stacked ensemble produced a usable "
            f"alternative.{detail}"
        )


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
    group_column_override: str | None = None,
    use_groups: bool = True,
    threshold_objective: str = DEFAULT_OBJECTIVE,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    cost_false_negative: float = 10.0,
    cost_false_positive: float = 1.0,
    parent_run_id: str | None = None,
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

    # Group detection needs roles to know what the target and time axis are, and
    # roles need the group column to exclude it from features, so roles are
    # built twice — cheap, and clearer than threading a placeholder through.
    provisional_roles = assign_feature_roles(profile, target_column=target_column, time_column=time_column)
    group_decision = detect_group_column(
        clean_df, profile, provisional_roles,
        override=group_column_override, disabled=not use_groups,
    )
    group_column = group_decision.column
    # The overlap scan uses what was DETECTED, not what is applied, so
    # --no-groups still reports the leak it is causing.
    detected_group_column = group_decision.detected_column
    roles = assign_feature_roles(profile, target_column=target_column, time_column=time_column,
                                 group_column=group_column)

    all_groups = group_values(clean_df, group_decision)

    leaderboard_dict = None
    hpo_results_list: list[dict[str, Any]] = []
    explanation_dict = None
    held_out_metrics = None
    # Only the supervised branches fit a single final estimator worth
    # persisting; time series may select a classical baseline (no fitted
    # object) and clustering has no model to serve.
    model_artifact: dict[str, Any] | None = None
    model_dir = out_dir / "models" / run_name
    # Binary classification only; None everywhere else means "0.5, and the
    # report should not pretend otherwise".
    threshold_choice: dict[str, Any] | None = None
    held_out_operating_point: dict[str, Any] | None = None
    pre_leak_dict = {"flags": []}
    post_leak_dict = {"flags": []}
    clustering_summary = None
    ts_baselines = None
    ts_setup_dict = None

    if chosen.problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        y_full = clean_df[target_column]
        X_full = clean_df[roles.feature_columns]
        n_classes = y_full.nunique()

        X_train, X_test, y_train, y_test, groups_train = _holdout_split(
            X_full, y_full, "classification", groups=all_groups,
        )
        pre_leak = scan_pre_training(clean_df.loc[X_train.index], profile, target_column, roles.feature_columns)
        overlap = check_train_test_row_overlap(clean_df.loc[X_train.index], clean_df.loc[X_test.index])
        pre_leak = merge_reports(pre_leak, overlap)
        # Runs whether or not grouping was applied: if an entity key was found
        # but grouping is off, this is exactly where that shows up.
        pre_leak = merge_reports(pre_leak, check_group_overlap(
            clean_df.loc[X_train.index], clean_df.loc[X_test.index], detected_group_column,
        ))
        pre_leak_dict = pre_leak.as_dict()

        leaderboard = run_classification_search(X_train, y_train, roles, n_classes=n_classes,
                                                 cv_folds=cv_folds, groups=groups_train)
        leaderboard_dict = leaderboard.as_dict()
        ranked = leaderboard.ranked()
        winner_name = ranked[0].name if ranked else None

        factories = get_classification_models(n_classes=n_classes)
        hpo_outcomes = optimize_top_candidates(
            leaderboard.results, factories, X_train, y_train, roles, "classification",
            leaderboard.primary_metric, top_n=hpo_top_n, n_trials=hpo_trials, groups=groups_train,
        )
        hpo_results_list = [o.as_dict() for o in hpo_outcomes]

        stack_result = evaluate_stacked_ensemble(
            leaderboard.results, factories, X_train, y_train, roles, "classification",
            leaderboard.primary_metric, CLASSIFICATION_SCORING if n_classes == 2
            else {"accuracy": "accuracy", "f1_macro": "f1_macro"}, cv_folds=cv_folds,
            groups=groups_train,
        )
        if stack_result is not None:
            leaderboard.results.append(stack_result)
            leaderboard_dict = leaderboard.as_dict()

        final_name, final_params, final_source, _, hpo_improvement = _select_final_model(
            ranked, hpo_outcomes, stack_result, leaderboard.primary_metric,
        )
        if final_name is None:
            raise NoViableModelError.from_leaderboard(leaderboard)
        if final_name == STACK_MODEL_NAME:
            final_pipeline = stack_result.pipeline
        else:
            final_pipeline = _build_pipeline_for_model(final_name, factories[final_name], roles, "classification")
            if final_params:
                final_pipeline.named_steps["model"].set_params(**final_params)
        # Selected BEFORE the final fit, from out-of-fold predictions on the
        # training partition only. Binary targets only — a single cut point is
        # not a meaningful object for multiclass.
        if n_classes == 2:
            threshold_choice = _select_operating_point(
                final_pipeline, X_train, y_train, cv_folds=cv_folds,
                objective=threshold_objective, precision_floor=precision_floor,
                cost_false_negative=cost_false_negative, cost_false_positive=cost_false_positive,
                groups=groups_train,
            )

        final_pipeline.fit(X_train, y_train)
        model_artifact = _persist_final_model(
            final_pipeline, X_train, y_train, profile, roles,
            problem_type=chosen.problem_type.value, model_name=final_name,
            selection_source=final_source, dataset_path=dataset_path, model_dir=model_dir,
            decision_threshold=threshold_choice, groups=groups_train,
        )

        model_artifact = _freeze_holdout_into_artifact(
            model_artifact, X_test, y_test, target_column, clean_df, group_column,
        )

        held_out_metrics = _classification_metric(final_pipeline, X_test, y_test, n_classes)
        if threshold_choice is not None:
            held_out_operating_point = _held_out_operating_point(
                final_pipeline, X_test, y_test, threshold_choice["threshold"],
                threshold_choice.get("calibration"),
            )
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

        X_train, X_test, y_train, y_test, groups_train = _holdout_split(
            X_full, y_full, "regression", groups=all_groups,
        )
        pre_leak = scan_pre_training(clean_df.loc[X_train.index], profile, target_column, roles.feature_columns)
        overlap = check_train_test_row_overlap(clean_df.loc[X_train.index], clean_df.loc[X_test.index])
        pre_leak = merge_reports(pre_leak, overlap)
        pre_leak = merge_reports(pre_leak, check_group_overlap(
            clean_df.loc[X_train.index], clean_df.loc[X_test.index], detected_group_column,
        ))
        pre_leak_dict = pre_leak.as_dict()

        leaderboard = run_regression_search(X_train, y_train, roles, cv_folds=cv_folds,
                                             groups=groups_train)
        leaderboard_dict = leaderboard.as_dict()
        ranked = leaderboard.ranked()
        winner_name = ranked[0].name if ranked else None

        factories = get_regression_models()
        hpo_outcomes = optimize_top_candidates(
            leaderboard.results, factories, X_train, y_train, roles, "regression",
            leaderboard.primary_metric, top_n=hpo_top_n, n_trials=hpo_trials, groups=groups_train,
        )
        hpo_results_list = [o.as_dict() for o in hpo_outcomes]

        stack_result = evaluate_stacked_ensemble(
            leaderboard.results, factories, X_train, y_train, roles, "regression",
            leaderboard.primary_metric, REGRESSION_SCORING, cv_folds=cv_folds,
            groups=groups_train,
        )
        if stack_result is not None:
            leaderboard.results.append(stack_result)
            leaderboard_dict = leaderboard.as_dict()

        final_name, final_params, final_source, _, hpo_improvement = _select_final_model(
            ranked, hpo_outcomes, stack_result, leaderboard.primary_metric,
        )
        if final_name is None:
            raise NoViableModelError.from_leaderboard(leaderboard)
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
            groups=groups_train,
        )

        model_artifact = _freeze_holdout_into_artifact(
            model_artifact, X_test, y_test, target_column, clean_df, group_column,
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

    model_artifact = _enrich_schema_after_evaluation(
        model_artifact, explanation_dict, held_out_metrics, threshold_choice,
        roles.feature_columns,
    )

    profile_summary = profile.as_dict()
    role_dict = {
        "numeric_columns": roles.numeric_columns, "categorical_columns": roles.categorical_columns,
        "low_card_categorical_columns": roles.low_card_categorical_columns,
        "high_card_categorical_columns": roles.high_card_categorical_columns,
        "datetime_columns": roles.datetime_columns, "text_columns": roles.text_columns,
        "excluded_columns": roles.excluded_columns, "group_column": roles.group_column,
    }

    report_text = generate_report(
        source_path=dataset_path, ingestion_report=ingestion_report.as_dict(), profile_summary=profile_summary,
        problem_decision=decision.as_dict(), structural_report=structural_report.as_dict(),
        role_assignment=role_dict, leaderboard=leaderboard_dict, hpo_results=hpo_results_list,
        pre_training_leakage=pre_leak_dict, post_training_leakage=post_leak_dict,
        explanation=explanation_dict, held_out_metrics=held_out_metrics,
        clustering_summary=clustering_summary, time_series_baselines=ts_baselines,
        target_source=target_source, time_series_setup=ts_setup_dict,
        model_artifact=model_artifact, threshold_choice=threshold_choice,
        held_out_operating_point=held_out_operating_point,
        group_decision=group_decision.as_dict(),
    )

    report_path = out_dir / f"{run_name}_report.md"
    # Explicit UTF-8: the report contains em dashes and arrows, and the
    # platform default on Windows is cp1252, which writes them as bytes no
    # UTF-8 reader can decode. Every report written on Windows before this
    # was silently mis-encoded.
    report_path.write_text(report_text, encoding="utf-8")

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
            decision_threshold={"selected": threshold_choice,
                                "held_out": held_out_operating_point} if threshold_choice else None,
            group_decision=group_decision.as_dict(),
            parent_run_id=parent_run_id,
        )
    except Exception as e:  # noqa: BLE001 - tracking must never take down the run
        (out_dir / f"{run_name}_mlflow_error.txt").write_text(
            f"{type(e).__name__}: {e}", encoding="utf-8")

    return PipelineRunResult(
        run_id=run_id, problem_type=chosen.problem_type.value, target_column=target_column,
        report_text=report_text, report_path=str(report_path), leaderboard=leaderboard_dict,
        held_out_metrics=held_out_metrics, model_artifact=model_artifact,
        decision_threshold=threshold_choice,
        group_decision=group_decision.as_dict(),
        extra={"mlflow_tracking_uri": mlflow_tracking_uri, "explanation": explanation_dict,
               "held_out_operating_point": held_out_operating_point},
    )
