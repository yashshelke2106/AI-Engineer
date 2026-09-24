"""
Assembles every stage's output into one human-readable Markdown report:
what problem the system decided this was and why, what it did to the data,
what it tried, what it picked and why, what it's suspicious of, and what
to watch out for. This is the artifact a human actually reads at the end
of a run — everything else (MLflow, JSON reports) is the machine-readable
backing for it.
"""
from __future__ import annotations

from typing import Any


def _fmt_flag_list(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "None found.\n"
    lines = []
    for f in flags:
        lines.append(f"- **[{f['severity'].upper()}] {f['kind']}** — {f['description']}")
    return "\n".join(lines) + "\n"


def generate_report(
    source_path: str,
    ingestion_report: dict[str, Any],
    profile_summary: dict[str, Any],
    problem_decision: dict[str, Any],
    structural_report: dict[str, Any],
    role_assignment: dict[str, Any],
    leaderboard: dict[str, Any] | None,
    hpo_results: list[dict[str, Any]] | None,
    pre_training_leakage: dict[str, Any] | None,
    post_training_leakage: dict[str, Any] | None,
    explanation: dict[str, Any] | None,
    held_out_metrics: dict[str, float] | None = None,
    clustering_summary: dict[str, Any] | None = None,
    time_series_baselines: list[dict[str, Any]] | None = None,
    target_source: str = "auto-detected",
    time_series_setup: dict[str, Any] | None = None,
    model_artifact: dict[str, Any] | None = None,
    threshold_choice: dict[str, Any] | None = None,
    held_out_operating_point: dict[str, Any] | None = None,
    group_decision: dict[str, Any] | None = None,
) -> str:
    chosen = problem_decision["chosen"]
    lines: list[str] = []

    lines.append(f"# Autonomous ML Engineer — Run Report\n")
    lines.append(f"**Dataset:** `{source_path}`\n")
    lines.append(
        f"**Shape:** {profile_summary['n_rows']} rows x {profile_summary['n_cols']} columns "
        f"(format: {ingestion_report['file_format']}, encoding: {ingestion_report['detected_encoding']})\n"
    )
    if ingestion_report["warnings"]:
        lines.append("**Ingestion warnings:** " + "; ".join(ingestion_report["warnings"]) + "\n")

    lines.append("## 1. Problem Type Detection\n")
    lines.append(f"**Target source:** {target_source}\n")
    lines.append(f"**Decision:** `{chosen['problem_type']}`" +
                 (f", target column `{chosen['target_column']}`" if chosen["target_column"] else "") +
                 (f", time column `{chosen['time_column']}`" if chosen["time_column"] else "") +
                 f" (confidence: {problem_decision['confidence']:.2f})\n")
    lines.append("**Reasoning:**")
    lines.extend(f"- {r}" for r in chosen["reasoning"])
    if problem_decision["alternatives"]:
        lines.append("\n**Alternative hypotheses considered (and why they lost):**")
        for alt in problem_decision["alternatives"]:
            lines.append(f"- `{alt['problem_type']}`" + (f" (target `{alt['target_column']}`)" if alt["target_column"] else "")
                          + f" — score {alt['score']:.3f}")
    lines.append("")

    lines.append("## 2. Data Cleaning\n")
    lines.append("**Structural actions (dataset-level, pre-split):**")
    lines.extend(f"- {a}" for a in (structural_report.get("actions") or ["(none needed)"]))
    lines.append(
        "\nPer-column imputation, outlier capping, and categorical encoding statistics were fit "
        "separately inside each cross-validation fold (never on the full dataset) to avoid leaking "
        "validation-fold statistics into training — see the pipeline architecture notes in "
        "`autoeng/features/pipeline_builder.py`.\n"
    )

    lines.append("## 3. Feature Roles\n")
    lines.append(f"- Numeric features: {role_assignment['numeric_columns'] or '(none)'}")
    lines.append(f"- Categorical features (low-card, one-hot): {role_assignment['low_card_categorical_columns'] or '(none)'}")
    lines.append(f"- Categorical features (high-card, target-encoded): {role_assignment['high_card_categorical_columns'] or '(none)'}")
    lines.append(f"- Datetime features (decomposed): {role_assignment['datetime_columns'] or '(none)'}")
    lines.append(f"- Text features (TF-IDF -> SVD components + length/word-count stats): "
                 f"{role_assignment['text_columns'] or '(none)'}")
    lines.append(f"- Excluded (identifiers/constants): {role_assignment['excluded_columns'] or '(none)'}\n")

    if group_decision:
        lines.append("### Grouping (repeated entities)\n")
        if group_decision.get("column"):
            lines.append(
                f"Rows are grouped by `{group_decision['column']}` "
                f"(source: {group_decision.get('source', 'auto-detected')}, "
                f"confidence {group_decision.get('confidence', 0):.2f}). Every split - the held-out "
                f"partition, model-search folds, tuning folds and the threshold's out-of-fold "
                f"predictions - keeps an entity's rows on one side.\n"
            )
        else:
            lines.append("Rows are treated as independent; no repeated-entity key was found.\n")
        lines.extend(f"- {r}" for r in group_decision.get("reasoning", []))
        others = [c for c in group_decision.get("candidates", [])
                  if c["column"] != group_decision.get("column")]
        if others:
            lines.append("\n**Other columns considered:**")
            lines.extend(
                f"- `{c['column']}` - {c['n_groups']} groups, {c['mean_group_size']} rows each "
                f"(score {c['score']:.2f})" for c in others
            )
        lines.append("")

    lines.append("## 4. Pre-Training Leakage Scan\n")
    lines.append(_fmt_flag_list(pre_training_leakage["flags"]) if pre_training_leakage else "Not applicable for this problem type.\n")

    if leaderboard:
        lines.append(f"## 5. Model Leaderboard ({leaderboard['problem_kind']}, primary metric: `{leaderboard['primary_metric']}`)\n")
        if leaderboard.get("budget_note"):
            lines.append(f"*{leaderboard['budget_note']}*\n")

        metric = leaderboard["primary_metric"]
        ok = sorted([r for r in leaderboard["results"] if r["status"] == "ok"],
                    key=lambda r: r["metrics"].get(metric, float("-inf")), reverse=True)
        lines.append(f"| Rank | Model | {metric} | Fit time (s) |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(ok, 1):
            lines.append(f"| {i} | {r['name']} | {r['metrics'].get(metric):.4f} | {r['fit_time_seconds']:.2f} |")

        screened = [r for r in leaderboard["results"] if r["status"] == "screened_out"]
        if screened:
            screened.sort(key=lambda r: r["metrics"].get(metric, float("-inf")), reverse=True)
            lines.append(
                "\n**Eliminated at the screening stage** (scored on a subsample with fewer folds, so "
                "these numbers are not comparable to the table above):\n"
            )
            lines.append(f"| Model | screening {metric} |")
            lines.append("|---|---|")
            for r in screened:
                lines.append(f"| {r['name']} | {r['metrics'].get(metric, float('nan')):.4f} |")

        failed = [r for r in leaderboard["results"] if r["status"] in ("failed", "skipped")]
        if failed:
            lines.append("\n**Not scored:**")
            for r in failed:
                lines.append(f"- {r['name']} ({r['status']}): {r['error']}")
        lines.append("")

    if time_series_setup:
        lines.append("### Time-series setup\n")
        period = time_series_setup.get("seasonal_period")
        lines.append(f"- **Seasonal period:** {period if period else 'none detected'} — "
                     f"{time_series_setup.get('seasonality_reason')}")
        lines.append(f"- **Target representation:** "
                     f"{'first differences' if time_series_setup.get('differenced') else 'levels'} — "
                     f"{time_series_setup.get('differencing_reason')}")
        if time_series_setup.get("differenced"):
            lines.append("- Predictions are reconstructed onto the original scale before scoring, so the "
                         "numbers below stay directly comparable to the baselines.")
        lines.append("")

    if time_series_baselines:
        lines.append("### Classical forecasting baselines (same folds, for comparison)\n")
        lines.append("| Baseline | r2 | RMSE |")
        lines.append("|---|---|---|")
        for b in time_series_baselines:
            lines.append(f"| {b['name']} | {b['metrics']['r2']:.4f} | {-b['metrics']['neg_rmse']:.4f} |")
        lines.append("")

    if hpo_results:
        lines.append("## 6. Hyperparameter Optimization\n")
        for r in hpo_results:
            if r["tuned"]:
                lines.append(f"- **{r['model_name']}**: {r['baseline_score']:.4f} -> {r['best_score']:.4f} "
                              f"({r['improvement']:+.4f}) over {r['n_trials']} Optuna trials.")
                lines.append(f"  - Best params: `{r['best_params']}`")
            else:
                lines.append(f"- **{r['model_name']}**: no tunable search space defined; baseline score kept ({r['baseline_score']:.4f}).")
        lines.append("")

    if explanation:
        lines.append("## 7. Selected Model & Explanation\n")
        lines.append(explanation["narrative"] + "\n")
        if explanation.get("feature_importances"):
            lines.append(f"**Top features ({explanation['importance_method']}):**")
            for name, val in explanation["feature_importances"][:10]:
                lines.append(f"- {name}: {val:.4f}")
            lines.append("")

    if held_out_metrics:
        lines.append("## 8. Held-Out Test Set Performance\n")
        lines.append("**Ranking metrics** (threshold-free):\n")
        for k, v in held_out_metrics.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

        if threshold_choice:
            lines.append("### Operating point\n")
            lines.append(
                "A ranking metric like ROC-AUC integrates over every threshold. The deployed "
                "model has to pick one. These are the numbers it actually produces.\n"
            )
            lines.append(f"**Decision threshold:** `{threshold_choice['threshold']:.4f}` "
                         f"(objective: `{threshold_choice['objective']}`)\n")
            lines.append(threshold_choice["reasoning"] + "\n")
            calibration = threshold_choice.get("calibration")
            if calibration:
                lines.append("### Probability calibration\n")
                lines.append(calibration["reasoning"] + "\n")
                before, after = calibration.get("reliability_before") or [], calibration.get("reliability_after") or []
                if before:
                    lines.append("Reliability, out of fold (each row is a tenth of the training partition, "
                                 "sorted by predicted probability):\n")
                    lines.append("| Mean predicted | Observed | Mean predicted, calibrated | Observed |")
                    lines.append("|---|---|---|---|")
                    for i, row in enumerate(before):
                        cal = after[i] if i < len(after) else None
                        lines.append(f"| {row['mean_predicted']:.3f} | {row['observed']:.3f} | "
                                     + (f"{cal['mean_predicted']:.3f} | {cal['observed']:.3f} |" if cal else "— | — |"))
                    lines.append("")
                held_out_cal = (held_out_operating_point or {}).get("calibration")
                if held_out_cal:
                    lines.append(
                        f"On the held-out test set ({held_out_cal.get('n_rows', '?')} rows): Brier "
                        f"{held_out_cal['brier_raw']:.4f} -> {held_out_cal['brier_calibrated']:.4f}, calibration "
                        f"error {held_out_cal['ece_raw']:.4f} -> {held_out_cal['ece_calibrated']:.4f}. A calibration "
                        f"curve needs far more rows than a ranking metric, so on a small holdout this can disagree "
                        f"with the out-of-fold evidence the choice was made on. Measured on the grouped fixture: "
                        f"worse on its 150-row holdout, and calibration error halved (0.088 -> 0.044) on 5,000 "
                        f"fresh rows.\n"
                    )

            if held_out_operating_point:
                lines.append("**On the held-out test set:**\n")
                lines.append("| Threshold | Precision | Recall | F1 | TP | FP | TN | FN |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for label, key in (("selected", "at_selected_threshold"),
                                    ("default 0.5", "at_default_threshold")):
                    m = held_out_operating_point.get(key)
                    if not m:
                        continue
                    lines.append(
                        f"| {label} (`{m['threshold']:.4f}`) | {m['precision']:.3f} | "
                        f"{m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fp']} | "
                        f"{m['tn']} | {m['fn']} |"
                    )
                sel = held_out_operating_point.get("at_selected_threshold")
                dfl = held_out_operating_point.get("at_default_threshold")
                if sel and dfl:
                    lines.append(
                        f"\nAt the selected threshold the model catches **{sel['tp']} of "
                        f"{sel['tp'] + sel['fn']}** positives, against **{dfl['tp']}** at the 0.5 "
                        f"default, for {sel['fp']} false alarms rather than {dfl['fp']}.\n"
                    )
            lines.append(
                "*The threshold was selected on out-of-fold predictions from the training "
                "partition. It was never fitted on the held-out rows above — doing so would be "
                "the same leakage this pipeline prevents everywhere else, arriving at the last "
                "step.*\n"
            )
        elif leaderboard and leaderboard.get("problem_kind") == "classification":
            lines.append(
                "*No decision threshold was selected — either the target is multiclass (a single "
                "cut point is not meaningful) or the winning model exposes no calibrated "
                "probabilities. Predictions use the 0.5 default.*\n"
            )

    # Placed immediately after the held-out metrics on purpose: those numbers
    # describe this artifact, and the two belong side by side.
    if model_artifact:
        lines.append("## 9. Persisted Model Artifact\n")
        if model_artifact.get("status") == "saved":
            lines.append(f"- **Model:** `{model_artifact['model_path']}`")
            lines.append(f"- **Training schema:** `{model_artifact['schema_path']}`")
            lines.append(f"- **Estimator class:** `{model_artifact['estimator_class']}`")
            if model_artifact.get("mlflow_model_dir"):
                lines.append(f"- **MLflow model:** `{model_artifact['mlflow_model_dir']}`")
            for w in model_artifact.get("warnings") or []:
                lines.append(f"- *Warning:* {w}")
            lines.append(
                "\nThe schema records column order, dtypes, semantic types, feature roles, target "
                "class labels, and per-feature reference distributions (quantiles for numeric "
                "columns, category frequencies for categorical ones) taken from the training "
                "partition — the baseline a later drift check compares against.\n"
            )
        else:
            lines.append(
                f"- **Model was NOT persisted.** {model_artifact.get('error', 'unknown error')}\n\n"
                "The metrics above therefore describe an estimator that no longer exists.\n"
            )

    if post_training_leakage:
        lines.append("## 10. Post-Training Leakage Scan\n")
        lines.append(_fmt_flag_list(post_training_leakage["flags"]))

    if clustering_summary:
        lines.append("## Clustering Results\n")
        lines.append(f"Selected k (silhouette sweep): {clustering_summary.get('best_k')}\n")
        lines.append("| Algorithm | k found | Silhouette | Calinski-Harabasz | Davies-Bouldin |")
        lines.append("|---|---|---|---|---|")
        for r in clustering_summary.get("ranked", []):
            m = r["metrics"]
            lines.append(f"| {r['name']} | {r['n_clusters_found']} | {m.get('silhouette', float('nan')):.3f} "
                          f"| {m.get('calinski_harabasz', float('nan')):.1f} | {m.get('davies_bouldin', float('nan')):.3f} |")
        lines.append("")

    lines.append("## Known Limitations of This Run\n")
    lines.append(
        "- Target/problem-type detection is a statistical best-guess in the absence of a data dictionary; "
        "the reasoning and alternative hypotheses above are logged specifically so this guess can be audited "
        "and overridden.\n"
        "- Leakage detection is heuristic (association thresholds, performance-suspicion thresholds) and "
        "cannot catch every leakage pattern — it's a screen, not a proof of a leak-free pipeline.\n"
        "- Automated feature interactions are pruned by mutual information on the training fold and can still "
        "include noise-driven artifacts on small or very noisy datasets.\n"
    )

    return "\n".join(lines)
