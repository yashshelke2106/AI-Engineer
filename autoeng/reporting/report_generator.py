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
    lines.append(f"- Text features (length/word-count stats): {role_assignment['text_columns'] or '(none)'}")
    lines.append(f"- Excluded (identifiers/constants): {role_assignment['excluded_columns'] or '(none)'}\n")

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
        for k, v in held_out_metrics.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    if post_training_leakage:
        lines.append("## 9. Post-Training Leakage Scan\n")
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
