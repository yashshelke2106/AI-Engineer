"""
Conversational Q&A over a run's experiment history.

This grounds every answer in the actual MLflow-logged artifacts for a run
(the leaderboard, leakage report, problem-type decision, explanation) —
never a free-text guess. The routing here is deliberately simple keyword
matching rather than a general LLM chat layer: the point this module makes
is the *grounding* (answers come from real logged numbers, so "why did you
reject model X" can quote its actual CV score and failure reason), which
is the part that's easy to get wrong. A production system would put an
LLM in front of this to handle open-ended phrasing and call these same
lookups as tools — swapping that in later doesn't change what's below,
because the retrieval/grounding logic is exactly the part an LLM would
need to call anyway.
"""
from __future__ import annotations

import re

from autoeng.tracking.mlflow_tracker import get_run_artifact

_MODEL_NAME_PATTERN = re.compile(r"[a-z_]+")


def _find_mentioned_model(question: str, model_names: list[str]) -> str | None:
    tokens = set(_MODEL_NAME_PATTERN.findall(question.lower().replace("-", "_").replace(" ", "_")))
    q_norm = question.lower().replace("-", " ").replace("_", " ")
    for name in model_names:
        if name.replace("_", " ") in q_norm or name in tokens:
            return name
    return None


def answer_question(tracking_uri: str, run_id: str, question: str) -> str:
    q = question.lower().strip()

    leaderboard = get_run_artifact(tracking_uri, run_id, "model_leaderboard.json")
    explanation = get_run_artifact(tracking_uri, run_id, "model_explanation.json")
    leakage_pre = get_run_artifact(tracking_uri, run_id, "pre_training_leakage_report.json")
    leakage_post = get_run_artifact(tracking_uri, run_id, "post_training_leakage_report.json")
    problem_decision = get_run_artifact(tracking_uri, run_id, "problem_type_decision.json")
    structural = get_run_artifact(tracking_uri, run_id, "structural_cleaning_report.json")
    hpo = get_run_artifact(tracking_uri, run_id, "hpo_results.json")
    promotion = get_run_artifact(tracking_uri, run_id, "promotion_decision.json")

    if leaderboard is None:
        return f"No logged run found with id '{run_id}' at tracking URI '{tracking_uri}'."

    model_names = [r["name"] for r in leaderboard["results"]]

    # "why did you reject the latest model / the challenger / the retrained one"
    # Checked BEFORE the per-model branch: "the latest model" names no
    # leaderboard entry, so that branch would fall through to a generic answer
    # while the actual comparison sits logged and unread.
    if any(kw in q for kw in ("latest model", "challenger", "new model", "retrain", "promote")):
        if promotion is None:
            return ("No champion-challenger comparison was logged for this run. That decision "
                    "is only recorded on a retrain, so this run is either the original "
                    "training run or a challenger that has not been gated yet.")
        comparison = promotion.get("comparison") or {}
        lines = [promotion.get("reason", "")]
        if comparison:
            lines.append(
                f"Measured on {comparison.get('n_rows')} held-out rows over "
                f"{comparison.get('n_bootstrap')} bootstrap resamples of the paired difference: "
                f"champion {comparison.get('metric')}={comparison.get('champion_score'):.4f}, "
                f"challenger={comparison.get('challenger_score'):.4f}, "
                f"difference {comparison.get('difference'):+.4f} with "
                f"{100 * (1 - comparison.get('alpha', 0.05)):.0f}% CI "
                f"[{comparison.get('ci_low'):+.4f}, {comparison.get('ci_high'):+.4f}]."
            )
            if not comparison.get("excludes_zero"):
                lines.append(
                    "The interval spans zero, so the two models are not distinguishable on this "
                    "data. A challenger that wins by less than the comparison's own noise has "
                    "not won."
                )
        # Per-window figures when the gate scored more than one set of rows —
        # "rejected on the frozen holdout" and "rejected on this week's traffic"
        # are different findings, and the answer should say which it was.
        for name, window in (promotion.get("windows") or {}).items():
            label = name.replace("_", " ")
            window_comparison = window.get("comparison") or {}
            if window_comparison:
                lines.append(
                    f"{label}: {window.get('verdict')} — {window_comparison.get('metric')} "
                    f"champion {window_comparison.get('champion_score'):.4f}, challenger "
                    f"{window_comparison.get('challenger_score'):.4f}, CI "
                    f"[{window_comparison.get('ci_low'):+.4f}, {window_comparison.get('ci_high'):+.4f}] "
                    f"over {window_comparison.get('n_rows')} rows."
                )
            else:
                lines.append(f"{label}: {window.get('verdict')} — {window.get('reason')}")
        lines.extend(promotion.get("notes") or [])
        return '\n'.join(line for line in lines if line)

    # "why did you reject / not use <model>" or "why isn't <model> the winner"
    if any(kw in q for kw in ("reject", "why not", "why isn't", "instead of", "worse than")):
        mentioned = _find_mentioned_model(q, model_names)
        if mentioned:
            result = next((r for r in leaderboard["results"] if r["name"] == mentioned), None)
            winner = explanation["winner_name"]
            if mentioned == winner:
                return f"'{mentioned}' was NOT rejected — it's the model that was selected."
            if result["status"] != "ok":
                return f"'{mentioned}' was excluded because it failed during evaluation: {result['error']}"
            winner_score = explanation["winner_score"]
            metric = leaderboard["primary_metric"]
            gap = winner_score - result["metrics"].get(metric, float("nan"))
            return (
                f"'{mentioned}' scored {result['metrics'].get(metric):.4f} on {metric}, versus "
                f"{winner_score:.4f} for the selected model '{winner}' — a gap of {gap:.4f}. "
                f"It was a valid candidate, just not the best-performing one on this data."
            )
        return "Which model did you mean? " + ", ".join(model_names)

    # "why did you choose / pick <model>" or "why is the best model"
    if any(kw in q for kw in ("why did you choose", "why did you pick", "why is", "best model", "which model won", "winning model")):
        return explanation.get("narrative", "No explanation narrative was logged for this run.")

    # leakage questions
    if "leak" in q:
        flags = leakage_pre.get("flags", []) + leakage_post.get("flags", [])
        if not flags:
            return "No data leakage was flagged for this run — pre-training scan and post-training performance both looked clean."
        lines = [f"- [{f['severity']}] {f['description']}" for f in flags]
        return "Leakage findings for this run:\n" + "\n".join(lines)

    # problem type / target questions
    if any(kw in q for kw in ("problem type", "target", "what kind of problem", "classification or regression")):
        chosen = problem_decision["chosen"]
        lines = [f"Detected problem type: {chosen['problem_type']} (target column: {chosen['target_column']})."]
        lines.extend(f"  - {r}" for r in chosen["reasoning"])
        lines.append(f"Confidence: {problem_decision['confidence']:.2f}")
        if problem_decision["alternatives"]:
            lines.append("Other hypotheses considered: " + ", ".join(
                f"{a['problem_type']} (score {a['score']:.3f})" for a in problem_decision["alternatives"]
            ))
        return "\n".join(lines)

    # cleaning questions
    if any(kw in q for kw in ("clean", "missing value", "duplicate", "outlier")):
        return "Cleaning actions taken:\n" + "\n".join(f"- {a}" for a in structural.get("actions", ["(none needed)"]))

    # HPO questions
    if any(kw in q for kw in ("hyperparameter", "tuning", "tuned", "optuna")):
        results = hpo.get("hpo_results", [])
        if not results:
            return "No hyperparameter optimization was run for this model."
        lines = []
        for r in results:
            if r["tuned"]:
                lines.append(
                    f"- {r['model_name']}: {r['baseline_score']:.4f} -> {r['best_score']:.4f} "
                    f"({r['improvement']:+.4f}) over {r['n_trials']} trials."
                )
            else:
                lines.append(f"- {r['model_name']}: no tunable hyperparameter space defined; used defaults.")
        return "\n".join(lines)

    # feature importance
    if any(kw in q for kw in ("feature", "important", "why does the model use")):
        feats = explanation.get("feature_importances", [])
        if not feats:
            return "Feature importances were not available for this run."
        lines = [f"Top features ({explanation['importance_method']}):"]
        lines.extend(f"  {i+1}. {name} ({value:.4f})" for i, (name, value) in enumerate(feats[:10]))
        return "\n".join(lines)

    return (
        "I can answer questions about: why a model was chosen or rejected, data leakage findings, "
        "the detected problem type/target, cleaning actions taken, hyperparameter tuning results, "
        "and feature importances for this run. Try rephrasing, or name a specific model."
    )
