"""
T2-6 — a model card per run, assembled from what the run itself measured.

Nothing here comes from a template of plausible claims. Every line is a number
the run computed, or a limitation the run detected: its leakage flags, its
detection confidence, a near-trivial threshold, entity grouping, what drift
cannot see. The one thing the system cannot know is what the model is *for*, so
intended use is taken from the caller or stated as not supplied — never guessed.

Per-segment results are the part an aggregate hides. A held-out ROC-AUC of 0.85
can be 0.92 on one channel and 0.60 on another, and nothing in the headline says
so. A segment is flagged only when its whole 95% interval sits below the overall
figure (resampling entities when the run is grouped), so a small segment's
ordinary wobble is not reported as a finding, and a segment too small to judge
says that instead of printing a number.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoeng.modeling.threshold import binary_indicator, roc_auc_standard_error

#: Below this many held-out rows a segment gets a count, not a metric.
MIN_SEGMENT_ROWS = 30
#: Binary segments also need this many of each class for ROC-AUC to mean anything.
MIN_SEGMENT_CLASS_COUNT = 5
#: Segment on at most this many columns, the most important first.
MAX_SEGMENT_COLUMNS = 4
#: A categorical with more levels than this is not segmented (it would be a list, not a finding).
MAX_SEGMENTS_PER_COLUMN = 12
#: Held-out sets smaller than this get a limitation line about their precision.
SMALL_HOLDOUT_ROWS = 200
Z_95 = 1.96
BOOTSTRAP_ROUNDS = 300


def _bootstrap_se(metric, y: np.ndarray, pred: np.ndarray, groups: np.ndarray | None,
                  rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0) -> float | None:
    """Standard error of `metric(y, pred)`, resampling whole entities when grouped."""
    rng = np.random.default_rng(seed)
    units = np.unique(groups) if groups is not None else np.arange(len(y))
    if len(units) < 3:
        return None
    index = {u: np.flatnonzero(groups == u) for u in units} if groups is not None else None
    draws = []
    for _ in range(rounds):
        picked = rng.choice(units, size=len(units), replace=True)
        rows = np.concatenate([index[u] for u in picked]) if index is not None else picked
        try:
            draws.append(float(metric(y[rows], pred[rows])))
        except ValueError:
            continue
    return float(np.std(draws, ddof=1)) if len(draws) > 10 else None


def _segment_frames(X_test: pd.DataFrame, columns: list[str], numeric_columns: list[str]):
    """(column, label, mask) for each segment: categorical levels, numeric quartiles."""
    for column in columns:
        values = X_test[column]
        if column in numeric_columns:
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.nunique() < 4:
                continue
            edges = np.unique(np.nanquantile(numeric, [0, 0.25, 0.5, 0.75, 1.0]))
            if len(edges) < 3:
                continue
            bins = pd.cut(numeric, edges, include_lowest=True, duplicates="drop")
            for interval in bins.cat.categories:
                yield column, f"{interval.left:.4g} to {interval.right:.4g}", (bins == interval).to_numpy()
            continue
        levels = values.astype("object").where(values.notna(), "(missing)")
        if levels.nunique() > MAX_SEGMENTS_PER_COLUMN:
            continue
        for level in levels.value_counts().index:
            yield column, str(level), (levels == level).to_numpy()


def segment_results(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, *, problem_kind: str,
    candidate_columns: list[str], numeric_columns: list[str],
    groups: np.ndarray | None = None, threshold: float | None = None,
) -> dict[str, Any]:
    """Held-out performance per segment of the most important columns."""
    y = np.asarray(y_test)
    groups = None if groups is None else np.asarray(groups)
    out: dict[str, Any] = {"columns": [], "segments": []}

    if problem_kind == "binary":
        indicator, positive = binary_indicator(y)
        # ROC-AUC needs a ranking, not a probability. A winner like LinearSVC has
        # only decision_function; asking it for predict_proba silently emptied
        # this section on a real run until this fallback existed.
        if hasattr(pipeline, "predict_proba"):
            scores = np.asarray(pipeline.predict_proba(X_test))[:, 1]
        else:
            scores = np.asarray(pipeline.decision_function(X_test), dtype=float)
            threshold = None  # a probability threshold means nothing on a margin
        metric_name, higher_is_better = "roc_auc", True

        def metric(yy, ss):
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(yy, ss)

        truth, pred = indicator, scores
    elif problem_kind == "multiclass":
        from sklearn.metrics import accuracy_score as metric
        truth, pred = y, np.asarray(pipeline.predict(X_test))
        metric_name, higher_is_better = "accuracy", True
    else:
        from sklearn.metrics import mean_absolute_error as metric
        truth, pred = y.astype(float), np.asarray(pipeline.predict(X_test), dtype=float)
        metric_name, higher_is_better = "mae", False

    overall = float(metric(truth, pred))
    out["metric"], out["overall"], out["higher_is_better"] = metric_name, overall, higher_is_better

    for column, label, mask in _segment_frames(X_test, candidate_columns, numeric_columns):
        if column not in out["columns"]:
            out["columns"].append(column)
        n = int(mask.sum())
        entry: dict[str, Any] = {"column": column, "segment": label, "n_rows": n,
                                 "value": None, "se": None, "flag": None, "note": None}
        seg_truth, seg_pred = truth[mask], pred[mask]
        seg_groups = groups[mask] if groups is not None else None
        if problem_kind == "binary":
            entry["positive_rate"] = float(seg_truth.mean()) if n else None
            if threshold is not None and n:
                flagged = seg_pred >= threshold
                tp = int((flagged & (seg_truth == 1)).sum())
                entry["precision"] = tp / int(flagged.sum()) if flagged.sum() else None
                entry["recall"] = tp / int(seg_truth.sum()) if seg_truth.sum() else None
        if n < MIN_SEGMENT_ROWS:
            entry["note"] = f"too few held-out rows to judge ({n} < {MIN_SEGMENT_ROWS})"
            out["segments"].append(entry)
            continue
        if problem_kind == "binary" and min(seg_truth.sum(), n - seg_truth.sum()) < MIN_SEGMENT_CLASS_COUNT:
            entry["note"] = (f"one class appears fewer than {MIN_SEGMENT_CLASS_COUNT} times here; "
                             f"a ranking metric would be noise")
            out["segments"].append(entry)
            continue
        entry["value"] = float(metric(seg_truth, seg_pred))
        entry["se"] = (roc_auc_standard_error(seg_truth, seg_pred, groups=seg_groups)
                       if problem_kind == "binary" else _bootstrap_se(metric, seg_truth, seg_pred, seg_groups))
        if entry["se"] is not None:
            # The whole 95% interval has to sit on the wrong side of the overall
            # figure; a gap inside the segment's own noise is not a finding.
            worse = (entry["value"] + Z_95 * entry["se"] < overall) if higher_is_better \
                else (entry["value"] - Z_95 * entry["se"] > overall)
            if worse:
                entry["flag"] = "worse than overall beyond noise"
        out["segments"].append(entry)
    return out


def choose_segment_columns(roles: dict[str, Any], importances: list | None) -> list[str]:
    """Low-cardinality categoricals and numerics, most important first."""
    eligible = list(roles.get("low_card_categorical_columns") or []) + list(roles.get("numeric_columns") or [])
    ranked = [name for name, _ in (importances or []) if name in eligible]
    ordered = ranked + [c for c in eligible if c not in ranked]
    # Categoricals are the natural segments; keep the most important numeric too.
    categorical = [c for c in ordered if c in (roles.get("low_card_categorical_columns") or [])]
    numeric = [c for c in ordered if c in (roles.get("numeric_columns") or [])][:1]
    return (categorical + numeric)[:MAX_SEGMENT_COLUMNS]


def _limitations(*, problem_decision, target_source, pre_leak, post_leak, threshold_choice,
                 held_out_operating_point, group_decision, role_assignment, model_artifact,
                 n_test, problem_type, segments) -> list[str]:
    notes: list[str] = []
    confidence = (problem_decision or {}).get("confidence")
    if target_source != "supplied by caller" and confidence is not None and confidence < 0.6:
        runners = [h.get("target_column") or h.get("problem_type") for h in
                   (problem_decision or {}).get("alternatives", [])[:2]]
        notes.append(
            f"The target and problem type were inferred, at confidence {confidence:.2f}"
            + (f" (runners-up: {', '.join(str(r) for r in runners if r)})" if any(runners) else "")
            + ". Pin them with --target / --problem-type if this is not what the model should predict."
        )
    for report, stage in ((pre_leak, "before training"), (post_leak, "after training")):
        for flag in (report or {}).get("flags", []):
            if str(flag.get("severity", "")).lower() in ("critical", "warning"):
                cols = ", ".join(f"`{c}`" for c in flag.get("columns") or [])
                notes.append(f"Leakage check {stage} ({flag['severity']}{', ' + cols if cols else ''}): "
                             f"{flag.get('description')}")
    group = group_decision or {}
    if group.get("detected_column") and not group.get("column"):
        notes.append(f"`{group['detected_column']}` identifies repeated entities but grouping was turned off: "
                     f"every score in this card may be inflated by entities appearing on both sides of the split.")
    elif group.get("column"):
        notes.append(f"Rows were grouped by `{group['column']}`; the figures are for entities the model has not seen. "
                     f"Serving payloads need that column for the lifecycle's leak exclusions to work.")
    if problem_type == "binary_classification" and not threshold_choice:
        notes.append("No decision threshold was selected: the winning model exposes no probabilities, so its "
                     "labels use the estimator's own default cut and no operating point was measured. The "
                     "ranking figures are unaffected.")
    if (threshold_choice or {}).get("near_trivial"):
        notes.append("The selected threshold labels almost everything positive, so its F1 barely beats doing that "
                     "outright. Judge this model by its ranking (ROC-AUC), and set a cost-based objective if the "
                     "decision matters.")
    calibration = (threshold_choice or {}).get("calibration") or {}
    held_cal = (held_out_operating_point or {}).get("calibration")
    if calibration.get("method") == "platt" and held_cal and held_cal["brier_calibrated"] > held_cal["brier_raw"]:
        notes.append(f"Calibration was chosen on out-of-fold evidence but looks worse on the {n_test}-row holdout; "
                     f"calibration curves need far more rows than ranking metrics.")
    if n_test and n_test < SMALL_HOLDOUT_ROWS:
        notes.append(f"The held-out set is {n_test} rows, so its figures carry wide uncertainty; the cross-validated "
                     f"figures use the whole training partition and are the steadier estimate.")
    if role_assignment.get("text_columns"):
        notes.append("Drift monitoring watches text columns only through length and word count. A change in "
                     "vocabulary would not be seen, even though the model reads the words.")
    flagged = [s for s in (segments or {}).get("segments", []) if s.get("flag")]
    for s in flagged:
        notes.append(f"Weaker on `{s['column']}` = {s['segment']}: {segments['metric']} {s['value']:.3f} against "
                     f"{segments['overall']:.3f} overall ({s['n_rows']} held-out rows).")
    if not model_artifact or model_artifact.get("status") != "saved":
        why = (model_artifact or {}).get("error") or (
            "time-series and clustering runs do not persist a model" if problem_type in
            ("time_series_forecasting", "clustering") else "no artifact was written")
        notes.append(f"No servable model was persisted: {why}.")
    return notes


def _selected_cv_metrics(winner, explanation, leaderboard, schema) -> dict[str, Any] | None:
    """The cross-validated figures of the model that was actually selected.

    The leaderboard row holds the UNTUNED configuration. For a tuned winner only
    the primary metric was re-measured (the selection's own score, carried in the
    explanation), so that is all the card shows rather than pairing a tuned
    model's held-out numbers with its untuned self's CV numbers.
    """
    metrics = dict((winner or {}).get("metrics") or {})
    primary = (leaderboard or {}).get("primary_metric")
    selected = (explanation or {}).get("winner_score")
    if (schema.get("model") or {}).get("selection_source") == "hyperparameter tuning" and primary:
        return {primary: selected} if selected is not None else {}
    return metrics or None


def build_model_card(
    *, run_name: str, run_id: str | None, dataset_path: str, problem_decision: dict[str, Any],
    target_source: str, role_assignment: dict[str, Any], profile_summary: dict[str, Any],
    explanation: dict[str, Any] | None, held_out_metrics: dict[str, float] | None,
    leaderboard: dict[str, Any] | None, threshold_choice: dict[str, Any] | None,
    held_out_operating_point: dict[str, Any] | None, group_decision: dict[str, Any] | None,
    pre_training_leakage: dict[str, Any] | None, post_training_leakage: dict[str, Any] | None,
    model_artifact: dict[str, Any] | None, segments: dict[str, Any] | None,
    n_train: int | None, n_test: int | None, intended_use: str | None = None,
) -> dict[str, Any]:
    chosen = (problem_decision or {}).get("chosen") or {}
    problem_type = chosen.get("problem_type")
    columns = (profile_summary or {}).get("columns", {})
    time_ranges = {name: col.get("datetime_range") for name, col in columns.items()
                   if col.get("datetime_range") and name in (role_assignment.get("datetime_columns") or [])}
    target = chosen.get("target_column")
    target_profile = columns.get(target, {}) if target else {}

    winner = next((r for r in (leaderboard or {}).get("results", [])
                   if r.get("name") == (explanation or {}).get("winner_name")), None)
    # The card cites the artifact's own record rather than re-deriving it, so the
    # two cannot disagree about which model this is.
    schema: dict[str, Any] = {}
    schema_path = (model_artifact or {}).get("schema_path")
    if schema_path and Path(schema_path).exists():
        try:
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            schema = {}
    card = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name, "run_id": run_id,
        "model": {
            "name": (explanation or {}).get("winner_name"),
            "selection": (schema.get("model") or {}).get("selection_source"),
            "artifact": (model_artifact or {}).get("model_path") if (model_artifact or {}).get("status") == "saved" else None,
            "library_versions": schema.get("library_versions"),
        },
        "intended_use": intended_use or None,
        "task": {
            "problem_type": problem_type, "target": target, "target_source": target_source,
            "detection_confidence": (problem_decision or {}).get("confidence"),
            "target_distribution": (target_profile.get("top_categories") or target_profile.get("numeric_stats")),
        },
        "training_data": {
            "source": dataset_path, "rows_train": n_train, "rows_held_out": n_test,
            "features": {k: role_assignment.get(k) or [] for k in
                         ("numeric_columns", "categorical_columns", "datetime_columns", "text_columns")},
            "excluded": role_assignment.get("excluded_columns") or [],
            "time_ranges": time_ranges,
            "group_column": (group_decision or {}).get("column"),
        },
        "evaluation": {
            "held_out": held_out_metrics,
            "cross_validated": _selected_cv_metrics(winner, explanation, leaderboard, schema),
            "cv_note": ("Tuning re-scored only the primary metric; the untuned configuration's other "
                        "cross-validated figures would not describe this model, so they are not shown."
                        if (schema.get("model") or {}).get("selection_source") == "hyperparameter tuning"
                        else None),
            "primary_metric": (leaderboard or {}).get("primary_metric"),
            "operating_point": {
                "threshold": (threshold_choice or {}).get("threshold"),
                "objective": (threshold_choice or {}).get("objective"),
                "held_out": (held_out_operating_point or {}).get("at_selected_threshold"),
                "calibration": ((threshold_choice or {}).get("calibration") or {}).get("method"),
            } if threshold_choice else None,
        },
        "segments": segments,
    }
    card["limitations"] = _limitations(
        problem_decision=problem_decision, target_source=target_source,
        pre_leak=pre_training_leakage, post_leak=post_training_leakage, threshold_choice=threshold_choice,
        held_out_operating_point=held_out_operating_point, group_decision=group_decision,
        role_assignment=role_assignment, model_artifact=model_artifact, n_test=n_test,
        problem_type=problem_type, segments=segments,
    )
    return card


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_model_card(card: dict[str, Any]) -> str:
    model, task, data, evaluation = card["model"], card["task"], card["training_data"], card["evaluation"]
    lines = [f"# Model card — {card['run_name']}", ""]
    lines.append(f"Generated {card['generated_at']} from run `{card['run_id'] or 'untracked'}`. Every figure "
                 f"below was measured by that run; nothing is filled in from a template.\n")

    lines.append("## Model\n")
    lines.append(f"- **Algorithm:** {_fmt(model['name'])} ({_fmt(model['selection'])})")
    lines.append(f"- **Artifact:** {_fmt(model['artifact'])}")
    if model.get("library_versions"):
        lines.append("- **Library versions:** " + ", ".join(f"{k} {v}" for k, v in model["library_versions"].items()))
    lines.append("")

    lines.append("## Intended use\n")
    lines.append(card["intended_use"] or
                 "*Not supplied.* The system can infer what this model predicts, not what it is for or who may rely "
                 "on it. Pass `--intended-use` to record that here.")
    lines.append("")

    lines.append("## Task\n")
    lines.append(f"- **Predicts:** `{_fmt(task['target'])}` ({_fmt(task['problem_type'])}; target "
                 f"{task['target_source']}"
                 + (f", detection confidence {task['detection_confidence']:.2f}" if task.get("detection_confidence")
                    is not None and task["target_source"] != "supplied by caller" else "") + ")")
    distribution = task.get("target_distribution")
    if isinstance(distribution, list):
        lines.append("- **Target in the data:** " + ", ".join(f"{v} ({n})" for v, n in distribution[:6]))
    elif isinstance(distribution, dict):
        lines.append(f"- **Target in the data:** mean {_fmt(distribution.get('mean'))}, "
                     f"sd {_fmt(distribution.get('std'))}")
    lines.append("")

    lines.append("## Training data\n")
    lines.append(f"- **Source:** `{data['source']}`")
    lines.append(f"- **Rows:** {_fmt(data['rows_train'])} trained on, {_fmt(data['rows_held_out'])} held out")
    for kind, names in data["features"].items():
        if names:
            lines.append(f"- **{kind.replace('_columns', '').replace('_', ' ').title()} features:** "
                         + ", ".join(f"`{n}`" for n in names))
    if data["excluded"]:
        lines.append("- **Excluded:** " + ", ".join(f"`{n}`" for n in data["excluded"]))
    for name, span in (data.get("time_ranges") or {}).items():
        lines.append(f"- **Time window (`{name}`):** {span[0]} to {span[1]}")
    if data.get("group_column"):
        lines.append(f"- **Grouped by:** `{data['group_column']}` (whole entities on one side of every split)")
    lines.append("")

    lines.append("## Evaluation\n")
    if evaluation.get("held_out"):
        lines.append("| Metric | Held out | Cross-validated (training partition) |")
        lines.append("|---|---|---|")
        cv = evaluation.get("cross_validated") or {}
        for metric, value in evaluation["held_out"].items():
            lines.append(f"| {metric} | {_fmt(value)} | {_fmt(cv.get(metric))} |")
        lines.append("")
        if evaluation.get("cv_note"):
            lines.append(f"*{evaluation['cv_note']}*\n")
    op = evaluation.get("operating_point")
    if op:
        held = op.get("held_out") or {}
        lines.append(f"**Operating point:** threshold {_fmt(op['threshold'])}, chosen by `{op['objective']}` on "
                     f"out-of-fold predictions. Held out: precision {_fmt(held.get('precision'), 3)}, recall "
                     f"{_fmt(held.get('recall'), 3)}. Probability calibration: {op.get('calibration') or 'none'}.\n")

    segments = card.get("segments")
    if segments and segments.get("error"):
        lines.append("## Results by segment\n")
        lines.append(f"*Could not be computed: {segments['error']}.* The aggregate figures above may hide a "
                     f"segment the model fails on.\n")
    elif segments and segments.get("segments"):
        better = "higher" if segments["higher_is_better"] else "lower"
        lines.append("## Results by segment\n")
        lines.append(f"Held-out `{segments['metric']}` ({better} is better) — overall "
                     f"{segments['overall']:.3f}. A segment is flagged only when its whole 95% interval is on the "
                     f"wrong side of the overall figure.\n")
        lines.append(f"| Column | Segment | Rows | {segments['metric']} | ± 1.96 SE | Note |")
        lines.append("|---|---|---|---|---|---|")
        for s in segments["segments"]:
            spread = f"{Z_95 * s['se']:.3f}" if s.get("se") is not None else "—"
            note = s.get("flag") or s.get("note") or ""
            lines.append(f"| `{s['column']}` | {s['segment']} | {s['n_rows']} | {_fmt(s.get('value'), 3)} | "
                         f"{spread} | {note} |")
        lines.append("")

    lines.append("## Limitations\n")
    if card["limitations"]:
        lines.extend(f"- {note}" for note in card["limitations"])
    else:
        lines.append("- None detected by this run's checks. That is not the same as none existing.")
    lines.append("")
    return "\n".join(lines)
