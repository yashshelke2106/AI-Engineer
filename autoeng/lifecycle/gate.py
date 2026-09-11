"""
The champion–challenger gate.

A retrained model is a hypothesis, not an improvement. This decides whether it
replaces the model in production, and the entire difficulty is in one line
from the brief: *a challenger winning by 0.002 on a metric that swings 0.02
between folds has not won.*

Comparing two point estimates and promoting the larger is a coin flip dressed
as a decision, and it ratchets — every deploy takes the lucky side of the
noise, so the recorded "improvements" accumulate while the model does not. The
gate therefore bootstraps the **paired** difference (both models scoring the
same resampled rows, so the resampling noise they share cancels) and promotes
only when the interval excludes zero.

**Three outcomes, not two.** Promoted, rejected, and inconclusive. "We cannot
tell yet" is the honest answer far more often than either of the others, and
collapsing it into one of them turns the gate into either a rubber stamp or a
wall. Inconclusive keeps the champion — the incumbent wins ties, because
replacing a known model with an indistinguishable one is churn with a
deployment risk attached.

**Two evaluation windows, and neither model may have trained on either.**

  - The *frozen holdout* is the champion's own held-out split, persisted with
    the artifact and excluded from every retraining frame. Common to both
    models, stable across generations.
  - The *forward window* is ground truth that arrived for predictions the
    challenger did not train on, per its retrain manifest. It is what the world
    looks like now, which the frozen holdout by definition is not.

The forward window leads when it has enough rows to say anything, and the
frozen holdout decides otherwise. See `combine_windows` for why the obvious
rule — "a regression on either window disqualifies" — would block exactly the
retrains that genuine concept drift makes necessary.

The decision is logged with the numbers that produced it, so
`ask <run_id> "why did you reject the latest model"` answers from the actual
intervals rather than from a stored sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Enough resamples that the interval endpoints are stable to ~0.001; beyond
# this the cost grows and the answer does not change.
DEFAULT_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05
# Below this the interval is so wide that every comparison is inconclusive
# anyway; saying so is more useful than returning a number nobody should read.
MIN_ROWS_FOR_GATE = 30

FROZEN_HOLDOUT = "frozen_holdout"
FORWARD_WINDOW = "forward_window"


class GateVerdict(str, Enum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) else 0.0


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def _neg_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Negated so that "higher is better" holds for every metric here, which is
    # what lets the comparison logic stay metric-agnostic.
    return -float(np.sqrt(((y_true - y_pred) ** 2).mean()))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "f1": _f1, "accuracy": _accuracy, "recall": _recall, "precision": _precision,
    "r2": _r2, "neg_rmse": _neg_rmse,
}
# These score class 1 as the positive class, so labels must be mapped onto it.
_POSITIVE_CLASS_METRICS = {"f1", "precision", "recall"}


@dataclass
class Comparison:
    metric: str
    champion_score: float
    challenger_score: float
    difference: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    n_rows: int
    alpha: float

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["excludes_zero"] = self.excludes_zero
        return d


@dataclass
class GateDecision:
    verdict: GateVerdict
    promote: bool
    reason: str
    comparison: Comparison | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "promote": self.promote,
            "reason": self.reason,
            "notes": self.notes,
            "comparison": self.comparison.as_dict() if self.comparison else None,
        }


def bootstrap_paired_difference(
    y_true, champion_pred, challenger_pred, metric: str = "f1",
    n_bootstrap: int = DEFAULT_BOOTSTRAP, alpha: float = DEFAULT_ALPHA,
    random_state: int = 42,
) -> Comparison:
    """
    Percentile CI for (challenger - champion) on the same rows.

    **Paired.** One set of resampled row indices is used to score both models,
    so the sampling noise they share cancels instead of being counted twice.
    Resampling them independently inflates the variance and turns genuine wins
    inconclusive — a gate that is wrong in the safe direction is still wrong,
    and it is the direction that quietly freezes a model in place forever.
    """
    y_true = np.asarray(y_true)
    champion_pred = np.asarray(champion_pred)
    challenger_pred = np.asarray(challenger_pred)
    score = METRICS[metric]

    champion_score = score(y_true, champion_pred)
    challenger_score = score(y_true, challenger_pred)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    differences = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        differences[i] = score(y_true[idx], challenger_pred[idx]) - score(y_true[idx], champion_pred[idx])

    low, high = np.percentile(differences, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Comparison(
        metric=metric, champion_score=float(champion_score),
        challenger_score=float(challenger_score),
        difference=float(challenger_score - champion_score),
        ci_low=float(low), ci_high=float(high),
        n_bootstrap=int(n_bootstrap), n_rows=int(n), alpha=float(alpha),
    )


def evaluate_gate(
    y_true, champion_pred, challenger_pred, metric: str = "f1",
    n_bootstrap: int = DEFAULT_BOOTSTRAP, alpha: float = DEFAULT_ALPHA,
    notes: list[str] | None = None,
) -> GateDecision:
    """Decide whether the challenger replaces the champion, on one set of rows."""
    y_true = np.asarray(y_true)
    notes = list(notes or [])

    if len(y_true) < MIN_ROWS_FOR_GATE:
        return GateDecision(
            verdict=GateVerdict.INCONCLUSIVE, promote=False, notes=notes,
            reason=(f"Too few rows to compare ({len(y_true)}, below {MIN_ROWS_FOR_GATE}). "
                    f"Any interval over this little data would exclude zero only by accident. "
                    f"The champion stays."),
        )

    comparison = bootstrap_paired_difference(
        y_true, champion_pred, challenger_pred, metric, n_bootstrap, alpha,
    )
    interval = (f"{metric} {comparison.challenger_score:.4f} against the champion's "
                f"{comparison.champion_score:.4f} ({comparison.difference:+.4f}), "
                f"{100 * (1 - alpha):.0f}% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] "
                f"over {comparison.n_rows} rows")

    if comparison.ci_low > 0:
        return GateDecision(
            verdict=GateVerdict.PROMOTED, promote=True, comparison=comparison, notes=notes,
            reason=(f"Challenger promoted: {interval}. The interval excludes zero, so the "
                    f"improvement is larger than the noise in the comparison."),
        )
    if comparison.ci_high < 0:
        return GateDecision(
            verdict=GateVerdict.REJECTED, promote=False, comparison=comparison, notes=notes,
            reason=(f"Challenger rejected: it is worse. {interval}. The whole interval lies "
                    f"below zero, so this is a real regression rather than an unlucky sample."),
        )
    return GateDecision(
        verdict=GateVerdict.INCONCLUSIVE, promote=False, comparison=comparison, notes=notes,
        reason=(f"Challenger not promoted: {interval}. The interval spans zero, so the "
                f"difference is within the noise of the comparison and the two models are "
                f"not distinguishable on this data. The champion stays — replacing a known "
                f"model with an indistinguishable one is churn with a deployment risk."),
    )


def _predict_with_threshold(estimator, X, threshold: float | None) -> np.ndarray:
    """Labels at the model's own stored operating point, as serving produces them."""
    if threshold is not None and hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))[:, 1]
        classes = list(getattr(estimator, "classes_", [0, 1]))
        return np.where(proba >= threshold, classes[1], classes[0])
    return np.asarray(estimator.predict(X))


def primary_metric_for(problem_type: str | None) -> str:
    if problem_type == "regression":
        return "r2"
    if problem_type == "multiclass_classification":
        return "accuracy"
    return "f1"


def compare_saved_models(
    champion_estimator, challenger_estimator, X, y_true,
    metric: str = "f1", champion_threshold: float | None = None,
    challenger_threshold: float | None = None, positive_label: Any = None, **kwargs: Any,
) -> GateDecision:
    """
    Score two models on the same rows and gate on the difference.

    Each model is applied at **its own** stored threshold. Comparing a
    challenger tuned for recall against a champion at 0.5 would measure the
    threshold, not the model — and the operating point is part of what was
    retrained.
    """
    notes = []
    if champion_threshold != challenger_threshold:
        notes.append(
            f"The models are scored at their own thresholds "
            f"({champion_threshold} and {challenger_threshold}); the retrained operating "
            f"point is part of what is being compared."
        )
    y = np.asarray(y_true)
    champion_pred = _predict_with_threshold(champion_estimator, X, champion_threshold)
    challenger_pred = _predict_with_threshold(challenger_estimator, X, challenger_threshold)
    if positive_label is not None and metric in _POSITIVE_CLASS_METRICS:
        # The metrics count class 1 as positive. A target labelled
        # "benign"/"malignant" would otherwise score F1 = 0 for both models and
        # every gate would read as a tie — silently freezing the champion.
        y, champion_pred, challenger_pred = (
            (np.asarray(a) == positive_label).astype(int) for a in (y, champion_pred, challenger_pred)
        )
    return evaluate_gate(y, champion_pred, challenger_pred, metric=metric, notes=notes, **kwargs)


@dataclass
class LifecycleGateDecision:
    verdict: GateVerdict
    promote: bool
    reason: str
    windows: dict[str, GateDecision] = field(default_factory=dict)
    primary_window: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        primary = self.windows.get(self.primary_window) if self.primary_window else None
        return {
            "verdict": self.verdict.value,
            "promote": self.promote,
            "reason": self.reason,
            "notes": self.notes,
            "primary_window": self.primary_window,
            # Kept at the top level in the shape autoeng/explain/qa.py reads.
            "comparison": primary.comparison.as_dict() if primary and primary.comparison else None,
            "windows": {name: decision.as_dict() for name, decision in self.windows.items()},
        }


def _label(window: str) -> str:
    return window.replace("_", " ")


def combine_windows(windows: dict[str, GateDecision], notes: list[str] | None = None) -> LifecycleGateDecision:
    """
    One verdict from up to two windows.

    The forward window leads when it has enough rows to say anything, because
    it is the world as it is now. The frozen holdout is the world the champion
    was built for: a stable guard against regressions, and the deciding vote
    when there is not yet enough forward traffic.

    That ordering is deliberate, and the obvious alternative is a trap. "A
    regression on either window disqualifies" sounds safer, but under genuine
    concept drift a correct challenger MUST score worse on the old holdout —
    the relationship it was retrained to learn is exactly what changed. That
    rule would block every adaptation precisely when drift detection has just
    said one is needed, and the lifecycle would fail silently at the one moment
    it exists for. So a frozen-holdout regression alongside a real forward-window
    win promotes, with the regression stated rather than hidden.

    A forward-window regression always rejects.
    """
    notes = list(notes or [])
    holdout = windows.get(FROZEN_HOLDOUT)
    forward = windows.get(FORWARD_WINDOW)
    # evaluate_gate only attaches a comparison once a window clears
    # MIN_ROWS_FOR_GATE, so this is "has enough rows to say anything".
    forward_speaks = forward is not None and forward.comparison is not None

    if holdout is None and not forward_speaks:
        if forward is not None:
            notes.append(f"The forward window is too small to decide on: {forward.reason}")
        return LifecycleGateDecision(
            GateVerdict.INCONCLUSIVE, False,
            "No usable evaluation data: the champion has no frozen holdout, and too few labelled "
            "predictions the challenger did not train on have arrived. The champion stays.",
            windows, None, notes,
        )

    if forward_speaks:
        if forward.verdict == GateVerdict.REJECTED:
            return LifecycleGateDecision(
                GateVerdict.REJECTED, False,
                f"Challenger rejected: it is measurably worse on the forward window, the world "
                f"as it is now. {forward.reason}",
                windows, FORWARD_WINDOW, notes,
            )
        if forward.verdict == GateVerdict.PROMOTED:
            if holdout is not None and holdout.verdict == GateVerdict.REJECTED:
                notes.append(
                    "The challenger is worse on the frozen holdout while better on recent traffic. "
                    "That is what a genuine change in the input-outcome relationship looks like: "
                    "the holdout describes the world the champion was built for, the forward "
                    f"window the world as it is now. {holdout.reason}"
                )
            elif holdout is not None and holdout.verdict == GateVerdict.INCONCLUSIVE:
                notes.append(f"The frozen holdout did not confirm the improvement: {holdout.reason}")
            return LifecycleGateDecision(
                GateVerdict.PROMOTED, True,
                f"Challenger promoted on the forward window. {forward.reason}",
                windows, FORWARD_WINDOW, notes,
            )
        notes.append(f"The forward window could not separate the models: {forward.reason}")
    elif forward is not None:
        notes.append(f"The forward window is too small to decide on: {forward.reason}")

    if holdout is None:
        return LifecycleGateDecision(
            GateVerdict.INCONCLUSIVE, False,
            "Challenger not promoted: the forward window cannot separate the models and the "
            "champion has no frozen holdout to break the tie.",
            windows, FORWARD_WINDOW, notes,
        )
    if holdout.verdict == GateVerdict.REJECTED:
        return LifecycleGateDecision(
            GateVerdict.REJECTED, False,
            f"Challenger rejected: it is measurably worse on the frozen holdout, and there is no "
            f"recent evidence to set against that. {holdout.reason}",
            windows, FROZEN_HOLDOUT, notes,
        )
    if holdout.verdict == GateVerdict.PROMOTED:
        return LifecycleGateDecision(
            GateVerdict.PROMOTED, True,
            f"Challenger promoted on the frozen holdout. {holdout.reason}",
            windows, FROZEN_HOLDOUT, notes,
        )
    return LifecycleGateDecision(
        GateVerdict.INCONCLUSIVE, False,
        f"Challenger not promoted: no window shows it is better beyond the noise. {holdout.reason}",
        windows, FROZEN_HOLDOUT, notes,
    )


def gate_challenger(
    champion_dir: str | Path,
    challenger_dir: str | Path,
    store=None,
    manifest: dict[str, Any] | None = None,
    metric: str | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> LifecycleGateDecision:
    """
    Compare two persisted models on data neither trained on, and decide.

    Without a retrain manifest the forward window is skipped rather than
    guessed at: there is no way to know which logged predictions the challenger
    was fitted to, and scoring it on those would rig the comparison.
    """
    from autoeng.registry.model_store import MODEL_FILENAME, SCHEMA_FILENAME, load_holdout, load_model
    from autoeng.serving.predictor import threshold_from_schema

    champion_dir, challenger_dir = Path(champion_dir), Path(challenger_dir)
    champion = load_model(champion_dir / MODEL_FILENAME, champion_dir / SCHEMA_FILENAME)
    challenger = load_model(challenger_dir / MODEL_FILENAME, challenger_dir / SCHEMA_FILENAME)
    champion_schema, challenger_schema = champion.schema or {}, challenger.schema or {}
    notes = list(champion.warnings) + list(challenger.warnings)

    target = (champion_schema.get("target") or {}).get("column")
    challenger_target = (challenger_schema.get("target") or {}).get("column")
    if target != challenger_target:
        return LifecycleGateDecision(
            GateVerdict.REJECTED, False,
            f"Challenger rejected without scoring: it predicts '{challenger_target}' but the "
            f"champion predicts '{target}'. A retrain that changed its target is a different "
            f"model, not a better one.",
            notes=notes,
        )

    metric = metric or primary_metric_for(champion_schema.get("problem_type"))
    champion_threshold, _ = threshold_from_schema(champion_schema)
    challenger_threshold, _ = threshold_from_schema(challenger_schema)
    class_labels = (champion_schema.get("target") or {}).get("class_labels")
    positive = class_labels[1] if class_labels and len(class_labels) == 2 else None
    champion_features = list(champion_schema.get("feature_columns") or [])
    challenger_features = list(challenger_schema.get("feature_columns") or [])

    def score(frame, y) -> GateDecision:
        missing = sorted({c for c in champion_features + challenger_features if c not in frame.columns})
        if missing:
            raise ValueError(f"missing feature column(s): {', '.join(missing)}")
        decision = compare_saved_models(
            champion.estimator, challenger.estimator, frame[champion_features], y,
            metric=metric, champion_threshold=champion_threshold,
            challenger_threshold=challenger_threshold, positive_label=positive,
            n_bootstrap=n_bootstrap, alpha=alpha,
        ) if champion_features == challenger_features else _score_different_features(
            frame, y,
        )
        return decision

    def _score_different_features(frame, y) -> GateDecision:
        notes.append("The two models use different feature sets; each is scored on its own.")
        yy = np.asarray(y)
        c_pred = _predict_with_threshold(champion.estimator, frame[champion_features], champion_threshold)
        n_pred = _predict_with_threshold(challenger.estimator, frame[challenger_features], challenger_threshold)
        if positive is not None and metric in _POSITIVE_CLASS_METRICS:
            yy, c_pred, n_pred = ((np.asarray(a) == positive).astype(int) for a in (yy, c_pred, n_pred))
        return evaluate_gate(yy, c_pred, n_pred, metric=metric, n_bootstrap=n_bootstrap, alpha=alpha)

    windows: dict[str, GateDecision] = {}

    holdout = load_holdout(champion_dir)
    if holdout is None:
        notes.append(
            "The champion has no frozen holdout (an artifact from before holdouts were kept), "
            "so there is no common evaluation set that neither model trained on."
        )
    elif target not in holdout.columns:
        notes.append(f"The frozen holdout has no '{target}' column and cannot be scored.")
    else:
        try:
            windows[FROZEN_HOLDOUT] = score(holdout, holdout[target])
        except ValueError as e:
            notes.append(f"The frozen holdout could not be scored: {e}.")

    if store is not None:
        labelled = store.labelled_frame()
        if manifest is None:
            if not labelled.empty:
                notes.append(
                    "No retrain manifest was supplied, so it is unknown which logged predictions "
                    "the challenger trained on. The forward window is skipped rather than rigged."
                )
        elif not labelled.empty:
            trained_on = set(str(r) for r in (manifest.get("included_request_ids") or []))
            forward = labelled[~labelled["request_id"].astype(str).isin(trained_on)]
            fingerprints = set(manifest.get("training_row_fingerprints") or [])
            fingerprint_columns = list(manifest.get("fingerprint_columns") or [])
            if fingerprints and fingerprint_columns and not forward.empty:
                if all(c in forward.columns for c in fingerprint_columns):
                    from autoeng.lifecycle.retrain import row_fingerprints

                    repeats = np.array(
                        [f in fingerprints for f in row_fingerprints(forward, fingerprint_columns)],
                        dtype=bool,
                    )
                    if repeats.any():
                        notes.append(
                            f"{int(repeats.sum())} of {len(forward)} forward-window row(s) repeat a "
                            f"feature vector the challenger was trained on under a different request "
                            f"id, and were excluded. Scoring it on rows it was fitted to inflates its "
                            f"advantage: measured end to end, it more than doubled the apparent gap."
                        )
                        forward = forward.loc[~repeats]
            elif not fingerprints:
                notes.append(
                    "The retrain manifest carries no training-row fingerprints, so forward-window "
                    "rows repeating a training payload under a new request id could not be excluded."
                )
            if forward.empty:
                notes.append(
                    "Every labelled prediction was used to train the challenger, so there is no "
                    "forward window yet. Wait for more ground truth to arrive."
                )
            else:
                try:
                    windows[FORWARD_WINDOW] = score(forward, forward["actual"])
                except ValueError as e:
                    notes.append(f"The forward window could not be scored: {e}.")

    return combine_windows(windows, notes)
