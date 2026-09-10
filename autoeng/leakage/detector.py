"""
Data leakage detection.

Two different failure modes, both covered:

1. Leakage the *architecture* can't prevent by construction — a raw column
   that directly or near-directly encodes the target (e.g. a "risk_score"
   column computed from the same process that produced the label). No
   amount of fitting transformers only-on-train saves you from this one,
   because the leak is in the feature's *content*, not in how it's fit.
   `scan_pre_training` catches this using the same association machinery
   the problem detector uses to find targets in the first place — the
   irony is deliberate: a feature that's "too good" a target-predictor by
   the same measure that makes columns look target-like is itself the
   red flag.

2. Leakage from *procedure* — train/test rows that overlap, or a temporal
   problem evaluated with a split that lets the model see the future.
   `check_train_test_row_overlap` and `check_temporal_split` catch these.

3. A defense-in-depth net for anything upstream missed: a model that
   performs implausibly well, cross-referenced against which single
   feature is doing the work. `scan_post_training` catches this.

None of this is perfect — leakage detection fundamentally can't be, since
"suspiciously predictive" and "actually genuinely predictive" look
identical from the data alone in the worst cases. Every flag here comes
with the evidence that triggered it so a human (or the conversational
explainer) can make the final call rather than the pipeline silently
deciding for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autoeng.common.associations import association
from autoeng.profiling.profiler import DatasetProfile

# A feature this associated with the target is treated as a likely leak
# rather than a strong predictor — real-world features are almost never
# this cleanly related to an outcome unless they're derived from it.
LEAKAGE_ASSOCIATION_THRESHOLD = 0.97

# Post-training performance this good, on a problem that isn't trivial by
# construction, is flagged for a leakage audit rather than celebrated.
SUSPICIOUS_METRIC_THRESHOLDS = {
    "roc_auc": 0.995,
    "accuracy": 0.995,
    "f1": 0.995,
    "r2": 0.999,
}
DOMINANT_FEATURE_IMPORTANCE_SHARE = 0.85


@dataclass
class LeakageFlag:
    severity: str  # "critical" | "warning" | "info"
    kind: str
    columns: list[str]
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class LeakageReport:
    flags: list[LeakageFlag] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"flags": [f.as_dict() for f in self.flags]}

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.flags)


def scan_pre_training(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: str,
    feature_columns: list[str],
) -> LeakageReport:
    flags: list[LeakageFlag] = []

    for col in feature_columns:
        if col == target_column or col not in profile.columns:
            continue
        score = association(df, col, target_column, profile)
        if score >= LEAKAGE_ASSOCIATION_THRESHOLD:
            flags.append(LeakageFlag(
                severity="critical",
                kind="target_derived_feature",
                columns=[col],
                description=(
                    f"'{col}' is associated with the target at {score:.4f} (>= "
                    f"{LEAKAGE_ASSOCIATION_THRESHOLD}) — almost certainly derived from or a proxy for "
                    "the target itself rather than a genuine independent predictor. Recommend dropping "
                    "it before training, or manually confirming it is legitimately available at "
                    "prediction time."
                ),
                evidence={"association": score},
            ))

    # Near-duplicate feature pairs aren't leakage in the strict sense, but a
    # pair correlated at ~1.0 usually means one column is a deterministic
    # transform of the other (e.g. total = a + b, with a, b, and total all
    # present) — worth surfacing so it doesn't masquerade as two independent
    # signals or quietly destabilize a linear model.
    for a, b, corr in profile.highly_correlated_pairs:
        if a == target_column or b == target_column:
            continue
        flags.append(LeakageFlag(
            severity="info",
            kind="redundant_feature_pair",
            columns=[a, b],
            description=f"'{a}' and '{b}' are correlated at {corr:.3f} — likely redundant/derived from each other.",
            evidence={"correlation": corr},
        ))

    return LeakageReport(flags=flags)


def check_train_test_row_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> LeakageReport:
    flags: list[LeakageFlag] = []
    try:
        combined = pd.concat([train_df, test_df])
        dup_mask = combined.duplicated(keep=False)
        n_overlap = int(dup_mask.sum())
    except Exception:
        n_overlap = 0
    if n_overlap > 0:
        flags.append(LeakageFlag(
            severity="critical",
            kind="train_test_row_overlap",
            columns=[],
            description=(
                f"{n_overlap} row(s) appear in both the train and test split (exact match across all "
                "columns). The model would be evaluated partly on data it trained on."
            ),
            evidence={"n_overlapping_rows": n_overlap},
        ))
    return LeakageReport(flags=flags)


def check_group_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str | None) -> LeakageReport:
    """
    Flag entities whose rows appear on both sides of a split.

    This is the leak `check_train_test_row_overlap` structurally cannot see:
    the rows differ, so nothing is duplicated, but they describe the same
    patient / customer / device. The model recognises the entity in the test
    fold rather than generalising to it, and every metric inflates.
    """
    flags: list[LeakageFlag] = []
    if not group_column or group_column not in train_df.columns or group_column not in test_df.columns:
        return LeakageReport(flags=flags)

    train_groups = set(train_df[group_column].dropna().unique())
    test_groups = set(test_df[group_column].dropna().unique())
    shared = train_groups & test_groups
    if shared:
        n_rows_affected = int(test_df[group_column].isin(shared).sum())
        flags.append(LeakageFlag(
            severity="critical",
            kind="group_overlap",
            columns=[group_column],
            description=(
                f"{len(shared)} of {len(test_groups)} '{group_column}' value(s) appear in both the "
                f"train and test split, covering {n_rows_affected} test row(s). The rows are not "
                "duplicates, so the row-overlap check cannot see this — but they describe the same "
                "entity, and the model can recognise it rather than generalise to it."
            ),
            evidence={
                "group_column": group_column,
                "n_shared_groups": len(shared),
                "n_test_groups": len(test_groups),
                "n_affected_test_rows": n_rows_affected,
                "examples": [str(g) for g in list(shared)[:5]],
            },
        ))
    return LeakageReport(flags=flags)


def check_temporal_split(train_df: pd.DataFrame, test_df: pd.DataFrame, time_column: str) -> LeakageReport:
    flags: list[LeakageFlag] = []
    try:
        train_times = pd.to_datetime(train_df[time_column], errors="coerce", format="mixed").dropna()
        test_times = pd.to_datetime(test_df[time_column], errors="coerce", format="mixed").dropna()
    except Exception:
        return LeakageReport(flags=flags)
    if train_times.empty or test_times.empty:
        return LeakageReport(flags=flags)

    train_max, test_min = train_times.max(), test_times.min()
    if train_max > test_min:
        n_violations = int((train_times > test_min).sum())
        flags.append(LeakageFlag(
            severity="critical",
            kind="temporal_leakage",
            columns=[time_column],
            description=(
                f"Training data extends to {train_max} but test data starts at {test_min} — the "
                f"training set contains {n_violations} row(s) from AFTER the test period begins. "
                "For a time-series problem, splits must respect chronological order; a random split "
                "lets the model see the future."
            ),
            evidence={"train_max_time": str(train_max), "test_min_time": str(test_min), "n_violations": n_violations},
        ))
    return LeakageReport(flags=flags)


def scan_post_training(
    metric_name: str,
    metric_value: float,
    feature_importances: dict[str, float] | None = None,
) -> LeakageReport:
    flags: list[LeakageFlag] = []
    threshold = SUSPICIOUS_METRIC_THRESHOLDS.get(metric_name.lower())
    if threshold is not None and metric_value >= threshold:
        evidence: dict[str, Any] = {"metric": metric_name, "value": metric_value, "threshold": threshold}
        dominant_note = ""
        if feature_importances:
            total = sum(abs(v) for v in feature_importances.values()) or 1.0
            top_feature, top_value = max(feature_importances.items(), key=lambda kv: abs(kv[1]))
            share = abs(top_value) / total
            evidence["dominant_feature"] = top_feature
            evidence["dominant_feature_share"] = share
            if share >= DOMINANT_FEATURE_IMPORTANCE_SHARE:
                dominant_note = (
                    f" A single feature ('{top_feature}') accounts for {share:.0%} of total feature "
                    "importance, which is the classic signature of one leaked column carrying the model."
                )
        flags.append(LeakageFlag(
            severity="warning",
            kind="suspiciously_perfect_performance",
            columns=[evidence.get("dominant_feature")] if evidence.get("dominant_feature") else [],
            description=(
                f"{metric_name} = {metric_value:.4f}, at or above the {threshold} suspicion threshold for "
                f"this metric.{dominant_note} Recommend a manual leakage audit before trusting this model."
            ),
            evidence=evidence,
        ))
    return LeakageReport(flags=flags)


def merge_reports(*reports: LeakageReport) -> LeakageReport:
    flags: list[LeakageFlag] = []
    for r in reports:
        flags.extend(r.flags)
    return LeakageReport(flags=flags)
