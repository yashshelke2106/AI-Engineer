"""
Group-key detection: does this dataset have repeated entities?

Several visits per patient, sessions per user, readings per device. When it
does, splitting by row puts one entity's rows on both sides of the split, and
the model can recognise the entity in the test fold rather than generalise to
it. Every metric inflates, and nothing else in the pipeline notices: the
duplicate-row check compares whole rows, and these rows genuinely differ.

**A group key is not a feature.** Once a column is identified as one it is
excluded from X, the same way an identifier is. Target-encoding a customer id
against a customer-level label is about the most direct leak available.

## What separates a group key from an ordinary categorical

Both repeat. The difference is shape, and it is the whole heuristic:

|  | group key | categorical feature |
|---|---|---|
| number of distinct values | many (hundreds) | few (a handful) |
| rows per value | few, and *consistent* | many, and arbitrary |

`region` with 4 values over 750 rows repeats 186 times each; `customer_id`
with 150 values repeats 5 times each. Grouping on `region` would hold out a
quarter of the feature space at a time and wreck the model, so the floor on
group count matters as much as the ceiling on cardinality.

Consistency is the sharpest of the signals. Real entity keys produce
near-uniform group sizes (every patient has roughly the same number of
visits); an incidental repeated value does not.

Like every other guess in this system (CLAUDE.md #6) this records its
reasoning and its runner-up candidates, and `--group-column` / `--no-groups`
override it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.profiling.profiler import DatasetProfile, SemanticType

# Below this many groups, grouped CV stops being meaningful — 5 folds over 20
# groups is already only 4 groups per fold — and the column is far more likely
# to be an ordinary categorical feature.
MIN_GROUPS = 20
# A column whose values barely repeat is an identifier, not a grouping.
MIN_MEAN_GROUP_SIZE = 1.8
# Above this, the column is essentially unique per row: an identifier.
MAX_UNIQUE_RATIO = 0.75
# Coefficient of variation of group sizes. Real entity keys repeat a consistent
# number of times; this is what separates them from incidental repetition.
MAX_SIZE_VARIATION = 0.60
# Below this the detection is reported but not acted on automatically.
CONFIDENCE_FLOOR = 0.55

# Names that suggest an entity key. A prior, never a rule — it cannot carry a
# column past the confidence floor alone, exactly as with the target name prior.
GROUP_NAME_HINTS = (
    "id", "key", "customer", "client", "user", "account", "patient", "subject",
    "session", "device", "sensor", "store", "shop", "school", "student",
    "employee", "household", "vehicle", "group", "cluster", "entity",
)

ELIGIBLE_TYPES = {
    SemanticType.CATEGORICAL_HIGH_CARD,
    SemanticType.CATEGORICAL_LOW_CARD,
    SemanticType.IDENTIFIER,
    SemanticType.NUMERIC_DISCRETE,
}


@dataclass
class GroupCandidate:
    column: str
    score: float
    n_groups: int
    mean_group_size: float
    size_variation: float
    unique_ratio: float
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GroupDecision:
    """
    Two column fields, deliberately.

    `column` is what splitting actually uses and is None whenever grouping is
    not applied. `detected_column` is what the heuristic found regardless — so
    that turning grouping off does not also turn off the leakage flag that says
    why the resulting scores are too good. Disabling a safety check should make
    the danger louder, not silent.
    """
    column: str | None
    confidence: float
    reasoning: list[str] = field(default_factory=list)
    candidates: list[GroupCandidate] = field(default_factory=list)
    source: str = "auto-detected"
    detected_column: str | None = None
    applied: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "detected_column": self.detected_column,
            "applied": self.applied,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "reasoning": self.reasoning,
            "candidates": [c.as_dict() for c in self.candidates],
        }


def _name_prior(column: str) -> float:
    lowered = str(column).lower()
    return 1.0 if any(hint in lowered for hint in GROUP_NAME_HINTS) else 0.0


def _score_candidate(series: pd.Series, name: str, n_rows: int) -> GroupCandidate | None:
    counts = series.value_counts(dropna=True)
    n_groups = int(len(counts))
    if n_groups < MIN_GROUPS:
        return None

    mean_size = float(counts.mean())
    if mean_size < MIN_MEAN_GROUP_SIZE:
        return None

    unique_ratio = n_groups / max(n_rows, 1)
    if unique_ratio > MAX_UNIQUE_RATIO:
        return None

    size_variation = float(counts.std() / mean_size) if mean_size > 0 and n_groups > 1 else 0.0
    if size_variation > MAX_SIZE_VARIATION:
        return None

    # Consistency is the strongest signal, so it carries the most weight.
    consistency = 1.0 - min(size_variation / MAX_SIZE_VARIATION, 1.0)
    # Repetition: more rows per entity means more opportunity to memorise it.
    repetition = min((mean_size - 1.0) / 4.0, 1.0)
    # Granularity: many small groups look like entities, few large ones like a
    # categorical feature.
    granularity = min(n_groups / (n_rows / 2.0), 1.0)

    score = 0.45 * consistency + 0.30 * repetition + 0.10 * granularity + 0.15 * _name_prior(name)

    return GroupCandidate(
        column=name, score=round(score, 4), n_groups=n_groups,
        mean_group_size=round(mean_size, 2), size_variation=round(size_variation, 4),
        unique_ratio=round(unique_ratio, 4),
        reasoning=(
            f"{n_groups} distinct values over {n_rows} rows ({mean_size:.1f} rows each, "
            f"size variation {size_variation:.2f}); "
            + ("name suggests an entity key" if _name_prior(name) else "name is not suggestive")
        ),
    )


def detect_group_column(
    df: pd.DataFrame,
    profile: DatasetProfile,
    roles: FeatureRoleAssignment,
    override: str | None = None,
    disabled: bool = False,
) -> GroupDecision:
    """
    Find the column whose repeated values represent one real-world entity.

    `override` pins a column (validated against the frame); `disabled` turns
    grouping off entirely. Both exist for the same reason `--target` does: this
    is a heuristic guess and the caller may simply know better.
    """
    if disabled:
        # Detection still runs. Grouping is not applied, but if an entity key is
        # there the leakage scan must still be able to say so — otherwise
        # --no-groups quietly returns a perfect-looking model with no warning.
        found = detect_group_column(df, profile, roles, override=None, disabled=False)
        note = (f"Group-aware splitting was disabled by the caller, but '{found.column}' looks "
                f"like a repeated-entity key. Rows sharing an entity will be split across train "
                f"and test, and the scores below will be inflated as a result.")
        return GroupDecision(
            column=None, confidence=0.0, source="disabled by caller",
            detected_column=found.column, applied=False,
            candidates=found.candidates,
            reasoning=([note] if found.column else
                       ["Group-aware splitting was explicitly disabled; no entity key was "
                        "detected either, so this changes nothing."]),
        )

    if override is not None:
        if override not in df.columns:
            return GroupDecision(
                column=None, confidence=0.0, source="supplied by caller (invalid)",
                detected_column=None, applied=False,
                reasoning=[f"Requested group column '{override}' is not in the dataset; "
                           f"falling back to ungrouped splitting."],
            )
        n_groups = int(df[override].nunique())
        return GroupDecision(
            column=override, confidence=1.0, source="supplied by caller",
            detected_column=override, applied=True,
            reasoning=[f"Group column '{override}' supplied by the caller "
                       f"({n_groups} groups over {len(df)} rows)."],
        )

    n_rows = len(df)
    candidates: list[GroupCandidate] = []
    for name, col in profile.columns.items():
        # The target is never a grouping, and the time axis is handled by
        # chronological splitting instead.
        if name in (roles.target_column, roles.time_column):
            continue
        if col.semantic_type not in ELIGIBLE_TYPES:
            continue
        scored = _score_candidate(df[name], name, n_rows)
        if scored is not None:
            candidates.append(scored)

    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        return GroupDecision(
            column=None, confidence=0.0, detected_column=None, applied=False,
            reasoning=["No column looks like a repeated-entity key: nothing had enough "
                       f"distinct values (>= {MIN_GROUPS}) repeating consistently enough "
                       f"(size variation <= {MAX_SIZE_VARIATION})."],
            candidates=[],
        )

    best = candidates[0]
    if best.score < CONFIDENCE_FLOOR:
        return GroupDecision(
            column=None, confidence=best.score, candidates=candidates[:3],
            detected_column=None, applied=False,
            reasoning=[
                f"Best group candidate '{best.column}' scored {best.score:.2f}, below the "
                f"{CONFIDENCE_FLOOR} floor — treating the rows as independent. {best.reasoning}",
                "Pass --group-column to force it if these rows do share an entity.",
            ],
        )

    return GroupDecision(
        column=best.column, confidence=best.score, candidates=candidates[:3],
        detected_column=best.column, applied=True,
        reasoning=[
            f"Grouping on '{best.column}': {best.reasoning}.",
            "Splits are made by entity, not by row, so no entity appears in both training "
            "and evaluation. This column is excluded from the feature set — a group key is "
            "an identifier, and encoding it against an entity-level label leaks directly.",
        ],
    )


def group_values(df: pd.DataFrame, decision: GroupDecision) -> np.ndarray | None:
    """The `groups` array scikit-learn's group-aware splitters expect."""
    if decision.column is None or decision.column not in df.columns:
        return None
    return df[decision.column].to_numpy()
