"""
Problem-type detection.

Given a DatasetProfile (facts only — no interpretation), decide what kind of
ML problem this dataset most plausibly represents: binary/multiclass
classification, regression, time-series forecasting, or clustering
(unsupervised, when no column looks like a usable label).

This is fundamentally a best-guess under uncertainty — there is no way to
*know* a dataset's intended target without being told. So this module never
returns a single silent answer: it returns a ranked set of hypotheses with
a reasoning trace and a confidence score, and the pipeline proceeds with the
top hypothesis while logging the runner-up(s) so a human (or the
conversational interface later) can see what else was considered and why
it was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from autoeng.common.associations import aggregate_explainability
from autoeng.detection.name_prior import name_prior
from autoeng.detection.target_validator import validate_target_candidate
from autoeng.profiling.profiler import DatasetProfile, SemanticType

# Below this target-candidate score, we don't trust *any* column enough to
# call this a supervised problem at all.
MIN_TARGET_CONFIDENCE = 0.25
# Autocorrelation (lag-1, on the candidate target sorted by the detected time
# column) above this threshold is treated as real temporal structure rather
# than noise.
TIME_SERIES_AUTOCORR_THRESHOLD = 0.3
# A supervised hypothesis (classification/regression/time-series) below this
# combined score is treated as too weak to trust; clustering wins instead.
# Calibrated empirically against the nine validation datasets, not guessed —
# see tests/test_detection_calibration.py.
SUPERVISED_CONFIDENCE_FLOOR = 0.25

# Weights for the three target-detection signals. Fit-and-check dominates
# because it's the only one that actually tests the hypothesis; the name prior
# is real but must never be able to carry a bad column on its own (0.30 alone
# cannot clear the floor without support from at least one other signal).
SHAPE_WEIGHT = 0.15
FIT_WEIGHT = 0.55
NAME_WEIGHT = 0.30

# How many candidates get the expensive fit-and-check screening. Each one costs
# a small cross-validated model fit, so this is a compute/coverage trade-off.
CANDIDATE_POOL_SIZE = 8


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    CLUSTERING = "clustering"


@dataclass
class Hypothesis:
    problem_type: ProblemType
    target_column: str | None
    time_column: str | None
    score: float
    reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["problem_type"] = self.problem_type.value
        return d


@dataclass
class ProblemTypeDecision:
    chosen: Hypothesis
    alternatives: list[Hypothesis]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen.as_dict(),
            "alternatives": [a.as_dict() for a in self.alternatives],
            "confidence": round(self.confidence, 4),
        }


def _lag1_autocorrelation(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 5:
        return 0.0
    a, b = v.iloc[:-1].to_numpy(), v.iloc[1:].to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def detect_problem_type(df: pd.DataFrame, profile: DatasetProfile) -> ProblemTypeDecision:
    hypotheses: list[Hypothesis] = []

    candidates = [c for c in profile.target_candidates if c["score"] >= MIN_TARGET_CONFIDENCE]

    if not candidates:
        hypotheses.append(Hypothesis(
            problem_type=ProblemType.CLUSTERING,
            target_column=None,
            time_column=None,
            score=0.6,
            reasoning=[
                "No column scored above the minimum target-candidate threshold "
                f"({MIN_TARGET_CONFIDENCE}); no plausible label/outcome column found.",
                "Falling back to unsupervised structure discovery (clustering).",
            ],
        ))
    else:
        # Three independent signals, because no one of them is sufficient:
        #
        #   1. SHAPE (from the profiler) — is this column even label-shaped?
        #      Balanced classes, real variance, not an ID. Cheap, but a
        #      perfectly balanced noise column scores just as well as a label.
        #   2. FIT-AND-CHECK — actually fit a cheap model against the candidate
        #      and measure both how predictable it is AND how concentrated the
        #      importance is. Catches sibling/derived columns (a "target" that
        #      is really just a restatement of one neighbouring column).
        #   3. NAME PRIOR — a weak but real signal that a purely statistical
        #      approach throws away for no good reason. See name_prior.py for
        #      why this was added after the diabetes dataset proved that
        #      `target` and `s5` are statistically indistinguishable.
        #
        # Association-based explainability is still computed and reported, but
        # it is no longer the primary ranker: correlation is symmetric, so it
        # can never tell a target from its own best predictor.
        exclude_cols = set(profile.id_like_columns)

        # Which candidates earn the (expensive) fit-and-check screening is decided
        # by the CHEAP signals first — shape plus name prior. This matters: on
        # datasets like diabetes, eleven continuous columns all score ~0.49 on
        # shape alone, so a plain top-N cut is effectively arbitrary and can drop
        # the real target before it is ever evaluated. Letting the name prior into
        # the prefilter guarantees a column called `target` gets looked at; it
        # still has to win on the full three-signal score afterwards.
        candidates = sorted(
            candidates,
            key=lambda c: 0.5 * c["score"] + name_prior(c["column"])[0],
            reverse=True,
        )
        for cand in candidates[:CANDIDATE_POOL_SIZE]:
            col_name = cand["column"]
            col_profile = profile.columns[col_name]
            base_score = cand["score"]
            explainability, best_partner = aggregate_explainability(
                df, col_name, profile, exclude=exclude_cols | {col_name}
            )
            prior, prior_reason = name_prior(col_name)
            validation = validate_target_candidate(df, profile, col_name, exclude_columns=exclude_cols)

            reasoning = [
                f"Evaluating '{col_name}' as target (shape-based candidate score {base_score:.3f}).",
                f"Joint explainability from other columns (mean of top-2 associations): {explainability:.3f}"
                + (f", strongest single one with '{best_partner}')" if best_partner else " (no related column found)."),
                f"Name prior: {prior:.2f} — {prior_reason}.",
            ]
            if validation is not None:
                reasoning.extend(validation.reasoning)
                fit_score = validation.score
            else:
                # Screening unavailable (too few rows, no usable features) — fall
                # back to the association signal rather than scoring this zero.
                fit_score = explainability
                reasoning.append("Fit-and-check screening unavailable; falling back to association strength.")

            is_time_series = False
            autocorr = 0.0
            if profile.datetime_columns and col_profile.semantic_type in (
                SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE,
            ):
                time_col = profile.datetime_columns[0]
                try:
                    order = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
                    sorted_target = df.loc[order.sort_values().index, col_name]
                    autocorr = _lag1_autocorrelation(sorted_target)
                except Exception:
                    autocorr = 0.0
                reasoning.append(
                    f"Datetime column '{time_col}' present; lag-1 autocorrelation of "
                    f"'{col_name}' when sorted by time = {autocorr:.3f}."
                )
                if autocorr >= TIME_SERIES_AUTOCORR_THRESHOLD and profile.is_row_order_meaningful_hint:
                    is_time_series = True
                    reasoning.append(
                        f"Autocorrelation >= {TIME_SERIES_AUTOCORR_THRESHOLD} and row order looks "
                        "temporally monotonic -> treating as time-series forecasting, not i.i.d. regression."
                    )

            if is_time_series:
                # For a time series the target's own history is the signal that
                # matters; a strong autocorrelation outweighs everything else.
                signal = max(fit_score, abs(autocorr))
                combined_score = 0.15 * base_score + 0.70 * signal + 0.15 * prior
                hypotheses.append(Hypothesis(
                    problem_type=ProblemType.TIME_SERIES_FORECASTING,
                    target_column=col_name,
                    time_column=profile.datetime_columns[0],
                    score=combined_score,
                    reasoning=reasoning,
                ))
                continue

            combined_score = SHAPE_WEIGHT * base_score + FIT_WEIGHT * fit_score + NAME_WEIGHT * prior

            if col_profile.semantic_type == SemanticType.NUMERIC_CONTINUOUS:
                reasoning.append("Target is continuous with high cardinality -> regression.")
                hypotheses.append(Hypothesis(
                    problem_type=ProblemType.REGRESSION,
                    target_column=col_name,
                    time_column=None,
                    score=combined_score,
                    reasoning=reasoning,
                ))
            else:
                n_classes = col_profile.n_unique
                ptype = ProblemType.BINARY_CLASSIFICATION if n_classes == 2 else ProblemType.MULTICLASS_CLASSIFICATION
                reasoning.append(f"Target has {n_classes} discrete classes -> {ptype.value}.")
                hypotheses.append(Hypothesis(
                    problem_type=ptype,
                    target_column=col_name,
                    time_column=None,
                    score=combined_score,
                    reasoning=reasoning,
                ))

        # Clustering is always logged as the "null hypothesis" alternative so the
        # decision record shows what supervised learning was chosen *over*. If
        # nothing cleared a real confidence floor, don't force a supervised
        # answer just because it narrowly beat the others — bump clustering
        # above the pack instead. Weak-to-moderate correlation between plain
        # numeric feature columns (common in unlabeled multivariate data) is
        # exactly the pattern that would otherwise get misread as "regression."
        best_supervised = max((h.score for h in hypotheses), default=0.0)
        clustering_score = 0.2
        clustering_reasoning = ["Always considered as the no-usable-label baseline."]
        if best_supervised < SUPERVISED_CONFIDENCE_FLOOR:
            clustering_score = SUPERVISED_CONFIDENCE_FLOOR + 0.05
            clustering_reasoning.append(
                f"No supervised hypothesis cleared the confidence floor ({SUPERVISED_CONFIDENCE_FLOOR}); "
                f"best was {best_supervised:.3f}. Preferring unsupervised structure discovery instead of "
                "forcing a weak supervised answer."
            )
        hypotheses.append(Hypothesis(
            problem_type=ProblemType.CLUSTERING,
            target_column=None,
            time_column=None,
            score=clustering_score,
            reasoning=clustering_reasoning,
        ))

    hypotheses.sort(key=lambda h: h.score, reverse=True)
    chosen, alternatives = hypotheses[0], hypotheses[1:]

    # Confidence reflects both the winning score and the margin over the runner-up —
    # a dataset with two near-tied plausible targets should report LOW confidence
    # even if both scores are individually high.
    margin = (chosen.score - alternatives[0].score) if alternatives else chosen.score
    confidence = max(0.0, min(1.0, 0.5 * chosen.score + 0.5 * min(margin * 2, 1.0)))

    return ProblemTypeDecision(chosen=chosen, alternatives=alternatives, confidence=confidence)


def decision_from_override(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: str | None,
    problem_type: str | None = None,
) -> ProblemTypeDecision:
    """
    Build a decision from a caller-supplied target and/or problem type.

    Auto-detection is a best guess at human intent, and the honest response to
    "the guess is wrong" is to let a human say so — not to keep tuning
    heuristics against one dataset. When the target is supplied, the problem
    type is still inferred from that column's measured shape unless it too was
    given explicitly, so `--target` alone is enough in the common case.
    """
    reasoning: list[str] = []

    if target_column is not None and target_column not in profile.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {sorted(profile.columns)}"
        )

    if problem_type is not None:
        try:
            ptype = ProblemType(problem_type)
        except ValueError as exc:
            raise ValueError(
                f"Unknown problem type '{problem_type}'. Valid values: {[p.value for p in ProblemType]}"
            ) from exc
        reasoning.append(f"Problem type supplied by the caller: {ptype.value} (auto-detection skipped).")
    elif target_column is None:
        ptype = ProblemType.CLUSTERING
        reasoning.append("No target supplied and no problem type given -> unsupervised clustering.")
    else:
        col = profile.columns[target_column]
        if col.semantic_type == SemanticType.NUMERIC_CONTINUOUS:
            ptype = ProblemType.REGRESSION
            reasoning.append(f"Target '{target_column}' supplied; it is continuous -> regression.")
        else:
            ptype = (ProblemType.BINARY_CLASSIFICATION if col.n_unique == 2
                     else ProblemType.MULTICLASS_CLASSIFICATION)
            reasoning.append(
                f"Target '{target_column}' supplied; it has {col.n_unique} discrete values -> {ptype.value}."
            )

    time_column = None
    if ptype == ProblemType.TIME_SERIES_FORECASTING:
        if not profile.datetime_columns:
            raise ValueError("Time-series forecasting requested but no datetime column was detected.")
        time_column = profile.datetime_columns[0]
        reasoning.append(f"Using '{time_column}' as the time axis.")

    chosen = Hypothesis(
        problem_type=ptype,
        target_column=target_column if ptype != ProblemType.CLUSTERING else None,
        time_column=time_column,
        score=1.0,
        reasoning=reasoning,
    )
    return ProblemTypeDecision(chosen=chosen, alternatives=[], confidence=1.0)
