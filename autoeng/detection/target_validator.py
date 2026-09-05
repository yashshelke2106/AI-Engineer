"""
Fit-and-check target validation.

Association-based ranking has a specific, reproducible failure: it cannot
tell an *outcome* from a *sibling measurement*. On sklearn's diabetes
dataset it picked `s1` (a blood serum reading, ~0.9 correlated with `s2`,
another blood serum reading) over the real `target` column (disease
progression). Both look "explainable from the rest of the data" to a
correlation metric — because correlation is symmetric and says nothing
about *how many* things explain the column.

The discriminating signal is the shape of the explanation, not its
strength:

  - A sibling measurement is explained by essentially ONE other column.
    Fit a model to predict `s1` and a single feature (`s2`) carries almost
    all the importance. High score, degenerate structure.
  - A real outcome is explained by SEVERAL features jointly and usually
    imperfectly. Fit a model to predict disease progression and importance
    spreads across bmi, bp, s5... Lower score, real structure.

So each candidate gets a cheap model actually fit against it, and is
scored on achievable performance DISCOUNTED by how concentrated the
importance is. This is the same "one feature carrying the whole model"
signal the leakage detector uses to spot a leaked column — the same
pathology, read for a different purpose.

Deliberately cheap: subsampled rows, few trees, 3-fold. This is a screen
run over ~6 candidates before the real work starts, not a model anyone
ships.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from autoeng.profiling.profiler import DatasetProfile, SemanticType

# Screening budget — enough signal to rank candidates, cheap enough to run
# on every candidate before the real search begins.
MAX_SCREENING_ROWS = 5000
SCREENING_FOLDS = 3
SCREENING_TREES = 60
MIN_ROWS_FOR_SCREENING = 40

# Importance concentration below this is "normal" (several features share the
# work) and carries no penalty. Above it, the penalty ramps linearly to a full
# discount at 1.0, where a single feature explains the entire column.
CONCENTRATION_FREE_ALLOWANCE = 0.5

CLASSIFICATION_LIKE = {
    SemanticType.CATEGORICAL_LOW_CARD, SemanticType.CATEGORICAL_HIGH_CARD,
    SemanticType.NUMERIC_DISCRETE, SemanticType.BOOLEAN,
}


@dataclass
class TargetValidation:
    column: str
    kind: str                       # "classification" | "regression"
    raw_score: float                # native metric (roc_auc / accuracy / r2)
    predictability: float           # normalized to [0, 1], 0 = no better than baseline
    top_feature: str | None
    top_feature_share: float        # fraction of total importance held by one feature
    concentration_penalty: float
    score: float                    # predictability discounted by concentration
    looks_like_sibling_column: bool
    reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Minimal, fast encoding — this is a screening fit, not the real pipeline."""
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)
        elif pd.api.types.is_datetime64_any_dtype(series):
            out[col] = (series - pd.Timestamp("1970-01-01")).dt.days.astype("float").fillna(0.0)
        else:
            codes = series.astype("category").cat.codes
            out[col] = codes.astype("float")
    return out


def _make_screening_models(kind: str):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        # importance_type="gain", NOT the default "split". Split counts how often
        # a feature is used, which stays diffuse even when one feature explains
        # nearly everything (trees keep splitting on correlated columns) — that
        # would make the concentration signal below useless. Gain measures actual
        # loss reduction, which is what "one column explains this" looks like.
        kwargs = dict(n_estimators=SCREENING_TREES, verbosity=-1, random_state=42,
                      n_jobs=1, importance_type="gain")
        return LGBMClassifier(**kwargs) if kind == "classification" else LGBMRegressor(**kwargs)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        # sklearn's feature_importances_ is already impurity-decrease (gain-like).
        if kind == "classification":
            return RandomForestClassifier(n_estimators=SCREENING_TREES, random_state=42, n_jobs=1)
        return RandomForestRegressor(n_estimators=SCREENING_TREES, random_state=42, n_jobs=1)


def _normalize_score(kind: str, raw: float, y: pd.Series) -> float:
    """Map a native metric onto [0, 1] where 0 means 'no better than the trivial baseline'."""
    if kind == "regression":
        return float(np.clip(raw, 0.0, 1.0))
    n_classes = y.nunique()
    if n_classes == 2:
        # raw is ROC-AUC: 0.5 is chance.
        return float(np.clip((raw - 0.5) * 2.0, 0.0, 1.0))
    # raw is accuracy: the majority-class rate is the trivial baseline, which
    # matters a lot on imbalanced multiclass columns.
    baseline = float(y.value_counts(normalize=True).max())
    if baseline >= 0.999:
        return 0.0
    return float(np.clip((raw - baseline) / (1.0 - baseline), 0.0, 1.0))


def validate_target_candidate(
    df: pd.DataFrame,
    profile: DatasetProfile,
    candidate_column: str,
    exclude_columns: set[str] | None = None,
) -> TargetValidation | None:
    """
    Returns None when screening isn't possible (too few rows, no usable
    features, model failure) — callers should fall back to the
    association-based ranking rather than treating None as a zero score.
    """
    exclude_columns = (exclude_columns or set()) | {candidate_column}
    feature_cols = [
        c for c in df.columns
        if c not in exclude_columns
        and profile.columns[c].semantic_type not in (
            SemanticType.IDENTIFIER, SemanticType.TEXT_FREE, SemanticType.CONSTANT,
        )
    ]
    if not feature_cols:
        return None

    frame = df[feature_cols + [candidate_column]].dropna(subset=[candidate_column])
    if len(frame) < MIN_ROWS_FOR_SCREENING:
        return None
    if len(frame) > MAX_SCREENING_ROWS:
        frame = frame.sample(MAX_SCREENING_ROWS, random_state=42)

    y = frame[candidate_column]
    X = _encode_features(frame[feature_cols])

    semantic = profile.columns[candidate_column].semantic_type
    kind = "classification" if semantic in CLASSIFICATION_LIKE else "regression"

    if kind == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        # Every class needs enough members for a 3-fold stratified split.
        if counts.min() < SCREENING_FOLDS or y.nunique() < 2:
            return None
        scoring = "roc_auc" if y.nunique() == 2 else "accuracy"
        if y.nunique() == 2:
            y = (y == counts.index[0]).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().all() or y.std() == 0:
            return None
        y = y.fillna(y.median())
        scoring = "r2"

    model = _make_screening_models(kind)
    # Shuffled folds are essential here: real datasets are very often sorted by
    # class or group (iris is sorted by species), and unshuffled K-fold would
    # then train on two species and test on a third — scoring every candidate
    # at ~0 and destroying the ranking for reasons that have nothing to do with
    # whether the column is a good target.
    cv = (StratifiedKFold(n_splits=SCREENING_FOLDS, shuffle=True, random_state=42)
          if kind == "classification"
          else KFold(n_splits=SCREENING_FOLDS, shuffle=True, random_state=42))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            raw_score = float(np.nanmean(scores))
            model.fit(X, y)
    except Exception:
        return None

    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != X.shape[1] or float(np.sum(importances)) == 0:
        top_feature, top_share = None, 0.0
    else:
        shares = np.asarray(importances, dtype=float) / float(np.sum(importances))
        top_idx = int(np.argmax(shares))
        top_feature, top_share = X.columns[top_idx], float(shares[top_idx])

    predictability = _normalize_score(kind, raw_score, frame[candidate_column])
    penalty = float(np.clip(
        (top_share - CONCENTRATION_FREE_ALLOWANCE) / (1.0 - CONCENTRATION_FREE_ALLOWANCE), 0.0, 1.0,
    ))
    score = predictability * (1.0 - penalty)
    looks_like_sibling = penalty > 0.5 and predictability > 0.5

    reasoning = [
        f"Screening fit predicting '{candidate_column}' from {len(feature_cols)} other column(s): "
        f"{scoring}={raw_score:.3f} (normalized predictability {predictability:.3f}).",
    ]
    if top_feature is not None:
        reasoning.append(
            f"Importance concentration: '{top_feature}' holds {top_share:.0%} of it "
            f"(penalty {penalty:.2f})."
        )
    if looks_like_sibling:
        reasoning.append(
            f"Highly predictable but almost entirely from one column ('{top_feature}') — this looks "
            "like a sibling measurement of that column rather than an outcome to predict."
        )

    return TargetValidation(
        column=candidate_column, kind=kind, raw_score=raw_score, predictability=predictability,
        top_feature=top_feature, top_feature_share=top_share, concentration_penalty=penalty,
        score=score, looks_like_sibling_column=looks_like_sibling, reasoning=reasoning,
    )
