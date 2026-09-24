"""
Data profiler.

Turns a raw DataFrame into a structured, machine-readable profile: what
kind of thing each column *is* (not what it's called), how it's
distributed, how columns relate to each other, and which columns look
like plausible prediction targets versus identifiers versus noise.

Everything here is derived from measurable properties of the data
(cardinality, dtype, parse success rates, distribution shape) — never
from column names or a lookup table of "known" dataset schemas. That's
what lets the same code run on a dataset it has never seen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class SemanticType(str, Enum):
    CONSTANT = "constant"
    BOOLEAN = "boolean"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"          # small-integer-valued, could be categorical
    CATEGORICAL_LOW_CARD = "categorical_low_card"
    CATEGORICAL_HIGH_CARD = "categorical_high_card"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"                      # near-unique, looks like an ID/key
    TEXT_FREE = "text_free"                        # free text: near-unique, or prose that repeats
    UNKNOWN = "unknown"


#: What separates prose from a category label. A repeated multi-word sentence is
#: still prose (see the routing measurement in _infer_semantic_type); "Mumbai" or
#: "premium tier" is a label however many rows it covers.
MIN_WORDS_FOR_PROSE = 5
MIN_CHARS_FOR_PROSE = 25


@dataclass
class ColumnProfile:
    name: str
    raw_dtype: str
    semantic_type: SemanticType
    n_missing: int
    missing_ratio: float
    n_unique: int
    unique_ratio: float
    is_constant: bool
    sample_values: list[Any]
    numeric_stats: dict[str, float] | None = None
    top_categories: list[tuple[Any, int]] | None = None
    datetime_range: tuple[str, str] | None = None
    avg_text_length: float | None = None
    reasoning: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["semantic_type"] = self.semantic_type.value
        return d


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: dict[str, ColumnProfile]
    duplicate_row_count: int
    duplicate_row_ratio: float
    numeric_correlations: pd.DataFrame | None
    highly_correlated_pairs: list[tuple[str, str, float]]
    target_candidates: list[dict[str, Any]]
    id_like_columns: list[str]
    datetime_columns: list[str]
    is_row_order_meaningful_hint: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": {k: v.as_dict() for k, v in self.columns.items()},
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_ratio": round(self.duplicate_row_ratio, 5),
            "highly_correlated_pairs": self.highly_correlated_pairs,
            "target_candidates": self.target_candidates,
            "id_like_columns": self.id_like_columns,
            "datetime_columns": self.datetime_columns,
            "is_row_order_meaningful_hint": self.is_row_order_meaningful_hint,
        }


def _try_parse_datetime(series: pd.Series, sample_size: int = 500) -> float:
    """Return the fraction of non-null values parseable as a datetime."""
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.sample(min(sample_size, len(non_null)), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _infer_semantic_type(series: pd.Series, n_rows: int) -> tuple[SemanticType, list[str]]:
    reasoning: list[str] = []
    non_null = series.dropna()
    n_unique = non_null.nunique()
    unique_ratio = n_unique / max(len(non_null), 1)

    if n_unique <= 1:
        reasoning.append("<=1 distinct value observed -> constant.")
        return SemanticType.CONSTANT, reasoning

    if pd.api.types.is_bool_dtype(series):
        reasoning.append("pandas dtype is bool.")
        return SemanticType.BOOLEAN, reasoning

    if pd.api.types.is_numeric_dtype(series):
        all_int_valued = bool(np.all(np.mod(non_null.dropna().astype(float), 1) == 0))

        # A near-unique, integer-valued column (order IDs, row numbers, primary
        # keys) is an identifier regardless of whether it happens to be numeric.
        # This is a cardinality/dtype signal, not a name lookup, so it also
        # catches numeric ID columns that a name-based check would miss.
        if all_int_valued and unique_ratio >= 0.98 and n_unique > 20:
            reasoning.append(
                f"Integer-valued, {unique_ratio:.0%} unique with {n_unique} distinct values "
                f"-> treated as an identifier, not a feature."
            )
            return SemanticType.IDENTIFIER, reasoning

        # Small integer-valued numeric columns are frequently encoded categoricals
        # (e.g. 0/1/2 class labels, star ratings). Distinguish by cardinality relative
        # to row count, not by name.
        low_cardinality = n_unique <= max(15, int(0.02 * n_rows))
        if all_int_valued and low_cardinality:
            reasoning.append(
                f"Integer-valued with only {n_unique} distinct values "
                f"(<= max(15, 2% of rows)) -> treated as discrete/categorical-like numeric."
            )
            return SemanticType.NUMERIC_DISCRETE, reasoning
        reasoning.append("Numeric dtype with high cardinality relative to row count -> continuous.")
        return SemanticType.NUMERIC_CONTINUOUS, reasoning

    # Object / string-like column from here on.
    datetime_success = _try_parse_datetime(non_null)
    if datetime_success >= 0.9:
        reasoning.append(f"{datetime_success:.0%} of sampled values parse as datetimes.")
        return SemanticType.DATETIME, reasoning

    if unique_ratio >= 0.95 and n_unique > 20:
        avg_len = non_null.astype(str).str.len().mean()
        avg_words = non_null.astype(str).str.split().map(len).mean()
        if avg_len > 25 or avg_words > 4:
            reasoning.append(
                f"Near-unique ({unique_ratio:.0%}) with avg length {avg_len:.0f} chars / "
                f"{avg_words:.1f} words -> free text."
            )
            return SemanticType.TEXT_FREE, reasoning
        reasoning.append(f"Near-unique ({unique_ratio:.0%}) short values -> identifier.")
        return SemanticType.IDENTIFIER, reasoning

    cardinality_cap = max(20, int(0.05 * n_rows))
    if n_unique <= cardinality_cap:
        reasoning.append(f"{n_unique} distinct values <= cap {cardinality_cap} -> low-cardinality categorical.")
        return SemanticType.CATEGORICAL_LOW_CARD, reasoning

    # Prose that repeats is still prose. Short free text — support tickets, product
    # titles, error messages — often falls well short of the 95%-unique bar above,
    # and as a high-cardinality categorical it is target-encoded from a handful of
    # rows per category. Measured on data/synthetic_text.csv (900 tickets, 55%
    # unique, so 499 categories): target encoding scored ROC-AUC 0.646 / 0.564 /
    # 0.578 for logistic regression / random forest / hist gradient boosting,
    # against 0.650 / 0.532 / 0.586 with the column DROPPED — it was worth
    # essentially nothing — while routing it to TF-IDF -> SVD gave 0.695 / 0.636 /
    # 0.642. Only ever diverts what would otherwise be high-cardinality: a
    # low-cardinality column of long answers is a category and stays one.
    avg_words = non_null.astype(str).str.split().map(len).mean()
    avg_len = non_null.astype(str).str.len().mean()
    if avg_words >= MIN_WORDS_FOR_PROSE and avg_len >= MIN_CHARS_FOR_PROSE:
        reasoning.append(
            f"{n_unique} distinct values > cap {cardinality_cap}, averaging {avg_words:.1f} words / "
            f"{avg_len:.0f} chars -> repetitive free text, not a category to encode."
        )
        return SemanticType.TEXT_FREE, reasoning

    reasoning.append(f"{n_unique} distinct values > cap {cardinality_cap} -> high-cardinality categorical.")
    return SemanticType.CATEGORICAL_HIGH_CARD, reasoning


def _profile_column(series: pd.Series, n_rows: int) -> ColumnProfile:
    n_missing = int(series.isna().sum())
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    semantic_type, reasoning = _infer_semantic_type(series, n_rows)

    numeric_stats = None
    top_categories = None
    datetime_range = None
    avg_text_length = None

    if semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE, SemanticType.BOOLEAN):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric) > 0:
            numeric_stats = {
                "mean": float(numeric.mean()),
                "std": float(numeric.std()) if len(numeric) > 1 else 0.0,
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "skew": float(numeric.skew()) if len(numeric) > 2 else 0.0,
                "zero_ratio": float((numeric == 0).mean()),
            }
        # Discrete/boolean numerics behave like categoricals for downstream
        # scoring (class balance, target-candidacy) too, so give them the
        # same category-count breakdown a categorical column would get.
        if semantic_type in (SemanticType.NUMERIC_DISCRETE, SemanticType.BOOLEAN):
            vc = non_null.value_counts().head(10)
            top_categories = [(str(k), int(v)) for k, v in vc.items()]
    elif semantic_type in (SemanticType.CATEGORICAL_LOW_CARD, SemanticType.CATEGORICAL_HIGH_CARD, SemanticType.IDENTIFIER):
        vc = non_null.value_counts().head(10)
        top_categories = [(str(k), int(v)) for k, v in vc.items()]
    elif semantic_type == SemanticType.DATETIME:
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
        if len(parsed) > 0:
            datetime_range = (str(parsed.min()), str(parsed.max()))
    elif semantic_type == SemanticType.TEXT_FREE:
        avg_text_length = float(non_null.astype(str).str.len().mean())

    sample_values = non_null.head(5).tolist() if len(non_null) else []

    return ColumnProfile(
        name=str(series.name),
        raw_dtype=str(series.dtype),
        semantic_type=semantic_type,
        n_missing=n_missing,
        missing_ratio=n_missing / max(n_rows, 1),
        n_unique=n_unique,
        unique_ratio=n_unique / max(len(non_null), 1),
        is_constant=(semantic_type == SemanticType.CONSTANT),
        sample_values=[_jsonable(v) for v in sample_values],
        numeric_stats=numeric_stats,
        top_categories=top_categories,
        datetime_range=datetime_range,
        avg_text_length=avg_text_length,
        reasoning=reasoning,
    )


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def _score_target_candidate(col: ColumnProfile, n_rows: int) -> float | None:
    """
    Heuristic 'how plausible is this column as a prediction target' score.
    Identifiers, free text, datetimes and constants are disqualified outright.
    Everything else is scored on variance / balance, not on its name.
    """
    if col.semantic_type in (SemanticType.IDENTIFIER, SemanticType.TEXT_FREE,
                              SemanticType.DATETIME, SemanticType.CONSTANT):
        return None
    if col.missing_ratio > 0.3:
        return None

    if col.semantic_type in (SemanticType.CATEGORICAL_LOW_CARD, SemanticType.NUMERIC_DISCRETE, SemanticType.BOOLEAN):
        # Reward balanced classes, penalize extreme imbalance or a class count that's
        # basically just "one row per value" (that's an ID, not a label).
        if not col.top_categories:
            return None
        counts = np.array([c for _, c in col.top_categories])
        if counts.sum() == 0:
            return None
        proportions = counts / counts.sum()
        entropy = -(proportions * np.log(proportions + 1e-12)).sum()
        max_entropy = math.log(len(proportions)) if len(proportions) > 1 else 1e-12
        balance = entropy / max_entropy if max_entropy > 0 else 0.0
        n_classes = col.n_unique
        class_count_score = 1.0 if 2 <= n_classes <= 20 else 0.3
        return 0.6 * balance + 0.4 * class_count_score

    if col.semantic_type == SemanticType.NUMERIC_CONTINUOUS:
        if not col.numeric_stats or col.numeric_stats["std"] == 0:
            return None
        # Continuous columns are plausible regression targets; mild preference
        # against extreme skew (often an ID-like or degenerate column).
        skew_penalty = min(abs(col.numeric_stats["skew"]) / 10.0, 0.5)
        return max(0.5 - skew_penalty, 0.1)

    return None


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    n_rows, n_cols = df.shape
    columns: dict[str, ColumnProfile] = {}
    for col_name in df.columns:
        columns[col_name] = _profile_column(df[col_name], n_rows)

    duplicate_row_count = int(df.duplicated().sum())

    numeric_cols = [c for c, p in columns.items()
                    if p.semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE, SemanticType.BOOLEAN)]
    numeric_correlations = None
    highly_correlated_pairs: list[tuple[str, str, float]] = []
    if len(numeric_cols) >= 2:
        try:
            numeric_correlations = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1:]:
                    val = numeric_correlations.loc[a, b]
                    if pd.notna(val) and abs(val) >= 0.95:
                        highly_correlated_pairs.append((a, b, float(val)))
        except Exception:
            numeric_correlations = None

    id_like_columns = [c for c, p in columns.items() if p.semantic_type == SemanticType.IDENTIFIER]
    datetime_columns = [c for c, p in columns.items() if p.semantic_type == SemanticType.DATETIME]

    target_candidates = []
    for name, prof in columns.items():
        score = _score_target_candidate(prof, n_rows)
        if score is not None:
            target_candidates.append({
                "column": name,
                "score": round(score, 4),
                "semantic_type": prof.semantic_type.value,
                "n_unique": prof.n_unique,
            })
    target_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Weak, non-authoritative signal only: a single well-formed datetime column
    # that looks monotonic (or nearly so) suggests the row order may itself be
    # meaningful (time series), which the problem-type detector will weigh
    # alongside everything else rather than trust blindly.
    is_row_order_meaningful_hint = False
    if len(datetime_columns) == 1:
        parsed = pd.to_datetime(df[datetime_columns[0]], errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9:
            diffs = parsed.dropna().diff().dropna()
            if len(diffs) > 0:
                is_row_order_meaningful_hint = bool((diffs.dt.total_seconds() >= 0).mean() > 0.95)

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_ratio=duplicate_row_count / max(n_rows, 1),
        numeric_correlations=numeric_correlations,
        highly_correlated_pairs=highly_correlated_pairs,
        target_candidates=target_candidates,
        id_like_columns=id_like_columns,
        datetime_columns=datetime_columns,
        is_row_order_meaningful_hint=is_row_order_meaningful_hint,
    )
