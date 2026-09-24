"""
Feature-generation transformers.

All stateless-where-possible, fit/transform where not, and all designed
to be dropped into an sklearn Pipeline so cross-validation fits them fresh
per fold — the same leakage-safety argument as the cleaning transformer.

- DatetimeFeaturizer: decomposes a real datetime column into calendar parts
  plus days-since-a-fixed-epoch. Stateless (the epoch is a constant, not a
  fit statistic), so leakage isn't even a risk here.
- TextStatsFeaturizer: cheap free-text signal (length, word count) without
  needing an embedding model. Also stateless.
- TextVectorFeaturizer: what the words actually say — TF-IDF reduced by
  TruncatedSVD (latent semantic analysis). This one is stateful (a vocabulary,
  document frequencies and a projection), so it is exactly the kind of thing
  invariant 1 exists for: as a Pipeline step the vocabulary is learned from the
  training fold only, and a word that appears solely in the held-out fold is
  ignored rather than given a column of its own.
- NumericInteractionFeaturizer: this one DOES need y, and DOES need to be
  fit-only-on-train — it searches pairwise products/ratios of the most
  target-relevant numeric columns and keeps only the ones that carry
  measurable extra signal (by mutual information), rather than exploding
  every pair combinatorially and hoping the model sorts it out.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

_EPOCH = pd.Timestamp("1970-01-01")


class DatetimeFeaturizer(BaseEstimator, TransformerMixin):
    # NOTE: __init__ stores constructor args verbatim (sklearn convention) —
    # no `x or []`-style default substitution here, because that silently
    # replaces an empty container with a *new* object each time __init__
    # runs, which breaks sklearn.base.clone()'s identity check on refit
    # (clone() recreates the estimator from get_params() and requires the
    # reconstructed parameter to be the same object, not just an equal one).
    # Any "treat None as empty" logic happens in fit/transform instead.
    def __init__(self, datetime_columns: list[str] | None = None):
        self.datetime_columns = datetime_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in (self.datetime_columns or []):
            if col not in out.columns:
                continue
            parsed = pd.to_datetime(out[col], errors="coerce", format="mixed")
            out[f"{col}__year"] = parsed.dt.year.astype("float")
            out[f"{col}__month"] = parsed.dt.month.astype("float")
            out[f"{col}__day"] = parsed.dt.day.astype("float")
            out[f"{col}__dayofweek"] = parsed.dt.dayofweek.astype("float")
            out[f"{col}__is_weekend"] = (parsed.dt.dayofweek >= 5).astype("float")
            hours = parsed.dt.hour
            if hours.nunique(dropna=True) > 1:
                out[f"{col}__hour"] = hours.astype("float")
            out[f"{col}__days_since_epoch"] = (parsed - _EPOCH).dt.days.astype("float")
            out = out.drop(columns=[col])
        return out

    def get_feature_names(self, base: list[str]) -> list[str]:
        # Best-effort helper for downstream naming; not required by sklearn's API.
        cols = self.datetime_columns or []
        names = [c for c in base if c not in cols]
        for col in cols:
            names += [f"{col}__{p}" for p in ("year", "month", "day", "dayofweek", "is_weekend", "days_since_epoch")]
        return names


class TextStatsFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns: list[str] | None = None):
        self.text_columns = text_columns  # verbatim — see note in DatetimeFeaturizer

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in (self.text_columns or []):
            if col not in out.columns:
                continue
            text = out[col].astype(str).fillna("")
            out[f"{col}__char_len"] = text.str.len().astype("float")
            words = text.str.split()
            out[f"{col}__word_count"] = words.map(len).astype("float")
            out[f"{col}__avg_word_len"] = out[f"{col}__char_len"] / out[f"{col}__word_count"].replace(0, np.nan)
            out[f"{col}__avg_word_len"] = out[f"{col}__avg_word_len"].fillna(0.0)
            out = out.drop(columns=[col])
        return out


#: Measured on 20-newsgroups posts through this pipeline (6 topics, 5,796 posts,
#: logistic regression, 5-fold ROC-AUC): 10 components -> 0.977, 20 -> 0.983,
#: 50 -> 0.985, 100 -> 0.986, 200 -> 0.985, against 0.591 with no components at
#: all. The curve is flat past 50 while the cost is not (46s -> 84s -> 145s), so
#: 50 is the knee, not a round number.
TEXT_SVD_COMPONENTS = 50
TEXT_MIN_DOCUMENT_FREQUENCY = 2
TEXT_MAX_VOCABULARY = 20_000

#: A minimum-vocabulary floor was tried here and REJECTED by measurement. The
#: templated notes in data/synthetic_classification.csv prune to 6 terms, are
#: worth nothing (ROC-AUC 0.705 -> 0.707) and cost 38% of the suite's runtime, so
#: skipping thin vocabularies looked free. But tests/test_text_features.py's
#: length-matched corpus prunes to TEN terms and goes 0.50 -> 0.95 on them: a
#: small vocabulary is not a useless one. Cost documented instead; see the trap in
#: CLAUDE.md.


class TextVectorFeaturizer(BaseEstimator, TransformerMixin):
    """
    Turns each free-text column into `n_components` latent semantic dimensions.

    TF-IDF gives one column per word — tens of thousands, sparse, and unusable by
    most of the zoo — so it is projected down with TruncatedSVD, which works on
    the sparse matrix directly. The raw column is left in place for
    TextStatsFeaturizer to consume and drop, so length and word count survive
    alongside the meaning.

    A column whose vocabulary is too thin to decompose is *recorded* in
    `skipped_` and left to the stats features, rather than raising: one
    unusable text column must not take down a run that has twelve good ones.
    """

    def __init__(self, text_columns: list[str] | None = None,
                 n_components: int = TEXT_SVD_COMPONENTS,
                 min_document_frequency: int = TEXT_MIN_DOCUMENT_FREQUENCY,
                 max_vocabulary: int = TEXT_MAX_VOCABULARY,
                 random_state: int = 0):
        self.text_columns = text_columns  # verbatim — see note in DatetimeFeaturizer
        self.n_components = n_components
        self.min_document_frequency = min_document_frequency
        self.max_vocabulary = max_vocabulary
        self.random_state = random_state

    @staticmethod
    def _as_text(column: pd.Series) -> pd.Series:
        # A null is an empty document, not the string "nan": TF-IDF would otherwise
        # learn "nan" as a token and score missingness as a word.
        return column.astype("object").where(column.notna(), "").astype(str)

    def _vectorize(self, text: pd.Series):
        """TF-IDF with English stop words, falling back to keeping them.

        Stop words measured better on real posts (0.985 vs 0.981), but a corpus of
        short, formulaic notes can be *made* of them, and pruning then leaves an
        empty vocabulary.
        """
        attempts = [
            {"stop_words": "english", "min_df": self.min_document_frequency},
            {"stop_words": None, "min_df": 1},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                vectorizer = TfidfVectorizer(max_features=self.max_vocabulary, sublinear_tf=True, **kwargs)
                return vectorizer, vectorizer.fit_transform(text), None
            except ValueError as exc:  # "empty vocabulary", "after pruning no terms remain"
                last_error = exc
        return None, None, f"no usable vocabulary ({last_error})"

    def fit(self, X: pd.DataFrame, y=None):
        self.vectorizers_: dict[str, tuple[Any, Any]] = {}
        self.skipped_: dict[str, str] = {}
        for col in (self.text_columns or []):
            if col not in X.columns:
                continue
            text = self._as_text(X[col])
            # A column of one repeated document has nothing to decompose: SVD on a
            # rank-1 matrix divides by a zero total variance and returns components
            # that are constant for every row.
            n_distinct = int(text.nunique())
            if n_distinct < 3:
                self.skipped_[col] = (
                    f"only {n_distinct} distinct value(s) in this column, which is not a corpus; "
                    f"length and word-count stats only"
                )
                continue
            vectorizer, matrix, error = self._vectorize(text)
            if error:
                self.skipped_[col] = error
                continue
            # SVD needs strictly fewer components than either dimension of the matrix.
            n_components = min(self.n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
            if n_components < 2:
                self.skipped_[col] = (
                    f"a vocabulary of {matrix.shape[1]} terms over {matrix.shape[0]} rows is too "
                    f"small to decompose; length and word-count stats only"
                )
                continue
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            svd.fit(matrix)
            self.vectorizers_[col] = (vectorizer, svd)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        made: dict[str, np.ndarray] = {}
        for col, (vectorizer, svd) in getattr(self, "vectorizers_", {}).items():
            if col not in X.columns:
                continue
            components = svd.transform(vectorizer.transform(self._as_text(X[col])))
            for i in range(components.shape[1]):
                made[f"{col}__svd_{i}"] = components[:, i]
        if not made:
            return X
        # One concat rather than n inserts: 50 columns assigned one at a time
        # fragments the frame and pandas warns about it.
        return pd.concat([X, pd.DataFrame(made, index=X.index)], axis=1)


class NumericInteractionFeaturizer(BaseEstimator, TransformerMixin):
    """
    Generates products and ratios among the top-N target-correlated numeric
    columns, then keeps only the candidates whose mutual information with y
    exceeds both a minimum floor AND the better of its two parent columns'
    individual mutual information — i.e. the interaction has to actually add
    something, not just inherit one parent's signal under a new name.
    """

    def __init__(self, problem_kind: Literal["classification", "regression"] = "classification",
                 top_n_base_columns: int = 8, max_interactions: int = 8,
                 min_mutual_info: float = 0.01, random_state: int = 0):
        self.problem_kind = problem_kind
        self.top_n_base_columns = top_n_base_columns
        self.max_interactions = max_interactions
        self.min_mutual_info = min_mutual_info
        self.random_state = random_state

    def _mi(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.problem_kind == "classification":
                return float(mutual_info_classif(x, y, random_state=self.random_state)[0])
            return float(mutual_info_regression(x, y, random_state=self.random_state)[0])

    def fit(self, X: pd.DataFrame, y=None):
        self.selected_: list[tuple[str, str, str]] = []
        if y is None:
            return self
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if len(numeric_cols) < 2:
            return self

        y_arr = np.asarray(y)
        base_mi: dict[str, float] = {}
        for col in numeric_cols:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
            try:
                base_mi[col] = self._mi(vals, y_arr)
            except Exception:
                base_mi[col] = 0.0

        top_cols = sorted(base_mi, key=base_mi.get, reverse=True)[: self.top_n_base_columns]
        candidates = []
        for a, b in itertools.combinations(top_cols, 2):
            va = pd.to_numeric(X[a], errors="coerce").fillna(0.0).to_numpy()
            vb = pd.to_numeric(X[b], errors="coerce").fillna(0.0).to_numpy()
            try:
                prod_mi = self._mi(va * vb, y_arr)
            except Exception:
                prod_mi = 0.0
            # Require a real margin over the better parent, not just any epsilon
            # improvement — MI estimates are noisy enough that a tiny margin is
            # as likely to be estimation noise as a genuine interaction effect.
            parent_best = max(base_mi.get(a, 0.0), base_mi.get(b, 0.0))
            required = max(parent_best * 1.15, self.min_mutual_info)
            if prod_mi >= required:
                candidates.append((prod_mi, a, b, "product"))

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(vb != 0, va / vb, 0.0)
            try:
                ratio_mi = self._mi(ratio, y_arr)
            except Exception:
                ratio_mi = 0.0
            if ratio_mi >= required:
                candidates.append((ratio_mi, a, b, "ratio"))

        candidates.sort(key=lambda c: c[0], reverse=True)
        self.selected_ = [(a, b, op) for _, a, b, op in candidates[: self.max_interactions]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for a, b, op in getattr(self, "selected_", []):
            if a not in out.columns or b not in out.columns:
                continue
            va = pd.to_numeric(out[a], errors="coerce").fillna(0.0)
            vb = pd.to_numeric(out[b], errors="coerce").fillna(0.0)
            if op == "product":
                out[f"{a}__x__{b}"] = va * vb
            else:
                out[f"{a}__div__{b}"] = np.where(vb != 0, va / vb, 0.0)
        return out
