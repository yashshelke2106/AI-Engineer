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
- NumericInteractionFeaturizer: this one DOES need y, and DOES need to be
  fit-only-on-train — it searches pairwise products/ratios of the most
  target-relevant numeric columns and keeps only the ones that carry
  measurable extra signal (by mutual information), rather than exploding
  every pair combinatorially and hoping the model sorts it out.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
