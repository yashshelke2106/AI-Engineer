"""
Model search: build a leaky-safe pipeline per candidate algorithm, cross-
validate it, and rank the results into a leaderboard.

Every candidate is evaluated as (preprocessing pipeline -> model) fit fresh
per CV fold — never as (preprocess once on everything, then cross-validate
just the model), which is exactly the fit-on-everything mistake that would
leak validation-fold statistics into imputation/encoding/interaction
selection. This is the same architectural guarantee described in
features/pipeline_builder.py, applied across the whole candidate set.

COMPUTE BUDGET. Running 21 algorithms x 5 folds on the full dataset is fine
at a few thousand rows and untenable at a million. Above a row threshold the
search switches to successive halving: every candidate is screened cheaply
(subsampled rows, fewer folds), only the strongest survivors are promoted to
full cross-validation, and the eliminated ones stay on the leaderboard marked
`screened_out` with the score that eliminated them — visible, not silently
dropped. Screening scores and full-CV scores are computed under different
budgets and are therefore NOT comparable, so only fully-evaluated candidates
are eligible to win; the stage is recorded on every result to keep that
distinction explicit.

A model that errors out (singular matrix, unsupported data shape, etc.) is
recorded as a failed candidate with the exception message rather than
crashing the whole search — one bad algorithm shouldn't take down a
leaderboard of twenty others.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.features.pipeline_builder import build_preprocessing_pipeline
from autoeng.modeling.model_zoo import (
    SCALE_SENSITIVE_MODELS, base_model_name, get_classification_models, get_regression_models,
)

# Known-slow candidates get skipped above this many training rows rather than
# silently hanging the search — a heuristic AutoML engine has to budget
# compute across candidates, not spend it all on the least scalable one.
SLOW_MODEL_ROW_LIMIT = {
    "svc_rbf": 20000, "mlp": 50000, "bagging": 50000, "gaussian_mixture": 50000,
}

# Below this many rows, a full search is cheap enough that halving would only
# add risk (screening on a subsample of a small dataset is noisy) for no real
# saving. Above it, screen first.
HALVING_MIN_ROWS = 3000
SCREENING_ROWS = 2000
SCREENING_FOLDS = 3
SURVIVORS_PROMOTED = 6

CLASSIFICATION_SCORING = {
    # Plain "roc_auc" (not "roc_auc_ovr") falls back to decision_function for
    # models without predict_proba (RidgeClassifier, LinearSVC) instead of
    # erroring out on them.
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
}
REGRESSION_SCORING = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}

# Grouped CV needs enough entities to fill every fold; below this a screening
# subsample would leave folds with almost no groups.
MIN_GROUPS_FOR_CV = 10

TREE_LIKE_MODELS = {
    "random_forest", "extra_trees", "decision_tree", "gradient_boosting",
    "hist_gradient_boosting", "xgboost", "lightgbm", "catboost", "adaboost", "bagging",
}


@dataclass
class ModelResult:
    name: str
    status: Literal["ok", "failed", "skipped", "screened_out"]
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    error: str | None = None
    evaluation_stage: Literal["full", "screening"] = "full"
    pipeline: Any = None  # unfit Pipeline template (cloned+fit later on full train set if selected)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "pipeline"}


@dataclass
class Leaderboard:
    problem_kind: Literal["classification", "regression"]
    primary_metric: str
    results: list[ModelResult]
    budget_note: str = ""

    def ranked(self) -> list[ModelResult]:
        """Only fully-evaluated candidates can win — screening scores were
        measured under a smaller budget and aren't comparable."""
        ok = [r for r in self.results if r.status == "ok" and r.evaluation_stage == "full"]
        return sorted(ok, key=lambda r: r.metrics.get(self.primary_metric, float("-inf")), reverse=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_kind": self.problem_kind,
            "primary_metric": self.primary_metric,
            "budget_note": self.budget_note,
            "results": [r.as_dict() for r in self.results],
        }


def _build_pipeline_for_model(name: str, model_factory, roles: FeatureRoleAssignment,
                               problem_kind: str) -> Any:
    # Tree ensembles are outlier-robust by construction, so IQR capping buys
    # them nothing and can only distort genuine extreme values.
    # Resolve through the base name so a class_weight="balanced" twin inherits
    # its original's preprocessing rather than silently getting IQR capping.
    base = base_model_name(name)
    cap_outliers = base not in TREE_LIKE_MODELS
    pre = build_preprocessing_pipeline(roles, problem_kind=problem_kind, cap_outliers=cap_outliers)
    steps = list(pre.steps)
    if base in SCALE_SENSITIVE_MODELS:
        steps.append(("scale", StandardScaler(with_mean=True)))
    steps.append(("model", model_factory()))
    return Pipeline(steps)


def _make_cv(problem_kind: str, n_splits: int, grouped: bool = False):
    """
    Fold splitter. When `grouped`, whole entities move together so no entity is
    ever scored by a model that trained on its other rows.

    StratifiedGroupKFold cannot always honour both constraints exactly — it
    balances classes as well as whole groups allow — which is the correct
    trade: an approximately balanced fold is a nuisance, an entity spanning the
    split is a leak.
    """
    if grouped:
        if problem_kind == "classification":
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if problem_kind == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _subsample(X: pd.DataFrame, y: pd.Series, problem_kind: str, n_rows: int, groups=None):
    """
    Take a screening subsample.

    With groups, sample whole ENTITIES rather than rows: a row-wise subsample
    would scatter an entity's rows across the screening folds and reintroduce
    exactly the leakage grouping exists to prevent — at the stage that decides
    which candidates survive.
    """
    if len(X) <= n_rows:
        return X, y, groups

    if groups is None:
        stratify = y if problem_kind == "classification" and y.value_counts().min() >= 2 else None
        X_small, _, y_small, _ = train_test_split(
            X, y, train_size=n_rows, random_state=42, stratify=stratify,
        )
        return X_small, y_small, None

    groups = np.asarray(groups)
    unique = pd.unique(groups)
    rng = np.random.default_rng(42)
    keep_fraction = n_rows / len(X)
    n_keep = max(int(round(len(unique) * keep_fraction)), MIN_GROUPS_FOR_CV)
    keep = set(rng.choice(unique, size=min(n_keep, len(unique)), replace=False).tolist())
    mask = np.array([g in keep for g in groups])
    return X[mask], y[mask], groups[mask]


def _evaluate_candidate(name: str, factory, roles: FeatureRoleAssignment, problem_kind: str,
                         X: pd.DataFrame, y: pd.Series, cv, scoring: dict[str, str],
                         stage: str, groups=None) -> ModelResult:
    limit = SLOW_MODEL_ROW_LIMIT.get(base_model_name(name))
    if limit and len(X) > limit:
        return ModelResult(name=name, status="skipped", evaluation_stage=stage,
                           error=f"n_rows={len(X)} > {limit} row cap for this model")
    try:
        pipe = _build_pipeline_for_model(name, factory, roles, problem_kind)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=scoring,
                                    n_jobs=1, error_score="raise")
        elapsed = time.time() - t0
        metrics = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")}
        return ModelResult(name=name, status="ok", metrics=metrics, fit_time_seconds=elapsed,
                           evaluation_stage=stage, pipeline=pipe)
    except Exception as e:  # noqa: BLE001 - one bad candidate must not kill the search
        return ModelResult(name=name, status="failed", evaluation_stage=stage,
                           error=f"{type(e).__name__}: {e}")


def _run_search(
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment, problem_kind: str,
    models: dict[str, Any], scoring: dict[str, str], primary_metric: str,
    cv_folds: int, budget: Literal["auto", "full"], groups=None,
) -> Leaderboard:
    use_halving = budget == "auto" and len(X) > HALVING_MIN_ROWS
    grouped = groups is not None
    full_cv = _make_cv(problem_kind, cv_folds, grouped=grouped)

    if not use_halving:
        note = (f"Full {cv_folds}-fold CV on all {len(models)} candidates "
                f"({len(X)} rows — below the {HALVING_MIN_ROWS}-row halving threshold).")
        results = [
            _evaluate_candidate(name, factory, roles, problem_kind, X, y, full_cv, scoring,
                                 "full", groups=groups)
            for name, factory in models.items()
        ]
        return Leaderboard(problem_kind=problem_kind, primary_metric=primary_metric,
                           results=results, budget_note=note)

    # Stage 1 — cheap screen of everything.
    X_screen, y_screen, groups_screen = _subsample(X, y, problem_kind, SCREENING_ROWS, groups)
    screen_cv = _make_cv(problem_kind, SCREENING_FOLDS, grouped=grouped)
    screened = {
        name: _evaluate_candidate(name, factory, roles, problem_kind, X_screen, y_screen,
                                   screen_cv, scoring, "screening", groups=groups_screen)
        for name, factory in models.items()
    }

    ok_screened = sorted(
        [r for r in screened.values() if r.status == "ok"],
        key=lambda r: r.metrics.get(primary_metric, float("-inf")), reverse=True,
    )
    survivors = [r.name for r in ok_screened[:SURVIVORS_PROMOTED]]

    # Stage 2 — full CV for survivors only.
    results: list[ModelResult] = []
    for name, factory in models.items():
        if name in survivors:
            promoted = _evaluate_candidate(name, factory, roles, problem_kind, X, y,
                                            full_cv, scoring, "full", groups=groups)
            promoted.fit_time_seconds += screened[name].fit_time_seconds
            results.append(promoted)
            continue
        eliminated = screened[name]
        if eliminated.status == "ok":
            eliminated.status = "screened_out"
            eliminated.error = (
                f"Eliminated at screening: {primary_metric}="
                f"{eliminated.metrics.get(primary_metric, float('nan')):.4f} on "
                f"{len(X_screen)} rows / {SCREENING_FOLDS} folds, outside the top {SURVIVORS_PROMOTED}."
            )
        results.append(eliminated)

    note = (
        f"Successive halving: all {len(models)} candidates screened on {len(X_screen)} rows / "
        f"{SCREENING_FOLDS} folds, top {len(survivors)} promoted to full {cv_folds}-fold CV on "
        f"{len(X)} rows. Screening and full scores are measured under different budgets and are "
        "not directly comparable; only fully-evaluated candidates are eligible to win."
    )
    return Leaderboard(problem_kind=problem_kind, primary_metric=primary_metric,
                       results=results, budget_note=note)


def run_classification_search(
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment,
    n_classes: int, cv_folds: int = 5, budget: Literal["auto", "full"] = "auto", groups=None,
) -> Leaderboard:
    models = get_classification_models(n_classes=n_classes)
    primary_metric = "roc_auc" if n_classes == 2 else "f1_macro"
    scoring = CLASSIFICATION_SCORING if n_classes == 2 else {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    return _run_search(X, y, roles, "classification", models, scoring, primary_metric,
                       cv_folds, budget, groups=groups)


def run_regression_search(
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment, cv_folds: int = 5,
    budget: Literal["auto", "full"] = "auto", groups=None,
) -> Leaderboard:
    models = get_regression_models()
    return _run_search(X, y, roles, "regression", models, REGRESSION_SCORING, "r2",
                       cv_folds, budget, groups=groups)
