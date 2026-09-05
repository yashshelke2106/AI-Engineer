"""
Stacked ensemble of the leaderboard's strongest candidates.

Model *selection* picks the single best inductive bias and throws away
every other one that was fit. Stacking keeps them: base models make
out-of-fold predictions, and a simple meta-learner learns how to weight
them. It is the main reason serious AutoML systems beat naive
best-model-wins search, and it's close to free here because the base
pipelines already exist.

Two properties this implementation is careful about:

  - LEAKAGE. Each base estimator is the FULL pipeline (preprocessing +
    model), not a pre-transformed matrix, and sklearn's Stacking* meta-
    learner generates its meta-features with internal cross-fitting. So
    every base model's preprocessing is still fit per inner fold, and the
    meta-learner never trains on a base prediction that saw the row.
  - HONESTY. The stack does not automatically win. It is evaluated on the
    exact same outer CV as every other candidate and added to the
    leaderboard as one more competitor. Stacking usually helps; on small
    or noisy data it can overfit the meta-level and lose, and when it
    loses it should lose visibly.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_validate

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.modeling.search import ModelResult, _build_pipeline_for_model, _make_cv

# More than a handful of base models multiplies fit cost without adding much
# diversity — the top few usually span the useful inductive biases already.
DEFAULT_N_BASE_MODELS = 4
# Inner cross-fitting folds for generating meta-features. Kept below the outer
# CV to control the (folds x base models) fit-count blowup.
INNER_CV_FOLDS = 3
STACK_MODEL_NAME = "stacked_ensemble"


def build_stacked_ensemble(
    base_model_names: list[str],
    model_factories: dict[str, Callable[[], Any]],
    roles: FeatureRoleAssignment,
    problem_kind: Literal["classification", "regression"],
):
    estimators = [
        (name, _build_pipeline_for_model(name, model_factories[name], roles, problem_kind))
        for name in base_model_names
        if name in model_factories
    ]
    if len(estimators) < 2:
        return None

    if problem_kind == "classification":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=INNER_CV_FOLDS, n_jobs=1, passthrough=False,
        )
    return StackingRegressor(
        estimators=estimators, final_estimator=RidgeCV(),
        cv=INNER_CV_FOLDS, n_jobs=1, passthrough=False,
    )


def evaluate_stacked_ensemble(
    leaderboard_results: list[ModelResult],
    model_factories: dict[str, Callable[[], Any]],
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment,
    problem_kind: Literal["classification", "regression"],
    primary_metric: str, scoring: dict[str, str], cv_folds: int = 5,
    n_base_models: int = DEFAULT_N_BASE_MODELS,
) -> ModelResult | None:
    """
    Returns a ModelResult for the stack, scored on the same outer CV as every
    other candidate, or None if there aren't enough viable base models.
    """
    ranked = sorted(
        [r for r in leaderboard_results if r.status == "ok" and r.evaluation_stage == "full"],
        key=lambda r: r.metrics.get(primary_metric, float("-inf")), reverse=True,
    )
    base_names = [r.name for r in ranked[:n_base_models]]
    if len(base_names) < 2:
        return None

    stack = build_stacked_ensemble(base_names, model_factories, roles, problem_kind)
    if stack is None:
        return None

    cv = _make_cv(problem_kind, cv_folds)
    try:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(stack, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        elapsed = time.time() - t0
        metrics = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")}
        return ModelResult(
            name=STACK_MODEL_NAME, status="ok", metrics=metrics, fit_time_seconds=elapsed,
            evaluation_stage="full", pipeline=stack,
        )
    except Exception as e:  # noqa: BLE001
        return ModelResult(name=STACK_MODEL_NAME, status="failed",
                           error=f"{type(e).__name__}: {e}")


def describe_stack(base_model_names: list[str]) -> str:
    return (
        f"Stacked ensemble over {len(base_model_names)} base pipelines "
        f"({', '.join(base_model_names)}), combined by a "
        "cross-fitted meta-learner."
    )
