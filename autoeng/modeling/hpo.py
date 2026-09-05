"""
Hyperparameter optimization for the top candidates off the leaderboard.

Optuna (TPE sampler) searches each candidate's hyperparameter space, scoring
every trial with the exact same cross-validation scheme (folds, metric,
leaky-safe pipeline) used to build the original leaderboard — so a trial's
score is directly comparable to the untuned baseline for that model, and
tuning can never look better than it is by evaluating on an easier split.

Only budget-tunes models with a defined search space below; anything else
(e.g. GaussianNB, LDA — essentially parameter-free) is carried forward at
its baseline CV score with an explicit note that HPO wasn't attempted,
rather than silently pretending it was tuned.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.modeling.model_zoo import SCALE_SENSITIVE_MODELS
from autoeng.modeling.search import _build_pipeline_for_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

SearchSpaceFn = Callable[["optuna.trial.Trial"], dict[str, Any]]


def _rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _boosting_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }


def _xgb_space(trial):
    d = _boosting_space(trial)
    d.update({
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    })
    return d


def _lgbm_space(trial):
    d = _boosting_space(trial)
    d.update({
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    })
    return d


def _catboost_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 600, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


def _decision_tree_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def _knn_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),
    }


def _logreg_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
    }


def _linear_alpha_space(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}


def _elasticnet_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _svc_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _linear_svc_space(trial):
    return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}


def _mlp_space(trial):
    n_layers = trial.suggest_int("n_layers", 1, 2)
    width = trial.suggest_categorical("width", [32, 64, 128])
    return {
        "hidden_layer_sizes": tuple([width] * n_layers),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


SEARCH_SPACES: dict[str, SearchSpaceFn] = {
    "random_forest": _rf_space, "extra_trees": _rf_space,
    "gradient_boosting": _boosting_space, "hist_gradient_boosting": _boosting_space,
    "adaboost": lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
                            "learning_rate": t.suggest_float("learning_rate", 0.01, 2.0, log=True)},
    "xgboost": _xgb_space, "lightgbm": _lgbm_space, "catboost": _catboost_space,
    "decision_tree": _decision_tree_space, "knn": _knn_space,
    "logistic_regression": _logreg_space,
    # NOTE: "ridge" in the regression zoo is RidgeCV, which already does its own
    # internal alpha search over a default grid — it takes no plain `alpha`
    # param, so it's deliberately NOT here (Optuna would just prune every trial).
    "lasso": _linear_alpha_space, "elastic_net": _elasticnet_space,
    "ridge_classifier": _linear_alpha_space,
    "svc_rbf": _svc_space, "svr_rbf": _svc_space,
    "linear_svc": _linear_svc_space, "linear_svr": _linear_svc_space,
    "mlp": _mlp_space,
}


@dataclass
class HPOResult:
    model_name: str
    tuned: bool
    baseline_score: float
    best_score: float
    best_params: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 0
    improvement: float = 0.0
    trial_history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _make_scorer_and_cv(problem_kind: Literal["classification", "regression"], primary_metric: str,
                         cv_folds: int, y: pd.Series):
    if problem_kind == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "roc_auc" if primary_metric == "roc_auc" else primary_metric
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "r2"
    return cv, scoring


def optimize_model(
    model_name: str, model_factory: Callable[[], Any], baseline_score: float,
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment,
    problem_kind: Literal["classification", "regression"], primary_metric: str,
    cv_folds: int = 5, n_trials: int = 25, timeout_seconds: int = 120,
) -> HPOResult:
    space_fn = SEARCH_SPACES.get(model_name)
    if space_fn is None:
        return HPOResult(model_name=model_name, tuned=False, baseline_score=baseline_score, best_score=baseline_score)

    cv, scoring = _make_scorer_and_cv(problem_kind, primary_metric, cv_folds, y)
    trial_history: list[dict[str, Any]] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        params = space_fn(trial)
        try:
            model = model_factory()
            model.set_params(**params)
        except Exception:
            raise optuna.TrialPruned()
        pipe = _build_pipeline_for_model(model_name, lambda: model, roles, problem_kind)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan)
            score = float(np.nanmean(scores))
        except Exception:
            score = float("-inf")
        trial_history.append({"trial": trial.number, "params": params, "score": score})
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    try:
        best_score = study.best_value
        best_params = study.best_params
    except ValueError:
        # No trial completed successfully (every one errored/pruned) — keep the
        # untuned baseline rather than crashing the whole pipeline over HPO.
        best_score = baseline_score
        best_params = {}
    improvement = best_score - baseline_score
    return HPOResult(
        model_name=model_name, tuned=True, baseline_score=baseline_score, best_score=best_score,
        best_params=best_params, n_trials=len(study.trials),
        improvement=improvement, trial_history=trial_history,
    )


def optimize_top_candidates(
    leaderboard_results, model_factories: dict[str, Callable[[], Any]],
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment,
    problem_kind: Literal["classification", "regression"], primary_metric: str,
    top_n: int = 3, n_trials: int = 25,
) -> list[HPOResult]:
    ok_results = sorted(
        [r for r in leaderboard_results if r.status == "ok"],
        key=lambda r: r.metrics.get(primary_metric, float("-inf")), reverse=True,
    )[:top_n]

    outcomes = []
    for r in ok_results:
        factory = model_factories.get(r.name)
        if factory is None:
            continue
        outcome = optimize_model(
            r.name, factory, r.metrics.get(primary_metric, float("-inf")),
            X, y, roles, problem_kind, primary_metric, n_trials=n_trials,
        )
        outcomes.append(outcome)
    return outcomes
