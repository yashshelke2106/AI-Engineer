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

PARALLELISM (T2-4) happens across candidates, never inside a candidate's CV.
The cores are divided between the workers, because half the zoo already asks
for every core (`n_jobs=-1` on the forests, bagging, XGBoost and LightGBM) and
16 workers each doing that is 256 threads fighting over 16 cores. It buys less
than the core count suggests, and the measurements say why: the slowest single
candidate is the floor, and every candidate ran ~4x slower inside a starved
worker than alone with the machine to itself. See `WORKER_SHARE_OF_CORES`.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# sklearn's wrappers, not joblib's: they carry sklearn's configuration into the
# workers, and sklearn warns once per nested call when the plain ones are used.
from sklearn.utils.parallel import Parallel, delayed

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

#: Candidates evaluated at once (T2-4). -1 picks `default_search_workers()`;
#: 1 is the old sequential search. See `_evaluate_all`.
SEARCH_JOBS = -1
#: Native threads each worker may use. None divides the cores evenly between
#: workers, so workers x threads never exceeds the machine.
SEARCH_THREADS_PER_WORKER: int | None = None
#: Measured on 16 cores (T2-4), every setting run twice in alternating order,
#: mean wall clock:
#:
#:                      500-row full search   12,000-row halving search
#:   sequential               196s                  109s
#:   4 workers x 4 threads     74s  (2.66x)          97s  (1.11x)
#:   8 x 1                     94s  (2.08x)          69s  (1.57x)
#:   8 x 2                     95s  (2.06x)          98s  (1.11x)
#:   16 x 1                    88s  (2.24x)         167s  (0.65x)
#:
#: Repeats of one setting differed by up to 2x, so 4 x 4 and 8 x 1 are a tie
#: (~1.9x on average). What is not noise: a worker per core loses to doing
#: nothing on the halving search, whose six survivors are a short critical
#: path. A quarter of the cores as workers, the rest as their threads.
WORKER_SHARE_OF_CORES = 4

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


def default_search_workers(cores: int | None = None) -> int:
    """Workers for `SEARCH_JOBS = -1`: a quarter of the cores, at least two if there are two."""
    cores = cores or os.cpu_count() or 1
    return max(min(2, cores), cores // WORKER_SHARE_OF_CORES)


def _evaluate_all(jobs: list[tuple], n_jobs: int) -> list[ModelResult]:
    """Evaluate candidates concurrently, in the order given.

    `inner_max_num_threads` is the point of this function as much as the pool
    is: the forests, bagging, XGBoost and LightGBM all ask for every core on
    their own, and nesting that inside N workers oversubscribes the machine
    badly enough to run slower than doing nothing. Each worker gets its share,
    cores // workers. joblib also makes nested joblib calls sequential inside its
    workers, so a candidate's own CV and its estimator's `n_jobs=-1` stay
    in-process without `_evaluate_candidate` having to know about any of it.

    A candidate that raises is already caught and recorded inside
    `_evaluate_candidate`, so the only failures reaching here are the pool's own
    (a worker killed by the OS, a memory error, a pickling failure). Those fall
    back to a sequential pass with a RuntimeWarning, rather than discarding a
    whole search.
    """
    if n_jobs == 1 or len(jobs) <= 1:
        return [_evaluate_candidate(*args) for args in jobs]
    cores = os.cpu_count() or 1
    workers = min(len(jobs), default_search_workers(cores) if n_jobs < 1 else n_jobs)
    if workers <= 1:
        return [_evaluate_candidate(*args) for args in jobs]
    threads = SEARCH_THREADS_PER_WORKER or max(1, cores // workers)
    try:
        with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=threads):
            return list(Parallel()(delayed(_evaluate_candidate)(*args) for args in jobs))
    except Exception as exc:  # noqa: BLE001 — the pool failing must not lose the search
        warnings.warn(f"Parallel candidate evaluation failed ({type(exc).__name__}: {exc}); "
                      f"falling back to sequential.", RuntimeWarning, stacklevel=2)
        return [_evaluate_candidate(*args) for args in jobs]


def _run_search(
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment, problem_kind: str,
    models: dict[str, Any], scoring: dict[str, str], primary_metric: str,
    cv_folds: int, budget: Literal["auto", "full"], groups=None, n_jobs: int = SEARCH_JOBS,
) -> Leaderboard:
    use_halving = budget == "auto" and len(X) > HALVING_MIN_ROWS
    grouped = groups is not None
    full_cv = _make_cv(problem_kind, cv_folds, grouped=grouped)

    if not use_halving:
        note = (f"Full {cv_folds}-fold CV on all {len(models)} candidates "
                f"({len(X)} rows — below the {HALVING_MIN_ROWS}-row halving threshold).")
        results = _evaluate_all(
            [(name, factory, roles, problem_kind, X, y, full_cv, scoring, "full", groups)
             for name, factory in models.items()],
            n_jobs,
        )
        return Leaderboard(problem_kind=problem_kind, primary_metric=primary_metric,
                           results=results, budget_note=note)

    # Stage 1 — cheap screen of everything.
    X_screen, y_screen, groups_screen = _subsample(X, y, problem_kind, SCREENING_ROWS, groups)
    screen_cv = _make_cv(problem_kind, SCREENING_FOLDS, grouped=grouped)
    screened = dict(zip(models, _evaluate_all(
        [(name, factory, roles, problem_kind, X_screen, y_screen, screen_cv, scoring,
          "screening", groups_screen) for name, factory in models.items()],
        n_jobs,
    )))

    ok_screened = sorted(
        [r for r in screened.values() if r.status == "ok"],
        key=lambda r: r.metrics.get(primary_metric, float("-inf")), reverse=True,
    )
    survivors = [r.name for r in ok_screened[:SURVIVORS_PROMOTED]]

    # Stage 2 — full CV for survivors only.
    promoted_results = dict(zip(survivors, _evaluate_all(
        [(name, models[name], roles, problem_kind, X, y, full_cv, scoring, "full", groups)
         for name in survivors],
        n_jobs,
    )))

    results: list[ModelResult] = []
    for name in models:
        if name in survivors:
            promoted = promoted_results[name]
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
    n_jobs: int = SEARCH_JOBS,
) -> Leaderboard:
    models = get_classification_models(n_classes=n_classes)
    primary_metric = "roc_auc" if n_classes == 2 else "f1_macro"
    scoring = CLASSIFICATION_SCORING if n_classes == 2 else {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    return _run_search(X, y, roles, "classification", models, scoring, primary_metric,
                       cv_folds, budget, groups=groups, n_jobs=n_jobs)


def run_regression_search(
    X: pd.DataFrame, y: pd.Series, roles: FeatureRoleAssignment, cv_folds: int = 5,
    budget: Literal["auto", "full"] = "auto", groups=None, n_jobs: int = SEARCH_JOBS,
) -> Leaderboard:
    models = get_regression_models()
    return _run_search(X, y, roles, "regression", models, REGRESSION_SCORING, "r2",
                       cv_folds, budget, groups=groups, n_jobs=n_jobs)
