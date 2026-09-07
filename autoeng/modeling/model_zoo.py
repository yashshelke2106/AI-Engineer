"""
Candidate model zoos, one per problem kind.

Every entry is a real, distinct algorithm (not the same estimator with a
cosmetic hyperparameter change relabeled as a new "model" — that would be
padding a count, not testing algorithms). Classification and regression
each have 20+ genuinely different algorithms, spanning linear models,
distance-based methods, naive Bayes, trees, bagging, boosting (both
classic and modern gradient-boosting libraries), kernel methods, and
neural nets, so the search covers meaningfully different inductive biases
rather than twenty flavors of the same idea.

Clustering does NOT get artificially padded to 20 — there simply aren't
20 conceptually distinct, commonly-used clustering algorithms in the way
there are for supervised learning. Inflating the count with redundant
parameterizations of the same algorithm would be dishonest, so clustering
gets an honest, well-chosen set of ~10 instead. Time-series forecasting
reuses the regression zoo via lag-feature engineering (see
autoeng/modeling/time_series.py) plus a handful of classical
forecasting-specific baselines, so it also benefits from the full
regression bench rather than a separate, smaller list.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.cluster import (
    AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans,
    MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression,
    LogisticRegression, RidgeClassifier, RidgeCV, SGDClassifier, SGDRegressor,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

RANDOM_STATE = 42
ModelFactory = Callable[[], Any]

# Suffix marking a `class_weight="balanced"` twin of an existing candidate.
# The twins COMPETE with their unweighted originals rather than replacing them:
# reweighting is a hypothesis about the data, not a known improvement, and on
# some datasets it costs more precision than the recall is worth. It should
# have to win the same cross-validation as everything else.
BALANCED_SUFFIX = "_balanced"


def base_model_name(name: str) -> str:
    """
    Strip the balanced marker to recover the underlying algorithm.

    Anything keyed by algorithm rather than by candidate — HPO search spaces,
    the tree-like set that decides outlier capping — must resolve through this,
    or the twins silently lose behaviour their originals have.
    """
    return name[: -len(BALANCED_SUFFIX)] if name.endswith(BALANCED_SUFFIX) else name


def get_classification_models(n_classes: int = 2) -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "ridge_classifier": lambda: RidgeClassifier(random_state=RANDOM_STATE),
        "sgd_classifier": lambda: SGDClassifier(loss="log_loss", random_state=RANDOM_STATE),
        "linear_svc": lambda: LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
        "svc_rbf": lambda: SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15),
        "gaussian_nb": lambda: GaussianNB(),
        "bernoulli_nb": lambda: BernoulliNB(),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "adaboost": lambda: AdaBoostClassifier(random_state=RANDOM_STATE),
        "bagging": lambda: BaggingClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        "mlp": lambda: MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
        "lda": lambda: LinearDiscriminantAnalysis(),
        # A small reg_param shrinks toward a shared covariance estimate, which
        # avoids the common "covariance matrix is not full rank" failure on
        # one-hot-encoded / collinear feature sets without changing the model.
        "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
    }
    if _HAS_XGB:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        models["xgboost"] = lambda: XGBClassifier(
            n_estimators=300, objective=objective, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    if _HAS_LGBM:
        models["lightgbm"] = lambda: LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    if _HAS_CATBOOST:
        models["catboost"] = lambda: CatBoostClassifier(iterations=300, random_state=RANDOM_STATE, verbose=False)

    # class_weight="balanced" twins (T0-2). Only for estimators that accept the
    # parameter; the boosters and the naive-Bayes/kNN family do not, and forcing
    # an equivalent through sample_weight would need it threaded through every
    # CV call site for a benefit the threshold selector already delivers more
    # directly.
    models.update({
        f"{name}{BALANCED_SUFFIX}": factory
        for name, factory in {
            "logistic_regression": lambda: LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "ridge_classifier": lambda: RidgeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE),
            "linear_svc": lambda: LinearSVC(
                max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
            "svc_rbf": lambda: SVC(
                kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            "decision_tree": lambda: DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
            "extra_trees": lambda: ExtraTreesClassifier(
                n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
        }.items()
    })
    return models


def get_regression_models() -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {
        "linear_regression": lambda: LinearRegression(),
        "ridge": lambda: RidgeCV(),
        "lasso": lambda: Lasso(random_state=RANDOM_STATE),
        "elastic_net": lambda: ElasticNet(random_state=RANDOM_STATE),
        "bayesian_ridge": lambda: BayesianRidge(),
        "huber": lambda: HuberRegressor(max_iter=500),
        "sgd_regressor": lambda: SGDRegressor(random_state=RANDOM_STATE),
        "linear_svr": lambda: LinearSVR(max_iter=5000, random_state=RANDOM_STATE),
        "svr_rbf": lambda: SVR(kernel="rbf"),
        "knn": lambda: KNeighborsRegressor(n_neighbors=15),
        "decision_tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "extra_trees": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "adaboost": lambda: AdaBoostRegressor(random_state=RANDOM_STATE),
        "bagging": lambda: BaggingRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "mlp": lambda: MLPRegressor(max_iter=500, random_state=RANDOM_STATE),
    }
    if _HAS_XGB:
        models["xgboost"] = lambda: XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if _HAS_LGBM:
        models["lightgbm"] = lambda: LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1)
    if _HAS_CATBOOST:
        models["catboost"] = lambda: CatBoostRegressor(iterations=300, random_state=RANDOM_STATE, verbose=False)
    return models


# Models whose objective assumes standardized-scale numeric input (distance-,
# gradient-, or margin-based). Tree/boosting ensembles are scale-invariant and
# deliberately excluded so we don't pay the (harmless but pointless) cost of
# scaling for them.
SCALE_SENSITIVE_MODELS = {
    "logistic_regression", "ridge_classifier", "sgd_classifier", "linear_svc", "svc_rbf", "knn",
    "mlp", "lda", "qda", "linear_regression", "ridge", "lasso", "elastic_net", "bayesian_ridge",
    "huber", "sgd_regressor", "linear_svr", "svr_rbf",
}


def get_clustering_models(n_samples: int, n_clusters: int | None = None) -> dict[str, ModelFactory]:
    # A handful of k choices for the partition-based methods count as configuring
    # the SAME algorithm, not new ones — the zoo's headline count only reflects
    # distinct algorithm entries below. `n_clusters` should come from an actual
    # silhouette-based sweep (see clustering_search.select_k) rather than this
    # fallback, which only fires if no sweep was run.
    default_k = n_clusters if n_clusters is not None else min(8, max(2, n_samples // 50))
    models: dict[str, ModelFactory] = {
        "kmeans": lambda: KMeans(n_clusters=default_k, n_init=10, random_state=RANDOM_STATE),
        "minibatch_kmeans": lambda: MiniBatchKMeans(n_clusters=default_k, n_init=10, random_state=RANDOM_STATE),
        "agglomerative_ward": lambda: AgglomerativeClustering(n_clusters=default_k, linkage="ward"),
        "agglomerative_average": lambda: AgglomerativeClustering(n_clusters=default_k, linkage="average"),
        "dbscan": lambda: DBSCAN(eps=0.5, min_samples=5),
        "gaussian_mixture": lambda: GaussianMixture(n_components=default_k, random_state=RANDOM_STATE),
        "birch": lambda: Birch(n_clusters=default_k),
        "mean_shift": lambda: MeanShift(),
        "spectral": lambda: SpectralClustering(n_clusters=default_k, random_state=RANDOM_STATE, affinity="nearest_neighbors"),
        "optics": lambda: OPTICS(min_samples=5),
    }
    if n_samples <= 2000:
        # O(n^2)-ish memory; fine for small data, skipped for large to avoid stalls.
        models["affinity_propagation"] = lambda: AffinityPropagation(random_state=RANDOM_STATE)
    return models


def available_library_versions() -> dict[str, bool]:
    return {"xgboost": _HAS_XGB, "lightgbm": _HAS_LGBM, "catboost": _HAS_CATBOOST}
