"""
Clustering search: unsupervised, so there's no y to cross-validate against.
Every candidate is fit once on the full (already-cleaned/encoded) feature
matrix, then scored with internal validity indices that don't require
ground-truth labels — silhouette (higher better, cohesion vs. separation),
Calinski-Harabasz (higher better, variance ratio), and Davies-Bouldin
(lower better, average cluster similarity). No single internal index is
authoritative, so the leaderboard reports all three and ranks by
silhouette as the primary (most widely trusted, bounded [-1, 1]) metric.

Since there's no target, preprocessing here always scales numeric features
(distance-based algorithms dominate this zoo) and one-hot-encodes
categoricals — there's no target encoding option without labels.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoeng.cleaning.transformer import AutoCleanerTransformer
from autoeng.common.roles import FeatureRoleAssignment
from autoeng.features.transformers import DatetimeFeaturizer, TextStatsFeaturizer
from autoeng.modeling.model_zoo import get_clustering_models

MAX_SILHOUETTE_SAMPLE = 5000  # silhouette is O(n^2); subsample for large datasets


@dataclass
class ClusteringResult:
    name: str
    status: str
    n_clusters_found: int = 0
    noise_ratio: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def build_unsupervised_preprocessing_pipeline(roles: FeatureRoleAssignment) -> Pipeline:
    steps = [
        ("clean", AutoCleanerTransformer(column_roles=roles.column_roles, cap_outliers=True)),
        ("datetime_features", DatetimeFeaturizer(datetime_columns=roles.datetime_columns)),
        ("text_features", TextStatsFeaturizer(text_columns=roles.text_columns)),
    ]
    encoders = []
    all_categorical = roles.low_card_categorical_columns + roles.high_card_categorical_columns
    if all_categorical:
        encoders.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), all_categorical))
    if encoders:
        steps.append(("encode", ColumnTransformer(transformers=encoders, remainder="passthrough",
                                                    verbose_feature_names_out=False)))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def select_k(X_processed: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> tuple[int, dict[int, float]]:
    """
    Nothing in the data tells us how many clusters actually exist, so we don't
    guess a fixed k — we sweep a small range with KMeans (cheap, deterministic
    enough with n_init) and pick the k that maximizes silhouette, then hand
    that single number to every other partition-based algorithm in the zoo.
    This is still a heuristic (KMeans's notion of a "good" k needn't match a
    density-based method's), but it beats an arbitrary constant.
    """
    n = X_processed.shape[0]
    scores: dict[int, float] = {}
    valid_range = [k for k in k_range if k < n]
    for k in valid_range:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X_processed)
            scores[k] = float(silhouette_score(X_processed, labels))
        except Exception:
            continue
    if not scores:
        return min(8, max(2, n // 50)), scores
    best_k = max(scores, key=scores.get)
    return best_k, scores


def run_clustering_search(X: pd.DataFrame, roles: FeatureRoleAssignment) -> list[ClusteringResult]:
    pre = build_unsupervised_preprocessing_pipeline(roles)
    X_processed = pre.fit_transform(X)
    n_samples = X_processed.shape[0]

    best_k, k_scan_scores = select_k(X_processed)
    models = get_clustering_models(n_samples=n_samples, n_clusters=best_k)
    results: list[ClusteringResult] = []

    sil_sample_idx = None
    if n_samples > MAX_SILHOUETTE_SAMPLE:
        rng = np.random.default_rng(42)
        sil_sample_idx = rng.choice(n_samples, MAX_SILHOUETTE_SAMPLE, replace=False)

    for name, factory in models.items():
        try:
            model = factory()
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(model, "fit_predict"):
                    labels = model.fit_predict(X_processed)
                else:
                    labels = model.fit(X_processed).predict(X_processed)
            elapsed = time.time() - t0

            labels = np.asarray(labels)
            noise_ratio = float((labels == -1).mean()) if (labels == -1).any() else 0.0
            unique_labels = set(labels[labels != -1]) if noise_ratio > 0 else set(labels)
            n_clusters_found = len(unique_labels)

            if n_clusters_found < 2 or n_clusters_found >= n_samples - 1:
                results.append(ClusteringResult(
                    name=name, status="failed", n_clusters_found=n_clusters_found,
                    noise_ratio=noise_ratio, fit_time_seconds=elapsed,
                    error=f"Degenerate clustering ({n_clusters_found} cluster(s) found) — cannot score.",
                ))
                continue

            eval_X, eval_labels = X_processed, labels
            if sil_sample_idx is not None:
                eval_X, eval_labels = X_processed[sil_sample_idx], labels[sil_sample_idx]

            metrics = {}
            try:
                metrics["silhouette"] = float(silhouette_score(eval_X, eval_labels))
            except Exception:
                metrics["silhouette"] = float("nan")
            try:
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_processed, labels))
            except Exception:
                metrics["calinski_harabasz"] = float("nan")
            try:
                metrics["davies_bouldin"] = float(davies_bouldin_score(X_processed, labels))
            except Exception:
                metrics["davies_bouldin"] = float("nan")

            results.append(ClusteringResult(
                name=name, status="ok", n_clusters_found=n_clusters_found,
                noise_ratio=noise_ratio, metrics=metrics, fit_time_seconds=elapsed,
            ))
        except Exception as e:  # noqa: BLE001
            results.append(ClusteringResult(name=name, status="failed", error=f"{type(e).__name__}: {e}"))

    return results, best_k, k_scan_scores


def rank_clustering_results(results: list[ClusteringResult]) -> list[ClusteringResult]:
    ok = [r for r in results if r.status == "ok" and not np.isnan(r.metrics.get("silhouette", float("nan")))]
    return sorted(ok, key=lambda r: r.metrics["silhouette"], reverse=True)
