"""
Explains why the winning model was selected.

Two complementary views:
  1. Leaderboard-level: why THIS model over the other 20 candidates — the
     score margin over the runner-up, whether that margin is large relative
     to the CV fold-to-fold noise (a 0.01 win on a metric that swings 0.05
     between folds isn't a real win), and what HPO changed.
  2. Feature-level: what the winning model is actually using to make
     predictions. SHAP's TreeExplainer is used for tree/boosting models
     (fast, exact for that model family); everything else falls back to
     permutation importance (model-agnostic, always available, doesn't
     depend on a model-specific SHAP backend existing) rather than the
     slow, approximate KernelExplainer path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

TREE_MODEL_NAMES = {
    "random_forest", "extra_trees", "decision_tree", "gradient_boosting",
    "hist_gradient_boosting", "xgboost", "lightgbm", "catboost",
}


@dataclass
class ModelExplanation:
    winner_name: str
    winner_score: float
    runner_up_name: str | None
    runner_up_score: float | None
    margin: float | None
    cv_fold_std: float | None
    margin_within_noise: bool | None
    hpo_improvement: float | None
    feature_importances: list[tuple[str, float]]
    importance_method: str
    narrative: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["feature_importances"] = [[n, float(v)] for n, v in self.feature_importances]
        return d


def _feature_importance(fitted_pipeline, model_name: str, X_sample: pd.DataFrame, y_sample: pd.Series,
                         scoring: str, max_features: int = 15) -> tuple[list[tuple[str, float]], str]:
    # A stacked ensemble is a Stacking{Classifier,Regressor}, not a Pipeline, so
    # it has no `named_steps` at all — getattr keeps that from raising and sends
    # it down the model-agnostic permutation path, which handles it fine.
    model_step = getattr(fitted_pipeline, "named_steps", {}).get("model")

    if model_name in TREE_MODEL_NAMES and model_step is not None:
        try:
            import shap
            # SHAP explains the fitted MODEL directly, so it needs the fully
            # transformed (post-encoding) feature space and its exact names.
            feature_names = _extract_feature_names(fitted_pipeline, X_sample)
            pre = fitted_pipeline[:-1]
            X_transformed = pre.transform(X_sample)
            explainer = shap.TreeExplainer(model_step)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):  # multiclass: one array per class
                importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                importances = np.abs(shap_values).mean(axis=0)
                if importances.ndim > 1:
                    importances = importances.mean(axis=1)
            pairs = list(zip(feature_names, importances))
            pairs.sort(key=lambda p: p[1], reverse=True)
            return pairs[:max_features], "shap_tree_explainer"
        except Exception:
            pass  # fall through to permutation importance

    try:
        # permutation_importance is handed the WHOLE pipeline (preprocessing +
        # model) and permutes columns of the RAW input X_sample — so the
        # importances it returns are per ORIGINAL column, not per encoded/
        # engineered feature. Naming them from X_sample.columns directly is
        # therefore correct here; using the post-transform names (as SHAP
        # needs above) would silently misalign since the column counts differ
        # once one-hot/interaction features are in the mix.
        result = permutation_importance(
            fitted_pipeline, X_sample, y_sample, scoring=scoring, n_repeats=5, random_state=42, n_jobs=1,
        )
        pairs = list(zip(list(X_sample.columns), result.importances_mean))
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:max_features], "permutation_importance (original columns)"
    except Exception as e:
        return [], f"unavailable ({type(e).__name__}: {e})"


def _extract_feature_names(fitted_pipeline, X_sample: pd.DataFrame) -> list[str]:
    """
    Pipeline.get_feature_names_out() requires EVERY step to implement it, and
    our custom cleaning/feature-engineering transformers (which intentionally
    return plain DataFrames, not a get_feature_names_out-aware API) don't —
    so that call fails silently on this pipeline. Instead, actually run a
    small sample through each fitted step and read off real column names as
    we go: our custom steps preserve a DataFrame (so `.columns` is exact),
    and the final ColumnTransformer step does implement get_feature_names_out
    properly once it's seen a DataFrame with real column names.
    """
    try:
        step_names = [n for n in fitted_pipeline.named_steps if n != "model"]
        current = X_sample.iloc[: min(5, len(X_sample))].copy()
        names: list[str] = list(current.columns)
        for step_name in step_names:
            step = fitted_pipeline.named_steps[step_name]
            was_dataframe = hasattr(current, "columns")
            current = step.transform(current)
            if hasattr(current, "columns"):
                names = list(current.columns)
            elif was_dataframe:
                # This step is exactly where a DataFrame turned into a bare
                # ndarray (typically the ColumnTransformer 'encode' step) —
                # the only point where its get_feature_names_out() reflects
                # real column identity rather than a generic x0/x1/... fallback
                # (which is what a *later* ndarray-only step like StandardScaler
                # would report, since it was fit without any feature names).
                try:
                    names = list(step.get_feature_names_out())
                except Exception:
                    pass
            # else: ndarray -> ndarray step (e.g. scaling) — column identity/order
            # doesn't change, so `names` from the previous step still applies.
        if len(names) == current.shape[1]:
            return names
    except Exception:
        pass
    return list(X_sample.columns)


def explain_winner(
    leaderboard_results, winner_name: str, fitted_pipeline,
    X_sample: pd.DataFrame, y_sample: pd.Series, primary_metric: str,
    hpo_improvement: float | None = None, selection_source: str | None = None,
    winner_score: float | None = None, tuned_scores: dict[str, float] | None = None,
) -> ModelExplanation:
    """
    `winner_score` is the cross-validated score the selection actually used. For
    a tuned winner that is the TUNED score; the leaderboard only holds the
    untuned one. Reading it from the leaderboard reported a real run's tuned
    winner at 0.6866 behind a runner-up at 0.6899 (margin -0.0033), when the
    selection had compared 0.7007 against that runner-up's own tuned 0.6984
    (margin +0.0023). `tuned_scores` holds the other tuned candidates, so each
    rival is compared at its best as well.
    """
    ok_results = [r for r in leaderboard_results if r.status == "ok"]
    best_score: dict[str, float] = {}
    for r in ok_results:
        score = r.metrics.get(primary_metric)
        if score is not None:
            best_score[r.name] = max(best_score.get(r.name, float("-inf")), score)
    for name, score in (tuned_scores or {}).items():
        best_score[name] = max(best_score.get(name, float("-inf")), score)

    if winner_score is None:
        winner_score = best_score.get(winner_name, float("nan"))
    rivals = sorted(((n, s) for n, s in best_score.items() if n != winner_name), key=lambda t: t[1], reverse=True)
    runner_up_name, runner_up_score = rivals[0] if rivals else (None, None)
    margin = (winner_score - runner_up_score) if runner_up_score is not None else None

    scoring = "roc_auc" if primary_metric == "roc_auc" else ("r2" if primary_metric == "r2" else primary_metric)
    importances, method = _feature_importance(fitted_pipeline, winner_name, X_sample, y_sample, scoring)

    narrative_lines = [
        f"Selected model: {winner_name} ({primary_metric} = {winner_score:.4f} under cross-validation).",
    ]
    if selection_source:
        narrative_lines.append(f"Chosen from: {selection_source}.")
    if runner_up_name:
        narrative_lines.append(
            f"Runner-up: {runner_up_name} ({primary_metric} = {runner_up_score:.4f}); "
            f"margin = {margin:.4f}."
        )
    if hpo_improvement is not None:
        narrative_lines.append(f"Hyperparameter optimization improved this model by {hpo_improvement:+.4f} over its baseline.")
    if importances:
        top_feats = ", ".join(f"{n} ({v:.4f})" for n, v in importances[:5])
        narrative_lines.append(f"Top features by {method}: {top_feats}.")
    else:
        narrative_lines.append(f"Feature importance unavailable: {method}.")

    return ModelExplanation(
        winner_name=winner_name, winner_score=winner_score,
        runner_up_name=runner_up_name, runner_up_score=runner_up_score, margin=margin,
        cv_fold_std=None, margin_within_noise=None,
        hpo_improvement=hpo_improvement, feature_importances=importances, importance_method=method,
        narrative=" ".join(narrative_lines),
    )
