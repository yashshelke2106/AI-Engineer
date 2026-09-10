"""
Model persistence — the fitted winner, plus everything needed to score a row
against it after the process that trained it has exited.

Before this module the pipeline fit a winner, evaluated it, explained it and
wrote a report about it, and then let it be garbage-collected. Every number in
that report described an artifact nobody had.

Two files are written per run:

  model.joblib          the fitted estimator, exactly as evaluated
  training_schema.json  the contract around it

The schema is the part that is easy to under-build. It carries four things,
each with a named consumer:

  - **Column names, order and dtypes.** T1-1's serving API validates incoming
    payloads against these. Order matters: a positional mismatch scores the
    wrong columns silently rather than raising.
  - **Semantic types and the full FeatureRoleAssignment.** How each column was
    interpreted, so a reload can reproduce the interpretation instead of
    re-deriving it from a single serving row (which has no distribution to
    profile).
  - **Target class labels.** `predict` returns encoded positions for some
    estimators; without the labels the caller cannot name what came back.
  - **Per-feature reference distributions.** Quantiles for numeric columns,
    category frequencies for categorical ones. T1-3 diffs live traffic against
    these to measure drift. They are captured *here*, at training time, because
    this is the last moment the training data is in hand — a drift detector
    that has to go re-read the original CSV is one that stops working the day
    that file moves.

Reference distributions are computed on the **raw** training columns, not the
transformed matrix. Drift has to be measured in the space data arrives in;
a stored transformed matrix stops being comparable the moment the pipeline
changes.

Two things this module is careful about:

  - **It takes an estimator, not a Pipeline.** The ordinary winner is a
    `Pipeline`; the stacked ensemble is a bare `StackingClassifier` holding
    pipelines as base estimators. Nothing here may reach for `.named_steps`.
  - **It records library versions and checks them on load.** An estimator
    unpickled under a different scikit-learn minor version is a documented
    correctness risk that produces no exception — just different numbers.
"""
from __future__ import annotations

import json
import shutil
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from autoeng.common.roles import FeatureRoleAssignment
from autoeng.profiling.profiler import DatasetProfile, SemanticType

SCHEMA_VERSION = 1
MODEL_FILENAME = "model.joblib"
SCHEMA_FILENAME = "training_schema.json"
MLFLOW_MODEL_DIRNAME = "mlflow_model"

# Deciles. PSI is conventionally computed over ten training-quantile bins, so
# storing the edges means T1-3 never has to re-read training data it no longer
# has. 0.0 and 1.0 are included so the reference range is explicit.
REFERENCE_QUANTILES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# A long tail of rare categories bloats the schema without informing drift. The
# mass outside the kept set is preserved as one bucket so frequencies still
# sum to 1 and PSI stays well-defined.
MAX_TRACKED_CATEGORIES = 50
OTHER_CATEGORY_KEY = "__other__"
# Rows used as the MLflow input example / signature sample.
INPUT_EXAMPLE_ROWS = 5

# Libraries whose version can change an unpickled estimator's behaviour.
TRACKED_LIBRARIES = [
    "scikit-learn", "numpy", "scipy", "pandas",
    "xgboost", "lightgbm", "catboost", "joblib",
]
# scikit-learn does not guarantee pickle compatibility across MINOR versions,
# so compare two components there. For everything else only a major-version
# change is worth a warning.
MINOR_VERSION_SENSITIVE = {"scikit-learn"}

# Semantic types whose distribution is better described by category frequencies
# than by quantiles. Mirrors how roles.py routes them: low-cardinality discrete
# numerics and booleans are modelled as categoricals, so they drift like
# categoricals too.
CATEGORICAL_REFERENCE_TYPES = {
    SemanticType.CATEGORICAL_LOW_CARD,
    SemanticType.CATEGORICAL_HIGH_CARD,
    SemanticType.NUMERIC_DISCRETE,
    SemanticType.BOOLEAN,
    SemanticType.IDENTIFIER,
    SemanticType.CONSTANT,
    SemanticType.UNKNOWN,
}


@dataclass
class SavedModel:
    """Where a run's model landed. Paths are absolute so a report can cite them."""
    model_dir: str
    model_path: str
    schema_path: str
    estimator_class: str
    mlflow_model_dir: str | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "saved",
            "model_dir": self.model_dir,
            "model_path": self.model_path,
            "schema_path": self.schema_path,
            "estimator_class": self.estimator_class,
            "mlflow_model_dir": self.mlflow_model_dir,
            "warnings": self.warnings,
        }


@dataclass
class LoadedModel:
    estimator: Any
    schema: dict[str, Any] | None
    warnings: list[str] = field(default_factory=list)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value]
    return value


def _version_or_none(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _library_versions() -> dict[str, str]:
    """Only libraries actually installed — an absent optional booster should
    not be recorded as a pinned requirement of the artifact."""
    return {lib: v for lib in TRACKED_LIBRARIES if (v := _version_or_none(lib)) is not None}


def _numeric_reference(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    observed = numeric.dropna()
    ref: dict[str, Any] = {
        "kind": "numeric",
        "n_observed": int(len(observed)),
        "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "quantiles": {},
    }
    if observed.empty:
        return ref
    # Keys are formatted rather than raw floats so the JSON round-trips to the
    # same edges; consumers sort by float(key).
    ref["quantiles"] = {f"{q:.2f}": float(observed.quantile(q)) for q in REFERENCE_QUANTILES}
    ref["min"] = float(observed.min())
    ref["max"] = float(observed.max())
    ref["mean"] = float(observed.mean())
    ref["std"] = float(observed.std()) if len(observed) > 1 else 0.0
    return ref


def _categorical_reference(series: pd.Series) -> dict[str, Any]:
    observed = series.dropna()
    ref: dict[str, Any] = {
        "kind": "categorical",
        "n_observed": int(len(observed)),
        "n_unique": int(observed.nunique()),
        "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "frequencies": {},
    }
    total = len(observed)
    if total == 0:
        return ref
    counts = observed.value_counts()
    kept = counts.head(MAX_TRACKED_CATEGORIES)
    frequencies = {str(_jsonable(k)): float(v / total) for k, v in kept.items()}
    tail_mass = float((total - int(kept.sum())) / total)
    if tail_mass > 0:
        frequencies[OTHER_CATEGORY_KEY] = tail_mass
    ref["frequencies"] = frequencies
    ref["truncated_to"] = MAX_TRACKED_CATEGORIES if len(counts) > MAX_TRACKED_CATEGORIES else None
    return ref


def _datetime_reference(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
    ref: dict[str, Any] = {
        "kind": "datetime",
        "n_observed": int(len(parsed)),
        "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "quantiles": {},
    }
    if parsed.empty:
        return ref
    ref["quantiles"] = {f"{q:.2f}": str(parsed.quantile(q)) for q in REFERENCE_QUANTILES}
    ref["min"] = str(parsed.min())
    ref["max"] = str(parsed.max())
    return ref


def _text_reference(series: pd.Series) -> dict[str, Any]:
    """Free text is modelled through length/word-count stats, so that is the
    space its drift is measurable in — not the raw strings."""
    observed = series.dropna().astype(str)
    ref: dict[str, Any] = {
        "kind": "text",
        "n_observed": int(len(observed)),
        "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
    }
    if observed.empty:
        return ref
    ref["length"] = _numeric_reference(observed.str.len())
    ref["word_count"] = _numeric_reference(observed.str.split().map(len))
    return ref


def _column_reference(series: pd.Series, semantic_type: SemanticType) -> dict[str, Any]:
    if semantic_type == SemanticType.DATETIME:
        return _datetime_reference(series)
    if semantic_type == SemanticType.TEXT_FREE:
        return _text_reference(series)
    if semantic_type in CATEGORICAL_REFERENCE_TYPES:
        return _categorical_reference(series)
    return _numeric_reference(series)


def _roles_as_dict(roles: FeatureRoleAssignment) -> dict[str, Any]:
    """The FULL assignment, reasoning included — a retrain or a serving-time
    question about why a column was dropped should be answerable from the
    artifact alone."""
    return {
        "numeric_columns": roles.numeric_columns,
        "categorical_columns": roles.categorical_columns,
        "low_card_categorical_columns": roles.low_card_categorical_columns,
        "high_card_categorical_columns": roles.high_card_categorical_columns,
        "datetime_columns": roles.datetime_columns,
        "text_columns": roles.text_columns,
        "excluded_columns": roles.excluded_columns,
        "target_column": roles.target_column,
        "time_column": roles.time_column,
        "reasoning": roles.reasoning,
    }


def build_training_schema(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    profile: DatasetProfile,
    roles: FeatureRoleAssignment,
    problem_type: str,
    model_name: str | None,
    selection_source: str | None = None,
    dataset_path: str | None = None,
    decision_threshold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a later process needs to feed this estimator correctly and
    to notice when the data it is being fed has moved."""
    columns: dict[str, Any] = {}
    for name in X_train.columns:
        col_profile = profile.columns.get(name)
        semantic_type = col_profile.semantic_type if col_profile else SemanticType.UNKNOWN
        columns[str(name)] = {
            "dtype": str(X_train[name].dtype),
            "semantic_type": semantic_type.value,
            "role": roles.column_roles.get(name, "other"),
            "nullable": bool(X_train[name].isna().any()),
            "reference": _column_reference(X_train[name], semantic_type),
        }

    # Pipeline delegates classes_ to its final step and regressors have none,
    # so getattr is the portable read across both estimator shapes.
    classes = getattr(estimator, "classes_", None)
    target: dict[str, Any] = {
        "column": roles.target_column,
        "class_labels": [_jsonable(c) for c in classes] if classes is not None else None,
        "n_classes": int(len(classes)) if classes is not None else None,
    }
    if y_train is not None:
        target["dtype"] = str(y_train.dtype)
        target_profile = profile.columns.get(roles.target_column)
        target_semantic = target_profile.semantic_type if target_profile else SemanticType.UNKNOWN
        target["semantic_type"] = target_semantic.value
        # Prediction drift (T1-3) needs a baseline for the output distribution,
        # and the training target is the honest one.
        target["reference"] = _column_reference(y_train, target_semantic)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "problem_type": problem_type,
        "n_training_rows": int(len(X_train)),
        "model": {
            "name": model_name,
            "selection_source": selection_source,
            "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        },
        # Top-level rather than nested under `target`: this is a serving-time
        # policy, and T1-1 must not have to go looking for it. `predict` alone
        # silently uses 0.5 — the artifact has to carry the number that the
        # reported operating point was measured at, or the two come apart.
        "decision_threshold": decision_threshold,
        # Order is part of the serving contract, not incidental.
        "feature_columns": [str(c) for c in X_train.columns],
        "columns": columns,
        "target": target,
        "feature_roles": _roles_as_dict(roles),
        "library_versions": _library_versions(),
    }


def _write_mlflow_model(
    estimator: Any, X_train: pd.DataFrame, model_dir: Path, schema: dict[str, Any],
) -> tuple[str | None, str | None]:
    """
    Write an MLflow model directory (MLmodel metadata, signature, input
    example) beside the joblib file.

    `mlflow.sklearn.save_model` rather than `log_model` deliberately: this runs
    immediately after `fit`, long before the tracking run is opened at the end
    of the pipeline, and save_model needs no active run. The tracking layer
    logs this directory afterwards to produce the `runs:/` URI.

    Returns (path, error). Failure is reported, never raised — MLflow's export
    is a convenience on top of the joblib artifact, which is the authoritative
    one, and a run that has already spent minutes searching models should not
    lose everything to an optional export.
    """
    target = model_dir / MLFLOW_MODEL_DIRNAME
    try:
        import mlflow.sklearn
        from mlflow.models import infer_signature

        if target.exists():
            shutil.rmtree(target)
        example = X_train.head(INPUT_EXAMPLE_ROWS)
        signature = infer_signature(example, estimator.predict(example))
        mlflow.sklearn.save_model(
            sk_model=estimator,
            path=str(target),
            signature=signature,
            input_example=example,
            # MLflow's default sklearn format is skops, which refuses to
            # serialize any non-sklearn class — and every pipeline here embeds
            # this project's own transformers (AutoCleanerTransformer,
            # DatetimeFeaturizer, ...), so the default fails outright. The
            # alternative to cloudpickle is a hand-maintained
            # `skops_trusted_types` list that breaks silently the next time a
            # transformer is added. Cloudpickle adds no trust surface that
            # model.joblib beside it does not already have.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            # Supplying requirements explicitly skips MLflow's dependency
            # inference, which is slow and can reach the network.
            pip_requirements=[f"{lib}=={ver}" for lib, ver in schema["library_versions"].items()],
        )
        return str(target.resolve()), None
    except Exception as e:  # noqa: BLE001 - optional export, never fatal
        return None, f"MLflow model export skipped: {type(e).__name__}: {e}"


def save_model(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series | None,
    profile: DatasetProfile,
    roles: FeatureRoleAssignment,
    problem_type: str,
    model_name: str | None,
    output_dir: str | Path,
    selection_source: str | None = None,
    dataset_path: str | None = None,
    decision_threshold: dict[str, Any] | None = None,
    write_mlflow_model: bool = True,
) -> SavedModel:
    """Persist a fitted estimator and its training schema into `output_dir`."""
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    schema = build_training_schema(
        estimator, X_train, y_train, profile, roles,
        problem_type=problem_type, model_name=model_name,
        selection_source=selection_source, dataset_path=dataset_path,
        decision_threshold=decision_threshold,
    )

    model_path = model_dir / MODEL_FILENAME
    joblib.dump(estimator, model_path)
    schema_path = model_dir / SCHEMA_FILENAME
    schema_path.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")

    mlflow_dir, export_error = (None, None)
    if write_mlflow_model:
        mlflow_dir, export_error = _write_mlflow_model(estimator, X_train, model_dir, schema)

    return SavedModel(
        model_dir=str(model_dir.resolve()),
        model_path=str(model_path.resolve()),
        schema_path=str(schema_path.resolve()),
        estimator_class=schema["model"]["estimator_class"],
        mlflow_model_dir=mlflow_dir,
        warnings=[export_error] if export_error else [],
    )


def check_library_versions(recorded: dict[str, str]) -> list[str]:
    """
    Compare the versions an artifact was fit under against the ones loading it.

    This is a warning, not a refusal: a mismatch usually works, and hard-failing
    would make old artifacts unloadable for auditing. But the failure mode it
    guards against is silent — an estimator unpickled under a different
    scikit-learn minor version can score differently with no exception — so it
    has to be said out loud.
    """
    messages: list[str] = []
    for lib, trained_version in sorted(recorded.items()):
        current = _version_or_none(lib)
        if current is None:
            messages.append(f"{lib} was present at training time (=={trained_version}) but is not installed now.")
            continue
        if current == trained_version:
            continue
        n_components = 2 if lib in MINOR_VERSION_SENSITIVE else 1
        if trained_version.split(".")[:n_components] != current.split(".")[:n_components]:
            messages.append(
                f"{lib} {trained_version} at training time, {current} now — "
                f"a reloaded estimator can behave differently without raising."
            )
    return messages


def load_training_schema(schema_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def load_model(model_path: str | Path, schema_path: str | Path | None = None) -> LoadedModel:
    """
    Reload a saved estimator, and its schema when one is alongside it.

    Version mismatches surface both as a `RuntimeWarning` (for humans running
    this interactively) and on `LoadedModel.warnings` (for callers that want to
    record or act on them).
    """
    model_path = Path(model_path)
    estimator = joblib.load(model_path)

    if schema_path is None:
        candidate = model_path.parent / SCHEMA_FILENAME
        schema_path = candidate if candidate.exists() else None

    schema = load_training_schema(schema_path) if schema_path is not None else None
    warnings_found = check_library_versions(schema.get("library_versions", {})) if schema else []
    for message in warnings_found:
        _warnings.warn(message, RuntimeWarning, stacklevel=2)

    return LoadedModel(estimator=estimator, schema=schema, warnings=warnings_found)
