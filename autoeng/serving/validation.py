"""
Payload validation against the training schema.

This is the part of a serving API that is easy to get wrong in a way nobody
notices. The tempting behaviour, when a request is missing a feature, is to
fill it in — the pipeline has an imputer, the median is right there, the
request succeeds. What comes back is a confident, plausible, wrong prediction
with nothing in the response saying a field was invented. Multiply that by a
caller who renamed a field in a deploy and it is a silent outage.

So the rule here is: **a missing feature is a 422 that names it.** Never a
median.

The distinction that matters, and the one worth getting right:

  - A column *absent from the payload* is a contract violation. The caller
    thinks they sent a complete row and did not. Reject it.
  - A column *present and null* is ordinary missing data. Training saw nulls
    too, the pipeline's imputer is fit per fold to handle exactly this, and
    rejecting it would make the API stricter than the model. Accept it — but
    warn if this column was never null during training, because that usually
    means an upstream join started failing.

Order is enforced, not assumed. The schema records `feature_columns` in the
order the estimator was fit on; a frame built from dict keys comes out in
insertion order, and a positional mismatch scores the wrong columns silently
rather than raising.

**The entity key is the one non-feature a payload may carry.** On grouped data
(T0-3) the group column is excluded from features — it identifies the entity
rather than describing it — so the unknown-column rule rejected it, and the key
never reached the prediction log. Everything downstream that needs to know
which rows are the same customer then had to guess: retraining gave every
served row its own singleton group, and the gate resampled recent traffic by
row, an interval measured 1.9-2.0x too narrow. It is accepted, never required,
and never scored: the frame is built from `feature_columns` alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Unknown columns are rejected by default. A caller sending a field the model
# has never seen is either on the wrong endpoint or ahead of a deploy; either
# way, silently discarding it hides the mismatch.
DEFAULT_ALLOW_UNKNOWN = False


@dataclass
class FieldError:
    kind: str
    column: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ValidationResult:
    frame: pd.DataFrame | None
    errors: list[FieldError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error_dicts(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.errors]


def _coerce_column(values: pd.Series, dtype: str, column: str) -> tuple[pd.Series, FieldError | None]:
    """
    Coerce to the dtype recorded at training time.

    Coercion failures are errors, not silent NaN. `pd.to_numeric(errors=
    "coerce")` would turn the string "N/A" into a NaN that the imputer then
    replaces with a median — the same invented-value problem as a missing
    column, arriving through a different door.
    """
    try:
        if dtype.startswith(("int", "uint", "float")):
            coerced = pd.to_numeric(values, errors="raise")
            # Ints cannot hold NaN, so a nullable integer column is read as
            # float. That is pandas' rule, not a decision worth fighting.
            if dtype.startswith(("int", "uint")) and not coerced.isna().any():
                coerced = coerced.astype(dtype, errors="ignore")
            return coerced, None
        if dtype.startswith("bool"):
            return values.astype("boolean").astype("object"), None
        if dtype.startswith("datetime"):
            return pd.to_datetime(values, errors="raise"), None
        return values.astype("object"), None
    except (ValueError, TypeError) as e:
        return values, FieldError(
            kind="uncoercible_value", column=column,
            detail=(f"could not be read as {dtype} (recorded at training time): {e}. "
                    "Values are not coerced to NaN on failure — that would be imputed "
                    "to a median and returned as a confident prediction."),
        )


def entity_key_column(schema: dict[str, Any]) -> str | None:
    """The group column the model was trained with, if it has one."""
    return (schema.get("feature_roles") or {}).get("group_column") or None


def validate_payload(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    allow_unknown: bool = DEFAULT_ALLOW_UNKNOWN,
) -> ValidationResult:
    """Turn raw request rows into a frame the estimator can score, or explain why not."""
    expected: list[str] = list(schema.get("feature_columns") or [])
    entity_key = entity_key_column(schema)
    column_meta: dict[str, Any] = schema.get("columns") or {}

    if not rows:
        return ValidationResult(frame=None, errors=[FieldError(
            kind="empty_payload", column="", detail="No rows were supplied.",
        )])

    errors: list[FieldError] = []
    warnings: list[str] = []

    # Missing features, reported per column and named. Reported for the whole
    # batch rather than per row: a caller that omitted a field omitted it
    # everywhere, and one error per row would bury that under a thousand copies.
    for name in expected:
        absent_in = [i for i, row in enumerate(rows) if name not in row]
        if absent_in:
            errors.append(FieldError(
                kind="missing_feature", column=name,
                detail=(f"required feature '{name}' is absent from "
                        f"{len(absent_in)} of {len(rows)} row(s) "
                        f"(first at index {absent_in[0]}). It is not imputed: a value "
                        "invented here would be returned as a confident prediction."),
            ))

    # The entity key is known, not unknown — but it is also not a feature, so
    # it is neither required here nor ever copied into the scored frame.
    unknown = sorted({k for row in rows for k in row} - set(expected) - {entity_key})
    if unknown:
        if allow_unknown:
            warnings.append(
                f"Ignored {len(unknown)} column(s) the model was not trained on: "
                f"{', '.join(unknown)}."
            )
        else:
            errors.append(FieldError(
                kind="unknown_column", column=", ".join(unknown),
                detail=(f"{len(unknown)} column(s) not in the training schema: "
                        f"{', '.join(unknown)}. Send allow_unknown=true to ignore them."),
            ))

    if errors:
        return ValidationResult(frame=None, errors=errors, warnings=warnings)

    # Built column-by-column in the SCHEMA's order, never from dict keys.
    frame = pd.DataFrame({name: [row.get(name) for row in rows] for name in expected},
                         columns=expected)

    for name in expected:
        meta = column_meta.get(name) or {}
        dtype = str(meta.get("dtype", "object"))
        coerced, error = _coerce_column(frame[name], dtype, name)
        if error is not None:
            errors.append(error)
            continue
        frame[name] = coerced

        n_null = int(frame[name].isna().sum())
        if n_null and not meta.get("nullable", False):
            warnings.append(
                f"'{name}' is null in {n_null} row(s) but was never null in training; "
                "the pipeline will impute it, so the prediction rests on a value that "
                "was not supplied."
            )

    if errors:
        return ValidationResult(frame=None, errors=errors, warnings=warnings)

    assert list(frame.columns) == expected, "column order must match the training contract"
    return ValidationResult(frame=frame, errors=[], warnings=warnings)
