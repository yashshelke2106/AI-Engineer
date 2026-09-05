"""
Column-name prior for target detection.

An earlier version of this system deliberately refused to look at column
names, on the principle that inferring from names is "cheating" and only
measurable properties of the data should count. Testing killed that
principle. On sklearn's diabetes dataset, the real target (disease
progression) and `s5` (a blood serum measurement) are structurally
indistinguishable: both continuous, both moderately predictable from the
other columns, both with diffuse feature importance. No statistic
separates them, because what separates them is human intent — and the only
place intent is recorded in a bare CSV is the column name.

A column named `target` IS part of the dataset. It isn't a data dictionary
handed over on the side. Declining to read it doesn't make the system more
autonomous, just worse.

So: names are a PRIOR, never a rule. A column named "target" that turns out
to be constant, near-unique, or unpredictable still loses on the other
signals; a dataset with no recognizable names still works exactly as
before, on shape and fit-and-check alone. This just stops the system from
ignoring the most informative three bytes in the file.
"""
from __future__ import annotations

import re

# Tiered because these are not equally diagnostic. "target" in a column name
# is close to decisive; "score" or "status" shows up on features constantly.
_STRONG = {
    "target", "label", "labels", "y", "outcome", "outcomes", "response",
    "dependent", "groundtruth", "gt", "class", "classes",
}
_MEDIUM = {
    "churn", "churned", "diagnosis", "survived", "default", "defaulted", "fraud",
    "fraudulent", "converted", "conversion", "result", "verdict", "winner",
    "success", "failed", "failure", "approved", "accepted", "clicked", "purchased",
    "subscribed", "attrition", "readmitted", "malignant",
}
_WEAK = {
    "price", "cost", "sales", "revenue", "demand", "rating", "score", "grade",
    "severity", "risk", "status", "amount", "value", "quality", "duration",
    "lifetime", "yield", "profit", "loss",
}

_SPLIT = re.compile(r"[^a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(name: str) -> set[str]:
    spaced = _CAMEL.sub(" ", str(name))
    return {t for t in _SPLIT.split(spaced.lower()) if t}


def name_prior(column_name: str) -> tuple[float, str]:
    """
    Returns (prior in [0, 1], reason). 0.0 means the name says nothing —
    which is the common case and must stay perfectly usable.
    """
    tokens = _tokenize(column_name)
    if not tokens:
        return 0.0, "no usable tokens in column name"

    hit_strong = tokens & _STRONG
    if hit_strong:
        return 1.0, f"column name contains a strong target token {sorted(hit_strong)}"
    hit_medium = tokens & _MEDIUM
    if hit_medium:
        return 0.7, f"column name contains a likely outcome token {sorted(hit_medium)}"
    hit_weak = tokens & _WEAK
    if hit_weak:
        return 0.4, f"column name contains a weak outcome-ish token {sorted(hit_weak)}"
    return 0.0, "column name carries no target-like tokens"
