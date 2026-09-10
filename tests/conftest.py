"""Shared fixtures. Everything is generated in-memory and deterministic —
tests must be fast enough to run on every change, so nothing here loads a
file or fits a 21-model leaderboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def clean_classification_df() -> pd.DataFrame:
    """A well-behaved binary classification frame with real (but not perfect) signal."""
    rng = np.random.default_rng(0)
    n = 300
    age = rng.integers(18, 80, n)
    income = rng.exponential(50_000, n).round(2)
    city = rng.choice(["Pune", "Mumbai", "Delhi"], n)
    logit = -0.05 * (age - 45) - 0.00002 * (income - 50_000) + rng.normal(0, 1, n)
    churned = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": age,
        "income": income,
        "city": city,
        "churned": churned,
    })


@pytest.fixture
def leaky_classification_df(clean_classification_df: pd.DataFrame) -> pd.DataFrame:
    """Same frame plus a column that is a near-perfect copy of the target."""
    df = clean_classification_df.copy()
    rng = np.random.default_rng(1)
    flip = rng.uniform(0, 1, len(df)) < 0.01
    df["internal_risk_flag"] = np.where(flip, 1 - df["churned"], df["churned"])
    return df


@pytest.fixture
def timeseries_df() -> pd.DataFrame:
    """Random walk: the naive last-value forecast is near-optimal by construction."""
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    walk = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame({
        "date": dates.astype(str),
        "sales": walk.round(3),
        "promo_flag": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })


@pytest.fixture
def imbalanced_classification_df() -> pd.DataFrame:
    """
    744 rows, 29 positives (3.90%) — the shape that breaks a 0.5 threshold.

    This is the fixture T0-2 is measured against. The point of it is that the
    signal is genuinely strong (ROC-AUC lands between 0.92 and 0.96 depending
    on the model) while recall at the default 0.5 threshold is dismal: roughly
    6 to 12 of the 29 positives, because a rare positive almost never pushes a
    calibrated probability past 0.5. That gap between the ranking metric and
    the deployed behaviour is the entire subject of T0-2.

    Parameters are tuned, not arbitrary: `noise` is small relative to `signal`
    so the features really are predictive, and the intercept is set so the
    positive rate lands at the intended rarity rather than wherever it fell.
    """
    rng = np.random.default_rng(0)
    n = 744
    signal, intercept, noise = 2.6, -7.7, 0.25

    account_age_days = rng.integers(30, 3000, n)
    n_transactions = rng.poisson(40, n) + 1
    avg_amount = rng.lognormal(4.0, 0.9, n).round(2)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.35, 0.3, 0.2, 0.15])
    device = rng.choice(["mobile", "web", "api"], n, p=[0.6, 0.3, 0.1])
    prior_disputes = rng.poisson(0.3, n)

    z = signal * (
        -1.10 * (account_age_days - 1500) / 1000
        + 0.95 * (np.log(avg_amount) - 4.0)
        + 0.85 * prior_disputes
        + 0.60 * (device == "api")
        - 0.40 * (n_transactions - 40) / 20
    ) + rng.normal(0, noise, n)
    p = 1 / (1 + np.exp(-(z + intercept)))
    is_fraud = (rng.uniform(0, 1, n) < p).astype(int)

    return pd.DataFrame({
        "account_id": [f"A{i:05d}" for i in range(n)],
        "account_age_days": account_age_days,
        "n_transactions": n_transactions,
        "avg_amount": avg_amount,
        "region": region,
        "device_type": device,
        "prior_disputes": prior_disputes,
        "is_fraud": is_fraud,
    })


@pytest.fixture
def grouped_leakage_df() -> pd.DataFrame:
    """
    750 rows, 150 customers, 5 visits each — the shape that makes plain K-fold
    lie. Measured through the project's own pipeline: **ROC-AUC 0.976 under
    row-wise K-fold, 0.682 under GroupKFold.** That 0.29 is the leak.

    The label is decided at the CUSTOMER level, so a customer's five rows carry
    an identical answer. Split by row and the model sees four of them in
    training and is asked about the fifth.

    **The stable per-customer attributes are what make that exploitable, and
    they are the part worth understanding.** An entity-level label alone is not
    enough: with only noisy per-visit readings the gap is around 0.06, because
    nothing tells the model which rows belong together. `device_fingerprint`
    and `home_region` are constant per customer and do not cause the outcome at
    all — they identify who the row belongs to. That is what converts an
    entity-level label into memorisation, and it is exactly what real entity
    data carries: device ids, demographics, account attributes, home location.

    The duplicate-row check cannot see any of this — the rows genuinely differ.
    What is shared is the entity, and nothing in the pipeline knew entities
    existed.

    `home_region` is also a deliberate decoy for the group detector: it repeats
    consistently, but with only 5 distinct values it is an ordinary categorical
    and must be rejected. Grouping on it would hold out a fifth of the feature
    space per fold.

    The label is only *partly* determined by the trait (Bernoulli on a logistic
    of it), so the honest ceiling stays modest and the memorisation gap stays
    wide.
    """
    rng = np.random.default_rng(7)
    n_customers, visits = 150, 5

    trait = rng.normal(0, 1, n_customers)          # drives the outcome
    fingerprint = rng.normal(0, 1, n_customers)    # identifies the customer, causes nothing
    home_region = rng.choice(["north", "south", "east", "west", "central"], n_customers)
    converted = (rng.uniform(0, 1, n_customers) < 1 / (1 + np.exp(-1.0 * trait))).astype(int)

    rows = []
    for c in range(n_customers):
        for v in range(visits):
            rows.append({
                "customer_id": f"C{c:04d}",
                "home_region": home_region[c],
                "device_fingerprint": round(fingerprint[c] + rng.normal(0, 0.01), 4),
                "measure_a": round(trait[c] + rng.normal(0, 0.01), 4),
                "measure_b": round(trait[c] * 0.4 + rng.normal(0, 0.9), 4),
                "measure_c": round(rng.normal(0, 1.0), 4),
                "converted": int(converted[c]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sorted_by_class_df() -> pd.DataFrame:
    """Rows sorted by class — the shape that breaks unshuffled K-fold screening."""
    rng = np.random.default_rng(3)
    frames = []
    for cls, centre in enumerate([0.0, 5.0, 10.0]):
        frames.append(pd.DataFrame({
            "measure_a": rng.normal(centre, 0.5, 50),
            "measure_b": rng.normal(centre * 0.8, 0.5, 50),
            "species_code": cls,
        }))
    return pd.concat(frames, ignore_index=True)
