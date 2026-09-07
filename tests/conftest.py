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
