"""
The cases entity recovery used to refuse, measured on no-drift windows (share
flagged) before fixing:

  crowded    2,000 training customers: the signature merged 14% of them, failed
             membership validation, and keyless 60-customer windows were flagged
             in 55%, 10-visit windows in 70%.
  anonymous  no feature constant per customer: no signature can exist, and
             keyless 30-customer windows of 10 visits were flagged in 22%.
  no holdout an artifact predating stored sizes, with no frozen holdout, could
             not estimate them at all.

Fixes, each measured:

  - a signature is validated on what drift uses it for — every column's
    effective size within -25% .. +10% of the true one — not on exact
    membership. The crowded signature sizes within -15% (merging reads as more
    clustering, the conservative side) and its windows flag 0-2%, at the keyed
    detector's power.
  - without a signature, a keyless window assumes training's design: each
    column's ICC at training's rows per entity (0% at 5 visits, 10% at 10).
  - without a holdout, sizes come from the original dataset (the schema's
    dataset_path, or --reference-data), or as a last resort from the window's
    own keyed rows.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from autoeng.common.entities import MAX_OVERSIZE, MAX_UNDERSIZE
from autoeng.monitoring.drift import DriftSeverity, check_data_drift, with_estimated_reference_sizes
from tests.test_drift_noise import IMPORTANCES, KEY, _entities, _grouped_schema

ANONYMOUS_IMPORTANCES = {"measure_a": 0.77, "measure_b": 0.15, "measure_c": 0.08}


def _anonymous(n_customers: int, visits: int, rng, shift: float = 0.0) -> pd.DataFrame:
    """Customer-level trait, but nothing a row carries is constant per customer."""
    trait = rng.normal(0, 1, n_customers)
    c = np.repeat(np.arange(n_customers), visits)
    n = len(c)
    return pd.DataFrame({
        KEY: [f"A{i:05d}" for i in c],
        "measure_a": (trait[c] + shift + rng.normal(0, 0.35, n)).round(4),
        "measure_b": (trait[c] * 0.4 + rng.normal(0, 0.9, n)).round(4),
        "measure_c": rng.normal(0, 1.0, n).round(4),
    })


def _stripped(schema: dict) -> dict:
    old = copy.deepcopy(schema)
    for meta in old["columns"].values():
        for key in ("n_effective", "n_effective_mean", "icc_bins", "icc_mean"):
            meta["reference"].pop(key, None)
    old.pop("entity_signature", None)
    old.pop("entity_design", None)
    return old


def _flagged(schema, importances, make, seed, n=30, keyed=False) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        window = make(rng)
        if not keyed:
            window = window.drop(columns=[KEY])
        count += check_data_drift(window, schema, importances).severity != DriftSeverity.OK
    return count / n


class TestCrowdedEntities:
    @pytest.fixture(scope="class")
    def schema(self):
        return _grouped_schema(_entities(2000, 5, np.random.default_rng(1)))

    def test_a_signature_that_merges_is_kept_when_it_sizes_correctly(self, schema):
        signature = schema["entity_signature"]
        assert signature is not None, "refused on membership before; sizing is the right bar"
        low, high = signature["sizing_error"]
        assert -MAX_UNDERSIZE <= low and high <= MAX_OVERSIZE

    @pytest.mark.parametrize("customers,visits", [(60, 5), (100, 10)])
    def test_keyless_no_drift_stays_quiet(self, schema, customers, visits):
        assert _flagged(schema, IMPORTANCES, lambda r: _entities(customers, visits, r), seed=visits) <= 0.07

    def test_keyless_power_matches_keyed(self, schema):
        make = lambda r: _entities(60, 5, r, shift=0.5)  # noqa: E731
        assert _flagged(schema, IMPORTANCES, make, seed=21) >= 0.7


class TestAnonymousEntities:
    @pytest.fixture(scope="class")
    def schema(self):
        return _grouped_schema(_anonymous(120, 5, np.random.default_rng(3)))

    def test_no_signature_but_the_training_design_is_stored(self, schema):
        assert schema["entity_signature"] is None
        assert schema["entity_design"]["mean_entity_size"] == pytest.approx(5.0)
        assert schema["columns"]["measure_a"]["reference"]["icc_mean"] > 0.8

    def test_keyless_windows_like_training_stay_quiet(self, schema):
        assert _flagged(schema, ANONYMOUS_IMPORTANCES, lambda r: _anonymous(60, 5, r), seed=35) <= 0.07

    def test_keyless_windows_with_more_visits_improve(self, schema):
        assert _flagged(schema, ANONYMOUS_IMPORTANCES, lambda r: _anonymous(30, 10, r), seed=40) <= 0.17

    def test_the_report_states_the_assumption(self, schema):
        report = check_data_drift(_anonymous(60, 5, np.random.default_rng(9)).drop(columns=[KEY]), schema,
                                  ANONYMOUS_IMPORTANCES)
        assert any("rows per entity" in note and "assume" in note for note in report.notes), report.notes

    def test_keyless_power_is_kept(self, schema):
        make = lambda r: _anonymous(60, 5, r, shift=0.5)  # noqa: E731
        assert _flagged(schema, ANONYMOUS_IMPORTANCES, make, seed=45) >= 0.65


class TestNoHoldout:
    @pytest.fixture(scope="class")
    def training(self):
        return _entities(120, 5, np.random.default_rng(1))

    @pytest.fixture(scope="class")
    def old(self, training):
        return _stripped(_grouped_schema(training))

    def test_sizes_come_from_the_dataset(self, training, old):
        estimated = with_estimated_reference_sizes(old, reference_data=training)
        truth = _grouped_schema(training)
        for column in ("measure_a", "device_fingerprint", "home_region", "measure_c"):
            assert (estimated["columns"][column]["reference"]["n_effective"]
                    == pytest.approx(truth["columns"][column]["reference"]["n_effective"], rel=0.05))
        assert estimated["entity_signature"] is not None
        assert "reference data" in estimated["reference_sizes_estimated_from"]

    @pytest.mark.parametrize("keyed", [True, False])
    def test_estimated_from_the_dataset_behaves_like_the_stored_sizes(self, training, old, keyed):
        """Not just quiet: the same verdicts the artifact would give had it stored its sizes."""
        estimated = with_estimated_reference_sizes(old, reference_data=training)
        stored = _grouped_schema(training)
        make = lambda r: _entities(60, 5, r)  # noqa: E731
        assert _flagged(estimated, IMPORTANCES, make, seed=50, keyed=keyed) == \
            _flagged(stored, IMPORTANCES, make, seed=50, keyed=keyed)
        assert _flagged(estimated, IMPORTANCES, make, seed=51, n=60, keyed=keyed) <= 0.07

    def test_with_nothing_but_a_keyed_window_it_sizes_from_the_window(self, old):
        assert _flagged(old, IMPORTANCES, lambda r: _entities(60, 5, r), seed=55, keyed=True) <= 0.07
        report = check_data_drift(_entities(60, 5, np.random.default_rng(56)), old, IMPORTANCES)
        assert any("this window's own keyed rows" in note for note in report.notes), report.notes
