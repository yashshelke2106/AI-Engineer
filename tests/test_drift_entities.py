"""
Drift on grouped data when the drift check cannot see the entity key.

Two cases over-read, measured on no-drift windows against a 120-customer
reference (share flagged investigate or worse):

                                   60 customers x 5   30 customers x 10
    payloads without the key            42%                83%
    artifact without stored sizes       77%                97%   (no key)
                                         5% / 13% with the key

Both now learn what they are missing from data where the key IS known:

  - an ENTITY SIGNATURE (feature columns constant per entity and distinctive
    across them) is learned from the training rows and kept only if it
    reproduces the true entities with at most 5% split or merged; a keyless
    window's rows are grouped by it;
  - an older artifact's reference sizes and signature are estimated from its
    frozen holdout, which carries the key.

No validated signature means no recovery, and the report still says so.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from autoeng.common.entities import MAX_OVERSIZE, MAX_UNDERSIZE, learn_entity_signature, recover_entities
from autoeng.monitoring.drift import DriftSeverity, check_data_drift, with_estimated_reference_sizes
from autoeng.monitoring.report import run_drift_report
from autoeng.serving.store import PredictionStore
from tests.test_drift_noise import IMPORTANCES, KEY, _entities, _grouped_schema


@pytest.fixture(scope="module")
def reference_frame() -> pd.DataFrame:
    return _entities(120, 5, np.random.default_rng(1))


@pytest.fixture(scope="module")
def schema(reference_frame) -> dict:
    return _grouped_schema(reference_frame)


def _stripped(schema: dict) -> dict:
    """What an artifact written before effective sizes looks like."""
    old = copy.deepcopy(schema)
    for meta in old["columns"].values():
        for key in ("n_effective", "n_effective_mean"):
            meta["reference"].pop(key, None)
    old.pop("entity_signature", None)
    return old


def _flagged(schema, customers, visits, seed, shift=0.0, keyed=False, n=30) -> int:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        window = _entities(customers, visits, rng, shift=shift)
        if not keyed:
            window = window.drop(columns=[KEY])
        count += check_data_drift(window, schema, IMPORTANCES).severity != DriftSeverity.OK
    return count


class TestTheSignature:
    def test_it_is_the_entity_constant_columns_and_it_is_validated(self, schema):
        signature = schema["entity_signature"]
        assert signature is not None
        assert set(signature["columns"]) <= {"device_fingerprint", "measure_a", "home_region"}
        assert "measure_c" not in signature["columns"], "a per-visit reading identifies nobody"
        low, high = signature["sizing_error"]
        assert -MAX_UNDERSIZE <= low and high <= MAX_OVERSIZE
        assert signature["split_rate"] <= 0.05 and signature["merge_rate"] <= 0.05

    def test_no_signature_is_learned_where_nothing_identifies_the_entity(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(100), 5)
        frame = pd.DataFrame({"a": rng.normal(size=500), "b": rng.choice(["x", "y"], 500)})
        assert learn_entity_signature(frame, groups, ["a", "b"]) is None

    def test_a_region_alone_is_not_a_signature(self):
        """Constant per customer, but five values cannot tell 100 customers apart."""
        frame = _entities(100, 5, np.random.default_rng(4))
        assert learn_entity_signature(frame, frame[KEY].to_numpy(), ["home_region"]) is None

    def test_recovery_on_a_fresh_window(self, schema):
        window = _entities(60, 5, np.random.default_rng(5))
        recovered = recover_entities(window, schema["entity_signature"])
        pairs = pd.DataFrame({"true": window[KEY], "recovered": recovered})
        assert (pairs.groupby("true")["recovered"].nunique() == 1).mean() >= 0.95
        assert (pairs.groupby("recovered")["true"].nunique() == 1).mean() >= 0.95

    def test_independent_rows_are_not_merged(self, schema):
        window = _entities(300, 1, np.random.default_rng(6))
        assert len(set(recover_entities(window, schema["entity_signature"]))) >= 285

    def test_a_window_missing_a_signature_column_recovers_nothing(self, schema):
        window = _entities(60, 5, np.random.default_rng(7)).drop(columns=[schema["entity_signature"]["columns"][0]])
        assert recover_entities(window, schema["entity_signature"]) is None


class TestKeylessWindows:
    @pytest.mark.parametrize("customers,visits", [(60, 5), (30, 10)])
    def test_no_drift_stays_quiet(self, schema, customers, visits):
        assert _flagged(schema, customers, visits, seed=20 + visits) <= 2

    def test_the_report_says_entities_were_recovered(self, schema):
        window = _entities(60, 5, np.random.default_rng(30)).drop(columns=[KEY])
        report = check_data_drift(window, schema, IMPORTANCES)
        assert any("recovered" in note for note in report.notes), report.notes

    def test_a_one_sd_shift_still_alarms(self, schema):
        rng = np.random.default_rng(31)
        for _ in range(10):
            window = _entities(60, 5, rng, shift=1.0).drop(columns=[KEY])
            assert check_data_drift(window, schema, IMPORTANCES).severity == DriftSeverity.ALARM


class TestOlderArtifacts:
    @pytest.fixture(scope="class")
    def estimated(self, schema):
        holdout = _entities(30, 5, np.random.default_rng(2))
        return with_estimated_reference_sizes(_stripped(schema), holdout)

    def test_sizes_are_estimated_close_to_the_stored_ones(self, schema, estimated):
        for column in ("measure_a", "device_fingerprint", "measure_c", "home_region"):
            truth = schema["columns"][column]["reference"]["n_effective"]
            guess = estimated["columns"][column]["reference"]["n_effective"]
            assert guess == pytest.approx(truth, rel=0.35), column
        assert estimated["entity_signature"] is not None

    @pytest.mark.parametrize("keyed", [True, False])
    def test_no_drift_stays_quiet(self, estimated, keyed):
        assert _flagged(estimated, 60, 5, seed=40, keyed=keyed) <= 2
        assert _flagged(estimated, 300, 1, seed=41, keyed=keyed) <= 2

    def test_without_a_holdout_it_still_says_so(self, schema):
        report = check_data_drift(_entities(60, 5, np.random.default_rng(42)).drop(columns=[KEY]),
                                  _stripped(schema), IMPORTANCES)
        assert any("predates effective sample sizes" in note for note in report.notes), report.notes


def test_the_report_estimates_from_the_holdout_and_recovers_keyless_traffic(tmp_path, schema):
    store = PredictionStore(tmp_path / "log.db")
    window = _entities(60, 5, np.random.default_rng(50))
    for _, row in window.iterrows():
        store.log_prediction(payload={c: row[c] for c in IMPORTANCES}, prediction=0, model_version="m")
    old = _stripped(schema)
    holdout = _entities(30, 5, np.random.default_rng(2))

    report = run_drift_report(store, old, holdout=holdout)
    assert report.data.severity == DriftSeverity.OK, report.data.summary
    notes = " ".join(report.data.notes + report.notes)
    assert "holdout" in notes and "recovered" in notes
