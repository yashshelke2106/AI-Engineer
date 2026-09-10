"""
T1-3 — drift detection.

The ROADMAP's three "done when" clauses are the first three tests here, and
the third is the one that makes this item worth doing carefully:

  1. a deliberately shifted stream alarms
  2. an unshifted one stays quiet across a long window
  3. a shift confined to a near-zero-importance feature is *reported without
     alarming*

Clause 3 is the whole thesis. **Drift is not degradation.** A feature the model
barely uses can move enormously and change nothing; the dominant feature can
shift slightly and break everything. A detector that treats those the same
produces alarms nobody trusts, and an alarm nobody trusts is worse than no
alarm — it costs attention and buys nothing. So per-feature drift is weighted
by the importances the explain stage already computed, and both the raw and
the weighted view are reported.

Clause 2 is the quiet one that catches over-eager detectors: run enough
features past enough windows and something will look shifted by chance. That
is what the multiple-testing correction is for, and the test uses many
features precisely so an uncorrected detector fails it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.monitoring.drift import (
    PSI_ALARM, PSI_INVESTIGATE, DriftSeverity, check_concept_drift,
    check_data_drift, check_prediction_drift, population_stability_index,
    raw_column_importances,
)
from autoeng.registry.model_store import build_training_schema
from autoeng.common.roles import assign_feature_roles
from autoeng.profiling.profiler import profile_dataset


def _training_frame(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": rng.lognormal(4.0, 0.8, n).round(2),
        "tenure": rng.normal(500, 120, n).round(1),
        "noise_a": rng.normal(0, 1, n).round(3),
        "noise_b": rng.normal(0, 1, n).round(3),
        "noise_c": rng.normal(0, 1, n).round(3),
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "y": rng.integers(0, 2, n),
    })


class _Estimator:
    """Minimal stand-in: build_training_schema only reads `classes_`."""
    classes_ = np.array([0, 1])


def _schema(df: pd.DataFrame) -> dict:
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y")
    return build_training_schema(
        _Estimator(), df[roles.feature_columns], df["y"], profile, roles,
        problem_type="binary_classification", model_name="test",
    )


@pytest.fixture
def schema():
    return _schema(_training_frame())


@pytest.fixture
def importances():
    """`amount` dominates; the noise columns are effectively unused."""
    return {"amount": 0.70, "tenure": 0.25, "region": 0.04,
            "noise_a": 0.003, "noise_b": 0.003, "noise_c": 0.004}


def _live(n=800, seed=1, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "amount": rng.lognormal(4.0, 0.8, n).round(2),
        "tenure": rng.normal(500, 120, n).round(1),
        "noise_a": rng.normal(0, 1, n).round(3),
        "noise_b": rng.normal(0, 1, n).round(3),
        "noise_c": rng.normal(0, 1, n).round(3),
        "region": rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.3, 0.2, 0.1]),
    })
    for column, values in overrides.items():
        frame[column] = values
    return frame


class TestTheThreeDoneWhens:
    def test_a_shifted_stream_alarms(self, schema, importances):
        rng = np.random.default_rng(2)
        # `amount` is the dominant feature and it has moved substantially.
        shifted = _live(amount=rng.lognormal(5.2, 0.8, 800).round(2))
        report = check_data_drift(shifted, schema, importances)

        assert report.severity == DriftSeverity.ALARM
        amount = next(f for f in report.features if f.column == "amount")
        assert amount.psi > PSI_ALARM
        assert amount.severity == DriftSeverity.ALARM

    def test_an_unshifted_stream_stays_quiet(self, schema, importances):
        """Across many windows and six features, an uncorrected detector will
        find something 'significant' by chance. That is the failure this
        guards."""
        alarms = 0
        for seed in range(12):
            report = check_data_drift(_live(seed=100 + seed), schema, importances)
            assert report.severity != DriftSeverity.ALARM, (
                f"window {seed} alarmed on unshifted data: "
                f"{[f.column for f in report.features if f.severity == DriftSeverity.ALARM]}"
            )
            alarms += sum(1 for f in report.features if f.severity != DriftSeverity.OK)
        assert alarms == 0, f"{alarms} spurious per-feature flags across 12 clean windows"

    def test_drift_in_an_unimportant_feature_is_reported_without_alarming(
        self, schema, importances,
    ):
        """The thesis. `noise_b` carries 0.3% of the model's importance; it can
        move as far as it likes without the model caring."""
        rng = np.random.default_rng(3)
        shifted = _live(noise_b=rng.normal(6.0, 1.0, 800).round(3))
        report = check_data_drift(shifted, schema, importances)

        noise_b = next(f for f in report.features if f.column == "noise_b")
        assert noise_b.psi > PSI_ALARM, "the raw shift must still be measured and shown"
        assert noise_b.severity == DriftSeverity.ALARM, "raw severity is per-feature and honest"
        # ...but the report as a whole must not cry wolf.
        assert report.severity != DriftSeverity.ALARM
        assert report.weighted_psi < PSI_INVESTIGATE
        assert "noise_b" in report.summary, "it is reported, not hidden"


class TestWeighting:
    def test_the_same_shift_matters_more_in_an_important_feature(self, schema, importances):
        rng = np.random.default_rng(4)
        moved = rng.normal(6.0, 1.0, 800).round(3)

        unimportant = check_data_drift(_live(noise_b=moved), schema, importances)
        important = check_data_drift(
            _live(tenure=rng.normal(1400, 120, 800).round(1)), schema, importances,
        )
        assert important.weighted_psi > unimportant.weighted_psi * 5
        assert important.severity == DriftSeverity.ALARM

    def test_missing_importances_fall_back_to_uniform_and_say_so(self, schema):
        rng = np.random.default_rng(5)
        report = check_data_drift(_live(noise_b=rng.normal(6.0, 1, 800)), schema, importances=None)
        assert report.weighted_psi > 0
        assert any("uniform" in n.lower() for n in report.notes), report.notes


class TestPSI:
    def test_identical_distributions_score_zero(self, schema):
        frame = _training_frame().drop(columns=["y"])
        report = check_data_drift(frame, schema, importances=None)
        for feature in report.features:
            assert feature.psi < 0.02, f"{feature.column} drifted against its own training data"

    def test_psi_is_symmetric_in_the_sense_that_matters(self):
        reference = {"a": 0.5, "b": 0.5}
        observed = {"a": 0.9, "b": 0.1}
        assert population_stability_index(reference, observed) > PSI_ALARM

    def test_an_unseen_category_is_drift_not_a_crash(self, schema):
        live = _live()
        live["region"] = "a_region_that_did_not_exist"
        report = check_data_drift(live, schema, importances=None)
        region = next(f for f in report.features if f.column == "region")
        assert np.isfinite(region.psi) and region.psi > PSI_ALARM
        assert "a_region_that_did_not_exist" in region.detail


class TestPredictionDrift:
    def test_a_shifted_output_distribution_is_caught(self, schema):
        """Catches what per-feature checks miss: features can each look fine
        while their combination moves the model's output."""
        reference = np.concatenate([np.full(900, 0.05), np.full(100, 0.9)])
        shifted = np.concatenate([np.full(400, 0.05), np.full(600, 0.9)])
        report = check_prediction_drift(shifted, reference)
        assert report.severity == DriftSeverity.ALARM
        assert report.psi > PSI_ALARM

    def test_a_stable_output_distribution_is_quiet(self):
        rng = np.random.default_rng(6)
        reference = rng.beta(2, 8, 2000)
        report = check_prediction_drift(rng.beta(2, 8, 800), reference)
        assert report.severity == DriftSeverity.OK


class TestConceptDrift:
    def test_degraded_live_performance_alarms(self):
        """The only check that measures what actually matters — and the only
        one that needs labels."""
        rng = np.random.default_rng(7)
        n = 400
        frame = pd.DataFrame({
            "prediction": rng.integers(0, 2, n),
            "actual": rng.integers(0, 2, n),  # predictions now uncorrelated with truth
        })
        report = check_concept_drift(frame, baseline={"f1": 0.82, "precision": 0.8, "recall": 0.85})
        assert report.severity == DriftSeverity.ALARM
        assert report.observed["f1"] < 0.6

    def test_matching_live_performance_is_quiet(self):
        frame = pd.DataFrame({"prediction": [1] * 80 + [0] * 20, "actual": [1] * 78 + [0] * 22})
        report = check_concept_drift(frame, baseline={"f1": 0.9, "precision": 0.9, "recall": 0.95})
        assert report.severity == DriftSeverity.OK

    def test_too_few_labels_is_reported_as_unknown_not_as_healthy(self):
        """Silence because nothing arrived is not the same as silence because
        nothing is wrong, and conflating them is how a broken label pipeline
        looks like a passing check."""
        frame = pd.DataFrame({"prediction": [1, 0], "actual": [1, 0]})
        report = check_concept_drift(frame, baseline={"f1": 0.9})
        assert report.severity == DriftSeverity.UNKNOWN
        assert "labelled" in report.summary.lower()


class TestImportanceMapping:
    def test_one_hot_importances_fold_back_onto_their_source_column(self):
        """SHAP explains the TRANSFORMED matrix; drift is measured on raw
        columns. Without folding `region_north`/`region_south` back onto
        `region`, the model's most-used categorical looks unused."""
        importances = raw_column_importances(
            [("region_north", 0.2), ("region_south", 0.1), ("amount", 0.6), ("tenure", 0.1)],
            feature_columns=["amount", "tenure", "region"],
        )
        assert importances["region"] == pytest.approx(0.3)
        assert importances["amount"] == pytest.approx(0.6)
        assert sum(importances.values()) == pytest.approx(1.0)

    def test_unmatched_importance_names_do_not_silently_vanish(self):
        importances = raw_column_importances(
            [("a_derived_interaction", 0.5), ("amount", 0.5)],
            feature_columns=["amount", "tenure"],
        )
        # Normalised over what could be attributed; `amount` keeps its share
        # rather than being inflated to 1.0 by a dropped name.
        assert importances["amount"] < 1.0
        assert importances["tenure"] == 0.0
