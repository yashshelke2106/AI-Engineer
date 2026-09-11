"""
T1-4 — retrain orchestration.

Two things this has to get right, and one of them is easy to get wrong in a
way that only shows up months later.

**The target is pinned, never re-detected.** Detection is a heuristic guess at
human intent. Re-running it on each retrain means the system can quietly
decide it is now predicting a different column — the same pipeline, the same
name, a different question. Nothing about the run would look wrong. So the
champion's schema supplies `--target` and `--problem-type` to the challenger,
and there is a test that plants a decoy column which would win detection on a
retraining frame in order to prove the pin holds.

**A retrain needs new labelled data to be worth running.** Re-fitting on the
original file plus nothing is how a demo appears to have a lifecycle: it runs
on schedule, produces a model, logs a green check, and has learned nothing
since the first day. The trigger requires labels that did not exist before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeng.lifecycle.retrain import (
    MIN_NEW_LABELS, RetrainTrigger, build_retraining_frame, should_retrain,
)
from autoeng.monitoring.drift import DriftSeverity
from autoeng.monitoring.report import DriftReport
from autoeng.serving.store import PredictionStore


@pytest.fixture
def original(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "account_id": [f"A{i:04d}" for i in range(n)],
        "amount": rng.lognormal(4, 0.8, n).round(2),
        "tenure": rng.normal(500, 120, n).round(1),
        "is_fraud": rng.integers(0, 2, n),
    })
    path = tmp_path / "original.csv"
    df.to_csv(path, index=False)
    return path, df


@pytest.fixture
def schema():
    return {
        "feature_columns": ["amount", "tenure"],
        "target": {"column": "is_fraud", "class_labels": [0, 1]},
        "problem_type": "binary_classification",
        "feature_roles": {"group_column": None},
    }


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "log.db")


def _serve_and_label(store, n, seed=1, labelled=True):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        request_id = store.log_prediction(
            payload={"amount": float(rng.lognormal(4, 0.8)), "tenure": float(rng.normal(500, 120))},
            prediction=int(rng.integers(0, 2)), probability=float(rng.beta(2, 5)),
            model_version="champion", model_name="test",
        )
        if labelled:
            store.record_outcome(request_id, actual=int(rng.integers(0, 2)))


def _report(severity: DriftSeverity) -> DriftReport:
    return DriftReport(model_version="champion", window_rows=500, severity=severity,
                       summary="test")


class TestTheTrigger:
    def test_a_drift_alarm_with_enough_labels_triggers(self, store):
        _serve_and_label(store, MIN_NEW_LABELS + 10)
        decision = should_retrain(_report(DriftSeverity.ALARM), store)
        assert decision.triggered
        assert decision.trigger == RetrainTrigger.DRIFT_ALARM

    def test_an_alarm_without_new_labels_does_not(self, store):
        """Re-fitting on the original file plus nothing produces a model that
        has learned nothing — while looking exactly like a working lifecycle."""
        _serve_and_label(store, 200, labelled=False)
        decision = should_retrain(_report(DriftSeverity.ALARM), store)
        assert not decision.triggered
        assert "labelled" in decision.reason.lower()

    def test_a_quiet_report_does_not_trigger(self, store):
        _serve_and_label(store, 500)
        assert not should_retrain(_report(DriftSeverity.OK), store).triggered

    def test_unknown_does_not_trigger_but_says_why(self, store):
        """UNKNOWN means the check could not run. Retraining on the strength of
        a check that did not happen is worse than waiting."""
        _serve_and_label(store, 500)
        decision = should_retrain(_report(DriftSeverity.UNKNOWN), store)
        assert not decision.triggered
        assert "unknown" in decision.reason.lower()

    def test_a_schedule_triggers_without_an_alarm_if_labels_arrived(self, store):
        _serve_and_label(store, MIN_NEW_LABELS + 10)
        decision = should_retrain(_report(DriftSeverity.OK), store, scheduled=True)
        assert decision.triggered
        assert decision.trigger == RetrainTrigger.SCHEDULED


class TestTheRetrainingFrame:
    def test_new_labelled_rows_are_appended_to_the_original(self, original, store, schema):
        path, df = original
        _serve_and_label(store, 60)
        frame, report = build_retraining_frame(path, store, schema)

        assert len(frame) == len(df) + 60
        assert report["n_original_rows"] == len(df)
        assert report["n_new_rows"] == 60
        assert frame["is_fraud"].notna().all(), "every row must carry the target"

    def test_the_outcome_becomes_the_target_column(self, original, store, schema):
        path, _ = original
        _serve_and_label(store, 60)
        frame, _ = build_retraining_frame(path, store, schema)
        # The store calls it `actual`; the pipeline needs it under the target's
        # own name or detection would have nothing to pin to.
        assert "actual" not in frame.columns
        assert set(frame["is_fraud"].dropna().unique()) <= {0, 1}

    def test_columns_absent_from_served_payloads_are_reported(self, original, store, schema):
        """`account_id` exists in the original file but is never sent to
        /predict, so new rows cannot have one. Silently filling it would invent
        data; dropping the column silently would change the dataset shape
        between generations."""
        path, _ = original
        _serve_and_label(store, 60)
        frame, report = build_retraining_frame(path, store, schema)
        assert "account_id" in report["columns_missing_from_new_rows"]
        assert frame["account_id"].isna().sum() == 60

    def test_refuses_when_no_new_labels_exist(self, original, store, schema):
        path, _ = original
        with pytest.raises(ValueError, match="no new labelled"):
            build_retraining_frame(path, store, schema)


class TestThePinHolds:
    def test_the_target_comes_from_the_champion_not_from_detection(self, tmp_path, store, schema):
        """The failure this prevents: a retraining frame where some other
        column now scores better as a target, so the system silently starts
        predicting a different thing under the same name."""
        from autoeng.detection.problem_detector import detect_problem_type
        from autoeng.lifecycle.retrain import pinned_overrides
        from autoeng.profiling.profiler import profile_dataset

        rng = np.random.default_rng(2)
        n = 500
        df = pd.DataFrame({
            "amount": rng.lognormal(4, 0.8, n).round(2),
            "tenure": rng.normal(500, 120, n).round(1),
            # A decoy with a target-shaped name and perfect balance.
            "target": rng.integers(0, 2, n),
            "is_fraud": rng.integers(0, 2, n),
        })
        detected = detect_problem_type(df, profile_dataset(df)).chosen.target_column
        assert detected != "is_fraud", (
            "fixture must actually tempt detection away, or the pin proves nothing"
        )

        overrides = pinned_overrides(schema)
        assert overrides["target_override"] == "is_fraud"
        assert overrides["problem_type_override"] == "binary_classification"

    def test_the_group_column_is_pinned_too(self):
        from autoeng.lifecycle.retrain import pinned_overrides
        overrides = pinned_overrides({
            "target": {"column": "y"}, "problem_type": "regression",
            "feature_roles": {"group_column": "patient_id"},
        })
        assert overrides["group_column_override"] == "patient_id"

    def test_a_real_artifact_carries_the_group_decision(self):
        """The pin test above passed against a hand-built dict while real
        artifacts never stored `group_column` at all — T0-1 serialised the
        roles before T0-3 added the field, so every champion silently read as
        ungrouped. Built through build_training_schema so this fails if the
        artifact ever stops carrying it again."""
        from autoeng.common.roles import assign_feature_roles
        from autoeng.lifecycle.retrain import pinned_overrides
        from autoeng.profiling.profiler import profile_dataset
        from autoeng.registry.model_store import build_training_schema

        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "patient_id": np.repeat([f"P{i:03d}" for i in range(60)], 5),
            "reading": rng.normal(size=300).round(3),
            "y": rng.integers(0, 2, 300),
        })
        profile = profile_dataset(df)
        roles = assign_feature_roles(profile, target_column="y", group_column="patient_id")

        class _Estimator:
            classes_ = np.array([0, 1])

        schema = build_training_schema(
            _Estimator(), df[roles.feature_columns], df["y"], profile, roles,
            problem_type="binary_classification", model_name="test",
        )
        assert schema["feature_roles"]["group_column"] == "patient_id"

        overrides = pinned_overrides(schema)
        assert overrides["group_column_override"] == "patient_id"
        assert overrides["use_groups"] is True

    def test_no_grouping_is_pinned_as_no_grouping(self):
        """Leaving detection on for an ungrouped champion lets the retraining
        frame choose a split scheme the champion never used."""
        from autoeng.lifecycle.retrain import pinned_overrides
        overrides = pinned_overrides({
            "target": {"column": "y"}, "problem_type": "regression",
            "feature_roles": {"group_column": None},
        })
        assert overrides["use_groups"] is False

    def test_an_artifact_predating_grouping_is_left_to_detect(self):
        """A missing key is unknown, not 'no groups' — pinning it to disabled
        would switch off a leakage guard on data that might need it."""
        from autoeng.lifecycle.retrain import pinned_overrides
        overrides = pinned_overrides({
            "target": {"column": "y"}, "problem_type": "regression", "feature_roles": {},
        })
        assert "use_groups" not in overrides
        assert "group_column_override" not in overrides

    def test_a_schema_without_a_target_refuses_rather_than_guessing(self):
        from autoeng.lifecycle.retrain import pinned_overrides
        with pytest.raises(ValueError, match="target"):
            pinned_overrides({"target": {}, "problem_type": "clustering"})
