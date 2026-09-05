"""
Leakage detection tests — planted leaks, asserted caught.

The whole value proposition of the leakage module is a promise: "I would
have caught that." Before these tests existed that promise rested on
having run it by hand once. Each test here plants a specific, known class
of leak and asserts it is flagged; the last one asserts a clean dataset
produces no critical flags, because a detector that flags everything is
exactly as useless as one that flags nothing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from autoeng.leakage.detector import (
    check_temporal_split, check_train_test_row_overlap, merge_reports,
    scan_post_training, scan_pre_training,
)
from autoeng.profiling.profiler import profile_dataset


def _flags_of_kind(report, kind: str):
    return [f for f in report.flags if f.kind == kind]


def test_detects_target_proxy_column(leaky_classification_df: pd.DataFrame):
    profile = profile_dataset(leaky_classification_df)
    features = [c for c in leaky_classification_df.columns if c != "churned"]
    report = scan_pre_training(leaky_classification_df, profile, "churned", features)

    flags = _flags_of_kind(report, "target_derived_feature")
    assert flags, "a 99%-accurate copy of the target must be flagged as leakage"
    assert flags[0].columns == ["internal_risk_flag"]
    assert flags[0].severity == "critical"
    assert report.has_critical


def test_clean_dataset_has_no_critical_leakage_flags(clean_classification_df: pd.DataFrame):
    profile = profile_dataset(clean_classification_df)
    features = [c for c in clean_classification_df.columns if c != "churned"]
    report = scan_pre_training(clean_classification_df, profile, "churned", features)

    assert not report.has_critical, (
        f"clean data must not raise critical leakage flags, got: "
        f"{[f.description for f in report.flags if f.severity == 'critical']}"
    )


def test_detects_rows_shared_between_train_and_test(clean_classification_df: pd.DataFrame):
    train = clean_classification_df.iloc[:200]
    test = pd.concat([clean_classification_df.iloc[200:], clean_classification_df.iloc[:5]])

    report = check_train_test_row_overlap(train, test)
    flags = _flags_of_kind(report, "train_test_row_overlap")
    assert flags, "rows present in both splits must be flagged"
    assert flags[0].evidence["n_overlapping_rows"] >= 5


def test_no_overlap_flag_for_disjoint_splits(clean_classification_df: pd.DataFrame):
    train = clean_classification_df.iloc[:200]
    test = clean_classification_df.iloc[200:]
    assert not check_train_test_row_overlap(train, test).flags


def test_detects_temporal_leakage_from_shuffled_split(timeseries_df: pd.DataFrame):
    shuffled = timeseries_df.sample(frac=1.0, random_state=0)
    train, test = shuffled.iloc[:150], shuffled.iloc[150:]

    report = check_temporal_split(train, test, "date")
    flags = _flags_of_kind(report, "temporal_leakage")
    assert flags, "a random split on time-series data must be flagged as temporal leakage"
    assert flags[0].severity == "critical"


def test_chronological_split_is_not_flagged(timeseries_df: pd.DataFrame):
    ordered = timeseries_df.sort_values("date")
    train, test = ordered.iloc[:150], ordered.iloc[150:]
    assert not check_temporal_split(train, test, "date").flags, (
        "a correctly ordered split must not be flagged"
    )


@pytest.mark.parametrize("metric,value", [("roc_auc", 0.999), ("accuracy", 0.998), ("r2", 0.9999)])
def test_flags_suspiciously_perfect_performance(metric: str, value: float):
    report = scan_post_training(metric, value, {"some_feature": 1.0, "other": 0.5})
    assert _flags_of_kind(report, "suspiciously_perfect_performance")


def test_names_the_dominant_feature_when_one_carries_the_model():
    report = scan_post_training("roc_auc", 0.999, {"leaked_col": 95.0, "age": 1.0, "income": 0.5})
    flag = _flags_of_kind(report, "suspiciously_perfect_performance")[0]
    assert flag.evidence["dominant_feature"] == "leaked_col"
    assert flag.evidence["dominant_feature_share"] > 0.85
    assert "leaked_col" in flag.description


def test_ordinary_performance_is_not_flagged():
    assert not scan_post_training("roc_auc", 0.82, {"a": 1.0, "b": 0.9}).flags


def test_merge_reports_preserves_all_flags(leaky_classification_df: pd.DataFrame):
    profile = profile_dataset(leaky_classification_df)
    features = [c for c in leaky_classification_df.columns if c != "churned"]
    pre = scan_pre_training(leaky_classification_df, profile, "churned", features)
    post = scan_post_training("roc_auc", 0.999, {"internal_risk_flag": 99.0, "age": 1.0})

    merged = merge_reports(pre, post)
    assert len(merged.flags) == len(pre.flags) + len(post.flags)
    assert merged.has_critical
