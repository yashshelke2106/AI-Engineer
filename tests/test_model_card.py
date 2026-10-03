"""
T2-6 — a model card per run, built only from what the run measured.

The card's value is in two places a report does not reach: results per segment
(an aggregate can hide a segment the model fails on), and limitations pulled from
the run's own checks rather than written as boilerplate. So these tests plant a
segment the model genuinely fails on and check it is flagged; plant a segment
that differs only by noise and check it is NOT; and feed each detected problem in
and check the card says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.pipeline import _card_segments, _role_dict, _write_model_card
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import save_model
from autoeng.reporting.model_card import (
    MIN_SEGMENT_ROWS, build_model_card, render_model_card, segment_results,
)


def _channel_frame(n: int, seed: int, broken_channel: str | None) -> pd.DataFrame:
    """`x` decides the label on every channel except `broken_channel`, where the
    label is a coin flip the model cannot learn."""
    rng = np.random.default_rng(seed)
    channel = rng.choice(["web", "store", "phone"], n, p=[0.4, 0.35, 0.25])
    x = rng.normal(size=n)
    label = (x + rng.normal(0, 0.6, n) > 0).astype(int)
    if broken_channel:
        broken = channel == broken_channel
        label[broken] = rng.integers(0, 2, broken.sum())
    return pd.DataFrame({"x": x, "channel": channel, "label": label})


def _fit(frame: pd.DataFrame):
    X = pd.get_dummies(frame[["x", "channel"]], columns=["channel"], dtype=float)
    return LogisticRegression(max_iter=1000).fit(X, frame["label"]), X.columns


class _Frame2Model:
    """Wraps a model fitted on dummies so it takes the raw frame, as a pipeline would."""

    def __init__(self, model, columns):
        self.model, self.columns = model, columns

    def _x(self, frame):
        return pd.get_dummies(frame[["x", "channel"]], columns=["channel"], dtype=float) \
            .reindex(columns=self.columns, fill_value=0.0)

    def predict_proba(self, frame):
        return self.model.predict_proba(self._x(frame))

    def predict(self, frame):
        return self.model.predict(self._x(frame))


def _segments(train, test, **kw):
    model = _Frame2Model(*_fit(train))
    return segment_results(model, test[["x", "channel"]], test["label"], problem_kind="binary",
                           candidate_columns=["channel"], numeric_columns=[], **kw)


class TestSegments:
    def test_a_segment_the_model_fails_on_is_flagged(self):
        train = _channel_frame(3000, 0, broken_channel="phone")
        test = _channel_frame(1500, 1, broken_channel="phone")
        result = _segments(train, test)
        by_name = {s["segment"]: s for s in result["segments"]}
        assert by_name["phone"]["flag"] == "worse than overall beyond noise"
        assert by_name["phone"]["value"] < 0.65 < result["overall"]
        assert by_name["web"]["flag"] is None and by_name["store"]["flag"] is None

    def test_segments_that_differ_only_by_noise_are_not_flagged(self):
        train = _channel_frame(3000, 2, broken_channel=None)
        test = _channel_frame(600, 3, broken_channel=None)
        result = _segments(train, test)
        assert all(s["flag"] is None for s in result["segments"]), \
            [(s["segment"], s["value"], s["se"]) for s in result["segments"]]

    def test_a_small_segment_gets_a_count_not_a_number(self):
        train = _channel_frame(2000, 4, broken_channel=None)
        test = _channel_frame(400, 5, broken_channel=None)
        test.loc[test.index[:MIN_SEGMENT_ROWS - 5], "channel"] = "kiosk"
        result = _segments(train, test)
        kiosk = next(s for s in result["segments"] if s["segment"] == "kiosk")
        assert kiosk["value"] is None and "too few" in kiosk["note"]

    def test_a_segment_missing_a_class_is_not_given_a_ranking(self):
        train = _channel_frame(2000, 6, broken_channel=None)
        test = _channel_frame(600, 7, broken_channel=None)
        store = test["channel"] == "store"
        test.loc[store, "label"] = 1
        test.loc[test.index[store.to_numpy()][:3], "label"] = 0
        result = _segments(train, test)
        s = next(s for s in result["segments"] if s["segment"] == "store")
        assert s["value"] is None and "ranking metric would be noise" in s["note"]

    def test_regression_segments_compare_absolute_error(self):
        rng = np.random.default_rng(8)
        n = 2000
        region = rng.choice(["north", "south"], n)
        x = rng.normal(size=n)
        noise = np.where(region == "south", 3.0, 0.3)
        y = 2 * x + rng.normal(0, 1, n) * noise
        frame = pd.DataFrame({"x": x, "region": region})

        class _Linear:
            def __init__(self):
                self.m = LinearRegression().fit(frame[["x"]].iloc[:1500], y[:1500])

            def predict(self, f):
                return self.m.predict(f[["x"]])

        result = segment_results(_Linear(), frame.iloc[1500:], pd.Series(y[1500:]), problem_kind="regression",
                                 candidate_columns=["region"], numeric_columns=[])
        by_name = {s["segment"]: s for s in result["segments"]}
        assert result["metric"] == "mae" and not result["higher_is_better"]
        assert by_name["south"]["flag"] == "worse than overall beyond noise"
        assert by_name["north"]["flag"] is None


def _card(**overrides):
    base = dict(
        run_name="demo", run_id="abc123", dataset_path="data/demo.csv",
        problem_decision={"chosen": {"problem_type": "binary_classification", "target_column": "label"},
                          "alternatives": [], "confidence": 0.9},
        target_source="auto-detected",
        role_assignment={"numeric_columns": ["x"], "categorical_columns": [], "datetime_columns": [],
                         "text_columns": [], "excluded_columns": []},
        profile_summary={"columns": {}}, explanation={"winner_name": "logistic_regression"},
        held_out_metrics={"roc_auc": 0.81}, leaderboard=None,
        threshold_choice={"threshold": 0.42, "objective": "f1", "near_trivial": False},
        held_out_operating_point=None, group_decision={"column": None, "detected_column": None},
        pre_training_leakage={"flags": []}, post_training_leakage={"flags": []},
        model_artifact={"status": "saved", "model_path": "m.joblib"}, segments=None,
        n_train=800, n_test=400, intended_use=None,
    )
    base.update(overrides)
    return build_model_card(**base)


class TestLimitationsComeFromTheRun:
    def test_a_clean_run_says_so_without_claiming_nothing_is_wrong(self):
        text = render_model_card(_card())
        assert "None detected by this run's checks. That is not the same as none existing." in text

    def test_low_detection_confidence_is_named(self):
        card = _card(problem_decision={
            "chosen": {"problem_type": "binary_classification", "target_column": "label"},
            "alternatives": [{"target_column": "other", "problem_type": "regression"}], "confidence": 0.41})
        assert any("confidence 0.41" in n and "other" in n for n in card["limitations"])

    def test_a_target_the_caller_pinned_is_not_second_guessed(self):
        card = _card(target_source="supplied by caller",
                     problem_decision={"chosen": {"problem_type": "regression", "target_column": "y"},
                                       "alternatives": [], "confidence": 0.2})
        assert not any("inferred" in n for n in card["limitations"])

    def test_leakage_flags_are_quoted(self):
        card = _card(pre_training_leakage={"flags": [
            {"severity": "critical", "kind": "target_proxy", "columns": ["refund_issued"],
             "description": "refund_issued is recorded after the outcome"},
            {"severity": "info", "kind": "x", "columns": [], "description": "informational only"}]})
        text = " ".join(card["limitations"])
        assert "refund_issued is recorded after the outcome" in text and "`refund_issued`" in text
        assert "informational only" not in text

    def test_grouping_turned_off_is_the_loudest_warning(self):
        card = _card(group_decision={"column": None, "detected_column": "customer_id"})
        assert any("grouping was turned off" in n and "customer_id" in n for n in card["limitations"])

    def test_a_near_trivial_threshold_is_named(self):
        card = _card(threshold_choice={"threshold": 0.1, "objective": "f1", "near_trivial": True})
        assert any("labels almost everything positive" in n for n in card["limitations"])

    def test_text_features_carry_their_drift_blind_spot(self):
        card = _card(role_assignment={"numeric_columns": [], "categorical_columns": [], "datetime_columns": [],
                                      "text_columns": ["ticket"], "excluded_columns": []})
        assert any("same vocabulary in new proportions" in n for n in card["limitations"])

    def test_an_unpersisted_model_is_stated(self):
        card = _card(model_artifact={"status": "failed", "error": "PicklingError: boom"})
        assert any("No servable model was persisted: PicklingError: boom" in n for n in card["limitations"])

    def test_a_small_holdout_is_named(self):
        card = _card(n_test=90)
        assert any("90 rows" in n for n in card["limitations"])


class TestIntendedUseIsNeverInvented:
    def test_absent_intended_use_says_not_supplied(self):
        text = render_model_card(_card())
        assert "*Not supplied.*" in text and "--intended-use" in text

    def test_supplied_intended_use_is_printed_verbatim(self):
        text = render_model_card(_card(intended_use="Rank support tickets for triage; not for staffing decisions."))
        assert "Rank support tickets for triage; not for staffing decisions." in text


class TestFoundByARealRun:
    """A real run's winner was linear_svc_balanced: no predict_proba. The segments
    section vanished without a word and the card never said no threshold was set."""

    def test_a_model_without_probabilities_is_still_segmented(self):
        from sklearn.svm import LinearSVC

        train = _channel_frame(3000, 10, broken_channel="phone")
        test = _channel_frame(1500, 11, broken_channel="phone")
        model = _Frame2Model(*_fit(train))
        model.model = LinearSVC().fit(model._x(train), train["label"])
        del _Frame2Model.predict_proba  # the shape of the real winner: margins only
        try:
            assert not hasattr(model, "predict_proba")
            model.decision_function = lambda frame: model.model.decision_function(model._x(frame))
            result = segment_results(model, test[["x", "channel"]], test["label"], problem_kind="binary",
                                     candidate_columns=["channel"], numeric_columns=[], threshold=0.5)
        finally:
            _Frame2Model.predict_proba = lambda self, frame: self.model.predict_proba(self._x(frame))
        by_name = {s["segment"]: s for s in result["segments"]}
        assert by_name["phone"]["flag"] == "worse than overall beyond noise"
        assert "precision" not in by_name["web"], "a probability threshold means nothing on a margin"

    def test_a_segment_failure_is_printed_not_dropped(self):
        text = render_model_card(_card(segments={"error": "AttributeError: boom", "segments": []}))
        assert "## Results by segment" in text and "Could not be computed: AttributeError: boom" in text

    def test_a_tuned_winner_shows_its_tuned_cross_validated_score(self, tmp_path):
        schema = tmp_path / "training_schema.json"
        schema.write_text(json.dumps({"model": {"name": "linear_svc_balanced",
                                                "selection_source": "hyperparameter tuning"}}), encoding="utf-8")
        card = _card(
            explanation={"winner_name": "linear_svc_balanced", "winner_score": 0.7007},
            leaderboard={"primary_metric": "roc_auc", "results": [
                {"name": "linear_svc_balanced", "status": "ok", "metrics": {"roc_auc": 0.6866, "accuracy": 0.66}}]},
            held_out_metrics={"roc_auc": 0.7157, "accuracy": 0.6833},
            model_artifact={"status": "saved", "model_path": "m.joblib", "schema_path": str(schema)},
        )
        assert card["evaluation"]["cross_validated"] == {"roc_auc": 0.7007}, "not the untuned leaderboard row"
        text = render_model_card(card)
        assert "| roc_auc | 0.7157 | 0.7007 |" in text and "| accuracy | 0.6833 | — |" in text
        assert "Tuning re-scored only the primary metric" in text

    def test_a_binary_model_without_a_threshold_says_so(self):
        card = _card(threshold_choice=None)
        assert any("No decision threshold was selected" in n for n in card["limitations"])
        regression = _card(problem_decision={"chosen": {"problem_type": "regression", "target_column": "y"},
                                             "alternatives": [], "confidence": 0.9})
        assert not any("No decision threshold" in n for n in regression["limitations"])


def test_the_card_travels_with_a_real_artifact(tmp_path, clean_classification_df):
    """Built through save_model, not a hand-written dict (see CLAUDE.md)."""
    frame = clean_classification_df
    profile = profile_dataset(frame)
    roles = assign_feature_roles(profile, target_column="churned")
    X, y = frame[roles.feature_columns], frame["churned"]
    X_train, X_test, y_train, y_test = X.iloc[:200], X.iloc[200:], y.iloc[:200], y.iloc[200:]
    factories = get_classification_models(n_classes=2)
    estimator = _build_pipeline_for_model("logistic_regression", factories["logistic_regression"],
                                          roles, "classification").fit(X_train, y_train)
    saved = save_model(estimator, X_train, y_train, profile, roles, problem_type="binary_classification",
                       model_name="logistic_regression", output_dir=tmp_path / "model",
                       selection_source="leaderboard", write_mlflow_model=False)
    role_dict = _role_dict(roles)
    segments = _card_segments(estimator, X_test, y_test, frame, None, role_dict, None, "binary", threshold=0.5)
    assert "error" not in segments, segments

    path = _write_model_card(
        out_dir=tmp_path, run_name="real", run_id=None, dataset_path="data/synthetic_classification.csv",
        problem_decision={"chosen": {"problem_type": "binary_classification", "target_column": "churned"},
                          "alternatives": [], "confidence": 0.9},
        target_source="supplied by caller", role_assignment=role_dict, profile_summary=profile.as_dict(),
        explanation={"winner_name": "logistic_regression"}, held_out_metrics={"roc_auc": 0.7},
        leaderboard=None, threshold_choice=None, held_out_operating_point=None,
        group_decision={"column": None, "detected_column": None},
        pre_training_leakage={"flags": []}, post_training_leakage={"flags": []},
        model_artifact=saved.as_dict(), segments=segments, n_train=len(X_train), n_test=len(X_test),
        intended_use=None,
    )
    assert path and Path(path).exists()
    text = Path(path).read_text(encoding="utf-8")
    assert "logistic_regression (leaderboard)" in text, "selection source must come from the artifact's schema"
    assert "scikit-learn" in text, "library versions must come from the artifact's schema"
    card_json = json.loads((Path(saved.model_dir) / "model_card.json").read_text(encoding="utf-8"))
    assert card_json["model"]["artifact"] == saved.model_path


def test_a_card_failure_never_takes_down_the_run(tmp_path):
    path = _write_model_card(out_dir=tmp_path, run_name="broken", model_artifact=None, run_id=None)
    assert path is None
    assert (tmp_path / "broken_model_card_error.txt").exists()
