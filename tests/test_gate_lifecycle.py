"""
T1-5 on real persisted models: frozen holdout, forward window, production.

`test_gate.py` pins the statistics on arrays. This file pins the three things
that make the ROADMAP's "done when" a real property instead of a boolean:

  - **"Stays out of production"** needs a production. A CHAMPION.json pointer
    is what serving follows, and a rejected challenger must leave it unmoved.
  - **The comparison must not be rigged.** The frozen holdout has to be kept
    out of the challenger's training data, and the forward window has to
    exclude the logged predictions the challenger was fitted to. Either leak
    scores the challenger partly on rows it memorised — in its favour, and
    invisibly.
  - **The reason has to be retrievable** from the challenger's own run, with
    the real figures.

Models here are fitted directly rather than through run_pipeline so the file
stays fast; the pipeline's own holdout freezing is covered by calling the
helper it uses, and end to end separately.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from autoeng.common.roles import assign_feature_roles
from autoeng.explain.qa import answer_question
from autoeng.lifecycle.gate import FORWARD_WINDOW, FROZEN_HOLDOUT, GateVerdict, gate_challenger
from autoeng.lifecycle.retrain import build_retraining_frame
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.champion import apply_gate_decision, read_champion, read_gate_log, write_champion
from autoeng.registry.model_store import HOLDOUT_FILENAME, ROW_INDEX_COLUMN, freeze_holdout, save_model
from autoeng.serving.app import create_app
from autoeng.serving.store import PredictionStore
from autoeng.tracking.mlflow_tracker import log_pipeline_run, log_promotion_decision
from tests.test_lifecycle_end_to_end import _minimal_run_payload


def _data(n: int = 1500, seed: int = 0, labels: tuple | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = ((1.8 * a - 1.2 * b + rng.normal(0, 0.6, n)) > 0).astype(int)
    frame = pd.DataFrame({"a": a.round(4), "b": b.round(4), "noise": rng.normal(size=n).round(4), "y": y})
    if labels:
        frame["y"] = np.where(frame["y"] == 1, labels[1], labels[0])
    return frame


def _build_world(root, labels: tuple | None = None) -> dict:
    """A genuinely good model and one trained on shuffled labels, both frozen
    against the same holdout."""
    df = _data(labels=labels)
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y")
    X, y = df[roles.feature_columns], df["y"]
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    factory = get_classification_models(n_classes=2)["logistic_regression"]

    def fit(train_labels, name):
        estimator = _build_pipeline_for_model("logistic_regression", factory, roles, "classification")
        estimator.fit(X_tr, train_labels)
        saved = save_model(
            estimator, X_tr, train_labels, profile, roles, problem_type="binary_classification",
            model_name=name, output_dir=root / name, write_mlflow_model=False,
        )
        freeze_holdout(saved.model_dir, X_ho, y_ho, "y")
        return saved.model_dir

    good = fit(y_tr, "good")
    shuffled = pd.Series(np.random.default_rng(1).permutation(y_tr.to_numpy()), index=y_tr.index)
    bad = fit(shuffled, "bad")
    return {"good": good, "bad": bad, "df": df, "X_ho": X_ho, "y_ho": y_ho}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return _build_world(tmp_path_factory.mktemp("gate"))


class TestTheDoneWhen:
    def test_a_degraded_challenger_is_rejected_and_production_does_not_move(self, world, tmp_path):
        models_root = tmp_path / "models"
        write_champion(models_root, world["good"], run_id="champion-run")

        decision = gate_challenger(world["good"], world["bad"])
        assert decision.verdict == GateVerdict.REJECTED
        assert not decision.promote
        assert FROZEN_HOLDOUT in decision.windows

        apply_gate_decision(models_root, decision.as_dict(), world["bad"], "challenger-run")
        assert read_champion(models_root)["model_dir"].endswith("good"), "production must not move"
        entry = read_gate_log(models_root)[-1]
        assert entry["verdict"] == "rejected" and entry["promoted"] is False
        assert entry["champion_dir_before"] == entry["champion_dir_after"]

    def test_a_better_challenger_is_promoted_and_production_moves(self, world, tmp_path):
        models_root = tmp_path / "models"
        write_champion(models_root, world["bad"])

        decision = gate_challenger(world["bad"], world["good"])
        assert decision.verdict == GateVerdict.PROMOTED and decision.promote

        apply_gate_decision(models_root, decision.as_dict(), world["good"])
        assert read_champion(models_root)["model_dir"].endswith("good")

    def test_the_rejection_is_answerable_from_the_challengers_run(self, world, tmp_path):
        decision = gate_challenger(world["good"], world["bad"])
        tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
        run_id = log_pipeline_run(tracking_uri=tracking_uri, run_name="challenger",
                                  **_minimal_run_payload())
        log_promotion_decision(tracking_uri, run_id, decision.as_dict())

        answer = answer_question(tracking_uri, run_id, "why did you reject the latest model?")
        comparison = decision.windows[FROZEN_HOLDOUT].comparison
        assert "rejected" in answer.lower()
        assert f"{comparison.champion_score:.4f}" in answer
        assert f"{comparison.challenger_score:.4f}" in answer
        assert "frozen holdout" in answer, "the answer must say which data the verdict rests on"

    def test_serving_follows_the_production_pointer(self, world, tmp_path):
        models_root = tmp_path / "models"
        write_champion(models_root, world["good"])
        client = TestClient(create_app(models_root, log_predictions=False))
        assert client.get("/health").json()["model_dir"].endswith("good")


class TestLabels:
    def test_string_labels_are_scored_rather_than_reading_as_a_tie(self, tmp_path):
        """The metrics treat class 1 as positive. With "neg"/"pos" labels both
        models would score F1 = 0 and every gate would read as a tie —
        silently freezing whichever model happened to be champion, forever."""
        world = _build_world(tmp_path, labels=("neg", "pos"))
        decision = gate_challenger(world["good"], world["bad"])
        comparison = decision.windows[FROZEN_HOLDOUT].comparison
        assert comparison.champion_score > 0.5, "the good model must not score F1 = 0"
        assert decision.verdict == GateVerdict.REJECTED


class TestForwardWindow:
    def _log(self, tmp_path, world):
        store = PredictionStore(tmp_path / "log.db")
        ids = []
        for i in range(len(world["X_ho"])):
            request_id = store.log_prediction(payload=world["X_ho"].iloc[i].to_dict(),
                                              prediction=0, model_version="champion")
            store.record_outcome(request_id, actual=int(world["y_ho"].iloc[i]))
            ids.append(request_id)
        return store, ids

    def test_rows_the_challenger_trained_on_are_excluded(self, world, tmp_path):
        store, ids = self._log(tmp_path, world)
        decision = gate_challenger(world["good"], world["bad"], store=store,
                                   manifest={"included_request_ids": ids[:200]})
        assert decision.windows[FORWARD_WINDOW].comparison.n_rows == len(ids) - 200

    def test_without_a_manifest_the_forward_window_is_skipped_not_rigged(self, world, tmp_path):
        store, _ = self._log(tmp_path, world)
        decision = gate_challenger(world["good"], world["bad"], store=store, manifest=None)
        assert FORWARD_WINDOW not in decision.windows
        assert any("manifest" in note for note in decision.notes), decision.notes

    def test_no_evaluation_data_at_all_is_inconclusive(self, world, tmp_path):
        for name in ("good", "bad"):
            shutil.copytree(world[name], tmp_path / name)
            (tmp_path / name / HOLDOUT_FILENAME).unlink()
        decision = gate_challenger(tmp_path / "good", tmp_path / "bad")
        assert decision.verdict == GateVerdict.INCONCLUSIVE and not decision.promote

    def test_a_challenger_with_a_different_target_is_rejected_unscored(self, world, tmp_path):
        shutil.copytree(world["bad"], tmp_path / "other")
        path = tmp_path / "other" / "training_schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema["target"]["column"] = "something_else"
        path.write_text(json.dumps(schema), encoding="utf-8")

        decision = gate_challenger(world["good"], tmp_path / "other")
        assert decision.verdict == GateVerdict.REJECTED
        assert "different model" in decision.reason


class TestTheHoldoutStaysFrozen:
    SCHEMA = {"feature_columns": ["a", "b"], "target": {"column": "y"},
              "feature_roles": {"group_column": None}}

    def _original(self, tmp_path):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"a": rng.normal(size=40).round(4), "b": rng.normal(size=40).round(4),
                           "y": rng.integers(0, 2, 40)})
        df.loc[39, ["a", "b", "y"]] = df.loc[3, ["a", "b", "y"]].to_numpy()
        path = tmp_path / "original.csv"
        df.to_csv(path, index=False)
        return path, pd.read_csv(path)

    def _store(self, tmp_path):
        store = PredictionStore(tmp_path / "log.db")
        request_id = store.log_prediction(payload={"a": 100.0, "b": 200.0}, prediction=1)
        store.record_outcome(request_id, actual=1)
        return store

    def test_the_champions_holdout_is_excluded_including_exact_copies(self, tmp_path):
        """Row 39 is a copy of frozen row 3. The champion never trained on it
        (structural cleaning dropped the duplicate before its split), but the
        challenger would — and would then be scored against its twin."""
        path, df = self._original(tmp_path)
        holdout = df.loc[[3, 7, 11]].copy()
        holdout.insert(0, ROW_INDEX_COLUMN, [3, 7, 11])

        frame, report = build_retraining_frame(path, self._store(tmp_path), self.SCHEMA, holdout=holdout)
        assert report["n_holdout_rows_excluded"] == 3
        assert report["n_holdout_copies_excluded"] == 1
        assert len(frame) == 40 - 3 - 1 + 1

        frozen = {tuple(r) for r in df.loc[[3, 7, 11], ["a", "b", "y"]].astype(float).to_numpy().tolist()}
        present = {tuple(r) for r in frame[["a", "b", "y"]].astype(float).to_numpy().tolist()}
        assert not frozen & present, "no frozen row may reach the challenger's training data"

    def test_a_changed_original_file_is_refused(self, tmp_path):
        """If the rows at the frozen positions no longer match, excluding by
        position would remove the wrong rows and keep the real holdout in."""
        path, df = self._original(tmp_path)
        holdout = df.loc[[3, 7]].copy()
        holdout.insert(0, ROW_INDEX_COLUMN, [3, 7])
        holdout.loc[holdout.index[0], "a"] = 999.0
        with pytest.raises(ValueError, match="changed"):
            build_retraining_frame(path, self._store(tmp_path), self.SCHEMA, holdout=holdout)

    def test_the_manifest_records_what_the_challenger_trained_on(self, tmp_path):
        path, _ = self._original(tmp_path)
        store = self._store(tmp_path)
        _, report = build_retraining_frame(path, store, self.SCHEMA)
        assert report["included_request_ids"] == store.labelled_frame()["request_id"].tolist()
        assert report["training_cutoff"]

    def test_the_pipeline_freezes_its_holdout_into_the_artifact(self, world, tmp_path):
        from autoeng.pipeline import _freeze_holdout_into_artifact

        shutil.copytree(world["good"], tmp_path / "m")
        (tmp_path / "m" / HOLDOUT_FILENAME).unlink()
        artifact = {"status": "saved", "model_dir": str(tmp_path / "m")}

        out = _freeze_holdout_into_artifact(artifact, world["X_ho"], world["y_ho"], "y", world["df"], None)
        assert out["holdout"]["n_rows"] == len(world["X_ho"])
        schema = json.loads((tmp_path / "m" / "training_schema.json").read_text(encoding="utf-8"))
        assert schema["holdout"]["row_index_column"] == ROW_INDEX_COLUMN
        frozen = pd.read_csv(tmp_path / "m" / HOLDOUT_FILENAME)
        assert frozen[ROW_INDEX_COLUMN].tolist() == list(world["X_ho"].index)


class TestForwardWindowContent:
    """
    Found end to end, not by reasoning: excluding the challenger's training
    rows from the forward window by request_id is not enough. A payload served
    again under a new id slipped straight through — in a real run every row of
    the "unseen" window repeated a training vector, and the contaminated window
    more than doubled the apparent gap between the two models.
    """

    def test_a_training_payload_served_again_under_a_new_id_is_excluded(self, world, tmp_path):
        from autoeng.lifecycle.retrain import row_fingerprints

        store = PredictionStore(tmp_path / "log.db")
        X_ho, y_ho = world["X_ho"], world["y_ho"]
        trained_ids = []
        for i in range(len(X_ho)):
            request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
            store.record_outcome(request_id, actual=int(y_ho.iloc[i]))
            trained_ids.append(request_id)
        # The same payloads again, under fresh request ids.
        for i in range(100):
            request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
            store.record_outcome(request_id, actual=int(y_ho.iloc[i]))

        columns = list(X_ho.columns)
        manifest = {
            "included_request_ids": trained_ids,
            "fingerprint_columns": columns,
            "training_row_fingerprints": row_fingerprints(X_ho, columns),
        }
        decision = gate_challenger(world["good"], world["bad"], store=store, manifest=manifest)
        assert FORWARD_WINDOW not in decision.windows, "every forward row repeats a training payload"
        assert any("repeat a feature vector" in note for note in decision.notes), decision.notes

    def test_fingerprints_agree_across_csv_json_and_int_float(self, tmp_path):
        """The training frame reaches the manifest through a CSV; forward rows
        reach the gate through JSON payloads. If those disagree about 5 versus
        5.0 or how a null is spelled, nothing ever matches and the exclusion
        silently does nothing."""
        from autoeng.lifecycle.retrain import row_fingerprints

        frame = pd.DataFrame({"n": [5, 7], "x": [0.1234, None], "c": ["a", "b"]})
        frame.to_csv(tmp_path / "frame.csv", index=False)
        reread = pd.read_csv(tmp_path / "frame.csv")
        via_json = pd.DataFrame(json.loads(frame.to_json(orient="records")))
        via_float = frame.assign(n=frame["n"].astype(float))
        columns = ["n", "x", "c"]
        expected = row_fingerprints(frame, columns)
        assert row_fingerprints(reread, columns) == expected
        assert row_fingerprints(via_json, columns) == expected
        assert row_fingerprints(via_float, columns) == expected
        assert expected[0] != expected[1]

    def test_the_retraining_report_fingerprints_every_training_row(self, tmp_path):
        rng = np.random.default_rng(8)
        original = pd.DataFrame({"a": rng.normal(size=30).round(4), "b": rng.normal(size=30).round(4),
                                 "y": rng.integers(0, 2, 30)})
        original.to_csv(tmp_path / "original.csv", index=False)
        store = PredictionStore(tmp_path / "log.db")
        request_id = store.log_prediction(payload={"a": 1.5, "b": -0.5}, prediction=1)
        store.record_outcome(request_id, actual=0)

        schema = {"feature_columns": ["a", "b"], "target": {"column": "y"}, "feature_roles": {}}
        frame, report = build_retraining_frame(tmp_path / "original.csv", store, schema)
        assert report["fingerprint_columns"] == ["a", "b"]
        assert len(report["training_row_fingerprints"]) == len(frame) == report["n_total_rows"]


class TestHoldoutPayloadsServedAgain:
    """
    The fourth leak, found by measuring rather than assuming. Retraining
    excluded the frozen holdout from the ORIGINAL data, but rows appended from
    the prediction log were never checked against it. When production served
    the holdout customers' payloads again, all 150 of 150 frozen vectors
    re-entered the retraining frame, and a random forest trained on it scored
    F1 1.000 on the "frozen" holdout against 0.464 without them.

    The fix has a trap of its own: over a small discrete feature space every
    vector recurs, so content matching would empty the data instead of
    de-leaking it. Both directions are pinned.
    """

    SCHEMA = {"feature_columns": ["a", "b"], "target": {"column": "y"}, "feature_roles": {}}

    @staticmethod
    def _original(tmp_path, discrete=False):
        rng = np.random.default_rng(11)
        n = 60
        if discrete:
            a, b = rng.integers(0, 3, n), rng.integers(0, 3, n)
        else:
            a, b = rng.normal(size=n).round(4), rng.normal(size=n).round(4)
        df = pd.DataFrame({"a": a, "b": b, "y": rng.integers(0, 2, n)})
        path = tmp_path / "original.csv"
        df.to_csv(path, index=False)
        holdout = df.iloc[:10].copy()
        holdout.insert(0, ROW_INDEX_COLUMN, list(range(10)))
        return path, df, holdout

    @staticmethod
    def _serve(store, df, rows):
        for i in rows:
            payload = {c: df.iloc[i][c].item() for c in ("a", "b")}
            request_id = store.log_prediction(payload=payload, prediction=0)
            store.record_outcome(request_id, actual=int(df.iloc[i]["y"]))

    def test_a_holdout_payload_served_again_is_kept_out_of_retraining(self, tmp_path):
        from autoeng.lifecycle.retrain import row_fingerprints

        path, df, holdout = self._original(tmp_path)
        store = PredictionStore(tmp_path / "log.db")
        self._serve(store, df, range(10))        # the holdout's customers, again
        self._serve(store, df, range(40, 45))    # ordinary traffic

        frame, report = build_retraining_frame(path, store, self.SCHEMA, holdout=holdout)
        assert report["n_new_rows_repeating_holdout_excluded"] == 10
        assert report["n_new_rows"] == 5
        assert len(report["included_request_ids"]) == 5, "excluded rows were not trained on"
        frozen = set(row_fingerprints(holdout, ["a", "b"]))
        assert not frozen & set(row_fingerprints(frame, ["a", "b"])), "no frozen vector may be trained on"

    def test_a_discrete_feature_space_is_not_emptied_by_content_matching(self, tmp_path):
        path, df, holdout = self._original(tmp_path, discrete=True)
        store = PredictionStore(tmp_path / "log.db")
        self._serve(store, df, range(10, 40))

        _, report = build_retraining_frame(path, store, self.SCHEMA, holdout=holdout)
        assert report["n_new_rows_repeating_holdout_excluded"] == 0
        assert report["n_new_rows"] == 30
        assert any("not distinctive" in w for w in report["warnings"]), report["warnings"]

    def test_the_gate_does_not_wipe_a_forward_window_over_a_discrete_space(self, world, tmp_path):
        from autoeng.lifecycle.retrain import row_fingerprints

        store = PredictionStore(tmp_path / "log.db")
        X_ho, y_ho = world["X_ho"], world["y_ho"]
        for i in range(len(X_ho)):
            request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
            store.record_outcome(request_id, actual=int(y_ho.iloc[i]))
        columns = list(X_ho.columns)
        # Flagged at retrain time as a discrete feature space, where a matching
        # vector is ordinary recurrence rather than the same observation.
        manifest = {"included_request_ids": [], "fingerprint_columns": columns,
                    "training_row_fingerprints": row_fingerprints(X_ho, columns),
                    "fingerprints_identify_observations": False}

        decision = gate_challenger(world["good"], world["bad"], store=store, manifest=manifest)
        assert decision.windows[FORWARD_WINDOW].comparison.n_rows == len(X_ho)
        assert any("not distinctive" in note for note in decision.notes), decision.notes


class TestRepeatedObservationsAreNotADiscreteSpace:
    """
    A regression introduced by the discrete-space guard itself, found by
    running the lifecycle on a multiclass target. The guard judged
    distinctiveness by how often training rows repeated. But production
    serving the same customers again repeats rows too, which is precisely what
    content exclusion exists for, so the guard read repeated continuous
    observations as "a discrete feature space" and switched the exclusion off,
    reopening the leak it was meant to refine. Distinctiveness is now judged on
    the columns of the distinct vectors, not on the duplication rate.
    """

    def test_continuous_observations_served_three_times_still_identify_rows(self):
        from autoeng.lifecycle.retrain import vectors_identify_observations

        rng = np.random.default_rng(4)
        base = pd.DataFrame({"a": rng.normal(size=100).round(4), "b": rng.normal(size=100).round(4)})
        assert vectors_identify_observations(pd.concat([base] * 3, ignore_index=True), ["a", "b"])

    def test_a_small_grid_does_not_identify_rows_however_many_rows_it_has(self):
        from autoeng.lifecycle.retrain import vectors_identify_observations

        rng = np.random.default_rng(5)
        grid = pd.DataFrame({"a": rng.integers(0, 3, 500), "b": rng.integers(0, 3, 500)})
        assert not vectors_identify_observations(grid, ["a", "b"])
        assert not vectors_identify_observations(grid.iloc[:0], ["a", "b"])

    def test_the_gate_excludes_repeats_when_training_repeated_observations(self, world, tmp_path):
        from autoeng.lifecycle.retrain import row_fingerprints

        store = PredictionStore(tmp_path / "log.db")
        X_ho, y_ho = world["X_ho"], world["y_ho"]
        for i in range(len(X_ho)):
            request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
            store.record_outcome(request_id, actual=int(y_ho.iloc[i]))
        columns = list(X_ho.columns)
        manifest = {"included_request_ids": [], "fingerprint_columns": columns,
                    # Each observation trained on three times over, as recurring traffic does.
                    "training_row_fingerprints": row_fingerprints(X_ho, columns) * 3,
                    "fingerprints_identify_observations": True}

        decision = gate_challenger(world["good"], world["bad"], store=store, manifest=manifest)
        assert FORWARD_WINDOW not in decision.windows
        assert any("repeat a feature vector" in note for note in decision.notes), decision.notes

    def test_a_manifest_without_the_flag_errs_towards_excluding(self, world, tmp_path):
        """An empty forward window defers to the frozen holdout; a rigged one
        promotes on memorised rows. When unsure, take the safe failure."""
        from autoeng.lifecycle.retrain import row_fingerprints

        store = PredictionStore(tmp_path / "log.db")
        X_ho, y_ho = world["X_ho"], world["y_ho"]
        for i in range(len(X_ho)):
            request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
            store.record_outcome(request_id, actual=int(y_ho.iloc[i]))
        columns = list(X_ho.columns)
        manifest = {"included_request_ids": [], "fingerprint_columns": columns,
                    "training_row_fingerprints": row_fingerprints(X_ho, columns) * 3}

        decision = gate_challenger(world["good"], world["bad"], store=store, manifest=manifest)
        assert FORWARD_WINDOW not in decision.windows
