"""
The entity key, from the payload to the gate.

On grouped data the group column is excluded from features: it identifies the
entity rather than describing it. Serving then rejected it as an unknown
column, so it never reached the prediction log, and everything after serving
had to act as if every served row were a different customer:

  - the retrain gave each served row its own singleton group, so a customer's
    new visits could sit on both sides of the challenger's CV and threshold
    folds;
  - a frozen-holdout customer coming back on a NEW visit passed the vector
    check (different vector, same customer) and was trained on;
  - the gate resampled recent traffic by row, an interval measured 1.9-2.0x
    too narrow on the T1-5 forward windows.

The key is now accepted without being required or scored, logged with the raw
payload, carried into the retraining frame, and used by the gate.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from autoeng.common.roles import assign_feature_roles
from autoeng.lifecycle.gate import FORWARD_WINDOW, gate_challenger
from autoeng.lifecycle.retrain import build_retraining_frame, normalise_entity_key
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import freeze_holdout, load_holdout, save_model
from autoeng.serving.app import create_app
from autoeng.serving.store import PredictionStore
from autoeng.serving.validation import validate_payload

KEY = "customer_id"


def _population(n: int, seed: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trait = rng.normal(size=n)
    return pd.DataFrame({
        KEY: [f"{prefix}{i:03d}" for i in range(n)],
        "trait": trait,
        "fingerprint": rng.normal(size=n),
        "y": (trait + rng.normal(0, 0.5, n) > 0).astype(int),
    })


def _visits(population: pd.DataFrame, seed: int, per_customer: int = 5) -> pd.DataFrame:
    """Each visit is a new vector; the customer's fingerprint and label are not."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        {KEY: c[KEY], "fingerprint": round(c["fingerprint"] + rng.normal(0, 0.01), 4),
         "a": round(c["trait"] + rng.normal(0, 0.3), 4), "b": round(rng.normal(), 4), "y": int(c["y"])}
        for _, c in population.iterrows() for _ in range(per_customer)
    ])


def _fit_and_save(tmp_path, name, X, y, profile, roles):
    estimator = _build_pipeline_for_model(
        "logistic_regression", get_classification_models(n_classes=2)["logistic_regression"],
        roles, "classification",
    )
    estimator.fit(X, y)
    return Path(save_model(estimator, X, y, profile, roles, problem_type="binary_classification",
                           model_name=name, output_dir=tmp_path / name, write_mlflow_model=False).model_dir)


def _grouped_champion(tmp_path) -> SimpleNamespace:
    """A real artifact: saved through save_model, holdout frozen by customer."""
    population = _population(80, seed=0, prefix="C")
    df = _visits(population, seed=1)
    path = tmp_path / "original.csv"
    df.to_csv(path, index=False)

    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y", group_column=KEY)
    assert KEY not in roles.feature_columns
    X, y = df[roles.feature_columns], df["y"]
    holdout_ids = sorted(population[KEY].iloc[::4])
    held = df[KEY].isin(holdout_ids)

    model_dir = _fit_and_save(tmp_path, "champion", X[~held], y[~held], profile, roles)
    freeze_holdout(model_dir, X[held], y[held], "y", extra_columns=df.loc[held, [KEY]])
    challenger_dir = _fit_and_save(
        tmp_path, "challenger", X[~held],
        pd.Series(np.random.default_rng(4).permutation(y[~held].to_numpy()), index=y[~held].index),
        profile, roles,
    )
    from autoeng.registry.model_store import MODEL_FILENAME, SCHEMA_FILENAME, load_model

    schema = load_model(model_dir / MODEL_FILENAME, model_dir / SCHEMA_FILENAME).schema
    return SimpleNamespace(path=path, df=df, population=population, features=list(roles.feature_columns),
                           model_dir=model_dir, challenger_dir=challenger_dir, schema=schema,
                           holdout_ids=holdout_ids)


def _log(store: PredictionStore, rows: pd.DataFrame, features: list[str], with_key: bool = True) -> None:
    for _, row in rows.iterrows():
        payload = {c: row[c] for c in features}
        if with_key:
            payload[KEY] = row[KEY]
        request_id = store.log_prediction(payload=payload, prediction=0)
        store.record_outcome(request_id, actual=int(row["y"]))


class TestServingAcceptsTheKey:
    def test_the_key_is_accepted_logged_and_never_scored(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        client = TestClient(create_app(champion.model_dir, store_path=tmp_path / "log.db"))
        row = {c: float(champion.df[c].iloc[0]) for c in champion.features}

        without = client.post("/predict", json={"row": row})
        with_key = client.post("/predict", json={"row": row | {KEY: "C000"}})
        assert without.status_code == 200, without.text
        assert with_key.status_code == 200, "the entity key must not be rejected as an unknown column"
        assert with_key.json()["probability"] == without.json()["probability"], "the key must never be scored"
        assert not with_key.json()["warnings"]

        logged = PredictionStore(tmp_path / "log.db").prediction_frame()
        assert logged[KEY].tolist() == [None, "C000"] or (
            pd.isna(logged[KEY].iloc[0]) and logged[KEY].iloc[1] == "C000"
        )

    def test_the_contract_publishes_the_key_as_optional_and_unscored(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        contract = TestClient(create_app(champion.model_dir, log_predictions=False)).get("/model").json()
        assert contract["entity_key"]["column"] == KEY
        assert contract["entity_key"]["required"] is False
        assert contract["entity_key"]["used_for_scoring"] is False
        assert KEY not in contract["feature_columns"]

    def test_a_model_without_groups_still_rejects_the_column(self):
        schema = {"feature_columns": ["a"], "columns": {"a": {"dtype": "float64"}},
                  "feature_roles": {"group_column": None}}
        result = validate_payload([{"a": 1.0, KEY: "C000"}], schema)
        assert not result.ok
        assert result.errors[0].kind == "unknown_column"


class TestTheRetrainingFrameUsesTheKey:
    def test_a_key_sent_to_the_api_reaches_the_retraining_frame(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        client = TestClient(create_app(champion.model_dir, store_path=tmp_path / "log.db"))
        new = _visits(_population(12, seed=5, prefix="N"), seed=6)

        rows = [{c: float(r[c]) for c in champion.features} | {KEY: r[KEY]} for _, r in new.iterrows()]
        served = client.post("/predict/batch", json={"rows": rows}).json()["predictions"]
        outcomes = [{"request_id": p["request_id"], "actual": int(y)} for p, y in zip(served, new["y"])]
        assert not client.post("/outcomes/batch", json={"outcomes": outcomes}).json()["unknown_request_ids"]

        store = PredictionStore(tmp_path / "log.db")
        frame, report = build_retraining_frame(champion.path, store, champion.schema,
                                               holdout=load_holdout(champion.model_dir))
        appended = frame.tail(report["n_new_rows"])
        assert appended[KEY].notna().all(), "served rows must keep their entity, not a singleton group each"
        assert sorted(appended[KEY].unique()) == sorted(new[KEY].unique())
        assert report["n_new_rows_without_group_key"] == 0
        assert not any("singleton" in w for w in report["warnings"]), report["warnings"]
        assert report["challenger_only_entities"] == sorted(new[KEY].unique())

    def test_a_holdout_customer_on_a_new_visit_is_kept_out(self, tmp_path):
        """Different vector, same customer: the vector check passes it."""
        champion = _grouped_champion(tmp_path)
        returning = champion.population[champion.population[KEY].isin(champion.holdout_ids[:5])]
        returning_visits = _visits(returning, seed=7)
        new_visits = _visits(_population(12, seed=8, prefix="N"), seed=9)
        store = PredictionStore(tmp_path / "log.db")
        _log(store, pd.concat([returning_visits, new_visits]), champion.features)

        frame, report = build_retraining_frame(champion.path, store, champion.schema,
                                               holdout=load_holdout(champion.model_dir))
        assert report["n_new_rows_repeating_holdout_excluded"] == 0, "fixture: no vector repeats"
        assert report["n_new_rows_of_holdout_entities_excluded"] == len(returning_visits)
        assert report["n_new_rows"] == len(new_visits)
        assert not set(frame[KEY].dropna()) & set(champion.holdout_ids)
        assert any("new visits" in w for w in report["warnings"]), report["warnings"]

    def test_without_the_key_it_says_what_could_not_be_checked(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        store = PredictionStore(tmp_path / "log.db")
        new_visits = _visits(_population(12, seed=8, prefix="N"), seed=9)
        _log(store, new_visits, champion.features, with_key=False)

        frame, report = build_retraining_frame(champion.path, store, champion.schema,
                                               holdout=load_holdout(champion.model_dir))
        assert report["n_new_rows_without_group_key"] == len(new_visits)
        assert frame.tail(len(new_visits))[KEY].isna().all()
        assert any("cannot be checked against the frozen holdout" in w for w in report["warnings"])
        assert report["challenger_only_entities"] == []

    def test_keys_match_across_csv_json_and_nulls(self):
        assert normalise_entity_key(1001) == normalise_entity_key(1001.0) == normalise_entity_key("1001")
        assert normalise_entity_key(np.int64(7)) == "7"
        assert normalise_entity_key("C001") == "C001"
        assert normalise_entity_key(None) is None
        assert normalise_entity_key(float("nan")) is None, "a null is no key, not an entity called 'nan'"


class TestTheGateUsesTheKey:
    def test_forward_rows_are_resampled_by_entity_and_challenger_only_entities_excluded(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        unseen = _visits(_population(12, seed=10, prefix="F"), seed=11)
        retrained_on = _visits(_population(5, seed=12, prefix="R"), seed=13)
        store = PredictionStore(tmp_path / "log.db")
        _log(store, pd.concat([unseen, retrained_on]), champion.features)

        manifest = {"included_request_ids": [], "challenger_only_entities": sorted(retrained_on[KEY].unique())}
        decision = gate_challenger(champion.model_dir, champion.challenger_dir, store=store, manifest=manifest)
        forward = decision.windows[FORWARD_WINDOW].comparison
        assert forward.n_rows == len(unseen), "rows from entities only the challenger trained on must go"
        assert forward.n_groups == 12, "the forward window must be resampled by customer"
        assert any("only the challenger trained on" in n for n in decision.notes), decision.notes
        assert not any("no forward-window payload carried it" in n for n in decision.notes)

    def test_partial_key_coverage_is_stated(self, tmp_path):
        champion = _grouped_champion(tmp_path)
        store = PredictionStore(tmp_path / "log.db")
        _log(store, _visits(_population(12, seed=10, prefix="F"), seed=11), champion.features)
        _log(store, _visits(_population(2, seed=14, prefix="K"), seed=15), champion.features, with_key=False)

        decision = gate_challenger(champion.model_dir, champion.challenger_dir, store=store,
                                   manifest={"included_request_ids": []})
        forward = decision.windows[FORWARD_WINDOW].comparison
        assert forward.n_rows == 70
        assert forward.n_groups == 12 + 10, "keyless rows count as independent entities"
        assert any("10 of 70 forward-window row(s) arrived without" in n for n in decision.notes), decision.notes
