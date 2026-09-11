"""
The gate on data with repeated entities.

Found by measuring rather than assuming. T0-3 made every training split
entity-aware, but the gate's bootstrap still resampled rows. On the grouped
fixture a customer's rows share a label and near-identical features, so the
frozen holdout's 150 rows carry about 30 customers' worth of evidence.
Resampled by row, the interval came out 2.07 times narrower than resampled by
customer, and the T1-5 comparison it produced flipped: [-0.140, -0.022],
"rejected", became [-0.204, +0.041], inconclusive. The row bootstrap had
claimed a regression the data could not support.

The flip is reproduced here deterministically rather than depending on a
random draw landing near the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autoeng.common.roles import assign_feature_roles
from autoeng.lifecycle.gate import (
    FORWARD_WINDOW, FROZEN_HOLDOUT, MIN_GROUPS_FOR_GATE, Comparison, GateDecision, GateVerdict,
    bootstrap_paired_difference, combine_windows, evaluate_gate, gate_challenger,
)
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import _build_pipeline_for_model
from autoeng.profiling.profiler import profile_dataset
from autoeng.registry.model_store import freeze_holdout, save_model
from autoeng.serving.store import PredictionStore


def _clustered(n_groups: int = 30, per_group: int = 5, seed: int = 0):
    """Correctness decided per entity: a customer's rows are right or wrong together."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    y = np.repeat(rng.integers(0, 2, n_groups), per_group)
    champion = np.where(np.repeat(rng.uniform(size=n_groups) < 0.80, per_group), y, 1 - y)
    challenger = np.where(np.repeat(rng.uniform(size=n_groups) < 0.68, per_group), y, 1 - y)
    return y, champion, challenger, groups


def _three_bad_customers():
    """Thirty customers, five rows each; the challenger is wrong on three of them.

    By row that is 15 bad rows of 150, and a resample with none of them is
    astronomically unlikely, so the interval sits below zero. By customer it is
    3 of 30, and a resample containing none of the three happens about 4% of the
    time, which puts zero inside a 95% interval.
    """
    groups = np.repeat(np.arange(30), 5)
    y = np.repeat(np.arange(30) % 2, 5)
    champion = y.copy()
    challenger = y.copy()
    worse = np.isin(groups, [0, 1, 2])
    challenger[worse] = 1 - challenger[worse]
    return y, champion, challenger, groups


class TestResamplingUnit:
    def test_resampling_entities_widens_the_interval_to_match_the_evidence(self):
        y, champion, challenger, groups = _clustered()
        by_row = bootstrap_paired_difference(y, champion, challenger, metric="accuracy", n_bootstrap=1000)
        by_entity = bootstrap_paired_difference(y, champion, challenger, metric="accuracy",
                                                n_bootstrap=1000, groups=groups)
        assert by_row.n_groups is None and by_entity.n_groups == 30
        # Fully clustered correctness over five rows per entity: about sqrt(5).
        assert (by_entity.ci_high - by_entity.ci_low) > 1.5 * (by_row.ci_high - by_row.ci_low)

    def test_a_regression_rows_would_claim_is_inconclusive_by_entity(self):
        y, champion, challenger, groups = _three_bad_customers()
        by_row = evaluate_gate(y, champion, challenger, metric="accuracy")
        by_entity = evaluate_gate(y, champion, challenger, metric="accuracy", groups=groups)
        assert by_row.verdict == GateVerdict.REJECTED
        assert by_entity.verdict == GateVerdict.INCONCLUSIVE
        assert "30 entities" in by_entity.reason

    def test_too_few_entities_is_inconclusive_however_many_rows_they_span(self):
        n = MIN_GROUPS_FOR_GATE - 1
        groups = np.repeat(np.arange(n), 20)
        y = np.repeat(np.arange(n) % 2, 20)
        decision = evaluate_gate(y, y, 1 - y, metric="accuracy", groups=groups)
        assert decision.verdict == GateVerdict.INCONCLUSIVE
        assert "entities" in decision.reason.lower()


def test_gate_challenger_resamples_the_frozen_holdout_by_entity(tmp_path):
    rng = np.random.default_rng(3)
    n_customers, visits = 80, 5
    trait = rng.normal(size=n_customers)
    label = (trait + rng.normal(0, 0.5, n_customers) > 0).astype(int)
    df = pd.DataFrame([
        {"customer_id": f"C{c:03d}", "a": round(trait[c] + rng.normal(0, 0.05), 4),
         "b": round(rng.normal(), 4), "y": int(label[c])}
        for c in range(n_customers) for _ in range(visits)
    ])
    profile = profile_dataset(df)
    roles = assign_feature_roles(profile, target_column="y", group_column="customer_id")
    X, y = df[roles.feature_columns], df["y"]
    held = df["customer_id"].isin([f"C{c:03d}" for c in range(0, n_customers, 4)])
    X_tr, X_ho, y_tr, y_ho = X[~held], X[held], y[~held], y[held]
    factory = get_classification_models(n_classes=2)["logistic_regression"]

    def fit(labels, name):
        estimator = _build_pipeline_for_model("logistic_regression", factory, roles, "classification")
        estimator.fit(X_tr, labels)
        saved = save_model(estimator, X_tr, labels, profile, roles, problem_type="binary_classification",
                           model_name=name, output_dir=tmp_path / name, write_mlflow_model=False)
        freeze_holdout(saved.model_dir, X_ho, y_ho, "y", extra_columns=df.loc[held, ["customer_id"]])
        return saved.model_dir

    good = fit(y_tr, "good")
    bad = fit(pd.Series(np.random.default_rng(4).permutation(y_tr.to_numpy()), index=y_tr.index), "bad")

    decision = gate_challenger(good, bad)
    comparison = decision.windows[FROZEN_HOLDOUT].comparison
    assert comparison.n_rows == 100
    assert comparison.n_groups == 20, "the frozen holdout must be resampled by customer"

    # Served payloads never carry the group key, so the forward window cannot
    # be resampled by entity — and the decision has to say so.
    store = PredictionStore(tmp_path / "log.db")
    for i in range(len(X_ho)):
        request_id = store.log_prediction(payload=X_ho.iloc[i].to_dict(), prediction=0)
        store.record_outcome(request_id, actual=int(y_ho.iloc[i]))
    with_forward = gate_challenger(good, bad, store=store, manifest={"included_request_ids": []})
    assert with_forward.windows[FORWARD_WINDOW].comparison.n_groups is None
    assert any("never carry it" in note for note in with_forward.notes), with_forward.notes


def test_ask_names_the_window_its_figures_come_from(tmp_path):
    """The summary line used to call any primary comparison "held-out rows",
    including a forward window of live traffic."""
    from autoeng.explain.qa import answer_question
    from autoeng.tracking.mlflow_tracker import log_pipeline_run, log_promotion_decision
    from tests.test_lifecycle_end_to_end import _minimal_run_payload

    comparison = Comparison(metric="f1", champion_score=0.6, challenger_score=0.7, difference=0.1,
                            ci_low=0.05, ci_high=0.15, n_bootstrap=100, n_rows=300, alpha=0.05)
    decision = combine_windows({FORWARD_WINDOW: GateDecision(GateVerdict.PROMOTED, True, "promoted", comparison)})

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    run_id = log_pipeline_run(tracking_uri=tracking_uri, run_name="challenger", **_minimal_run_payload())
    log_promotion_decision(tracking_uri, run_id, decision.as_dict())

    answer = answer_question(tracking_uri, run_id, "why did you promote the challenger?")
    assert "rows of the forward window" in answer
    assert "held-out rows" not in answer
