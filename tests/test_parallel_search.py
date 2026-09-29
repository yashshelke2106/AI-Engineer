"""
T2-4 — candidates evaluated in parallel, with nothing about the result changing.

Half the zoo already asks for every core on its own (`n_jobs=-1` on the forests,
bagging, XGBoost and LightGBM). Running N of those at once, each still asking for
all 16 cores, is N x 16 threads contending for 16 — slower than not
parallelising at all, and silent about it. So the pool divides the cores between
its workers, and the test that matters most here checks that from inside a
worker rather than trusting the configuration.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from autoeng.common.roles import assign_feature_roles
from autoeng.modeling import search
from autoeng.modeling.model_zoo import get_classification_models
from autoeng.modeling.search import CLASSIFICATION_SCORING, _run_search
from autoeng.profiling.profiler import profile_dataset


def _frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.8, n) > 0).astype(int)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["label"] = y
    return frame


def _search(frame: pd.DataFrame, models: dict, n_jobs: int, budget: str = "full"):
    roles = assign_feature_roles(profile_dataset(frame), target_column="label")
    return _run_search(frame[roles.feature_columns], frame["label"], roles, "classification", models,
                       CLASSIFICATION_SCORING, "roc_auc", cv_folds=3, budget=budget, n_jobs=n_jobs)


class _Exploding(ClassifierMixin, BaseEstimator):
    """A candidate that fails on every fit, the way a singular matrix would."""

    def fit(self, X, y):
        raise np.linalg.LinAlgError("singular matrix")


class _ThreadProbe(ClassifierMixin, BaseEstimator):
    """Records, from inside whatever process fits it, how many native threads it may use."""

    def __init__(self, out_dir: str | None = None):
        self.out_dir = out_dir

    def fit(self, X, y):
        from threadpoolctl import threadpool_info

        widest = max((pool["num_threads"] for pool in threadpool_info()), default=1)
        Path(self.out_dir, f"{os.getpid()}_{id(self)}.txt").write_text(str(widest), encoding="utf-8")
        self._model = LogisticRegression(max_iter=500).fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def decision_function(self, X):
        return self._model.decision_function(X)


# Models whose scores do not depend on thread count. LightGBM and XGBoost sum in a
# thread-dependent order and can differ in the last digits, which is real but is
# not what these tests are about.
DETERMINISTIC = ("logistic_regression", "decision_tree", "random_forest", "knn")


def _zoo_subset(names=DETERMINISTIC) -> dict:
    zoo = get_classification_models(n_classes=2)
    return {name: zoo[name] for name in names}


class TestSameLeaderboard:
    def test_parallel_and_sequential_agree_candidate_by_candidate(self):
        frame = _frame()
        sequential = _search(frame, _zoo_subset(), n_jobs=1)
        parallel = _search(frame, _zoo_subset(), n_jobs=4)
        assert [r.name for r in parallel.results] == [r.name for r in sequential.results], \
            "results must come back in candidate order, not completion order"
        for seq, par in zip(sequential.results, parallel.results):
            assert par.status == seq.status
            for metric, value in seq.metrics.items():
                assert par.metrics[metric] == pytest.approx(value, abs=1e-12), (seq.name, metric)
        assert parallel.ranked()[0].name == sequential.ranked()[0].name

    def test_the_halving_path_keeps_its_stages_and_order(self, monkeypatch):
        # Force halving on a small frame so both stages run through the pool.
        monkeypatch.setattr(search, "HALVING_MIN_ROWS", 100)
        monkeypatch.setattr(search, "SCREENING_ROWS", 150)
        monkeypatch.setattr(search, "SURVIVORS_PROMOTED", 2)
        frame = _frame(400)
        sequential = _search(frame, _zoo_subset(), n_jobs=1, budget="auto")
        parallel = _search(frame, _zoo_subset(), n_jobs=4, budget="auto")
        assert [(r.name, r.status, r.evaluation_stage) for r in parallel.results] == \
               [(r.name, r.status, r.evaluation_stage) for r in sequential.results]
        assert sum(r.evaluation_stage == "full" for r in parallel.results) == 2


class TestFailuresStayContained:
    def test_a_candidate_that_raises_is_recorded_not_raised(self):
        models = {**_zoo_subset(("logistic_regression",)), "exploding": _Exploding}
        board = _search(_frame(), models, n_jobs=2)
        exploded = next(r for r in board.results if r.name == "exploding")
        assert exploded.status == "failed" and "singular matrix" in exploded.error
        assert board.ranked()[0].name == "logistic_regression"

    def test_a_broken_pool_falls_back_to_sequential(self, monkeypatch):
        class _BrokenParallel:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, tasks):
                raise OSError("worker died")

        monkeypatch.setattr(search, "Parallel", _BrokenParallel)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            board = _search(_frame(), _zoo_subset(("logistic_regression", "decision_tree")), n_jobs=4)
        assert [r.status for r in board.results] == ["ok", "ok"], "a pool failure must not lose the search"
        assert any("falling back to sequential" in str(w.message) for w in caught)


def _probe_widths(tmp_path, n_jobs):
    probe = lambda: _ThreadProbe(out_dir=str(tmp_path))  # noqa: E731
    board = _search(_frame(), {f"probe_{i}": probe for i in range(4)}, n_jobs=n_jobs)
    assert all(r.status == "ok" for r in board.results), [r.error for r in board.results]
    widths = [int(p.read_text(encoding="utf-8")) for p in tmp_path.glob("*.txt")]
    assert widths, "the probe never ran"
    return widths


def test_workers_times_threads_never_exceeds_the_machine(tmp_path):
    # The whole reason sequential is not "one core": without the cap, every
    # worker's BLAS / OpenMP pools size themselves to the whole machine.
    cores = os.cpu_count() or 1
    widths = _probe_widths(tmp_path, n_jobs=4)
    assert max(widths) <= max(1, cores // 4), f"a worker could use {max(widths)} threads on {cores} cores"


def test_an_explicit_thread_count_per_worker_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SEARCH_THREADS_PER_WORKER", 1)
    assert max(_probe_widths(tmp_path, n_jobs=4)) == 1


def test_the_default_uses_a_quarter_of_the_cores_as_workers():
    assert search.default_search_workers(16) == 4
    assert search.default_search_workers(8) == 2
    assert search.default_search_workers(2) == 2
    assert search.default_search_workers(1) == 1


def test_one_job_means_no_pool_at_all(monkeypatch):
    def _no_pool(*args, **kwargs):
        raise AssertionError("n_jobs=1 must not start a pool")

    monkeypatch.setattr(search, "Parallel", _no_pool)
    board = _search(_frame(), _zoo_subset(("logistic_regression",)), n_jobs=1)
    assert board.results[0].status == "ok"
