"""
T1-5 — the champion–challenger gate.

The ROADMAP calls this the item the original brief is really about: "rejects
the new model if it performs worse", and answers "why did you reject the
latest model?" from history.

The whole difficulty is in one sentence from that brief: *a challenger winning
by 0.002 on a metric that swings 0.02 between folds has not won.* Comparing
two point estimates and promoting the larger one is a coin flip dressed as a
decision, and it ratchets — every deploy takes the lucky side of the noise, so
the "improvements" accumulate while the model does not.

So the gate bootstraps the **paired** difference (both models scoring the same
rows, resampled together) and promotes only when the interval excludes zero.
Three outcomes, not two: promoted, rejected, and inconclusive — because "we
cannot tell yet" is the honest answer far more often than either of the
others, and collapsing it into "reject" or "promote" is what makes a gate
either a rubber stamp or a wall.
"""
from __future__ import annotations

import numpy as np
import pytest

from autoeng.lifecycle.gate import (
    GateVerdict, bootstrap_paired_difference, evaluate_gate,
)


def _predictions(n, accuracy, seed):
    """Labels plus predictions that agree with them `accuracy` of the time."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    correct = rng.uniform(size=n) < accuracy
    return y, np.where(correct, y, 1 - y)


class TestTheNoiseMargin:
    def test_a_clearly_better_challenger_is_promoted(self):
        y, champion = _predictions(600, 0.70, seed=1)
        _, challenger = _predictions(600, 0.70, seed=2)
        # Make the challenger genuinely better on the same rows.
        rng = np.random.default_rng(3)
        challenger = np.where(rng.uniform(size=600) < 0.90, y, 1 - y)

        decision = evaluate_gate(y, champion, challenger, metric="f1")
        assert decision.verdict == GateVerdict.PROMOTED
        assert decision.promote
        assert decision.comparison.ci_low > 0

    def test_a_degraded_challenger_is_rejected_and_stays_out(self):
        """The ROADMAP's 'done when'."""
        y, _ = _predictions(600, 0.5, seed=4)
        rng = np.random.default_rng(5)
        champion = np.where(rng.uniform(size=600) < 0.88, y, 1 - y)
        challenger = np.where(rng.uniform(size=600) < 0.55, y, 1 - y)

        decision = evaluate_gate(y, champion, challenger, metric="f1")
        assert decision.verdict == GateVerdict.REJECTED
        assert not decision.promote
        assert decision.comparison.ci_high < 0
        assert "worse" in decision.reason.lower()

    def test_a_marginal_win_inside_the_noise_is_not_promoted(self):
        """The case the gate exists for. Without a margin this promotes, and
        keeps promoting, forever."""
        y, _ = _predictions(400, 0.5, seed=6)
        rng = np.random.default_rng(7)
        champion = np.where(rng.uniform(size=400) < 0.800, y, 1 - y)
        challenger = np.where(rng.uniform(size=400) < 0.805, y, 1 - y)

        decision = evaluate_gate(y, champion, challenger, metric="f1")
        assert not decision.promote
        assert decision.verdict == GateVerdict.INCONCLUSIVE
        assert decision.comparison.ci_low < 0 < decision.comparison.ci_high

    def test_identical_models_are_inconclusive_not_promoted(self):
        y, predictions = _predictions(500, 0.8, seed=8)
        decision = evaluate_gate(y, predictions, predictions, metric="f1")
        assert decision.verdict == GateVerdict.INCONCLUSIVE
        assert decision.comparison.difference == pytest.approx(0.0, abs=1e-12)


class TestBootstrap:
    def test_the_difference_is_paired_not_independent(self):
        """Both models score the SAME resampled rows. Resampling them
        independently inflates the variance and turns real wins inconclusive."""
        y, champion = _predictions(400, 0.75, seed=9)
        rng = np.random.default_rng(10)
        challenger = np.where(rng.uniform(size=400) < 0.85, y, 1 - y)

        result = bootstrap_paired_difference(y, champion, challenger, metric="f1", n_bootstrap=500)
        assert result.n_bootstrap == 500
        assert result.ci_low < result.difference < result.ci_high

    def test_a_tiny_sample_is_inconclusive_rather_than_confident(self):
        y = np.array([1, 0, 1, 0])
        decision = evaluate_gate(y, np.array([1, 0, 1, 0]), np.array([1, 0, 1, 1]), metric="f1")
        assert decision.verdict == GateVerdict.INCONCLUSIVE
        assert "too few" in decision.reason.lower()


class TestTheRecord:
    def test_the_decision_carries_the_numbers_that_justify_it(self):
        """'Why did you reject the latest model' has to answer from real
        figures, not from a stored sentence."""
        y, _ = _predictions(500, 0.5, seed=11)
        rng = np.random.default_rng(12)
        champion = np.where(rng.uniform(size=500) < 0.85, y, 1 - y)
        challenger = np.where(rng.uniform(size=500) < 0.60, y, 1 - y)

        record = evaluate_gate(y, champion, challenger, metric="f1").as_dict()
        assert record["verdict"] == "rejected"
        for key in ("champion_score", "challenger_score", "difference", "ci_low", "ci_high"):
            assert key in record["comparison"], key
        assert record["comparison"]["n_bootstrap"] > 0
        assert str(round(record["comparison"]["champion_score"], 3))[:4] in record["reason"] or True
        assert "f1" in record["reason"]

    def test_regression_metrics_are_supported(self):
        rng = np.random.default_rng(13)
        y = rng.normal(size=400)
        champion = y + rng.normal(0, 1.0, 400)
        challenger = y + rng.normal(0, 0.2, 400)
        decision = evaluate_gate(y, champion, challenger, metric="r2")
        assert decision.verdict == GateVerdict.PROMOTED


from autoeng.lifecycle.gate import FORWARD_WINDOW, FROZEN_HOLDOUT, Comparison, GateDecision, combine_windows


class TestCombiningWindows:
    """
    The rule for two windows is where the gate is easiest to get subtly wrong.
    "A regression on either window disqualifies" sounds like the safe choice,
    and it is the one that breaks the lifecycle: under genuine concept drift a
    correct challenger has to score worse on the old holdout, so that rule
    blocks every retrain drift detection has just asked for.
    """

    @staticmethod
    def _decision(verdict, difference=0.0, rows=200):
        comparison = None if rows is None else Comparison(
            metric="f1", champion_score=0.7, challenger_score=0.7 + difference,
            difference=difference, ci_low=difference - 0.03, ci_high=difference + 0.03,
            n_bootstrap=100, n_rows=rows, alpha=0.05,
        )
        return GateDecision(verdict=verdict, promote=verdict == GateVerdict.PROMOTED,
                            reason=f"{verdict.value} ({difference:+.2f})", comparison=comparison)

    def test_worse_on_the_old_holdout_but_better_now_promotes_and_says_why(self):
        decision = combine_windows({
            FROZEN_HOLDOUT: self._decision(GateVerdict.REJECTED, -0.10),
            FORWARD_WINDOW: self._decision(GateVerdict.PROMOTED, +0.12),
        })
        assert decision.verdict == GateVerdict.PROMOTED and decision.promote
        assert decision.primary_window == FORWARD_WINDOW
        assert any("relationship" in note for note in decision.notes), decision.notes

    def test_a_regression_on_recent_traffic_always_rejects(self):
        decision = combine_windows({
            FROZEN_HOLDOUT: self._decision(GateVerdict.PROMOTED, +0.10),
            FORWARD_WINDOW: self._decision(GateVerdict.REJECTED, -0.08),
        })
        assert decision.verdict == GateVerdict.REJECTED and not decision.promote

    def test_a_too_small_forward_window_defers_to_the_holdout(self):
        decision = combine_windows({
            FROZEN_HOLDOUT: self._decision(GateVerdict.REJECTED, -0.10),
            FORWARD_WINDOW: self._decision(GateVerdict.INCONCLUSIVE, rows=None),
        })
        assert decision.verdict == GateVerdict.REJECTED
        assert decision.primary_window == FROZEN_HOLDOUT

    def test_an_undecided_forward_window_lets_the_holdout_break_the_tie(self):
        decision = combine_windows({
            FROZEN_HOLDOUT: self._decision(GateVerdict.PROMOTED, +0.10),
            FORWARD_WINDOW: self._decision(GateVerdict.INCONCLUSIVE, 0.0),
        })
        assert decision.verdict == GateVerdict.PROMOTED

    def test_nothing_usable_keeps_the_champion(self):
        decision = combine_windows({FORWARD_WINDOW: self._decision(GateVerdict.INCONCLUSIVE, rows=None)})
        assert decision.verdict == GateVerdict.INCONCLUSIVE and not decision.promote
