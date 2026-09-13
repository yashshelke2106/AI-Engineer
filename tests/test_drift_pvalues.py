"""
Drift p-values that mean what they say, and the power they buy.

The per-feature p-values were not honest. A one-sample KS against a CDF
interpolated between stored deciles treats both the interpolation and the
reference sample as exact, and counts rows as independent. Measured, with no
drift anywhere, the share of windows in which some feature came out
Benjamini-Hochberg significant at 5%:

    800 independent rows, 2,000-row reference        62%
    5,000 independent rows                           100%
    60 customers x 5 visits, 120-customer reference  100%

An honest test says about 5%. Now:

  - numeric: the smaller of a KS statistic taken only at the stored CDF points
    (no interpolation) and a cluster-robust z-test on the mean, Bonferroni-
    doubled; both two-sample, both with effective sizes on both sides;
  - categorical: Rao-Scott chi-square, effective sizes on both sides.

Honest p-values are then worth listening to. When features carrying at least a
quarter of the model's importance are significant, an otherwise-quiet report
reads `investigate` (never `alarm` — that still needs PSI beyond its noise). It
lifted a 0.5 sd shift behind 30 customers from 30% flagged to 58%, behind 60
customers from 59% to 87%, at no measured cost in false flags. The rest of the
gap is the data's limit: a single two-sided test on 30 against 120 customers has
about 69% power there.
"""
from __future__ import annotations

import numpy as np

from autoeng.monitoring.drift import FDR_ALPHA, DriftSeverity, check_data_drift
from tests.test_drift import _live, _schema, _training_frame
from tests.test_drift_noise import IMPORTANCES, KEY, _entities, _grouped_schema

UNGROUPED_IMPORTANCES = {"amount": 0.70, "tenure": 0.25, "region": 0.04,
                         "noise_a": 0.003, "noise_b": 0.003, "noise_c": 0.004}


def _any_significant(report) -> bool:
    return any(f.p_value_adjusted is not None and f.p_value_adjusted < FDR_ALPHA for f in report.features)


class TestHonestUnderNoDrift:
    def test_independent_windows(self):
        schema = _schema(_training_frame())
        hits = sum(_any_significant(check_data_drift(_live(800, seed=1000 + i), schema, UNGROUPED_IMPORTANCES))
                   for i in range(40))
        assert hits <= 4, f"{hits} of 40 no-drift windows had a significant feature"

    def test_large_windows_do_not_find_the_interpolation(self):
        schema = _schema(_training_frame())
        hits = sum(_any_significant(check_data_drift(_live(5000, seed=2000 + i), schema, UNGROUPED_IMPORTANCES))
                   for i in range(15))
        assert hits <= 2, f"{hits} of 15 no-drift 5,000-row windows had a significant feature"

    def test_recurring_customers_with_their_keys(self):
        schema = _grouped_schema(_entities(120, 5, np.random.default_rng(1)))
        rng = np.random.default_rng(7)
        hits = sum(_any_significant(check_data_drift(_entities(60, 5, rng), schema, IMPORTANCES))
                   for _ in range(40))
        assert hits <= 4, f"{hits} of 40 no-drift windows had a significant feature"


class TestPowerFromHonestTests:
    @staticmethod
    def _flagged(customers: int, visits: int, shift: float, seed: int, n: int = 40) -> float:
        schema = _grouped_schema(_entities(120, 5, np.random.default_rng(1)))
        rng = np.random.default_rng(seed)
        return np.mean([check_data_drift(_entities(customers, visits, rng, shift=shift), schema,
                                         IMPORTANCES).severity != DriftSeverity.OK for _ in range(n)])

    def test_a_half_sd_shift_behind_thirty_customers(self):
        assert self._flagged(30, 10, 0.5, seed=60) >= 0.45

    def test_a_half_sd_shift_behind_sixty_customers(self):
        assert self._flagged(60, 5, 0.5, seed=55) >= 0.75

    def test_significance_alone_raises_investigate_never_alarm(self):
        """A small shift in the dominant feature: significant, PSI within noise."""
        schema = _schema(_training_frame())
        severities = []
        for i in range(20):
            shifted = _live(800, seed=3000 + i, amount=np.random.default_rng(i).lognormal(4.1, 0.8, 800).round(2))
            report = check_data_drift(shifted, schema, UNGROUPED_IMPORTANCES)
            severities.append(report.severity)
            if report.weighted_excess_psi < 0.1 and report.severity == DriftSeverity.INVESTIGATE:
                assert "significant" in report.summary
        assert DriftSeverity.ALARM not in severities
        assert severities.count(DriftSeverity.INVESTIGATE) >= 10, severities


def test_a_categorical_shift_is_significant_and_its_absence_is_not():
    schema = _schema(_training_frame())
    rng = np.random.default_rng(9)
    moved = _live(800, seed=4000, region=rng.choice(["north", "south", "east", "west"], 800, p=[0.25, 0.25, 0.25, 0.25]))
    region = next(f for f in check_data_drift(moved, schema, UNGROUPED_IMPORTANCES).features if f.column == "region")
    assert region.p_value < 1e-3
    same = _live(800, seed=4001)
    region = next(f for f in check_data_drift(same, schema, UNGROUPED_IMPORTANCES).features if f.column == "region")
    assert region.p_value > 0.01
