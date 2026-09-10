"""
What happens when every candidate fails.

`_select_final_model` returns `None` for the winner's name when nothing scored
— every model errored, HPO produced nothing, the stack could not be built. The
call sites then did `factories[final_name]`, so the run died with

    KeyError: None

which says nothing about the twenty-one exceptions that actually caused it. A
total search failure is rare but it is exactly the moment a user most needs the
report: the real errors are all sitting on the leaderboard, recorded per
candidate, and a KeyError throws that away.
"""
from __future__ import annotations

import pandas as pd
import pytest

from autoeng.modeling.search import Leaderboard, ModelResult
from autoeng.pipeline import NoViableModelError, _select_final_model


def _all_failed_leaderboard() -> Leaderboard:
    return Leaderboard(
        problem_kind="classification", primary_metric="roc_auc",
        results=[
            ModelResult(name="random_forest", status="failed", error="ValueError: singular matrix"),
            ModelResult(name="logistic_regression", status="failed", error="ValueError: singular matrix"),
        ],
    )


class TestNothingScored:
    def test_select_returns_no_winner(self):
        name, params, source, score, improvement = _select_final_model(
            _all_failed_leaderboard().ranked(), [], None, "roc_auc",
        )
        assert name is None
        assert source == "none"

    def test_raises_a_diagnosable_error_not_a_keyerror(self):
        """The message has to carry the candidates' own errors — they are the
        only explanation of why the search failed."""
        leaderboard = _all_failed_leaderboard()
        with pytest.raises(NoViableModelError) as excinfo:
            raise NoViableModelError.from_leaderboard(leaderboard)

        message = str(excinfo.value)
        assert "random_forest" in message
        assert "singular matrix" in message
        assert "KeyError" not in message
        # Real line breaks, not a literal backslash-n. An escaping slip makes
        # the message technically correct and practically unreadable, and a
        # substring assertion alone will not notice.
        assert chr(10) in message
        assert chr(92) + "n" not in message
        assert len(message.splitlines()) >= 3

    def test_error_message_survives_an_empty_leaderboard(self):
        empty = Leaderboard(problem_kind="regression", primary_metric="r2", results=[])
        message = str(NoViableModelError.from_leaderboard(empty))
        assert message, "an empty leaderboard must still produce a usable message"
