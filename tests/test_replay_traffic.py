"""
scripts/replay_traffic.py's --shift option: the way to watch drift fire by hand.

The parsing and the change applied to payloads are tested here; the HTTP
round trip was verified against a live server (see the commit message).
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "replay_traffic", Path(__file__).resolve().parents[1] / "scripts" / "replay_traffic.py",
)
replay = importlib.util.module_from_spec(_SPEC)
# Registered first: @dataclass resolves its class's module through sys.modules.
sys.modules[_SPEC.name] = replay
_SPEC.loader.exec_module(replay)


class TestParsing:
    @pytest.mark.parametrize("spec,kind,amount", [
        ("measure_a=+1sd", "sd", 1.0), ("measure_a=-0.5SD", "sd", -0.5),
        ("avg_amount=x8", "multiply", 8.0), ("avg_amount=*1.5", "multiply", 1.5),
        ("measure_b=+0.3", "add", 0.3), ("measure_b=-2", "add", -2.0),
    ])
    def test_numeric_changes(self, spec, kind, amount):
        shift = replay.parse_shift(spec)
        assert (shift.kind, shift.amount) == (kind, amount)

    def test_categories_with_and_without_a_share(self):
        assert replay.parse_shift("home_region=north") == replay.Shift("home_region", "category", value="north")
        assert replay.parse_shift("home_region=north:0.4").share == 0.4

    @pytest.mark.parametrize("spec,value", [("device=xbox", "xbox"), ("disk=ssd", "ssd")])
    def test_a_category_that_looks_like_a_change_is_still_a_category(self, spec, value):
        shift = replay.parse_shift(spec)
        assert (shift.kind, shift.value) == ("category", value)

    @pytest.mark.parametrize("spec", ["measure_a", "=+1sd", "measure_a=", "measure_a=+onesd",
                                      "home_region=north:1.5"])
    def test_malformed_specs_say_what_to_write(self, spec):
        with pytest.raises(ValueError, match="--shift"):
            replay.parse_shift(spec)


class TestApplying:
    def test_numeric_shifts(self):
        records = [{"a": 1.0, "b": 2.0, "c": None}, {"a": 3.0, "b": 4.0, "c": 5.0}]
        shifts = [replay.parse_shift("a=+2sd"), replay.parse_shift("b=x10"), replay.parse_shift("c=-1")]
        replay.apply_shifts(records, shifts, {"a": 0.5}, random.Random(0))
        assert records == [{"a": 2.0, "b": 20.0, "c": None}, {"a": 4.0, "b": 40.0, "c": 4.0}]

    def test_a_partial_category_shift_hits_about_its_share(self):
        records = [{"region": "south"} for _ in range(2000)]
        replay.apply_shifts(records, [replay.parse_shift("region=north:0.4")], {}, random.Random(1))
        share = sum(r["region"] == "north" for r in records) / len(records)
        assert share == pytest.approx(0.4, abs=0.03)
