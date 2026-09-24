"""
Header detection on files whose columns are free text.

`csv.Sniffer().has_header` decides by comparing row one with the rows below:
numeric where they are numeric, or a different string *length* — and it only
counts length when every row in that column shares one. A column of free text
has no two documents the same length, so it casts no vote, and a file whose
every column is free text collects no votes and is declared headerless. Its
header row then becomes a data row and every column is renamed col_0, col_1.

Found by running this pipeline on 1,961 newsgroup posts: `--target topic` died
with "Available columns: ['col_0', 'col_1']". Auto-detection would not have
died — it would have modelled a corrupted column under a made-up name.

Calibrated over every committed dataset plus a header-stripped copy of each
(34 cases): the sniffer got 31, the replacement got 34.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from autoeng.ingestion.loader import load_raw_dataset

DATA = Path(__file__).resolve().parents[1] / "data"
POSTS = [
    "the pitcher walked three batters in the seventh and still got the win somehow",
    "my doctor prescribed a lower dosage after the second round of blood tests came back",
    "anyone remember the bases loaded double play that ended that game in extra innings",
    "the clinic called to say the results were normal so the symptoms are unexplained",
    "he swings at everything outside the zone which is why his average collapsed",
    "a diagnosis took four visits and two specialists before anyone agreed on it",
]


def _write(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return path


class TestFreeTextFiles:
    def test_a_header_over_free_text_columns_is_not_eaten(self, tmp_path):
        rows = [["post", "topic"]] + [[p, "sport" if i % 2 == 0 else "health"]
                                     for i, p in enumerate(POSTS * 4)]
        df, report = load_raw_dataset(_write(tmp_path / "posts.csv", rows))
        assert list(df.columns) == ["post", "topic"], "the header row became data"
        assert len(df) == len(POSTS) * 4, "a data row went missing into the header"
        assert not any("synthetic column names" in w for w in report.warnings)
        assert "row one reads as column names" in " ".join(report.notes)

    def test_a_headerless_free_text_file_is_still_called_headerless(self, tmp_path):
        rows = [[p, "sport" if i % 2 == 0 else "health"] for i, p in enumerate(POSTS * 4)]
        df, report = load_raw_dataset(_write(tmp_path / "posts_bare.csv", rows))
        assert list(df.columns) == ["col_0", "col_1"]
        assert len(df) == len(POSTS) * 4, "no row may be consumed as a header"
        assert any("synthetic column names" in w for w in report.warnings)

    def test_the_basis_for_the_decision_is_always_reported(self, tmp_path):
        # Invariant 6: a guess is logged as one, whichever way it went.
        rows = [["post", "topic"]] + [[p, "sport"] for p in POSTS]
        _, report = load_raw_dataset(_write(tmp_path / "with.csv", rows))
        assert any(n.startswith("Header decision:") for n in report.notes)
        _, bare = load_raw_dataset(_write(tmp_path / "without.csv", rows[1:]))
        assert any(n.startswith("Header decision:") for n in bare.notes)


class TestOrdinaryFiles:
    @pytest.mark.parametrize("name", [p.name for p in sorted(DATA.glob("*.csv"))])
    def test_every_committed_dataset_keeps_its_header(self, name):
        df, _ = load_raw_dataset(DATA / name)
        assert not any(str(c).startswith("col_") for c in df.columns), f"{name} lost its header"

    @pytest.mark.parametrize("name", [p.name for p in sorted(DATA.glob("*.csv"))])
    def test_every_committed_dataset_stripped_of_its_header_is_detected_as_such(self, name, tmp_path):
        original = (DATA / name).read_text(encoding="utf-8").splitlines()
        stripped = tmp_path / name
        stripped.write_text("\n".join(original[1:]), encoding="utf-8")
        df, _ = load_raw_dataset(stripped)
        assert all(str(c).startswith("col_") for c in df.columns), f"{name}: row one was mistaken for names"
        assert len(df) == len(pd.read_csv(DATA / name)), "the first data row was eaten as a header"


class TestEdges:
    def test_a_numeric_first_row_is_data(self, tmp_path):
        rows = [["1.5", "2.5"], ["3.5", "4.5"], ["5.5", "6.5"], ["7.5", "8.5"]]
        df, _ = load_raw_dataset(_write(tmp_path / "numbers.csv", rows))
        assert list(df.columns) == ["col_0", "col_1"] and len(df) == 4

    def test_a_first_row_repeating_its_own_column_values_is_data(self, tmp_path):
        rows = [["north", "1"], ["south", "2"], ["north", "3"], ["east", "4"], ["south", "5"]]
        df, _ = load_raw_dataset(_write(tmp_path / "regions.csv", rows))
        assert list(df.columns) == ["col_0", "col_1"], "'north' recurs below, so it is a value"

    def test_duplicate_names_in_row_one_are_not_names(self, tmp_path):
        rows = [["x", "x"], ["1", "2"], ["3", "4"], ["5", "6"]]
        df, _ = load_raw_dataset(_write(tmp_path / "dupes.csv", rows))
        assert list(df.columns) == ["col_0", "col_1"]

    def test_a_two_row_file_falls_back_to_the_sniffer(self, tmp_path):
        rows = [["height_cm", "weight_kg"], ["181", "77"]]
        _, report = load_raw_dataset(_write(tmp_path / "tiny.csv", rows))
        assert any("too few rows to judge" in n for n in report.notes)
