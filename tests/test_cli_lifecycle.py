"""
The lifecycle commands as a person runs them: `retrain`, then `gate --apply`.

Found walking the loop by hand. `gate --apply` against a models root with no
CHAMPION.json yet printed "Production model: None" after an inconclusive
verdict: the gate only ever writes the pointer on a promotion, so nothing had
recorded which model was in production in the first place. The first applied
gate now records the champion it was run against, before deciding. That is a
record, not a move — only a promotion moves an existing pointer.
"""
from __future__ import annotations

import json

from autoeng.cli import main
from autoeng.registry.champion import CHAMPION_POINTER, GATE_LOG, write_champion
from tests.test_entity_key import _grouped_champion


def test_the_first_applied_gate_records_the_champion_it_kept(tmp_path, capsys):
    champion = _grouped_champion(tmp_path)
    root = tmp_path / "production"

    code = main(["gate", str(champion.model_dir), str(champion.challenger_dir),
                 "--models-root", str(root), "--apply"])
    output = capsys.readouterr().out
    pointer = json.loads((root / CHAMPION_POINTER).read_text(encoding="utf-8"))

    assert code in (1, 2), "a challenger fitted to permuted labels must not be promoted"
    assert pointer["model_dir"] == str(champion.model_dir.resolve())
    assert "Production model: None" not in output
    assert len((root / GATE_LOG).read_text(encoding="utf-8").splitlines()) == 1


def test_an_existing_pointer_is_not_overwritten_by_the_first_gate(tmp_path):
    champion = _grouped_champion(tmp_path)
    root = tmp_path / "production"
    elsewhere = tmp_path / "some_other_model"
    elsewhere.mkdir()
    write_champion(root, elsewhere)

    main(["gate", str(champion.model_dir), str(champion.challenger_dir), "--models-root", str(root), "--apply"])
    pointer = json.loads((root / CHAMPION_POINTER).read_text(encoding="utf-8"))
    assert pointer["model_dir"] == str(elsewhere.resolve())


def test_retrain_without_a_log_says_what_it_needs(tmp_path, capsys):
    champion = _grouped_champion(tmp_path)
    assert main(["retrain", str(champion.model_dir), "--dataset", str(champion.path)]) == 2
    assert "needs served traffic with outcomes" in capsys.readouterr().out
