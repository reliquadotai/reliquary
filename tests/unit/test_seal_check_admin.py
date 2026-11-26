"""``scripts/seal_check_admin.py``: what it prints for sealed, drifted and legacy cards, and for set ids given."""

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

from reliquary.eval import storage, verifiers_source
from reliquary.eval.protocol import grader_sha256

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "seal_check_admin.py"
PINS = ["1.0.0", "0.3.1", "a" * 64]


def _card(*, grader):
    card = {"taskset": {"package": "aime24", "package_version": PINS[0], "verifiers_version": PINS[1],
                        "artifact_sha256": None if grader is None else PINS[2]}}
    if grader is not None:
        card["protocol"] = {"source": {"grader_sha256": grader}}
    return card


def _run(monkeypatch, capsys, cards, argv):
    class Store:
        async def get_bytes(self, key):
            card = cards.get(key.split("/")[-2])
            return None if card is None else json.dumps(card).encode()

    monkeypatch.setattr(storage, "SubnetEvalStore", Store)
    monkeypatch.setattr(verifiers_source, "installed_artifacts", lambda package: PINS[2])
    monkeypatch.setattr(importlib.metadata, "version", lambda name: PINS[1] if name == "verifiers" else PINS[0])
    monkeypatch.setattr(sys, "argv", ["-", *argv])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return lines[0], {line["set"]: line for line in lines[1:]}


def test_a_legacy_card_reads_null_where_no_comparison_applies(monkeypatch, capsys):
    here, lines = _run(monkeypatch, capsys, {
        "sealed": _card(grader=grader_sha256()),
        "drifted": _card(grader="0" * 64),
        "legacy": _card(grader=None),
    }, ["sealed", "drifted", "legacy", "absent"])
    assert here == {"grader_sha256_here": grader_sha256(), "verifiers_here": PINS[1]}
    assert (lines["sealed"]["grader_ok"], lines["sealed"]["taskset_ok"]) == (True, True)
    assert (lines["drifted"]["grader_ok"], lines["drifted"]["taskset_ok"]) == (False, True)
    assert lines["legacy"]["sealed"] is False
    assert (lines["legacy"]["grader_ok"], lines["legacy"]["taskset_ok"]) == (None, None)
    assert lines["absent"] == {"set": "absent", "card": None}


def test_without_arguments_it_checks_its_own_list(monkeypatch, capsys):
    _, lines = _run(monkeypatch, capsys, {}, [])
    sets = list(lines)
    assert "verifiers-math500-r0-n500-gc0b08e92" in sets and "verifiers-scicode-steps-a024c0a1e-r0-n64" in sets
    assert not any("graphwalks" in s or "mrcr" in s for s in sets)
    assert all(lines[s] == {"set": s, "card": None} for s in sets)
