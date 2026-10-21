"""The harness goldens (`harness_goldens`): today's code renders every frozen case id for id.

A fixture whose tokenizer is not on this host is skipped by name; set
RELIQUARY_QWEN38_TOKENIZER and/or RELIQUARY_HARNESS_TOKENIZERS (paths joined with
os.pathsep) to run them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.unit import harness_goldens as goldens

pytest.importorskip("renderers")

FIXTURES = sorted(p for p in goldens.FIXTURES.glob("*.*.json") if not p.name.startswith("miner-loop."))


def _tokenizers() -> dict[str, str]:
    paths = [os.environ.get("RELIQUARY_QWEN38_TOKENIZER", "")]
    paths += os.environ.get("RELIQUARY_HARNESS_TOKENIZERS", "").split(os.pathsep)
    found = {}
    for path in filter(None, paths):
        if all((Path(path) / name).is_file() for name in goldens.TOKENIZER_FILES):
            found[goldens.tokenizer_digest(path)] = path
    return found


TOKENIZERS = _tokenizers()


def test_every_case_has_a_fixture_per_frozen_tokenizer():
    names = {*goldens.QWEN_CASES, "real-turns", "envscaler-v2-chatml"}
    by_tokenizer: dict[str, set[str]] = {}
    for path in FIXTURES:
        case, tokenizer, _ = path.name.rsplit(".", 2)
        by_tokenizer.setdefault(tokenizer, set()).add(case)
    assert by_tokenizer and all(cases == names for cases in by_tokenizer.values())


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_the_code_renders_the_frozen_golden(path):
    case, tokenizer, _ = path.name.rsplit(".", 2)
    if tokenizer not in TOKENIZERS:
        pytest.skip(f"tokenizer {tokenizer} is not on this host")
    directory = TOKENIZERS[tokenizer]
    if case in goldens.QWEN_CASES:
        got = goldens.qwen_case(directory, case)
    elif case == "real-turns":
        got = goldens.real_turns_case(directory)
    else:
        got = goldens.envscaler_case(directory)
    assert json.loads(goldens.dump(got)) == json.loads(path.read_text(encoding="utf-8"))
