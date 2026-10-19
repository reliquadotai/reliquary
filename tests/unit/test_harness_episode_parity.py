"""The signed miner's loop, recorded at the generate endpoint (`harness_episode_parity`).

`miner-loop.<tokenizer>.json` is what verifiers b2e4e81's bash harness did in each
scenario. RUN_VERIFIERS_BRIDGE=1 plays verifiers again and requires the same record
(needs verifiers, uv and network)."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402

FIXTURES = sorted(parity.FIXTURES.glob("miner-loop.*.json"))


def _fixture(path):
    tokenizer = path.name.split(".")[1]
    if tokenizer not in TOKENIZERS:
        pytest.skip(f"tokenizer {tokenizer} is not on this host")
    return TOKENIZERS[tokenizer], json.loads(path.read_text(encoding="utf-8"))


def test_every_scenario_is_frozen_for_some_tokenizer():
    assert FIXTURES
    for path in FIXTURES:
        assert set(json.loads(path.read_text(encoding="utf-8"))) == set(parity.SCENARIOS)


@pytest.mark.skipif(os.getenv("RUN_VERIFIERS_BRIDGE") != "1",
                    reason="set RUN_VERIFIERS_BRIDGE=1 (needs verifiers, uv and network)")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
@pytest.mark.parametrize("scenario", sorted(parity.SCENARIOS))
def test_verifiers_still_plays_the_frozen_record(path, scenario, tmp_path):
    directory, frozen = _fixture(path)
    got = asyncio.run(parity.play(scenario, directory, tmp_path, parity.verifiers_runner))
    assert json.loads(parity.dump(got)) == frozen[scenario]
