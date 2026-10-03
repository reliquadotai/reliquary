"""Golden digests of every compiled profile and of the live registry entries.

The fleet attests these bytes. A change here is a protocol fork, never a
refactor, so every value is a literal rather than something recomputed.
"""

import hashlib
import json
from pathlib import Path

import pytest

from reliquary.protocol.profiles import PROFILES, profile_from_contract
from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared.task_registry import (
    parse_registry,
    render_registry,
    validate_entry,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# Snapshot of the production registry, 2026-09-30. It holds no secret.
LIVE_REGISTRY = FIXTURES / "live_registry_2026_09_30.json"
# `reliquary tasks contract --task-id corpus-code-v1`, byte for byte.
CORPUS_CODE_V1_CONTRACT = FIXTURES / "corpus_code_v1_contract.json"

# profile id -> (canonical sha256, sha256 of json.dumps(sort_keys=True)).
GOLDEN = {
    "qwen35-2b-auction-v2": (
        "d763f85b1366ac1b25d0de1251899d6a8aceee11f5235f18817176a039d15dd1",
        "b69cddfe0f9f450e00adf7851674bbb2bfb276e381824c5413930a1c291ec3c3",
    ),
    "qwen35-4b-auction-v3": (
        "3d0b7013d980b565065e19627fbaeb592c96a1604fb7bd3af8e31d15d04c87e8",
        "f61d0a730e50878ded4aae36bcc911ab0932f418fc34724cfa3f9b786805e53b",
    ),
    "qwen3-4b-base-dapo-v4": (
        "e3098b00582d395fc7176ff835c988588e29a530343942c081781f1bab65de91",
        "6222172340951409e14a2936a4563b6df2d2f7373af405a4870cca86a60a25e5",
    ),
    "qwen3-4b-base-dapo-reasoning-v5": (
        "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd",
        "73ed7c4fa4ce6bf5a446762b88771b3d0a4ce30a2479f52a5dea836cd8f05fa0",
    ),
    "qwen3-4b-base-dapo-fill-closed-v6": (
        "1696eef2a8ff52284842f2253d6f699b50bc657dc93b20fc61a257db7d449385",
        "3cc57240684d246e2c95b136fe912158d821e60f00c952dca7667b013cbd4e30",
    ),
    "qwen3-4b-base-dapo-reliquary-v1": (
        "8637701be242332857e059ae886f35c87d93d4be4730593c8d9e98087cd6bee1",
        "d89fe2439785c6586299ee5efafa1e1518957f8bbf826bead0cf4fa273c5ff8f",
    ),
    "qwen3-4b-reliquary-verifiable-v6-dev1": (
        "b0a6d440a6e40097ba3c05b14a9e4d150e39b34870537abffd414bb71323434b",
        "4aea134f614d6aff859d0aa369ad1062d59174e66a4b02493eabbc3245fe6795",
    ),
    "qwen3-4b-reliquary-episode-v7-dev1": (
        "0165289b5e5de93e8ab5fbc80773c8f6c3d8dd8b67c369ba975153f8531329fb",
        "f952e8aeac5c3d567eea851b25a66fbd659f7e85b329f2ac1362971c9bbe6dcd",
    ),
    "qwen3-4b-reliquary-logic-v8-dev1": (
        "35c3cf2f1330687b9e3da7dc3c83fee51b1313474614d0a0339278bb42d903cd",
        "4a8e0c030b8dc0cf98aa71f26659cad2586d06ebf5c24d161b9afac07ee35762",
    ),
    "teutonic-9b-reliquary-suite-v9-dev1": (
        "d13e5eaa6a2a04f7434d98f7694e0997b1a848d8268eb8c02dbb46463b0fcf09",
        "da7bbbb0fa0b2138ff42231df196dfbe9c20af0669327d69b36919597a23eaa9",
    ),
}
CORPUS_CODE_V1_SHA256 = (
    "ffa86eaf42a3034f3c7bfa67b8b5ae3658019acc4fb0ee62d5485607a66b818d"
)


def _sort_keys_sha(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _live_entries():
    return parse_registry(LIVE_REGISTRY.read_bytes())


def test_every_compiled_profile_is_pinned():
    assert set(PROFILES) == set(GOLDEN)


@pytest.mark.parametrize("profile_id", sorted(GOLDEN))
def test_compiled_profile_digests_are_unchanged(profile_id):
    contract = PROFILES[profile_id].to_generation_contract()
    canonical, sort_keys = GOLDEN[profile_id]
    assert canonical_sha256(contract) == canonical
    assert _sort_keys_sha(contract) == sort_keys


def test_reliquary_v1_matches_the_externally_quoted_digest():
    contract = PROFILES["qwen3-4b-base-dapo-reliquary-v1"].to_generation_contract()
    assert _sort_keys_sha(contract).startswith("d89fe2439785c658")


@pytest.mark.parametrize("profile_id", sorted(GOLDEN))
def test_compiled_profile_contract_is_a_round_trip_fixed_point(profile_id):
    contract = PROFILES[profile_id].to_generation_contract()
    assert profile_from_contract(contract).to_generation_contract() == contract


def test_live_registry_renders_back_to_its_own_bytes():
    raw = LIVE_REGISTRY.read_bytes()
    assert render_registry(_live_entries()) == raw.rstrip(b"\n")


@pytest.mark.parametrize("task_id", ["default", "corpus-code-v1"])
def test_live_entries_validate(task_id):
    validate_entry(_live_entries()[task_id])


def test_live_default_pins_the_compiled_reliquary_v1_digest():
    entry = _live_entries()["default"]
    assert entry.contract is None
    assert entry.profile_id == "qwen3-4b-base-dapo-reliquary-v1"
    assert entry.profile_sha256 == GOLDEN[entry.profile_id][0]


def test_live_corpus_code_v1_contract_is_a_fixed_point_with_its_digest():
    entry = _live_entries()["corpus-code-v1"]
    contract = dict(entry.contract)
    assert entry.profile_sha256 == CORPUS_CODE_V1_SHA256
    assert canonical_sha256(contract) == CORPUS_CODE_V1_SHA256
    rebuilt = profile_from_contract(contract).to_generation_contract()
    assert rebuilt == contract
    assert canonical_sha256(rebuilt) == CORPUS_CODE_V1_SHA256


def test_corpus_code_v1_fixture_is_what_tasks_contract_prints():
    contract = _live_entries()["corpus-code-v1"].contract
    printed = json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    assert CORPUS_CODE_V1_CONTRACT.read_text() == printed
