"""Epoch prototyping must not mutate existing V4/V5 generation contracts."""

from reliquary.protocol.profiles import to_generation_contract
from reliquary.shared.checkpoint_epoch import (
    canonical_json_bytes,
    generation_contract_sha256,
)


def test_v4_generation_contract_bytes_and_hash_are_unchanged() -> None:
    contract = to_generation_contract("qwen3-4b-base-dapo-v4")

    assert len(canonical_json_bytes(contract)) == 551
    assert generation_contract_sha256(contract) == (
        "e3098b00582d395fc7176ff835c988588e29a530343942c081781f1bab65de91"
    )
    assert "checkpoint_epoch" not in contract
    assert "experimental_capability_id" not in contract


def test_v5_generation_contract_bytes_and_hash_are_unchanged() -> None:
    contract = to_generation_contract("qwen3-4b-base-dapo-reasoning-v5")

    assert len(canonical_json_bytes(contract)) == 1_204
    assert generation_contract_sha256(contract) == (
        "19e98f5a3ddac1980efe66fd80db1ec0f8db87a5e60934efd5d0e8985435eadd"
    )
    assert "checkpoint_epoch" not in contract
    assert "experimental_capability_id" not in contract
