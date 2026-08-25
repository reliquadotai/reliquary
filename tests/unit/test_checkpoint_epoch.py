from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    canonical_manifest_bytes,
    derive_prompt_slices,
    derive_window_seed,
    manifest_sha256,
    parse_epoch_plan,
)


PROTOCOL = ProtocolBinding(
    profile_id="experimental-fixture",
    protocol_version=99,
    generation_contract_sha256="1" * 64,
)
CHECKPOINT = CheckpointBinding(
    number=7,
    repo_id="example/checkpoint",
    revision="a" * 40,
    commit_observed_round=100,
)
BEACON = BeaconBinding(
    source="drand",
    chain="quicknet",
    chain_hash="2" * 64,
    round=101,
    randomness="3" * 64,
)
SCHEDULE = WindowSchedule(
    mode="ordinary_window_state_machine",
    collection_seconds=60.0,
    timeout_seconds=7200,
)
UNIVERSES = {"code": 90_000, "math": 100_000}


def _plan(*, count: int = 16, **overrides):
    values = {
        "protocol": PROTOCOL,
        "checkpoint": CHECKPOINT,
        "first_window": 500,
        "window_count": count,
        "epoch_beacon": BEACON,
        "beacon_delay_rounds": 1,
        "warmup_rounds": 20,
        "window_schedule": SCHEDULE,
        "prompt_range_size": 5_000,
        "environment_universes": UNIVERSES,
    }
    values.update(overrides)
    return build_epoch_plan(**values)


def test_manifest_is_canonical_deterministic_and_stable():
    first = _plan()
    second = _plan(
        environment_universes={"math": 100_000, "code": 90_000}
    )

    assert first == second
    assert canonical_manifest_bytes(first) == canonical_manifest_bytes(second)
    assert parse_epoch_plan(canonical_manifest_bytes(first)) == first
    assert manifest_sha256(first) == (
        "54b00df08420fa642bc149d6f0a6f276e8ee75054a9ca791d3324226d3b242b2"
    )


def test_production_horizon_has_sixteen_unique_domain_separated_seeds():
    plan = _plan()
    seeds = [window.generation_randomness for window in plan.windows]

    assert len(plan.windows) == 16
    assert len(set(seeds)) == 16
    assert seeds[0] == derive_window_seed(
        plan.epoch_seed,
        offset=0,
        window_number=plan.first_window,
    )


def test_small_horizon_and_validator_miner_slice_derivation_match():
    plan = _plan(count=4)
    shared = derive_prompt_slices(
        plan.epoch_seed,
        environment_universes=UNIVERSES,
        prompt_range_size=plan.prompt_range_size,
        window_count=4,
    )

    assert tuple(window.prompt_slices for window in plan.windows) == shared
    for environment in UNIVERSES:
        ranges = [
            (item.start, item.stop)
            for window in plan.windows
            for item in window.prompt_slices
            if item.environment == environment
        ]
        assert len(ranges) == len(set(ranges)) == 4


def test_overlap_fallback_is_explicit_and_deterministic():
    plan = _plan(
        count=4,
        prompt_range_size=5,
        environment_universes={"tiny": 12},
    )
    slices = [window.prompt_slices[0] for window in plan.windows]

    assert [item.policy for item in slices] == ["deterministic_cycle"] * 4
    assert [item.cycle for item in slices] == [0, 0, 1, 1]
    assert slices == list(_plan(
        count=4,
        prompt_range_size=5,
        environment_universes={"tiny": 12},
    ).windows[index].prompt_slices[0] for index in range(4))


@pytest.mark.parametrize(
    "change",
    [
        {"checkpoint": replace(CHECKPOINT, revision="b" * 40)},
        {
            "protocol": replace(
                PROTOCOL,
                generation_contract_sha256="4" * 64,
            )
        },
        {"epoch_beacon": replace(BEACON, randomness="5" * 64)},
        {
            "window_schedule": replace(
                SCHEDULE,
                collection_seconds=61.0,
            )
        },
    ],
)
def test_manifest_bindings_change_hash(change):
    assert manifest_sha256(_plan(**change)) != manifest_sha256(_plan())


def test_window_offset_and_number_are_both_seed_bindings():
    root = _plan().epoch_seed
    assert derive_window_seed(root, offset=0, window_number=500) != (
        derive_window_seed(root, offset=1, window_number=500)
    )
    assert derive_window_seed(root, offset=0, window_number=500) != (
        derive_window_seed(root, offset=0, window_number=501)
    )


def test_commitment_must_precede_the_first_epoch_beacon():
    with pytest.raises(ValueError, match="first round after checkpoint"):
        _plan(epoch_beacon=replace(BEACON, round=102))


def test_manifest_never_contains_seal_randomness():
    value = json.loads(canonical_manifest_bytes(_plan()))
    encoded = json.dumps(value, sort_keys=True).lower()

    assert "seal_randomness" not in encoded
    assert "selection_randomness" not in encoded
    assert "auction_randomness" not in encoded


def test_manifest_is_immutable_and_rejects_noncanonical_json():
    plan = _plan(count=4)
    with pytest.raises(FrozenInstanceError):
        plan.first_window = 1

    pretty = json.dumps(json.loads(canonical_manifest_bytes(plan)), indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        parse_epoch_plan(pretty)
