from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    EpochAdmissionCommitment,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    canonical_manifest_bytes,
    derive_prompt_slices,
    derive_window_seed,
    manifest_sha256,
    parse_epoch_plan,
    select_epoch_reveals,
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
    mode="concurrent_checkpoint_epoch",
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
        "training_mode": "sequential_steps",
        "prompt_range_size": 5_000,
        "target_groups_per_environment_lane": 16,
        "candidate_limit_per_environment_lane": 24,
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
        "40b49c6970a1df9d1fd77200112c7cce60bc914c1b4f9b01d1e5bf37d6156c50"
    )
    assert first.window_schedule.mode == "concurrent_checkpoint_epoch"
    assert first.training_mode == "sequential_steps"


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
        {"training_mode": "aggregate_one_step"},
        {"candidate_limit_per_environment_lane": 25},
        {"commitments_per_operator_per_environment_lane": 17},
        {"reveal_seconds": 61.0},
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
    assert "admission_beacon" not in encoded


def test_post_commit_selection_is_arrival_neutral_and_operator_rounded():
    plan = _plan()
    digest = manifest_sha256(plan)
    commitments = [
        EpochAdmissionCommitment(
            commitment_id=f"commit-{operator}-{index}",
            operator_id=operator,
            window_number=plan.first_window,
            environment="math",
            prompt_idx=index,
            payload_sha256=f"{index + 1:064x}",
        )
        for operator in ("alice", "bob", "carol")
        for index in range(4)
    ]
    selected = select_epoch_reveals(
        commitments,
        admission_randomness="9" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=digest,
        limit=6,
        per_prompt_limit=10,
    )
    reversed_selected = select_epoch_reveals(
        list(reversed(commitments)),
        admission_randomness="9" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=digest,
        limit=6,
        per_prompt_limit=10,
    )

    assert selected == reversed_selected
    assert len(selected) == 6
    assert {
        operator: sum(item.startswith(f"commit-{operator}-") for item in selected)
        for operator in ("alice", "bob", "carol")
    } == {"alice": 2, "bob": 2, "carol": 2}


def test_post_commit_selection_applies_prompt_cap_deterministically():
    plan = _plan()
    commitments = [
        EpochAdmissionCommitment(
            commitment_id=f"commit-{index}",
            operator_id=f"operator-{index}",
            window_number=plan.first_window,
            environment="math",
            prompt_idx=7,
            payload_sha256=f"{index + 1:064x}",
        )
        for index in range(8)
    ]
    selected = select_epoch_reveals(
        commitments,
        admission_randomness="8" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=manifest_sha256(plan),
        limit=8,
        per_prompt_limit=2,
    )

    assert len(selected) == 2


def test_post_commit_selection_gives_one_ticket_per_operator_prompt():
    plan = _plan()
    commitments = [
        EpochAdmissionCommitment(
            commitment_id=f"commit-{index}",
            operator_id="operator",
            window_number=plan.first_window,
            environment="math",
            prompt_idx=7,
            payload_sha256=f"{index + 1:064x}",
        )
        for index in range(4)
    ]

    selected = select_epoch_reveals(
        commitments,
        admission_randomness="8" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=manifest_sha256(plan),
        limit=4,
        per_prompt_limit=10,
    )

    assert len(selected) == 1


def test_post_commit_selection_binds_operator_order_to_one_lane():
    plan = _plan()
    commitments = [
        EpochAdmissionCommitment(
            commitment_id=f"commit-{index}",
            operator_id=f"operator-{index}",
            window_number=plan.first_window,
            environment="math",
            prompt_idx=index,
            payload_sha256=f"{index + 1:064x}",
        )
        for index in range(8)
    ]
    digest = manifest_sha256(plan)
    selected = select_epoch_reveals(
        commitments,
        admission_randomness="7" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=digest,
        limit=4,
        per_prompt_limit=10,
    )
    next_lane = [
        replace(item, window_number=plan.first_window + 1)
        for item in commitments
    ]
    next_selected = select_epoch_reveals(
        next_lane,
        admission_randomness="7" * 64,
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=digest,
        limit=4,
        per_prompt_limit=10,
    )

    assert selected != next_selected
    with pytest.raises(ValueError, match="share one lane"):
        select_epoch_reveals(
            [commitments[0], next_lane[1]],
            admission_randomness="7" * 64,
            epoch_id=plan.epoch_id,
            manifest_sha256_hex=digest,
            limit=2,
            per_prompt_limit=10,
        )


def test_training_mode_is_strict_and_bound_before_beacon():
    sequential = _plan(training_mode="sequential_steps")
    aggregate = _plan(training_mode="aggregate_one_step")

    assert sequential.epoch_id != aggregate.epoch_id
    assert manifest_sha256(sequential) != manifest_sha256(aggregate)
    with pytest.raises(ValueError, match="training mode"):
        _plan(training_mode="unknown")


def test_candidate_limit_cannot_be_smaller_than_useful_target():
    with pytest.raises(ValueError, match="candidate_limit"):
        _plan(
            target_groups_per_environment_lane=16,
            candidate_limit_per_environment_lane=15,
        )


def test_manifest_is_immutable_and_rejects_noncanonical_json():
    plan = _plan(count=4)
    with pytest.raises(FrozenInstanceError):
        plan.first_window = 1

    pretty = json.dumps(json.loads(canonical_manifest_bytes(plan)), indent=2)
    with pytest.raises(ValueError, match="not canonical"):
        parse_epoch_plan(pretty)
