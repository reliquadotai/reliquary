from __future__ import annotations

from dataclasses import replace
import json

import pytest

from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    ProtocolBinding,
    WindowSchedule,
    canonical_json_bytes,
    canonical_manifest_bytes,
)
from reliquary.validator.checkpoint_epoch_runtime import (
    EpochEquivocationError,
    EpochStore,
    EpochStoreError,
    build_epoch_intent,
    canonical_intent_bytes,
    parse_epoch_intent,
    plan_from_intent,
)


def _intent(**overrides):
    values = {
        "protocol": ProtocolBinding(
            profile_id="experimental-fixture",
            protocol_version=99,
            generation_contract_sha256="a" * 64,
        ),
        "checkpoint_number": 7,
        "checkpoint_repo_id": "example/checkpoint",
        "checkpoint_revision": "b" * 40,
        "commit_observed_round": 1_000,
        "first_window": 501,
        "window_count": 4,
        "beacon_chain": "quicknet",
        "beacon_chain_hash": "c" * 64,
        "warmup_rounds": 5,
        "window_schedule": WindowSchedule(
            mode="concurrent_checkpoint_epoch",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
        "training_mode": "sequential_steps",
        "prompt_range_size": 5_000,
        "environment_universes": {"code": 80_000, "math": 100_000},
    }
    values.update(overrides)
    return build_epoch_intent(**values)


def _beacon(randomness: str = "d" * 64):
    return BeaconBinding(
        source="drand",
        chain="quicknet",
        chain_hash="c" * 64,
        round=1_001,
        randomness=randomness,
    )


def test_intent_is_canonical_and_targets_exactly_the_next_round():
    intent = _intent()

    assert intent.beacon_target_round == intent.checkpoint.commit_observed_round + 1
    assert parse_epoch_intent(canonical_intent_bytes(intent)) == intent


def test_intent_rejects_unknown_schedule_and_training_modes():
    with pytest.raises(ValueError, match="window schedule"):
        _intent(window_schedule=WindowSchedule(
            mode="sequential_windows",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ))
    with pytest.raises(ValueError, match="training mode"):
        _intent(training_mode="unknown")


def test_intent_rejects_a_schema_version_mutation():
    value = json.loads(canonical_intent_bytes(_intent()))
    value["schema_version"] += 1

    with pytest.raises(ValueError, match="version differs"):
        parse_epoch_intent(canonical_json_bytes(value))


def test_store_enforces_commit_before_beacon(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)

    with pytest.raises(EpochStoreError, match="before beacon"):
        store.confirm_before_beacon(intent, observed_round=1_001)

    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    raw = store.install_plan(intent, plan)

    assert raw == canonical_manifest_bytes(plan)
    assert store.load_current_plan() == plan


def test_restart_reload_is_byte_identical(tmp_path):
    first = EpochStore(tmp_path)
    intent = _intent()
    first.install_intent(intent)
    first.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    expected = first.install_plan(intent, plan)

    restarted = EpochStore(tmp_path)
    restored = restarted.load_current_plan()

    assert restored == plan
    assert canonical_manifest_bytes(restored) == expected


def test_manifest_equivocation_is_rejected(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    path = tmp_path / f"plan-{plan.epoch_id}.json"
    path.write_bytes(b"different")

    with pytest.raises(EpochEquivocationError, match="different bytes"):
        store.install_plan(intent, plan)


def test_plan_cannot_change_checkpoint_contract_or_beacon():
    intent = _intent()

    with pytest.raises(ValueError, match="does not match"):
        plan_from_intent(intent, beacon=replace(_beacon(), randomness="e" * 64, round=1_002))

    changed_contract = _intent(
        protocol=replace(
            intent.protocol,
            generation_contract_sha256="f" * 64,
        )
    )
    assert changed_contract.intent_id != intent.intent_id

    changed_checkpoint = _intent(checkpoint_revision="0" * 40)
    assert changed_checkpoint.intent_id != intent.intent_id

    changed_training = _intent(training_mode="aggregate_one_step")
    assert changed_training.intent_id != intent.intent_id
