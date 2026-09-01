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
    manifest_sha256,
)
from reliquary.shared.checkpoint_epoch_market import (
    GenerationIntent,
    SignedGenerationIntentSet,
    build_generation_intent_set,
    generation_intent_set_sha256,
)
from reliquary.validator.checkpoint_epoch_runtime import (
    EpochEquivocationError,
    EpochStore,
    EpochStoreError,
    SignedEpochIntent,
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
        "target_groups_per_environment_lane": 16,
        "candidate_limit_per_environment_lane": 24,
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


def _install_signed_intent(store: EpochStore, intent) -> None:
    store.install_signed_intent(
        SignedEpochIntent(
            intent=intent,
            intent_sha256=intent.intent_id,
            validator_hotkey="validator",
            validator_signature="aa",
        )
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


def test_intent_requires_an_immutable_checkpoint_revision():
    with pytest.raises(ValueError, match="immutable commit OID"):
        _intent(checkpoint_revision="main")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valuation_policy", "raw_tokens"),
        ("ranking_policy", "throughput_admission"),
        ("reward_policy", "per_token"),
        ("finalization_policy", "arrival_stream"),
    ],
)
def test_intent_rejects_uncoordinated_economic_policy_changes(field, value):
    with pytest.raises(ValueError, match="admission bounds"):
        _intent(**{field: value})


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", 2])
def test_intent_rejects_a_schema_version_mutation(schema_version):
    value = json.loads(canonical_intent_bytes(_intent()))
    value["schema_version"] = schema_version

    with pytest.raises(ValueError, match="version differs"):
        parse_epoch_intent(canonical_json_bytes(value))


def test_store_enforces_commit_before_beacon(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)

    with pytest.raises(EpochStoreError, match="before beacon"):
        store.confirm_before_beacon(intent, observed_round=1_001)

    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    raw = store.install_plan(intent, plan)

    assert raw == canonical_manifest_bytes(plan)
    assert store.load_current_plan() == plan


@pytest.mark.parametrize("observed_round", [True, 1_000.0, "1000"])
def test_commit_confirmation_does_not_coerce_observed_round(
    tmp_path,
    observed_round,
):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)
    confirmation = tmp_path / f"confirmed-{intent.intent_id}.json"

    with pytest.raises(EpochStoreError, match="before beacon"):
        store.confirm_before_beacon(intent, observed_round=observed_round)

    assert not confirmation.exists()


@pytest.mark.parametrize("observed_round", [True, 1_000.0, "1000"])
def test_persisted_confirmation_requires_an_exact_integer_round(
    tmp_path,
    observed_round,
):
    store = EpochStore(tmp_path)
    intent = _intent()
    path = tmp_path / f"confirmed-{intent.intent_id}.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "intent_id": intent.intent_id,
                "observed_round": observed_round,
                "beacon_target_round": intent.beacon_target_round,
            }
        )
    )

    assert store.is_confirmed(intent) is False


def test_persisted_confirmation_rejects_duplicate_json_keys(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    path = tmp_path / f"confirmed-{intent.intent_id}.json"
    path.write_bytes(
        (
            '{"beacon_target_round":1001,"intent_id":"%s",'
            '"observed_round":1000,"observed_round":999}'
        ).encode("utf-8") % intent.intent_id.encode("ascii")
    )

    assert store.is_confirmed(intent) is False


def test_restart_reload_is_byte_identical(tmp_path):
    first = EpochStore(tmp_path)
    intent = _intent()
    first.install_intent(intent)
    _install_signed_intent(first, intent)
    first.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    expected = first.install_plan(intent, plan)

    restarted = EpochStore(tmp_path)
    restored = restarted.load_current_plan()

    assert restored == plan
    assert canonical_manifest_bytes(restored) == expected


def test_generation_intent_set_restart_is_byte_identical_and_create_only(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)
    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    store.install_plan(intent, plan)
    store.mark_activated(plan)
    frozen = build_generation_intent_set(
        (
            GenerationIntent(
                intent_id="intent-a",
                operator_id="operator-a",
                miner_hotkey="miner-a",
                window_number=plan.first_window,
                environment="math",
                prompt_idx=next(
                    item.start
                    for item in plan.windows[0].prompt_slices
                    if item.environment == "math"
                ),
                prompt_content_sha256="1" * 64,
                generation_nonce="nonce-a",
            ),
        ),
        epoch_id=plan.epoch_id,
        manifest_sha256_hex=manifest_sha256(plan),
        intent_close_round=plan.epoch_beacon.round + 1,
        validator_hotkey="validator",
    )
    publication = SignedGenerationIntentSet(
        intent_set=frozen,
        intent_set_sha256=generation_intent_set_sha256(frozen),
        validator_signature="aa",
    )

    expected = store.install_generation_intent_set(plan, publication)
    restored = EpochStore(tmp_path).load_generation_intent_set(plan)

    assert restored == publication
    assert store.install_generation_intent_set(plan, publication) == expected
    changed = replace(
        publication,
        validator_signature="bb",
    )
    with pytest.raises(EpochEquivocationError):
        store.install_generation_intent_set(plan, changed)


def test_activation_and_terminal_outcome_are_create_only(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)
    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    store.install_plan(intent, plan)

    assert store.is_activated(plan) is False
    store.mark_activated(plan)
    store.mark_activated(plan)
    assert store.is_activated(plan) is True
    assert store.terminal_status(plan) is None

    store.mark_terminal(plan, status="completed")
    store.mark_terminal(plan, status="completed")
    assert store.terminal_status(plan) == "completed"
    with pytest.raises(EpochEquivocationError, match="different bytes"):
        store.mark_terminal(plan, status="aborted")


def test_corrupt_activation_marker_is_rejected(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)
    store.confirm_before_beacon(intent, observed_round=1_000)
    plan = plan_from_intent(intent, beacon=_beacon())
    store.install_plan(intent, plan)
    store.mark_activated(plan)
    marker = tmp_path / f"activated-{plan.epoch_id}.json"
    marker.write_bytes(b"{}")

    with pytest.raises(EpochStoreError, match="activation does not match"):
        store.is_activated(plan)


def test_manifest_equivocation_is_rejected(tmp_path):
    store = EpochStore(tmp_path)
    intent = _intent()
    store.install_intent(intent)
    _install_signed_intent(store, intent)
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
