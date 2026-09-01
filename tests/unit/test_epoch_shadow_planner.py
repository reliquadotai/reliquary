from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from reliquary.miner.epoch_shadow import (
    EpochShadowPlanner,
    PreparedGroup,
    ShadowPlannerConfig,
    ShadowPlannerDisabled,
    ShadowPlannerError,
    ShadowWorkSpec,
)
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    generation_contract_sha256,
    manifest_sha256,
)
from reliquary.shared.checkpoint_epoch_market import GenerationTicket


CONTRACT = {"profile_id": "experimental-fixture", "sampling": {"temperature": 1.0}}
INTENT_SET_SHA256 = "d" * 64


def _plan(*, checkpoint_revision: str = "a" * 40):
    return build_epoch_plan(
        protocol=ProtocolBinding(
            profile_id="experimental-fixture",
            protocol_version=99,
            generation_contract_sha256=generation_contract_sha256(CONTRACT),
        ),
        checkpoint=CheckpointBinding(
            number=12,
            repo_id="example/checkpoint",
            revision=checkpoint_revision,
            commit_observed_round=100,
        ),
        epoch_beacon=BeaconBinding(
            source="drand",
            chain="quicknet",
            chain_hash="b" * 64,
            round=101,
            randomness="c" * 64,
        ),
        beacon_delay_rounds=1,
        first_window=50,
        window_count=3,
        warmup_rounds=3,
        window_schedule=WindowSchedule(
            mode="concurrent_checkpoint_epoch",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
        training_mode="sequential_steps",
        target_groups_per_environment_lane=16,
        candidate_limit_per_environment_lane=24,
        environment_universes={"code": 100, "math": 100},
        prompt_range_size=8,
    )


def _prompt_digest(environment: str, prompt_idx: int) -> str:
    return hashlib.sha256(f"{environment}:{prompt_idx}".encode()).hexdigest()


def _spec(plan, offset=0, environment="math"):
    prompt_slice = next(
        item
        for item in plan.windows[offset].prompt_slices
        if item.environment == environment
    )
    prompt_idx = prompt_slice.start
    return ShadowWorkSpec(
        window_offset=offset,
        environment=environment,
        prompt_idx=prompt_idx,
        prompt_content_sha256=_prompt_digest(environment, prompt_idx),
        estimated_payload_bytes=128,
    )


def _planner(tmp_path, **config):
    return EpochShadowPlanner(
        ShadowPlannerConfig(
            spool_root=tmp_path,
            enabled=True,
            **config,
        ),
        beacon_verifier=lambda _beacon: True,
    )


def _payload(record):
    binding = record.binding
    return {
        "prompt_idx": binding.prompt_idx,
        "window_start": binding.window_number,
        "checkpoint_hash": binding.checkpoint_revision,
        "protocol_version": binding.protocol_version,
        "generation_profile_id": binding.profile_id,
        "generation_randomness": binding.generation_randomness,
        "rollouts": [{
            "env_name": binding.environment,
            "commit": {
                "beacon": {"randomness": binding.generation_randomness}
            },
        }],
    }


def _state(plan, *, window=50, state="OPEN", **overrides):
    values = {
        "state": state,
        "window_n": window,
        "checkpoint_epoch_id": plan.epoch_id,
        "checkpoint_epoch_manifest_sha256": manifest_sha256(plan),
        "checkpoint_epoch_target_groups": (
            plan.target_groups_per_environment_lane
        ),
        "checkpoint_epoch_candidate_limit": (
            plan.candidate_limit_per_environment_lane
        ),
        "checkpoint_epoch_candidate_remaining": (
            plan.candidate_limit_per_environment_lane
        ),
        "checkpoint_epoch_collection_seconds": (
            plan.window_schedule.collection_seconds
        ),
        "checkpoint_epoch_intent_seconds": plan.intent_seconds,
        "checkpoint_epoch_backup_activation_fractions": list(
            plan.backup_activation_fractions
        ),
        "checkpoint_epoch_phase": "generation",
        "checkpoint_epoch_generation_intent_set_sha256": INTENT_SET_SHA256,
        "generation_profile_id": plan.protocol.profile_id,
        "protocol_version": plan.protocol.protocol_version,
        "generation_contract": CONTRACT,
        "checkpoint_n": plan.checkpoint.number,
        "checkpoint_repo_id": plan.checkpoint.repo_id,
        "checkpoint_revision": plan.checkpoint.revision,
        "randomness": plan.windows[window - plan.first_window].generation_randomness,
    }
    values.update(overrides)
    return values


def _select_all(planner):
    intents = planner.generation_intents(
        operator_id="operator-a", miner_hotkey="miner-a"
    )
    tickets = tuple(
        GenerationTicket(
            intent_id=intent.intent_id,
            role="primary",
            activation_wave=0,
            operator_round=index,
            selection_rank=index,
        )
        for index, intent in enumerate(intents)
    )
    planner.apply_generation_tickets(
        tickets,
        intent_set_sha256=INTENT_SET_SHA256,
    )
    return tickets


def _prepare_one(planner, plan, spec=None):
    spec = spec or _spec(plan)
    queued = planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[spec],
    )
    assert len(queued) == 1
    _select_all(planner)
    prepared = planner.prepare_next(
        lambda record: PreparedGroup(payload=_payload(record), gpu_seconds=2.5)
    )
    assert prepared is not None
    return prepared


def test_planner_is_disabled_and_has_no_transport_by_default(tmp_path):
    planner = EpochShadowPlanner(ShadowPlannerConfig(spool_root=tmp_path))

    with pytest.raises(ShadowPlannerDisabled):
        planner.enqueue(
            _plan(),
            expected_manifest_sha256=manifest_sha256(_plan()),
            specs=[],
        )
    assert "submit" not in EpochShadowPlanner.__dict__


def test_queue_is_bounded_and_deterministic(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path, max_queue_groups=2)
    accepted = planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan, 2), _spec(plan, 0), _spec(plan, 1)],
    )

    assert [item.binding.window_offset for item in accepted] == [0, 1]
    assert len(planner.records()) == 2


def test_future_work_never_releases_before_exact_open(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    prepared = _prepare_one(planner, plan)
    cooldowns = {prepared.binding.environment: set()}

    assert planner.release_ready(
        live_state=_state(plan, state="READY"),
        cooldown_prompts_by_environment=cooldowns,
        prompt_sha256=_prompt_digest,
    ) == []
    assert planner.release_ready(
        live_state=_state(plan, window=51),
        cooldown_prompts_by_environment=cooldowns,
        prompt_sha256=_prompt_digest,
    ) == []
    assert planner.records("released") == []


@pytest.mark.parametrize("window_number", [True, 50.0, "50"])
def test_release_requires_an_exact_integer_live_window(
    tmp_path,
    window_number,
):
    plan = _plan()
    planner = _planner(tmp_path)
    prepared = _prepare_one(planner, plan)
    state = _state(plan)
    state["window_n"] = window_number

    assert planner.release_ready(
        live_state=state,
        cooldown_prompts_by_environment={
            prepared.binding.environment: set(),
        },
        prompt_sha256=_prompt_digest,
    ) == []
    assert [record.status for record in planner.records()] == ["prepared"]
    assert planner.records("released") == []


def test_exact_open_release_is_local_and_fully_bound(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    prepared = _prepare_one(planner, plan)

    released = planner.release_ready(
        live_state=_state(plan),
        cooldown_prompts_by_environment={prepared.binding.environment: set()},
        prompt_sha256=_prompt_digest,
    )

    assert len(released) == 1
    assert planner.records() == []
    assert planner.records("released")[0].status == "released"
    assert planner.metrics()["network_send_capable"] is False


def test_concurrent_epoch_release_polls_each_exact_lane(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan, 0), _spec(plan, 1)],
    )
    _select_all(planner)
    first = planner.prepare_next(
        lambda record: PreparedGroup(payload=_payload(record), gpu_seconds=1.0)
    )
    second = planner.prepare_next(
        lambda record: PreparedGroup(payload=_payload(record), gpu_seconds=1.0)
    )
    assert first is not None and second is not None

    states = {
        (record.binding.environment, record.binding.window_number): {
            **_state(plan, window=record.binding.window_number),
            "cooldown_prompts": [],
        }
        for record in (first, second)
    }
    released = planner.release_epoch_ready(
        live_states=states,
        prompt_sha256=_prompt_digest,
    )

    assert len(released) == 2
    assert planner.records() == []


@pytest.mark.parametrize(
    ("mapping_window", "state_window"),
    [
        (True, 50),
        (50.0, 50),
        ("50", 50),
        (50, True),
        (50, 50.0),
        (50, "50"),
    ],
)
def test_epoch_release_requires_exact_integer_window_identities(
    tmp_path,
    mapping_window,
    state_window,
):
    plan = _plan()
    planner = _planner(tmp_path)
    prepared = _prepare_one(planner, plan)
    state = _state(plan)
    state["window_n"] = state_window
    state["cooldown_prompts"] = []

    assert planner.release_epoch_ready(
        live_states={
            (prepared.binding.environment, mapping_window): state,
        },
        prompt_sha256=_prompt_digest,
    ) == []
    assert [record.status for record in planner.records()] == ["prepared"]
    assert planner.records("released") == []


def test_commitment_release_is_not_limited_by_reveal_cohort_size(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    specs = [
        replace(
            _spec(plan),
            prompt_idx=_spec(plan).prompt_idx + offset,
            prompt_content_sha256=_prompt_digest(
                "math", _spec(plan).prompt_idx + offset
            ),
        )
        for offset in range(2)
    ]
    planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=specs,
    )
    _select_all(planner)
    for _ in specs:
        assert planner.prepare_next(
            lambda record: PreparedGroup(
                payload=_payload(record), gpu_seconds=1.0
            )
        ) is not None

    released = planner.release_ready(
        live_state=_state(
            plan,
            checkpoint_epoch_candidate_remaining=1,
        ),
        cooldown_prompts_by_environment={"math": set()},
        prompt_sha256=_prompt_digest,
    )

    assert len(released) == 2
    assert len(planner.records()) == 0


def test_checkpoint_or_manifest_change_invalidates_without_replay(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    spec = _spec(plan)
    _prepare_one(planner, plan, spec)

    assert planner.release_ready(
        live_state=_state(plan, checkpoint_revision="d" * 40),
        cooldown_prompts_by_environment={spec.environment: set()},
        prompt_sha256=_prompt_digest,
    ) == []
    assert planner.records() == []
    assert planner.records("quarantine")[0].status == "quarantined"
    assert planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[spec],
    ) == []


def test_release_rechecks_cooldown_and_prompt_content(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    prepared = _prepare_one(planner, plan)

    assert planner.release_ready(
        live_state=_state(plan),
        cooldown_prompts_by_environment={
            prepared.binding.environment: {prepared.binding.prompt_idx}
        },
        prompt_sha256=_prompt_digest,
    ) == []
    assert planner.records("quarantine")[0].quarantine_reason == (
        "prompt_in_cooldown"
    )


def test_ambiguous_generation_is_quarantined_on_restart(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    record = planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan)],
    )[0]
    path = planner.queue_dir / f"{record.identity}.json"
    planner._write(
        path,
        planner._record_dict(replace(record, status="generating")),
    )

    restarted = _planner(tmp_path)

    assert restarted.records() == []
    assert restarted.records("quarantine")[0].quarantine_reason == (
        "ambiguous_restart"
    )


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_durable_shadow_record_requires_exact_integer_schema(
    tmp_path,
    schema_version,
):
    plan = _plan()
    planner = _planner(tmp_path)
    record = planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan)],
    )[0]
    path = planner.queue_dir / f"{record.identity}.json"
    value = planner._record_dict(record)
    value["schema_version"] = schema_version
    planner._write(path, value)

    assert planner.records() == []
    assert not path.exists()
    assert (planner.quarantine_dir / f"corrupt-{path.name}").exists()


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_durable_active_plan_requires_exact_integer_schema(
    tmp_path,
    schema_version,
):
    plan = _plan()
    planner = _planner(tmp_path)
    digest = planner.adopt_plan(plan, manifest_sha256(plan))
    value = json.loads(planner.active_plan_path.read_bytes())
    value["schema_version"] = schema_version
    planner._write(planner.active_plan_path, value)

    with pytest.raises(ValueError, match="invalid active checkpoint epoch plan"):
        planner.adopt_plan(plan, digest)


def test_prepared_payload_cannot_contain_final_selection_randomness(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan)],
    )
    _select_all(planner)

    def prepare(record):
        payload = _payload(record)
        payload["seal_randomness"] = "not-available-yet"
        return PreparedGroup(payload=payload, gpu_seconds=1.0)

    with pytest.raises(ShadowPlannerError, match="selection randomness"):
        planner.prepare_next(prepare)
    assert planner.records("quarantine")[0].quarantine_reason == (
        "generation_failed_ambiguous"
    )


def test_metrics_report_prepared_compute_coverage_and_age(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    _prepare_one(planner, plan)

    observation_time = 10_000.0
    metrics = planner.metrics(now=observation_time)

    assert observation_time > 0
    assert metrics["gpu_seconds_generated"] == 2.5
    assert metrics["valid_groups_prepared_local"] == 1
    assert metrics["coverage_by_environment_window"]
