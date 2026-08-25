from __future__ import annotations

from dataclasses import replace
import hashlib

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


CONTRACT = {"profile_id": "experimental-fixture", "sampling": {"temperature": 1.0}}


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
            mode="ordinary_window_state_machine",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
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


def _prepare_one(planner, plan, spec=None):
    spec = spec or _spec(plan)
    queued = planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[spec],
    )
    assert len(queued) == 1
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


def test_prepared_payload_cannot_contain_final_selection_randomness(tmp_path):
    plan = _plan()
    planner = _planner(tmp_path)
    planner.enqueue(
        plan,
        expected_manifest_sha256=manifest_sha256(plan),
        specs=[_spec(plan)],
    )

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
