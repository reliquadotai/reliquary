"""Independent bounded-fill contract tests; fixtures are never release evidence."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from queue import Queue
import subprocess
import sys
import threading

import pytest

from reliquary.validator.proof_scheduler import (
    CapacityAbortReason,
    GlobalProofScheduler,
    ProofDecisionStatus,
    ProofExecution,
    ProofPlanOutcome,
)
from tests.unit.test_proof_scheduler import _candidate, _plan, _wait_until

ENVIRONMENTS = ("openmathinstruct", "opencodeinstruct", "reliquary_logic_v2")


def _bounded_plan(plan_id="bounded", environment=ENVIRONMENTS[0], *, count=3):
    return replace(
        _plan(plan_id, environment,
              [_candidate(i, prefix=plan_id) for i in range(count)],
              required=count + 1, deadline=110.0, open_ended=True),
        allow_shortfall=True, dispatch_deadline_at=70.0,
    )


def _scheduler(prove, now):
    return GlobalProofScheduler(
        devices=("gpu-0",), environments=ENVIRONMENTS,
        proof_callable=prove, checkpoint_revision="rev-a",
        clock=lambda: now[0], deadline_poll_seconds=0.005,
    )


@pytest.mark.parametrize("at", [70.0, 70.001])
def test_no_pending_proof_starts_at_or_after_dispatch_cutoff(at):
    now, called = [10.0], []
    scheduler = _scheduler(lambda invocation: called.append(invocation) or True, now)
    try:
        # Submit while valid, then advance under the same lock workers use.
        with scheduler._condition:
            handle = scheduler.submit(_bounded_plan())
            now[0] = at
        scheduler.expire_deadlines()
        result = handle.result(2)
        assert called == []
        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert result.abort_reason is None
        assert result.attempts_started == 0
        assert len(result.decisions) == 3
        assert all(d.status is ProofDecisionStatus.NOT_NEEDED for d in result.decisions)
        assert {d.reason for d in result.decisions} == {"dispatch_budget_exhausted"}
        assert result.winner_job_ids == ()
        assert scheduler.snapshot()["state"] == "running"
    finally:
        assert scheduler.close()


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_active_proof_settles_while_pending_work_is_not_needed(explicit_stop):
    now = [10.0]
    started, release = threading.Event(), threading.Event()
    actual_result = object()
    calls = []

    def prove(invocation):
        calls.append(invocation.candidate.job_id)
        started.set()
        assert release.wait(2)
        return ProofExecution(passed=True, value=actual_result)

    scheduler = _scheduler(prove, now)
    try:
        handle = scheduler.submit(_bounded_plan())
        assert started.wait(2)
        if explicit_stop:
            scheduler.stop_dispatch(handle.plan_id, reason="dispatch_budget_exhausted")
        else:
            now[0] = 70.0
            scheduler.expire_deadlines()
        assert not handle.done()
        release.set()
        result = handle.result(2)
        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert calls == ["bounded-0"]
        assert result.attempts_started == 1
        assert result.winner_job_ids == ("bounded-0",)
        assert result.decisions[0].status is ProofDecisionStatus.PASSED
        assert result.decisions[0].value is actual_result
        assert all(d.status is ProofDecisionStatus.NOT_NEEDED for d in result.decisions[1:])
        assert all(d.reason == "dispatch_budget_exhausted" for d in result.decisions[1:])
        assert scheduler.snapshot()["state"] == "running"
    finally:
        release.set()
        assert scheduler.close()


def test_three_environments_get_round_robin_service_before_shared_cutoff():
    now = [10.0]
    entered = Queue()
    advance = threading.Semaphore(0)

    def prove(invocation):
        entered.put(invocation.environment)
        assert advance.acquire(timeout=2)
        return True

    scheduler = _scheduler(prove, now)
    try:
        handles = scheduler.submit_many(tuple(
            _bounded_plan(environment, environment, count=2)
            for environment in ENVIRONMENTS
        ))
        order = []
        for index in range(3):
            order.append(entered.get(timeout=2))
            if index < 2:
                advance.release()
        assert tuple(order) == ENVIRONMENTS
        now[0] = 70.0
        scheduler.expire_deadlines()
        advance.release()
        for handle in handles:
            result = handle.result(2)
            assert result.outcome is ProofPlanOutcome.COMPLETED
            assert result.attempts_started == 1
            assert [d.status for d in result.decisions] == [
                ProofDecisionStatus.PASSED, ProofDecisionStatus.NOT_NEEDED,
            ]
            assert result.decisions[1].reason == "dispatch_budget_exhausted"
        assert entered.empty()
        assert scheduler.snapshot()["state"] == "running"
    finally:
        for _ in range(6):
            advance.release()
        assert scheduler.close()


def test_dispatch_cutoff_does_not_hide_an_active_hard_deadline_overrun():
    now = [10.0]
    started, release = threading.Event(), threading.Event()

    def prove(_invocation):
        started.set()
        assert release.wait(2)
        return True

    scheduler = _scheduler(prove, now)
    try:
        handle = scheduler.submit(_bounded_plan())
        assert started.wait(2)
        now[0] = 70.0
        scheduler.expire_deadlines()
        now[0] = 110.0
        scheduler.expire_deadlines()
        release.set()
        result = handle.result(2)
        assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
        assert result.abort_reason is CapacityAbortReason.ACTIVE_PROOF_TIMEOUT
        assert result.winner_job_ids == ()
        assert scheduler.snapshot()["state"] == "faulted"
    finally:
        release.set()
        assert scheduler.close()


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf"), 110.0, 111.0])
def test_nonfinite_or_inverted_dispatch_window_is_refused(cutoff):
    scheduler = _scheduler(lambda _invocation: pytest.fail("must not dispatch"), [10.0])
    try:
        with pytest.raises(ValueError):
            scheduler.submit(replace(_bounded_plan(), dispatch_deadline_at=cutoff))
    finally:
        assert scheduler.close()


@pytest.mark.parametrize("overrides", [
    {"open_ended": False}, {"allow_shortfall": False},
    {"complete_all": True, "required_passes": 0},
])
def test_dispatch_cutoff_cannot_silently_weaken_a_strict_plan(overrides):
    scheduler = _scheduler(lambda _invocation: pytest.fail("must not dispatch"), [10.0])
    try:
        with pytest.raises(ValueError):
            scheduler.submit(replace(_bounded_plan(), **overrides))
    finally:
        assert scheduler.close()


def _bounded_script(source, **overrides):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    environment.update({
        "RELIQUARY_PROTOCOL_PROFILE": "qwen3-4b-base-dapo-reliquary-v1",
        "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED": "1",
        "RELIQUARY_FILL_CLOSED_PROOF_SERVICE_MODE": "bounded",
        "RELIQUARY_FILL_CLOSED_PROOF_DRAIN_SECONDS": "240",
        "RELIQUARY_FILL_CLOSED_MAX_SECONDS": "1800",
    })
    environment.update(overrides)
    return subprocess.run([sys.executable, "-c", source], env=environment,
                          capture_output=True, text=True, timeout=30)


def test_native_precommit_horizon_leaves_upload_grace_before_dispatch_cutoff():
    result = _bounded_script("""
import json
from reliquary import constants as c
from reliquary.validator.proof_capacity import capacity_budget
cutoff=c.FILL_CLOSED_MAX_SECONDS-c.FILL_CLOSED_PROOF_DRAIN_SECONDS
assert c.FILL_CLOSED_PRECOMMIT_SECONDS+c.SUBMISSION_UPLOAD_GRACE_SECONDS <= cutoff
print(json.dumps(capacity_budget()))
""")
    assert result.returncode == 0, result.stderr
    budget = json.loads(result.stdout)
    assert budget["mode"] == "fill_closed_bounded"
    assert budget["drain_seconds"] == 240
    assert budget["proofs_per_environment"] == 512
    assert budget["wall_seconds"] == 1800


@pytest.mark.parametrize("overrides", [
    {"RELIQUARY_FILL_CLOSED_PROOF_DRAIN_SECONDS": "0"},
    {"RELIQUARY_FILL_CLOSED_PROOF_DRAIN_SECONDS": "nan"},
    {"RELIQUARY_FILL_CLOSED_PROOF_DRAIN_SECONDS": "1800"},
    {"RELIQUARY_FILL_CLOSED_PRECOMMIT_SECONDS": "1560"},
    {"RELIQUARY_FILL_CLOSED_PROOF_SERVICE_MODE": "invented"},
])
def test_incoherent_bounded_operator_settings_refuse_import(overrides):
    result = _bounded_script("from reliquary import constants", **overrides)
    assert result.returncode != 0
    assert "FILL_CLOSED" in result.stderr


@pytest.mark.parametrize("cutoff", [-1.0, 10.0])
def test_expired_initial_plan_settles_all_work_without_a_gpu_call(cutoff):
    scheduler = _scheduler(lambda _invocation: pytest.fail("must not dispatch"), [10.0])
    try:
        result = scheduler.submit(replace(_bounded_plan(), dispatch_deadline_at=cutoff)).result(2)
        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert result.attempts_started == 0
        assert len(result.decisions) == 3
        assert all(d.status is ProofDecisionStatus.NOT_NEEDED for d in result.decisions)
        assert {d.reason for d in result.decisions} == {"dispatch_budget_exhausted"}
    finally:
        assert scheduler.close()


def test_native_batcher_retains_active_valid_submission_for_pick_and_releases_pending(monkeypatch):
    """Run the existing fixture proof path; this is unit evidence, not GPU acceptance."""
    from reliquary.validator import batcher as module
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _make_batcher, _request,
    )

    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(module, "FILL_CLOSED_BOUNDED_PROOFS", True)
    monkeypatch.setattr(module, "FILL_CLOSED_PROOF_DISPATCH_SECONDS", 60.0)
    monkeypatch.setattr(module, "FILL_CLOSED_MAX_SECONDS", 100.0)
    monkeypatch.setattr(module, "B_BATCH", 1)
    now = [10.0]
    started, release = threading.Event(), threading.Event()
    executed, emitted = [], []

    def prove(invocation):
        started.set()
        assert release.wait(5)
        result = _execute_scheduler_payload(invocation)
        executed.append(result)
        return result

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",), environments=ENVIRONMENTS,
        proof_callable=prove, checkpoint_revision="", clock=lambda: now[0],
        deadline_poll_seconds=0.005,
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler, time_fn=lambda: now[0])
        batcher.fill_state = module.FillState(budgets={ENVIRONMENTS[0]: 4}, picks_target=4)
        batcher._emit_training_batch_fn = lambda env, groups, *_args: emitted.append((env, groups))
        with scheduler._condition:
            assert batcher.accept_submission(_request(prompt_idx=21, hotkey="a")).accepted
            assert batcher.accept_submission(_request(prompt_idx=22, hotkey="b")).accepted
        assert started.wait(2)
        before = batcher.fill_state.snapshot()
        assert before["admitted"][ENVIRONMENTS[0]] == 2
        assert before["in_flight"][ENVIRONMENTS[0]] == 2
        now[0] = 70.0
        assert batcher.poll_deadline() is False
        assert not batcher.is_sealed()
        release.set()
        handle = batcher._open_proof_plan_handle
        _wait_until(handle.done, timeout=5)
        assert batcher.can_pick() is True  # native reconcile, not a test-side counter update
        snapshot = batcher.fill_state.snapshot()
        assert snapshot["admitted"][ENVIRONMENTS[0]] == 2  # cutoff never refunds demand
        assert snapshot["proven"][ENVIRONMENTS[0]] == 1
        assert snapshot["in_flight"][ENVIRONMENTS[0]] == 0
        decisions = handle.decisions()
        assert [d.status for d in decisions] == [
            ProofDecisionStatus.PASSED, ProofDecisionStatus.NOT_NEEDED,
        ]
        assert decisions[1].reason == "dispatch_budget_exhausted"
        assert len(executed) == 1 and executed[0].passed
        assert decisions[0].value is executed[0].value
        assert batcher.pick_training_batch() is True
        assert emitted == [(ENVIRONMENTS[0], [executed[0].value])]
        assert batcher.can_pick() is False
        # Repeated observation cannot release or credit either reservation twice.
        batcher._reconcile_fill_state_decisions(ENVIRONMENTS[0])
        after = batcher.fill_state.snapshot()
        assert after["admitted"] == snapshot["admitted"]
        assert after["in_flight"] == snapshot["in_flight"]
        assert after["proven"] == snapshot["proven"]
    finally:
        release.set()
        assert scheduler.close()


@pytest.mark.parametrize("offset", [60.0, 60.001])
@pytest.mark.parametrize("prepared_route", [False, True])
def test_native_admission_at_dispatch_cutoff_cancels_reservation(monkeypatch, offset, prepared_route):
    from reliquary.validator import batcher as module
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(module, "FILL_CLOSED_BOUNDED_PROOFS", True)
    monkeypatch.setattr(module, "FILL_CLOSED_PROOF_DISPATCH_SECONDS", 60.0)
    now = [10.0]
    batcher = _make_batcher(time_fn=lambda: now[0], collection_seconds=100.0)
    batcher.fill_state = module.FillState(budgets={ENVIRONMENTS[0]: 4}, picks_target=16)
    before = batcher.fill_state.snapshot()
    request = _request(prompt_idx=21, hotkey="late")
    assert batcher.try_reserve_logical_group(request) == (True, None)
    assert batcher.logical_group_reservation_count == 1
    now[0] += offset
    if prepared_route:
        from reliquary.validator.admission import PreparedSubmission
        prepared = PreparedSubmission(request=request, completion_texts=[],
            rewards=[r.reward for r in request.rollouts], rollout_hashes=[],
            selection_digest=None, prompt_content_sha256="a" * 64,
            target_content_sha256="b" * 64)
        response = batcher.accept_prepared_submission(prepared)
    else:
        response = batcher.accept_submission(request)
    assert response.accepted is False
    assert response.reason is module.RejectReason.BATCH_FILLED
    assert batcher.fill_state.snapshot() == before
    assert batcher._open_proof_plan_handle is None
    assert batcher.logical_group_reservation_count == 0
    assert batcher.rejected_submissions[-1].reject_stage == "proof_dispatch_closed"


def test_timely_native_precommit_keeps_its_upload_grace_but_late_precommit_is_refused():
    result = _bounded_script("""
from reliquary import constants as c
from tests.unit.test_grpo_window_batcher import _make_batcher
now,wall=[1000.0],[10000.0]
b=_make_batcher(time_fn=lambda:now[0],wall_clock_fn=lambda:wall[0],collection_seconds=c.FILL_CLOSED_PRECOMMIT_SECONDS)
collection_wall=b.window_opened_wall_ts+b.collection_seconds
now[0]=b.window_opened_at+b.collection_seconds-1
accepted,reason,deadline=b.try_register_upload_precommit('timely','miner',t_arrival_wall=collection_wall-1,payload_bytes=100)
assert accepted and reason is None
assert deadline <= b.window_opened_at+c.FILL_CLOSED_PROOF_DISPATCH_SECONDS
late=b.try_register_upload_precommit('late','other',t_arrival_wall=collection_wall+.001,payload_bytes=100)
assert late == (False,'collection_closed',None)
# Upload starts before collection ends; remaining bytes retain completion grace.
assert b.account_upload_precommit_bytes('timely',t_arrival_wall=collection_wall-.5,chunk_bytes=1)==(True,None)
now[0]=b.window_opened_at+b.collection_seconds+.1
assert b.account_upload_precommit_bytes('timely',t_arrival_wall=collection_wall+.1,chunk_bytes=99)==(True,None)
assert b.mark_upload_precommit_transport_complete('timely',completed_at_wall=collection_wall+.1)==(True,None)
assert b.reserved_payload_bytes == 100
assert b.upload_precommit_conservation()['conserved']
""")
    assert result.returncode == 0, result.stderr


# Reuse the established typed-wire/manifest fixture, not fabricated release receipts.
from tests.unit.test_proof_capacity_combined import evidence as capacity_evidence  # noqa: E402

evidence = capacity_evidence


def _bounded_capacity_value(evidence, monkeypatch, *, drain_seconds=240.0):
    from reliquary import constants as constants
    from tests.unit.test_proof_capacity_combined import manifest

    monkeypatch.setattr(constants, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(constants, "FILL_CLOSED_BOUNDED_PROOFS", True)
    monkeypatch.setattr(constants, "FILL_CLOSED_PROOF_SERVICE_MODE", "bounded")
    monkeypatch.setattr(constants, "FILL_CLOSED_MAX_SECONDS", 1800.0)
    monkeypatch.setattr(constants, "FILL_CLOSED_PROOF_DRAIN_SECONDS", drain_seconds)
    monkeypatch.setattr(constants, "FILL_CLOSED_PROOF_DISPATCH_SECONDS", 1800.0 - drain_seconds)
    monkeypatch.setattr(constants, "FILL_CLOSED_PRECOMMIT_SECONDS",
                        1800.0 - drain_seconds - constants.SUBMISSION_UPLOAD_GRACE_SECONDS)
    monkeypatch.setattr(constants, "FILL_CLOSED_ADMISSION_BUDGET_PER_ENV", 512)
    value = manifest(evidence[2]())
    value.update(service_mode="bounded", proof_wall_seconds=1800.0,
                 proofs_per_environment={env: 512 for env in ENVIRONMENTS})
    return value


def _activate_bounded(value, **overrides):
    from tests.unit.test_proof_capacity_combined import activate

    return activate(value, proof_wall_seconds=1800.0,
                    minimum_proofs_per_environment=512, **overrides)


def test_bounded_qualification_states_its_real_limited_guarantee(evidence, monkeypatch):
    value = _bounded_capacity_value(evidence, monkeypatch)
    result = _activate_bounded(value)
    assert result["qualified"] is True
    assert result["service_mode"] == "bounded"
    assert result["all_admitted_proofs_qualified"] is False
    assert result["guaranteed_picks_per_window"] == 0
    assert result["qualified_group_seconds_with_headroom"] == 3.5 / .8
    assert result["required_device_seconds"] is None
    assert result["minimum_device_count"] is None
    assert result["all_admission_demand_device_seconds"] > result["available_device_seconds"]


def test_insufficient_drain_margin_is_refused_even_with_correct_bound_receipts(evidence, monkeypatch):
    from reliquary.validator.proof_capacity import ProofCapacityQualificationError

    value = _bounded_capacity_value(evidence, monkeypatch, drain_seconds=4.0)
    with pytest.raises(ProofCapacityQualificationError, match="drain margin"):
        _activate_bounded(value)


@pytest.mark.parametrize("mismatch", ["strict_manifest", "schema3", "runtime_drain", "runtime_cpu", "runtime_utility"])
def test_bounded_qualification_rejects_mismatched_contract_or_runtime(evidence, monkeypatch, mismatch):
    from reliquary import constants as constants
    from reliquary.validator import proof_capacity_combined, proof_stress
    from reliquary.validator.proof_capacity import ProofCapacityQualificationError
    from tests.unit.test_proof_capacity_combined import CONTROLLER

    value = _bounded_capacity_value(evidence, monkeypatch)
    if mismatch == "strict_manifest":
        value["service_mode"] = "strict"
    elif mismatch == "schema3":
        value["schema_version"] = 3
    elif mismatch == "runtime_drain":
        monkeypatch.setattr(constants, "FILL_CLOSED_PROOF_DRAIN_SECONDS", 239.0)
    elif mismatch == "runtime_cpu":
        monkeypatch.setattr(proof_capacity_combined, "controller_identity",
                            lambda: {**CONTROLLER, "cpu.max": "200000 100000"})
    else:
        monkeypatch.setattr(proof_stress, "runtime_settings",
                            lambda: {"counterfactual": False, "utility_telemetry_enabled": True})
    with pytest.raises(ProofCapacityQualificationError):
        _activate_bounded(value)


def test_cutoff_racing_after_reserve_before_extend_releases_once_without_refund(monkeypatch):
    from reliquary.validator import batcher as module
    from tests.unit.test_grpo_window_batcher import _execute_scheduler_payload, _make_batcher, _request

    monkeypatch.setattr(module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(module, "FILL_CLOSED_BOUNDED_PROOFS", True)
    monkeypatch.setattr(module, "FILL_CLOSED_PROOF_DISPATCH_SECONDS", 60.0)
    monkeypatch.setattr(module, "FILL_CLOSED_MAX_SECONDS", 100.0)
    monkeypatch.setattr(module, "B_BATCH", 1)
    now = [10.0]
    scheduler = GlobalProofScheduler(devices=("gpu-0",), environments=ENVIRONMENTS,
        proof_callable=_execute_scheduler_payload, checkpoint_revision="", clock=lambda: now[0])
    try:
        batcher = _make_batcher(proof_scheduler=scheduler, time_fn=lambda: now[0])
        batcher.fill_state = module.FillState(budgets={ENVIRONMENTS[0]: 4}, picks_target=4)
        assert batcher.accept_submission(_request(prompt_idx=21, hotkey="a")).accepted
        handle = batcher._open_proof_plan_handle
        _wait_until(lambda: len(handle.decisions()) == 1, timeout=5)
        assert not handle.done()
        original_extend = batcher._extend_proof_plan

        def cross_cutoff(candidates):
            # Exact race: native reservation has happened; extension has not.
            assert batcher.fill_state.snapshot()["admitted"][ENVIRONMENTS[0]] == 2
            now[0] = 70.0
            scheduler.expire_deadlines()
            original_extend(candidates)

        monkeypatch.setattr(batcher, "_extend_proof_plan", cross_cutoff)
        assert batcher.accept_submission(_request(prompt_idx=22, hotkey="b")).accepted
        assert handle.done()
        batcher._reconcile_fill_state_decisions(ENVIRONMENTS[0])
        snapshot = batcher.fill_state.snapshot()
        assert snapshot["admitted"][ENVIRONMENTS[0]] == 2
        assert snapshot["proven"][ENVIRONMENTS[0]] == 1
        assert snapshot["in_flight"][ENVIRONMENTS[0]] == 0
        assert batcher._arrival_proof_meta == {}
        assert len(handle.decisions()) == 1
        assert handle.result().attempts_started == 1
        assert scheduler.snapshot()["state"] == "running"
    finally:
        assert scheduler.close()


def test_early_fill_stops_surplus_proofs_before_close(monkeypatch):
    from reliquary.validator import batcher as module
    from tests.unit.test_grpo_window_batcher import _execute_scheduler_payload, _make_batcher, _request

    for name, value in (("FILL_CLOSED_ENABLED", True), ("FILL_CLOSED_BOUNDED_PROOFS", True),
                        ("FILL_CLOSED_PROOF_DISPATCH_SECONDS", 60.0),
                        ("FILL_CLOSED_MAX_SECONDS", 100.0), ("B_BATCH", 1)):
        monkeypatch.setattr(module, name, value)
    now = [10.0]
    executed, emitted = [], []

    def prove(invocation):
        result = _execute_scheduler_payload(invocation)
        executed.append(result)
        return result

    scheduler = GlobalProofScheduler(devices=("gpu-0",), environments=ENVIRONMENTS,
        proof_callable=prove, checkpoint_revision="", clock=lambda: now[0])
    try:
        batcher = _make_batcher(proof_scheduler=scheduler, time_fn=lambda: now[0])
        batcher.fill_state = module.FillState(budgets={ENVIRONMENTS[0]: 4}, picks_target=1)
        batcher._emit_training_batch_fn = lambda env, groups, *_args: emitted.append((env, groups))
        with scheduler._condition:
            for prompt in (21, 22, 23):
                assert batcher.accept_submission(_request(prompt_idx=prompt, hotkey=str(prompt))).accepted
        handle = batcher._open_proof_plan_handle
        _wait_until(handle.done, timeout=5)
        assert batcher.can_pick() and batcher.pick_training_batch()
        assert batcher.poll_deadline() is True
        assert batcher.is_sealed()
        assert len(executed) == 1 and executed[0].passed
        assert [d.status for d in handle.decisions()] == [
            ProofDecisionStatus.PASSED,
            ProofDecisionStatus.NOT_NEEDED,
            ProofDecisionStatus.NOT_NEEDED,
        ]
        assert batcher.fill_state.snapshot()["in_flight"][ENVIRONMENTS[0]] == 0
        assert batcher.fill_state.snapshot()["admitted"][ENVIRONMENTS[0]] == 3
        assert emitted == [(ENVIRONMENTS[0], [executed[0].value])]
        assert batcher._burned_unpicked_groups == 0
        assert len(batcher._proven_groups[ENVIRONMENTS[0]]) == 1
    finally:
        assert scheduler.close()
