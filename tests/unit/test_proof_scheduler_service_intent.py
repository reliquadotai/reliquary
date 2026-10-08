from dataclasses import replace
import threading
import time

import pytest

from reliquary.validator.proof_scheduler import (
    CapacityAbortReason, GlobalProofScheduler, ProofDecisionStatus, ProofPlan,
    ProofPlanOutcome, RankedProof, SchedulerState,
)


def candidate(rank, training=True, prompt=None, resources=()):
    return RankedProof(str(rank), rank, prompt or str(rank), None, resources,
                       counts_toward_target=training)


def plan(candidates, *, required=1, allow_shortfall=False, **changes):
    return ProofPlan("mixed", "env", "checkpoint", candidates, required,
                     time.monotonic() + 5, allow_shortfall=allow_shortfall, **changes)


def scheduler(prove, devices=("cpu-a",), **kwargs):
    return GlobalProofScheduler(devices=devices, environments=("env",), proof_callable=prove,
                                 checkpoint_revision="checkpoint", deadline_poll_seconds=0.01, **kwargs)


def test_exploration_pass_does_not_fill_training_target():
    with scheduler(lambda call: True) as service:
        result = service.submit(plan([candidate(0, False), candidate(1), candidate(2)])).result(1)
    assert result.winner_job_ids == ("1",)
    assert result.attempts_started == 2
    assert [d.status for d in result.decisions] == [ProofDecisionStatus.PASSED,
                                                   ProofDecisionStatus.PASSED, ProofDecisionStatus.NOT_NEEDED]
    assert result.decisions[1].details["proof_plan_passes_before_decision"] == 0


def test_active_exploration_reserves_no_training_slot():
    release, explore_started, training_started = threading.Event(), threading.Event(), threading.Event()
    def prove(call):
        if call.candidate.rank == 0:
            explore_started.set()
            assert release.wait(2)
        else:
            training_started.set()
        return True
    service = scheduler(prove, devices=("cpu-a", "cpu-b"))
    try:
        handle = service.submit(plan([candidate(0, False), candidate(1)]))
        assert explore_started.wait(1)
        assert training_started.wait(1)
        assert not handle.done()
        release.set()
        result = handle.result(1)
        assert result.winner_job_ids == ("1",)
        assert [d.status for d in result.decisions] == [ProofDecisionStatus.PASSED] * 2
    finally:
        release.set()
        assert service.close()


def test_raw_exploration_reserves_no_training_slot_behind_training_rank_gap():
    release, leader_started, explore_finished, fallback_started = [threading.Event() for _ in range(4)]
    def prove(call):
        rank = call.candidate.rank
        if rank == 0:
            leader_started.set()
            assert release.wait(2)
            return False
        if rank == 1:
            fallback_started.set()
        else:
            explore_finished.set()
        return True
    service = scheduler(prove, devices=("cpu-a", "cpu-b"))
    try:
        handle = service.submit(plan([candidate(0, prompt="same"), candidate(1, prompt="same"),
                                      candidate(2, False)]))
        assert leader_started.wait(1)
        assert explore_finished.wait(1)
        release.set()
        assert fallback_started.wait(1)
        result = handle.result(1)
        assert result.winner_job_ids == ("1",)
        assert [d.status for d in result.decisions] == [ProofDecisionStatus.REJECTED,
                                                       ProofDecisionStatus.PASSED, ProofDecisionStatus.PASSED]
        assert result.attempts_started == 3
    finally:
        release.set()
        assert service.close()


def test_exploration_keeps_prompt_claim_and_failure_debt_effects():
    calls = []
    def prove(call):
        calls.append(call.candidate.rank)
        return call.candidate.rank != 2
    with scheduler(prove) as service:
        result = service.submit(plan([
            candidate(0, False, prompt="claimed"), candidate(1, prompt="claimed"),
            candidate(2, False, resources=(("operator", 1),)),
            candidate(3, resources=(("operator", 1),)), candidate(4),
        ], allow_shortfall=True)).result(1)
    assert calls == [0, 2, 4]
    assert result.winner_job_ids == ("4",)
    assert [d.status for d in result.decisions] == [ProofDecisionStatus.PASSED,
        ProofDecisionStatus.SKIPPED_PROMPT_CLAIMED, ProofDecisionStatus.REJECTED,
        ProofDecisionStatus.SKIPPED_RESOURCE_LIMIT, ProofDecisionStatus.PASSED]


def test_shortfall_waits_for_all_pending_and_active_exploration_proofs():
    release, started = threading.Event(), threading.Event()
    calls = []
    def prove(call):
        calls.append(call.candidate.rank)
        if call.candidate.rank == 0:
            started.set()
            assert release.wait(2)
        return True
    service = scheduler(prove)
    try:
        handle = service.submit(plan([candidate(0, False), candidate(1, False)], allow_shortfall=True))
        assert started.wait(1)
        assert not handle.done()
        release.set()
        result = handle.result(1)
        assert calls == [0, 1]
        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert result.winner_job_ids == ()
        assert [d.status for d in result.decisions] == [ProofDecisionStatus.PASSED] * 2
    finally:
        release.set()
        assert service.close()


def test_complete_all_keeps_existing_pass_and_winner_behavior():
    with scheduler(lambda call: True) as service:
        result = service.submit(plan([candidate(0, False), candidate(1)], required=0,
                                      complete_all=True)).result(1)
    assert result.winner_job_ids == ("0", "1")
    assert result.decisions[1].details["proof_plan_passes_before_decision"] == 1


def test_observation_attempt_limit_does_not_leave_shortfall_plan_stuck():
    with scheduler(lambda call: True) as service:
        result = service.submit(plan([candidate(0, False), candidate(1, False)],
                                      allow_shortfall=True, max_attempts=1)).result(1)
    assert result.abort_reason is CapacityAbortReason.ATTEMPT_LIMIT
    assert result.attempts_started == 1


def test_observation_at_hard_deadline_faults_and_late_result_cannot_win():
    release, started, returned = [threading.Event() for _ in range(3)]
    clock = type("Clock", (), {"now": 0.0, "__call__": lambda self: self.now})()
    def prove(call):
        started.set()
        assert release.wait(2)
        returned.set()
        return True
    service = scheduler(prove, clock=clock)
    try:
        handle = service.submit(replace(plan([candidate(0, False)], allow_shortfall=True), deadline_at=10))
        assert started.wait(1)
        clock.now = 11
        service.expire_deadlines()
        result = handle.result(1)
        assert result.abort_reason is CapacityAbortReason.ACTIVE_PROOF_TIMEOUT
        assert result.winner_job_ids == ()
        assert result.decisions[0].status is ProofDecisionStatus.CAPACITY_ABORTED
        assert service.state is SchedulerState.FAULTED
        release.set()
        assert returned.wait(1)
    finally:
        release.set()
        assert service.close()


def test_non_boolean_target_flags_are_rejected():
    with scheduler(lambda call: True) as service:
        with pytest.raises(ValueError, match="must be a bool"):
            service.submit(plan([replace(candidate(0), counts_toward_target=0)]))
