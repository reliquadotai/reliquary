from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import threading
import time

import pytest

from reliquary.validator.proof_scheduler import (
    CapacityAbortReason,
    CheckpointNotReady,
    DeviceNotReady,
    GlobalProofScheduler,
    ProofDecisionStatus,
    ProofExecution,
    ProofPlan,
    ProofPlanOutcome,
    RankedProof,
    SchedulerNotRunning,
    SchedulerState,
)


MATH = "openmathinstruct"
CODE = "opencodeinstruct"


def _candidate(
    rank: int,
    *,
    prompt: str | None = None,
    prefix: str = "job",
    resources: tuple[tuple[str, int], ...] = (),
) -> RankedProof:
    return RankedProof(
        job_id=f"{prefix}-{rank}",
        rank=rank,
        prompt_key=prompt or f"prompt-{rank}",
        payload={"rank": rank},
        resources=resources,
    )


def _plan(
    plan_id: str,
    environment: str,
    candidates: list[RankedProof],
    *,
    required: int,
    deadline: float | None = None,
    revision: str = "rev-a",
    max_attempts: int | None = None,
) -> ProofPlan:
    return ProofPlan(
        plan_id=plan_id,
        environment=environment,
        checkpoint_revision=revision,
        candidates=candidates,
        required_passes=required,
        deadline_at=time.monotonic() + 10 if deadline is None else deadline,
        max_attempts=max_attempts,
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.005)


def test_one_worker_per_device_prevents_same_device_overlap():
    lock = threading.Lock()
    release = threading.Event()
    active = defaultdict(int)
    max_active = defaultdict(int)
    global_max = 0

    def prove(invocation):
        nonlocal global_max
        with lock:
            active[invocation.device_id] += 1
            max_active[invocation.device_id] = max(
                max_active[invocation.device_id],
                active[invocation.device_id],
            )
            global_max = max(global_max, sum(active.values()))
        assert release.wait(2)
        with lock:
            active[invocation.device_id] -= 1
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0", "gpu-1"),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(i) for i in range(4)],
                required=4,
            )
        )
        _wait_until(
            lambda: sum(
                entry is not None
                for entry in scheduler.snapshot()["active_by_device"].values()
            )
            == 2
        )
        release.set()
        result = handle.result(2)

        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert global_max == 2
        assert max_active == {"gpu-0": 1, "gpu-1": 1}
    finally:
        release.set()
        assert scheduler.close()


def test_shared_resource_is_serialized_across_devices():
    release_first = threading.Event()
    second_started = threading.Event()
    unrelated_started = threading.Event()

    def prove(invocation):
        rank = invocation.candidate.rank
        if rank == 1:
            assert release_first.wait(2)
        elif rank == 2:
            second_started.set()
        else:
            unrelated_started.set()
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0", "gpu-1"),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [
                    _candidate(
                        1,
                        resources=(("operator-a", 4),),
                    ),
                    _candidate(
                        2,
                        resources=(("operator-a", 4),),
                    ),
                    _candidate(
                        3,
                        resources=(("operator-b", 4),),
                    ),
                ],
                required=3,
            )
        )
        assert unrelated_started.wait(2)
        assert not second_started.is_set()
        release_first.set()
        assert second_started.wait(2)
        assert handle.result(2).outcome is ProofPlanOutcome.COMPLETED
    finally:
        release_first.set()
        assert scheduler.close()


def test_resource_failure_limit_skips_identity_and_promotes_other_operator():
    invoked = []

    def prove(invocation):
        invoked.append(invocation.candidate.job_id)
        return invocation.candidate.job_id != "job-1"

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        result = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [
                    _candidate(
                        1,
                        prompt="first",
                        resources=(("operator-a", 1),),
                    ),
                    _candidate(
                        2,
                        prompt="second",
                        resources=(("operator-a", 1),),
                    ),
                    _candidate(
                        3,
                        prompt="second",
                        resources=(("operator-b", 1),),
                    ),
                ],
                required=1,
            )
        ).result(2)

    assert invoked == ["job-1", "job-3"]
    assert result.decisions[1].status is (
        ProofDecisionStatus.SKIPPED_RESOURCE_LIMIT
    )
    assert result.winner_job_ids == ("job-3",)


def test_round_robin_dispatch_is_fair_between_environments():
    dispatch_order = []

    def prove(invocation):
        dispatch_order.append(invocation.environment)
        return True

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        math, code = scheduler.submit_many(
            (
                _plan(
                    "math-window",
                    MATH,
                    [_candidate(i, prefix="math") for i in range(2)],
                    required=2,
                ),
                _plan(
                    "code-window",
                    CODE,
                    [_candidate(i, prefix="code") for i in range(2)],
                    required=2,
                ),
            )
        )
        assert math.result(2).outcome is ProofPlanOutcome.COMPLETED
        assert code.result(2).outcome is ProofPlanOutcome.COMPLETED

    assert dispatch_order == [MATH, CODE, MATH, CODE]


def test_winner_plan_dispatches_before_lower_priority_forensics():
    dispatch_order = []
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def prove(invocation):
        dispatch_order.append(invocation.plan_id)
        if invocation.plan_id == "math-winner":
            blocker_started.set()
            assert release_blocker.wait(2)
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        winner = replace(
            _plan(
                "math-winner",
                MATH,
                [_candidate(1, prefix="winner")],
                required=1,
            ),
            priority=0,
        )
        forensic = replace(
            _plan(
                "code-forensic",
                CODE,
                [_candidate(1, prefix="forensic")],
                required=0,
            ),
            priority=10,
            complete_all=True,
        )
        winner_handle, forensic_handle = scheduler.submit_many(
            (winner, forensic)
        )
        assert blocker_started.wait(2)
        release_blocker.set()
        assert winner_handle.result(2).outcome is (
            ProofPlanOutcome.COMPLETED
        )
        assert forensic_handle.result(2).outcome is (
            ProofPlanOutcome.COMPLETED
        )
    finally:
        release_blocker.set()
        assert scheduler.close()

    assert dispatch_order == ["math-winner", "code-forensic"]


def test_decisions_apply_in_rank_order_not_completion_order():
    rank_one_release = threading.Event()
    rank_two_finished = threading.Event()

    def prove(invocation):
        if invocation.candidate.rank == 1:
            assert rank_one_release.wait(2)
        else:
            rank_two_finished.set()
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0", "gpu-1"),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1), _candidate(2)],
                required=2,
            )
        )
        assert rank_two_finished.wait(2)
        assert handle.decisions() == ()
        rank_one_release.set()

        result = handle.result(2)
        assert [decision.rank for decision in result.decisions] == [1, 2]
        assert result.winner_job_ids == ("job-1", "job-2")
    finally:
        rank_one_release.set()
        assert scheduler.close()


def test_unique_prompt_fallback_waits_for_higher_ranked_candidate():
    leader_release = threading.Event()
    fallback_started = threading.Event()
    other_started = threading.Event()
    starts = []
    lock = threading.Lock()

    def prove(invocation):
        rank = invocation.candidate.rank
        with lock:
            starts.append(rank)
        if rank == 1:
            assert leader_release.wait(2)
            return ProofExecution(False, reason="fabricated")
        if rank == 2:
            fallback_started.set()
        if rank == 3:
            other_started.set()
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0", "gpu-1"),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [
                    _candidate(1, prompt="same"),
                    _candidate(2, prompt="same"),
                    _candidate(3, prompt="other"),
                ],
                required=2,
            )
        )
        assert other_started.wait(2)
        assert not fallback_started.is_set()
        leader_release.set()
        assert fallback_started.wait(2)

        result = handle.result(2)
        assert [decision.status for decision in result.decisions] == [
            ProofDecisionStatus.REJECTED,
            ProofDecisionStatus.PASSED,
            ProofDecisionStatus.PASSED,
        ]
        assert starts.index(1) < starts.index(2)
    finally:
        leader_release.set()
        assert scheduler.close()


def test_passing_prompt_leader_skips_same_prompt_tail():
    invoked = []

    def prove(invocation):
        invoked.append(invocation.candidate.job_id)
        return True

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        result = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [
                    _candidate(1, prompt="same"),
                    _candidate(2, prompt="same"),
                    _candidate(3, prompt="other"),
                ],
                required=2,
            )
        ).result(2)

    assert invoked == ["job-1", "job-3"]
    assert result.decisions[1].status is (
        ProofDecisionStatus.SKIPPED_PROMPT_CLAIMED
    )


def test_checkpoint_revision_requires_every_device_to_be_ready():
    scheduler = GlobalProofScheduler(
        devices=("gpu-0", "gpu-1"),
        environments=(MATH, CODE),
        proof_callable=lambda _invocation: True,
        checkpoint_revision="rev-a",
    )
    try:
        with pytest.raises(CheckpointNotReady):
            scheduler.submit(
                _plan(
                    "wrong-revision",
                    MATH,
                    [_candidate(1)],
                    required=1,
                    revision="rev-b",
                )
            )

        assert scheduler.drain(1)
        scheduler.mark_device_ready("gpu-0", "rev-b")
        with pytest.raises(DeviceNotReady):
            scheduler.resume("rev-b")
        scheduler.mark_device_ready("gpu-1", "rev-b")
        scheduler.resume("rev-b")

        assert scheduler.checkpoint_ready("rev-b")
        result = scheduler.submit(
            _plan(
                "new-revision",
                MATH,
                [_candidate(1)],
                required=1,
                revision="rev-b",
            )
        ).result(2)
        assert result.checkpoint_revision == "rev-b"
    finally:
        assert scheduler.close()


def test_drain_refuses_new_work_and_finishes_the_queue():
    release = threading.Event()
    started = threading.Event()

    def prove(_invocation):
        started.set()
        assert release.wait(2)
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1)],
                required=1,
            )
        )
        assert started.wait(2)
        assert scheduler.drain(0.01) is False
        assert scheduler.state is SchedulerState.DRAINING
        with pytest.raises(SchedulerNotRunning):
            scheduler.submit(
                _plan(
                    "code-window",
                    CODE,
                    [_candidate(1)],
                    required=1,
                )
            )
        release.set()
        assert handle.result(2).outcome is ProofPlanOutcome.COMPLETED
        assert scheduler.drain(1)
        assert scheduler.state is SchedulerState.QUIESCED
    finally:
        release.set()
        assert scheduler.close()


def test_quiesce_aborts_pending_work_and_waits_for_active_call():
    release = threading.Event()
    started = threading.Event()

    def prove(_invocation):
        started.set()
        assert release.wait(2)
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1), _candidate(2)],
                required=2,
            )
        )
        assert started.wait(2)
        assert scheduler.quiesce(0.01) is False
        assert scheduler.state is SchedulerState.QUIESCING
        release.set()

        result = handle.result(2)
        assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
        assert result.abort_reason is CapacityAbortReason.QUIESCED
        _wait_until(lambda: scheduler.state is SchedulerState.QUIESCED)
        assert all(
            decision.status is ProofDecisionStatus.CAPACITY_ABORTED
            for decision in result.decisions
        )
    finally:
        release.set()
        assert scheduler.close()


def test_active_proof_deadline_faults_without_waiting_for_cuda_return():
    class ManualClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = ManualClock()
    release = threading.Event()
    started = threading.Event()

    def prove(_invocation):
        started.set()
        assert release.wait(2)
        return True

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
        clock=clock,
    )
    try:
        handle = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1)],
                required=1,
                deadline=10.0,
            )
        )
        assert started.wait(2)
        clock.now = 11.0
        scheduler.expire_deadlines()

        result = handle.result(2)
        assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
        assert result.abort_reason is (
            CapacityAbortReason.ACTIVE_PROOF_TIMEOUT
        )
        assert result.decisions[0].status is (
            ProofDecisionStatus.CAPACITY_ABORTED
        )
        assert scheduler.state is SchedulerState.FAULTED
        assert scheduler.snapshot()["fault_reason"] == (
            CapacityAbortReason.ACTIVE_PROOF_TIMEOUT.value
        )

        release.set()
        _wait_until(
            lambda: (
                scheduler.snapshot()["active_by_device"]["gpu-0"] is None
            )
        )
        assert scheduler.snapshot()["totals"]["late_results"] == 1
    finally:
        release.set()
        assert scheduler.close()


def test_impossible_distinct_prompt_capacity_aborts_without_proving():
    calls = 0

    def prove(_invocation):
        nonlocal calls
        calls += 1
        return True

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        result = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [
                    _candidate(1, prompt="same"),
                    _candidate(2, prompt="same"),
                ],
                required=2,
            )
        ).result(1)

    assert calls == 0
    assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
    assert result.abort_reason is (
        CapacityAbortReason.INSUFFICIENT_DISTINCT_PROMPTS
    )
    assert result.attempts_started == 0


def test_sparse_population_can_complete_with_an_explicit_shortfall():
    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=lambda _invocation: True,
        checkpoint_revision="rev-a",
    ) as scheduler:
        plan = _plan(
            "math-window",
            MATH,
            [_candidate(1)],
            required=2,
        )
        plan = replace(plan, allow_shortfall=True)
        result = scheduler.submit(plan).result(2)

    assert result.outcome is ProofPlanOutcome.COMPLETED
    assert result.abort_reason is None
    assert result.winner_job_ids == ("job-1",)


def test_complete_all_plan_runs_rejected_and_passing_forensic_jobs():
    invoked = []

    def prove(invocation):
        invoked.append(invocation.candidate.job_id)
        return invocation.candidate.rank == 2

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        plan = _plan(
            "math-forensic",
            MATH,
            [_candidate(1), _candidate(2)],
            required=0,
        )
        plan = replace(plan, complete_all=True)
        result = scheduler.submit(plan).result(2)

    assert invoked == ["job-1", "job-2"]
    assert result.outcome is ProofPlanOutcome.COMPLETED
    assert [decision.status for decision in result.decisions] == [
        ProofDecisionStatus.REJECTED,
        ProofDecisionStatus.PASSED,
    ]


def test_attempt_limit_aborts_partial_success_explicitly():
    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=lambda _invocation: True,
        checkpoint_revision="rev-a",
    ) as scheduler:
        result = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1), _candidate(2), _candidate(3)],
                required=2,
                max_attempts=1,
            )
        ).result(2)

    assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
    assert result.abort_reason is CapacityAbortReason.ATTEMPT_LIMIT
    assert result.winner_job_ids == ()
    assert result.attempts_started == 1


def test_proof_exception_faults_scheduler_without_promoting_lower_rank():
    invoked = []

    def prove(invocation):
        invoked.append(invocation.candidate.job_id)
        if invocation.candidate.rank == 1:
            raise RuntimeError("bad proof")
        return ProofExecution(True, value="verified")

    with GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    ) as scheduler:
        result = scheduler.submit(
            _plan(
                "math-window",
                MATH,
                [_candidate(1), _candidate(2)],
                required=1,
            )
        ).result(2)
        snapshot = scheduler.snapshot()

    assert [decision.status for decision in result.decisions] == [
        ProofDecisionStatus.CAPACITY_ABORTED,
        ProofDecisionStatus.CAPACITY_ABORTED,
    ]
    assert invoked == ["job-1"]
    assert result.outcome is ProofPlanOutcome.CAPACITY_ABORTED
    assert result.abort_reason is (
        CapacityAbortReason.PROOF_EXECUTION_ERROR
    )
    assert result.winner_job_ids == ()
    assert snapshot["state"] == SchedulerState.FAULTED.value
    assert snapshot["totals"]["proof_errors"] == 1
    assert snapshot["totals"]["jobs_completed"] == 1
    assert snapshot["dispatches_by_environment"][MATH] == 1
