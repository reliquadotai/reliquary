"""Open-ended plans: prove candidates as they arrive, close on the target.

The fill-closed window design proves each submission on arrival and closes the
window as soon as enough groups are PROVEN. The scheduler already implements
the closing half — ``required_passes`` reached sets ``stop_dispatch`` with
``completion_reason = "target_reached"``. What it cannot do is accept work
after submission: ``candidates`` is frozen, and a plan finalises the moment its
queue drains (``_finalize_if_terminal_locked``), which for a window still
admitting submissions would abort it as INSUFFICIENT_DISTINCT_PROMPTS.

An open-ended plan is the missing concept: it does not finalise on exhaustion,
only on its target, its deadline, or an explicit seal.
"""

from __future__ import annotations

import time

import pytest

from reliquary.validator.proof_scheduler import (
    CapacityAbortReason,
    GlobalProofScheduler,
    ProofPlanOutcome,
)

from tests.unit.test_proof_scheduler import (
    MATH,
    CODE,
    _candidate,
    _plan,
    _wait_until,
)


def _scheduler(prove):
    return GlobalProofScheduler(
        devices=("gpu-0",),
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )


def test_an_open_plan_survives_running_out_of_candidates():
    """A closed plan short of its target aborts; an open one waits for more."""
    scheduler = _scheduler(lambda invocation: True)
    try:
        handle = scheduler.submit(
            _plan("w", MATH, [_candidate(0)], required=4, open_ended=True)
        )
        _wait_until(lambda: len(handle.decisions()) == 1)
        time.sleep(0.05)

        assert not handle.done()
    finally:
        scheduler.close()


def test_extend_feeds_candidates_to_a_running_plan():
    proven: list[int] = []

    def prove(invocation):
        proven.append(invocation.candidate.rank)
        return True

    scheduler = _scheduler(prove)
    try:
        handle = scheduler.submit(
            _plan("w", MATH, [_candidate(0)], required=3, open_ended=True)
        )
        _wait_until(lambda: len(handle.decisions()) == 1)

        scheduler.extend("w", [_candidate(1), _candidate(2)])

        _wait_until(lambda: handle.done())
        assert sorted(proven) == [0, 1, 2]
    finally:
        scheduler.close()


def test_sealing_an_open_plan_lets_it_finalise_short_of_its_target():
    """The backstop: the window closes on its maximum duration, batch unfilled.

    Sealing restores ordinary semantics — exhaustion becomes terminal again —
    so the plan reports the shortfall instead of hanging.
    """
    scheduler = _scheduler(lambda invocation: True)
    try:
        handle = scheduler.submit(
            _plan("w", MATH, [_candidate(0)], required=4, open_ended=True)
        )
        _wait_until(lambda: len(handle.decisions()) == 1)
        assert not handle.done()

        scheduler.seal("w")

        _wait_until(lambda: handle.done())
        assert handle.result().outcome is ProofPlanOutcome.CAPACITY_ABORTED
    finally:
        scheduler.close()


def test_extending_behind_the_applied_rank_is_refused():
    """Decisions are applied in rank order, tracked by ``next_apply_index``.

    A candidate landing behind that index would be applied never or twice, so
    arrival order must keep producing increasing ranks.
    """
    scheduler = _scheduler(lambda invocation: True)
    try:
        scheduler.submit(
            _plan(
                "w", MATH, [_candidate(5)], required=4, open_ended=True
            )
        )
        with pytest.raises(ValueError, match="rank after"):
            scheduler.extend("w", [_candidate(3)])
    finally:
        scheduler.close()


def test_an_open_plan_still_aborts_on_its_deadline():
    """The backstop must survive open-endedness.

    Open-endedness only makes EXHAUSTION non-terminal. A window that runs past
    its maximum duration still has to end, or a stalled fleet would hold the
    plan open forever.
    """
    scheduler = _scheduler(lambda invocation: True)
    try:
        handle = scheduler.submit(
            _plan(
                "w",
                MATH,
                [_candidate(0)],
                required=4,
                deadline=time.monotonic() + 0.05,
                open_ended=True,
            )
        )
        _wait_until(lambda: handle.done(), timeout=3.0)

        assert handle.result().abort_reason is CapacityAbortReason.DEADLINE_EXCEEDED
    finally:
        scheduler.close()
