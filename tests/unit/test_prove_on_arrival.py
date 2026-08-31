"""Under v6 a submission is proven when it arrives, not at seal."""
import hashlib

from reliquary.validator.batcher import PendingSubmission

from tests.unit.test_grpo_window_batcher import _make_batcher


def _pending_stub(
    prompt_idx: int,
    *,
    rewards=None,
    truncated_index=None,
    truncated_count=0,
    attainable_rewards=(),
) -> PendingSubmission:
    """Copied from ``tests.unit.test_batch_fill_offset._pending``, extended
    with the truncation/attainable-rewards fields the arrival-proof gate
    reads off ``pending`` (see ``robust_utility_admits``)."""
    root = str(prompt_idx).encode().ljust(32, b"\x00")
    if rewards is None:
        rewards = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return PendingSubmission(
        hotkey=f"hk{prompt_idx}",
        prompt_idx=prompt_idx,
        request=None,
        rewards=rewards,
        drand_round=1,
        merkle_root=root,
        selection_digest=root,
        prompt_content_sha256=hashlib.sha256(
            f"prompt:{prompt_idx}".encode()
        ).hexdigest(),
        target_content_sha256=hashlib.sha256(b"target").hexdigest(),
        truncated_index=truncated_index,
        truncated_count=truncated_count,
        attainable_rewards=attainable_rewards,
    )


def test_accept_submission_wires_the_arrival_proof_call(monkeypatch):
    """The unit tests above call ``_submit_arrival_proof`` directly; this
    proves it is actually reached from the real admission entrypoint at both
    ``_pending.append`` sites' shared caller (``accept_submission`` ->
    ``_accept_locked``), not just callable in isolation."""
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import _request

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    response = batcher.accept_submission(_request(prompt_idx=9, hotkey="miner"))

    assert response.accepted is True
    assert len(extended) == 1
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1


def test_an_arriving_submission_is_extended_onto_the_open_plan(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=1))

    assert len(extended) == 1
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1


def test_a_manufactured_zero_is_refused_before_capacity_is_reserved(monkeypatch):
    """robust_utility_admits is the admission-time analogue of the auction's
    least-favourable pricing: a truncated rollout that COULD have been
    correct must not buy a reservation. Refusal must not touch fill_state."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    pending = _pending_stub(
        prompt_idx=2,
        rewards=[1.0] * 15 + [0.0],
        truncated_index=15,
        truncated_count=1,
        attainable_rewards=(0.0, 1.0),
    )

    batcher._submit_arrival_proof(pending)

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 0


def test_a_full_environment_refuses_without_reserving(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    batcher.fill_state.reserve("openmathinstruct")  # already at target

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=3))

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1


def test_a_buffered_group_is_not_reserved_on_arrival(monkeypatch):
    """A body that arrives while the environment is at budget sits in the
    buffer; buffering must reserve nothing."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    env = "openmathinstruct"
    batcher.fill_state.reserve(env)  # already at budget

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=41))

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"][env] == 1
    assert len(batcher._arrival_proof_buffer) == 1


def test_a_buffered_group_stays_buffered_once_the_budget_is_spent(monkeypatch):
    """R33: ``admitted`` is monotone -- a release frees in-flight PROOF
    capacity (a failed proof no longer occupies a proof-worker slot) but
    never refunds the budget a reservation already spent. A body that
    landed while the environment was at budget therefore stays buffered
    even after something else releases: the grading cost the budget paid
    for was real, and does not come back on someone else's failure."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    env = "openmathinstruct"
    batcher.fill_state.reserve(env)  # already at budget

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=41))
    assert len(batcher._arrival_proof_buffer) == 1

    batcher.fill_state.release(env)  # frees in_flight, NOT the budget
    batcher._drain_arrival_proof_buffer(env)

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"][env] == 0
    assert len(batcher._arrival_proof_buffer) == 1


def test_ranks_handed_to_extend_strictly_increase_across_drains(monkeypatch):
    """``extend`` refuses a rank at or behind the plan's current highest
    (``proof_scheduler.py``'s ``next_apply_index`` invariant). The rank
    counter must keep climbing across every drain, not just within one.

    R33: a release no longer reopens the budget (``admitted`` is
    monotone), so ample budget -- not a release -- is what lets each of
    the three submissions below drain on its own turn, one separate
    ``_drain_arrival_proof_buffer`` call per submission."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 3, "opencodeinstruct": 3}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=61))  # drains now
    batcher._submit_arrival_proof(_pending_stub(prompt_idx=62))  # drains now
    batcher._submit_arrival_proof(_pending_stub(prompt_idx=63))  # drains now

    assert len(extended) == 3
    ranks = [candidate.rank for candidate in extended]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_an_unknown_rate_falls_back_to_lowest_priority_not_a_crash(monkeypatch):
    """``rate_of`` misses when a receipt was never offered (or was offered
    in a different window). The group must still drain -- after every
    known-rate group, however small its rate -- rather than raise.

    Driven through the REAL intake (``_submit_arrival_proof`` doing its
    own ``rate_of``/``payload_bytes_of`` lookups), with only the
    auto-drain held off so both bodies sit in the buffer together, the
    way a genuine concurrent race leaves them -- the same shape
    ``test_rate_ordered_admission`` uses. That also covers the miss
    fallback on the SIZE: the queue has no entry for the unknown
    receipt, so ``payload_bytes`` degrades to the request's own accounted
    size (0 for a body that was never precommitted) instead of raising.

    R33: a release no longer reopens the budget, so the buffer cannot be
    filled by arrival + release; holding the drain off is what produces
    two competing entries now."""
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_rate_ordered_admission import _pending_for_receipt

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.mark_window_opened()
    env = "openmathinstruct"
    batcher.admission_queue = batcher_module.ThroughputAdmissionQueue(
        window_opened_at=batcher.window_opened_at
    )
    # Only the "known" receipt was ever offered: 1000 bytes over 1000 s
    # of window, a rate of 1.0 -- feeble, but known.
    batcher.admission_queue.offer(
        receipt_id="known", environment=env, payload_bytes=1_000,
        precommit_arrived_at=batcher.window_opened_at + 1_000.0,
    )
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    real_drain = batcher._drain_arrival_proof_buffer
    batcher._drain_arrival_proof_buffer = lambda environment: None
    batcher._submit_arrival_proof(_pending_for_receipt(71, "never-offered"))
    batcher._submit_arrival_proof(_pending_for_receipt(72, "known"))
    batcher._drain_arrival_proof_buffer = real_drain

    unknown, known = batcher._arrival_proof_buffer
    assert (unknown.rate, unknown.payload_bytes) == (None, 0)
    assert known.rate == 1.0
    assert known.payload_bytes == 1_000

    batcher._drain_arrival_proof_buffer(env)

    assert len(extended) == 1
    assert extended[0].payload.pending.prompt_idx == 72  # "known" wins
    assert len(batcher._arrival_proof_buffer) == 1
    assert batcher._arrival_proof_buffer[0] is unknown


def test_a_plan_extension_failure_releases_the_reservation(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )

    def _boom(candidates):
        raise RuntimeError("scheduler unavailable")

    batcher._extend_proof_plan = _boom

    try:
        batcher._submit_arrival_proof(_pending_stub(prompt_idx=4))
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 0


def test_a_passing_proof_records_proven_and_a_failing_one_releases(monkeypatch):
    """Accounting flows through the scheduler's own decision list, not
    through ``_execute_scheduled_proof``'s return value directly (see
    ``_reconcile_fill_state_decisions``) -- the scheduler settles some
    candidates without ever calling that method, so a real scheduler is
    what makes a PASSED/REJECTED decision exist to walk. A passing proof's
    PASSED decision records proven; a failing proof's REJECTED decision
    releases."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _execute_scheduler_payload, _make_batcher,
        _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    passing_scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        passing = _make_batcher(proof_scheduler=passing_scheduler)
        passing.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
        )
        assert passing.accept_submission(
            _request(prompt_idx=11, hotkey="miner-pass")
        ).accepted
        # accept_submission already reserved via _submit_arrival_proof.
        assert passing.fill_state.snapshot()["in_flight"][env] == 1

        def _passed() -> bool:
            passing._drain_arrival_proof_buffer(env)
            return passing.fill_state.snapshot()["proven"][env] == 1

        _wait_until(_passed, timeout=5.0)
        snap = passing.fill_state.snapshot()
        assert snap["proven"][env] == 1
        assert snap["in_flight"][env] == 0
    finally:
        assert passing_scheduler.close()

    failing_scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        failing = _make_batcher(
            proof_scheduler=failing_scheduler,
            verify_commitment_proofs_fn=_always_false_grail,
        )
        failing.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
        )
        assert failing.accept_submission(
            _request(prompt_idx=12, hotkey="miner-fail")
        ).accepted
        assert failing.fill_state.snapshot()["in_flight"][env] == 1

        def _released() -> bool:
            failing._drain_arrival_proof_buffer(env)
            return failing.fill_state.snapshot()["in_flight"][env] == 0

        _wait_until(_released, timeout=5.0)
        snap = failing.fill_state.snapshot()
        assert snap["proven"][env] == 0
        assert snap["in_flight"][env] == 0
    finally:
        assert failing_scheduler.close()


def test_v6_does_not_consult_the_seal_time_proof_wall(monkeypatch):
    """The wall bounded ONE seal burst. v6 has no burst -- it proves
    continuously -- so the window backstop is the only time bound."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "MAX_PROOF_WALL_SECONDS", 0.0)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher.mark_window_opened()

    # A zero wall would abort the auction path instantly; v6 must ignore it.
    assert batcher.poll_deadline() is False


def test_a_skipped_prompt_claimed_decision_releases_its_reservation(monkeypatch):
    """Reproduces the leak found in code review: 3 candidates on 2 prompts,
    budget 3. Only 2 groups can ever pass -- one per prompt -- so the loser
    on the shared prompt is settled by the scheduler as SKIPPED_PROMPT_
    CLAIMED, entirely inside its own coordinator, WITHOUT ever calling
    ``_execute_scheduled_proof``. Only the decisions walk can catch that;
    without it this reservation leaks forever and ``in_flight`` never goes
    back to 0 even though nothing is actually still proving.

    R33: the budget itself does NOT reopen for the loser -- all 3 were
    real reservations, and a failed one still spent real budget, so
    ``may_admit`` stays False once ``admitted`` reaches it. Only
    ``in_flight`` (proof-worker capacity, not admission) is what this
    reservation-leak fix actually frees."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _make_batcher, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        batcher.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 3, "opencodeinstruct": 3}, picks_target=16
        )

        # Two submissions competing for the SAME prompt (only one can win
        # it), one on a different prompt: 3 candidates, 2 distinct prompts.
        assert batcher.accept_submission(
            _request(prompt_idx=1, hotkey="hk-a")
        ).accepted
        assert batcher.accept_submission(
            _request(prompt_idx=1, hotkey="hk-b")
        ).accepted
        assert batcher.accept_submission(
            _request(prompt_idx=2, hotkey="hk-c")
        ).accepted

        def _settled() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            snap = batcher.fill_state.snapshot()
            return snap["proven"][env] == 2 and snap["in_flight"][env] == 0

        _wait_until(_settled, timeout=5.0)

        snap = batcher.fill_state.snapshot()
        assert snap["proven"][env] == 2
        assert snap["in_flight"][env] == 0
        assert batcher.fill_state.may_admit(env) is False
    finally:
        assert scheduler.close()


def test_a_raising_proof_callable_releases_its_reservation(monkeypatch):
    """An uncaught exception from ``_verify_expensive`` faults the whole
    scheduler (infrastructure failure, not a miner-attributable reject --
    see the comment in ``_execute_scheduled_proof``). The faulted candidate
    still gets a terminal, non-PASSED decision synthesized once the fault
    applies, so the decisions walk must release its reservation exactly
    like any other non-PASSED outcome -- accounting must not depend on
    ``_execute_scheduled_proof`` returning normally."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    def _boom(_invocation):
        raise RuntimeError("proof device failed")

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_boom,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        batcher.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
        )
        assert batcher.accept_submission(
            _request(prompt_idx=5, hotkey="hk-fault")
        ).accepted
        assert batcher.fill_state.snapshot()["in_flight"][env] == 1

        def _released() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            return batcher.fill_state.snapshot()["in_flight"][env] == 0

        _wait_until(_released, timeout=5.0)

        snap = batcher.fill_state.snapshot()
        assert snap["proven"][env] == 0
        assert snap["in_flight"][env] == 0
    finally:
        assert scheduler.close()


def test_a_decision_is_accounted_exactly_once_across_repeated_walks(monkeypatch):
    """Two separate triggers (a graded body landing, a proof completing)
    both call the walk; a job_id already accounted must never be
    double-released or double-counted just because the walk ran twice
    before any NEW decision arrived."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import (
        ProofDecision, ProofDecisionStatus,
    )

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher.fill_state.reserve(env)
    batcher.fill_state.reserve(env)

    class _FakeHandle:
        def __init__(self, decisions):
            self._decisions = tuple(decisions)

        def decisions(self):
            return self._decisions

    batcher._open_proof_plan_handle = _FakeHandle([
        ProofDecision(
            job_id="j1", rank=1, prompt_key=("prompt", 1),
            status=ProofDecisionStatus.PASSED, device_id="gpu-0",
            started_at=0.0, finished_at=1.0,
        ),
        ProofDecision(
            job_id="j2", rank=2, prompt_key=("prompt", 2),
            status=ProofDecisionStatus.REJECTED, device_id="gpu-0",
            started_at=0.0, finished_at=1.0,
        ),
    ])

    batcher._reconcile_fill_state_decisions(env)
    snap = batcher.fill_state.snapshot()
    assert snap["proven"][env] == 1
    assert snap["in_flight"][env] == 0

    # Same two decisions, walked again -- must be a no-op, not a second
    # record_proven/release for job ids already accounted.
    batcher._reconcile_fill_state_decisions(env)
    snap = batcher.fill_state.snapshot()
    assert snap["proven"][env] == 1
    assert snap["in_flight"][env] == 0


def test_concurrent_drains_do_not_race_on_rank_allocation(monkeypatch):
    """``_arrival_proof_rank`` is allocated under ``_fill_state_lock``, but
    (pre-fix) ``_extend_proof_plan`` ran AFTER that block released it. Two
    threads draining the same environment concurrently -- which
    ``_execute_scheduled_proof``'s own comment says is routine in
    production with more than one device -- could allocate ranks 5 then 6
    but call ``extend`` in the opposite order, which the scheduler refuses
    (``rank <= its current highest`` -> ``ValueError``).

    This drives ``_drain_arrival_proof_buffer`` directly from N threads,
    released together by a ``Barrier``, against a buffer pre-populated
    with N entries and a REAL ``GlobalProofScheduler`` (so ``extend``'s
    refusal is genuine, not simulated). Going through the full admission
    -> scheduler -> device-worker-completion chain to manufacture this
    same interleaving turned out to not reproduce reliably even over many
    attempts (see the report) -- decision application happens strictly
    AFTER a proof callable returns, so two callables released from the
    same barrier still can't race on `` _drain_arrival_proof_buffer`` this
    directly. Driving the method under test straight from N threads is
    the deterministic version of the same race. Repeated in a loop:
    GIL-interleaving races are not guaranteed on any single pass.
    """
    import threading
    import time

    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler, ProofExecution,
    )
    from tests.unit.test_grpo_window_batcher import _make_batcher
    from tests.unit.test_proof_scheduler import _wait_until

    # Widens the exact window under test (between allocating a rank and
    # calling extend/submit with it) so the interleaving is not left to
    # chance GIL preemption, which -- empirically, see the report -- does
    # not land in that window reliably even over many bare attempts.
    # Standard technique for a concurrency bug: this is test-only
    # instrumentation, not a change to the code under test's logic.
    original_ranked_proof_for = batcher_module.GrpoWindowBatcher._ranked_proof_for

    def _slow_ranked_proof_for(self, *args, **kwargs):
        time.sleep(0.005)
        return original_ranked_proof_for(self, *args, **kwargs)

    monkeypatch.setattr(
        batcher_module.GrpoWindowBatcher,
        "_ranked_proof_for",
        _slow_ranked_proof_for,
    )

    env = "openmathinstruct"
    n_threads = 8

    def _trivial_prove(_invocation):
        return ProofExecution(passed=True, value=object())

    for attempt in range(20):
        scheduler = GlobalProofScheduler(
            devices=("gpu-0", "gpu-1"),
            environments=("openmathinstruct", "opencodeinstruct"),
            proof_callable=_trivial_prove,
            checkpoint_revision="",
        )
        try:
            batcher = _make_batcher(proof_scheduler=scheduler)
            batcher.fill_state = batcher_module.FillState(
                budgets={"openmathinstruct": n_threads, "opencodeinstruct": n_threads}, picks_target=16
            )

            # Pre-populate the buffer directly -- this test targets
            # ``_drain_arrival_proof_buffer``'s own atomicity, not the
            # admission path that normally fills it.
            for i in range(n_threads):
                batcher._arrival_proof_sequence += 1
                batcher._arrival_proof_buffer.append(
                    batcher_module._BufferedArrivalProof(
                        pending=_pending_stub(prompt_idx=attempt * 100 + i),
                        rate=None,
                        payload_bytes=0,
                        receipt_id="",
                        sequence=batcher._arrival_proof_sequence,
                    )
                )

            barrier = threading.Barrier(n_threads)
            errors: list[Exception] = []

            def _drain_worker() -> None:
                barrier.wait(timeout=5.0)
                try:
                    batcher._drain_arrival_proof_buffer(env)
                except Exception as exc:  # noqa: BLE001 -- captured, not swallowed
                    errors.append(exc)

            threads = [
                threading.Thread(target=_drain_worker) for _ in range(n_threads)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10.0)

            assert errors == [], (attempt, errors)
            assert batcher._arrival_proof_buffer == []
            # Total committed (proven + in_flight) must equal exactly the
            # number of entries drained -- some may already have resolved
            # PASSED and been reconciled (moving in_flight -> proven) by
            # the time the last thread finishes, which is expected, not a
            # bug: reconcile runs at the top of every drain call too.
            snap = batcher.fill_state.snapshot()
            assert snap["proven"][env] + snap["in_flight"][env] == n_threads

            handle = batcher._open_proof_plan_handle
            _wait_until(
                lambda: len(handle.decisions()) >= n_threads, timeout=5.0
            )
            ranks = [decision.rank for decision in handle.decisions()]
            assert ranks == sorted(ranks), (attempt, ranks)
            assert len(set(ranks)) == len(ranks), (attempt, ranks)
        finally:
            scheduler.close()
