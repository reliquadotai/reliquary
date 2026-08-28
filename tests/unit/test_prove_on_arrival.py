"""Under v6 a submission is proven when it arrives, not at seal."""
import hashlib
import types

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
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
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
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
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
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
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
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    batcher.fill_state.reserve("openmathinstruct")  # already at target

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=3))

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1


def test_a_buffered_group_is_not_reserved_until_drained(monkeypatch):
    """A body that arrives while the environment is full sits in the
    buffer; buffering must reserve nothing. Only ``_drain_arrival_proof_
    buffer`` -- run once capacity frees -- reserves and extends it."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    env = "openmathinstruct"
    batcher.fill_state.reserve(env)  # already at target

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=41))

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"][env] == 1
    assert len(batcher._arrival_proof_buffer) == 1

    batcher.fill_state.release(env)  # capacity frees elsewhere
    batcher._drain_arrival_proof_buffer(env)

    assert len(extended) == 1
    assert batcher.fill_state.snapshot()["in_flight"][env] == 1
    assert len(batcher._arrival_proof_buffer) == 0


def test_ranks_handed_to_extend_strictly_increase_across_drains(monkeypatch):
    """``extend`` refuses a rank at or behind the plan's current highest
    (``proof_scheduler.py``'s ``next_apply_index`` invariant). The rank
    counter must keep climbing across every drain, not just within one."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    env = "openmathinstruct"

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=61))  # drains now
    batcher._submit_arrival_proof(_pending_stub(prompt_idx=62))  # buffers: full
    batcher._submit_arrival_proof(_pending_stub(prompt_idx=63))  # buffers: full

    assert len(extended) == 1

    batcher.fill_state.release(env)
    batcher._drain_arrival_proof_buffer(env)  # a second, separate drain

    batcher.fill_state.release(env)
    batcher._drain_arrival_proof_buffer(env)  # a third, separate drain

    assert len(extended) == 3
    ranks = [candidate.rank for candidate in extended]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_an_unknown_rate_falls_back_to_lowest_priority_not_a_crash(monkeypatch):
    """``rate_of`` misses when a receipt was never offered (or was offered
    in a different window). The group must still drain -- after every
    known-rate group, however small its rate -- rather than raise."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.mark_window_opened()
    batcher.admission_queue = batcher_module.ThroughputAdmissionQueue(
        window_opened_at=batcher.window_opened_at
    )
    env = "openmathinstruct"
    batcher.admission_queue.offer(
        receipt_id="known", environment=env, payload_bytes=1,
        precommit_arrived_at=batcher.window_opened_at + 100.0,  # tiny rate
    )
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 2, "opencodeinstruct": 2}
    )
    batcher.fill_state.reserve(env)
    batcher.fill_state.reserve(env)  # full: both bodies below buffer
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    unknown = _pending_stub(prompt_idx=71)
    unknown.request = types.SimpleNamespace(_precommit_receipt_id="never-offered")
    known = _pending_stub(prompt_idx=72)
    known.request = types.SimpleNamespace(_precommit_receipt_id="known")

    batcher._submit_arrival_proof(unknown)  # arrives first, unknown rate
    batcher._submit_arrival_proof(known)    # arrives second, a known (tiny) rate

    assert extended == []

    batcher.fill_state.release(env)
    batcher._drain_arrival_proof_buffer(env)

    assert len(extended) == 1
    assert extended[0].payload.pending.prompt_idx == 72  # "known" wins despite tiny rate


def test_a_plan_extension_failure_releases_the_reservation(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
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
    """The completion path: ``_execute_scheduled_proof`` runs once per
    candidate, synchronously, on whatever thread executes the proof --
    before the scheduler's coordinator later applies decisions in rank
    order. That is the one place the pass/fail fact is known, for both the
    seal-ranked path (fill_state is None there, so this is a no-op) and the
    v6 arrival path."""
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    passing = _make_batcher(verify_commitment_proofs_fn=_always_true_grail)
    passing.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    passing._extend_proof_plan = lambda candidates: None
    passing.accept_submission(_request(prompt_idx=11, hotkey="miner-pass"))
    pending = passing.pending_submissions()[0]
    # accept_submission already reserved via _submit_arrival_proof.
    assert passing.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1

    verified = passing._execute_scheduled_proof(
        pending, model=None, count_operator_debt=True,
    )

    assert verified is not None
    snap = passing.fill_state.snapshot()
    assert snap["proven"]["openmathinstruct"] == 1
    assert snap["in_flight"]["openmathinstruct"] == 0

    failing = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    failing.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    failing._extend_proof_plan = lambda candidates: None
    failing.accept_submission(_request(prompt_idx=12, hotkey="miner-fail"))
    pending = failing.pending_submissions()[0]
    assert failing.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1

    verified = failing._execute_scheduled_proof(
        pending, model=None, count_operator_debt=True,
    )

    assert verified is None
    snap = failing.fill_state.snapshot()
    assert snap["proven"]["openmathinstruct"] == 0
    assert snap["in_flight"]["openmathinstruct"] == 0


def test_v6_does_not_consult_the_seal_time_proof_wall(monkeypatch):
    """The wall bounded ONE seal burst. v6 has no burst -- it proves
    continuously -- so the window backstop is the only time bound."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "MAX_PROOF_WALL_SECONDS", 0.0)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    batcher.mark_window_opened()

    # A zero wall would abort the auction path instantly; v6 must ignore it.
    assert batcher.poll_deadline() is False
