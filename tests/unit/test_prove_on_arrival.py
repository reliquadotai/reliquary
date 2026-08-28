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
