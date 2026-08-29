"""The window ends on its fill, and batches leave while it is still open."""
from reliquary.constants import B_BATCH
from tests.unit.test_grpo_window_batcher import _make_batcher


def test_the_window_seals_when_every_environment_is_full(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher.mark_window_opened()

    batcher.fill_state.record_proven("openmathinstruct")
    assert batcher.poll_deadline() is False

    batcher.fill_state.record_proven("opencodeinstruct")

    assert batcher.poll_deadline() is True


def test_a_batch_is_emitted_every_b_batch_proven_groups(monkeypatch):
    """16 x 32 = 512 is arithmetic, not a schedule the miner can see."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    emitted = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 64, "opencodeinstruct": 64}
    )
    batcher._emit_training_batch = lambda: emitted.append(1)

    for _ in range(B_BATCH):
        batcher.fill_state.record_proven("openmathinstruct")
        batcher.fill_state.record_proven("opencodeinstruct")
        batcher._maybe_emit_batch()

    assert len(emitted) == 1


def test_state_advertises_which_environments_are_still_admitting(monkeypatch):
    """Math has historically under-filled, so Code will close first and
    Math will set the window duration. Code miners need to see that."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher.fill_state.record_proven("opencodeinstruct")

    fill = batcher.upload_precommit_conservation()["fill_state"]

    assert fill["proven"]["opencodeinstruct"] == 1
    assert fill["closed"] is False


def test_v6_does_not_consult_the_seal_time_proof_wall(monkeypatch):
    """The wall bounded ONE seal burst. v6 has no burst -- it proves
    continuously -- so the window backstop is the only time bound.

    Task 6's brief specified this as its own Step 3b and skipped it
    (NEEDS_CONTEXT); it belongs here since ``poll_deadline`` is this
    task's file (see task-7-addendum.md, "Unchanged from the brief").
    """
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


def test_emission_is_hooked_from_the_reconcile_walk_not_record_proven(
    monkeypatch,
):
    """Pins the hook point named in task-7-addendum.md: ``_maybe_emit_batch``
    must be called from ``_reconcile_fill_state_decisions``, after its walk
    -- not from ``FillState.record_proven`` directly, which is now the
    scheduler's only path to accounting a PASSED group. A direct
    ``record_proven`` call (bypassing the walk entirely) must NOT emit on
    its own; only a REAL scheduler's terminal decision, walked, does.
    """
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "B_BATCH", 1)
    env = "openmathinstruct"

    emitted = []
    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        batcher.fill_state = batcher_module.FillState(
            targets={"openmathinstruct": 1, "opencodeinstruct": 1}
        )
        batcher._emit_training_batch = lambda: emitted.append(1)

        # Bypasses the reconcile walk entirely -- must not trigger emission.
        batcher.fill_state.record_proven("opencodeinstruct")
        assert emitted == []

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _proven() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            return batcher.fill_state.snapshot()["proven"][env] == 1

        _wait_until(_proven, timeout=5.0)

        assert len(emitted) == 1
    finally:
        assert scheduler.close()
