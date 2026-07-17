"""The window is time-boxed. Its MINIMUM duration is the deadline itself — an
early seal is exactly the speed race we are removing: whoever triggered it would
cut off the slow-but-hard submissions still generating.
"""


def test_eighth_distinct_prompt_does_not_seal_the_window():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(12):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}"))

    assert b.is_sealed() is False       # 12 > B_BATCH, and still open


def test_window_seals_when_the_deadline_expires():
    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    now = [1000.0]
    b = _make_batcher(time_fn=lambda: now[0])
    b.accept_submission(_request(prompt_idx=1, hotkey="m"))

    assert b.is_sealed() is False
    now[0] += WINDOW_COLLECTION_SECONDS + 1
    b.poll_deadline()

    assert b.is_sealed() is True


def test_late_submission_is_accepted_while_the_window_is_open():
    """No more count-based BATCH_FILLED. A miner who took 250s on a hard prompt
    still gets in — that is the entire point of the deadline."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(20):
        assert b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}")).accepted


def test_multiple_submissions_per_prompt_are_admitted_and_resolved_at_seal():
    """v2 replaces the PROMPT_CLAIMED admission reject: several hotkeys may submit
    the same prompt (bounded by MAX_SUBMISSIONS_PER_PROMPT). The same-prompt
    winner is resolved at SEAL — the first submission that PASSES the proof takes
    the slot, the rest are dropped — so no wire-level reject is needed and
    admission never runs the GPU."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    assert b.accept_submission(_request(prompt_idx=7, hotkey="first")).accepted
    assert b.accept_submission(_request(prompt_idx=7, hotkey="second")).accepted
    assert len(b.pending_submissions()) == 2

    b.seal_batch()

    winners = [s for s in b.valid_submissions() if s.prompt_idx == 7]
    assert len(winners) == 1


def test_code_uses_deferred_proof_and_collection_deadline():
    from reliquary.constants import B_BATCH, WINDOW_COLLECTION_SECONDS
    from tests.unit.test_grpo_window_batcher import (
        PrivateRewardFakeEnv,
        _make_batcher,
        _request,
    )

    now = [1000.0]
    proof_calls = []

    def _proof(commit, model, randomness):
        from tests.unit.test_grpo_window_batcher import _always_true_grail

        proof_calls.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(
        env=PrivateRewardFakeEnv(),
        time_fn=lambda: now[0],
        verify_commitment_proofs_fn=_proof,
    )
    assert b.difficulty_auction_enabled is True

    for i in range(B_BATCH):
        req = _request(prompt_idx=i, hotkey=f"code-{i}")
        for rollout in req.rollouts:
            rollout.env_name = "opencodeinstruct"
        assert b.accept_submission(req).accepted

    assert proof_calls == []
    assert len(b.pending_submissions()) == B_BATCH
    assert b.valid_submissions() == []
    assert b.is_sealed() is False

    now[0] += WINDOW_COLLECTION_SECONDS + 1
    assert b.poll_deadline() is True
    b.seal_batch()

    assert len(proof_calls) == B_BATCH * 8
    assert len(b.valid_submissions()) == B_BATCH


def test_auction_kill_switch_restores_legacy_path(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    monkeypatch.setattr(batcher_module, "DIFFICULTY_AUCTION_ENFORCE", False)
    b = _make_batcher()

    assert b.difficulty_auction_enabled is False
    assert b.accept_submission(_request(prompt_idx=1, hotkey="miner")).accepted
    assert b.pending_submissions() == []
    assert len(b.valid_submissions()) == 1
