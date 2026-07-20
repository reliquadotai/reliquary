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


def test_collection_deadline_starts_at_activation_not_construction():
    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from tests.unit.test_grpo_window_batcher import _make_batcher

    now = [1000.0]
    wall = [10_000.0]
    b = _make_batcher(
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: wall[0],
    )
    now[0] += 45.0
    wall[0] += 45.0

    b.mark_window_opened()

    assert b.window_opened_at == 1045.0
    assert b.window_opened_wall_ts == 10_045.0
    now[0] += WINDOW_COLLECTION_SECONDS - 0.1
    assert b.poll_deadline() is False
    now[0] += 0.2
    assert b.poll_deadline() is True


def test_late_submission_is_accepted_while_the_window_is_open():
    """No more count-based BATCH_FILLED. A miner who took 250s on a hard prompt
    still gets in — that is the entire point of the deadline."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(20):
        assert b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}")).accepted


def test_multiple_submissions_per_prompt_are_admitted_and_resolved_at_seal():
    """v2 replaces the PROMPT_CLAIMED admission reject: several hotkeys may submit
    the same prompt (bounded by MAX_SUBMISSIONS_PER_PROMPT). Same-prompt
    resolution happens at SEAL — an exact (score, arrival) tie is proven in
    full and splits the prompt's share; one representative trains — so no
    wire-level reject is needed and admission never runs the GPU."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    assert b.accept_submission(_request(prompt_idx=7, hotkey="first")).accepted
    assert b.accept_submission(_request(prompt_idx=7, hotkey="second")).accepted
    assert len(b.pending_submissions()) == 2

    batch, rewards = b.seal_batch()

    winners = [s for s in b.valid_submissions() if s.prompt_idx == 7]
    assert len(winners) == 2                       # exact tie: both proven
    assert len([s for s in batch if s.prompt_idx == 7]) == 1
    assert rewards["first"] == rewards["second"]   # split the prompt share


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


def test_precommit_bytes_transfer_and_conserve_at_terminal_decision():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    request = _request(prompt_idx=9, hotkey="miner")
    accepted, reason, _deadline = b.try_register_upload_precommit(
        "receipt",
        "miner",
        t_arrival_wall=b.window_opened_wall_ts,
        payload_bytes=1234,
    )

    assert accepted is True
    assert reason is None
    assert b.reserved_payload_bytes == 1234
    assert b.upload_precommit_conservation()["pending"] == 1
    assert b.mark_upload_precommit_revealed("receipt") is True
    assert b.start_revealed_admission("receipt", request) == (True, None)
    assert b.upload_precommit_payload_bytes == 0
    assert b.inflight_payload_bytes == 1234

    b.finish_proof_admission(request)
    assert b.resolve_upload_precommit("receipt") is True
    assert b.reserved_payload_bytes == 0
    assert b.upload_precommit_conservation() == {
        "accepted_receipts": 1,
        "revealed": 1,
        "revealed_terminal": 1,
        "expired": 0,
        "terminal_decisions": 1,
        "pending": 0,
        "conserved": True,
    }


def test_unrevealed_precommit_expires_and_releases_exact_bytes():
    from reliquary.constants import SUBMISSION_UPLOAD_GRACE_SECONDS
    from tests.unit.test_grpo_window_batcher import _make_batcher

    now = [1000.0]
    wall = [10_000.0]
    b = _make_batcher(
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: wall[0],
    )
    accepted, _reason, _deadline = b.try_register_upload_precommit(
        "receipt",
        "miner",
        t_arrival_wall=wall[0],
        payload_bytes=2048,
    )
    assert accepted is True

    now[0] += SUBMISSION_UPLOAD_GRACE_SECONDS + 0.1
    assert b.pending_upload_precommits == 0
    assert b.reserved_payload_bytes == 0
    conservation = b.upload_precommit_conservation()
    assert conservation["expired"] == 1
    assert conservation["conserved"] is True


def test_precommit_capacity_is_a_per_window_accepted_total(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import _make_batcher

    monkeypatch.setattr(
        batcher_module, "MAX_PENDING_UPLOAD_PRECOMMITS_PER_ENV", 2
    )
    batcher = _make_batcher()
    for index in range(2):
        accepted, reason, _deadline = batcher.try_register_upload_precommit(
            f"receipt-{index}",
            f"miner-{index}",
            t_arrival_wall=batcher.window_opened_wall_ts,
            payload_bytes=100,
        )
        assert accepted is True
        assert reason is None
        assert batcher.resolve_upload_precommit(f"receipt-{index}") is True

    accepted, reason, _deadline = batcher.try_register_upload_precommit(
        "receipt-over-cap",
        "miner-over-cap",
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=100,
    )

    assert accepted is False
    assert reason == "precommit_capacity_full"
    assert batcher.upload_precommit_conservation()["accepted_receipts"] == 2
