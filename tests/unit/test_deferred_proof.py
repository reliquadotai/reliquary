"""Proof deferral: a submission is graded and scored at admission, but not proven
until it is ranked high enough to win. See
docs/superpowers/specs/2026-07-14-difficulty-auction-design.md §7
"""
from reliquary.validator.batcher import PendingSubmission


def _pending(hotkey="a", prompt_idx=1, k=2, m=8, drand_round=1):
    return PendingSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        request=None,
        rewards=[1.0] * k + [0.0] * (m - k),
        drand_round=drand_round,
        merkle_root=hotkey.encode().ljust(32, b"\x00"),
        selection_digest=hotkey.encode().ljust(32, b"\x00"),
    )


def test_pending_submission_is_scored_at_admission():
    """Scoring is cheap (it only needs the rewards), so it happens before the GPU
    ever sees the submission — that is what lets us rank before proving."""
    assert _pending(k=2).value > _pending(k=6).value


def test_pending_submission_ranks_in_the_auction():
    """It must satisfy the duck-type select_batch_auction consumes, so the same
    ranking code works on unproven candidates."""
    from reliquary.validator.batch_auction import select_batch_auction
    from reliquary.validator.cooldown import CooldownMap

    hard = _pending(hotkey="hard", prompt_idx=1, k=2)
    easy = _pending(hotkey="easy", prompt_idx=2, k=6)

    batch, _ = select_batch_auction(
        [easy, hard], b=1,
        cooldown_map=CooldownMap(cooldown_windows=0), current_window=1, pool=1.0,
    )

    assert [s.hotkey for s in batch] == ["hard"]


def test_accept_does_not_touch_the_gpu():
    """Admission must be proof-free. If the GPU is called during accept, the
    whole design collapses — we would be proving ~69 submissions per window."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    calls = []

    def _exploding_proof(*a, **kw):
        calls.append(1)
        raise AssertionError("GRAIL must not run during admission")

    b = _make_batcher(verify_commitment_proofs_fn=_exploding_proof)
    resp = b.accept_submission(_request(prompt_idx=7, hotkey="miner"))

    assert resp.accepted is True
    assert calls == []
    assert len(b.pending_submissions()) == 1
    assert b.pending_submissions()[0].value > 0.0   # graded + scored


def test_verify_expensive_runs_the_proof_and_returns_a_valid_submission():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=7, hotkey="miner"))
    pending = b.pending_submissions()[0]

    proven = b._verify_expensive(pending)

    assert proven is not None
    assert proven.hotkey == "miner"
    assert proven.value == pending.value
