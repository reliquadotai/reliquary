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
