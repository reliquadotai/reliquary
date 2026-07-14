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
    """No more BATCH_FILLED. A miner who took 250s on a hard prompt still gets in
    — that is the entire point of the deadline."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(20):
        assert b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}")).accepted


def test_second_submission_for_a_claimed_prompt_is_rejected():
    """One prompt = one slot, decided at ADMISSION. This also kills the
    variance-farming sybil: the forced-seed seed contains the hotkey, so N hotkeys
    on one prompt would otherwise buy N independent draws of k and the operator
    would submit whichever landed nearest k=2."""
    from reliquary.validator.batcher import RejectReason
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    assert b.accept_submission(_request(prompt_idx=7, hotkey="first")).accepted

    resp = b.accept_submission(_request(prompt_idx=7, hotkey="second"))

    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_CLAIMED
