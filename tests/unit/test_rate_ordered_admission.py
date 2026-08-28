"""A precommit that lands later is served first if it was produced faster.

The queue itself (``ThroughputAdmissionQueue``) never hands out a provable
candidate -- a ``PendingSubmission`` does not exist until the body has
arrived and graded, later and on a different path than the precommit the
queue holds. So the queue's rate order becomes the batcher's arrival-proof
BUFFER drain order instead: both bodies below grade while the environment is
momentarily full (so neither drains immediately on arrival), and when
capacity frees, the higher-rate precommit is what gets extended first,
regardless of which body arrived -- and buffered -- first.
"""
import types

from tests.unit.test_grpo_window_batcher import _make_batcher
from tests.unit.test_prove_on_arrival import _pending_stub


def _pending_for_receipt(prompt_idx, receipt_id):
    pending = _pending_stub(prompt_idx=prompt_idx)
    pending.request = types.SimpleNamespace(_precommit_receipt_id=receipt_id)
    return pending


def test_a_faster_later_precommit_is_extended_before_a_slower_earlier_one(
    monkeypatch,
):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.mark_window_opened()
    batcher.admission_queue = batcher_module.ThroughputAdmissionQueue(
        window_opened_at=batcher.window_opened_at
    )
    env = "openmathinstruct"

    # slow: 1000 bytes over 50 s. fast: 9000 bytes over 60 s, arriving LAST.
    batcher.admission_queue.offer(
        receipt_id="slow", environment=env, payload_bytes=1_000,
        precommit_arrived_at=batcher.window_opened_at + 50.0,
    )
    batcher.admission_queue.offer(
        receipt_id="fast", environment=env, payload_bytes=9_000,
        precommit_arrived_at=batcher.window_opened_at + 60.0,
    )

    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 2, "opencodeinstruct": 2}
    )
    batcher.fill_state.reserve(env)
    batcher.fill_state.reserve(env)  # full: both bodies below buffer instead
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    # The slow group's body grades FIRST; the fast group's body grades
    # second. Neither can drain -- the environment is full -- so both sit in
    # the buffer together.
    batcher._submit_arrival_proof(_pending_for_receipt(1, "slow"))
    batcher._submit_arrival_proof(_pending_for_receipt(2, "fast"))

    assert extended == []
    assert batcher.fill_state.snapshot()["in_flight"][env] == 2

    # One reservation frees elsewhere (a proof completes) -- only room for
    # one more. It must go to "fast", not to "slow" merely because "slow"
    # graded first.
    batcher.fill_state.release(env)
    batcher._drain_arrival_proof_buffer(env)

    assert len(extended) == 1
    assert extended[0].payload.pending.prompt_idx == 2  # the "fast" group
