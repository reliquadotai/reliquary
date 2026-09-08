"""A precommit that lands later is served first if it was produced faster.

The queue itself (``ThroughputAdmissionQueue``) never hands out a provable
candidate -- a ``PendingSubmission`` does not exist until the body has
arrived and graded, later and on a different path than the precommit the
queue holds. So the queue's rate order becomes the batcher's arrival-proof
BUFFER drain order instead: both bodies below grade and land in the buffer
before either's own drain runs (what a genuine concurrent race would
produce -- see the R33 note in the test below, since under the budget
model a release no longer reopens room for a SECOND drain to matter), and
when the one drain that has room runs, the higher-rate precommit is what
gets extended, regardless of which body arrived -- and buffered -- first.
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
    """R33: ``admitted`` is monotone, so a release no longer reopens
    ``may_admit`` -- the old two-reserve-then-release setup that used to
    force both bodies into the buffer together no longer produces a
    second, room-having drain. What DOES still produce two competing
    buffer entries is exactly what a genuine concurrent race would: both
    bodies landing before either's own auto-drain call actually runs.
    That race is pinned here by holding the real drain off (a no-op
    stand-in) while both bodies are submitted -- ``_submit_arrival_proof``
    still does the real ``admission_queue.rate_of()`` lookup and buffer
    append for each -- then restoring it and running the one drain that
    has room."""
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
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    # The slow group's body grades FIRST; the fast group's body grades
    # second. Neither drains here -- the real drain is held off -- so both
    # sit in the buffer together, same as a genuine race would leave them.
    real_drain = batcher._drain_arrival_proof_buffer
    batcher._drain_arrival_proof_buffer = lambda environment: None
    batcher._submit_arrival_proof(_pending_for_receipt(1, "slow"))
    batcher._submit_arrival_proof(_pending_for_receipt(2, "fast"))
    batcher._drain_arrival_proof_buffer = real_drain

    assert extended == []
    assert len(batcher._arrival_proof_buffer) == 2

    # Budget for only one. It must go to "fast", not to "slow" merely
    # because "slow" graded first.
    batcher._drain_arrival_proof_buffer(env)

    assert len(extended) == 1
    assert extended[0].payload.pending.prompt_idx == 2  # the "fast" group
    assert len(batcher._arrival_proof_buffer) == 1
