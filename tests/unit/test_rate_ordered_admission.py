"""A precommit that lands later is served first if it was produced faster."""
from tests.unit.test_grpo_window_batcher import _make_batcher


def test_a_faster_later_precommit_is_admitted_before_a_slower_earlier_one(
    monkeypatch,
):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

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

    assert batcher._next_admission(env).receipt_id == "fast"
    assert batcher._next_admission(env).receipt_id == "slow"
    assert batcher._next_admission(env) is None
