"""Characterization tests for the disabled fill priority queue."""

from __future__ import annotations

from reliquary.validator.admission_priority import ThroughputAdmissionQueue

MATH = "openmathinstruct"


def _offer(queue, receipt, *, at, payload_bytes, env=MATH):
    return queue.offer(
        receipt_id=receipt,
        environment=env,
        payload_bytes=payload_bytes,
        precommit_arrived_at=at,
    )


def test_rate_of_reports_the_throughput_a_receipt_registered():
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)

    _offer(queue, "slow", at=50.0, payload_bytes=1000)   # 20 B/s
    _offer(queue, "fast", at=60.0, payload_bytes=9000)   # 150 B/s, arrived LAST

    assert queue.rate_of("fast") == 150.0
    assert queue.rate_of("slow") == 20.0


def test_rate_of_an_unknown_receipt_is_none_not_a_crash():
    """An unknown receipt has an explicit, non-raising result."""
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "slow", at=50.0, payload_bytes=1000)

    assert queue.rate_of("never-offered") is None


def test_the_rate_runs_from_window_open_and_carries_no_identity():
    """Equivalent receipts produce the same deterministic priority."""
    queue = ThroughputAdmissionQueue(window_opened_at=100.0)

    first = _offer(queue, "r1", at=125.0, payload_bytes=5000)
    second = _offer(queue, "r2", at=125.0, payload_bytes=5000)

    assert first.elapsed == 25.0
    assert first.throughput == second.throughput == 200.0


def test_multiple_receipts_share_one_elapsed_time_origin():
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)

    solo = _offer(queue, "solo", at=25.0, payload_bytes=8000)
    burst = [
        _offer(queue, f"burst-{i}", at=25.0 + 0.1 * i, payload_bytes=8000)
        for i in range(8)
    ]

    assert max(b.throughput for b in burst) <= solo.throughput
    assert min(b.throughput for b in burst) > solo.throughput * 0.97


def test_rate_of_looks_up_by_receipt_regardless_of_environment():
    """Receipt identity is sufficient for a deterministic lookup."""
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "math", at=10.0, payload_bytes=9000)
    _offer(queue, "code", at=10.0, payload_bytes=100, env="opencodeinstruct")

    assert queue.rate_of("code") == 10.0
    assert queue.rate_of("math") == 900.0
