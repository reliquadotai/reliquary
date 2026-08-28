"""Rate-ordered admission: producing long must not cost you your place.

The fill-closed window closes when the batch is full, and the slots still open
near the close would otherwise go to whoever finishes first — systematically
whoever produced the SHORTEST rollouts. Per-token payment does not fix it: a
long group has to get in before it can be paid.

So the queue is ordered by production rate:

    rate = payload_bytes / (precommit arrival - window open)

It is measured at the PRECOMMIT, not the upload, so transport latency is
inside the measure and a fat uplink cannot buy a place. And the denominator
runs from window open for every group — no identity in the formula at all —
so splitting across hotkeys changes nothing, and parallel producers gain
volume without gaining rank.
"""

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
    """A graded body whose receipt never went through ``offer`` (never
    queued, or queued in a different window) must degrade to a defined
    priority rather than raise -- the batcher's buffer sort depends on
    this returning cleanly."""
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "slow", at=50.0, payload_bytes=1000)

    assert queue.rate_of("never-offered") is None


def test_the_rate_runs_from_window_open_and_carries_no_identity():
    """Two identical groups from any two senders get the same rate: the
    formula has no hotkey and no operator in it. What decides is only how far
    into the window the precommit landed, and how many bytes it binds."""
    queue = ThroughputAdmissionQueue(window_opened_at=100.0)

    first = _offer(queue, "r1", at=125.0, payload_bytes=5000)
    second = _offer(queue, "r2", at=125.0, payload_bytes=5000)

    assert first.elapsed == 25.0
    assert first.throughput == second.throughput == 200.0


def test_parallel_producers_gain_volume_not_rank():
    """Eight groups from eight GPUs all landing at 25 s each rate as a single
    25 s group does. They win by having eight tickets, not by out-ranking.

    Measured from the sender's previous arrival instead, the eighth would show
    0.1 s elapsed and a 250x rate — a double count of the same hardware.
    """
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)

    solo = _offer(queue, "solo", at=25.0, payload_bytes=8000)
    burst = [
        _offer(queue, f"burst-{i}", at=25.0 + 0.1 * i, payload_bytes=8000)
        for i in range(8)
    ]

    assert max(b.throughput for b in burst) <= solo.throughput
    assert min(b.throughput for b in burst) > solo.throughput * 0.97


def test_rate_of_looks_up_by_receipt_regardless_of_environment():
    """``rate_of`` has no environment parameter: a receipt_id is unique on
    its own, and the batcher that calls it already knows which environment
    it is (its own -- one batcher per environment)."""
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "math", at=10.0, payload_bytes=9000)
    _offer(queue, "code", at=10.0, payload_bytes=100, env="opencodeinstruct")

    assert queue.rate_of("code") == 10.0
    assert queue.rate_of("math") == 900.0
