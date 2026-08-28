"""Throughput-ordered admission: producing long must not cost you your place.

The fill-closed window closes when the batch is full, and admission is by
arrival. That hands the marginal slots — the ones still open near the close —
to whoever finishes first, which is systematically whoever produced the
SHORTEST rollouts. Per-token payment does not fix it: you have to get in first.

So the queue is ordered by throughput rather than by arrival. A precommit that
lands while an earlier one is still being validated goes ahead of it if it was
produced at a better rate. At fixed hardware the rate is the same whether the
group is 500 or 5000 tokens per rollout, so length stops deciding who gets in.

Both terms are safe from the miner. ``payload_bytes`` is bound by the signed
precommit and enforced against the upload; elapsed is validator-observed.
"""

from __future__ import annotations

from reliquary.validator.admission_priority import ThroughputAdmissionQueue

MATH = "openmathinstruct"


def _offer(queue, receipt, *, hotkey, at, payload_bytes, env=MATH):
    return queue.offer(
        receipt_id=receipt,
        hotkey=hotkey,
        environment=env,
        payload_bytes=payload_bytes,
        arrived_at=at,
    )


def test_the_faster_producer_is_served_first_even_when_it_arrives_later():
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)

    # slow: 1000 bytes over 50 s  = 20 B/s
    _offer(queue, "slow", hotkey="a", at=50.0, payload_bytes=1000)
    # fast: 9000 bytes over 60 s  = 150 B/s, and it arrived LAST
    _offer(queue, "fast", hotkey="b", at=60.0, payload_bytes=9000)

    assert queue.take_best(MATH).receipt_id == "fast"
    assert queue.take_best(MATH).receipt_id == "slow"
    assert queue.take_best(MATH) is None


def test_a_steady_miner_keeps_a_steady_rate_across_the_window():
    """Elapsed is per-submission, not since window open.

    Measured from window open, a miner's Nth precommit shows elapsed
    N x generation_time, so its apparent rate decays as 1/N and only its first
    submission ever competes. The rate must describe the group that was just
    produced, so it runs from that hotkey's previous arrival.
    """
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)

    first = _offer(queue, "r1", hotkey="steady", at=20.0, payload_bytes=2000)
    second = _offer(queue, "r2", hotkey="steady", at=40.0, payload_bytes=2000)
    third = _offer(queue, "r3", hotkey="steady", at=60.0, payload_bytes=2000)

    assert first.throughput == second.throughput == third.throughput


def test_the_first_submission_of_a_hotkey_runs_from_window_open():
    """There is no previous arrival to run from, and the window open is the
    earliest moment the work could have started — the seed did not exist
    before it."""
    queue = ThroughputAdmissionQueue(window_opened_at=100.0)

    entry = _offer(queue, "r1", hotkey="a", at=125.0, payload_bytes=5000)

    assert entry.elapsed == 25.0


def test_equal_rates_break_on_arrival_then_receipt():
    """Order must not depend on insertion accidents.

    Rates collide often — two miners on the same hardware produce the same
    ratio — and an order that fell out of list ordering would make a window
    unreproducible when replaying it from the archive.
    """
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "later", hotkey="b", at=30.0, payload_bytes=3000)
    _offer(queue, "earlier", hotkey="a", at=20.0, payload_bytes=2000)

    assert queue.take_best(MATH).receipt_id == "earlier"


def test_environments_queue_independently():
    """Math and Code fill at different rates and close independently, so one
    cannot be allowed to starve the other's queue."""
    queue = ThroughputAdmissionQueue(window_opened_at=0.0)
    _offer(queue, "math", hotkey="a", at=10.0, payload_bytes=9000)
    _offer(queue, "code", hotkey="b", at=10.0, payload_bytes=100,
           env="opencodeinstruct")

    assert queue.take_best("opencodeinstruct").receipt_id == "code"
    assert queue.take_best("opencodeinstruct") is None
    assert queue.take_best(MATH).receipt_id == "math"


def test_at_capacity_a_better_arrival_displaces_the_worst_queued_one():
    """The queue is bounded, and refusing the newest arrival would be wrong.

    Arrivals can outrun proof capacity, so the queue needs a bound. Dropping
    whatever arrives last would make the bound a second arrival race — the very
    thing ordering by rate exists to remove. The bound drops the WORST instead.
    """
    queue = ThroughputAdmissionQueue(window_opened_at=0.0, max_pending=2)
    _offer(queue, "slow", hotkey="a", at=100.0, payload_bytes=100)     # 1 B/s
    _offer(queue, "mid", hotkey="b", at=10.0, payload_bytes=1000)      # 100 B/s

    displaced = queue.offer(
        receipt_id="fast", hotkey="c", environment=MATH,
        payload_bytes=9000, arrived_at=10.0,                           # 900 B/s
    )

    assert displaced is not None
    assert queue.take_best(MATH).receipt_id == "fast"
    assert queue.take_best(MATH).receipt_id == "mid"
    assert queue.take_best(MATH) is None


def test_at_capacity_a_worse_arrival_is_refused():
    queue = ThroughputAdmissionQueue(window_opened_at=0.0, max_pending=1)
    _offer(queue, "fast", hotkey="a", at=10.0, payload_bytes=9000)

    assert queue.offer(
        receipt_id="slow", hotkey="b", environment=MATH,
        payload_bytes=10, arrived_at=100.0,
    ) is None
    assert queue.take_best(MATH).receipt_id == "fast"


def test_a_refused_offer_still_advances_that_hotkey_s_clock():
    """The work was produced; we simply had no room for it.

    Leaving the clock behind would make the NEXT submission measure from the
    refused one's start, inflating its elapsed and depressing its rate —
    penalising a miner twice for a refusal that was not its doing.
    """
    queue = ThroughputAdmissionQueue(window_opened_at=0.0, max_pending=1)
    _offer(queue, "other", hotkey="other", at=5.0, payload_bytes=9000)

    refused = _offer(queue, "r1", hotkey="a", at=40.0, payload_bytes=10)
    assert refused is None

    queue.take_best(MATH)
    accepted = _offer(queue, "r2", hotkey="a", at=60.0, payload_bytes=2000)

    assert accepted.elapsed == 20.0
