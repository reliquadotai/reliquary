"""The controller records when the trainer waits for rollouts, so a run's bill can leave those hours out."""
from __future__ import annotations

from reliquary.validator.rl_buffer import SubmittedGroup, TrainingBuffer, starved_intervals
from tests.unit.test_rl_leases import Clock, book_at


def _group(lease_id: str) -> SubmittedGroup:
    return SubmittedGroup(lease_id, "hk", "math", 0, 1, "00", 0.0, (1.0,), (False,), (1,), ((2, 3),), (None,))


def _setup(tmp_path):
    clock = Clock(1000.0)
    book = book_at(tmp_path, clock=clock)
    return clock, book, TrainingBuffer(book, clock=clock)


def test_a_short_wait_is_not_starvation(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    buffer.take(1)                       # empty: the wait starts
    clock.t += 120.0
    buffer.push(_group("a"))
    buffer.take(1)                       # served
    assert starved_intervals(book.db) == []


def test_a_long_wait_counts_after_its_first_five_minutes(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    for _ in range(40):                  # still waiting, asking every 20 s: one wait, not forty
        buffer.take(1)
        clock.t += 20.0
    buffer.push(_group("a"))
    buffer.take(1)
    assert starved_intervals(book.db) == [(1300.0, 1800.0)]


def test_an_open_wait_is_reported_up_to_now(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    for _ in range(46):                  # the trainer keeps asking every 20 s
        buffer.take(1)
        clock.t += 20.0
    assert starved_intervals(book.db, now=1910.0) == [(1300.0, 1910.0)]
    assert starved_intervals(book.db, now=1200.0) == []


def test_an_open_wait_stops_at_the_last_poll(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    for _ in range(26):                  # asks until 1500, then the trainer is gone
        buffer.take(1)
        clock.t += 20.0
    assert starved_intervals(book.db, now=9000.0) == [(1300.0, 1530.0)]


def _poll_until(clock, buffer, until):
    while clock.t < until:
        buffer.take(1)
        clock.t += 20.0


def test_a_dead_trainer_gap_is_never_starvation(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1401.0)   # asks until 1400, then the pod dies
    clock.t = 5000.0                     # a replacement asks again
    published = starved_intervals(book.db, now=4000.0)
    assert published == [(1300.0, 1430.0)]
    _poll_until(clock, buffer, 5401.0)
    buffer.push(_group("a"))
    clock.t = 5400.0
    buffer.take(1)                       # served: the wait ends
    # the dead piece (1430, 5000) is left out, and the grace is not restarted after it
    assert starved_intervals(book.db, now=9000.0) == [(1300.0, 1430.0), (5000.0, 5400.0)]


def test_the_grace_counts_live_waiting_only(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1101.0)   # 130 s live
    clock.t = 3000.0
    _poll_until(clock, buffer, 3401.0)
    buffer.push(_group("a"))
    clock.t = 3400.0
    buffer.take(1)
    assert starved_intervals(book.db) == [(3170.0, 3400.0)]   # 130 + 170 s of grace, then starved
