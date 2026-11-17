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
    buffer.take(1)
    clock.t += 20.0
    buffer.take(1)                       # still waiting: one wait, not two
    clock.t += 780.0
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
