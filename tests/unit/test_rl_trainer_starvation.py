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
    assert starved_intervals(book.db, now=9000.0) == [(1300.0, 1680.0)]   # last take + 180 s


def _poll_until(clock, buffer, until):
    while clock.t < until:
        buffer.take(1)
        clock.t += 20.0


def test_a_dead_trainer_gap_is_never_starvation(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1401.0)   # asks until 1400, then the pod dies
    clock.t = 5000.0                     # a replacement asks again
    published = starved_intervals(book.db, now=4000.0)
    assert published == [(1300.0, 1580.0)]
    _poll_until(clock, buffer, 5401.0)
    buffer.push(_group("a"))
    clock.t = 5400.0
    buffer.take(1)                       # served: the wait ends
    # the dead piece (1580, 5000) is left out, and the grace is not restarted after it
    assert starved_intervals(book.db, now=9000.0) == [(1300.0, 1580.0), (5000.0, 5400.0)]


def test_the_grace_counts_live_waiting_only(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1101.0)   # 280 s live: up to the last take + 180 s
    clock.t = 3000.0
    _poll_until(clock, buffer, 3401.0)
    buffer.push(_group("a"))
    clock.t = 3400.0
    buffer.take(1)
    assert starved_intervals(book.db) == [(3020.0, 3400.0)]   # 280 + 20 s of grace, then starved


def test_a_gap_as_long_as_an_audit_round_stays_starved(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    while clock.t < 1900.0:              # the trainer judges audits between takes: 150 s apart
        buffer.take(1)
        clock.t += 150.0
    buffer.push(_group("a"))
    clock.t = 1900.0
    buffer.take(1)
    assert starved_intervals(book.db) == [(1300.0, 1900.0)]


def test_a_long_gap_is_dead_past_its_first_180_s(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1401.0)
    clock.t = 2000.0                     # 600 s without a take
    _poll_until(clock, buffer, 2401.0)
    buffer.push(_group("a"))
    clock.t = 2400.0
    buffer.take(1)
    assert starved_intervals(book.db) == [(1300.0, 1580.0), (2000.0, 2400.0)]


def test_a_controller_restart_starts_a_fresh_wait(tmp_path):
    clock, book, buffer = _setup(tmp_path)
    _poll_until(clock, buffer, 1401.0)
    clock.t = 5000.0
    buffer = TrainingBuffer(book, clock=clock)   # the controller restarts: its downtime is neither starved nor dead
    _poll_until(clock, buffer, 5401.0)
    buffer.push(_group("a"))
    clock.t = 5400.0
    buffer.take(1)
    assert starved_intervals(book.db) == [(1300.0, 1400.0), (5300.0, 5400.0)]
