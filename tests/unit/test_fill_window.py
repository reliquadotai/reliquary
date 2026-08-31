"""Admission is bounded by a monotone per-env budget; the window closes at
the Nth pick, not on proven count (v6.1, R33/R35)."""
import threading

import pytest

from reliquary.validator.fill_window import FillState

MATH, CODE = "openmathinstruct", "opencodeinstruct"


def _state(budget=2, picks_target=16):
    return FillState(
        budgets={MATH: budget, CODE: budget}, picks_target=picks_target
    )


def test_admission_counts_in_flight_work():
    """Gating admission on proven alone would over-admit by the whole proof
    pipeline depth: every reservation still in flight would look like room."""
    state = _state(budget=2)
    state.reserve(MATH)
    state.reserve(MATH)

    assert state.may_admit(MATH) is False


def test_a_released_reservation_frees_in_flight_but_not_budget():
    """A failed grade or proof frees proof CAPACITY (``in_flight``) but not
    the budget it already spent: the grading cost was real. Budget is a
    monotone counter -- only ``reserve()`` ever moves it, never
    ``release()`` -- which is the whole point of a budget over a target."""
    state = _state(budget=1)
    state.reserve(MATH)
    assert state.may_admit(MATH) is False

    state.release(MATH)

    assert state.snapshot()["in_flight"][MATH] == 0
    assert state.may_admit(MATH) is False


def test_budget_is_monotone_across_many_reserve_release_cycles():
    """Reserve-then-release, repeated, must never reopen admission: each
    ``reserve()`` is a real grading attempt the fleet already paid for,
    regardless of how many times capacity later frees."""
    state = _state(budget=3)
    for _ in range(3):
        state.reserve(MATH)
        state.release(MATH)

    assert state.may_admit(MATH) is False
    assert state.snapshot()["admitted"][MATH] == 3


def test_releasing_what_was_never_reserved_is_a_bug_not_a_no_op():
    """Silent tolerance here would hide a double-release, which would let
    in-flight bookkeeping go negative."""
    state = _state()
    with pytest.raises(ValueError):
        state.release(MATH)


def test_record_proven_does_not_by_itself_close_the_window():
    """R35: the window used to close when every environment reached its
    proven target. It no longer does -- proven groups now only accumulate
    in the pool for a pick to choose from; only picks close the window."""
    state = _state(budget=1, picks_target=1)
    state.record_proven(MATH)
    state.record_proven(CODE)

    assert state.is_closed() is False


def test_the_window_closes_at_the_nth_pick():
    state = _state(picks_target=2)
    state.record_pick()
    assert state.is_closed() is False

    state.record_pick()

    assert state.is_closed() is True


def test_a_pick_past_the_target_raises():
    """A pick past ``picks_target`` is a caller bug -- the window should
    already have sealed at the target-th pick."""
    state = _state(picks_target=1)
    state.record_pick()

    with pytest.raises(ValueError):
        state.record_pick()


def test_picks_target_must_be_positive():
    with pytest.raises(ValueError):
        FillState(budgets={MATH: 1}, picks_target=0)


def test_the_instance_owns_a_lock_for_sharing_across_batchers():
    """One ``FillState`` is shared across every per-environment batcher for
    a window (R10). Sharing the OBJECT but not a lock on it would mean two
    batchers each taking their own separate lock around the same instance
    -- no lock at all. The lock has to live on the shared instance itself."""
    state = _state()

    assert isinstance(state.lock, type(threading.Lock()))


def test_snapshot_exposes_the_new_accounting_fields():
    state = _state(budget=5, picks_target=3)
    state.reserve(MATH)
    state.record_pick()

    snap = state.snapshot()

    assert snap["budgets"] == {MATH: 5, CODE: 5}
    assert snap["admitted"] == {MATH: 1, CODE: 0}
    assert snap["in_flight"] == {MATH: 1, CODE: 0}
    assert snap["proven"] == {MATH: 0, CODE: 0}
    assert snap["picks_emitted"] == 1
    assert snap["picks_target"] == 3
    assert snap["closed"] is False
