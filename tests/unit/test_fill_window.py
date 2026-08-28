"""Admission stops on proven + in-flight; the close fires on proven alone."""
import pytest

from reliquary.validator.fill_window import FillState

MATH, CODE = "openmathinstruct", "opencodeinstruct"


def _state(target=2):
    return FillState(targets={MATH: target, CODE: target})


def test_admission_counts_in_flight_work():
    """Gating admission on proven alone would over-admit by the whole proof
    pipeline depth: every reservation still in flight would look like room."""
    state = _state(target=2)
    state.reserve(MATH)
    state.reserve(MATH)

    assert state.may_admit(MATH) is False


def test_the_close_ignores_in_flight_work():
    """Closing on proven + in-flight would close on work that may still fail
    GRAIL, and a failed proof is not a group."""
    state = _state(target=2)
    state.reserve(MATH)
    state.reserve(MATH)
    state.record_proven(MATH)
    state.record_proven(CODE)
    state.record_proven(CODE)

    assert state.is_closed() is False


def test_a_released_reservation_reopens_capacity():
    """A failed grade or proof must not consume a slot forever, or a run of
    failures would stall the window below its target."""
    state = _state(target=1)
    state.reserve(MATH)
    assert state.may_admit(MATH) is False

    state.release(MATH)

    assert state.may_admit(MATH) is True


def test_the_window_closes_only_when_every_environment_is_full():
    state = _state(target=1)
    state.record_proven(MATH)
    assert state.is_closed() is False

    state.record_proven(CODE)

    assert state.is_closed() is True


def test_releasing_what_was_never_reserved_is_a_bug_not_a_no_op():
    """Silent tolerance here would hide a double-release, which would let the
    environment admit past its target."""
    state = _state()
    with pytest.raises(ValueError):
        state.release(MATH)
