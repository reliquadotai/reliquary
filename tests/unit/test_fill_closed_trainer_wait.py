"""How long ready picks waited on each environment's supply.

Only a wait while the trainer's pacing gate is OPEN counts: a closed gate means
the trainer is still busy, and nobody's supply is holding it back. Pick 1 is
paced by the opening floor, not by the trainer, so it never counts either.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.validator.service import ValidationService, _window_price_signal


def _service(*, gate_open: bool):
    service = ValidationService.__new__(ValidationService)
    service._fill_trainer_wait_seconds = {}
    service._fill_trainer_wait_last_tick = None
    service._fill_closed_pick_gate_open = lambda batcher, *, next_pick: gate_open
    service._trainer_consumed_through = lambda batcher, *, pick: gate_open
    return service


def _batcher(name: str, can_pick: bool):
    return SimpleNamespace(env=SimpleNamespace(name=name), can_pick=lambda: can_pick)


def _ticks(service, picking, *, next_pick, times):
    for now in times:
        service._accrue_trainer_wait(picking, next_pick=next_pick, now=now)


def test_an_open_gate_charges_the_short_environment_only():
    service = _service(gate_open=True)
    picking = [_batcher("math", False), _batcher("code", True)]
    _ticks(service, picking, next_pick=3, times=[100.0, 100.5, 101.0, 101.5])
    assert service._fill_trainer_wait_seconds == {"math": pytest.approx(1.5)}


def test_a_closed_gate_charges_nobody():
    service = _service(gate_open=False)
    _ticks(service, [_batcher("math", False)], next_pick=3, times=[100.0, 100.5, 101.0])
    assert service._fill_trainer_wait_seconds == {}


def test_the_first_pick_is_paced_by_the_floor_not_the_trainer():
    service = _service(gate_open=True)
    _ticks(service, [_batcher("math", False)], next_pick=1, times=[100.0, 100.5, 101.0])
    assert service._fill_trainer_wait_seconds == {}


def test_a_stalled_loop_is_clamped_at_five_seconds():
    service = _service(gate_open=True)
    _ticks(service, [_batcher("math", False)], next_pick=2, times=[100.0, 130.0])
    assert service._fill_trainer_wait_seconds == {"math": pytest.approx(5.0)}


def test_opening_a_window_resets_the_accumulator():
    service = _service(gate_open=True)
    service._fill_trainer_wait_seconds = {"math": 12.0}
    service._fill_trainer_wait_last_tick = 99.0
    service._reset_fill_trainer_wait()
    assert service._fill_trainer_wait_seconds == {}
    assert service._fill_trainer_wait_last_tick is None


def test_the_window_price_signal_forwards_the_map():
    pending = SimpleNamespace(drand_round=12)
    batcher = SimpleNamespace(
        _submissions_per_prompt={7: [pending]}, window_first_pick_drand_round=20,
        _seal_trigger_round=None, window_close_drand_round=160, window_open_drand_round=10,
    )
    fields = _window_price_signal(
        batcher, {"math": batcher}, lambda name, b: 1,
        trainer_wait_seconds_by_environment={"math": 41.26},
    )
    assert fields["trainer_wait_seconds_by_environment"] == {"math": 41.3}


def test_a_trainer_still_busy_on_the_buffered_batch_is_not_waiting():
    """Pipeline depth 1: the gate for pick k opens when the trainer reached pick
    k-2, while pick k-1 is still queued. Only a trainer that consumed every
    emitted pick (cursor at k-1) is idle and waiting on supply (I2)."""
    service = _service(gate_open=True)
    service._trainer_consumed_through = lambda batcher, *, pick: False
    _ticks(service, [_batcher("math", False)], next_pick=4, times=[100.0, 100.5, 101.0])
    assert service._fill_trainer_wait_seconds == {}
