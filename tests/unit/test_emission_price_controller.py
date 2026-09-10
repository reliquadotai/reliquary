"""The emission price controller, replayed from window outcomes.

The controller answers one question per window: is collection on the cycle's
critical path? When it is not, the subnet is paying for speed it cannot
consume, and the price walks down; the unpaid share burns.

Every constant is expressed in BLOCKS, never in windows. ``EMA_ALPHA`` is the
cautionary tale: calibrated for ~5-minute windows, it silently became a ~28-hour
time constant when fill-closed made windows ten times longer. Nobody chose that.
A controller counted in windows would recalibrate itself the same way the next
time the cadence moves -- and the cadence is expected to move.
"""

from __future__ import annotations

import pytest

from reliquary.validator.emission_price import (
    PriceParams,
    PriceState,
    WindowOutcome,
    advance,
    replay,
)


def _params(**overrides) -> PriceParams:
    """Parameters chosen for legibility in tests, not for production."""
    base = dict(
        start=1.0,
        decay=0.5,          # a big step so arithmetic stays readable
        blocks_per_step=100,
        deadband=0.80,
        snap=1.20,
        floor=0.01,
        cap=1.0,
        median_blocks=1_000_000,   # effectively "all history" unless overridden
    )
    base.update(overrides)
    return PriceParams(**base)


_INCOMPRESSIBLE = 100


def _window(open_block: int, span: int, *, r: float | None) -> WindowOutcome:
    """A window whose collection landed at ratio ``r``; ``r=None`` never filled."""
    ready = None if r is None else open_block + round(r * _INCOMPRESSIBLE)
    return WindowOutcome(
        open_block=open_block,
        close_block=open_block + span,
        collect_ready_block=ready,
        training_blocks=_INCOMPRESSIBLE,
        validation_blocks=_INCOMPRESSIBLE // 2,
    )


def _masked(open_block: int, span: int) -> WindowOutcome:
    """Collection finished well inside the incompressible time.

    ``r`` lands at 0.1, far below any sane deadband, so this is the unambiguous
    "we are paying for speed we cannot use" case.
    """
    return _window(open_block, span, r=0.1)


def test_oversupply_walks_the_price_down_geometrically():
    params = _params()

    # One window covering exactly two decay steps.
    decision = replay([_masked(0, 200)], params)

    assert decision.price == 0.25
    assert decision.regime == "descend"


def test_price_depends_on_blocks_elapsed_not_on_window_count():
    """The property that keeps a cadence change from recalibrating the controller.

    Four short windows and one long window covering the same 400 blocks must
    leave the price at the same place. Without this, halving the window
    duration would double the controller's speed -- exactly the way EMA_ALPHA
    drifted when fill-closed lengthened windows.
    """
    params = _params()

    one_long = replay([_masked(0, 400)], params)
    four_short = replay(
        [_masked(0, 100), _masked(100, 100), _masked(200, 100), _masked(300, 100)],
        params,
    )

    assert one_long.price == four_short.price
    assert one_long.price == 0.0625  # 0.5 ** 4


def test_deadband_holds_the_price():
    """Collection close to the incompressible time is the target, not a signal.

    Without this the controller twitches every window around r=1, which is the
    main oscillation source once supply is elastic and every miner reads the
    same public price at the same moment.
    """
    params = _params(deadband=0.80)

    decision = replay([_window(0, 200, r=0.9)], params)

    assert decision.price == 1.0
    assert decision.regime == "hold"


def test_shortage_snaps_the_price_up_immediately():
    """A fill-closed window that never gathers its target stops the trainer.

    There is no graceful degradation to ride out, so recovery does not walk --
    it jumps. This is what lets the descent be fast in the first place.
    """
    params = _params(snap=1.20)

    # Two decay steps down (price 0.25, all filled), then one window that did not.
    decision = replay([_masked(0, 200), _window(200, 100, r=None)], params)

    assert decision.price == pytest.approx(0.30)
    assert decision.regime == "snap"


def test_repeated_shortage_keeps_escalating():
    """Liveness: one snap that fails to restore supply must not be a dead end.

    A snap pinned to the last good price would park there forever with the
    trainer stopped.
    """
    params = _params(snap=1.20)

    once = replay([_masked(0, 200), _window(200, 100, r=None)], params)
    twice = replay(
        [_masked(0, 200), _window(200, 100, r=None), _window(300, 100, r=None)],
        params,
    )

    assert twice.price > once.price
    assert twice.price == pytest.approx(0.36)


def test_a_window_that_never_filled_does_not_become_the_good_price():
    """``last_good`` is the lowest price at which a window actually filled."""
    params = _params(snap=1.20)

    decision = replay([_masked(0, 200), _window(200, 100, r=None)], params)

    assert decision.last_good == pytest.approx(0.25)


def test_price_never_falls_below_the_floor():
    """The floor is a liveness guard, not an economic opinion."""
    params = _params(floor=0.10)

    decision = replay([_masked(0, 1000)], params)   # ten decay steps

    assert decision.price == 0.10


def test_snap_never_exceeds_the_cap():
    params = _params(cap=1.0, snap=1.20)

    decision = replay([_window(0, 100, r=None)], params)

    assert decision.price == 1.0


def test_one_noisy_window_does_not_move_the_price():
    """Descending must be earned: a single fast window is not evidence.

    Fill time is polluted by things that have nothing to do with supply -- the
    precommit stall that forces STALE_ROUND (12.5 s = 4 drand rounds), network
    latency, beacon jitter. Spending less on one sample of that would be noise
    trading.
    """
    params = _params(deadband=0.80)

    decision = replay(
        [_window(0, 100, r=0.9), _window(100, 100, r=0.9), _window(200, 100, r=0.1)],
        params,
    )

    assert decision.price == 1.0
    assert decision.regime == "hold"


def test_smoothing_never_delays_the_snap():
    """The evidence bar is asymmetric because the cost of being wrong is.

    Paying less must be confirmed by the smoothed signal; restoring liveness is
    not up for confirmation and fires on the instantaneous one.
    """
    params = _params(deadband=0.80, cap=2.0, snap=1.20)

    decision = replay(
        [_window(0, 100, r=0.9), _window(100, 100, r=0.9), _window(200, 100, r=None)],
        params,
    )

    assert decision.regime == "snap"
    assert decision.price == pytest.approx(1.20)


def test_the_smoothing_window_is_counted_in_blocks():
    """Same history, two lookbacks, two answers -- so the lookback is real.

    A lookback counted in windows would silently shrink the moment the cadence
    changes, which is precisely how EMA_ALPHA drifted to ~28 hours.
    """
    history = [
        _window(0, 100, r=0.1),
        _window(100, 100, r=0.1),
        _window(200, 100, r=0.9),
        _window(300, 100, r=0.9),
    ]

    long_lookback = replay(history, _params(median_blocks=1000))
    short_lookback = replay(history, _params(median_blocks=150))

    assert long_lookback.regime == "descend"
    assert short_lookback.regime == "hold"


def test_one_step_needs_only_the_previous_state_and_the_lookback():
    """A reader must not need the archive chain back to genesis.

    ``_replay_ema`` reads a BOUNDED slice of archives (216). If reproducing the
    price required folding every window since the run began, a weight-only node
    would land on a different number than the validator that wrote it -- and
    the two would submit different weight vectors. So each archive carries its
    own state, and the next step consumes only that plus the smoothing
    lookback.
    """
    params = _params(median_blocks=250)
    history = [_masked(0, 100), _masked(100, 100), _masked(200, 100), _masked(300, 100)]

    whole_chain = replay(history, params)

    previous = replay(history[:-1], params)
    # close_block > 400 - 250 keeps the last three windows, the decided one included.
    one_step = advance(previous.state, history[-3:], params)

    assert one_step.price == whole_chain.price
    assert one_step.last_good == whole_chain.last_good
    assert one_step.regime == whole_chain.regime


def test_state_survives_a_round_trip_through_plain_values():
    """The state crosses an archive, so it must be ordinary JSON scalars."""
    params = _params()
    decision = replay([_masked(0, 200)], params)

    revived = PriceState(price=float(decision.price), last_good=float(decision.last_good))

    assert advance(revived, [_masked(200, 100)], params).price == pytest.approx(
        replay([_masked(0, 300)], params).price
    )


def test_missing_stage_telemetry_is_not_a_shortage():
    """A window that filled but was not measured carries no ratio, not bad news.

    Collapsing "we did not record the stage durations" into "the window never
    filled" would snap the price UP every time instrumentation hiccups --
    exactly backwards, and self-reinforcing since the snap is the one regime
    that needs no confirmation.
    """
    params = _params(cap=2.0)
    unmeasured = WindowOutcome(
        open_block=0,
        close_block=100,
        collect_ready_block=10,      # it DID fill
        training_blocks=0,           # but nothing was measured
        validation_blocks=0,
    )

    decision = replay([unmeasured], params)

    assert decision.regime == "hold"
    assert decision.price == 1.0


def test_an_unmeasured_window_does_not_enter_the_median():
    """It carries no information, so it must not dilute the ones that do."""
    params = _params(deadband=0.80)
    unmeasured = WindowOutcome(
        open_block=100,
        close_block=200,
        collect_ready_block=110,
        training_blocks=0,
        validation_blocks=0,
    )

    decision = replay([_masked(0, 100), unmeasured, _masked(200, 100)], params)

    # The median is over [0.1, 0.1], not over [0.1, <nothing>, 0.1] read as noise.
    assert decision.regime == "descend"
    assert decision.r_smoothed == pytest.approx(0.1)
