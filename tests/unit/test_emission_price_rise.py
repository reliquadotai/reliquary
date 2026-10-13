"""The price rises when the trainer waited on an environment's supply.

``r`` cannot carry that signal: its denominator, the training span, grows with
the very wait it should report, so it saturates near 1 exactly when collection
is the bottleneck. The service measures the wait itself and the archive carries
it per environment.
"""

from __future__ import annotations

import pytest

from reliquary.validator.emission_price import (
    PRODUCTION_PRICE_PARAMS,
    ROUND_SECONDS,
    EnvironmentOutcome,
    PriceParams,
    PriceState,
    advance,
    outcomes_by_environment_from_archive,
    price_signal_fields,
    restore_walk,
)


def _params(**overrides) -> PriceParams:
    base = dict(
        start=1.0, decay=0.5, rounds_per_step=100, deadband=0.80, snap=1.20,
        floor=0.01, cap=1.0, median_rounds=1_000_000, last_good_fills=1_000_000,
        rise=1.10, rise_wait_fraction=0.05,
    )
    base.update(overrides)
    return PriceParams(**base)


def _outcome(*, wait, ready=10, elapsed=150, timed_out=False, training=100.0):
    # 150 rounds = 450 s: a typical window.
    return EnvironmentOutcome(
        environment="math", open_round=0, close_round=elapsed,
        ready_round=ready, incompressible_rounds=training,
        timed_out=timed_out, trainer_wait_seconds=wait,
    )


def _state(price=0.5):
    return PriceState(price=price, last_good=price, recent_fill_prices=(price,))


def test_production_params_rise_ten_percent_past_five_percent_of_the_window():
    assert PRODUCTION_PRICE_PARAMS.rise == pytest.approx(1.10)
    assert PRODUCTION_PRICE_PARAMS.rise_wait_fraction == pytest.approx(0.05)
    assert ROUND_SECONDS == 3.0


def test_a_wait_over_the_threshold_raises_the_price():
    decision = advance(_state(), [_outcome(wait=45.0)], _params())  # 10 % of 450 s
    assert decision.regime == "rise"
    assert decision.price == pytest.approx(0.55)


@pytest.mark.parametrize("wait", [9.0, None])
def test_a_small_or_unmeasured_wait_does_not_rise(wait):
    # 9 s = 2 % of the window; None = no measurement. ready=10 of 100 training
    # rounds gives r = 0.1, below the deadband: today's descend.
    decision = advance(_state(), [_outcome(wait=wait)], _params())
    assert decision.regime == "descend"


def test_an_unfilled_window_snaps_rather_than_rises():
    decision = advance(_state(), [_outcome(wait=400.0, ready=None)], _params())
    assert decision.regime == "snap"


def test_a_timed_out_window_does_not_rise():
    decision = advance(_state(), [_outcome(wait=400.0, timed_out=True)], _params())
    assert decision.regime != "rise"


def test_the_rise_is_clamped_at_the_cap():
    decision = advance(_state(0.95), [_outcome(wait=400.0)], _params())
    assert decision.price == pytest.approx(1.0)


def test_a_rise_never_lifts_last_good():
    decision = advance(_state(0.5), [_outcome(wait=400.0)], _params())
    assert decision.last_good == pytest.approx(0.5)


def test_the_math_shortage_doubles_the_price_in_seven_windows():
    state, history = _state(0.084), []
    for _ in range(7):
        history.append(_outcome(wait=400.0, ready=145, training=150.0))
        decision = advance(state, history, PRODUCTION_PRICE_PARAMS)
        state = decision.state
    assert state.price == pytest.approx(0.084 * 1.1 ** 7, rel=1e-6)


def _record(wait_map):
    record = {
        "window_status": "completed", "window_open_round": 0, "window_close_round": 150,
        "collect_ready_round": 40, "training_rounds": 120,
        "collect_ready_round_by_environment": {"math": 40, "code": 30},
    }
    if wait_map is not None:
        record["trainer_wait_seconds_by_environment"] = wait_map
    return record


def test_the_archive_map_reaches_each_environment_outcome():
    outcomes = outcomes_by_environment_from_archive(_record({"math": 33.3, "code": "x"}))
    assert outcomes["math"].trainer_wait_seconds == pytest.approx(33.3)
    assert outcomes["code"].trainer_wait_seconds is None  # unreadable: that env only


def test_an_archive_without_the_map_carries_no_wait():
    outcomes = outcomes_by_environment_from_archive(_record(None))
    assert all(o.trainer_wait_seconds is None for o in outcomes.values())


def test_price_signal_fields_write_the_rounded_map_only_when_given():
    common = dict(open_round=0, close_round=150, arrivals_by_environment={"math": {}},
                  targets_by_environment={"math": 1}, first_pick_round=10)
    fields = price_signal_fields(**common, trainer_wait_seconds_by_environment={"math": 12.345})
    assert fields["trainer_wait_seconds_by_environment"] == {"math": 12.3}
    assert "trainer_wait_seconds_by_environment" not in price_signal_fields(**common)


def test_restoring_from_archives_without_the_map_is_unchanged():
    shadow = {"price": 0.3, "last_good": 0.3, "by_environment": {"math": {"price": 0.2, "last_good": 0.2}}}
    archive = {**_record(None), "emission_price_shadow": shadow}
    walk = restore_walk([archive], PRODUCTION_PRICE_PARAMS)
    assert walk.states_by_environment["math"].price == pytest.approx(0.2)
    assert walk.history_by_environment["math"][0].trainer_wait_seconds is None
