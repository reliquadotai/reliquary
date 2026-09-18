"""What a miner can read about the price before spending a GPU-hour on a window."""

from __future__ import annotations

import pytest

from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS, WindowOutcome
from reliquary.validator.price_view import price_view

ROUND_SECONDS = 3.0


def _outcomes(n=6, *, cycle_rounds=336, elapsed_rounds=244, ready_after=25):
    """Consecutive windows opening ``cycle_rounds`` apart, each filling after
    ``ready_after`` rounds and sealing ``elapsed_rounds`` after it opened."""
    return [
        WindowOutcome(
            open_round=1000 + i * cycle_rounds,
            close_round=1000 + i * cycle_rounds + elapsed_rounds,
            collect_ready_round=1000 + i * cycle_rounds + ready_after,
            incompressible_rounds=float(elapsed_rounds),
        )
        for i in range(n)
    ]


def _shadow(price=0.654, regime="descend", applied=False, **extra):
    return {
        "price": price, "last_good": price, "r": 0.1, "r_smoothed": 0.1,
        "regime": regime, "applied": applied, **extra,
    }


def _view(shadow, outcomes=None, *, window_pool=1.0, places=336):
    return price_view(
        shadow=shadow,
        outcomes=_outcomes() if outcomes is None else outcomes,
        params=PRODUCTION_PRICE_PARAMS,
        window_pool=window_pool,
        places_per_window=places,
        round_seconds=ROUND_SECONDS,
    )


def test_no_decision_yet_means_no_price_block():
    assert _view(None) is None


def test_the_block_says_whether_the_price_is_actually_paid():
    view = _view(_shadow(applied=False))

    assert view["applied"] is False
    assert view["value"] == 0.654
    assert view["regime"] == "descend"
    assert (view["floor"], view["cap"], view["decay"]) == (0.05, 1.0, 0.98)


def test_fill_time_and_its_target_are_in_seconds():
    view = _view(_shadow(), _outcomes(ready_after=25, elapsed_rounds=244))

    assert view["fill_seconds_recent"] == pytest.approx(25 * ROUND_SECONDS)
    assert view["fill_target_seconds"] == pytest.approx(0.80 * 244 * ROUND_SECONDS)


def test_no_filled_window_means_no_fill_time():
    unfilled = [
        WindowOutcome(open_round=1000, close_round=1244, collect_ready_round=None,
                      incompressible_rounds=244.0),
        WindowOutcome(open_round=1336, close_round=1580, collect_ready_round=None,
                      incompressible_rounds=244.0),
    ]

    assert _view(_shadow(regime="snap"), unfilled)["fill_seconds_recent"] is None


def test_pay_per_place_shows_what_is_paid_now_and_what_the_price_would_pay():
    view = _view(_shadow(price=0.654, applied=False), window_pool=1.0)

    assert view["pay_per_place_share_of_window"] == pytest.approx(1.0 / 336)
    assert view["pay_per_place_if_applied"] == pytest.approx(0.654 / 336)


def test_an_applied_price_pays_what_it_shows():
    view = _view(_shadow(price=0.654, applied=True), window_pool=0.654)

    assert view["pay_per_place_share_of_window"] == pytest.approx(0.654 / 336)
    assert "pay_per_place_if_applied" not in view


def test_unknown_places_mean_no_pay_per_place():
    view = _view(_shadow(), places=None)

    assert view["pay_per_place_share_of_window"] is None


def test_a_descending_price_projects_its_own_rule_forward():
    """Descent accrues only inside windows, so wall time is scaled by the share
    of each cycle spent in one: 244 of 336 rounds here."""
    view = _view(_shadow(price=0.654), _outcomes(cycle_rounds=336, elapsed_rounds=244))

    share = 244 / 336
    for label, seconds in (("1h", 3600), ("6h", 21600), ("24h", 86400)):
        expected = 0.654 * 0.98 ** (seconds / ROUND_SECONDS * share / 1000)
        assert view["projection"][label] == pytest.approx(expected)


def test_a_projection_never_falls_below_the_floor():
    assert _view(_shadow(price=0.06))["projection"]["24h"] == 0.05


def test_a_holding_price_projects_flat():
    assert _view(_shadow(price=0.4, regime="hold"))["projection"] == {
        "1h": 0.4, "6h": 0.4, "24h": 0.4,
    }


@pytest.mark.parametrize("regime", ["snap", "frozen"])
def test_a_snapping_or_frozen_price_makes_no_projection(regime):
    assert _view(_shadow(regime=regime))["projection"] is None


def test_one_window_is_not_enough_history_to_project():
    assert _view(_shadow(), _outcomes(n=1))["projection"] is None


def test_each_environment_shows_its_own_price():
    shadow = _shadow(by_environment={
        "openmathinstruct": {"price": 0.6, "regime": "descend", "applied": False},
        "opencodeinstruct": {"price": 0.7, "regime": "hold", "applied": False},
    })

    assert _view(shadow)["by_environment"] == {
        "openmathinstruct": {"value": 0.6, "regime": "descend"},
        "opencodeinstruct": {"value": 0.7, "regime": "hold"},
    }


def test_the_fill_target_follows_the_denominator_the_controller_divides_by():
    """A window that reports its own training span is judged against THAT span,
    so the target a miner reads has to be the same one -- not the window's
    duration, which is longer and would advertise a target nobody is held to."""
    outcomes = [
        WindowOutcome(
            open_round=1000 + i * 336,
            close_round=1000 + i * 336 + 244,
            collect_ready_round=1000 + i * 336 + 25,
            incompressible_rounds=200.0,
        )
        for i in range(6)
    ]

    view = _view(_shadow(), outcomes)

    assert view["fill_target_seconds"] == pytest.approx(0.80 * 200 * ROUND_SECONDS)
