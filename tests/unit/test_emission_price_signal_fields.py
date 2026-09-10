"""The three fields a window contributes to the archive.

They travel together or not at all. A record that carries the window bounds but
not the readiness would be read as a SHORTAGE -- the one regime that needs no
confirmation and snaps the price up -- when all that happened is that the
validator could not measure. Silence has to look like silence.
"""

from __future__ import annotations

from reliquary.validator.emission_price import price_signal_fields


def test_a_measurable_window_yields_all_three():
    fields = price_signal_fields(
        open_round=1000,
        close_round=1100,
        arrivals_by_environment={"openmathinstruct": {7: [1010], 9: [1020]}},
        targets_by_environment={"openmathinstruct": 2},
    )

    assert fields == {
        "window_open_round": 1000,
        "window_close_round": 1100,
        "collect_ready_round": 1020,
    }


def test_a_target_never_met_is_reported_as_a_shortage():
    """This one DOES travel: the window was measured and it did not fill."""
    fields = price_signal_fields(
        open_round=1000,
        close_round=1100,
        arrivals_by_environment={"openmathinstruct": {7: [1010]}},
        targets_by_environment={"openmathinstruct": 2},
    )

    assert fields == {
        "window_open_round": 1000,
        "window_close_round": 1100,
        "collect_ready_round": None,
    }


def test_an_unmeasurable_window_yields_nothing():
    """Arrivals unavailable is not the same as arrivals insufficient."""
    assert (
        price_signal_fields(
            open_round=1000,
            close_round=1100,
            arrivals_by_environment=None,
            targets_by_environment={"openmathinstruct": 2},
        )
        is None
    )


def test_a_window_without_a_beacon_round_yields_nothing():
    """``window_open_drand_round`` is None outside fill-closed, or if the
    beacon was unavailable. No clock, no signal."""
    assert (
        price_signal_fields(
            open_round=None,
            close_round=1100,
            arrivals_by_environment={"openmathinstruct": {7: [1010]}},
            targets_by_environment={"openmathinstruct": 1},
        )
        is None
    )


def test_a_window_that_never_stamped_its_seal_yields_nothing():
    assert (
        price_signal_fields(
            open_round=1000,
            close_round=None,
            arrivals_by_environment={"openmathinstruct": {7: [1010]}},
            targets_by_environment={"openmathinstruct": 1},
        )
        is None
    )
