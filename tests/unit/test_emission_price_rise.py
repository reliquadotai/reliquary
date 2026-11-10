"""The price rises when the trainer waited on an environment's supply.

``r`` cannot carry that signal: its denominator, the training span, grows with
the very wait it should report, so it saturates near 1 exactly when collection
is the bottleneck. The service measures the wait itself and the archive carries
it per environment.
"""

from __future__ import annotations

import pytest

from reliquary.validator.emission_price import PRICE_RISE, PRICE_RISE_ENV, PRODUCTION_PRICE_PARAMS, ROUND_SECONDS, PriceParams, PriceState, advance, price_rise_from_env


def _params(**overrides) -> PriceParams:
    base = dict(
        start=1.0, decay=0.5, rounds_per_step=100, deadband=0.80, snap=1.20,
        floor=0.01, cap=1.0, median_rounds=1_000_000, last_good_fills=1_000_000,
        rise=1.10, rise_wait_fraction=0.05,
    )
    base.update(overrides)
    return PriceParams(**base)


def _state(price=0.5):
    return PriceState(price=price, last_good=price, recent_fill_prices=(price,))


def test_production_params_rise_is_off_by_default_past_five_percent_of_the_window(monkeypatch):
    monkeypatch.delenv(PRICE_RISE_ENV, raising=False)
    assert price_rise_from_env() == 1.0
    assert PRODUCTION_PRICE_PARAMS.rise == pytest.approx(PRICE_RISE)
    assert PRODUCTION_PRICE_PARAMS.rise_wait_fraction == pytest.approx(0.05)
    assert ROUND_SECONDS == 3.0


def _record(wait_map):
    record = {
        "window_status": "completed", "window_open_round": 0, "window_close_round": 150,
        "collect_ready_round": 40, "training_rounds": 120,
        "collect_ready_round_by_environment": {"math": 40, "code": 30},
    }
    if wait_map is not None:
        record["trainer_wait_seconds_by_environment"] = wait_map
    return record


def test_params_built_like_the_registry_carry_the_rise():
    """task_config builds PriceParams from PRICE_PARAM_FIELDS only; the rise
    fields are image-versioned like breaker_timeouts, so the defaults ARE the
    production values (C1: inert defaults left the rise dead in production)."""
    from reliquary.shared.task_registry import PRICE_PARAM_FIELDS

    params = PriceParams(**{f: getattr(PRODUCTION_PRICE_PARAMS, f) for f in PRICE_PARAM_FIELDS})
    assert params.rise == pytest.approx(PRICE_RISE)
    assert params.rise_wait_fraction == pytest.approx(0.05)


def test_the_unset_env_gives_no_rise_in_a_fresh_process():
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != PRICE_RISE_ENV}
    code = ("from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS as p, "
            "PriceParams as P; print(p.rise, P.__dataclass_fields__['rise'].default)")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["1.0", "1.0"]


@pytest.mark.parametrize("raw, expected", [("", 1.0), ("1.10", 1.10), (" 1.0 ", 1.0)])
def test_the_env_flag_sets_the_rise(raw, expected):
    assert price_rise_from_env({PRICE_RISE_ENV: raw}) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["0.9", "nan", "inf", "x"])
def test_a_bad_env_flag_is_refused(raw):
    with pytest.raises(ValueError):
        price_rise_from_env({PRICE_RISE_ENV: raw})
