"""The window pool follows the price: each environment's cap, scaled by its own price."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS, PriceState

ENVIRONMENTS = ["openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"]

SIGNAL = {
    "window_open_round": 0,
    "window_close_round": 1000,
    "collect_ready_round": 100,
    "collect_ready_round_by_environment": {"openmathinstruct": 100},
}


def _service(*, env_caps=None, emission_cap=1.0, params=PRODUCTION_PRICE_PARAMS):
    from reliquary.validator.service import ValidationService

    service = ValidationService.__new__(ValidationService)
    service._price_params = params
    service._env_caps = dict(env_caps or {})
    service._emission_cap = emission_cap
    return service


@pytest.fixture
def disarmed(monkeypatch):
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)


def test_the_price_is_armed_by_default():
    import reliquary.constants as constants

    assert constants.EMISSION_PRICE_ARMED is True


def test_an_unpriced_walk_pays_exactly_what_it_pays_today():
    """Until the price moves, the assembler receives the very same object as before."""
    scalar = _service()
    assert scalar._window_pool_for(ENVIRONMENTS) is scalar._emission_cap

    split = _service(env_caps={
        "openmathinstruct": 0.5, "opencodeinstruct": 0.3, "reliquary_logic_v2": 0.2,
    })
    split._price_shadow_state = PriceState(price=1.0, last_good=1.0)
    assert split._window_pool_for(ENVIRONMENTS) is split._env_caps


def test_a_descended_price_scales_every_environment_pool():
    service = _service()
    service._price_shadow_state = PriceState(price=0.5, last_good=0.5)

    assert service._window_pool_for(ENVIRONMENTS) == pytest.approx(
        {environment: 0.5 / 3 for environment in ENVIRONMENTS}
    )


def test_each_environment_is_paid_at_its_own_price():
    service = _service()
    service._price_shadow_state = PriceState(price=0.6, last_good=0.6)
    service._price_shadow_states_by_environment = {
        "openmathinstruct": PriceState(price=0.4, last_good=0.4),
        "opencodeinstruct": PriceState(price=0.8, last_good=0.8),
    }

    assert service._window_pool_for(ENVIRONMENTS) == pytest.approx({
        "openmathinstruct": 0.4 / 3,
        "opencodeinstruct": 0.8 / 3,
        # No walk of its own yet: it is paid at the window's price.
        "reliquary_logic_v2": 0.6 / 3,
    })


def test_a_narrowed_mix_is_renormalised_before_the_price_applies():
    service = _service(env_caps={
        "openmathinstruct": 0.6, "opencodeinstruct": 0.3, "reliquary_logic_v2": 0.1,
    })
    service._price_shadow_state = PriceState(price=0.5, last_good=0.5)

    assert service._window_pool_for(["openmathinstruct", "opencodeinstruct"]) == pytest.approx({
        "openmathinstruct": 0.6 / 0.9 * 0.5,
        "opencodeinstruct": 0.3 / 0.9 * 0.5,
    })


def test_a_smaller_task_prices_relative_to_its_own_cap():
    params = replace(PRODUCTION_PRICE_PARAMS, start=0.5, cap=0.5)
    service = _service(emission_cap=0.5, params=params)
    service._price_shadow_state = PriceState(price=0.25, last_good=0.25)

    assert service._window_pool_for(ENVIRONMENTS) == pytest.approx(
        {environment: 0.5 / 3 * 0.5 for environment in ENVIRONMENTS}
    )


def test_the_pool_never_exceeds_what_the_task_declared():
    rng = random.Random(81)
    for _ in range(500):
        service = _service()
        service._price_shadow_states_by_environment = {
            environment: PriceState(price=rng.uniform(0.05, 1.0), last_good=0.05)
            for environment in ENVIRONMENTS
        }

        assert math.fsum(service._window_pool_for(ENVIRONMENTS).values()) <= 1.0


def test_a_missing_declared_environment_is_left_for_the_assembler_to_refuse():
    service = _service(env_caps={"openmathinstruct": 0.5, "opencodeinstruct": 0.5})
    service._price_shadow_state = PriceState(price=0.5, last_good=0.5)

    assert service._window_pool_for(ENVIRONMENTS) is service._env_caps


def test_the_kill_switch_pays_the_declared_pool_whatever_the_price(disarmed):
    service = _service()
    service._price_shadow_state = PriceState(price=0.2, last_good=0.2)

    assert service._window_pool_for(ENVIRONMENTS) is service._emission_cap


def test_an_armed_decision_says_it_is_applied():
    shadow = _service()._advance_price_shadow(dict(SIGNAL))

    assert shadow["applied"] is True
    assert shadow["by_environment"]["openmathinstruct"]["applied"] is True


def test_a_disarmed_decision_says_it_is_not_applied(disarmed):
    shadow = _service()._advance_price_shadow(dict(SIGNAL))

    assert shadow["applied"] is False
    assert shadow["by_environment"]["openmathinstruct"]["applied"] is False


def _built_window_assembler(monkeypatch, *, price=None):
    from tests.unit.test_fill_close_and_emit import _build_two_env_fill_closed_service

    service = _build_two_env_fill_closed_service(monkeypatch, enabled=True)
    if price is not None:
        service._price_shadow_state = PriceState(price=price, last_good=price)
    batchers = service._build_window_batchers(999)
    return service, batchers["openmathinstruct"]._emit_training_batch_fn.__self__


def test_a_window_is_built_at_the_price_in_force_when_it_opens(monkeypatch):
    """The pool is wired where windows are built, not only computed beside it."""
    service, assembler = _built_window_assembler(monkeypatch, price=0.5)

    assert assembler.window_pool == pytest.approx(0.5 * service._emission_cap)
    assert assembler.pool_for("openmathinstruct") == pytest.approx(
        0.5 * service._emission_cap / 2
    )


def test_a_window_built_before_any_price_pays_the_declared_pool(monkeypatch):
    service, assembler = _built_window_assembler(monkeypatch)

    assert assembler.window_pool == service._emission_cap
