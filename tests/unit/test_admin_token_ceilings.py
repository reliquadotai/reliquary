"""An API order's token ceiling: USD per million prompt and generated tokens,
converted to alpha and to a pool share when its task is declared."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.admin import service
from reliquary.validator import corpus_periods as cp
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import _order, pins  # noqa: F401

ALPHA_TAO, TAO_USD, PER_BLOCK = 0.01, 400.0, 0.4    # alpha = 4 USD
CEILING = {"input": 0.5, "output": 2.0}


@pytest.fixture
def rates(monkeypatch):
    """The chain and TAO price the admin reads, counted; ``down`` fails them."""
    state = {"reads": 0, "down": False, "alpha_tao": ALPHA_TAO}

    async def pool():
        state["reads"] += 1
        if state["down"]:
            raise OSError("chain down")
        return {"alpha_price_tao": state["alpha_tao"], "miner_alpha_per_block": PER_BLOCK}

    monkeypatch.setattr(service, "_read_subnet_pool", pool)
    monkeypatch.setenv(service.TAO_USD_ENV, str(TAO_USD))
    return state


def test_an_eval_order_converts_its_usd_ceiling_into_the_task(admin, pins, rates):  # noqa: F811
    created = admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING))
    assert created.status_code == 201, created.text
    params = admin.registry["entries"]["order-eval-p1"].params
    emission = PER_BLOCK * cp.PERIOD_SECONDS / 12
    assert params[cp.MAX_PRICE_ALPHA_PARAM] == pytest.approx(0.5)              # 2 USD / 4 USD
    assert params[cp.MAX_PRICE_INPUT_ALPHA_PARAM] == pytest.approx(0.125)      # 0.5 USD / 4 USD
    assert cp.max_price(params) == pytest.approx(0.5e-6 / emission)
    assert cp.max_input_price(params) == pytest.approx(0.125e-6 / emission)
    assert params[cp.MAX_PRICE_ALPHA_USD_PARAM] == pytest.approx(4.0)
    assert params[cp.MAX_PRICE_TAO_USD_PARAM] == TAO_USD
    price = created.json()["max_price"]
    assert price["usd"] == {"input_per_mtoken": 0.5, "output_per_mtoken": 2.0,
                           "alpha_usd": pytest.approx(4.0), "tao_usd": TAO_USD}
    status = admin("GET", "/admin/v1/jobs/order-eval-p1/status").json()
    assert status["tasks"][0]["max_price"]["max_price_alpha_per_mtoken"] == pytest.approx(0.5)


def test_a_replay_keeps_the_first_price_even_when_the_rate_moved_or_is_down(
        admin, pins, rates):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING)).status_code == 201
    rates["alpha_tao"] = 0.02
    replay = admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING))
    assert replay.status_code == 200, replay.text
    assert replay.json()["max_price"]["max_price_alpha_per_mtoken"] == pytest.approx(0.5)
    rates["down"] = True
    assert admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING)).status_code == 200


def test_a_free_input_prices_output_tokens_alone(admin, pins, rates):  # noqa: F811
    created = admin("POST", "/admin/v1/jobs",
                    _order(max_price_usd_per_mtoken={"input": 0, "output": 2.0}))
    assert created.status_code == 201, created.text
    params = admin.registry["entries"]["order-eval-p1"].params
    assert cp.max_input_price(params) is None and cp.max_price(params) is not None


def test_no_rate_no_task(admin, pins, rates, monkeypatch):  # noqa: F811
    rates["down"] = True
    refused = admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING))
    assert refused.status_code == 503 and "token_price_unavailable" in refused.text
    assert "order-eval-p1" not in admin.registry["entries"]
    # Once the rate is back, the same create declares it.
    rates["down"] = False
    assert admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING)).status_code == 201


def test_an_order_without_a_ceiling_reads_no_rate(admin, pins, rates):  # noqa: F811
    created = admin("POST", "/admin/v1/jobs", _order())
    assert created.status_code == 201 and "max_price" not in created.json()
    assert rates["reads"] == 0
    assert cp.max_price(admin.registry["entries"]["order-eval-p1"].params) is None


@pytest.mark.parametrize("ceiling", [{"input": -1, "output": 2.0}, {"input": 0.5, "output": 0},
                                     {"input": 0.5}, {"input": 0.5, "output": 2.0, "cached": 1}])
def test_a_malformed_ceiling_is_refused(admin, pins, rates, ceiling):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs",
                 _order(max_price_usd_per_mtoken=ceiling)).status_code == 422


def test_the_capability_documents_announce_token_ceilings(admin):  # noqa: F811
    assert admin("GET", "/admin/v1/architectures").json()["orders"]["token_ceilings"] is True
    assert admin("GET", "/admin/v1/task-catalog").json()["orders"]["token_ceilings"] is True


def test_tao_usd_is_fetched_once_per_cache_window(monkeypatch):
    monkeypatch.delenv(service.TAO_USD_ENV, raising=False)
    now, fetched = [0.0], []

    async def fetch():
        fetched.append(now[0])
        return 300.0 + len(fetched)

    read = service.tao_usd_reader(fetch=fetch, clock=lambda: now[0], ttl=300)
    assert asyncio.run(read()) == 301.0
    now[0] = 299.0
    assert asyncio.run(read()) == 301.0
    now[0] = 301.0
    assert asyncio.run(read()) == 302.0
    monkeypatch.setenv(service.TAO_USD_ENV, "250")
    assert asyncio.run(read()) == 250.0 and len(fetched) == 2
    monkeypatch.setenv(service.TAO_USD_ENV, "nan")
    with pytest.raises(ValueError):
        asyncio.run(read())


def test_an_order_is_priced_by_auction_when_the_admin_says_so(admin, pins, rates, monkeypatch):  # noqa: F811
    from reliquary.corpus.token_auction import auction_params

    monkeypatch.setenv(service.ORDER_PRICING_ENV, "auction")
    created = admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING))
    assert created.status_code == 201, created.text
    declared = auction_params(admin.registry["entries"]["order-eval-p1"].params)
    assert declared is not None and declared.alpha_of(declared.ceiling) == pytest.approx(0.5)


def test_an_order_without_a_ceiling_is_never_priced_by_auction(admin, pins, rates, monkeypatch):  # noqa: F811
    monkeypatch.setenv(service.ORDER_PRICING_ENV, "auction")
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    assert "pricing" not in admin.registry["entries"]["order-eval-p1"].params


def test_orders_are_first_come_unless_the_admin_says_auction(admin, pins, rates, monkeypatch):  # noqa: F811
    monkeypatch.delenv(service.ORDER_PRICING_ENV, raising=False)
    assert admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING)).status_code == 201
    assert "pricing" not in admin.registry["entries"]["order-eval-p1"].params


def test_a_misspelt_order_pricing_refuses_the_order(admin, pins, rates, monkeypatch):  # noqa: F811
    monkeypatch.setenv(service.ORDER_PRICING_ENV, "auctoin")
    refused = admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING))
    assert refused.status_code == 503 and "order_pricing_misconfigured" in refused.text
