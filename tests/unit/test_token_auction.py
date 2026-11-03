"""The token price auction of a draw job: the round price follows the asks, the cheapest miners keep the work,
the leases follow the audit queue, and a lease that ends unused is paid back."""
from __future__ import annotations

import pytest

from reliquary.corpus.token_auction import (
    ASK_ABOVE_PRICE,
    ASK_TTL_SECONDS,
    INITIAL_LEASES_PARAM,
    NO_ASK,
    PRICING_PARAM,
    ROUND_FULL,
    ROUND_SECONDS_PARAM,
    AuctionParams,
    Terms,
    TokenAuction,
    auction_params,
    job_auction,
)
from reliquary.validator.corpus_periods import price_params

EMISSION = 100.0                     # miner alpha per period
ROUND = 600.0


class Clock:
    def __init__(self, now=6000.0):
        self.now = now

    def __call__(self):
        return self.now


def _auction(ceiling_alpha=0.5, leases=4, backlog=None, clock=None):
    params = AuctionParams(ceiling=ceiling_alpha / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                           round_seconds=ROUND, initial_leases=leases)
    clock = clock or Clock()
    return TokenAuction(params, clock=clock, backlog=backlog, mean_tokens=1000), clock


def _alpha(auction, terms):
    return auction.params.alpha_of(terms.price)


def _next_round(clock):
    clock.now += ROUND


def test_a_hotkey_without_an_ask_gets_no_lease():
    auction, _ = _auction()
    assert auction.admit("hk") == NO_ASK


def test_the_round_price_is_the_highest_ask_under_the_target():
    auction, clock = _auction(ceiling_alpha=0.5)
    for hotkey, ask in (("a", 0.15), ("b", 0.20), ("c", 0.30), ("d", 0.45), ("e", 0.70)):
        assert auction.post_ask(hotkey, ask) is None
    _next_round(clock)
    terms = auction.admit("a")
    assert isinstance(terms, Terms)
    # Not the ceiling: the most expensive admitted miner asked 0.45.
    assert _alpha(auction, terms) == pytest.approx(0.45)
    # Over the ceiling: never served.
    assert auction.admit("e") == ASK_ABOVE_PRICE


def test_a_saturated_round_drops_the_price_to_the_next_ask():
    auction, clock = _auction(leases=2)
    for hotkey, ask in (("a", 0.15), ("b", 0.20), ("d", 0.45)):
        auction.post_ask(hotkey, ask)
    _next_round(clock)
    assert isinstance(auction.admit("a"), Terms)
    assert isinstance(auction.admit("b"), Terms)
    assert auction.admit("d") == ROUND_FULL           # admitted miners wanted more than the round's leases
    _next_round(clock)
    assert _alpha(auction, auction.admit("a")) == pytest.approx(0.20)
    assert auction.admit("d") == ASK_ABOVE_PRICE      # the most expensive miner is out


def test_leftover_leases_with_a_miner_priced_out_raise_the_price():
    auction, clock = _auction(leases=2)
    for hotkey, ask in (("a", 0.15), ("b", 0.20)):
        auction.post_ask(hotkey, ask)
    _next_round(clock)
    auction.admit("a"); auction.admit("b"); auction.admit("a")   # saturated: down to 0.15
    _next_round(clock)
    assert auction.admit("b") == ASK_ABOVE_PRICE
    _next_round(clock)                                           # a alone left leases: b is needed again
    assert _alpha(auction, auction.admit("b")) == pytest.approx(0.20)


def test_a_quiet_round_holds_the_price():
    auction, clock = _auction(leases=8)
    for hotkey, ask in (("a", 0.15), ("b", 0.20)):
        auction.post_ask(hotkey, ask)
    _next_round(clock)
    price = _alpha(auction, auction.admit("a"))
    _next_round(clock)
    assert _alpha(auction, auction.admit("a")) == pytest.approx(price)


def test_a_lower_ask_mid_round_is_admitted_at_the_round_price():
    auction, clock = _auction()
    auction.post_ask("a", 0.30)
    _next_round(clock)
    auction.post_ask("b", 0.10)
    assert _alpha(auction, auction.admit("b")) == pytest.approx(0.30)


def test_a_reservation_without_a_slot_is_given_back():
    auction, clock = _auction(leases=1)
    auction.post_ask("a", 0.1)
    _next_round(clock)
    terms = auction.admit("a")
    auction.unreserve(terms)
    assert isinstance(auction.admit("a"), Terms)


def test_an_ask_expires_unless_reposted():
    auction, clock = _auction()
    auction.post_ask("a", 0.1)
    clock.now += ASK_TTL_SECONDS + 1
    assert auction.admit("a") == NO_ASK


@pytest.mark.parametrize("ask", [0, -1, float("nan"), float("inf"), "x", 2e9])
def test_an_ask_must_be_a_positive_number(ask):
    auction, _ = _auction()
    assert auction.post_ask("a", ask) == "invalid_ask"


def test_the_leases_follow_the_audit_queue():
    queue = {"waiting": 0}
    auction, clock = _auction(leases=10, backlog=lambda: queue["waiting"])
    auction.post_ask("a", 0.1)
    _next_round(clock)
    for _ in range(11):
        auction.admit("a")                       # saturated, queue empty: grow
    _next_round(clock)
    assert auction.view()["leases"] == 12
    queue["waiting"] = 50                        # the card falls behind: cut
    _next_round(clock)
    assert auction.view()["leases"] == 8


def test_without_an_audit_queue_the_leases_hold():
    auction, clock = _auction(leases=3)
    auction.post_ask("a", 0.1)
    _next_round(clock)
    for _ in range(5):
        auction.admit("a")
    _next_round(clock)
    assert auction.view()["leases"] == 3


def test_an_unused_lease_is_owed_back_from_the_next_submissions():
    auction, _ = _auction()
    price = auction.params.share_of(0.2)
    auction.lease_ended("a", price, 1.0)                       # expired: 1000 tokens at 0.2
    owed = price * 1000
    assert auction.debt_alpha("a") == pytest.approx(owed * EMISSION)
    first = auction.take_debit("a", price * 600)
    assert first == pytest.approx(price * 600)
    second = auction.take_debit("a", price * 600)
    assert second == pytest.approx(price * 400)
    assert auction.take_debit("a", price * 600) == 0
    auction.refund_debit("a", second)                          # its submission was not accepted
    assert auction.take_debit("a", 1.0) == pytest.approx(second)


def test_a_release_owes_the_share_of_the_lease_time_held():
    auction, _ = _auction()
    price = auction.params.share_of(0.2)
    auction.lease_ended("a", price, 0.25)
    assert auction.take_debit("a", 1.0) == pytest.approx(price * 250)


def test_the_public_view_names_no_hotkey_and_no_ask():
    auction, clock = _auction()
    auction.post_ask("5Fsecret", 0.123)
    _next_round(clock)
    auction.admit("5Fsecret")
    _next_round(clock)
    view = auction.view()
    assert "5Fsecret" not in repr(view)
    assert view["rounds"][-1]["granted"] == 1
    assert view["ceiling_alpha_per_mtoken"] == pytest.approx(0.5)


def _params(**extra):
    return {"settlement": "period-ema-v1", **price_params(0.5, EMISSION, 1.0), PRICING_PARAM: "auction", **extra}


def test_auction_params_come_from_the_registry_params():
    declared = auction_params(_params(**{ROUND_SECONDS_PARAM: 300, INITIAL_LEASES_PARAM: 32}))
    assert declared.round_seconds == 300 and declared.initial_leases == 32
    assert declared.alpha_of(declared.ceiling) == pytest.approx(0.5)
    assert auction_params({"settlement": "period-ema-v1"}) is None


@pytest.mark.parametrize("bad", [
    {PRICING_PARAM: "dutch"},
    {ROUND_SECONDS_PARAM: 10},
    {INITIAL_LEASES_PARAM: 1.5},
])
def test_a_bad_auction_declaration_is_refused(bad):
    with pytest.raises(ValueError):
        auction_params(_params(**bad))


def test_an_auction_needs_a_price_ceiling():
    with pytest.raises(ValueError):
        auction_params({PRICING_PARAM: "auction"})
    assert job_auction({PRICING_PARAM: "auction"}) is None


# ------------------------------------------------------------------ the registry


def test_a_task_is_priced_by_auction_only_under_a_price_ceiling():
    from reliquary.shared.task_registry import RegistryError, set_cap
    from tests.unit.test_corpus_token_price_ceiling import _entry

    entries = {"corpus-x": _entry()}
    with pytest.raises(RegistryError, match="ceiling"):
        set_cap(entries, "corpus-x", None, pricing={PRICING_PARAM: "auction"})
    priced = set_cap(entries, "corpus-x", None, token_price=price_params(0.5, EMISSION, 1.0),
                     pricing={PRICING_PARAM: "auction", ROUND_SECONDS_PARAM: 300.0})
    assert auction_params(priced["corpus-x"].params).round_seconds == 300
    # Kept through a cap change, removed on request.
    assert set_cap(priced, "corpus-x", 0.2)["corpus-x"].params[PRICING_PARAM] == "auction"
    cleared = set_cap(priced, "corpus-x", None, pricing=None)
    assert auction_params(cleared["corpus-x"].params) is None
    assert ROUND_SECONDS_PARAM not in cleared["corpus-x"].params


def test_auction_settings_without_pricing_are_refused():
    from reliquary.shared.task_registry import RegistryError, validate_entry
    from tests.unit.test_corpus_token_price_ceiling import _entry

    with pytest.raises(RegistryError, match="pricing"):
        validate_entry(_entry(**price_params(0.5, EMISSION, 1.0), **{ROUND_SECONDS_PARAM: 300.0}))
