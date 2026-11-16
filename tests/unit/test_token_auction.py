"""The price auction of a draw job: offers are served cheapest first up to their capacity, the leases out at once
follow the audit queue, a missed lease is charged and halves its holder's capacity, and the state survives a
restart."""
from __future__ import annotations

import pytest

from reliquary.corpus.token_auction import (
    ACTIVE_SECONDS,
    ALLOWANCE_FULL,
    ASK_ABOVE_PRICE,
    ASK_TTL_SECONDS,
    BACKLOG_HIGH_PARAM,
    BACKLOG_LOW_PARAM,
    INITIAL_LEASES_PARAM,
    LEASE_FLOOR_SECONDS,
    NO_ASK,
    PENALTY_SECONDS,
    PRICE_HALF_LIFE_SECONDS,
    PRICE_HORIZON_SECONDS,
    PRICING_PARAM,
    STEER_SECONDS,
    AuctionParams,
    Terms,
    TokenAuction,
    auction_params,
    job_auction,
    market_key,
)
from reliquary.validator.corpus_periods import price_params

EMISSION = 100.0                     # miner alpha per period


class Clock:
    def __init__(self, now=6000.0):
        self.now = now

    def __call__(self):
        return self.now


class Book:
    """Live leases per hotkey, as ``DrawLeases.counts`` reports them."""

    def __init__(self):
        self.held = {}

    def __call__(self):
        return dict(self.held)

    def take(self, terms):
        self.held[terms.hotkey] = self.held.get(terms.hotkey, 0) + 1


def _auction(ceiling_alpha=0.5, budget=8, backlog=None, clock=None, state_path=None, low=8, high=32,
             market_name=None, job_id="job"):
    params = AuctionParams(ceiling=ceiling_alpha / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                           initial_leases=budget, backlog_low=low, backlog_high=high)
    clock = clock or Clock()
    book = Book()
    auction = TokenAuction(params, clock=clock, backlog=backlog, holdings=book, mean_tokens=1000,
                           state_path=state_path, market_name=market_name, job_id=job_id)
    return auction, clock, book


def _lease(auction, book, hotkey):
    """A lease as the route grants it: reserve, count as held, give the reservation back."""
    terms = auction.admit(hotkey)
    if isinstance(terms, Terms):
        book.take(terms)
        auction.unreserve(terms)
    return terms


def _deliver(auction, hotkey, count, alpha=None):
    """``count`` leases of ``hotkey`` delivered, each won at ``alpha`` (default: its live offer)."""
    offer = auction.ask_of(hotkey) if alpha is None else auction.params.share_of(alpha)
    for _ in range(count):
        auction.observe_delivery(hotkey, 60.0, offer=offer)


def _active(auction, *hotkeys):
    for hotkey in hotkeys:
        auction.admit(hotkey)                      # each asks once: all are active
        auction._reserved.clear()


def _fill(auction, book, hotkey):
    """Lease until refused; the leases granted and the refusal."""
    granted = 0
    while True:
        terms = _lease(auction, book, hotkey)
        if not isinstance(terms, Terms):
            return granted, terms
        granted += 1


def test_a_hotkey_without_an_ask_gets_no_lease():
    auction, _, _ = _auction()
    assert auction.admit("5A") == NO_ASK


def test_a_lease_alone_in_the_budget_carries_its_holders_own_offer():
    auction, _, book = _auction()
    auction.post_ask("5A", 0.3, 4)
    terms = _lease(auction, book, "5A")
    assert auction.params.alpha_of(terms.price) == pytest.approx(0.3)


def test_a_lease_carries_the_price_of_the_moment_never_under_its_own_offer():
    """2026-11-12: an offer at 0 took a small budget's leases and was paid the period's price. A lease is paid the
    price of the delivered volume when it is granted, never above the dearest offer getting work: alone in a small
    budget, an offer at 0 gets 0."""
    auction, _, book = _auction(budget=4)
    auction.post_ask("zero", 1e-6, 4)            # the lowest a miner can offer
    auction.post_ask("mid", 0.2, 4)
    auction.post_ask("dear", 0.5, 4)
    _active(auction, "zero", "mid", "dear")
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)   # nothing delivered
    _deliver(auction, "zero", 6)
    _deliver(auction, "mid", 24)                   # most of the work was won at 0.2
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)   # zero alone works
    auction._budget = 8                            # mid works too: the moment's price is mid's
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(0.2)
    auction._budget = 12                           # dear works too, but has delivered nothing: still mid's
    assert auction.params.alpha_of(_lease(auction, book, "mid").price) == pytest.approx(0.2)
    assert auction.params.alpha_of(_lease(auction, book, "dear").price) == pytest.approx(0.5)
    _deliver(auction, "dear", 8)                   # over a fifth of the work at 0.5: the price for all
    assert auction.params.alpha_of(_lease(auction, book, "mid").price) == pytest.approx(0.5)


def test_an_offer_under_its_real_cost_only_wins_leases_that_pay_under_that_cost():
    """A miner whose card costs 0.3 against one at 0.15: offering ~0 instead of 0.3 wins it more leases only when the
    budget is short, and those are paid the rival's 0.15; with room for both, the same leases at the same price."""
    def grant(own, budget):
        auction, _, book = _auction(budget=budget)
        auction.post_ask("a", own, 6)
        auction.post_ask("b", 0.15, 6)
        auction.post_ask("c", 0.4, 6)
        _active(auction, "a", "b", "c")
        for hotkey in ("a", "b", "c"):
            _deliver(auction, hotkey, 7)
        prices = []
        while isinstance(terms := _lease(auction, book, "a"), Terms):
            prices.append(auction.params.alpha_of(terms.price))
        return prices

    honest, low = grant(0.3, 18), grant(1e-6, 18)          # room for all: same leases, same price
    assert honest == pytest.approx(low) and len(honest) == 6 and honest == pytest.approx([0.4] * 6)
    honest, low = grant(0.3, 8), grant(1e-6, 8)            # short: a lies its way ahead of b
    assert len(honest) == 2 and all(p == pytest.approx(0.3) for p in honest)
    assert len(low) == 6 and all(p < 0.3 for p in low)    # every extra lease pays under a's cost


def test_the_cheapest_offers_get_the_budget_up_to_their_capacity():
    auction, _, book = _auction(budget=10)
    auction.post_ask("cheap", 0.2, 6)
    auction.post_ask("mid", 0.3, 6)
    auction.post_ask("dear", 0.4, 6)
    for hotkey in ("cheap", "mid", "dear"):
        auction.admit(hotkey)                      # each asks once: all three are active
        auction._reserved.clear()
    assert _fill(auction, book, "cheap") == (6, ALLOWANCE_FULL)
    assert _fill(auction, book, "mid") == (4, ALLOWANCE_FULL)
    assert _fill(auction, book, "dear") == (0, ASK_ABOVE_PRICE)
    assert auction.view()["price_alpha_per_mtoken"] == pytest.approx(0.3)


def test_a_cheaper_offer_takes_the_budget_from_a_dearer_one():
    auction, _, book = _auction(budget=4)
    auction.post_ask("dear", 0.4, 4)
    assert _fill(auction, book, "dear") == (4, ALLOWANCE_FULL)
    auction.post_ask("cheap", 0.2, 4)
    assert _fill(auction, book, "cheap") == (0, ALLOWANCE_FULL)    # the budget is held by dear's leases
    book.held["dear"] = 0                                          # dear delivers
    assert _fill(auction, book, "dear") == (0, ASK_ABOVE_PRICE)
    assert _fill(auction, book, "cheap") == (4, ALLOWANCE_FULL)


def test_an_offer_over_the_ceiling_never_works():
    auction, _, book = _auction(ceiling_alpha=0.5)
    auction.post_ask("5A", 0.9, 4)
    assert _lease(auction, book, "5A") == ASK_ABOVE_PRICE
    assert auction.view()["price_alpha_per_mtoken"] is None


def test_an_idle_offer_holds_no_budget():
    auction, clock, book = _auction(budget=4)
    auction.post_ask("idle", 0.1, 4)
    auction.admit("idle")
    auction._reserved.clear()
    auction.post_ask("busy", 0.3, 4)
    assert _fill(auction, book, "busy") == (0, ASK_ABOVE_PRICE)
    clock.now += ACTIVE_SECONDS + 1
    assert _fill(auction, book, "busy") == (4, ALLOWANCE_FULL)


def test_an_offer_expires_unless_reposted():
    auction, clock, _ = _auction()
    auction.post_ask("5A", 0.3, 1)
    clock.now += ASK_TTL_SECONDS + 1
    assert auction.admit("5A") == NO_ASK


@pytest.mark.parametrize("price,capacity", [(0, 1), (-1, 1), (float("nan"), 1), ("x", 1), (0.3, 0), (0.3, 10**6)])
def test_an_offer_must_be_a_positive_price_and_capacity(price, capacity):
    auction, _, _ = _auction()
    assert auction.post_ask("5A", price, capacity) == "invalid_ask"


def test_without_a_capacity_an_offer_holds_one_lease():
    auction, _, book = _auction()
    auction.post_ask("5A", 0.3)
    assert _fill(auction, book, "5A") == (1, ALLOWANCE_FULL)


def test_a_reservation_counts_until_given_back():
    auction, _, _ = _auction()
    auction.post_ask("5A", 0.3, 1)
    terms = auction.admit("5A")
    assert auction.admit("5A") == ALLOWANCE_FULL
    auction.unreserve(terms)
    assert isinstance(auction.admit("5A"), Terms)


# ------------------------------------------------------------------ the budget follows the audit queue


def test_a_long_audit_queue_shrinks_the_budget():
    queue = [100]
    auction, clock, _ = _auction(budget=50, backlog=lambda: queue[0])
    auction.post_ask("5A", 0.3, 1)
    auction.admit("5A")
    assert auction.view()["budget"] == 40
    clock.now += STEER_SECONDS
    auction.view()
    assert auction.view()["budget"] == 32


def test_a_short_queue_grows_the_budget_only_while_it_is_used():
    queue = [0]
    auction, clock, book = _auction(budget=8, backlog=lambda: queue[0])
    auction.view()
    assert auction.view()["budget"] == 8                # nothing leased: no reason to grow
    clock.now += STEER_SECONDS
    book.held["5A"] = 8
    assert auction.view()["budget"] == 10


def test_a_queue_between_the_marks_holds_the_budget():
    auction, clock, book = _auction(budget=8, backlog=lambda: 16)
    book.held["5A"] = 8
    auction.view()
    clock.now += STEER_SECONDS
    assert auction.view()["budget"] == 8


def test_without_an_audit_queue_the_budget_holds():
    auction, clock, book = _auction(budget=8)
    book.held["5A"] = 8
    clock.now += STEER_SECONDS
    assert auction.view()["budget"] == 8


# ------------------------------------------------------------------ missed leases


def test_a_missed_lease_is_owed_back_from_the_next_submissions():
    auction, _, _ = _auction()
    price = auction.params.share_of(0.2)
    auction.lease_ended("5A", price, 1.0)
    owed = price * 1000
    assert auction.debt_alpha("5A") == pytest.approx(owed * EMISSION)
    assert auction.take_debit("5A", owed / 2) == pytest.approx(owed / 2)
    assert auction.take_debit("5A", owed) == pytest.approx(owed / 2)
    assert auction.take_debit("5A", owed) == 0


def test_an_early_release_owes_its_share_and_keeps_the_capacity():
    auction, _, book = _auction()
    auction.post_ask("5A", 0.2, 4)
    auction.lease_ended("5A", auction.params.share_of(0.2), 0.25)
    assert auction.debt_alpha("5A") == pytest.approx(0.2 / 1e6 * 1000 * 0.25)
    auction.take_debit("5A", 1.0)
    assert _fill(auction, book, "5A") == (4, ALLOWANCE_FULL)


def test_a_missed_lease_halves_the_capacity_for_a_while():
    auction, clock, book = _auction(budget=100)
    auction.post_ask("5A", 0.2, 8)
    auction.lease_ended("5A", auction.params.share_of(0.2), 1.0)
    auction.take_debit("5A", 1.0)                       # paid back
    assert _fill(auction, book, "5A") == (4, ALLOWANCE_FULL)
    book.held.clear()
    clock.now += PENALTY_SECONDS + 1
    auction.post_ask("5A", 0.2, 8)
    assert _fill(auction, book, "5A") == (8, ALLOWANCE_FULL)


def test_a_hotkey_in_debt_holds_one_lease():
    auction, _, book = _auction(budget=100)
    auction.post_ask("5A", 0.2, 8)
    auction.lease_ended("5A", auction.params.share_of(0.2), 0.1)
    assert _fill(auction, book, "5A") == (1, ALLOWANCE_FULL)


# ------------------------------------------------------------------ lease time


def test_the_lease_time_follows_the_hotkeys_deliveries():
    auction, _, _ = _auction()
    assert auction.lease_seconds("5A", 3600.0) == 3600.0
    for seconds in (400, 500, 600):
        auction.observe_delivery("5A", seconds)
    assert auction.lease_seconds("5A", 3600.0) == pytest.approx(1200.0)     # twice its slow delivery
    for _ in range(20):
        auction.observe_delivery("fast", 10)
    assert auction.lease_seconds("fast", 3600.0) == LEASE_FLOOR_SECONDS
    for _ in range(3):
        auction.observe_delivery("slow", 5000)
    assert auction.lease_seconds("slow", 3600.0) == 3600.0


def test_a_batch_miners_slow_deliveries_set_its_lease_not_its_median():
    # 2026-11-11: a miner delivering in 50-280 s had 300 s leases (twice its 101 s median) and lost whole batches.
    auction, _, _ = _auction()
    for seconds in (50, 60, 70, 80, 90, 100, 101, 110, 120, 130, 140, 147, 150, 160, 170, 180, 200, 230, 260, 278):
        auction.observe_delivery("batch", seconds)
    assert auction.lease_seconds("batch", 3600.0) == pytest.approx(2 * 230.0)     # its 18th of 20


def test_a_batch_still_out_keeps_the_default_lease_time():
    # 2026-11-15: 3 of 188 answers back in 23-87 s cut the next 466 leases to 300 s; the batch took 15-34 min.
    auction, _, _ = _auction()
    for seconds in (23, 25, 87):
        auction.observe_delivery("big", seconds)
    assert auction.lease_seconds("big", 3600.0) == LEASE_FLOOR_SECONDS
    assert auction.lease_seconds("big", 3600.0, held=185) == 3600.0
    # Most of the batch delivered: its tail is observed.
    for _ in range(200):
        auction.observe_delivery("big", 900)
    assert auction.lease_seconds("big", 3600.0, held=10) == pytest.approx(1800.0)


# ------------------------------------------------------------------ views and state


def test_the_public_view_names_no_other_hotkey():
    auction, _, book = _auction()
    auction.post_ask("5Fsecret", 0.123, 2)
    _lease(auction, book, "5Fsecret")
    view = auction.view()
    assert "5Fsecret" not in repr(view)
    assert view["leased"] == 1 and view["miners_working"] == 1
    assert view["ceiling_alpha_per_mtoken"] == pytest.approx(0.5)
    own = auction.view("5Fsecret")
    assert (own["allowance"], own["held"]) == (2, 1)


def test_the_state_survives_a_restart(tmp_path):
    path = tmp_path / "auction" / "job.json"
    auction, clock, _ = _auction(state_path=path)
    auction.post_ask("5A", 0.2, 8)
    auction.lease_ended("5A", auction.params.share_of(0.2), 1.0)
    for seconds in (400, 500, 600):
        auction.observe_delivery("5A", seconds)
    for _ in range(20):
        auction.observe_delivery("5B", 60, offer=auction.params.share_of(0.3))
    auction.save(force=True)
    again, _, book = _auction(state_path=path, clock=clock)
    assert again._market.of("job") == auction._market.of("job")
    assert again._volume_price(clock.now) == pytest.approx(auction.params.share_of(0.3))
    assert again.debt_alpha("5A") == pytest.approx(auction.debt_alpha("5A"))
    assert again.ask_of("5A") == pytest.approx(auction.ask_of("5A"))
    assert again.lease_seconds("5A", 3600.0) == pytest.approx(1200.0)
    again.take_debit("5A", 1.0)
    assert _fill(again, book, "5A") == (4, ALLOWANCE_FULL)      # still halved


def test_an_unreadable_state_starts_empty(tmp_path):
    path = tmp_path / "job.json"
    path.write_text("{not json")
    auction, _, _ = _auction(state_path=path)
    assert auction.admit("5A") == NO_ASK


# ------------------------------------------------------------------ declaration


def _params(**extra):
    return {"settlement": "period-ema-v1", **price_params(0.5, EMISSION, 1.0), PRICING_PARAM: "auction", **extra}


def test_auction_params_come_from_the_registry_params():
    declared = auction_params(_params(**{INITIAL_LEASES_PARAM: 32, BACKLOG_LOW_PARAM: 4, BACKLOG_HIGH_PARAM: 12}))
    assert (declared.initial_leases, declared.backlog_low, declared.backlog_high) == (32, 4, 12)
    assert declared.alpha_of(declared.ceiling) == pytest.approx(0.5)
    assert auction_params({"settlement": "period-ema-v1"}) is None


@pytest.mark.parametrize("bad", [
    {PRICING_PARAM: "dutch"},
    {INITIAL_LEASES_PARAM: 1.5},
    {INITIAL_LEASES_PARAM: 1},
    {BACKLOG_LOW_PARAM: 10, BACKLOG_HIGH_PARAM: 10},
])
def test_a_bad_auction_declaration_is_refused(bad):
    with pytest.raises(ValueError):
        auction_params(_params(**bad))


def test_an_auction_needs_a_price_ceiling():
    with pytest.raises(ValueError):
        auction_params({PRICING_PARAM: "auction"})
    assert job_auction({PRICING_PARAM: "auction"}) is None


def test_the_state_file_follows_the_environment(monkeypatch, tmp_path):
    from reliquary.corpus.token_auction import STATE_DIR_ENV, state_path_for

    monkeypatch.delenv(STATE_DIR_ENV, raising=False)
    assert state_path_for("order-eval-1") is None
    monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path))
    assert state_path_for("order/../x") == tmp_path / "order_.._x.json"


# ------------------------------------------------------------------ the registry


def test_a_task_is_priced_by_auction_only_under_a_price_ceiling():
    from reliquary.shared.task_registry import RegistryError, set_cap
    from tests.unit.test_corpus_token_price_ceiling import _entry

    entries = {"corpus-x": _entry()}
    with pytest.raises(RegistryError, match="ceiling"):
        set_cap(entries, "corpus-x", None, pricing={PRICING_PARAM: "auction"})
    priced = set_cap(entries, "corpus-x", None, token_price=price_params(0.5, EMISSION, 1.0),
                     pricing={PRICING_PARAM: "auction", BACKLOG_HIGH_PARAM: 40})
    assert auction_params(priced["corpus-x"].params).backlog_high == 40
    # Kept through a cap change, removed on request.
    assert set_cap(priced, "corpus-x", 0.2)["corpus-x"].params[PRICING_PARAM] == "auction"
    cleared = set_cap(priced, "corpus-x", None, pricing=None)
    assert auction_params(cleared["corpus-x"].params) is None
    assert BACKLOG_HIGH_PARAM not in cleared["corpus-x"].params


def test_auction_settings_without_pricing_are_refused():
    from reliquary.shared.task_registry import RegistryError, validate_entry
    from tests.unit.test_corpus_token_price_ceiling import _entry

    with pytest.raises(RegistryError, match="pricing"):
        validate_entry(_entry(**price_params(0.5, EMISSION, 1.0), **{BACKLOG_HIGH_PARAM: 40}))


def test_an_expiry_charged_while_a_lease_is_admitted_never_deadlocks():
    """The leases charge the auction under their lock; the auction reads the leases before taking its own."""
    import threading

    from reliquary.corpus.draw_leases import DrawLeases
    from tests.unit.test_draw_leases import _job

    clock = Clock()
    auction, _, _ = _auction(clock=clock, budget=1000)
    leases = DrawLeases(_job(prompts=50), clock=clock, limit_per_hotkey=False,
                        on_unused=lambda lease, held: auction.lease_ended(lease.hotkey, lease.price, held))
    auction.bind_holdings(leases.counts)
    auction.post_ask("5A", 0.2, 1000)

    def churn():
        for _ in range(300):
            leases.grant("5B", lambda start: ((p, 0) for p in range(50)), price=1e-9, seconds=1.0)
            clock.now += 2.0
            leases.expire()

    worker = threading.Thread(target=churn)
    worker.start()
    for _ in range(300):
        terms = auction.admit("5A")
        if isinstance(terms, Terms):
            auction.unreserve(terms)
    worker.join(timeout=20)
    assert not worker.is_alive()



def test_unmet_demand_grows_the_budget_even_while_a_cheaper_offer_leaves_it_unused():
    """2026-11-13: one slow offer held a 15-lease budget at 4 leases; the thermostat waited for 80 % of the budget to
    be leased, so it never grew and the offers behind got nothing. A refused offer now counts as demand."""
    backlog = [0]
    auction, clock, book = _auction(budget=4, backlog=lambda: backlog[0])
    auction.post_ask("slow", 0.1, 4)
    auction.post_ask("next", 0.2, 4)
    _lease(auction, book, "slow")                           # 1 of 4 leased: far under 80 %
    assert auction.admit("next") == ASK_ABOVE_PRICE         # no room behind the slow offer: unmet demand
    clock.now += STEER_SECONDS + 1
    _lease(auction, book, "slow")                           # the next request steers: the budget grows
    assert auction.view()["budget"] > 4
    assert isinstance(_lease(auction, book, "next"), Terms)


def test_without_unmet_demand_an_unused_budget_does_not_grow():
    backlog = [0]
    auction, clock, book = _auction(budget=4, backlog=lambda: backlog[0])
    auction.post_ask("slow", 0.1, 4)
    _lease(auction, book, "slow")
    clock.now += STEER_SECONDS + 1
    _lease(auction, book, "slow")
    assert auction.view()["budget"] == 4


def test_a_hotkey_set_aside_gets_no_allowance_until_then():
    auction, clock, book = _auction(budget=4)
    auction.post_ask("aside", 0.1, 4)
    auction.post_ask("next", 0.2, 4)
    for hotkey in ("aside", "next"):
        auction.admit(hotkey)
        auction._reserved.clear()
    auction.set_aside("aside", clock.now + 100)
    assert auction.admit("aside") == ASK_ABOVE_PRICE
    assert _fill(auction, book, "next")[0] == 4             # its share goes to the offer behind it
    clock.now += 101
    book.held.clear()
    assert isinstance(_lease(auction, book, "aside"), Terms)


# -- the price of the moment follows the delivered volume (2026-11-16) ----------------------------------------------

def _two_keys(budget=100, honest=15):
    """One operator offers ~0 on ``zero`` for the room the honest offers leave but one lease, and the ceiling on
    ``dear`` for one lease; ``mid`` stands for the honest offers at 0.3."""
    auction, clock, book = _auction(ceiling_alpha=0.5, budget=budget)
    auction.post_ask("zero", 1e-6, budget - honest - 1)
    auction.post_ask("mid", 0.3, honest)
    auction.post_ask("dear", 0.5, 1)
    _active(auction, "zero", "mid", "dear")
    return auction, clock, book


def test_a_second_key_at_the_ceiling_for_one_lease_no_longer_prices_every_lease():
    auction, _, book = _two_keys()
    assert _fill(auction, book, "dear") == (1, ALLOWANCE_FULL)      # dear is given work: the old moment's price
    assert auction.view()["price_alpha_per_mtoken"] == pytest.approx(0.5)    # nothing delivered: at most that
    _deliver(auction, "zero", 84)
    _deliver(auction, "mid", 15)
    _deliver(auction, "dear", 1)                   # it even delivers its one lease
    terms = _lease(auction, book, "zero")
    assert auction.params.alpha_of(terms.price) == pytest.approx(1e-6)       # was the ceiling, 0.5
    assert auction.params.alpha_of(_lease(auction, book, "mid").price) == pytest.approx(0.3)
    assert auction.view()["price_alpha_per_mtoken"] == pytest.approx(1e-6)


def test_taking_more_leases_at_zero_lowers_the_price_it_is_paid():
    auction, _, book = _two_keys()
    _deliver(auction, "mid", 15)
    _deliver(auction, "dear", 21)
    _deliver(auction, "zero", 64)
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(0.5)
    _deliver(auction, "zero", 20)                  # more of its own work at 0: under a fifth at the ceiling
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(0.3)


def test_leases_not_delivered_set_no_price():
    """A second key that takes leases and gives them back, or lets them expire, is in debt: it counts for nothing."""
    auction, _, book = _two_keys()
    _deliver(auction, "zero", 64)
    _deliver(auction, "mid", 15)
    _deliver(auction, "dear", 21)
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(0.5)
    auction.lease_ended("dear", auction.params.share_of(0.5), 0.01)          # one lease given back at once
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)


def test_a_delivery_counts_at_its_holders_offer_never_at_the_price_it_was_paid():
    """A lease paid the price of the moment counted at that price would keep the price up by itself."""
    auction, _, book = _two_keys()
    _deliver(auction, "dear", 25)
    _deliver(auction, "zero", 75)
    terms = _lease(auction, book, "zero")
    assert auction.params.alpha_of(terms.price) == pytest.approx(0.5)
    assert auction.params.alpha_of(terms.offer) == pytest.approx(1e-6)
    for _ in range(50):
        auction.observe_delivery("zero", 60.0, offer=terms.offer)
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)


def test_a_delivery_reposted_dearer_still_counts_at_its_grant_offer():
    auction, _, book = _two_keys()
    terms = _lease(auction, book, "zero")
    auction.post_ask("zero", 0.5, 84)              # reposted at the ceiling before delivering
    auction.observe_delivery("zero", 60.0, offer=terms.offer)
    assert auction._market.of("job")[-1][2] == pytest.approx(1e-6)


def test_old_deliveries_leave_the_price():
    auction, clock, book = _two_keys()
    _deliver(auction, "dear", 50)
    clock.now += PRICE_HORIZON_SECONDS + 1
    _active(auction, "zero", "mid", "dear")
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)   # its own offer
    assert auction._market.entries(clock.now) == []


def test_a_delivery_without_its_offer_sets_no_price():
    """A lease granted before the switch carries no offer: it times the next leases and prices nothing."""
    auction, _, _ = _two_keys()
    auction.observe_delivery("dear", 60.0)
    assert auction._volume_price(auction._clock()) is None
    assert len(auction._deliveries["dear"]) == 1


def test_a_few_deliveries_set_no_price():
    """Under PRICE_MIN_WEIGHT the quantile of a few deliveries is their dearest: one delivered at the ceiling would
    price the next wave."""
    auction, _, book = _two_keys()
    _deliver(auction, "dear", 4)
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(1e-6)


def test_older_deliveries_weigh_less():
    auction, clock, book = _two_keys()
    _deliver(auction, "zero", 80)
    clock.now += 3 * PRICE_HALF_LIFE_SECONDS       # weigh 80 / 8 = 10 now
    _active(auction, "zero", "mid", "dear")
    _deliver(auction, "mid", 15)                   # 16 % of the count, 60 % of the weight
    assert auction.params.alpha_of(_lease(auction, book, "zero").price) == pytest.approx(0.3)


def _market_name(request):
    return f"{request.node.name}|turn"


def test_a_new_order_starts_at_its_markets_price(request):
    first, _, _ = _auction(market_name=_market_name(request), job_id="first")
    first.post_ask("mid", 0.3, 4)
    _deliver(first, "mid", 30)
    second, clock, book = _auction(market_name=_market_name(request), job_id="second", budget=20)
    second.post_ask("zero", 1e-6, 10)
    second.post_ask("mid", 0.3, 10)
    _active(second, "zero", "mid")
    assert second.params.alpha_of(_lease(second, book, "zero").price) == pytest.approx(0.3)
    alone, _, alone_book = _auction(market_name=_market_name(request) + "-other", job_id="third", budget=20)
    alone.post_ask("zero", 1e-6, 10)
    alone.post_ask("mid", 0.3, 10)
    _active(alone, "zero", "mid")
    assert alone.params.alpha_of(_lease(alone, alone_book, "zero").price) == pytest.approx(1e-6)


def test_a_market_counts_offers_in_alpha_whatever_each_orders_emission(request):
    first, _, _ = _auction(market_name=_market_name(request), job_id="first")
    first.post_ask("mid", 0.3, 4)
    _deliver(first, "mid", 30)
    params = AuctionParams(ceiling=0.5 / 1e6 / (2 * EMISSION), miner_alpha_per_period=2 * EMISSION, initial_leases=20)
    second = TokenAuction(params, clock=Clock(), holdings=Book(), market_name=_market_name(request), job_id="second")
    assert second.params.alpha_of(second._volume_price(6000.0)) == pytest.approx(0.3)


def test_a_jobs_market_is_its_model_and_its_kind():
    class Job:
        checkpoint_repo = "org/model"
        episode = None
        job_id = "j"
    assert market_key(Job()) == "org/model|turn"
    Job.episode = object()
    assert market_key(Job()) == "org/model|episode"
    assert market_key(None) is None


def test_a_state_written_by_the_first_volume_price_keeps_its_deliveries(tmp_path):
    """#540 kept each hotkey's deliveries at its offer in pool share: a restart onto markets reads them in alpha."""
    import json
    path = tmp_path / "job.json"
    auction, clock, _ = _auction()
    share = auction.params.share_of(0.3)
    path.write_text(json.dumps({"budget": 8, "delivered": {"5B": [[clock.now - 10, share]] * 20,
                                                           "5A": [[clock.now - 20, share]]}}))
    again, _, _ = _auction(state_path=path, clock=clock)
    entries = again._market.of("job")
    assert len(entries) == 21 and entries[0][1] == "5A"
    assert [alpha for _, _, alpha in entries] == pytest.approx([0.3] * 21)
    assert again.params.alpha_of(again._volume_price(clock.now)) == pytest.approx(0.3)
