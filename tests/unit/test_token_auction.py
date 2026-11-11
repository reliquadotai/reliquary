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
    PRICING_PARAM,
    STEER_SECONDS,
    AuctionParams,
    Terms,
    TokenAuction,
    auction_params,
    job_auction,
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


def _auction(ceiling_alpha=0.5, budget=8, backlog=None, clock=None, state_path=None, low=8, high=32):
    params = AuctionParams(ceiling=ceiling_alpha / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                           initial_leases=budget, backlog_low=low, backlog_high=high)
    clock = clock or Clock()
    book = Book()
    auction = TokenAuction(params, clock=clock, backlog=backlog, holdings=book, mean_tokens=1000,
                           state_path=state_path)
    return auction, clock, book


def _lease(auction, book, hotkey):
    """A lease as the route grants it: reserve, count as held, give the reservation back."""
    terms = auction.admit(hotkey)
    if isinstance(terms, Terms):
        book.take(terms)
        auction.unreserve(terms)
    return terms


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


def test_a_lease_carries_its_holders_own_offer():
    auction, _, book = _auction()
    auction.post_ask("5A", 0.3, 4)
    terms = _lease(auction, book, "5A")
    assert auction.params.alpha_of(terms.price) == pytest.approx(0.3)


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
    auction.save(force=True)
    again, _, book = _auction(state_path=path, clock=clock)
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
