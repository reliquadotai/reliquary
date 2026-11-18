"""Leases at the consumer's pace, cheapest miners full (docs/design/2026-11-17-lease-allocation.md): speeds with
two counters, trusted slots declared then measured, the merit order and its clearing price, the market's pace and
its split between orders; then a job's auction allocating by merit order."""
from __future__ import annotations

import pytest

from reliquary.corpus import lease_allocator as la
from reliquary.corpus.token_auction import (
    ALLOWANCE_FULL,
    ASK_ABOVE_PRICE,
    AuctionParams,
    Terms,
    TokenAuction,
)

EMISSION = 100.0


class Clock:
    def __init__(self, now=10_000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_merit_enabled_names_markets():
    assert not la.merit_enabled("Qwen/Qwen3.5-9B|turn", {})
    assert la.merit_enabled("Qwen/Qwen3.5-9B|turn", {la.MERIT_ENV: "all"})
    assert la.merit_enabled("Qwen/Qwen3.5-9B|turn", {la.MERIT_ENV: "Qwen3.5-9B, other"})
    assert not la.merit_enabled("Qwen/Qwen3.6-27B|turn", {la.MERIT_ENV: "Qwen3.5-9B"})
    assert not la.merit_enabled(None, {la.MERIT_ENV: "Qwen3.5-9B"})


# -- speeds -------------------------------------------------------------------------------------------------------


def _run(book, clock, hotkey, held, per_delivery, every, seconds):
    """``hotkey`` holds ``held`` leases and delivers ``per_delivery`` tokens every ``every`` seconds."""
    end = clock.now + seconds
    while clock.now < end:
        clock.now += every
        book.tick(clock.now, {hotkey: held})
        book.delivered(hotkey, clock.now, per_delivery)


def test_speed_counts_every_lease_held_not_only_the_delivered():
    clock = Clock()
    honest, hoarder = la.SpeedBook(), la.SpeedBook()
    # Both deliver 2 000 tokens every 10 s; the hoarder holds 40 leases to do it, the honest miner 10.
    _run(honest, clock, "h", 10, 2000, 10.0, 600)
    clock.now -= 600
    _run(hoarder, clock, "h", 40, 2000, 10.0, 600)
    assert honest.speed("h", clock.now) == pytest.approx(20.0, rel=0.05)
    assert hoarder.speed("h", clock.now) == pytest.approx(5.0, rel=0.05)


def test_no_speed_before_enough_deliveries():
    clock = Clock()
    book = la.SpeedBook()
    _run(book, clock, "h", 10, 2000, 60.0, 60.0 * (la.MIN_DELIVERIES - 1))
    assert book.speed("h", clock.now) is None
    _run(book, clock, "h", 10, 2000, 60.0, 60.0)
    assert book.speed("h", clock.now) is not None


def test_speed_forgets_past_its_window():
    clock = Clock()
    book = la.SpeedBook()
    _run(book, clock, "h", 10, 2000, 10.0, 600)      # 20 tok/lease-s
    _run(book, clock, "h", 10, 500, 10.0, 900)       # then 5
    assert book.speed("h", clock.now) == pytest.approx(5.0, rel=0.1)


# -- the merit order ----------------------------------------------------------------------------------------------


def test_merit_order_fills_cheapest_first_and_cuts_only_the_last():
    # The design's example: R* = 25k tok/s; M4 is the marginal, partly served; M5 gets nothing.
    candidates = [la.Candidate("M1", 0.01, 100, 60.0), la.Candidate("M2", 0.02, 200, 40.0),
                  la.Candidate("M3", 0.03, 50, 30.0), la.Candidate("M4", 0.04, 300, 40.0),
                  la.Candidate("M5", 0.05, 80, 50.0)]
    allocation = la.merit_order(candidates, 25_000.0, median=40.0)
    assert allocation.slots == {"M1": 100, "M2": 200, "M3": 50, "M4": 238}
    assert allocation.clearing == 0.04
    assert allocation.reached


def test_a_slow_card_counts_for_what_it_delivers():
    cheap_slow = la.Candidate("slow", 0.0, 100, 2.0)        # 200 tok/s whatever its 100 slots
    dear_fast = la.Candidate("fast", 0.5, 100, 40.0)
    allocation = la.merit_order([cheap_slow, dear_fast], 1_000.0, median=20.0)
    assert allocation.slots == {"slow": 100, "fast": 20}
    assert allocation.clearing == 0.5


def test_an_added_offer_never_raises_the_clearing_price():
    """2026-11-18: nineteen hotkeys at 0 and a twentieth dear one. The dear one is either not needed, or the
    marginal: without it the marginal would be the next offer up."""
    book = [la.Candidate(f"zero{i}", 1e-6, 10, 20.0) for i in range(19)]
    book += [la.Candidate("rival-a", 0.10, 50, 20.0), la.Candidate("rival-b", 0.35, 50, 20.0)]
    for target in (3_000.0, 4_500.0, 5_500.0):
        without = la.merit_order(book, target, 20.0).clearing
        with_dear = la.merit_order(book + [la.Candidate("dear-sybil", 0.30, 20, 20.0)], target, 20.0).clearing
        assert with_dear <= without


def test_volume_delivered_does_not_set_the_price():
    """Three hotkeys at 1e-6 delivering three quarters of the volume do not choose the price: the last needed
    miner does."""
    book = [la.Candidate("zero-a", 1e-6, 200, 20.0), la.Candidate("zero-b", 1e-6, 64, 20.0),
            la.Candidate("zero-c", 1e-6, 64, 20.0), la.Candidate("mid", 0.10, 190, 20.0),
            la.Candidate("dear", 0.35, 288, 20.0)]
    assert la.merit_order(book, 20.0 * 400, 20.0).clearing == 0.10
    assert la.merit_order(book, 20.0 * 300, 20.0).clearing == 1e-6       # the cheap offers alone cover it


# -- trusted slots ------------------------------------------------------------------------------------------------


def test_an_honest_newcomer_keeps_its_whole_capacity():
    clock = Clock()
    book, trust = la.SpeedBook(), la.Allowances()
    trust.needed("new", clock.now, book)
    _run(book, clock, "new", 200, 2000, 0.5, 300)    # 200 slots at 20 tok/lease-s
    trust.step("new", clock.now, declared=200, held=200, speeds=book, median=20.0, median_seconds=100.0,
               tail=False)
    assert trust.of("new", 200) == 200


def test_an_over_declared_capacity_drops_to_what_is_delivered_at_the_first_verdict():
    clock = Clock()
    book, trust = la.SpeedBook(), la.Allowances()
    trust.needed("liar", clock.now, book)
    _run(book, clock, "liar", 500, 2000, 1.0, 300)   # 2 000 tok/s for 500 slots declared
    changed = trust.step("liar", clock.now, declared=500, held=500, speeds=book, median=20.0,
                         median_seconds=100.0, tail=False)
    assert changed.startswith("verdict")
    assert trust.of("liar", 500) == 100              # 2 000 tok/s / 20 per slot


def test_a_slow_full_hotkey_is_cut_then_grows_back_when_fast():
    clock = Clock()
    book, trust = la.SpeedBook(), la.Allowances()
    trust.needed("h", clock.now, book)
    _run(book, clock, "h", 100, 2000, 1.0, 300)
    trust.step("h", clock.now, declared=100, held=100, speeds=book, median=20.0, median_seconds=100.0, tail=False)
    assert trust.of("h", 100) == 100                 # judged honest
    _run(book, clock, "h", 100, 200, 1.0, 600)       # then 2 tok/lease-s, under 1/4 of 20
    trust.step("h", clock.now, declared=100, held=100, speeds=book, median=20.0, median_seconds=100.0, tail=False)
    assert trust.of("h", 100) == 80
    _run(book, clock, "h", 80, 2000, 1.0, 600)       # fast again, full
    trust.step("h", clock.now, declared=100, held=80, speeds=book, median=20.0, median_seconds=100.0, tail=False)
    assert trust.of("h", 100) == 100


def test_the_tail_cuts_at_half_the_median():
    assert la.vmin(20.0, tail=False) == 5.0
    assert la.vmin(20.0, tail=True) == 10.0


# -- the market's pace --------------------------------------------------------------------------------------------


def test_one_order_takes_the_audit_rate_over_its_audit_share():
    clock = Clock()
    pace = la.MarketPace(clock)
    pace.bind_audit(lambda: (5_000.0, 0.9))
    pace.report("job", q=0.2, remaining_tokens=1e9, deadline=None, supply=None, backlog_tokens=0, unmet_at=None)
    assert pace.target("job") == pytest.approx(25_000.0)


def test_the_queue_lowers_k_when_the_wait_grows():
    clock = Clock()
    pace = la.MarketPace(clock)
    pace.bind_audit(lambda: (1_000.0, 0.99))
    pace.report("job", q=1.0, remaining_tokens=1e9, deadline=None, supply=None, backlog_tokens=240_000,
                unmet_at=None)
    pace.target("job")
    assert pace.k == pytest.approx(0.5)              # a 240 s wait, the card full: down, at most by half


def test_an_idle_card_with_miners_refused_raises_k():
    clock = Clock()
    pace = la.MarketPace(clock)
    pace.bind_audit(lambda: (1_000.0, 0.5))
    pace.report("job", q=1.0, remaining_tokens=1e9, deadline=None, supply=None, backlog_tokens=0,
                unmet_at=clock.now)
    pace.target("job")
    assert pace.k == pytest.approx(la.K_GROW)


def test_orders_get_their_floor_then_the_surplus_goes_to_the_closest_to_done():
    jobs = {
        "big": la._JobReport(q=1.0, remaining_tokens=86_400 * 100.0, deadline=86_400.0, supply=None),
        "small": la._JobReport(q=1.0, remaining_tokens=60_000.0, deadline=86_400.0, supply=2_000.0),
    }
    rates = la.split(jobs, 3_000.0, now=0.0, first_seen={})
    assert rates["big"] == pytest.approx(100.0 + (3_000.0 - 100.0 - 2_000.0))
    assert rates["small"] == pytest.approx(2_000.0)  # its whole supply: it finishes in 30 s


def test_floors_over_the_card_are_scaled_down():
    jobs = {"a": la._JobReport(q=1.0, remaining_tokens=2e6, deadline=1_000.0),
            "b": la._JobReport(q=1.0, remaining_tokens=2e6, deadline=1_000.0)}
    rates = la.split(jobs, 2_000.0, now=0.0, first_seen={})
    assert rates == {"a": pytest.approx(1_000.0), "b": pytest.approx(1_000.0)}


# -- a job's auction by merit order -------------------------------------------------------------------------------


class Book:
    def __init__(self):
        self.held = {}

    def __call__(self):
        return dict(self.held)

    def take(self, terms):
        self.held[terms.hotkey] = self.held.get(terms.hotkey, 0) + 1


def _merit_auction(monkeypatch, clock, audit=(400.0, 0.9), q=1.0, market="m-model|turn"):
    monkeypatch.setenv(la.MERIT_ENV, "all")
    monkeypatch.setattr(la, "_PACES", {})
    params = AuctionParams(ceiling=0.706 / 1e6 / EMISSION, miner_alpha_per_period=EMISSION, initial_leases=1000)
    book = Book()
    auction = TokenAuction(params, clock=clock, holdings=book, market_name=market, job_id="job",
                           backlog=lambda: 0)
    auction.bind_merit(audit=lambda: audit, q=lambda: q)
    return auction, book


def _warm(auction, book, clock, hotkeys, seconds=700):
    """Each hotkey holds its leases and delivers 200 tokens per lease every 10 s (20 tok/lease-s)."""
    end = clock.now + seconds
    while clock.now < end:
        clock.now += 10.0
        for hotkey in hotkeys:
            for _ in range(book.held.get(hotkey, 0)):
                auction.observe_delivery(hotkey, 10.0, tokens=200)
        auction.view()


def test_the_cheapest_are_full_the_dear_wait_and_the_price_is_the_marginal_offer(monkeypatch):
    clock = Clock()
    auction, book = _merit_auction(monkeypatch, clock, audit=(400.0, 0.9))   # 400 tok/s = 20 leases at 20
    auction.post_ask("zero", 1e-6, 10)
    auction.post_ask("mid", 0.10, 15)
    auction.post_ask("dear", 0.70, 20)
    # Speeds start unmeasured: the thermostat stands in until a first measure.
    for hotkey in ("zero", "mid", "dear"):
        for _ in range(5):
            terms = auction.admit(hotkey)
            if isinstance(terms, Terms):
                book.take(terms)
                auction.unreserve(terms)
    _warm(auction, book, clock, ["zero", "mid", "dear"])
    book.held = {}
    granted = {}
    for hotkey in ("dear", "mid", "zero"):           # the dear one asks first: timing no longer matters
        while True:
            terms = auction.admit(hotkey)
            if not isinstance(terms, Terms):
                break
            book.take(terms)
            auction.unreserve(terms)
            granted[hotkey] = granted.get(hotkey, 0) + 1
            last = terms
    assert granted.get("zero") == 10
    assert granted.get("mid") == 10                  # partly served: the marginal
    assert "dear" not in granted
    assert auction.admit("dear") == ASK_ABOVE_PRICE
    assert auction.admit("zero") == ALLOWANCE_FULL
    assert last.price == pytest.approx(auction.params.share_of(0.10))
    view = auction.view("zero")
    assert view["allocator"] == "merit"
    assert view["price_alpha_per_mtoken"] == pytest.approx(0.10)


def test_without_the_flag_nothing_changes(monkeypatch):
    monkeypatch.delenv(la.MERIT_ENV, raising=False)
    params = AuctionParams(ceiling=0.706 / 1e6 / EMISSION, miner_alpha_per_period=EMISSION, initial_leases=8)
    auction = TokenAuction(params, clock=Clock(), holdings=Book(), market_name="m|turn", job_id="job")
    assert not auction.merit
    auction.post_ask("a", 0.1, 4)
    assert isinstance(auction.admit("a"), Terms)
    assert "allocator" not in auction.view()


def test_an_rl_env_fills_its_trainer_budget_cheapest_first(monkeypatch):
    """RL: the budget is what the trainer's next steps take (in leases); once speeds are measured it goes to the
    cheapest offers, not to whoever asks when a lease frees."""
    monkeypatch.setenv(la.MERIT_ENV, "all")
    monkeypatch.setattr(la, "_PACES", {})
    clock = Clock()
    params = AuctionParams(ceiling=0.706 / 1e6 / EMISSION, miner_alpha_per_period=EMISSION, initial_leases=4)
    book = Book()
    auction = TokenAuction(params, clock=clock, holdings=book, market_name="rl|task|math", job_id="rl-task-math",
                           budget=lambda: 6)
    auction.post_ask("cheap", 0.05, 6)
    auction.post_ask("dear", 0.50, 6)
    for hotkey in ("cheap", "dear"):
        for _ in range(3):
            terms = auction.admit(hotkey)
            if isinstance(terms, Terms):
                book.take(terms)
                auction.unreserve(terms)
    _warm(auction, book, clock, ["cheap", "dear"])
    book.held = {}
    granted = {}
    for hotkey in ("dear", "cheap"):
        while isinstance(terms := auction.admit(hotkey), Terms):
            book.take(terms)
            auction.unreserve(terms)
            granted[hotkey] = granted.get(hotkey, 0) + 1
    assert granted == {"cheap": 6}
