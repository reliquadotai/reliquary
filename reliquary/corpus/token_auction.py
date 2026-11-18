"""The price auction of one draw job (docs/design/2026-11-10-order-price-auction-design.md).

A job priced by auction (registry ``pricing: "auction"``) hands its samples out cheapest first, as fast as its audit
verifies them, and pays every verified token of a period the same price:

* **Offers.** Each hotkey posts one offer (``post_ask``): the alpha per million verified tokens it works for, and the
  samples it can hold at once (its capacity, taken as declared). An offer lives ``ASK_TTL_SECONDS`` unless reposted.
* **Budget.** The samples leased at once across the job follow the audit queue (``backlog``), as a thermostat: under
  ``backlog_low`` records waiting, with the budget in use, it grows; over ``backlog_high`` it shrinks.
* **Allowances.** Walking the offers from the cheapest, each active hotkey gets up to its capacity until the budget
  is covered. A hotkey is granted a lease while it holds fewer than its allowance.
* **The price of the moment.** A lease is paid, from its grant, the higher of its holder's own offer and the price of
  the moment: the offer under which ``PRICE_QUANTILE`` of the recently delivered leases were won (each counted at its
  holder's own offer when granted, its weight halving every ``PRICE_HALF_LIFE_SECONDS``), never above the dearest
  offer given work now. Weighted by delivered volume, not by allowances: a second hotkey offering the ceiling for
  one lease it never delivers no longer sets the price of every lease (``_volume_price``). The deliveries are those
  of the job's market (``market_key``: its model, agentic or not), so a new order starts at its market's price; under
  ``PRICE_MIN_WEIGHT`` of them, a lease is paid its own offer.
* **Leases not delivered.** A lease that ends without a submission charges its holder its price times the job's
  mean submission size (by the share of its time held on a release), taken back from its next accepted
  submissions (``take_debit``). An expiry, or a release past half its time, halves the holder's capacity for
  ``PENALTY_SECONDS``; a hotkey in debt holds one lease at most until it has paid.
* **Lease time** follows each hotkey's measured delivery time (``lease_seconds``).
* **Merit order** (``RELIQUARY_MERIT_ORDER``, ``corpus.lease_allocator``): on the markets it names, the leases
  follow the rate the market's audit can take instead of the thermostat, filled cheapest first in measured tokens
  per second; each hotkey gets its declared capacity at once, then what its deliveries show; only the last needed
  hotkey is partly served, and its offer is the price of every lease (not a quantile of the delivered volume).
  Until the audit and the speeds are measured, the thermostat and the volume price stand in.

Prices are kept in pool share per token, the settlement's unit (``corpus_periods.max_price``); offers and the public
view are in alpha per million tokens, converted with the miner emission per period declared with the ceiling.

The state (offers, debts, penalties, budget, delivery times) can be kept in a file (``state_path``), so a restart
keeps it.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import statistics
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from reliquary.corpus import lease_allocator as la

logger = logging.getLogger(__name__)

PRICING_PARAM = "pricing"
PRICING_AUCTION = "auction"
INITIAL_LEASES_PARAM = "auction_initial_leases"
BACKLOG_LOW_PARAM = "auction_backlog_low"
BACKLOG_HIGH_PARAM = "auction_backlog_high"
AUCTION_PARAMS = (PRICING_PARAM, INITIAL_LEASES_PARAM, BACKLOG_LOW_PARAM, BACKLOG_HIGH_PARAM)
# The local directory a validator keeps each auction's state in (one file per job); unset: in memory only.
STATE_DIR_ENV = "RELIQUARY_AUCTION_STATE_DIR"

DEFAULT_INITIAL_LEASES = 64
DEFAULT_BACKLOG_LOW = 8
DEFAULT_BACKLOG_HIGH = 32
MIN_LEASES = 4
MAX_LEASES = 100_000
MAX_BACKLOG = 1_000_000
# The budget: x SHRINK over the high mark, x GROW (at least +1) under the low mark while USED of it is leased.
BUDGET_SHRINK = 0.8
BUDGET_GROW = 1.25
BUDGET_USED = 0.8
STEER_SECONDS = 30.0
ASK_TTL_SECONDS = 3600.0
ASK_REPOST_SECONDS = 600.0
# A hotkey counts for the allowances while it asked for a lease this recently, or holds one.
ACTIVE_SECONDS = 180.0
# The price of the moment: the offer under which this share of the recently delivered leases were won, each
# delivery's weight halving every PRICE_HALF_LIFE_SECONDS and dropped past PRICE_HORIZON_SECONDS; with under
# PRICE_MIN_WEIGHT of them (a few deliveries: their quantile is their dearest), no price of the moment.
PRICE_QUANTILE = 0.8
PRICE_HALF_LIFE_SECONDS = 300.0
PRICE_HORIZON_SECONDS = 1800.0
PRICE_MIN_WEIGHT = 20.0
PRICE_DELIVERIES_KEPT = 16384
# The largest ask accepted, in alpha per million tokens (a typo guard; the ceiling decides who works).
MAX_ASK_ALPHA_PER_MTOKEN = 1e9
MAX_CAPACITY = 10_000
# A missed lease halves its holder's capacity for this long.
PENALTY_SECONDS = 3600.0
# A release past this share of its lease time counts as a miss.
LATE_RELEASE = 0.5
# Lease time: twice the median of a hotkey's recent deliveries, once it has this many, within [floor, the job's].
DELIVERIES_KEPT = 256
DELIVERIES_NEEDED = 3
# The share of a hotkey's recent deliveries its lease time covers twice over (`lease_seconds`).
DELIVERY_TAIL = 0.9
LEASE_FLOOR_SECONDS = 300.0
# The mean submission size, in tokens, an unused lease is charged at: an EMA of accepted submissions.
TOKENS_EMA_WEIGHT = 0.05
SAVE_SECONDS = 5.0
# The auditor's audited share is trusted over this many verdicts; under it, the share kept from before a restart.
Q_TRUSTED_DECISIONS = 300

NO_ASK = "no_ask"
ASK_ABOVE_PRICE = "ask_above_price"
ALLOWANCE_FULL = "allowance_full"


class Market:
    """The deliveries of the jobs sold on one market (``market_key``), each at its holder's own offer in alpha per
    million tokens (the unit every job's offers share): each job's auction records into it and prices from all of
    it, so a new order starts at its market's price, not at nothing.

    2026-11-16: one history per job left every order's first wave of leases paid each holder's own offer (a short
    eval, all of its life) and priced the next wave on its first few deliveries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, collections.deque] = {}    # job -> (delivered at, hotkey, alpha per Mtoken)

    def record(self, job: str, at: float, hotkey: str, alpha: float) -> None:
        with self._lock:
            self._jobs.setdefault(job, collections.deque(maxlen=PRICE_DELIVERIES_KEPT)).append(
                (float(at), str(hotkey), float(alpha)))

    def replace(self, job: str, entries) -> None:
        with self._lock:
            self._jobs[job] = collections.deque(((float(t), str(h), float(a)) for t, h, a in entries),
                                                maxlen=PRICE_DELIVERIES_KEPT)

    def of(self, job: str) -> list[tuple[float, str, float]]:
        with self._lock:
            return list(self._jobs.get(job, ()))

    def entries(self, now: float) -> list[tuple[float, str, float]]:
        """Every job's deliveries within the horizon (older ones dropped)."""
        with self._lock:
            kept = []
            for job in list(self._jobs):
                entries = self._jobs[job]
                while entries and now - entries[0][0] > PRICE_HORIZON_SECONDS:
                    entries.popleft()
                if not entries:
                    del self._jobs[job]
                    continue
                kept.extend(entries)
            return kept


_MARKETS: dict[str, Market] = {}
_MARKETS_LOCK = threading.Lock()


def market(key: str) -> Market:
    """The process's market named ``key`` (``market_key``)."""
    with _MARKETS_LOCK:
        return _MARKETS.setdefault(str(key), Market())


def market_key(job) -> str | None:
    """A job's market: its model, and whether its work is agentic (episodes) or single turns. None without a
    model (the job then prices from its own deliveries alone)."""
    repo = getattr(job, "checkpoint_repo", None)
    if not repo:
        return None
    return f"{repo}|{'episode' if getattr(job, 'episode', None) is not None else 'turn'}"


@dataclass(frozen=True, slots=True)
class AuctionParams:
    ceiling: float                      # pool share per token
    miner_alpha_per_period: float
    initial_leases: int = DEFAULT_INITIAL_LEASES
    backlog_low: int = DEFAULT_BACKLOG_LOW
    backlog_high: int = DEFAULT_BACKLOG_HIGH

    def share_of(self, alpha_per_mtoken: float) -> float:
        return float(alpha_per_mtoken) / 1e6 / self.miner_alpha_per_period

    def alpha_of(self, share: float) -> float:
        return float(share) * 1e6 * self.miner_alpha_per_period


def _whole(params: Mapping, name: str, default: int, low: int, high: int) -> int:
    value = params.get(name, default)
    if isinstance(value, bool) or float(value) != int(float(value)) or not low <= int(float(value)) <= high:
        raise ValueError(f"{name} must be an integer within [{low}, {high}]")
    return int(float(value))


def auction_params(params: Mapping | None) -> AuctionParams | None:
    """The auction of a job's registry params; None when the job is not priced by auction. Raises ``ValueError``
    on an auction declared without a price ceiling or with bad settings (``task_registry`` refuses them)."""
    from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM, max_price

    if not isinstance(params, Mapping) or params.get(PRICING_PARAM) is None:
        return None
    if params[PRICING_PARAM] != PRICING_AUCTION:
        raise ValueError(f"unknown pricing {params[PRICING_PARAM]!r}")
    ceiling = max_price(params)
    emission = params.get(MAX_PRICE_EMISSION_PARAM)
    if ceiling is None or emission is None:
        raise ValueError("an auction needs a price ceiling (max_price_per_token)")
    if not (math.isfinite(float(ceiling)) and float(ceiling) > 0 and float(emission) > 0):
        raise ValueError("the price ceiling and the miner emission must be positive")
    initial = _whole(params, INITIAL_LEASES_PARAM, DEFAULT_INITIAL_LEASES, MIN_LEASES, MAX_LEASES)
    low = _whole(params, BACKLOG_LOW_PARAM, DEFAULT_BACKLOG_LOW, 0, MAX_BACKLOG)
    high = _whole(params, BACKLOG_HIGH_PARAM, DEFAULT_BACKLOG_HIGH, 1, MAX_BACKLOG)
    if high <= low:
        raise ValueError(f"{BACKLOG_HIGH_PARAM} must be above {BACKLOG_LOW_PARAM}")
    return AuctionParams(ceiling=float(ceiling), miner_alpha_per_period=float(emission),
                         initial_leases=initial, backlog_low=low, backlog_high=high)


@dataclass(frozen=True, slots=True)
class Terms:
    """What a lease granted now pays: ``price`` per token (pool share), the price of the moment never under its
    holder's own offer, which ``offer`` keeps (the price its delivery counts at, ``observe_delivery``)."""
    price: float
    hotkey: str
    round: int | None = None
    offer: float | None = None


@dataclass(frozen=True, slots=True)
class Offer:
    price: float                        # pool share per token
    capacity: int
    posted_at: float
    swap: bool = False                  # takes its next leases in its submissions' answers


class TokenAuction:
    def __init__(self, params: AuctionParams, *, clock: Callable[[], float] = time.time,
                 backlog: Callable[[], int | None] | None = None,
                 holdings: Callable[[], Mapping[str, int]] | None = None,
                 mean_tokens: float = 1000.0, state_path: str | os.PathLike | None = None,
                 market_name: str | None = None, job_id: str = "job",
                 budget: Callable[[], int | None] | None = None) -> None:
        self.params = params
        # The deliveries this job prices from: its market's, else its own (``Market``).
        self._market = market(market_name) if market_name is not None else Market()
        self._job = str(job_id)
        self._clock = clock
        self._backlog = backlog
        # A budget the caller knows (an RL env: what the trainer's next steps take), read at each lease instead of
        # the thermostat on the audit queue. None from it: keep the last value.
        self._budget_of = budget
        self._holdings = holdings
        self._lock = threading.RLock()
        self._offers: dict[str, Offer] = {}
        self._seen: dict[str, float] = {}                      # hotkey -> last lease request
        # The last time an active offer was refused for want of budget: unmet demand the thermostat may grow for.
        self._unmet_at: float | None = None
        # Hotkeys the lease route set aside (a suspect miner) until the time given: no allowance meanwhile.
        self._aside: dict[str, float] = {}
        self._reserved: collections.Counter = collections.Counter()
        self._budget = int(params.initial_leases)
        self._steered_at: float | None = None
        self._last_backlog: int | None = None
        self._debt: dict[str, float] = {}
        self._missed: dict[str, list[float]] = {}
        self._deliveries: dict[str, collections.deque] = {}
        self._mean_tokens = float(mean_tokens)
        self._state_path = Path(state_path) if state_path is not None else None
        self._saved_at = 0.0
        self._dirty = False
        # Merit order (``corpus.lease_allocator``): speeds, trusted slots, the market's pace, the job's progress.
        self._merit = market_name is not None and la.merit_enabled(market_name)
        # The shadow: everything computed, logged and shown, nothing used for a lease or a price.
        self._shadow = (not self._merit and market_name is not None
                        and la.merit_enabled(market_name, name=la.SHADOW_ENV))
        self._measures = self._merit or self._shadow
        self._shadow_logged_at: float | None = None
        self._pace = la.market_pace(market_name, clock) if self._measures else None
        self._speeds = la.SpeedBook()
        self._trusted = la.Allowances()
        self._q: Callable[[], float] | None = None
        self._deadline: float | None = None
        self._progress: tuple[int, int] | None = None     # (filled, total) slots
        self._ticked_at: float | None = None
        self._allocation: la.Allocation | None = None
        self._allocated_at: float | None = None
        self._target: float | None = None
        # The median speed last measured, kept over a restart: the merit order resumes at once instead of
        # leaving the job to the thermostat while speeds are measured again (2026-11-18: 5 min at 4 leases).
        self._median_hint: float | None = None
        self._q_weight: Callable[[], int] | None = None
        self._q_hint: float | None = None
        self._load()

    def bind_holdings(self, holdings: Callable[[], Mapping[str, int]]) -> None:
        """The live leases per hotkey (``DrawLeases.counts``), once the job's leases exist."""
        with self._lock:
            self._holdings = holdings

    @property
    def merit(self) -> bool:
        """Whether this job allocates by merit order (``corpus.lease_allocator``)."""
        return self._merit

    @property
    def measures(self) -> bool:
        """Whether this job computes the merit order, to allocate by it or in the shadow."""
        return self._measures

    def bind_merit(self, *, audit: Callable[[], tuple[float, float | None] | None] | None = None,
                   q: Callable[[], float] | None = None, deadline: float | None = None,
                   q_weight: Callable[[], int] | None = None) -> None:
        """The merit order's inputs: the market's audit (output tokens per second its cards verify, their busy
        share), the share of this job's records audited, its deadline (unix seconds; None: a day after it opens)."""
        if self._pace is not None and audit is not None:
            self._pace.bind_audit(audit)
        with self._lock:
            self._q = q
            self._q_weight = q_weight
            self._deadline = deadline

    def note_progress(self, filled: int, total: int) -> None:
        """The job's slots filled and in all (its ledger, read by the lease route)."""
        with self._lock:
            self._progress = (int(filled), int(total))

    # -- offers --------------------------------------------------------------

    def post_ask(self, hotkey: str, alpha_per_mtoken: float, capacity: int | None = None,
                 swap: bool = False) -> str | None:
        """Record ``hotkey``'s offer; a refusal string when it is not a positive price and capacity. ``swap``: it
        takes its next leases in its accepted submissions' answers."""
        try:
            alpha = float(alpha_per_mtoken)
            held = 1 if capacity is None else int(capacity)
        except (TypeError, ValueError):
            return "invalid_ask"
        if not (math.isfinite(alpha) and 0 < alpha <= MAX_ASK_ALPHA_PER_MTOKEN and 1 <= held <= MAX_CAPACITY):
            return "invalid_ask"
        with self._lock:
            self._offers[hotkey] = Offer(self.params.share_of(alpha), held, self._clock(), bool(swap))
            self._changed()
        return None

    def swaps(self, hotkey: str) -> bool:
        """Whether ``hotkey``'s live offer takes its next leases in its submissions' answers."""
        with self._lock:
            offer = self._live(hotkey, self._clock())
            return offer is not None and offer.swap

    def ask_of(self, hotkey: str) -> float | None:
        """``hotkey``'s live offer price (pool share per token), or None."""
        with self._lock:
            offer = self._live(hotkey, self._clock())
            return None if offer is None else offer.price

    def _live(self, hotkey: str, now: float) -> Offer | None:
        offer = self._offers.get(hotkey)
        if offer is None:
            return None
        if now - offer.posted_at > ASK_TTL_SECONDS:
            del self._offers[hotkey]
            self._changed()
            return None
        return offer

    # -- the budget ----------------------------------------------------------

    def _held(self) -> Mapping[str, int]:
        if self._holdings is None:
            return {}
        try:
            return self._holdings()
        except Exception:                     # a status read never stops leasing
            logger.exception("token auction: lease holdings unavailable")
            return {}

    def _given_budget(self) -> int | None:
        """The caller's budget (``budget``), read before this auction's lock: it may read the leases' own store."""
        if self._budget_of is None:
            return None
        try:
            given = self._budget_of()
        except Exception:                     # a status read never stops leasing
            logger.exception("token auction: budget unavailable")
            return None
        return None if given is None else max(0, min(MAX_LEASES, int(given)))

    def _steer(self, now: float, held: Mapping[str, int], given: int | None = None) -> None:
        if self._budget_of is not None:
            if given is not None and given != self._budget:
                self._budget = given
                self._changed()
            return
        if self._steered_at is not None and now - self._steered_at < STEER_SECONDS:
            return
        self._steered_at = now
        if self._backlog is None:
            return
        try:
            backlog = self._backlog()
        except Exception:                     # a status read never stops leasing
            logger.exception("token auction: audit backlog unavailable")
            return
        if backlog is None:
            return
        self._last_backlog = int(backlog)
        before = self._budget
        if backlog > self.params.backlog_high:
            self._budget = max(MIN_LEASES, int(self._budget * BUDGET_SHRINK))
        elif backlog < self.params.backlog_low and (
                sum(held.values()) >= BUDGET_USED * self._budget
                # An offer refused for want of budget since the last step: the budget grows for it even while a
                # cheaper hotkey leaves its allowance unused (2026-11-13: one slow offer held a 15-lease budget at
                # 4 leases, and the thirty offers behind it got nothing for an hour).
                or (self._unmet_at is not None and now - self._unmet_at <= 2 * STEER_SECONDS)):
            self._budget = min(MAX_LEASES, max(self._budget + 1, int(self._budget * BUDGET_GROW)))
        if self._budget != before:
            self._changed()

    # -- allowances ----------------------------------------------------------

    def _capacity(self, hotkey: str, offer: Offer, now: float) -> int:
        missed = [t for t in self._missed.get(hotkey, ()) if now - t < PENALTY_SECONDS]
        if missed:
            self._missed[hotkey] = missed
        else:
            self._missed.pop(hotkey, None)
        capacity = max(1, offer.capacity >> len(missed))
        return 1 if self._debt.get(hotkey, 0.0) > 0 else capacity

    def _allowances(self, now: float, held: Mapping[str, int]) -> tuple[dict[str, int], float | None]:
        """Each active hotkey's allowance, cheapest first until the budget is covered, and the highest price given
        one (None when nobody has an allowance)."""
        offers = []
        for hotkey in list(self._offers):
            offer = self._live(hotkey, now)
            if offer is None or offer.price > self.params.ceiling:
                continue
            if self._aside.get(hotkey, -math.inf) > now:
                continue
            if held.get(hotkey, 0) > 0 or now - self._seen.get(hotkey, -math.inf) <= ACTIVE_SECONDS:
                offers.append((offer.price, offer.posted_at, hotkey, offer))
        offers.sort()
        left = self._budget
        given: dict[str, int] = {}
        marginal = None
        for price, _, hotkey, offer in offers:
            if left <= 0:
                break
            share = min(self._capacity(hotkey, offer, now), left)
            given[hotkey] = share
            left -= share
            marginal = price
        return given, marginal

    # -- merit order ---------------------------------------------------------

    def _pace_target(self) -> float | None:
        """This job's target rate from its market's pace, read before this auction's lock (it reads the audit)."""
        if self._pace is None:
            return None
        try:
            return self._pace.target(self._job)
        except Exception:                     # a status read never stops leasing
            logger.exception("token auction: market pace unavailable")
            return None

    def _tail(self) -> bool:
        if self._progress is None:
            return False
        filled, total = self._progress
        return total > 0 and (total - filled) <= la.TAIL_SHARE * total

    def _median_seconds(self) -> float | None:
        recent = [s for times in self._deliveries.values() for s in list(times)[-32:]]
        return statistics.median(recent) if recent else None

    def _merit_allowances(self, now: float, held: Mapping[str, int],
                          target: float | None) -> tuple[dict[str, int], float | None] | None:
        """The merit order's slots per hotkey and its clearing price (the last needed hotkey's offer), or None
        while the audit or every speed is unmeasured (the thermostat then stands in)."""
        if not self._measures:
            return None
        if self._ticked_at is None or now - self._ticked_at >= 1.0:
            self._ticked_at = now
            self._speeds.tick(now, held)
        median = self._speeds.median(now)
        if median is not None:
            self._median_hint = median
        else:
            median = self._median_hint
        if (self._allocation is None or self._allocated_at is None or now - self._allocated_at >= 5.0
                or target != self._target):
            self._allocated_at, self._target = now, target
            self._allocation = None
            if target is not None and median is not None and median > 0:
                candidates = []
                for hotkey in list(self._offers):
                    offer = self._live(hotkey, now)
                    if offer is None or offer.price > self.params.ceiling:
                        continue
                    if self._aside.get(hotkey, -math.inf) > now:
                        continue
                    if not (held.get(hotkey, 0) > 0 or now - self._seen.get(hotkey, -math.inf) <= ACTIVE_SECONDS):
                        continue
                    declared = self._capacity(hotkey, offer, now)
                    candidates.append(la.Candidate(hotkey, offer.price, self._trusted.of(hotkey, declared),
                                                   self._speeds.speed(hotkey, now)))
                self._allocation = la.merit_order(candidates, target, median)
                tail = self._tail()
                median_seconds = self._median_seconds()
                for c in candidates:
                    if self._allocation.slots.get(c.hotkey, 0) > 0:
                        self._trusted.needed(c.hotkey, now, self._speeds)
                    offer = self._offers[c.hotkey]
                    changed = self._trusted.step(c.hotkey, now, declared=self._capacity(c.hotkey, offer, now),
                                                 held=held.get(c.hotkey, 0), speeds=self._speeds, median=median,
                                                 median_seconds=median_seconds, tail=tail,
                                                 trim_unused=not self._shadow)
                    if changed:
                        logger.info("merit order%s job=%s hotkey=%s %s", " shadow" if self._shadow else "",
                                    self._job, c.hotkey[:8], changed)
            self._report(now, held)
        if self._allocation is None:
            return None
        return dict(self._allocation.slots), self._allocation.clearing

    def _report(self, now: float, held: Mapping[str, int]) -> None:
        if self._pace is None:
            return
        q = 1.0
        if self._q is not None:
            try:
                q = float(self._q())
                weight = None if self._q_weight is None else int(self._q_weight())
            except Exception:
                logger.exception("token auction: audit share unavailable")
                weight = None
            else:
                # Freshly started, the auditor's share stands on few verdicts and reads near q (its prior): the
                # share measured before the restart stands until enough land (2026-11-18: 0.2 read against 0.43
                # audited, R* twice too high, the queue swelled, k fell to 0.25 and the 27B price 2.01 -> 0.034).
                if weight is not None and weight < Q_TRUSTED_DECISIONS and self._q_hint is not None:
                    q = max(q, self._q_hint)
                elif weight is None or weight >= Q_TRUSTED_DECISIONS:
                    self._q_hint = q
        remaining = None
        if self._progress is not None:
            remaining = max(0, self._progress[1] - self._progress[0]) * self._mean_tokens
        self._pace.report(self._job, q=q, remaining_tokens=remaining, deadline=self._deadline,
                          supply=None if self._allocation is None else self._allocation.supply,
                          backlog_tokens=(self._last_backlog or 0) * self._mean_tokens, unmet_at=self._unmet_at)

    def _volume_price(self, now: float) -> float | None:
        """The offer under which ``PRICE_QUANTILE`` of the market's recently delivered leases were won, each
        weighted by its age; None under ``PRICE_MIN_WEIGHT``. A hotkey in debt or set aside here counts for nothing.

        2026-11-16: the price was the dearest offer given an allowance, even of one lease. One operator could
        offer 0 for the budget less one on a hotkey and the ceiling for one lease, never delivered, on another:
        every lease of the first was paid the ceiling. Here raising the price takes delivering a fifth of the work
        at that price, and in a market with enough cheaper offers that work goes to them."""
        weighted = []
        for at, hotkey, alpha in self._market.entries(now):
            if self._debt.get(hotkey, 0.0) > 0 or self._aside.get(hotkey, -math.inf) > now:
                continue
            weighted.append((alpha, 0.5 ** (max(0.0, now - at) / PRICE_HALF_LIFE_SECONDS)))
        total = sum(weight for _, weight in weighted)
        if total < PRICE_MIN_WEIGHT:
            return None
        weighted.sort()
        reach, summed = PRICE_QUANTILE * total * (1.0 - 1e-9), 0.0
        for alpha, weight in weighted:
            summed += weight
            if summed >= reach:
                return self.params.share_of(alpha)
        return self.params.share_of(weighted[-1][0])

    def _moment_price(self, now: float, marginal: float | None) -> float | None:
        """The price of the moment: the delivered volume's price, never above the dearest offer given work now."""
        volume = self._volume_price(now)
        if volume is None or marginal is None:
            return None
        return min(volume, marginal)

    # -- leases --------------------------------------------------------------

    def note_request(self, hotkey: str) -> None:
        """``hotkey`` asks for leases now: it counts for the allowances before any request of the same batch is
        served (an RL interval resolves its requests together, cheapest first, whatever their arrival order)."""
        with self._lock:
            if hotkey in self._offers:
                self._seen[hotkey] = self._clock()

    def admit(self, hotkey: str) -> Terms | str:
        """Reserve one lease for ``hotkey`` under its allowance: its terms, or why it gets none. Every reservation
        is given back with ``unreserve`` once its lease is granted or not."""
        # Read before taking this lock: the leases call this auction under their own lock (an expiry's charge).
        held = self._held()
        given_budget = self._given_budget()
        target = self._pace_target()
        with self._lock:
            now = self._clock()
            offer = self._live(hotkey, now)
            if offer is None:
                return NO_ASK
            self._seen[hotkey] = now
            self._steer(now, held, given_budget)
            merit = self._merit_allowances(now, held, self._merit_target(target))
            if merit is not None and self._shadow:
                self._log_shadow(now, held, merit)
                merit = None
            if merit is not None:
                return self._admit_merit(hotkey, offer, now, held, *merit)
            if self._budget <= 0:
                return ALLOWANCE_FULL               # nothing to hand out now (an RL env whose next steps are full)
            given, marginal = self._allowances(now, held)
            allowance = given.get(hotkey, 0)
            if allowance <= 0:
                if self._aside.get(hotkey, -math.inf) <= now:
                    self._unmet_at = now
                return ASK_ABOVE_PRICE
            # Within its allowance, and within the budget: a dearer hotkey's leases out from before keep their place
            # until delivered, so the audit is never handed more than the budget.
            if held.get(hotkey, 0) + self._reserved[hotkey] >= allowance:
                return ALLOWANCE_FULL
            if sum(held.values()) + sum(self._reserved.values()) >= self._budget:
                self._unmet_at = now                # under its allowance, but the whole budget is out
                return ALLOWANCE_FULL
            self._reserved[hotkey] += 1
            # Paid the price of the moment, not its own offer: an offer under its true cost then only wins work
            # at moments that pay under that cost (2026-11-12: offers at 0 took the leases of a small budget and
            # were paid the period's price, set by the dear offers needed at other moments). Nothing delivered
            # lately: its own offer.
            moment = self._moment_price(now, marginal)
            return Terms(max(offer.price, moment if moment is not None else offer.price), hotkey,
                         offer=offer.price)

    def _merit_target(self, target: float | None) -> float | None:
        """The rate to fill: the market's pace, or for a job with a budget of its own (an RL env: the leases the
        trainer's next steps take) that budget at the median speed."""
        if self._budget_of is not None:
            median = self._speeds.median(self._clock()) or self._median_hint
            return None if median is None else self._budget * median
        return target

    def _log_shadow(self, now: float, held: Mapping[str, int], merit) -> None:
        """Once a minute, what the merit order would do against what the thermostat does."""
        if self._shadow_logged_at is not None and now - self._shadow_logged_at < la.SHADOW_LOG_SECONDS:
            return
        self._shadow_logged_at = now
        given, clearing = merit
        legacy, marginal = self._allowances(now, held)
        moment = self._moment_price(now, marginal)
        pace = self._pace.last if self._pace is not None else {}
        median = self._speeds.median(now)
        alpha = self.params.alpha_of
        logger.info(
            "merit order shadow job=%s target=%s audit=%s k=%.2f wait=%s busy=%s median_v=%s "
            "merit_leases=%d merit_hotkeys=%d merit_price=%s budget=%d held=%d legacy_hotkeys=%d live_price=%s "
            "reached=%s supply=%s tail=%s",
            self._job, _fmt(self._target), _fmt(pace.get("audit_rate")), self._pace.k if self._pace else 1.0,
            _fmt(pace.get("wait_seconds")), _fmt(pace.get("busy"), 2), _fmt(median, 2),
            sum(given.values()), len(given), _fmt(None if clearing is None else alpha(clearing), 4),
            self._budget, sum(held.values()), len(legacy),
            _fmt(None if moment is None and marginal is None else alpha(moment if moment is not None else marginal), 4),
            None if self._allocation is None else self._allocation.reached,
            _fmt(None if self._allocation is None else self._allocation.supply), self._tail())

    def _admit_merit(self, hotkey: str, offer: Offer, now: float, held: Mapping[str, int],
                     given: dict[str, int], clearing: float | None) -> Terms | str:
        """Granted iff ``hotkey`` holds fewer than its slots: no shared pool, no race. Leases of a hotkey cut
        since drain as they are delivered (its renewals are refused)."""
        allowance = given.get(hotkey, 0)
        if allowance <= 0:
            if self._aside.get(hotkey, -math.inf) <= now:
                self._unmet_at = now
            return ASK_ABOVE_PRICE
        if held.get(hotkey, 0) + self._reserved[hotkey] >= allowance:
            return ALLOWANCE_FULL
        self._reserved[hotkey] += 1
        return Terms(offer.price if clearing is None else max(offer.price, clearing), hotkey, offer=offer.price)

    def set_aside(self, hotkey: str, until: float) -> None:
        """``hotkey`` gets no allowance until ``until`` (unix seconds): its share goes to the offers behind it."""
        with self._lock:
            now = self._clock()
            self._aside = {h: t for h, t in self._aside.items() if t > now}
            if until > now:
                self._aside[hotkey] = float(until)

    def unreserve(self, terms: Terms) -> None:
        """The reservation ``admit`` made is over: its lease is granted (and counted as held) or not."""
        with self._lock:
            if self._reserved[terms.hotkey] > 0:
                self._reserved[terms.hotkey] -= 1
            if self._reserved[terms.hotkey] <= 0:
                del self._reserved[terms.hotkey]

    def lease_seconds(self, hotkey: str, default: float, held: int = 0) -> float:
        """How long ``hotkey``'s next lease lasts: twice its recent slow delivery (the DELIVERY_TAIL quantile of
        the last DELIVERIES_KEPT), within [floor, ``default``]. ``held``: its live leases, deliveries still to
        come and each at least as slow as now; while they are more than the tail's share, that quantile is not
        observed yet and the lease keeps ``default``."""
        with self._lock:
            times = self._deliveries.get(hotkey)
            if not times or len(times) < DELIVERIES_NEEDED:
                return float(default)
            # 2026-11-15: a miner that leased 188 slots at once delivered its shortest answers first (23-87 s);
            # three of them cut its next 466 leases to 300 s while its batch took 15-34 min.
            if held > (1.0 - DELIVERY_TAIL) * (len(times) + held):
                return float(default)
            # Its slow deliveries, not its median: a miner working in batches delivers a whole batch at once,
            # and a lease cut at twice its median expired every lease of a batch a little slower than usual.
            ordered = sorted(times)
            tail = ordered[math.ceil(DELIVERY_TAIL * len(ordered)) - 1]
            return min(float(default), max(LEASE_FLOOR_SECONDS, 2.0 * tail))

    def observe_delivery(self, hotkey: str, seconds: float, offer: float | None = None,
                         tokens: int | None = None) -> None:
        """A lease of ``hotkey`` was delivered ``seconds`` after its grant; ``offer``: its holder's own offer when
        granted (``Terms.offer``), the price this delivery counts at for the price of the moment. Never the lease's
        price: that is the price of the moment already, and would feed it back into itself."""
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            self._deliveries.setdefault(hotkey, collections.deque(maxlen=DELIVERIES_KEPT)).append(float(seconds))
            if tokens is not None:
                self._speeds.delivered(hotkey, self._clock(), float(tokens))
            if offer is not None and math.isfinite(float(offer)) and float(offer) > 0:
                self._market.record(self._job, self._clock(), hotkey, self.params.alpha_of(float(offer)))
            self._changed()

    def lease_ended(self, hotkey: str, price: float | None, held_fraction: float) -> None:
        """A lease ended without a submission (expired: ``held_fraction`` 1; released: the share of its time held)."""
        if price is None:
            return
        fraction = min(1.0, max(0.0, float(held_fraction)))
        with self._lock:
            owed = float(price) * self._mean_tokens * fraction
            if owed > 0:
                self._debt[hotkey] = self._debt.get(hotkey, 0.0) + owed
            if fraction >= LATE_RELEASE:
                self._missed.setdefault(hotkey, []).append(self._clock())
            self._changed()

    # -- submissions ---------------------------------------------------------

    def take_debit(self, hotkey: str, value: float) -> float:
        """Take up to ``value`` (pool share) of ``hotkey``'s debt against an accepted submission worth ``value``."""
        with self._lock:
            owed = self._debt.get(hotkey, 0.0)
            taken = min(owed, max(0.0, float(value)))
            if taken > 0:
                left = owed - taken
                if left > 1e-18:
                    self._debt[hotkey] = left
                else:
                    del self._debt[hotkey]
                self._changed()
            return taken

    def refund_debit(self, hotkey: str, amount: float) -> None:
        if amount > 0:
            with self._lock:
                self._debt[hotkey] = self._debt.get(hotkey, 0.0) + float(amount)
                self._changed()

    def observe_tokens(self, token_count: int) -> None:
        with self._lock:
            self._mean_tokens += TOKENS_EMA_WEIGHT * (float(token_count) - self._mean_tokens)

    # -- views ---------------------------------------------------------------

    def debt_alpha(self, hotkey: str) -> float:
        """``hotkey``'s debt in alpha (a pool share is a share of the miners' emission per period)."""
        with self._lock:
            return self._debt.get(hotkey, 0.0) * self.params.miner_alpha_per_period

    def view(self, hotkey: str | None = None) -> dict:
        """The public state: the price of the moment (until something is delivered, the dearest offer given work
        now: what a lease pays at most), the budget and the audit queue.
        Never another hotkey's offer; with ``hotkey``, that hotkey's own allowance and debt."""
        held = self._held()                   # before this lock, as in ``admit``
        given_budget = self._given_budget()
        target = self._pace_target()
        with self._lock:
            now = self._clock()
            self._steer(now, held, given_budget)
            merit = self._merit_allowances(now, held, self._merit_target(target))
            shadow = None
            if merit is not None and self._shadow:
                shadow, merit = merit, None
            if merit is not None:
                given, shown = merit
            else:
                given, marginal = self._allowances(now, held)
                moment = self._moment_price(now, marginal)
                shown = moment if moment is not None else marginal
            view = {
                "pricing": PRICING_AUCTION,
                "price_alpha_per_mtoken": None if shown is None else self.params.alpha_of(shown),
                "ceiling_alpha_per_mtoken": self.params.alpha_of(self.params.ceiling),
                "budget": self._budget,
                "leased": sum(held.values()),
                "audit_backlog": self._last_backlog,
                "miners_working": len(given),
            }
            if shadow is not None:
                view["merit_shadow"] = {
                    "leases_wanted": sum(shadow[0].values()), "hotkeys": len(shadow[0]),
                    "price_alpha_per_mtoken": None if shadow[1] is None else self.params.alpha_of(shadow[1]),
                    "target_tokens_per_s": self._target, "median_tokens_per_lease_s": self._speeds.median(now),
                    "audit_tokens_per_s": self._pace.last.get("audit_rate") if self._pace else None,
                    "audit_wait_s": self._pace.last.get("wait_seconds") if self._pace else None,
                    "k": self._pace.k if self._pace else None}
            if self._merit:
                median = self._speeds.median(now)
                view.update(allocator="merit" if merit is not None else "budget",
                            target_tokens_per_s=self._target,
                            leases_wanted=sum(given.values()),
                            median_tokens_per_lease_s=median,
                            tail=self._tail())
                if self._pace is not None:
                    view.update(audit_tokens_per_s=self._pace.last.get("audit_rate"),
                                audit_wait_s=self._pace.last.get("wait_seconds"), k=self._pace.k)
            if hotkey is not None:
                view.update(allowance=given.get(hotkey, 0), held=held.get(hotkey, 0),
                            debt_alpha=self._debt.get(hotkey, 0.0) * self.params.miner_alpha_per_period)
                if self._merit:
                    median = self._speeds.median(now)
                    offer = self._offers.get(hotkey)
                    view.update(tokens_per_lease_s=self._speeds.speed(hotkey, now),
                                v_min=None if median is None else la.vmin(median, self._tail()),
                                trusted_slots=None if offer is None else self._trusted.of(
                                    hotkey, self._capacity(hotkey, offer, now)))
            return view

    # -- state ---------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "offers": {h: [o.price, o.capacity, o.posted_at, o.swap] for h, o in self._offers.items()},
                "budget": self._budget,
                "debt": dict(self._debt),
                "missed": {h: list(t) for h, t in self._missed.items()},
                "deliveries": {h: list(d) for h, d in self._deliveries.items()},
                "delivered": [list(e) for e in self._market.of(self._job)],
                "mean_tokens": self._mean_tokens,
                "trusted": self._trusted.snapshot(),
                "median_speed": self._median_hint,
                "audit_share": self._q_hint,
                "audit_rate": None if self._pace is None else self._pace.rate,
            }

    def restore(self, state: Mapping) -> None:
        with self._lock:
            self._offers = {str(h): Offer(float(row[0]), int(row[1]), float(row[2]),
                                          bool(row[3]) if len(row) > 3 else False)
                            for h, row in (state.get("offers") or {}).items()}
            budget = int(state.get("budget") or self.params.initial_leases)
            self._budget = min(MAX_LEASES, max(MIN_LEASES, budget))
            self._debt = {str(h): float(v) for h, v in (state.get("debt") or {}).items() if float(v) > 0}
            self._missed = {str(h): [float(t) for t in ts] for h, ts in (state.get("missed") or {}).items()}
            self._deliveries = {str(h): collections.deque((float(s) for s in d), maxlen=DELIVERIES_KEPT)
                                for h, d in (state.get("deliveries") or {}).items()}
            delivered = state.get("delivered")
            if isinstance(delivered, Mapping):
                # The format of #540 (2026-11-16): per hotkey, (delivered at, offer in pool share per token).
                delivered = sorted([float(t), str(h), self.params.alpha_of(float(p))]
                                   for h, entries in delivered.items() for t, p in entries)
            if isinstance(delivered, list):
                self._market.replace(self._job, delivered)
            if state.get("mean_tokens") is not None:
                self._mean_tokens = float(state["mean_tokens"])
            if isinstance(state.get("trusted"), Mapping):
                self._trusted.restore(state["trusted"], self._clock())
            if state.get("median_speed") is not None:
                self._median_hint = float(state["median_speed"])
            if state.get("audit_share") is not None:
                self._q_hint = float(state["audit_share"])
            if self._pace is not None and state.get("audit_rate") is not None:
                self._pace.seed_rate(float(state["audit_rate"]))

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            self.restore(json.loads(self._state_path.read_text()))
        except Exception:
            logger.exception("token auction: state %s unreadable; starting empty", self._state_path)

    @property
    def state_path(self) -> Path | None:
        """Where this auction's state is written, if anywhere: its job's leases are written beside it."""
        return self._state_path

    def _changed(self) -> None:
        self._dirty = True
        self.save()

    def save(self, *, force: bool = False) -> None:
        """Write the state to ``state_path`` (at most every ``SAVE_SECONDS`` unless forced)."""
        if self._state_path is None:
            return
        with self._lock:
            now = time.monotonic()
            if not self._dirty or (not force and now - self._saved_at < SAVE_SECONDS):
                return
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.snapshot()))
                os.replace(tmp, self._state_path)
                self._saved_at = now
                self._dirty = False
            except OSError:
                logger.exception("token auction: state %s not written", self._state_path)


def _fmt(value, digits: int = 0) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def state_path_for(job_id: str) -> Path | None:
    """Where a job's auction state lives (``RELIQUARY_AUCTION_STATE_DIR``), or None to keep it in memory."""
    root = os.environ.get(STATE_DIR_ENV)
    if not root:
        return None
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(job_id))
    return Path(root) / f"{safe}.json"


def job_auction(params: Mapping | None, *, backlog: Callable[[], int | None] | None = None,
                clock: Callable[[], float] = time.time, mean_tokens: float = 1000.0,
                state_path: str | os.PathLike | None = None, job=None) -> TokenAuction | None:
    """The auction a job's registry params declare, or None (an invalid declaration is logged and not run: the
    registry refuses it, so this only guards an older registry read). ``job``: its spec, which names its market."""
    try:
        declared = auction_params(params)
    except ValueError:
        logger.exception("token auction: invalid pricing params; the job is not priced by auction")
        return None
    if declared is None:
        return None
    return TokenAuction(declared, clock=clock, backlog=backlog, mean_tokens=mean_tokens, state_path=state_path,
                        market_name=market_key(job), job_id=str(getattr(job, "job_id", None) or "job"))
