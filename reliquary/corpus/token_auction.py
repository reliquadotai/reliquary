"""The price auction of one draw job (docs/design/2026-11-10-order-price-auction-design.md).

A job priced by auction (registry ``pricing: "auction"``) hands its samples out cheapest first, as fast as its audit
verifies them, and pays every verified token of a period the same price:

* **Offers.** Each hotkey posts one offer (``post_ask``): the alpha per million verified tokens it works for, and the
  samples it can hold at once (its capacity, taken as declared). An offer lives ``ASK_TTL_SECONDS`` unless reposted.
* **Budget.** The samples leased at once across the job follow the audit queue (``backlog``), as a thermostat: under
  ``backlog_low`` records waiting, with the budget in use, it grows; over ``backlog_high`` it shrinks.
* **Allowances.** Walking the offers from the cheapest, each active hotkey gets up to its capacity until the budget
  is covered. A hotkey is granted a lease while it holds fewer than its allowance. A lease carries its holder's
  own offer; the settlement pays every token of a period at the highest offer delivered in it.
* **Leases not delivered.** A lease that ends without a submission charges its holder its price times the job's
  mean submission size (by the share of its time held on a release), taken back from its next accepted
  submissions (``take_debit``). An expiry, or a release past half its time, halves the holder's capacity for
  ``PENALTY_SECONDS``; a hotkey in debt holds one lease at most until it has paid.
* **Lease time** follows each hotkey's measured delivery time (``lease_seconds``).

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
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

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
# The largest ask accepted, in alpha per million tokens (a typo guard; the ceiling decides who works).
MAX_ASK_ALPHA_PER_MTOKEN = 1e9
MAX_CAPACITY = 10_000
# A missed lease halves its holder's capacity for this long.
PENALTY_SECONDS = 3600.0
# A release past this share of its lease time counts as a miss.
LATE_RELEASE = 0.5
# Lease time: twice the median of a hotkey's recent deliveries, once it has this many, within [floor, the job's].
DELIVERIES_KEPT = 20
DELIVERIES_NEEDED = 3
# The share of a hotkey's recent deliveries its lease time covers twice over (`lease_seconds`).
DELIVERY_TAIL = 0.9
LEASE_FLOOR_SECONDS = 300.0
# The mean submission size, in tokens, an unused lease is charged at: an EMA of accepted submissions.
TOKENS_EMA_WEIGHT = 0.05
SAVE_SECONDS = 5.0

NO_ASK = "no_ask"
ASK_ABOVE_PRICE = "ask_above_price"
ALLOWANCE_FULL = "allowance_full"


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
    """What a lease granted now pays: ``price`` per token (pool share), the price of the moment -- the dearest
    offer given work right now, never under its holder's own offer."""
    price: float
    hotkey: str
    round: int | None = None


@dataclass(frozen=True, slots=True)
class Offer:
    price: float                        # pool share per token
    capacity: int
    posted_at: float


class TokenAuction:
    def __init__(self, params: AuctionParams, *, clock: Callable[[], float] = time.time,
                 backlog: Callable[[], int | None] | None = None,
                 holdings: Callable[[], Mapping[str, int]] | None = None,
                 mean_tokens: float = 1000.0, state_path: str | os.PathLike | None = None) -> None:
        self.params = params
        self._clock = clock
        self._backlog = backlog
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
        self._load()

    def bind_holdings(self, holdings: Callable[[], Mapping[str, int]]) -> None:
        """The live leases per hotkey (``DrawLeases.counts``), once the job's leases exist."""
        with self._lock:
            self._holdings = holdings

    # -- offers --------------------------------------------------------------

    def post_ask(self, hotkey: str, alpha_per_mtoken: float, capacity: int | None = None) -> str | None:
        """Record ``hotkey``'s offer; a refusal string when it is not a positive price and capacity."""
        try:
            alpha = float(alpha_per_mtoken)
            held = 1 if capacity is None else int(capacity)
        except (TypeError, ValueError):
            return "invalid_ask"
        if not (math.isfinite(alpha) and 0 < alpha <= MAX_ASK_ALPHA_PER_MTOKEN and 1 <= held <= MAX_CAPACITY):
            return "invalid_ask"
        with self._lock:
            self._offers[hotkey] = Offer(self.params.share_of(alpha), held, self._clock())
            self._changed()
        return None

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

    def _steer(self, now: float, held: Mapping[str, int]) -> None:
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

    # -- leases --------------------------------------------------------------

    def admit(self, hotkey: str) -> Terms | str:
        """Reserve one lease for ``hotkey`` under its allowance: its terms, or why it gets none. Every reservation
        is given back with ``unreserve`` once its lease is granted or not."""
        # Read before taking this lock: the leases call this auction under their own lock (an expiry's charge).
        held = self._held()
        with self._lock:
            now = self._clock()
            offer = self._live(hotkey, now)
            if offer is None:
                return NO_ASK
            self._seen[hotkey] = now
            self._steer(now, held)
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
            # were paid the period's price, set by the dear offers needed at other moments).
            return Terms(max(offer.price, marginal if marginal is not None else offer.price), hotkey)

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

    def lease_seconds(self, hotkey: str, default: float) -> float:
        """How long ``hotkey``'s next lease lasts: twice its recent slow delivery (the DELIVERY_TAIL quantile of
        the last DELIVERIES_KEPT), within [floor, ``default``]."""
        with self._lock:
            times = self._deliveries.get(hotkey)
            if not times or len(times) < DELIVERIES_NEEDED:
                return float(default)
            # Its slow deliveries, not its median: a miner working in batches delivers a whole batch at once,
            # and a lease cut at twice its median expired every lease of a batch a little slower than usual.
            ordered = sorted(times)
            tail = ordered[math.ceil(DELIVERY_TAIL * len(ordered)) - 1]
            return min(float(default), max(LEASE_FLOOR_SECONDS, 2.0 * tail))

    def observe_delivery(self, hotkey: str, seconds: float) -> None:
        """A lease of ``hotkey`` was delivered ``seconds`` after its grant."""
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            self._deliveries.setdefault(hotkey, collections.deque(maxlen=DELIVERIES_KEPT)).append(float(seconds))
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
        """The public state: the price of the most expensive offer working now, the budget and the audit queue.
        Never another hotkey's offer; with ``hotkey``, that hotkey's own allowance and debt."""
        held = self._held()                   # before this lock, as in ``admit``
        with self._lock:
            now = self._clock()
            self._steer(now, held)
            given, marginal = self._allowances(now, held)
            view = {
                "pricing": PRICING_AUCTION,
                "price_alpha_per_mtoken": None if marginal is None else self.params.alpha_of(marginal),
                "ceiling_alpha_per_mtoken": self.params.alpha_of(self.params.ceiling),
                "budget": self._budget,
                "leased": sum(held.values()),
                "audit_backlog": self._last_backlog,
                "miners_working": len(given),
            }
            if hotkey is not None:
                view.update(allowance=given.get(hotkey, 0), held=held.get(hotkey, 0),
                            debt_alpha=self._debt.get(hotkey, 0.0) * self.params.miner_alpha_per_period)
            return view

    # -- state ---------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "offers": {h: [o.price, o.capacity, o.posted_at] for h, o in self._offers.items()},
                "budget": self._budget,
                "debt": dict(self._debt),
                "missed": {h: list(t) for h, t in self._missed.items()},
                "deliveries": {h: list(d) for h, d in self._deliveries.items()},
                "mean_tokens": self._mean_tokens,
            }

    def restore(self, state: Mapping) -> None:
        with self._lock:
            self._offers = {str(h): Offer(float(p), int(c), float(t))
                            for h, (p, c, t) in (state.get("offers") or {}).items()}
            budget = int(state.get("budget") or self.params.initial_leases)
            self._budget = min(MAX_LEASES, max(MIN_LEASES, budget))
            self._debt = {str(h): float(v) for h, v in (state.get("debt") or {}).items() if float(v) > 0}
            self._missed = {str(h): [float(t) for t in ts] for h, ts in (state.get("missed") or {}).items()}
            self._deliveries = {str(h): collections.deque((float(s) for s in d), maxlen=DELIVERIES_KEPT)
                                for h, d in (state.get("deliveries") or {}).items()}
            if state.get("mean_tokens") is not None:
                self._mean_tokens = float(state["mean_tokens"])

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


def state_path_for(job_id: str) -> Path | None:
    """Where a job's auction state lives (``RELIQUARY_AUCTION_STATE_DIR``), or None to keep it in memory."""
    root = os.environ.get(STATE_DIR_ENV)
    if not root:
        return None
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(job_id))
    return Path(root) / f"{safe}.json"


def job_auction(params: Mapping | None, *, backlog: Callable[[], int | None] | None = None,
                clock: Callable[[], float] = time.time, mean_tokens: float = 1000.0,
                state_path: str | os.PathLike | None = None) -> TokenAuction | None:
    """The auction a job's registry params declare, or None (an invalid declaration is logged and not run: the
    registry refuses it, so this only guards an older registry read)."""
    try:
        declared = auction_params(params)
    except ValueError:
        logger.exception("token auction: invalid pricing params; the job is not priced by auction")
        return None
    if declared is None:
        return None
    return TokenAuction(declared, clock=clock, backlog=backlog, mean_tokens=mean_tokens, state_path=state_path)
