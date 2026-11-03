"""The token price auction of one draw job (docs/design/2026-11-03-token-price-auction-design.md).

A job priced by auction (registry ``pricing: "auction"``) pays every leased token the price of the round the lease
was granted in, instead of splitting the period's pool by tokens:

* **Asks.** Each hotkey posts one ask, the alpha per million verified tokens it works for (``post_ask``). An ask
  lives ``ASK_TTL_SECONDS`` unless reposted. No quantity, no capacity.
* **Rounds.** Rounds are aligned on ``round_seconds``. A round's price is fixed when it opens: the highest active
  ask at or under the target, so a miner knows its price when it gets a lease. A lease is granted only to a hotkey
  whose ask is at or under the round's price (``ask_above_price``), while the round still has leases
  (``round_full``).
* **Price.** When a round closes, the target moves one ask at a time from what happened:
  - leases ran out while admitted miners still asked for more: supply exceeds the work, so the target drops to
    the next lower ask and the most expensive miner leaves;
  - leases were left while a miner was refused for its price: the work needs more miners, so the target rises
    to the next higher ask, never past the ceiling;
  - otherwise it holds.
* **Leases per round** follow the audit queue when one is wired (``backlog``). More waiting than a round's leases
  cuts them; a saturated round with a short queue grows them. Without a queue they stay at ``initial_leases``.
* **Expiry.** A lease that ends without a submission charges its holder its price times the job's mean submission
  size: in full at expiry, by the share of its time held on a release. The debt is taken back from the hotkey's next
  accepted submissions (``take_debit``) and recorded with them, so the settlement pays ``tokens x price - debit``.

Prices are kept in pool share per token, the settlement's unit (``corpus_periods.max_price``); asks and the public
view are in alpha per million tokens, converted with the miner emission per period declared with the ceiling.

State is in memory, in the process that serves the job's routes. A restart drops asks, debts and the target price,
which restarts at the ceiling. Miners repost asks at least every ``ASK_REPOST_SECONDS``.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PRICING_PARAM = "pricing"
PRICING_AUCTION = "auction"
ROUND_SECONDS_PARAM = "auction_round_seconds"
INITIAL_LEASES_PARAM = "auction_initial_leases"
AUCTION_PARAMS = (PRICING_PARAM, ROUND_SECONDS_PARAM, INITIAL_LEASES_PARAM)

DEFAULT_ROUND_SECONDS = 600.0
MIN_ROUND_SECONDS = 60.0
MAX_ROUND_SECONDS = 86400.0
DEFAULT_INITIAL_LEASES = 64
MIN_LEASES = 4
MAX_LEASES = 100_000
# A round's leases: x CUT when the audit queue holds more than a round's worth, x GROW (at least +1) when the round
# ran out with the queue at most half a round.
LEASES_CUT = 0.7
LEASES_GROW = 1.25
ASK_TTL_SECONDS = 3600.0
ASK_REPOST_SECONDS = 600.0
# The largest ask accepted, in alpha per million tokens (a typo guard; the ceiling decides who works).
MAX_ASK_ALPHA_PER_MTOKEN = 1e9
HISTORY_ROUNDS = 144
# The mean submission size, in tokens, an expiry is charged at: an EMA of accepted submissions.
TOKENS_EMA_WEIGHT = 0.05

NO_ASK = "no_ask"
ASK_ABOVE_PRICE = "ask_above_price"
ROUND_FULL = "round_full"


@dataclass(frozen=True, slots=True)
class AuctionParams:
    ceiling: float                      # pool share per token
    miner_alpha_per_period: float
    round_seconds: float = DEFAULT_ROUND_SECONDS
    initial_leases: int = DEFAULT_INITIAL_LEASES

    def share_of(self, alpha_per_mtoken: float) -> float:
        return float(alpha_per_mtoken) / 1e6 / self.miner_alpha_per_period

    def alpha_of(self, share: float) -> float:
        return float(share) * 1e6 * self.miner_alpha_per_period


def auction_params(params: Mapping | None) -> AuctionParams | None:
    """The auction of a job's registry params; None when the job is not priced by auction. Raises ``ValueError``
    on an auction declared without a price ceiling or with bad round settings (``task_registry`` refuses them)."""
    from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM, max_price

    if not isinstance(params, Mapping) or params.get(PRICING_PARAM) is None:
        return None
    if params[PRICING_PARAM] != PRICING_AUCTION:
        raise ValueError(f"unknown pricing {params[PRICING_PARAM]!r}")
    ceiling = max_price(params)
    emission = params.get(MAX_PRICE_EMISSION_PARAM)
    if ceiling is None or emission is None:
        raise ValueError("an auction needs a price ceiling (max_price_per_token)")
    round_seconds = float(params.get(ROUND_SECONDS_PARAM, DEFAULT_ROUND_SECONDS))
    if not MIN_ROUND_SECONDS <= round_seconds <= MAX_ROUND_SECONDS:
        raise ValueError(f"{ROUND_SECONDS_PARAM} must be within [{MIN_ROUND_SECONDS:.0f}, "
                         f"{MAX_ROUND_SECONDS:.0f}], got {round_seconds}")
    initial = params.get(INITIAL_LEASES_PARAM, DEFAULT_INITIAL_LEASES)
    if float(initial) != int(float(initial)) or not MIN_LEASES <= int(float(initial)) <= MAX_LEASES:
        raise ValueError(f"{INITIAL_LEASES_PARAM} must be an integer within [{MIN_LEASES}, {MAX_LEASES}]")
    if not (math.isfinite(float(ceiling)) and float(ceiling) > 0 and float(emission) > 0):
        raise ValueError("the price ceiling and the miner emission must be positive")
    return AuctionParams(ceiling=float(ceiling), miner_alpha_per_period=float(emission),
                         round_seconds=round_seconds, initial_leases=int(float(initial)))


@dataclass(frozen=True, slots=True)
class Terms:
    """What a lease granted now pays: ``price`` per token (pool share), fixed for ``round``."""
    price: float
    round: int


class TokenAuction:
    def __init__(self, params: AuctionParams, *, clock: Callable[[], float] = time.time,
                 backlog: Callable[[], int | None] | None = None, mean_tokens: float = 1000.0) -> None:
        self.params = params
        self._clock = clock
        self._backlog = backlog
        self._lock = threading.RLock()
        self._asks: dict[str, tuple[float, float]] = {}       # hotkey -> (price share, posted at)
        self._target = params.ceiling
        self._round: int | None = None
        self._round_price = params.ceiling
        self._leases = int(params.initial_leases)
        self._granted = 0
        self._full = 0                                         # refusals for round_full
        self._priced_out = 0                                   # refusals for ask_above_price
        self._debt: dict[str, float] = {}
        self._mean_tokens = float(mean_tokens)
        self.history: collections.deque[dict] = collections.deque(maxlen=HISTORY_ROUNDS)

    # -- asks ----------------------------------------------------------------

    def post_ask(self, hotkey: str, alpha_per_mtoken: float) -> str | None:
        """Record ``hotkey``'s ask; a refusal string when the ask is not a positive number."""
        try:
            alpha = float(alpha_per_mtoken)
        except (TypeError, ValueError):
            return "invalid_ask"
        if not (math.isfinite(alpha) and 0 < alpha <= MAX_ASK_ALPHA_PER_MTOKEN):
            return "invalid_ask"
        with self._lock:
            self._roll()
            self._asks[hotkey] = (self.params.share_of(alpha), self._clock())
        return None

    def ask_of(self, hotkey: str) -> float | None:
        """``hotkey``'s live ask (pool share per token), or None."""
        with self._lock:
            return self._live_ask(hotkey, self._clock())

    def _live_ask(self, hotkey: str, now: float) -> float | None:
        held = self._asks.get(hotkey)
        if held is None:
            return None
        if now - held[1] > ASK_TTL_SECONDS:
            del self._asks[hotkey]
            return None
        return held[0]

    def _live_prices(self, now: float) -> list[float]:
        for hotkey in list(self._asks):
            self._live_ask(hotkey, now)
        return sorted({price for price, _ in self._asks.values()})

    # -- rounds --------------------------------------------------------------

    def _index(self, now: float) -> int:
        return math.floor(now / self.params.round_seconds)

    def _roll(self) -> None:
        now = self._clock()
        index = self._index(now)
        if self._round == index:
            return
        if self._round is not None:
            self._close(now)
        self._open(index, now)

    def _close(self, now: float) -> None:
        backlog = None
        if self._backlog is not None:
            try:
                backlog = self._backlog()
            except Exception:                     # a status read never stops leasing
                logger.exception("token auction: audit backlog unavailable")
        saturated = self._granted >= self._leases and self._full > 0
        prices = self._live_prices(now)
        if saturated:
            lower = [p for p in prices if p < self._round_price]
            if lower:
                self._target = lower[-1]
        elif self._granted < self._leases and self._priced_out > 0:
            higher = [p for p in prices if self._round_price < p <= self.params.ceiling]
            if higher:
                self._target = higher[0]
        leases = self._leases
        if backlog is not None and backlog > leases:
            self._leases = max(MIN_LEASES, int(leases * LEASES_CUT))
        elif saturated and backlog is not None and backlog <= leases / 2:
            self._leases = min(MAX_LEASES, max(leases + 1, int(leases * LEASES_GROW)))
        self.history.append({
            "round": self._round,
            "price_alpha_per_mtoken": self.params.alpha_of(self._round_price),
            "leases": leases, "granted": self._granted,
            "refused_round_full": self._full, "refused_ask_above_price": self._priced_out,
            "audit_backlog": backlog,
        })

    def _open(self, index: int, now: float) -> None:
        self._round = index
        under = [p for p in self._live_prices(now) if p <= self._target]
        # The highest ask the target admits: never pay past what the most expensive admitted miner asked for.
        self._round_price = under[-1] if under else self._target
        self._target = self._round_price
        self._granted = self._full = self._priced_out = 0

    # -- leases --------------------------------------------------------------

    def admit(self, hotkey: str) -> Terms | str:
        """Reserve one of this round's leases for ``hotkey``: its terms, or why it gets none. A reservation that
        found no slot is given back with ``unreserve``."""
        with self._lock:
            self._roll()
            ask = self._live_ask(hotkey, self._clock())
            if ask is None:
                return NO_ASK
            if ask > self._round_price:
                self._priced_out += 1
                return ASK_ABOVE_PRICE
            if self._granted >= self._leases:
                self._full += 1
                return ROUND_FULL
            self._granted += 1
            return Terms(self._round_price, self._round)

    def unreserve(self, terms: Terms) -> None:
        with self._lock:
            if terms.round == self._round and self._granted > 0:
                self._granted -= 1

    def lease_ended(self, hotkey: str, price: float | None, held_fraction: float) -> None:
        """A lease ended without a submission (expired: ``held_fraction`` 1; released: the share of its time held)."""
        if price is None:
            return
        fraction = min(1.0, max(0.0, float(held_fraction)))
        with self._lock:
            owed = float(price) * self._mean_tokens * fraction
            if owed > 0:
                self._debt[hotkey] = self._debt.get(hotkey, 0.0) + owed

    # -- submissions ---------------------------------------------------------

    def take_debit(self, hotkey: str, value: float) -> float:
        """Take up to ``value`` (pool share) of ``hotkey``'s debt against an accepted submission worth ``value``."""
        with self._lock:
            owed = self._debt.get(hotkey, 0.0)
            taken = min(owed, max(0.0, float(value)))
            if taken > 0:
                left = owed - taken
                if left > 0:
                    self._debt[hotkey] = left
                else:
                    del self._debt[hotkey]
            return taken

    def refund_debit(self, hotkey: str, amount: float) -> None:
        if amount > 0:
            with self._lock:
                self._debt[hotkey] = self._debt.get(hotkey, 0.0) + float(amount)

    def observe_tokens(self, token_count: int) -> None:
        with self._lock:
            self._mean_tokens += TOKENS_EMA_WEIGHT * (float(token_count) - self._mean_tokens)

    # -- views ---------------------------------------------------------------

    def debt_alpha(self, hotkey: str) -> float:
        """``hotkey``'s debt in alpha (a pool share is a share of the miners' emission per period)."""
        with self._lock:
            return self._debt.get(hotkey, 0.0) * self.params.miner_alpha_per_period

    def view(self) -> dict:
        """The public state: the round's price and leases, and the closed rounds. Never an ask or a hotkey."""
        with self._lock:
            self._roll()
            ends = (self._round + 1) * self.params.round_seconds
            return {
                "pricing": PRICING_AUCTION,
                "round": self._round,
                "round_seconds": self.params.round_seconds,
                "round_ends_at": ends,
                "price_alpha_per_mtoken": self.params.alpha_of(self._round_price),
                "ceiling_alpha_per_mtoken": self.params.alpha_of(self.params.ceiling),
                "leases": self._leases,
                "granted": self._granted,
                "rounds": list(self.history),
            }


def job_auction(params: Mapping | None, *, backlog: Callable[[], int | None] | None = None,
                clock: Callable[[], float] = time.time, mean_tokens: float = 1000.0) -> TokenAuction | None:
    """The auction a job's registry params declare, or None (an invalid declaration is logged and not run: the
    registry refuses it, so this only guards an older registry read)."""
    try:
        declared = auction_params(params)
    except ValueError:
        logger.exception("token auction: invalid pricing params; the job is not priced by auction")
        return None
    if declared is None:
        return None
    return TokenAuction(declared, clock=clock, backlog=backlog, mean_tokens=mean_tokens)
