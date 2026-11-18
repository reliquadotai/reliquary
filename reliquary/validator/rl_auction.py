"""The RL price auction (docs/design/2026-11-16-rl-auction.md): one ``corpus.token_auction.TokenAuction`` per env.

A task priced by auction (registry ``pricing: "auction"`` and a price ceiling, as an order) replaces the per-env
price multiplier (``rl_price``) and the per-hotkey quota (``rl_lease_quota``):

* **Offers per env.** A hotkey posts one offer per env (price per million matched tokens, capacity in groups in
  flight); envs differ in cost (an episode needs a sandbox), so each has its own market under the task's ceiling.
* **The budget keeps n checkpoints ahead.** An env's buffered groups should hold its next ``lag_target`` steps
  (``share x batch_groups x lag_target``). A lease yields a trainable group only when it comes back in zone, so
  the leases out are the groups missing divided by the env's measured yield: ``(target - buffered) / yield``. Every
  submitted lease frees its place at once, trained or not, so the leases keep turning over. No thermostat: the
  trainer's demand is known, and the yield is measured.
* **A lease's price** is the auction's: the higher of its holder's offer and the price of the moment (the offer
  under which 80 % of the env's recently delivered leases were won), never above the dearest offer given work.
* **A lease not delivered** (expired, forfeited, refused for integrity) charges its price times the env's mean
  matched tokens per trained group, taken from the holder's next accruals on the env. Our failures charge nothing.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.corpus.token_auction import (
    PRICING_AUCTION, PRICING_PARAM, AuctionParams, Terms, TokenAuction, auction_params, state_path_for,
)

LAG_TARGET_PARAM = "rl_auction_lag_target"
DEFAULT_LAG_TARGET = 2
MAX_LAG_TARGET = 16
# Before any trained group: the matched tokens an undelivered lease is charged at (``LedgerConfig`` default).
DEFAULT_GROUP_TOKENS = 16384
# The yield (share of an env's graded groups that train) the budget divides by: measured over the last
# YIELD_GROUPS, assumed PRIOR_YIELD until YIELD_MIN_GROUPS are graded, never under YIELD_FLOOR (a dry env would
# otherwise ask for unbounded leases; the trainer's max lag drops what waits too long anyway).
YIELD_GROUPS = 200
YIELD_MIN_GROUPS = 32
PRIOR_YIELD = 0.5
YIELD_FLOOR = 0.1


def rl_auction_declared(params: Mapping | None) -> bool:
    return isinstance(params, Mapping) and params.get(PRICING_PARAM) == PRICING_AUCTION


def lag_target(params: Mapping | None) -> int:
    value = (params or {}).get(LAG_TARGET_PARAM, DEFAULT_LAG_TARGET)
    if isinstance(value, bool) or float(value) != int(float(value)) or not 1 <= int(float(value)) <= MAX_LAG_TARGET:
        raise ValueError(f"{LAG_TARGET_PARAM} must be an integer within [1, {MAX_LAG_TARGET}]")
    return int(float(value))


def env_budget(share: float, batch_groups: int, lag: int, buffered: int, trainable: float = 1.0) -> int:
    """Leases an env may have out: the groups its next ``lag`` steps train, less those already buffered, divided
    by the share of its leases that come back trainable (``trainable``, at least ``YIELD_FLOOR``)."""
    missing = math.ceil(float(share) * int(batch_groups) * int(lag)) - int(buffered)
    if missing <= 0:
        return 0
    return math.ceil(missing / max(YIELD_FLOOR, min(1.0, float(trainable))))


def auction_settings(params: Mapping, targets: Mapping[str, float], *, batch_groups: int) -> dict | None:
    """The registry params an RL auction reads (pricing, the ceiling and its emission, the lag target), checked;
    None when the task is not priced by auction. ValueError on a bad declaration."""
    from reliquary.validator.corpus_periods import MAX_PRICE_PARAMS

    if not rl_auction_declared(params):
        if PRICING_PARAM in params:
            raise ValueError(f"unknown pricing {params[PRICING_PARAM]!r}")
        return None
    declared = auction_params(params)
    if declared is None:
        return None
    lag_target(params)
    if batch_groups < 1 or not targets:
        raise ValueError("an RL auction needs the trainer's batch and a target mix")
    return {k: params[k] for k in (PRICING_PARAM, LAG_TARGET_PARAM, *MAX_PRICE_PARAMS) if k in params}


class RlAuctions:
    """One auction per served env, its budget from the trainer's batch and its holdings from the lease book."""

    def __init__(self, params: AuctionParams, targets: Mapping[str, float], *, batch_groups: int, lag: int,
                 task_id: str, clock: Callable[[], float] = time.time,
                 mean_tokens: float = DEFAULT_GROUP_TOKENS, state_dir_job: Callable[[str], Any] = state_path_for):
        total = sum(float(v) for v in targets.values())
        if not targets or total <= 0:
            raise ValueError("an RL auction needs a positive target mix")
        self.params = params
        self.shares = {str(env): float(v) / total for env, v in targets.items()}
        self.batch_groups, self.lag = int(batch_groups), int(lag)
        self._book = None
        self.auctions: dict[str, TokenAuction] = {
            env: TokenAuction(params, clock=clock, mean_tokens=mean_tokens,
                              holdings=lambda env=env: self._holdings(env),
                              budget=lambda env=env: self.budget(env),
                              # Its own market: an RL env's cost (a sandbox, a context length) is not an order's.
                              market_name=f"rl|{task_id}|{env}", job_id=f"rl-{task_id}-{env}",
                              state_path=state_dir_job(f"rl-{task_id}-{env}"))
            for env in self.shares}

    @classmethod
    def from_params(cls, params: Mapping, targets: Mapping[str, float], *, batch_groups: int, task_id: str,
                    clock: Callable[[], float] = time.time) -> RlAuctions | None:
        """The auctions a task's registry params declare, or None (ValueError on a bad declaration)."""
        declared = auction_params(params)
        if declared is None:
            return None
        return cls(declared, targets, batch_groups=batch_groups, lag=lag_target(params), task_id=task_id,
                   clock=clock)

    def bind(self, book) -> None:
        """The lease book (its open leases and its buffered groups), once it is open."""
        self._book = book

    # -- reads of the lease book (the auction calls them before taking its own lock) --------------------------

    def _holdings(self, env: str) -> Mapping[str, int]:
        if self._book is None:
            return {}
        return self._book.open_by_hotkey(env)

    def buffered(self, env: str) -> int:
        if self._book is None:
            return 0
        with self._book.lock:
            return int(self._book._pending_by_env().get(env, 0))

    def trainable(self, env: str) -> float:
        """The share of the env's recent graded groups that train (in zone, not refused by the trainer)."""
        if self._book is None:
            return PRIOR_YIELD
        with self._book.lock:
            db = self._book.db
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rl_groups'").fetchone():
                return PRIOR_YIELD
            rows = db.execute("SELECT verdict, training FROM rl_groups WHERE environment=? AND verdict!='rejected' "
                              "ORDER BY seq DESC LIMIT ?", (env, YIELD_GROUPS)).fetchall()
        if len(rows) < YIELD_MIN_GROUPS:
            return PRIOR_YIELD
        return sum(1 for verdict, training in rows if verdict == "in_zone" and training != "rejected") / len(rows)

    def budget(self, env: str) -> int:
        return env_budget(self.shares[env], self.batch_groups, self.lag, self.buffered(env), self.trainable(env))

    # -- the lease book's calls -----------------------------------------------------------------------------

    def auction(self, env: str) -> TokenAuction:
        return self.auctions[env]

    def note_request(self, env: str, hotkey: str) -> None:
        if env in self.auctions:
            self.auctions[env].note_request(hotkey)

    def admit(self, env: str, hotkey: str) -> Terms | str:
        return self.auctions[env].admit(hotkey)

    def unreserve(self, env: str, terms: Terms) -> None:
        self.auctions[env].unreserve(terms)

    def lease_seconds(self, env: str, hotkey: str, default: float, held: int = 0) -> float:
        return self.auctions[env].lease_seconds(hotkey, default, held)

    def delivered(self, env: str, hotkey: str, seconds: float, offer: float | None) -> None:
        self.auctions[env].observe_delivery(hotkey, seconds, offer)

    def missed(self, env: str, hotkey: str, price: float | None) -> None:
        """An undelivered lease, charged in full (an RL lease has no partial use: a group is whole or nothing)."""
        self.auctions[env].lease_ended(hotkey, price, 1.0)

    # -- the ledger's calls ---------------------------------------------------------------------------------

    def trained_tokens(self, env: str, tokens: int) -> None:
        """A trained group's matched tokens: what an undelivered lease of the env is charged at."""
        if env in self.auctions and tokens > 0:
            self.auctions[env].observe_tokens(int(tokens))

    def take_debit(self, env: str, hotkey: str, value: float) -> float:
        if env not in self.auctions:
            return 0.0
        return self.auctions[env].take_debit(hotkey, value)

    # -- offers and views -----------------------------------------------------------------------------------

    def post_ask(self, env: str, hotkey: str, alpha_per_mtoken: float, capacity: int | None) -> str | None:
        if env not in self.auctions:
            return "environment_not_served"
        why = self.auctions[env].post_ask(hotkey, alpha_per_mtoken, capacity)
        if why is None:
            self.auctions[env].note_request(hotkey)   # an offer is posted to work: it counts as active at once
        return why

    def view(self, hotkey: str | None = None) -> dict:
        out = {}
        for env, auction in self.auctions.items():
            view = auction.view(hotkey)
            view.update(share=self.shares[env], buffered=self.buffered(env), trainable_share=self.trainable(env),
                        target_groups=math.ceil(self.shares[env] * self.batch_groups * self.lag))
            out[env] = view
        return {"lag_target": self.lag, "batch_groups": self.batch_groups, "environments": out}

    def allowance(self, hotkey: str) -> int:
        """The leases ``hotkey`` may hold now over every env (its allowances; held ones included). Asking counts
        as being active: a miner polls this before it requests, and an inactive hotkey is given no allowance."""
        for auction in self.auctions.values():
            auction.note_request(hotkey)
        return sum(int(auction.view(hotkey).get("allowance", 0)) for auction in self.auctions.values())

    def save(self) -> None:
        for auction in self.auctions.values():
            auction.save(force=True)


__all__ = ["DEFAULT_LAG_TARGET", "auction_settings", "LAG_TARGET_PARAM", "RlAuctions", "env_budget", "lag_target",
           "rl_auction_declared"]
