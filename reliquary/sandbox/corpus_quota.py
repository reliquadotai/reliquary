"""The corpus signed-sandbox session quota: demand-driven, bounded by the fleet, fair and split-invariant.

A corpus signed job's sessions used to be capped by fixed per-hotkey constants (``SandboxPolicy``: 4 live per job,
8 live in total, 120 opens an hour), whatever the fleet's size. With this quota (``install``ed on the corpus session
book and passed to the issuer) a hotkey may hold as many sessions as it asks for, bounded only by the fleet's
capacity and the other miners' fair shares. It reuses the RL quota's arithmetic (``rl_capacity``: weighted
water-filling over entities, ``allocate``, ``potential``, ``sessions_of``, ``opens_for``) with a group size of one
session; what differs is where the standings come from (the session book itself, so a restart restores them with
the book) and the corpus thresholds below.

* CAPACITY ``C``: the session slots of the fresh, active, undrained machines serving a served signed job's env at its
  package (``Fleet.slots_serving``).
* DEMAND of a hotkey: its held sessions (``SessionBook._holds``) plus the ADMISSIBLE opens it asked for in the last
  ``DEMAND_TTL_S`` (recorded only once the book's caps and the prompt's slots passed: ``admit``) for prompts it does
  not hold yet, raised to what it last held, decaying linearly to zero over ``DEMAND_MEMORY_S`` (a turnover never
  shows an idle fleet; a hotkey that left frees its share within that).
* STANDING, from the hotkey's own sessions in the book (the last 24 h; the book is restored from the store at start):
  a session is decided USEFUL when it was submitted (and not handed back), WASTED when it closed unpaid (expired,
  box failed, budget exhausted, a failed open, a withdrawal) or lapsed, and NEUTRAL when it was aborted (machine-
  signed) or voided (a drain: ours). Fewer than ``MIN_DECIDED`` decided sessions: a NEWCOMER (unless already
  ``NEWCOMER_EXIT_DECIDED`` decided at or below the yield floor: LOW_YIELD). Otherwise LOW_YIELD at or below
  ``YIELD_FLOOR``, else MAIN. A MAIN hotkey with no useful session in the window is back among the newcomers.
* WEIGHT of a MAIN hotkey: its useful held session-seconds over the window (24 h, or the hotkey's age in the book
  when shorter, at least ``MIN_HORIZON_S``) per second of the window, times the yield multiplier (0 at the floor,
  1 at ``YIELD_FULL``): its concurrent useful sessions. Linear in the volume, so a hotkey split into k parts gets
  exactly the whole's share (``rl_capacity`` docstring).
* ENTITIES (``rl_capacity.allocate``, group size 1): the LOW_YIELD entity gets at most ``L = max(1, 3 % of C)``
  (a hard cap); the NEWCOMER entity is guaranteed ``N = max(1, 10 % of C)`` and takes spare capacity beyond it
  with the weight ``NEWCOMER_WEIGHT``; each MAIN hotkey gets ``min(demand, weighted share)`` and what capped ones
  leave is re-spread. Inside a pooled entity its members share it by equal-weight water-filling of their demands
  (each may hold the ceiling of its share, the entity never more than the floor of its allocation): splitting
  never buys more than the entity.
* ADMISSION (``admit``, under the issuer lock): a hotkey opens while its held sessions + 1 fit within its
  allocation (MAIN: ``floor``; pooled: its member share and the entity's room), and the fleet's held sessions + 1
  fit within ``C`` less the room the pooled entities were allocated and have not filled yet. A refusal is
  ``quota_live`` (its own allocation, or the fleet) or ``quota_pool`` (its pooled entity), 429 with ``Retry-After``
  and the hotkey's ``allowance`` in the detail.
* ALLOWANCE (what the miner is told: on every grant and in every quota refusal): its POTENTIAL, the sessions it
  would get if it asked for the whole fleet, the others' demands as measured (``rl_capacity.potential``; for a pooled
  hotkey, its member share of its entity's potential). The miner grows its live sessions to it.
* The per-hotkey rate caps scale with the allowance: opens per hour ``max(policy, opens_for(max(allowance,
  held)))``, the aborted and refused-open rates by ``max(1, allowance / policy.max_live_per_hotkey_job)``. The fixed
  live caps no longer bind: they become safety ceilings (``max_live_per_hotkey`` / ``max_live_per_hotkey_job``,
  by default the fleet's capacity, never below the policy's own). One live session per (hotkey, prompt) and the
  prompt's own slots stay as they were.

Every method is synchronous and called on the event loop (the issuer calls ``admit`` under its lock)."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from reliquary.sandbox.rl_capacity import (
    ALLOCATION_REFRESH_S,
    DEMAND_MEMORY_S,
    DEMAND_TTL_S,
    LOW_YIELD,
    MAIN,
    NEWCOMER,
    Allocation,
    YieldStats,
    allocate,
    opens_for,
    potential,
    sessions_of,
    water_fill,
)
from reliquary.sandbox.sessions import (
    ABORTED,
    CLOSED,
    HANDED_BACK,
    LAPSED,
    SUBMITTED,
    VOIDED,
    Refusal,
    SandboxPolicy,
    SessionBook,
)

# --- standing (corpus episodes are long: fewer decided sessions graduate a hotkey than in RL) ---
WINDOW_S = 86_400              # the sessions a standing reads (the book keeps a day and an hour of them)
YIELD_SESSIONS = 200           # the yield is over a hotkey's last this many decided sessions
MIN_DECIDED = 24               # below this many decided sessions a hotkey is a newcomer
NEWCOMER_EXIT_DECIDED = 8      # a newcomer with this many decided at or below the floor is low-yield at once
YIELD_FULL = 0.50              # multiplier 1 at and above (an agentic episode fails more often than an RL seed)
YIELD_FLOOR = 0.15             # at and below: the low-yield entity (the multiplier reaches 0 there)
MIN_HORIZON_S = 3_600          # the weight's window is the hotkey's age in the book, at least this

# --- entities ---
NEWCOMER_WEIGHT = 4.0          # the newcomer entity's weight for spare capacity: one legacy miner (4 live), in w units
GROUP = 1                      # a corpus session is its own group (rl_capacity's ``pool_seeds``)

USEFUL, WASTED, NEUTRAL = "useful", "wasted", "neutral"


def outcome(record: Any) -> str | None:
    """A session's verdict for its miner's standing: USEFUL, WASTED, NEUTRAL, or None while it still holds."""
    state = record.state
    if state == SUBMITTED:
        return NEUTRAL if record.closed_status == HANDED_BACK else USEFUL
    if state in (ABORTED, VOIDED):
        return NEUTRAL
    if state in (CLOSED, LAPSED):
        return WASTED
    return None


def multiplier(yield_ratio: float) -> float:
    if yield_ratio >= YIELD_FULL:
        return 1.0
    if yield_ratio <= YIELD_FLOOR:
        return 0.0
    return (yield_ratio - YIELD_FLOOR) / (YIELD_FULL - YIELD_FLOOR)


def standing(stats: YieldStats) -> str:
    if stats.decided < MIN_DECIDED:
        if stats.decided >= NEWCOMER_EXIT_DECIDED and stats.yield_ratio <= YIELD_FLOOR:
            return LOW_YIELD
        return NEWCOMER
    return LOW_YIELD if stats.yield_ratio <= YIELD_FLOOR else MAIN


def stats_of(records, now: int) -> tuple[YieldStats, float]:
    """(the hotkey's YieldStats over the window, its weight's horizon in seconds) from its session records."""
    decided: list[tuple[int, str, float]] = []
    first = None
    for record in records:
        if getattr(record, "job_id", None) is None:
            continue                                    # not a corpus session
        issued = int(record.issued_at)
        first = issued if first is None else min(first, issued)
        verdict = outcome(record)
        at = int(record.closed_at or issued)
        if verdict is None or verdict == NEUTRAL or at < now - WINDOW_S:
            continue
        held_s = 0.0
        if verdict == USEFUL:
            held_s = float(max(0, min(at, int(record.expires_at)) - issued))
        decided.append((at, verdict, held_s))
    decided.sort(reverse=True)
    last = decided[:YIELD_SESSIONS]
    useful = sum(1 for _at, verdict, _s in last if verdict == USEFUL)
    useful_24h = sum(1 for _at, verdict, _s in decided if verdict == USEFUL)
    useful_s = sum(held for _at, verdict, held in decided if verdict == USEFUL)
    horizon = float(WINDOW_S if first is None else min(WINDOW_S, max(MIN_HORIZON_S, now - first)))
    return YieldStats(useful=useful, decided=len(last), useful_24h=useful_24h, useful_s_24h=useful_s), horizon


def weight(stats: YieldStats, horizon_s: float) -> float:
    """``w`` of a MAIN hotkey: its concurrent useful sessions over its window x the yield multiplier."""
    if standing(stats) != MAIN:
        return 0.0
    return float(stats.useful_s_24h) / max(1.0, float(horizon_s)) * multiplier(stats.yield_ratio)


class CorpusQuota:
    """``capacity()``: C, the fleet's slots serving the signed jobs. ``max_live_per_hotkey`` /
    ``max_live_per_hotkey_job``: the safety ceilings (None or 0: the fleet's capacity, never below the policy's own
    caps)."""

    def __init__(self, *, book: SessionBook, capacity: Callable[[], int], policy: SandboxPolicy | None = None,
                 max_live_per_hotkey: int | None = None, max_live_per_hotkey_job: int | None = None,
                 refresh_s: float = ALLOCATION_REFRESH_S, clock: Callable[[], float] = time.time) -> None:
        self._book = book
        self._capacity_of = capacity
        self._policy = policy or book.policy
        self._ceiling_live = int(max_live_per_hotkey or 0)
        self._ceiling_job = int(max_live_per_hotkey_job or 0)
        self._refresh_s = float(refresh_s)
        self._clock = clock
        self._wants: dict[str, dict[str, float]] = {}       # hotkey -> {engagement: last admissible ask}
        self._memory: dict[str, tuple[float, float]] = {}   # hotkey -> (demand last held, when)
        self._alloc: Allocation | None = None
        self._at = -math.inf
        self._lanes: dict[str, str] = {}
        self._demands: dict[str, float] = {}                # the demands the allocation used
        self._claims: dict[str, tuple[float, float]] = {}   # MAIN hotkey -> (w, demand)
        self._members: dict[str, dict[str, float]] = {NEWCOMER: {}, LOW_YIELD: {}}
        self._shares: dict[str, float] = {}                 # pooled hotkey -> its member share
        self._weights: dict[str, float] = {}
        self._potentials: dict[str, int] = {}

    def install(self) -> None:
        """The session book's per-hotkey caps follow the quota from now on."""
        self._book.live_caps = self.ceilings
        self._book.opens_cap = self.opens_per_hour
        self._book.cap_scale = self.cap_scale

    # --- counting (from the book) ---
    def _held_by_hotkey(self, now: int) -> dict[str, int]:
        held: dict[str, int] = {}
        for record in self._book.records():
            if record.job_id is not None and SessionBook._holds(record, now):
                held[record.hotkey] = held.get(record.hotkey, 0) + 1
        return held

    def _held_engagements(self, hotkey: str, now: int) -> set[str]:
        return {r.engagement for r in self._book.of_hotkey(hotkey)
                if r.job_id is not None and SessionBook._holds(r, now)}

    def _asks(self, hotkey: str, now: int) -> int:
        """Its admissible asks of the last DEMAND_TTL_S for prompts it does not hold (an ask that opened counts as
        a held session, not twice)."""
        wants = self._wants.get(hotkey)
        if not wants:
            return 0
        for engagement in [e for e, at in wants.items() if now - at > DEMAND_TTL_S]:
            del wants[engagement]
        if not wants:
            del self._wants[hotkey]
            return 0
        held = self._held_engagements(hotkey, now)
        return sum(1 for engagement in wants if engagement not in held)

    def _remembered(self, hotkey: str, held: int, demand: float, now: int) -> float:
        """``demand`` raised to what the hotkey last held, decaying over DEMAND_MEMORY_S (``rl_quota``'s rule)."""
        memory = self._memory.get(hotkey)
        kept = 0.0 if memory is None else memory[0] * max(0.0, min(1.0, 1.0 - (now - memory[1]) / DEMAND_MEMORY_S))
        if held >= kept:
            if held > 0:
                self._memory[hotkey] = (float(held), float(now))
            else:
                self._memory.pop(hotkey, None)
            kept = float(held)
        elif kept < 1.0:
            self._memory.pop(hotkey, None)
            kept = 0.0
        return max(float(demand), kept)

    def _capacity(self) -> int:
        try:
            return max(0, int(self._capacity_of()))
        except Exception:
            return 0

    # --- the allocation ---
    def allocation(self, now: float | None = None, *, force: bool = False) -> Allocation:
        now = int(self._clock() if now is None else now)
        if self._alloc is not None and not force and now - self._at < self._refresh_s:
            return self._alloc
        capacity = self._capacity()
        held = self._held_by_hotkey(now)
        lanes: dict[str, str] = {}
        demands: dict[str, float] = {}
        claims: dict[str, tuple[float, float]] = {}
        members: dict[str, dict[str, float]] = {NEWCOMER: {}, LOW_YIELD: {}}
        weights: dict[str, float] = {}
        for hotkey in set(held) | set(self._wants) | set(self._memory):
            demand = self._remembered(hotkey, held.get(hotkey, 0), held.get(hotkey, 0) + self._asks(hotkey, now),
                                      now)
            if demand <= 0:
                continue
            stats, horizon = stats_of(self._book.of_hotkey(hotkey), now)
            lane = standing(stats)
            w = weight(stats, horizon)
            if lane == MAIN and w <= 0:
                lane = NEWCOMER                          # no useful session in the window: a newcomer again
            lanes[hotkey], demands[hotkey], weights[hotkey] = lane, demand, w
            if lane == MAIN:
                claims[hotkey] = (w, demand)
            else:
                members[lane][hotkey] = demand
        alloc = allocate(capacity, pool_seeds=GROUP, main=claims,
                         newcomer_demand=sum(members[NEWCOMER].values()),
                         low_yield_demand=sum(members[LOW_YIELD].values()), newcomer_weight=NEWCOMER_WEIGHT)
        shares: dict[str, float] = {}
        for lane, total in ((NEWCOMER, alloc.newcomer), (LOW_YIELD, alloc.low_yield)):
            shares.update(water_fill(total, {hotkey: (1.0, d) for hotkey, d in members[lane].items()}))
        self._alloc, self._at = alloc, now
        self._lanes, self._demands, self._claims, self._members = lanes, demands, claims, members
        self._shares, self._weights, self._potentials = shares, weights, {}
        return alloc

    def lane(self, hotkey: str) -> str:
        self.allocation()
        found = self._lanes.get(str(hotkey))
        if found is not None:
            return found
        stats, horizon = stats_of(self._book.of_hotkey(str(hotkey)), int(self._clock()))
        lane = standing(stats)
        return NEWCOMER if lane == MAIN and weight(stats, horizon) <= 0 else lane

    def allowance(self, hotkey: str) -> int:
        """The sessions the hotkey could hold if it asked for the whole fleet (what the miner is told)."""
        hotkey = str(hotkey)
        alloc = self.allocation()
        found = self._potentials.get(hotkey)
        if found is not None:
            return found
        capacity = int(alloc.capacity)
        lane = self.lane(hotkey)
        newcomers = {k: d for k, d in self._members[NEWCOMER].items() if k != hotkey}
        lows = {k: d for k, d in self._members[LOW_YIELD].items() if k != hotkey}
        if lane == MAIN:
            w = self._weights.get(hotkey)
            if w is None:
                stats, horizon = stats_of(self._book.of_hotkey(hotkey), int(self._clock()))
                w = weight(stats, horizon)
            found = sessions_of(potential(capacity, pool_seeds=GROUP, main=self._claims, hotkey=hotkey, weight_h=w,
                                          newcomer_demand=sum(newcomers.values()),
                                          low_yield_demand=sum(lows.values()), newcomer_weight=NEWCOMER_WEIGHT))
        else:
            others = newcomers if lane == NEWCOMER else lows
            claims = {k: v for k, v in self._claims.items() if k != hotkey}
            entity = allocate(capacity, pool_seeds=GROUP, main=claims,
                              newcomer_demand=sum(newcomers.values()) + (capacity if lane == NEWCOMER else 0),
                              low_yield_demand=sum(lows.values()) + (capacity if lane == LOW_YIELD else 0),
                              newcomer_weight=NEWCOMER_WEIGHT)
            total = entity.newcomer if lane == NEWCOMER else entity.low_yield
            share = water_fill(total, {**{k: (1.0, d) for k, d in others.items()},
                                       hotkey: (1.0, float(capacity))})[hotkey]
            found = min(sessions_of(total), math.ceil(share - 1e-9))
        found = max(0, int(found))
        self._potentials[hotkey] = found
        return found

    def _limit(self, hotkey: str, lane: str, alloc: Allocation) -> int:
        if lane == MAIN:
            return sessions_of(alloc.main.get(hotkey, 0.0))
        return math.ceil(self._shares.get(hotkey, 0.0) - 1e-9)

    def _pool_room(self, alloc: Allocation, held: dict[str, int], but: str | None) -> int:
        """The sessions the pooled entities (but ``but``) were allocated and do not hold yet: kept for them."""
        room = 0
        for lane, total in ((NEWCOMER, alloc.newcomer), (LOW_YIELD, alloc.low_yield)):
            if lane != but:
                room += max(0, sessions_of(total) - sum(held.get(k, 0) for k in self._members[lane]))
        return room

    # --- admission ---
    def admit(self, hotkey: str, now: int, *, engagement: str | None = None) -> Refusal | None:
        """None admits. The issuer calls it under its lock for an ADMISSIBLE open (the book's caps and the prompt's
        slots passed): the ask counts as demand for DEMAND_TTL_S."""
        hotkey, now = str(hotkey), int(now)
        if engagement is not None:
            self._wants.setdefault(hotkey, {})[str(engagement)] = float(now)
        held = self._held_by_hotkey(now)
        mine = held.get(hotkey, 0)
        demand = mine + self._asks(hotkey, now)
        # a hotkey asking for more than the allocation knew of is judged on a fresh one (never a stale refusal)
        self.allocation(now, force=demand > self._demands.get(hotkey, 0.0))
        alloc = self._alloc
        retry = self._policy.retry_after_s
        if alloc is None or alloc.capacity <= 0:
            return Refusal("quota_live", {"why": "no fleet capacity known yet", "allowance": 0}, retry_after=retry)
        lane = self._lanes.get(hotkey) or self.lane(hotkey)
        detail = {"standing": lane, "live": mine, "allowance": self.allowance(hotkey),
                  "fleet_sessions": int(alloc.capacity)}
        limit = self._limit(hotkey, lane, alloc)
        if mine + 1 > limit:
            return Refusal("quota_live", {**detail, "max": limit}, retry_after=retry)
        fleet = sum(held.values())
        kept = self._pool_room(alloc, held, None if lane == MAIN else lane)
        if fleet + 1 > alloc.capacity - kept:
            return Refusal("quota_live", {**detail, "why": "fleet", "fleet_held": fleet, "kept_for_pools": kept},
                           retry_after=retry)
        if lane != MAIN:
            total = sessions_of(alloc.newcomer if lane == NEWCOMER else alloc.low_yield)
            pooled = sum(held.get(k, 0) for k in self._members[lane])
            if pooled + 1 > total:
                return Refusal("quota_pool", {**detail, "pool": lane, "pool_held": pooled, "max": total},
                               retry_after=retry)
        return None

    # --- the session book's caps ---
    def ceilings(self, hotkey: str) -> tuple[int, int]:
        """(live sessions in total, live per job): safety ceilings only, the allocation binds below them."""
        fleet = self._capacity()
        live = self._ceiling_live or max(fleet, self._policy.max_live_per_hotkey)
        job = self._ceiling_job or max(fleet, self._policy.max_live_per_hotkey_job)
        return int(live), int(job)

    def opens_per_hour(self, hotkey: str) -> int:
        held = self._held_by_hotkey(int(self._clock())).get(str(hotkey), 0)
        return max(self._policy.max_opens_per_hour, opens_for(max(self.allowance(hotkey), held)))

    def cap_scale(self, hotkey: str) -> float:
        """The aborted and refused-open rates scale with the allowance (the policy's are sized for its per-job cap)."""
        return max(1.0, self.allowance(hotkey) / max(1, self._policy.max_live_per_hotkey_job))

    def view(self) -> dict:
        """The operator's view: the fleet size, each entity's allocation and each hotkey's standing and demand."""
        alloc = self.allocation()
        return {"capacity": int(alloc.capacity), "newcomer": round(alloc.newcomer, 2),
                "low_yield": round(alloc.low_yield, 2),
                "hotkeys": {k: {"standing": lane, "demand": round(self._demands.get(k, 0.0), 2),
                                "allocation": round(alloc.main.get(k, self._shares.get(k, 0.0)), 2)}
                            for k, lane in self._lanes.items()}}


__all__ = ["CorpusQuota", "multiplier", "outcome", "standing", "stats_of", "weight"]
