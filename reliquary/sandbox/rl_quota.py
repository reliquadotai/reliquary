"""The allocation of RL sandbox sessions per hotkey, split-invariant: the ``SessionQuota`` hook of the RL
engagements.

Every ``refresh_s`` at most (and at once when the ledger's snapshot or the fleet's size changes), the capacity is
allocated by weighted water-filling over entities (``rl_capacity.allocate``): each MAIN hotkey with its weight and its
CURRENT demand, the NEWCOMER entity and the LOW_YIELD entity. A hotkey's demand is its counted sessions plus the
ADMISSIBLE open requests it made in the last ``DEMAND_TTL_S`` (the caller records a request only after the seed and
the session book's checks passed).

* What the MINER is told is its POTENTIAL (``rl_capacity.potential``): its weighted share with its own demand
  unbounded, the others' as measured. Admission uses the allocation (its current demand).
* GROUPS ARE NEVER CUT: a seed of a started precommit (one with a counted session) is admitted
  until the precommit holds M counted sessions, whatever the allocation, the entity or the standing became since. A
  NEW group starts only when the hotkey's committed sessions (each started group counts at least M) plus M fit in its
  allocation (MAIN) or its member maximum and its entity (pooled): a hotkey whose allocation is below M never opens a
  session that cannot form a group.
* STANDINGS ARE MONOTONE: a hotkey that leaves the newcomer or low-yield entity for MAIN stays in it
  until its MAIN potential reaches what it could hold there.
* Pooled members: at most ``NEWCOMER_MAX_GROUPS`` / ``LOW_YIELD_MAX_GROUPS`` groups each, first opens FIFO by group.

A session COUNTS while it holds (``SessionBook._holds``) and, while ``live``, only if a machine report listed it
(``reported``) or it was issued less than ``PRESENT_WINDOW_S`` ago (the gateway refuses it later).

Counting never scans the session book: the quota is a ``SessionEvents`` listener."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from reliquary.sandbox.rl_capacity import (
    ALLOCATION_REFRESH_S,
    ASK_FORGET_S,
    DEMAND_TTL_S,
    LOW_YIELD,
    LOW_YIELD_MAX_GROUPS,
    LOW_YIELD_OPENS_PERIOD_S,
    MAIN,
    NEWCOMER,
    NEWCOMER_MAX_GROUPS,
    NEWCOMER_TAKES_SPARE,
    OPENS_PERIOD_S,
    PRESENT_WINDOW_S,
    Allocation,
    YieldStats,
    allocate,
    opens_for,
    potential,
    sessions_of,
    standing,
    weight,
)
from reliquary.sandbox.sessions import (
    ABORTED,
    CLOSED_GRADED,
    LAPSED,
    LIVE,
    VOIDED,
    Refusal,
    SessionBook,
)

_LANE_GROUPS = {NEWCOMER: NEWCOMER_MAX_GROUPS, LOW_YIELD: LOW_YIELD_MAX_GROUPS}
_LANE_PERIOD = {MAIN: OPENS_PERIOD_S, NEWCOMER: OPENS_PERIOD_S, LOW_YIELD: LOW_YIELD_OPENS_PERIOD_S}
_KEEP_S = LOW_YIELD_OPENS_PERIOD_S
_LATE_GRADED_S = 60          # a lapsed graded session may still be claimed by a group received in time


def _precommit_of(record: Any) -> str | None:
    engagement = getattr(record, "engagement", None)
    if not isinstance(engagement, str) or not engagement.startswith("rl:"):
        return None
    from reliquary.protocol.service_episode import EpisodeWireError, parse_rl_engagement

    try:
        return parse_rl_engagement(engagement)[1]
    except EpisodeWireError:
        return None


class YieldQuota:
    def __init__(self, *, ledger: Any, pool_seeds: int, group_size: int, capacity: Callable[[], int],
                 reported: Callable[[str], bool] | None = None, takes_spare: bool = NEWCOMER_TAKES_SPARE,
                 refresh_s: float = ALLOCATION_REFRESH_S, retry_after_s: int = 10,
                 group_sizes: Mapping[str, int] | None = None, clock: Callable[[], float] = time.time) -> None:
        self._ledger = ledger
        self._pool_seeds, self._group_size = int(pool_seeds), int(group_size)
        self._group_sizes = dict(group_sizes or {})       # sandbox env -> M (its group size)
        self._capacity_of = capacity
        self._reported = reported
        self._takes_spare = bool(takes_spare)
        self._refresh_s = float(refresh_s)
        self._retry_after_s = int(retry_after_s)
        self._clock = clock
        self._index: dict[str, dict[str, Any]] = {}              # hotkey -> {session_id: record}
        self._sha: dict[str, str] = {}                           # session_id -> precommit (parsed once)
        self._wants: dict[str, dict[tuple[str, int], float]] = {}   # hotkey -> {(precommit, seed): last try}
        self._asks: dict[str, dict[str, tuple[str, float, float]]] = {NEWCOMER: {}, LOW_YIELD: {}}
        self._lane_opened: dict[str, list[Any]] = {NEWCOMER: [], LOW_YIELD: []}
        self._lane_counted: dict[str, int] = {NEWCOMER: 0, LOW_YIELD: 0}
        self._previous: dict[str, str] = {}                      # hotkey -> the pooled entity it was last in
        self._capacity_cache: tuple[float, int] | None = None
        self._alloc: Allocation | None = None
        self._claims: dict[str, tuple[float, float]] = {}
        self._entity_demand = {NEWCOMER: 0.0, LOW_YIELD: 0.0}
        self._lanes: dict[str, str] = {}
        self._potentials: dict[str, int] = {}
        self._alloc_key: tuple | None = None
        self._alloc_at = -math.inf
        self._dirty = True

    # --- the index (SessionEvents) ---
    def opened(self, record: Any) -> None:
        sha = _precommit_of(record)
        if sha is None:
            return
        hotkey = str(record.hotkey)
        self._index.setdefault(hotkey, {})[record.session_id] = record
        self._sha[record.session_id] = sha
        wants = self._wants.get(hotkey)
        if wants is not None:
            wants.pop((sha, self._seed_of(record)), None)
        lane = self.lane(hotkey)
        if lane != MAIN:
            self._lane_opened[lane].append(record)
            self._lane_counted[lane] += 1            # until the next refresh recounts
        self._dirty = True

    def closed(self, record: Any, final: Any) -> None:
        self._dirty = True                           # the record is updated in place by the book

    def rebuild(self, records: Iterable[Any]) -> None:
        now = int(self._clock())
        for record in records:
            self.opened(record)
        for lane in self._lane_opened:
            self._lane_opened[lane] = [r for r in self._lane_opened[lane] if r.issued_at > now - _LANE_PERIOD[lane]]
        self._dirty, self._alloc_key = True, None

    @staticmethod
    def _seed_of(record: Any) -> int:
        try:
            return int(str(record.engagement).rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return -1

    def _records(self, hotkey: str, now: int) -> list[Any]:
        records = self._index.get(hotkey)
        if not records:
            return []
        for session_id in [sid for sid, r in records.items()
                           if r.issued_at <= now - _KEEP_S and not SessionBook._holds(r, now)]:
            del records[session_id]
            self._sha.pop(session_id, None)
        if not records:
            del self._index[hotkey]
            return []
        return list(records.values())

    def _counts(self, record: Any, now: int) -> bool:
        if not SessionBook._holds(record, now):
            return False
        if record.state != LIVE or now - int(record.issued_at) <= PRESENT_WINDOW_S:
            return True
        return bool(self._reported(record.session_id)) if self._reported is not None else True

    def _counted(self, hotkey: str, now: int) -> list[Any]:
        return [r for r in self._records(hotkey, now) if self._counts(r, now)]

    def _demand(self, hotkey: str, now: int) -> int:
        wants = self._wants.get(hotkey)
        if wants:
            for key in [k for k, at in wants.items() if now - at > DEMAND_TTL_S]:
                del wants[key]
            if not wants:
                del self._wants[hotkey]
        return len(self._counted(hotkey, now)) + len(self._wants.get(hotkey, ()))

    def _m(self, records: Iterable[Any]) -> int:
        for record in records:
            return int(self._group_sizes.get(getattr(record, "env", None), self._group_size))
        return self._group_size

    # --- standings and the allocation ---
    def _capacity(self, now: float) -> int:
        cached = self._capacity_cache
        if cached is None or now - cached[0] >= 1.0:          # a directory scan: at most once a second
            cached = self._capacity_cache = (now, int(self._capacity_of()))
        return cached[1]

    def _stats(self, hotkey: str) -> YieldStats:
        return self._ledger.snapshot().get(str(hotkey), YieldStats())

    def _stats_lane(self, hotkey: str) -> str:
        stats = self._stats(hotkey)
        lane = standing(stats, pool_seeds=self._pool_seeds)
        if lane == MAIN and weight(stats, pool_seeds=self._pool_seeds) <= 0:
            return NEWCOMER                       # no useful session in 24 h: back among the newcomers
        return lane

    def lane(self, hotkey: str) -> str:
        """MAIN, NEWCOMER or LOW_YIELD: the entity the hotkey belongs to now (with the graduation grace)."""
        hotkey = str(hotkey)
        if self._capacity(self._clock()) > 0:
            self.allocation()                     # the standings as of the current allocation
        known = self._lanes.get(hotkey)
        if known is not None:
            return known
        lane = self._stats_lane(hotkey)
        previous = self._previous.get(hotkey)
        return previous if lane == MAIN and previous is not None else lane

    def allocation(self, now: float | None = None) -> Allocation:
        now = int(self._clock() if now is None else now)
        key = (getattr(self._ledger, "version", None), self._capacity(now))
        stale = key != self._alloc_key or (self._dirty and now - self._alloc_at >= self._refresh_s)
        if self._alloc is None or stale:
            group = self._pool_seeds
            lanes: dict[str, str] = {}
            demands: dict[str, int] = {}
            main: dict[str, tuple[float, float]] = {}
            graduates: list[str] = []
            for hotkey in set(self._index) | set(self._wants):
                lane = self._stats_lane(hotkey)
                demands[hotkey] = self._demand(hotkey, now)
                if lane == MAIN and hotkey in self._previous:
                    graduates.append(hotkey)
                    continue
                lanes[hotkey] = lane
                if lane == MAIN:
                    main[hotkey] = (weight(self._stats(hotkey), pool_seeds=group), float(demands[hotkey]))
            entity = {NEWCOMER: 0.0, LOW_YIELD: 0.0}
            for hotkey, lane in lanes.items():
                if lane != MAIN:
                    entity[lane] += min(demands[hotkey], _LANE_GROUPS[lane] * group)
            for hotkey in graduates:              # leave a pool only when MAIN gives at least as much
                previous = self._previous[hotkey]
                w = weight(self._stats(hotkey), pool_seeds=group)
                could = potential(key[1], pool_seeds=group, main=main, hotkey=hotkey, weight_h=w,
                                  newcomer_demand=entity[NEWCOMER], low_yield_demand=entity[LOW_YIELD],
                                  takes_spare=self._takes_spare)
                if sessions_of(could) >= _LANE_GROUPS[previous] * group:
                    lanes[hotkey] = MAIN
                    main[hotkey] = (w, float(demands[hotkey]))
                else:
                    lanes[hotkey] = previous
                    entity[previous] += min(demands[hotkey], _LANE_GROUPS[previous] * group)
            counted = {NEWCOMER: 0, LOW_YIELD: 0}
            for hotkey, lane in lanes.items():
                if lane != MAIN:
                    counted[lane] += len(self._counted(hotkey, now))
                    self._previous[hotkey] = lane
                else:
                    self._previous.pop(hotkey, None)
            self._alloc = allocate(key[1], pool_seeds=group, main=main, newcomer_demand=entity[NEWCOMER],
                                   low_yield_demand=entity[LOW_YIELD], takes_spare=self._takes_spare)
            self._claims, self._entity_demand = main, entity
            self._lanes, self._lane_counted, self._potentials = lanes, counted, {}
            self._alloc_key, self._alloc_at, self._dirty = key, now, False
        return self._alloc

    def potential_sessions(self, hotkey: str) -> int:
        """The sessions the hotkey could hold if it asked for them (what the miner is told). A pooled
        hotkey: its member maximum."""
        alloc = self.allocation()
        hotkey = str(hotkey)
        lane = self.lane(hotkey)
        if lane != MAIN:
            return _LANE_GROUPS[lane] * self._pool_seeds
        found = self._potentials.get(hotkey)
        if found is None:
            found = self._potentials[hotkey] = sessions_of(potential(
                alloc.capacity, pool_seeds=self._pool_seeds, main=self._claims, hotkey=hotkey,
                weight_h=weight(self._stats(hotkey), pool_seeds=self._pool_seeds),
                newcomer_demand=self._entity_demand[NEWCOMER], low_yield_demand=self._entity_demand[LOW_YIELD],
                takes_spare=self._takes_spare))
        return found

    def _entity_sessions(self, lane: str, alloc: Allocation) -> int:
        return sessions_of(alloc.newcomer if lane == NEWCOMER else alloc.low_yield)

    # --- the quota ---
    def _opens_refusal(self, reason: str, opened: list[int], limit: int, period: int, now: int,
                       detail: dict) -> Refusal | None:
        opens = sorted(opened)
        if len(opens) < limit:
            return None
        retry = max(1, opens[len(opens) - limit] + period - now) if limit and opens else self._retry_after_s
        return Refusal(reason, {"opens": len(opens), "max": limit, "period_s": period, **detail}, retry_after=retry)

    def refusal(self, hotkey: str, now: int, *, precommit_sha256: str | None = None, seed_index: int | None = None,
                take: bool = False) -> Refusal | None:
        """None admits. The caller passes ``seed_index`` only for an ADMISSIBLE request (its seed free, the session
        book's caps passed): only those count as demand."""
        hotkey, now = str(hotkey), int(now)
        if precommit_sha256 is not None and seed_index is not None:
            self._wants.setdefault(hotkey, {})[(precommit_sha256, int(seed_index))] = float(now)
            self._dirty = True
        if self._capacity(now) <= 0:
            return Refusal("quota_live", {"why": "no fleet capacity known yet"}, retry_after=self._retry_after_s)
        alloc = self.allocation(now)
        lane = self.lane(hotkey)
        stats = self._stats(hotkey)
        detail = {"lane": lane, "yield": round(stats.yield_ratio, 3), "useful_24h": stats.useful_24h}
        records = self._records(hotkey, now)
        counted = [r for r in records if self._counts(r, now)]
        groups: dict[str, list[Any]] = {}
        for record in counted:
            groups.setdefault(self._sha.get(record.session_id), []).append(record)
        started = groups.get(precommit_sha256) if precommit_sha256 is not None else None
        if started and len(started) < self._m(started):
            return None                    # a group in progress is never cut
        starts = precommit_sha256 is not None and not started
        committed = sum(max(len(members), self._m(members)) for members in groups.values())
        m_new = self._group_size
        period = _LANE_PERIOD[lane]
        mine = [r.issued_at for r in records if r.issued_at > now - period and r.state not in (ABORTED, VOIDED)]
        if lane == MAIN:
            limit = sessions_of(alloc.main.get(hotkey, 0.0))
            refused = self._opens_refusal("quota_opens", mine, opens_for(max(self.potential_sessions(hotkey),
                                                                             len(counted))), period, now, detail)
            if refused is not None:
                return refused
            if (starts and committed + m_new > limit) or (not starts and len(counted) >= limit):
                return Refusal("quota_live", {"live": len(counted), "committed": committed, "max": limit, **detail},
                               retry_after=self._retry_after_s)
            return None
        member_groups = _LANE_GROUPS[lane]
        member_max = member_groups * self._pool_seeds
        refused = self._opens_refusal("quota_opens", mine, opens_for(member_max), period, now, detail)
        if refused is not None:
            return refused
        entity = self._entity_sessions(lane, alloc)
        self._lane_opened[lane] = [r for r in self._lane_opened[lane] if r.issued_at > now - period]
        pooled = [r.issued_at for r in self._lane_opened[lane] if r.state not in (ABORTED, VOIDED)]
        guarantee = alloc.newcomer_guarantee if lane == NEWCOMER else alloc.low_yield_cap
        refused = self._opens_refusal("quota_pool_opens", pooled, opens_for(max(entity, guarantee)), period, now,
                                      detail)
        if refused is not None:
            return refused
        if starts and (len(groups) >= member_groups or committed + m_new > member_max):
            return Refusal("quota_live", {"groups": len(groups), "max": member_groups, **detail},
                           retry_after=self._retry_after_s)
        if not starts and len(counted) >= member_max:
            return Refusal("quota_live", {"live": len(counted), "max": member_max, **detail},
                           retry_after=self._retry_after_s)
        asks = self._asks[lane]
        for sha in [sha for sha, (_h, _first, last) in asks.items() if now - last > ASK_FORGET_S]:
            del asks[sha]
        first = asks.get(precommit_sha256, (hotkey, float(now), float(now)))[1] if starts else float(now)
        room = entity - self._lane_counted[lane]
        if room < (m_new if starts else 1):
            if starts:
                asks[precommit_sha256] = (hotkey, first, float(now))      # it waits its turn, as a group
            return Refusal("quota_pool", {"pool": lane, "live": self._lane_counted[lane], "max": entity, **detail},
                           retry_after=self._retry_after_s)
        if not starts:
            return None                                          # over-generation of a group it runs
        older = sum(1 for sha, (_h, at, _last) in asks.items() if sha != precommit_sha256 and at < first)
        if older:                                                # FIFO by group, never one turn per hotkey
            asks[precommit_sha256] = (hotkey, first, float(now))
            return Refusal("quota_pool", {"pool": lane, "ahead": older, **detail}, retry_after=self._retry_after_s)
        if take:
            asks.pop(precommit_sha256, None)
        return None

    def headroom(self, hotkey: str, now: int) -> int:
        """How many more sessions the hotkey may hold now (the open queue's bound per key)."""
        now = int(now)
        alloc, lane = self.allocation(now), self.lane(hotkey)
        counted = len(self._counted(str(hotkey), now))
        if lane == MAIN:
            return max(0, sessions_of(alloc.main.get(str(hotkey), 0.0)) - counted)
        member = _LANE_GROUPS[lane] * self._pool_seeds - counted
        return max(0, min(member, self._entity_sessions(lane, alloc) - self._lane_counted[lane]))

    async def admit_open(self, hotkey: str, *, environment: str, precommit_sha256: str,
                         seed_index: int | None = None) -> Refusal | None:
        """The pre-check before the issuer lock; it records an admissible request as demand."""
        return self.refusal(hotkey, int(self._clock()), precommit_sha256=precommit_sha256, seed_index=seed_index)

    def admit_locked(self, hotkey: str, now: int, *, precommit_sha256: str | None = None,
                     seed_index: int | None = None) -> Refusal | None:
        """The same check under the issuer lock; a first open of a group leaves the FIFO."""
        return self.refusal(hotkey, int(now), precommit_sha256=precommit_sha256, seed_index=seed_index, take=True)

    def rank(self, hotkey: str) -> tuple[str, float, bool]:
        """The open queue's (key, score, pooled)."""
        if self.lane(hotkey) == MAIN:
            return str(hotkey), weight(self._stats(hotkey), pool_seeds=self._pool_seeds), False
        return str(hotkey), 0.0, True

    def sessions_allowed(self, hotkey: str) -> int:
        """What the miner is told (Task 16b): its potential."""
        return self.potential_sessions(hotkey)

    def groups_allowed(self, hotkey: str) -> int:
        """Groups the hotkey may hold at once (at least one): the cap on capacity verdicts per window."""
        return max(1, math.ceil(self.potential_sessions(hotkey) / self._pool_seeds))

    def cap_scale(self, hotkey: str) -> float:
        """How much a per-hotkey rate constant of the session policy scales for this hotkey."""
        return max(1.0, self.potential_sessions(hotkey) / self._pool_seeds)

    def groups_formable(self, hotkey: str, now: int) -> int:
        """Precommits of the hotkey with at least M (their env's) graded, unsubmitted sessions."""
        now = int(now)
        graded: dict[str, list[Any]] = {}
        for record in self._records(str(hotkey), now):
            fresh_lapse = record.state == LAPSED and (record.closed_at or 0) >= now - _LATE_GRADED_S
            if record.closed_status == "graded" and (record.state == CLOSED_GRADED or fresh_lapse):
                graded.setdefault(self._sha.get(record.session_id), []).append(record)
        return sum(1 for members in graded.values() if len(members) >= self._m(members))


__all__ = ["YieldQuota"]
