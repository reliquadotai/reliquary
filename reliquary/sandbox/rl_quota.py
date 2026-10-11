"""The allocation of RL sandbox sessions per hotkey, split-invariant: the ``SessionQuota`` hook of the RL
engagements.

Every ``refresh_s`` at most (and at once when the ledger's snapshot or the fleet's size changes), the capacity is
allocated by weighted water-filling over entities (``rl_capacity.allocate``): each MAIN hotkey with its weight and its
CURRENT demand, the NEWCOMER entity and the LOW_YIELD entity. A hotkey's demand is its counted sessions plus the
ADMISSIBLE open requests it made in the last ``DEMAND_TTL_S`` (the caller records a request only after the seed and
the session book's checks passed, and withdraws it when the precommit's task then fails: ``forget``), counted by
group: each precommit held or asked counts at least its M, as at admission. A miner sending its seeds one at a time
shows a whole group's demand from its first request.

* What the MINER is told is its POTENTIAL (``rl_capacity.potential``): its weighted share with its own demand
  unbounded, the others' as measured. Admission uses the allocation (its current demand).
* GROUPS ARE NEVER CUT: a seed of a started precommit (one with a counted session) is admitted until the precommit
  holds M counted sessions (M of its env), whatever the allocation, the entity or the standing became since. A NEW
  group starts only when the hotkey's committed sessions (each started group counts at least its M) plus the M of the
  new group's env fit in its allocation (MAIN) or its member maximum and its entity (pooled): a hotkey whose
  allocation is below M never opens a session that cannot form a group.
* LEAVING A POOL (read from the ledger, so it survives a restart): a hotkey whose ledger standing is MAIN stays in a
  pooled entity while its MAIN potential is below one group (M), only
  - as a newcomer while ``YIELD_MIN_DECIDED <= graduation < YIELD_MIN_DECIDED + GRACE_SESSIONS`` (the graduation
    grace: bounded, it ends after ``GRACE_SESSIONS`` more decided sessions; ``YieldStats.graduation`` also counts
    graded sessions a capacity verdict made neutral), or
  - in the low-yield entity while its yield is below ``YIELD_FULL`` (a recovering hotkey keeps its one group).
  Splitting a hotkey into parts below one group therefore buys at most the pooled entities' rules.
* Pooled members: at most ``NEWCOMER_MAX_GROUPS`` / ``LOW_YIELD_MAX_GROUPS`` groups each. First opens of groups are
  FIFO by group; in the newcomer entity a hotkey with fewer than ``YIELD_MIN_DECIDED`` decided sessions goes before a
  graduate in grace (a graduate starts a group only in the room the newcomers' current demand leaves). A waiting group keeps its place until one of its sessions opens (or it stops asking for
  ``ASK_FORGET_S``), so an open that fails after the check does not lose it.

A session COUNTS while it holds (``SessionBook._holds``) and, while ``live``, only if a machine report listed it
(``reported``) or it was issued less than ``PRESENT_WINDOW_S`` ago (the gateway refuses it later).

Counting never scans the session book: the quota is a ``SessionEvents`` listener. Its index keeps only the sessions
that still hold (and a graded session lapsed in the last minute), plus each hotkey's and each pooled entity's open
times over its period. Every method is synchronous and must be called on the event loop (never from a worker
thread): each call is then atomic."""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from reliquary.constants import CODE_ADMISSION_WALL_SECONDS, MATH_ADMISSION_WALL_SECONDS
from reliquary.sandbox.rl_capacity import (
    ALLOCATION_REFRESH_S,
    ASK_FORGET_S,
    DEMAND_TTL_S,
    GRACE_SESSIONS,
    LOW_YIELD,
    LOW_YIELD_MAX_GROUPS,
    LOW_YIELD_OPENS_PERIOD_S,
    MAIN,
    NEWCOMER,
    NEWCOMER_MAX_GROUPS,
    NEWCOMER_TAKES_SPARE,
    OPENS_PERIOD_S,
    PRESENT_WINDOW_S,
    YIELD_FULL,
    YIELD_MIN_DECIDED,
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
    GRADING_GRACE_S,
    LAPSED,
    LIVE,
    VOIDED,
    Refusal,
    SessionBook,
)

_LANE_GROUPS = {NEWCOMER: NEWCOMER_MAX_GROUPS, LOW_YIELD: LOW_YIELD_MAX_GROUPS}
_LANE_PERIOD = {MAIN: OPENS_PERIOD_S, NEWCOMER: OPENS_PERIOD_S, LOW_YIELD: LOW_YIELD_OPENS_PERIOD_S}
_LATE_GRADED_S = 60          # a lapsed graded session may still be claimed by a group received in time
# A group received by its sessions' deadline is admitted within its admission wall, and the lapse sweep runs after
# that deadline: the index keeps a lapsed graded session for as long as such a group can still claim it.
assert _LATE_GRADED_S >= max(MATH_ADMISSION_WALL_SECONDS, CODE_ADMISSION_WALL_SECONDS)
_NOT_COUNTED = (ABORTED, VOIDED)


def _precommit_of(record: Any) -> str | None:
    engagement = getattr(record, "engagement", None)
    if not isinstance(engagement, str) or not engagement.startswith("rl:"):
        return None
    from reliquary.protocol.service_episode import EpisodeWireError, parse_rl_engagement

    try:
        return parse_rl_engagement(engagement)[1]
    except EpisodeWireError:
        return None


def _prune(opens: deque, period: int, now: int) -> deque:
    while opens and opens[0][0] <= now - period:
        opens.popleft()
    return opens


def _opens_refusal(reason: str, opens: deque, limit: int, period: int, now: int, detail: dict,
                   retry_after_s: int) -> Refusal | None:
    """None while fewer than ``limit`` opens of the period count (aborted and voided ones do not)."""
    if len(opens) < limit:                       # the fast path: even counting every open, below the limit
        return None
    times = [at for at, record in opens if getattr(record, "state", None) not in _NOT_COUNTED]   # issue order
    if len(times) < limit:
        return None
    retry = max(1, int(times[len(times) - limit] + period - now)) if limit else retry_after_s
    return Refusal(reason, {"opens": len(times), "max": limit, "period_s": period, **detail}, retry_after=retry)


class YieldQuota:
    def __init__(self, *, ledger: Any, pool_seeds: int, group_size: int, capacity: Callable[[], int],
                 reported: Callable[[str], bool] | None = None, takes_spare: bool = NEWCOMER_TAKES_SPARE,
                 refresh_s: float = ALLOCATION_REFRESH_S, retry_after_s: int = 10,
                 group_sizes: Mapping[str, int] | None = None, clock: Callable[[], float] = time.time) -> None:
        self._ledger = ledger
        self._pool_seeds, self._group_size = int(pool_seeds), int(group_size)
        # env -> M (its group size), keyed by every name an env goes by: the name an open gives and the name a
        # session record stores (its sandbox env).
        self._group_sizes = {str(k): int(v) for k, v in (group_sizes or {}).items()}
        self._capacity_of = capacity
        self._reported = reported
        self._takes_spare = bool(takes_spare)
        self._refresh_s = float(refresh_s)
        self._retry_after_s = int(retry_after_s)
        self._clock = clock
        self._index: dict[str, dict[str, Any]] = {}              # hotkey -> {session_id: record}, while it holds
        self._sha: dict[str, str] = {}                           # session_id -> precommit (parsed once)
        self._last_m: dict[str, int] = {}                        # hotkey -> M of its last opened session's env
        self._opens: dict[str, deque] = {}                       # hotkey -> (issued_at, record) over its period
        self._wants: dict[str, dict[tuple[str, int], float]] = {}   # hotkey -> {(precommit, seed): last try}
        self._want_m: dict[str, int] = {}                        # asked precommit -> M of its env
        # pooled entity -> {precommit: (hotkey, first ask, last ask, a genuine newcomer)}
        self._asks: dict[str, dict[str, tuple[str, float, float, bool]]] = {NEWCOMER: {}, LOW_YIELD: {}}
        self._lane_opens: dict[str, deque] = {NEWCOMER: deque(), LOW_YIELD: deque()}
        self._lane_counted: dict[str, int] = {NEWCOMER: 0, LOW_YIELD: 0}
        self._genuine_waiting = 0                                # newcomers' (decided < 96) demand not yet held
        self._capacity_cache: tuple[float, int] | None = None
        self._alloc: Allocation | None = None
        self._snapshot: Mapping[str, YieldStats] = {}
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
        self._last_m[hotkey] = self._m_env(getattr(record, "env", None))
        wants = self._wants.get(hotkey)
        if wants is not None:
            wants.pop((sha, self._seed_of(record)), None)
        for asks in self._asks.values():
            asks.pop(sha, None)                              # its group started: it leaves the line
        self._opens.setdefault(hotkey, deque()).append((int(record.issued_at), record))
        lane = self.lane(hotkey)
        if lane != MAIN:
            self._lane_opens[lane].append((int(record.issued_at), record))
            self._lane_counted[lane] += 1            # until the next refresh recounts
        self._dirty = True

    def closed(self, record: Any, final: Any) -> None:
        self._dirty = True                           # the record is updated in place by the book

    def rebuild(self, records: Iterable[Any]) -> None:
        now = int(self._clock())
        for record in sorted(records, key=lambda r: int(r.issued_at)):
            self.opened(record)
        for lane, opens in self._lane_opens.items():
            _prune(opens, _LANE_PERIOD[lane], now)
        for hotkey in list(self._index):
            self._records(hotkey, now)
        self._dirty, self._alloc_key = True, None

    @staticmethod
    def _seed_of(record: Any) -> int:
        try:
            return int(str(record.engagement).rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return -1

    @staticmethod
    def _kept(record: Any, now: int) -> bool:
        if SessionBook._holds(record, now):
            return True
        return (record.state == LAPSED and record.closed_status == "graded"
                and (record.closed_at or 0) >= now - _LATE_GRADED_S)

    def _records(self, hotkey: str, now: int) -> list[Any]:
        records = self._index.get(hotkey)
        if not records:
            return []
        for session_id in [sid for sid, r in records.items() if not self._kept(r, now)]:
            del records[session_id]
            self._sha.pop(session_id, None)
        if not records:
            del self._index[hotkey]
            return []
        return list(records.values())

    def _key_opens(self, hotkey: str, period: int, now: int) -> deque:
        opens = self._opens.get(hotkey)
        if opens is None:
            return deque()
        _prune(opens, period, now)
        if not opens and hotkey not in self._index:
            del self._opens[hotkey]
        return opens

    def _counts(self, record: Any, now: int) -> bool:
        if not SessionBook._holds(record, now):
            return False
        if record.state != LIVE or now - int(record.issued_at) <= PRESENT_WINDOW_S:
            return True
        return bool(self._reported(record.session_id)) if self._reported is not None else True

    def _counted(self, hotkey: str, now: int) -> list[Any]:
        return [r for r in self._records(hotkey, now) if self._counts(r, now)]

    def _wanted(self, hotkey: str, now: int) -> int:
        wants = self._wants.get(hotkey)
        if wants:
            for key in [k for k, at in wants.items() if now - at > DEMAND_TTL_S]:
                del wants[key]
            if not wants:
                del self._wants[hotkey]
        return len(self._wants.get(hotkey, ()))

    def _demand(self, hotkey: str, now: int, counted: list[Any] | None = None) -> int:
        """Counted sessions plus admissible requests, by group: each precommit held or asked counts at least its
        M (as ``committed`` does at admission), since asking for one seed of a group is asking for the group. A
        miner that sends its seeds one at a time therefore shows a group's demand from its first request."""
        self._wanted(hotkey, now)
        per: dict[str | None, list[int]] = {}                    # precommit -> [sessions and requests, M]
        for record in self._counted(hotkey, now) if counted is None else counted:
            sha = self._sha.get(record.session_id)
            entry = per.setdefault(sha, [0, self._m_env(getattr(record, "env", None))])
            entry[0] += 1
        for sha, _seed in self._wants.get(hotkey, ()):
            entry = per.setdefault(sha, [0, self._want_m.get(sha, self._group_size)])
            entry[0] += 1
        return sum(max(n, m) for n, m in per.values())

    def forget(self, hotkey: str, precommit_sha256: str) -> None:
        """A request the quota admitted was then refused for its precommit's task (unresolvable, too long): its
        precommit's requests and its place in a pooled entity's line go, at once (no demand from a precommit
        that cannot open)."""
        hotkey = str(hotkey)
        wants = self._wants.get(hotkey)
        if wants:
            for key in [k for k in wants if k[0] == precommit_sha256]:
                del wants[key]
            if not wants:
                del self._wants[hotkey]
        for asks in self._asks.values():
            ask = asks.get(precommit_sha256)
            if ask is not None and ask[0] == hotkey:
                del asks[precommit_sha256]
        self._dirty = True

    def _m_env(self, environment: str | None) -> int:
        if environment is None:
            return self._group_size
        return int(self._group_sizes.get(str(environment), self._group_size))

    def _m(self, records: Iterable[Any]) -> int:
        for record in records:
            return self._m_env(getattr(record, "env", None))
        return self._group_size

    # --- standings and the allocation ---
    def _capacity(self, now: float) -> int:
        cached = self._capacity_cache
        if cached is None or now - cached[0] >= 1.0:          # a directory scan: at most once a second
            cached = self._capacity_cache = (now, int(self._capacity_of()))
        return cached[1]

    def _stats(self, hotkey: str, snapshot: Mapping[str, YieldStats] | None = None) -> YieldStats:
        snapshot = self._ledger.snapshot() if snapshot is None else snapshot
        return snapshot.get(str(hotkey), YieldStats())

    def _stats_lane(self, stats: YieldStats) -> str:
        lane = standing(stats, pool_seeds=self._pool_seeds)
        if lane == MAIN and weight(stats, pool_seeds=self._pool_seeds) <= 0:
            return NEWCOMER                       # no useful session in 24 h: back among the newcomers
        return lane

    @staticmethod
    def _pool_kept(stats: YieldStats) -> str | None:
        """The pooled entity a MAIN-standing hotkey stays in while MAIN would not give it a group (or None)."""
        if YIELD_MIN_DECIDED <= stats.graduation < YIELD_MIN_DECIDED + GRACE_SESSIONS:
            return NEWCOMER                       # the graduation grace, bounded by the ledger's count
        if stats.yield_ratio < YIELD_FULL:
            return LOW_YIELD                      # a recovering hotkey keeps its group
        return None

    def _main_potential(self, capacity: int, hotkey: str, w: float, main: Mapping[str, tuple[float, float]],
                        entity: Mapping[str, float]) -> float:
        return potential(capacity, pool_seeds=self._pool_seeds, main=main, hotkey=hotkey, weight_h=w,
                         newcomer_demand=entity[NEWCOMER], low_yield_demand=entity[LOW_YIELD],
                         takes_spare=self._takes_spare)

    def _settle_lane(self, hotkey: str, stats: YieldStats, capacity: int, main: Mapping[str, tuple[float, float]],
                     entity: Mapping[str, float]) -> str:
        lane = self._stats_lane(stats)
        if lane != MAIN:
            return lane
        kept = self._pool_kept(stats)
        if kept is None:
            return MAIN
        w = weight(stats, pool_seeds=self._pool_seeds)
        m = self._last_m.get(hotkey, self._group_size)
        return MAIN if sessions_of(self._main_potential(capacity, hotkey, w, main, entity)) >= m else kept

    def lane(self, hotkey: str) -> str:
        """MAIN, NEWCOMER or LOW_YIELD: the entity the hotkey belongs to now."""
        hotkey = str(hotkey)
        capacity = self._capacity(self._clock())
        if capacity > 0:
            self.allocation()                     # the standings as of the current allocation
        known = self._lanes.get(hotkey)
        if known is not None:
            return known
        return self._settle_lane(hotkey, self._stats(hotkey), capacity, self._claims, self._entity_demand)

    def _genuine(self, hotkey: str) -> bool:
        return self._stats(hotkey, self._snapshot or None).graduation < YIELD_MIN_DECIDED

    def allocation(self, now: float | None = None) -> Allocation:
        now = int(self._clock() if now is None else now)
        key = (getattr(self._ledger, "version", None), self._capacity(now))
        stale = key != self._alloc_key or (self._dirty and now - self._alloc_at >= self._refresh_s)
        if self._alloc is None or stale:
            group = self._pool_seeds
            snapshot = self._ledger.snapshot()    # one snapshot for the whole recompute
            lanes: dict[str, str] = {}
            counted: dict[str, int] = {}
            demands: dict[str, int] = {}
            main: dict[str, tuple[float, float]] = {}
            kept: list[tuple[str, YieldStats]] = []
            for hotkey in set(self._index) | set(self._wants):
                stats = snapshot.get(hotkey, YieldStats())
                rows = self._counted(hotkey, now)
                counted[hotkey] = len(rows)
                demands[hotkey] = self._demand(hotkey, now, rows)
                lane = self._stats_lane(stats)
                if lane == MAIN and self._pool_kept(stats) is not None:
                    kept.append((hotkey, stats))
                    continue
                lanes[hotkey] = lane
                if lane == MAIN:
                    main[hotkey] = (weight(stats, pool_seeds=group), float(demands[hotkey]))
            entity = {NEWCOMER: 0.0, LOW_YIELD: 0.0}
            for hotkey, lane in lanes.items():
                if lane != MAIN:
                    entity[lane] += min(demands[hotkey], _LANE_GROUPS[lane] * group)
            for hotkey, stats in sorted(kept):    # leave the pool as soon as MAIN gives a group
                lane = self._settle_lane(hotkey, stats, key[1], main, entity)
                lanes[hotkey] = lane
                if lane == MAIN:
                    main[hotkey] = (weight(stats, pool_seeds=group), float(demands[hotkey]))
                else:
                    entity[lane] += min(demands[hotkey], _LANE_GROUPS[lane] * group)
            lane_counted = {NEWCOMER: 0, LOW_YIELD: 0}
            genuine_waiting = 0
            for hotkey, lane in lanes.items():
                if lane != MAIN:
                    lane_counted[lane] += counted[hotkey]
                if lane == NEWCOMER and snapshot.get(hotkey, YieldStats()).graduation < YIELD_MIN_DECIDED:
                    member_max = _LANE_GROUPS[NEWCOMER] * group
                    genuine_waiting += max(0, min(demands[hotkey], member_max) - counted[hotkey])
            for lane, opens in self._lane_opens.items():
                _prune(opens, _LANE_PERIOD[lane], now)
            for hotkey in list(self._opens):      # bounded memory: open times over the period only
                if not _prune(self._opens[hotkey], _LANE_PERIOD[lanes.get(hotkey, MAIN)], now):
                    del self._opens[hotkey]
            for hotkey in [h for h in self._last_m if h not in self._index and h not in self._opens]:
                del self._last_m[hotkey]
            asked = {sha for wants in self._wants.values() for sha, _seed in wants}
            for sha in [sha for sha in self._want_m if sha not in asked]:
                del self._want_m[sha]
            self._alloc = allocate(key[1], pool_seeds=group, main=main, newcomer_demand=entity[NEWCOMER],
                                   low_yield_demand=entity[LOW_YIELD], takes_spare=self._takes_spare)
            self._snapshot, self._claims, self._entity_demand = snapshot, main, entity
            self._lanes, self._lane_counted, self._potentials = lanes, lane_counted, {}
            self._genuine_waiting = genuine_waiting
            self._alloc_key, self._alloc_at, self._dirty = key, now, False
        return self._alloc

    def potential_sessions(self, hotkey: str) -> int:
        """The sessions the hotkey could hold if it asked for them (what the miner is told). A pooled hotkey: its
        member maximum."""
        alloc = self.allocation()
        hotkey = str(hotkey)
        lane = self.lane(hotkey)
        if lane != MAIN:
            return _LANE_GROUPS[lane] * self._pool_seeds
        found = self._potentials.get(hotkey)
        if found is None:
            w = weight(self._stats(hotkey, self._snapshot or None), pool_seeds=self._pool_seeds)
            found = self._potentials[hotkey] = sessions_of(
                self._main_potential(alloc.capacity, hotkey, w, self._claims, self._entity_demand))
        return found

    def _entity_sessions(self, lane: str, alloc: Allocation) -> int:
        return sessions_of(alloc.newcomer if lane == NEWCOMER else alloc.low_yield)

    # --- the quota ---
    def refusal(self, hotkey: str, now: int, *, precommit_sha256: str | None = None, seed_index: int | None = None,
                environment: str | None = None) -> Refusal | None:
        """None admits. The caller passes ``seed_index`` only for an ADMISSIBLE request (its seed free, the session
        book's caps passed): only those count as demand. ``environment``: the env of the precommit (its M)."""
        hotkey, now = str(hotkey), int(now)
        if precommit_sha256 is not None and seed_index is not None:
            self._wants.setdefault(hotkey, {})[(precommit_sha256, int(seed_index))] = float(now)
            self._want_m[precommit_sha256] = self._m_env(environment)
            self._dirty = True
        if self._capacity(now) <= 0:
            return Refusal("quota_live", {"why": "no fleet capacity known yet"}, retry_after=self._retry_after_s)
        alloc = self.allocation(now)
        lane = self.lane(hotkey)
        stats = self._stats(hotkey, self._snapshot or None)
        detail = {"lane": lane, "yield": round(stats.yield_ratio, 3), "useful_24h": stats.useful_24h}
        counted = self._counted(hotkey, now)
        groups: dict[str | None, list[Any]] = {}
        for record in counted:
            groups.setdefault(self._sha.get(record.session_id), []).append(record)
        started = groups.get(precommit_sha256) if precommit_sha256 is not None else None
        if started and len(started) < self._m(started):
            return None                    # a group in progress is never cut
        starts = precommit_sha256 is not None and not started
        committed = sum(max(len(members), self._m(members)) for members in groups.values())
        m_new = self._m_env(environment)
        period = _LANE_PERIOD[lane]
        mine = self._key_opens(hotkey, period, now)
        if lane == MAIN:
            limit = sessions_of(alloc.main.get(hotkey, 0.0))
            refused = _opens_refusal("quota_opens", mine, opens_for(max(self.potential_sessions(hotkey),
                                                                         len(counted))), period, now, detail,
                                     self._retry_after_s)
            if refused is not None:
                return refused
            if (starts and committed + m_new > limit) or (not starts and len(counted) >= limit):
                return Refusal("quota_live", {"live": len(counted), "committed": committed, "max": limit, **detail},
                               retry_after=self._retry_after_s)
            return None
        member_groups = _LANE_GROUPS[lane]
        member_max = member_groups * self._pool_seeds
        refused = _opens_refusal("quota_opens", mine, opens_for(member_max), period, now, detail,
                                 self._retry_after_s)
        if refused is not None:
            return refused
        entity = self._entity_sessions(lane, alloc)
        pooled = _prune(self._lane_opens[lane], period, now)
        guarantee = alloc.newcomer_guarantee if lane == NEWCOMER else alloc.low_yield_cap
        refused = _opens_refusal("quota_pool_opens", pooled, opens_for(max(entity, guarantee)), period, now, detail,
                                 self._retry_after_s)
        if refused is not None:
            return refused
        if starts and (len(groups) >= member_groups or committed + m_new > member_max):
            return Refusal("quota_live", {"groups": len(groups), "max": member_groups, **detail},
                           retry_after=self._retry_after_s)
        if not starts and len(counted) >= member_max:
            return Refusal("quota_live", {"live": len(counted), "max": member_max, **detail},
                           retry_after=self._retry_after_s)
        asks = self._asks[lane]
        for sha in [sha for sha, ask in asks.items() if now - ask[2] > ASK_FORGET_S]:
            del asks[sha]
        genuine = lane != NEWCOMER or self._genuine(hotkey)
        first = asks[precommit_sha256][1] if starts and precommit_sha256 in asks else float(now)
        room = entity - self._lane_counted[lane]
        if starts and not genuine:
            room -= self._genuine_waiting                        # a graduate in grace leaves the newcomers' share
        if room < (m_new if starts else 1):
            if starts:
                asks[precommit_sha256] = (hotkey, first, float(now), genuine)   # it waits its turn, as a group
            return Refusal("quota_pool", {"pool": lane, "live": self._lane_counted[lane], "max": entity, **detail},
                           retry_after=self._retry_after_s)
        if not starts:
            return None                                          # over-generation of a group it runs
        # FIFO by group, never one turn per hotkey; a genuine newcomer goes before a graduate in grace
        older = sum(1 for sha, (asker, at, _last, other) in asks.items()
                    if asker != hotkey and ((other and not genuine) or (other == genuine and at < first)))
        if older:
            asks[precommit_sha256] = (hotkey, first, float(now), genuine)
            return Refusal("quota_pool", {"pool": lane, "ahead": older, **detail}, retry_after=self._retry_after_s)
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
        return self.refusal(hotkey, int(self._clock()), precommit_sha256=precommit_sha256, seed_index=seed_index,
                            environment=environment)

    def admit_locked(self, hotkey: str, now: int, *, precommit_sha256: str | None = None,
                     seed_index: int | None = None, environment: str | None = None) -> Refusal | None:
        """The same check under the issuer lock. A waiting group leaves the line when one of its sessions opens
        (``opened``), so an open that fails after this check keeps its place."""
        return self.refusal(hotkey, int(now), precommit_sha256=precommit_sha256, seed_index=seed_index,
                            environment=environment)

    def rank(self, hotkey: str) -> tuple[str, float, bool]:
        """The open queue's (key, score, pooled)."""
        if self.lane(hotkey) == MAIN:
            return str(hotkey), weight(self._stats(hotkey), pool_seeds=self._pool_seeds), False
        return str(hotkey), 0.0, True

    def sessions_allowed(self, hotkey: str) -> int:
        """What the miner is told: its potential."""
        return self.potential_sessions(hotkey)

    def groups_allowed(self, hotkey: str) -> int:
        """Groups the hotkey may hold at once (at least one): the cap on capacity verdicts per window."""
        return max(1, math.ceil(self.potential_sessions(hotkey) / self._pool_seeds))

    def cap_scale(self, hotkey: str) -> float:
        """How much a per-hotkey rate constant of the session policy scales for this hotkey. The RL policy's
        constants are sized for ``NEWCOMER_MAX_GROUPS`` groups in flight (a newcomer's maximum), so the scale is the
        hotkey's potential in such sets of groups, never below 1: a newcomer keeps the policy's own budget."""
        return max(1.0, self.potential_sessions(hotkey) / (NEWCOMER_MAX_GROUPS * self._pool_seeds))

    def submittable(self, hotkey: str, now: int, received: float | None = None) -> frozenset[str]:
        """The precommits of the hotkey with at least M (their env's) graded, unsubmitted sessions: the groups it
        can submit. A lapsed graded session counts as the issuer's claim judges it: for a group ``received`` by
        the session's deadline (``expires_at + GRADING_GRACE_S``); without ``received``, while it lapsed in the
        last minute."""
        now = int(now)
        graded: dict[str, list[Any]] = {}
        for record in self._records(str(hotkey), now):
            if record.state == LAPSED:
                late = ((record.closed_at or 0) >= now - _LATE_GRADED_S if received is None
                        else float(received) <= record.expires_at + GRADING_GRACE_S)
            else:
                late = False
            sha = self._sha.get(record.session_id)
            if sha is not None and record.closed_status == "graded" and (record.state == CLOSED_GRADED or late):
                graded.setdefault(sha, []).append(record)
        return frozenset(sha for sha, members in graded.items() if len(members) >= self._m(members))

    def groups_formable(self, hotkey: str, now: int, received: float | None = None) -> int:
        """How many groups the hotkey can submit (``submittable``)."""
        return len(self.submittable(hotkey, now, received))


__all__ = ["YieldQuota"]
