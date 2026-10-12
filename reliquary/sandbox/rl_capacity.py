"""The RL sandbox session quota and the fair open queue: every constant in one place and the allocation arithmetic
(keyed by hotkey and split-invariant: more hotkeys never buy more capacity). The values are defaults; they are
measured and tuned at qualification. Change them here only.

Allocation (weighted water-filling, recomputed at each ledger flush and on demand changes). The capacity ``C`` (the
session slots of the fresh RL machines) is shared among ENTITIES that have a current demand, in sessions:

* the LOW_YIELD entity (every hotkey at or below ``YIELD_FLOOR``) first gets ``min(L, its demand)``: a hard cap,
  never more (it bounds the wasters);
* the NEWCOMER entity (every hotkey with fewer than ``YIELD_MIN_DECIDED`` decided sessions) is guaranteed
  ``min(N, its demand)``; beyond it, it takes part in the water-filling with the constant weight
  ``NEWCOMER_WEIGHT`` when ``NEWCOMER_TAKES_SPARE``;
* each MAIN hotkey takes part with weight ``w = useful held session-seconds over 24 h / 86 400 x m(yield)`` and its
  CURRENT demand (counted live sessions + open requests it is making now): it gets ``min(demand, weighted share)``,
  and what the capped entities leave is re-spread by weight until it is used up.

Water-filling with weights and demands is split-invariant (k parts with proportional weights and demands get exactly
the whole's total), monotonic (more weight, or more demand, never lowers an entity's allocation) and dormant-proof
(no demand, nothing). Allocations are fractional sessions; a hotkey may open while its counted sessions are below
``floor(allocation)`` (sum of floors never exceeds the floor of the sum). There is no per-key constant."""
from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping
from typing import Any
from dataclasses import dataclass, field

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT

# --- yield and standing ---
YIELD_SESSIONS = 200
YIELD_MIN_DECIDED = 96          # below this many decided sessions a hotkey is a newcomer
YIELD_FULL = 0.80               # multiplier 1 at and above
YIELD_FLOOR = 0.50              # at and below: the low-yield entity (the multiplier reaches 0 there: no step)
USEFUL_HORIZON_S = 86_400
MAIN, NEWCOMER, LOW_YIELD = "main", "newcomer", "low_yield"

# --- the entities ---
NEWCOMER_SHARE = 0.10           # N = max(P, floor(NEWCOMER_SHARE x C)): the newcomer entity's guarantee
NEWCOMER_WEIGHT = 32.0          # its weight for spare capacity (units of w: concurrent useful sessions)
NEWCOMER_TAKES_SPARE = True     # newcomers may use spare capacity beyond N (the operator grows the fleet by hand)
NEWCOMER_MAX_GROUPS = GROUPS_IN_FLIGHT      # groups a newcomer holds at once inside its entity
NEWCOMER_EXIT_GROUPS = 2        # a newcomer that wasted this many groups' worth leaves for the low-yield entity
LOW_YIELD_SHARE = 0.03          # L = max(P, floor(LOW_YIELD_SHARE x C)): the low-yield entity's hard cap
LOW_YIELD_MAX_GROUPS = 1
GRACE_SESSIONS = 64            # a graduate (YIELD_MIN_DECIDED <= decided < this more) stays a newcomer while MAIN
                               # would not give it a group; read from the ledger, so it survives a restart

# --- demand, tokens, refresh ---
DEMAND_TTL_S = 30.0             # an open request (refused or waiting) counts as demand this long after its last try
PRESENT_WINDOW_S = 120          # a token no machine report lists within this stops counting (the gateway refuses it later)
ALLOCATION_REFRESH_S = 1.0      # the allocation is recomputed at most this often when only demand changed
ASK_FORGET_S = 20.0             # a refused first open of a group waits its turn this long (FIFO by group)

# --- opens ---
OPENS_PER_LIVE_PER_HOUR = 12        # turnovers an hour per allocated session: sessions as short as 5 min
OPENS_PERIOD_S = 3_600
LOW_YIELD_OPENS_PERIOD_S = 4 * 3_600

# --- checks (the intake) ---
MAX_CHECKS_RUNNING = GROUPS_IN_FLIGHT + 1   # transcript checks at once, all miners (one is always left to others)

# --- the open queue ---
QUEUE_WAIT_S = 20.0           # below the miner's 60 s sessions HTTP timeout (episode_mining)
QUEUE_MAX_WAITING = 512
QUEUE_MAX_PER_KEY = 64
QUEUE_AGING_S = 60.0
QUEUE_AGING_FORGET_S = 120.0
QUEUE_FAIR_EVERY = 4
QUEUE_HOLD_AFTER_S = 5.0      # a head waiter that waited this long and missed gets one machine held for it

# --- servable tasks ---
MIN_SERVING_MACHINES = 1

# --- saturation: what tells the operator to grow the fleet (the fleet is fixed, grown by hand) ---
SATURATION_WINDOW_S = 300               # metrics cover the last five minutes
SATURATION_ALERT_FOR_S = 300            # an ALERT when the condition holds this long
SATURATION_ALERT_WAIT_S = 10.0          # p90 of the queue's waits above this
SATURATION_ALERT_REFUSALS_PER_MIN = 30  # or DISTINCT opens refused for the fleet's capacity a minute
SATURATION_ALERT_FREE_FRACTION = 0.05   # and only while less than this share of the fleet's sessions is free
SATURATION_CHECK_S = 10.0
CHECKER_BUSY_ALERT_PER_MIN = 6          # distinct groups refused ``episode_checker_busy`` a minute: raise the
                                        # check slots (MAX_CHECKS_RUNNING), sized from the check times' p90
CAPACITY_VERDICT_ALERT_SESSIONS = 32    # a hotkey's sessions a capacity verdict made neutral (since start) ...
CAPACITY_VERDICT_ALERT_SHARE = 0.25     # ... at least this share of its decided + neutral sessions: an ALERT
CAPACITY_VERDICT_TOP = 20               # the hotkeys the saturation view lists (most neutral sessions first)
CAPACITY_VERDICT_SAMPLE_S = 300         # the capacity-verdict counts are sampled this often (a 24 h window)

# --- the fleet size's per-session reservation, seeded at start from the env's own tasks ---
TIER_SEED_TASKS = 512                   # tasks sampled per env (evenly over its index range)
TIER_QUANTILE = 0.95                    # the reservation: this quantile of their CPUs and memory budgets
TIER_SEED_TIMEOUT_S = 60.0              # the whole seeding, per env

NEWCOMER_ENTITY = "\x00newcomer"        # never a hotkey (SS58 has no NUL)


@dataclass(frozen=True)
class YieldStats:
    """``useful`` and ``decided`` (useful + wasted) among the hotkey's last ``YIELD_SESSIONS`` decided sessions;
    ``useful_24h`` and ``useful_s_24h``: its useful sessions decided in the last ``USEFUL_HORIZON_S``, and their held
    seconds (issue to close: an env-neutral volume); ``capacity_neutral``: its graded sessions the ledger still keeps
    that a capacity verdict made neither useful nor wasted. They count toward graduation (``graduation``), never
    toward the yield, so a hotkey whose sessions are all capacity-neutral cannot stay a newcomer forever."""

    useful: int = 0
    decided: int = 0
    useful_24h: int = 0
    useful_s_24h: float = 0.0
    capacity_neutral: int = 0

    @property
    def yield_ratio(self) -> float:
        return self.useful / self.decided if self.decided else 0.0

    @property
    def graduation(self) -> int:
        """The sessions that count toward leaving the newcomers: decided ones plus capacity-neutral graded ones."""
        return self.decided + self.capacity_neutral


def multiplier(yield_ratio: float) -> float:
    if yield_ratio >= YIELD_FULL:
        return 1.0
    if yield_ratio <= YIELD_FLOOR:
        return 0.0
    return (yield_ratio - YIELD_FLOOR) / (YIELD_FULL - YIELD_FLOOR)


def standing(stats: YieldStats, *, pool_seeds: int) -> str:
    if stats.graduation < YIELD_MIN_DECIDED:
        if stats.decided >= NEWCOMER_EXIT_GROUPS * int(pool_seeds) and stats.yield_ratio <= YIELD_FLOOR:
            return LOW_YIELD
        return NEWCOMER
    return LOW_YIELD if stats.yield_ratio <= YIELD_FLOOR else MAIN


def weight(stats: YieldStats, *, pool_seeds: int) -> float:
    """``w`` of a MAIN hotkey: its concurrent useful sessions over 24 h x the yield multiplier; 0 otherwise. Linear
    in the volume (split-invariant); continuous through YIELD_FLOOR (the multiplier is 0 there)."""
    if standing(stats, pool_seeds=pool_seeds) != MAIN:
        return 0.0
    return float(stats.useful_s_24h) / USEFUL_HORIZON_S * multiplier(stats.yield_ratio)


def water_fill(capacity: float, claims: Mapping[Hashable, tuple[float, float]]) -> dict[Hashable, float]:
    """Weighted max-min fair shares of ``capacity`` among ``claims`` = {entity: (weight, demand)}: each gets
    ``min(demand, weighted share)``; what the capped ones leave is re-spread by weight until it is used up."""
    alloc = {key: 0.0 for key in claims}
    active = {key for key, (w, d) in claims.items() if w > 0 and d > 0}
    left = max(0.0, float(capacity))
    while active and left > 1e-9:
        total = sum(claims[key][0] for key in active)
        unit = left / total
        capped = {key for key in active if claims[key][1] - alloc[key] <= claims[key][0] * unit + 1e-12}
        if not capped:
            for key in active:
                alloc[key] += claims[key][0] * unit
            break
        for key in capped:
            left -= claims[key][1] - alloc[key]
            alloc[key] = float(claims[key][1])
        active -= capped
    return alloc


@dataclass(frozen=True)
class Allocation:
    capacity: int
    newcomer_guarantee: int         # N
    low_yield_cap: int              # L
    newcomer: float                 # the newcomer entity's allocation
    low_yield: float                # the low-yield entity's allocation
    main: dict = field(default_factory=dict)   # MAIN hotkey -> allocation (sessions, fractional)


def entity_sizes(capacity: int, *, pool_seeds: int) -> tuple[int, int]:
    """(N, L) of a fleet of ``capacity`` session slots: never below one group."""
    capacity, group = max(0, int(capacity)), int(pool_seeds)
    return (max(group, math.floor(NEWCOMER_SHARE * capacity)), max(group, math.floor(LOW_YIELD_SHARE * capacity)))


def allocate(capacity: int, *, pool_seeds: int, main: Mapping[str, tuple[float, float]], newcomer_demand: float,
             low_yield_demand: float, takes_spare: bool = NEWCOMER_TAKES_SPARE) -> Allocation:
    """The allocation of ``capacity`` sessions: the low-yield entity's ``min(L, demand)`` (hard cap), the newcomer
    entity's guaranteed ``min(N, demand)``, then weighted water-filling of the rest among the MAIN hotkeys
    (``main`` = {hotkey: (w, current demand)}) and, when ``takes_spare``, the newcomer entity's unmet demand."""
    capacity = max(0, int(capacity))
    newcomer_n, low_l = entity_sizes(capacity, pool_seeds=pool_seeds)
    low = min(float(low_l), max(0.0, float(low_yield_demand)), float(capacity))
    new = min(float(newcomer_n), max(0.0, float(newcomer_demand)), capacity - low)
    claims: dict[Hashable, tuple[float, float]] = {key: (float(w), float(d)) for key, (w, d) in main.items()}
    if takes_spare and newcomer_demand > new:
        claims[NEWCOMER_ENTITY] = (NEWCOMER_WEIGHT, float(newcomer_demand) - new)
    filled = water_fill(capacity - low - new, claims)
    new += filled.pop(NEWCOMER_ENTITY, 0.0)
    return Allocation(capacity=capacity, newcomer_guarantee=newcomer_n, low_yield_cap=low_l, newcomer=new,
                      low_yield=low, main=filled)


def potential(capacity: int, *, pool_seeds: int, main: Mapping[str, tuple[float, float]], hotkey: str,
              weight_h: float, newcomer_demand: float, low_yield_demand: float,
              takes_spare: bool = NEWCOMER_TAKES_SPARE) -> float:
    """What MAIN ``hotkey`` would get if it asked for everything, the others' demands as measured: the number the
    miner is told (an allocation capped by the miner's own requests would pin it to what it holds).
    Admission still uses ``allocate`` (its current demand), so the potential is advice, never a grant."""
    claims = {key: value for key, value in main.items() if key != hotkey}
    claims[hotkey] = (float(weight_h), float(max(0, int(capacity))))
    return allocate(capacity, pool_seeds=pool_seeds, main=claims, newcomer_demand=newcomer_demand,
                    low_yield_demand=low_yield_demand, takes_spare=takes_spare).main[hotkey]


def sessions_of(allocation: float) -> int:
    """Sessions an allocation lets a holder count (a floor: k parts never sum above the whole)."""
    return math.floor(float(allocation) + 1e-9)


def opens_for(sessions: int) -> int:
    return OPENS_PER_LIVE_PER_HOUR * int(sessions)


class SessionTiers:
    """What one RL session of each env reserves on its machine: the fleet size the quota shares is counted in such
    sessions (``Fleet.session_room(reservations=)``), so an allocation never exceeds what placement can host.
    ``policies``: {env name: its episode policy}, read at each call (an env added later counts at once).

    SEEDED at start (``seed``) from a sample of the env's own tasks: the reservation is the ``TIER_QUANTILE`` of
    their CPUs (a task declaring none takes the whole box: it ranks above every figure) and of their memory budgets
    (the policy's, raised by the task). Frozen once seeded: no miner's precommit on a rare task moves it; the tasks
    resolved above it are only counted (``view``). Until an env is seeded (its source cannot be sampled), it is
    LEARNT from the tasks resolved so far: the largest seen, the whole box before any (never smaller than what was
    seen)."""

    def __init__(self, policies: Mapping[str, object]) -> None:
        self._policies = policies
        self._cpu: dict[str, int | None] = {}            # learnt: env -> largest cpu_millis seen (None: a box)
        self._memory: dict[str, int] = {}                # learnt: env -> largest memory budget seen
        self._seeded: dict[str, dict] = {}               # env -> {"cpu_millis", "memory_bytes", "tasks", "above"}

    def _memory_of(self, environment: str, task: object) -> int:
        memory = int(self._policies[environment].budgets_dict().get("memory_bytes", 0))
        limits = getattr(task, "limits", None) or {}
        if isinstance(limits.get("memory_bytes"), int):
            memory = max(memory, int(limits["memory_bytes"]))
        return memory

    def seed(self, environment: str, tasks: list) -> None:
        """Freeze the env's reservation at the ``TIER_QUANTILE`` of these tasks (a sample of its own)."""
        if environment not in self._policies or not tasks:
            return
        def at_quantile(values: list) -> Any:
            ordered = sorted(values, key=lambda v: math.inf if v is None else v)
            return ordered[max(0, math.ceil(TIER_QUANTILE * len(ordered)) - 1)]
        cpus = [None if getattr(t, "cpu_millis", None) is None else int(t.cpu_millis) for t in tasks]
        self._seeded[environment] = {"cpu_millis": at_quantile(cpus),
                                     "memory_bytes": at_quantile([self._memory_of(environment, t) for t in tasks]),
                                     "tasks": len(tasks), "above": 0}

    def seeded(self, environment: str) -> bool:
        return environment in self._seeded

    def observe(self, environment: str, task: object) -> None:
        """A task of ``environment`` was resolved."""
        if environment not in self._policies:
            return
        cpu = getattr(task, "cpu_millis", None)
        seeded = self._seeded.get(environment)
        if seeded is not None:
            tier = seeded["cpu_millis"]
            if (tier is not None and (cpu is None or int(cpu) > tier)) or (
                    self._memory_of(environment, task) > seeded["memory_bytes"]):
                seeded["above"] += 1                     # counted, never moving the frozen tier
            return
        if environment not in self._cpu:
            self._cpu[environment] = None if cpu is None else int(cpu)
        elif self._cpu[environment] is not None:
            self._cpu[environment] = None if cpu is None else max(int(cpu), self._cpu[environment])
        self._memory[environment] = max(self._memory_of(environment, task), self._memory.get(environment, 0))

    def reservation(self, environment: str) -> dict:
        seeded = self._seeded.get(environment)
        if seeded is not None:
            return {"cpu_millis": seeded["cpu_millis"], "memory_bytes": seeded["memory_bytes"]}
        policy = self._policies[environment]
        memory = max(int(policy.budgets_dict().get("memory_bytes", 0)), self._memory.get(environment, 0))
        return {"cpu_millis": self._cpu.get(environment), "memory_bytes": memory}

    def reservations(self, names: Collection[str] | None = None) -> dict[tuple[str, str], dict]:
        """{(sandbox env, env package): reservation} for ``Fleet.session_room``; two envs on one sandbox env and
        package keep the larger reservation. ``names``: only these envs count (None: every env)."""
        found: dict[tuple[str, str], dict] = {}
        for name, policy in list(self._policies.items()):
            if names is not None and name not in names:
                continue
            key, mine = (policy.sandbox_env, policy.env_package), self.reservation(name)
            other = found.get(key)
            if other is not None:
                cpu = (None if other["cpu_millis"] is None or mine["cpu_millis"] is None
                       else max(other["cpu_millis"], mine["cpu_millis"]))
                mine = {"cpu_millis": cpu, "memory_bytes": max(other["memory_bytes"], mine["memory_bytes"])}
            found[key] = mine
        return found

    def view(self) -> dict[str, dict]:
        """Per env: the reservation the fleet is counted in, whether it was seeded, from how many tasks, and how
        many resolved tasks lie above it (the operator's view)."""
        found = {}
        for name in list(self._policies):
            seeded = self._seeded.get(name)
            found[name] = {**self.reservation(name), "seeded": seeded is not None,
                           "tasks": None if seeded is None else seeded["tasks"],
                           "above": None if seeded is None else seeded["above"]}
        return found


__all__ = [
    "ALLOCATION_REFRESH_S",
    "ASK_FORGET_S",
    "CAPACITY_VERDICT_ALERT_SESSIONS",
    "CAPACITY_VERDICT_ALERT_SHARE",
    "CAPACITY_VERDICT_TOP",
    "CHECKER_BUSY_ALERT_PER_MIN",
    "DEMAND_TTL_S",
    "GRACE_SESSIONS",
    "LOW_YIELD",
    "LOW_YIELD_MAX_GROUPS",
    "LOW_YIELD_OPENS_PERIOD_S",
    "LOW_YIELD_SHARE",
    "MAIN",
    "MAX_CHECKS_RUNNING",
    "MIN_SERVING_MACHINES",
    "NEWCOMER",
    "NEWCOMER_ENTITY",
    "NEWCOMER_EXIT_GROUPS",
    "NEWCOMER_MAX_GROUPS",
    "NEWCOMER_SHARE",
    "NEWCOMER_TAKES_SPARE",
    "NEWCOMER_WEIGHT",
    "OPENS_PERIOD_S",
    "OPENS_PER_LIVE_PER_HOUR",
    "PRESENT_WINDOW_S",
    "QUEUE_AGING_FORGET_S",
    "QUEUE_AGING_S",
    "QUEUE_FAIR_EVERY",
    "QUEUE_HOLD_AFTER_S",
    "QUEUE_MAX_PER_KEY",
    "QUEUE_MAX_WAITING",
    "QUEUE_WAIT_S",
    "SATURATION_ALERT_FOR_S",
    "SATURATION_ALERT_FREE_FRACTION",
    "SATURATION_ALERT_REFUSALS_PER_MIN",
    "SATURATION_ALERT_WAIT_S",
    "SATURATION_CHECK_S",
    "SATURATION_WINDOW_S",
    "USEFUL_HORIZON_S",
    "YIELD_FLOOR",
    "YIELD_FULL",
    "YIELD_MIN_DECIDED",
    "YIELD_SESSIONS",
    "Allocation",
    "SessionTiers",
    "YieldStats",
    "allocate",
    "entity_sizes",
    "multiplier",
    "opens_for",
    "potential",
    "sessions_of",
    "standing",
    "water_fill",
    "weight",
]
