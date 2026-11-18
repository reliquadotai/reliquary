"""Leases at the consumer's pace, cheapest miners full (docs/design/2026-11-17-lease-allocation.md).

Three pieces, each kept free of the auction's own lock so it can be tested alone:

* ``SpeedBook`` measures each hotkey's speed with two counters: output tokens accepted, and leases held integrated
  over time. ``v_i = tokens over the window / lease-seconds held over the window``: every held lease counts,
  delivered or not, so a hotkey sitting on leases sees its speed drop (3.3).
* ``Allowances`` turns a declared capacity into the slots a hotkey is trusted with: its whole capacity at once, a
  first verdict after a couple of sample times that brings it straight down to what it delivers, then +1 per
  delivery while full and fast enough, -20 % per step under ``v_min`` (3.3).
* ``MarketPace`` (one per market, shared by every order on its model) sets the rate the audit card can take,
  ``A x k``, and splits it between the orders: each its floor (remaining work over the time to its deadline),
  the surplus to the order closest to done (3.1, 3.7, 3.11).

``merit_order`` walks the offers cheapest first, filling the job's target rate in measured tokens per second;
only the last needed hotkey is partly served and its offer is the clearing price (3.4, 3.8).
"""
from __future__ import annotations

import collections
import math
import os
import statistics
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

# Which markets allocate by merit order (``market_key`` substrings, comma separated, or "all"); unset: none.
MERIT_ENV = "RELIQUARY_MERIT_ORDER"

SPEED_WINDOW_SECONDS = 600.0
CHECKPOINT_SECONDS = 5.0
# A speed is trusted over this many deliveries in its window.
MIN_DELIVERIES = 8
# v_min, as a share of the order's median speed: the bulk only cuts leases held without being worked, the tail
# keeps the last prompts on the fast miners.
VMIN_BULK = 0.25
VMIN_TAIL = 0.5
TAIL_SHARE = 0.10
# First verdict: after this many median sample times (at least VERDICT_MIN_SECONDS) and MIN_DELIVERIES since
# the hotkey was first needed; well under (VERDICT_SHORT of) what its capacity at the median speed would deliver,
# its allowance drops to what it delivers.
VERDICT_SAMPLE_TIMES = 2.0
VERDICT_MIN_SECONDS = 60.0
VERDICT_MAX_SECONDS = 1800.0
VERDICT_SHORT = 0.5
CUT = 0.8
STEP_SECONDS = 30.0
FULL = 0.9

# MarketPace: the audit queue in seconds of the card's work (wait), corrected around TARGET_WAIT_SECONDS.
TARGET_WAIT_SECONDS = 60.0
BUSY_HIGH = 0.95
BUSY_LOW = 0.85
K_MIN, K_MAX = 0.05, 4.0
K_GROW = 1.15
K_SHRINK_FLOOR = 0.5
DEFAULT_DEADLINE_SECONDS = 24 * 3600.0
MIN_TIME_LEFT_SECONDS = 600.0
UNMET_SECONDS = 2 * STEP_SECONDS
# Every offer needed (the target not reached): the price is the offer under which this share of the measured
# supply is offered; dearer offers are paid their own. One small dear hotkey cannot price every lease.
SHORT_SUPPLY_QUANTILE = 0.9


def merit_enabled(market_key: str | None, env: Mapping[str, str] | None = None) -> bool:
    """Whether ``market_key`` allocates by merit order (``RELIQUARY_MERIT_ORDER``)."""
    value = (env if env is not None else os.environ).get(MERIT_ENV, "").strip()
    if not value or value.lower() in {"0", "off", "false", "none"}:
        return False
    if value.lower() in {"all", "1", "on", "true"}:
        return True
    if market_key is None:
        return False
    return any(part.strip() and part.strip() in market_key for part in value.split(","))


# -- speeds -------------------------------------------------------------------------------------------------------


@dataclass
class _Counters:
    tokens: float = 0.0          # output tokens accepted, cumulated
    area: float = 0.0            # leases held x seconds, cumulated
    deliveries: int = 0
    held: int = 0
    at: float | None = None      # when ``held`` was last read
    marks: collections.deque = field(default_factory=collections.deque)   # (t, tokens, area, deliveries)


class SpeedBook:
    """Each hotkey's output tokens per lease-second over the last ``window`` seconds."""

    def __init__(self, window: float = SPEED_WINDOW_SECONDS) -> None:
        self._window = float(window)
        self._hotkeys: dict[str, _Counters] = {}

    def _of(self, hotkey: str) -> _Counters:
        return self._hotkeys.setdefault(hotkey, _Counters())

    @staticmethod
    def _integrate(c: _Counters, now: float) -> None:
        if c.at is not None and now > c.at:
            c.area += c.held * (now - c.at)
        c.at = now if c.at is None else max(c.at, now)

    def tick(self, now: float, held: Mapping[str, int]) -> None:
        """The leases each hotkey holds now (hotkeys not named hold none)."""
        for hotkey in set(held) | set(self._hotkeys):
            c = self._of(hotkey)
            self._integrate(c, now)
            c.held = max(0, int(held.get(hotkey, 0)))
            if not c.marks or now - c.marks[-1][0] >= CHECKPOINT_SECONDS:
                c.marks.append((now, c.tokens, c.area, c.deliveries))
            while len(c.marks) > 2 and c.marks[1][0] <= now - self._window:
                c.marks.popleft()
        for hotkey in [h for h, c in self._hotkeys.items()
                       if c.held == 0 and c.marks and c.marks[-1][1] == c.marks[0][1]
                       and now - c.marks[0][0] > 2 * self._window]:
            del self._hotkeys[hotkey]

    def delivered(self, hotkey: str, now: float, tokens: float) -> None:
        c = self._of(hotkey)
        self._integrate(c, now)
        c.tokens += max(0.0, float(tokens))
        c.deliveries += 1

    def window(self, hotkey: str, now: float) -> tuple[float, float, int, float] | None:
        """(tokens, lease-seconds, deliveries, seconds) over the window, or None before two marks."""
        c = self._hotkeys.get(hotkey)
        if c is None or not c.marks:
            return None
        self._integrate(c, now)
        start = c.marks[0]
        for mark in c.marks:
            if mark[0] <= now - self._window:
                start = mark
            else:
                break
        seconds = now - start[0]
        if seconds <= 0:
            return None
        return c.tokens - start[1], c.area - start[2], c.deliveries - start[3], seconds

    def speed(self, hotkey: str, now: float) -> float | None:
        """Output tokens per lease-second held, once MIN_DELIVERIES are in the window."""
        measured = self.window(hotkey, now)
        if measured is None:
            return None
        tokens, area, deliveries, _ = measured
        if deliveries < MIN_DELIVERIES or area <= 0:
            return None
        return tokens / area

    def rate(self, hotkey: str, now: float) -> float | None:
        """Output tokens per second over the window."""
        measured = self.window(hotkey, now)
        return None if measured is None else measured[0] / measured[3]

    def median(self, now: float) -> float | None:
        speeds = [v for v in (self.speed(h, now) for h in list(self._hotkeys)) if v is not None]
        return statistics.median(speeds) if speeds else None

    def totals(self, hotkey: str) -> tuple[float, int]:
        c = self._hotkeys.get(hotkey)
        return (0.0, 0) if c is None else (c.tokens, c.deliveries)


# -- allowances ---------------------------------------------------------------------------------------------------


@dataclass
class _Trust:
    since: float                              # first needed
    tokens0: float                            # its counters then
    deliveries0: int
    allowance: int | None = None              # None: its declared capacity
    judged: bool = False
    stepped_at: float = 0.0
    deliveries_at_step: int = 0


class Allowances:
    """The slots each hotkey is trusted with: declared at once, then measured (3.3)."""

    def __init__(self) -> None:
        self._trust: dict[str, _Trust] = {}

    def of(self, hotkey: str, declared: int) -> int:
        trust = self._trust.get(hotkey)
        if trust is None or trust.allowance is None:
            return declared
        return max(1, min(declared, trust.allowance))

    def needed(self, hotkey: str, now: float, speeds: SpeedBook) -> None:
        """``hotkey`` is among the needed now: its first verdict counts from the first time."""
        if hotkey not in self._trust:
            tokens, deliveries = speeds.totals(hotkey)
            self._trust[hotkey] = _Trust(since=now, tokens0=tokens, deliveries0=deliveries, stepped_at=now,
                                         deliveries_at_step=deliveries)

    def forget(self, hotkey: str) -> None:
        self._trust.pop(hotkey, None)

    def step(self, hotkey: str, now: float, *, declared: int, held: int, speeds: SpeedBook,
             median: float | None, median_seconds: float | None, tail: bool) -> str | None:
        """Re-judge ``hotkey``; what changed (for the log), or None."""
        trust = self._trust.get(hotkey)
        if trust is None:
            return None
        current = self.of(hotkey, declared)
        tokens, deliveries = speeds.totals(hotkey)
        if not trust.judged:
            wait = VERDICT_SAMPLE_TIMES * (median_seconds or VERDICT_MIN_SECONDS)
            wait = min(VERDICT_MAX_SECONDS, max(VERDICT_MIN_SECONDS, wait))
            elapsed = now - trust.since
            if elapsed < wait or (deliveries - trust.deliveries0 < MIN_DELIVERIES and elapsed < 2 * wait):
                return None
            trust.judged = True
            trust.stepped_at, trust.deliveries_at_step = now, deliveries
            if median is None or median <= 0:
                return None
            rate = (tokens - trust.tokens0) / elapsed
            expected = current * median
            if rate < VERDICT_SHORT * expected:
                trust.allowance = max(1, math.ceil(rate / median))
                return f"verdict {current}->{trust.allowance} (rate {rate:.0f} vs {expected:.0f} tok/s expected)"
            return None
        if now - trust.stepped_at < STEP_SECONDS:
            return None
        delivered = deliveries - trust.deliveries_at_step
        trust.stepped_at, trust.deliveries_at_step = now, deliveries
        speed = speeds.speed(hotkey, now)
        if speed is None or median is None:
            return None
        v_min = vmin(median, tail)
        if speed < v_min and held >= FULL * current:
            trust.allowance = max(1, int(current * CUT))
            return f"cut {current}->{trust.allowance} (v {speed:.2f} < v_min {v_min:.2f})"
        if held >= current and current < declared and delivered > 0:
            trust.allowance = min(declared, current + delivered)
            if trust.allowance >= declared:
                trust.allowance = None
            return f"grow {current}->{self.of(hotkey, declared)}"
        return None

    def snapshot(self) -> dict:
        return {h: [t.since, t.tokens0, t.deliveries0, t.allowance, t.judged] for h, t in self._trust.items()}

    def restore(self, state: Mapping, now: float) -> None:
        for hotkey, row in (state or {}).items():
            try:
                since, tokens0, deliveries0, allowance, judged = row
                # Counters restart at 0 after a restart: the verdict counts afresh, the allowance stays.
                self._trust[str(hotkey)] = _Trust(since=now, tokens0=0.0, deliveries0=0,
                                                  allowance=None if allowance is None else int(allowance),
                                                  judged=bool(judged), stepped_at=now)
            except (TypeError, ValueError):
                continue


def vmin(median: float, tail: bool) -> float:
    """The speed under which a full hotkey is cut: a share of the order's median. No absolute floor: an agentic
    order's tokens per lease-second include its tool time, and the median already follows the task."""
    return (VMIN_TAIL if tail else VMIN_BULK) * median


# -- the merit order ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    hotkey: str
    price: float          # its offer
    slots: int            # what it is trusted with
    speed: float | None   # tokens per lease-second, None: not measured


@dataclass(frozen=True, slots=True)
class Allocation:
    slots: dict[str, int]
    clearing: float | None        # the last needed hotkey's offer
    reached: bool                 # the target was reached (else every offer is needed)
    supply: float                 # tokens per second every candidate together would deliver


def merit_order(candidates: Iterable[Candidate], target: float, median: float) -> Allocation:
    """Fill ``target`` tokens per second cheapest first; a hotkey not measured counts at ``median``.

    The clearing price: the last needed offer when the target is reached. When it is not (every offer needed),
    the offer under which SHORT_SUPPLY_QUANTILE of the measured supply is offered: with every offer needed, the
    dearest one would price every lease, and a hotkey offering the ceiling for one slot would set it (the lever
    #542 closed). Unmeasured hotkeys do not count there: a fresh capacity only moves the price once it delivers."""
    ordered = sorted(candidates, key=lambda c: (c.price, c.hotkey))
    slots: dict[str, int] = {}
    reached = False
    clearing = None
    filled = 0.0
    supply = 0.0
    for c in ordered:
        speed = c.speed if c.speed is not None and c.speed > 0 else median
        supply += c.slots * speed
        if reached or c.slots <= 0:
            continue
        missing = target - filled
        if c.slots * speed >= missing:
            slots[c.hotkey] = max(1, min(c.slots, math.ceil(missing / speed)))
            filled += slots[c.hotkey] * speed
            clearing = c.price
            reached = True
            continue
        slots[c.hotkey] = c.slots
        filled += c.slots * speed
        clearing = c.price
    if not reached and slots:
        measured = [(c.price, c.slots * c.speed) for c in ordered
                    if slots.get(c.hotkey, 0) > 0 and c.speed is not None and c.speed > 0]
        total = sum(rate for _, rate in measured)
        if total > 0:
            reach, summed = SHORT_SUPPLY_QUANTILE * total * (1.0 - 1e-9), 0.0
            for price, rate in measured:
                summed += rate
                if summed >= reach:
                    clearing = price
                    break
    return Allocation(slots=slots, clearing=clearing, reached=reached, supply=supply)


# -- the market's pace --------------------------------------------------------------------------------------------


@dataclass
class _JobReport:
    q: float = 1.0
    remaining_tokens: float | None = None
    deadline: float | None = None
    supply: float | None = None
    backlog_tokens: float = 0.0
    unmet_at: float | None = None
    at: float = 0.0


class MarketPace:
    """The rate a market's audit can take (``A x k``) and its split between the orders on it."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobReport] = {}
        self._first_seen: dict[str, float] = {}
        self._audit: Callable[[], tuple[float, float | None] | None] | None = None
        self.k = 1.0
        self._wait: float | None = None
        self._rate: float | None = None
        self._steered_at: float | None = None
        self._targets: dict[str, float] = {}
        self.last: dict = {}

    def bind_audit(self, audit: Callable[[], tuple[float, float | None] | None]) -> None:
        """``audit()``: (output tokens per second the market's audit cards verify together, their busy share or
        None), or None while not measured."""
        with self._lock:
            self._audit = audit

    def report(self, job: str, *, q: float, remaining_tokens: float | None, deadline: float | None,
               supply: float | None, backlog_tokens: float, unmet_at: float | None) -> None:
        now = self._clock()
        with self._lock:
            self._first_seen.setdefault(job, now)
            self._jobs[job] = _JobReport(q=max(1e-3, min(1.0, float(q))), remaining_tokens=remaining_tokens,
                                         deadline=deadline, supply=supply, backlog_tokens=max(0.0, backlog_tokens),
                                         unmet_at=unmet_at, at=now)

    def forget(self, job: str) -> None:
        with self._lock:
            self._jobs.pop(job, None)
            self._targets.pop(job, None)

    def target(self, job: str) -> float | None:
        """``job``'s target rate (output tokens per second), None while the audit is not measured."""
        audit = None
        if self._audit is not None:
            try:
                audit = self._audit()
            except Exception:
                audit = None
        with self._lock:
            now = self._clock()
            if self._steered_at is None or now - self._steered_at >= STEP_SECONDS:
                self._steered_at = now
                self._steer(now, audit)
            return self._targets.get(job)

    def _steer(self, now: float, audit) -> None:
        for job in [j for j, r in self._jobs.items() if now - r.at > 10 * STEP_SECONDS]:
            del self._jobs[job]
        if audit is not None and audit[0] is not None and audit[0] > 0:
            self._rate = float(audit[0])
            busy = audit[1]
        else:
            # An idle card is not measured: its last measure stands (a card that idles has room).
            busy = 0.0 if self._rate is not None else None
        if self._rate is None or not self._jobs:
            self._targets = {}
            self.last = {"audit_rate": None, "k": self.k}
            return
        rate = self._rate
        wait = sum(r.backlog_tokens for r in self._jobs.values()) / rate
        unmet = any(r.unmet_at is not None and now - r.unmet_at <= UNMET_SECONDS for r in self._jobs.values())
        before = self.k
        if wait > TARGET_WAIT_SECONDS and ((self._wait is not None and wait > self._wait)
                                           or (busy is not None and busy > BUSY_HIGH)):
            self.k *= max(K_SHRINK_FLOOR, TARGET_WAIT_SECONDS / wait)
        elif (wait < TARGET_WAIT_SECONDS / 2 and unmet and (busy is None or busy < BUSY_LOW)):
            self.k *= K_GROW
        self.k = min(K_MAX, max(K_MIN, self.k))
        self._wait = wait
        self._targets = split(self._jobs, rate * self.k, now, self._first_seen)
        self.last = {"audit_rate": rate, "busy": busy, "wait_seconds": wait, "k": self.k, "k_before": before,
                     "targets": dict(self._targets)}


def split(jobs: Mapping[str, _JobReport], audit_rate: float, now: float,
          first_seen: Mapping[str, float]) -> dict[str, float]:
    """Each order's rate: its floor (remaining work over the time to its deadline), then the surplus of audit
    to the order closest to done, up to what its offers supply, then the next (3.11)."""
    floors: dict[str, float] = {}
    for job, r in jobs.items():
        if r.remaining_tokens is None:
            floors[job] = 0.0
            continue
        deadline = r.deadline if r.deadline is not None else first_seen.get(job, now) + DEFAULT_DEADLINE_SECONDS
        floors[job] = r.remaining_tokens / max(MIN_TIME_LEFT_SECONDS, deadline - now)
    used = sum(floors[j] * jobs[j].q for j in jobs)
    if used >= audit_rate:
        scale = audit_rate / used if used > 0 else 0.0
        return {j: floors[j] * scale for j in jobs}
    rates = dict(floors)
    left = audit_rate - used
    order = sorted(jobs, key=lambda j: (math.inf if jobs[j].remaining_tokens is None else jobs[j].remaining_tokens, j))
    for job in order:
        if left <= 0:
            break
        r = jobs[job]
        room = math.inf if r.supply is None else max(0.0, r.supply - rates[job])
        extra = min(room, left / r.q)
        rates[job] += extra
        left -= extra * r.q
    if left > 0:
        # Every order is supply-bound: the rest goes to all in proportion of their audit share, so a new offer
        # anywhere finds room.
        for job in jobs:
            rates[job] += left / len(jobs) / jobs[job].q
    return rates


_PACES: dict[str, MarketPace] = {}
_PACES_LOCK = threading.Lock()


def market_pace(key: str, clock: Callable[[], float]) -> MarketPace:
    with _PACES_LOCK:
        return _PACES.setdefault(str(key), MarketPace(clock))


__all__ = ["Allocation", "Allowances", "Candidate", "MERIT_ENV", "MarketPace", "SpeedBook", "market_pace",
           "merit_enabled", "merit_order", "split", "vmin"]
