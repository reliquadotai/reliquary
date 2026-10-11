"""The fair open queue of RL sandbox sessions at saturation.

Only an open that some fresh machine could take waits here (the issuer refuses a never-placeable open at once,
``task_unplaceable``), never holding the issuer lock and never in a worker thread. Waiters are grouped by
placement class (the issuer's ``(env, cpu_millis, memory_bytes)``: not the image, so the many per-task images of
one env share a class, and each waiter's own attempt filters on its image): a new open places directly when no
waiter of its own class waits, so a class nobody can serve now never blocks another.

ONE turn is outstanding at a time. Each wake-up (``notify``: a fleet poll, a session closing, lapsing or
voided) starts a sweep: the best untried waiter gets the turn; a waiter that finds no room (``missed``) passes
it at once to the next untried one; one that placed or left (``done``) passes it on too (room may remain).
When every waiter has tried, nobody holds the turn until the next wake-up; a newcomer whose class already
missed in this sweep waits for the next one too. Wake-ups that arrive while a sweep runs are coalesced into ONE
follow-up sweep when it ends (a burst of closes costs one more sweep, not one per close). Within a sweep, once
a waiter of a given room key (``missed(room=...)``: the env, image and reservation) found no room, the others of
that key are passed over without an attempt (``room_missed``): nothing freed in between.

No starvation of big reservations: when the head waiter (the best priority of all tickets) misses after its key
has waited ``hold_after_s``, the issuer holds one machine for it (``hold``): until it places or leaves, nobody
else places there (``held_for_others``), so freed room accumulates where it can fit, while other opens keep
placing on the other machines.

Priority = ``score x (1 + key_waited / aging_s)``: ``score`` = a MAIN hotkey's weight (useful volume x yield
multiplier) from ``rank``; the key's waiting time starts at its first ticket and is kept across retries (each
HTTP retry is a new ticket), forgotten ``aging_forget_s`` after its last ticket left, reset when it places. One
PLACEMENT in ``fair_every`` goes to a POOLED waiter (``rank``'s third element: a newcomer's, a low-yield or a
small hotkey's, score 0): once ``fair_every - 1`` MAIN tickets placed since the last pooled placement, the turn
goes to the untried pooled waiter whose key was served least recently, until one places (a pooled miss does not
use the pooled share up); with no pooled waiter it goes by priority. A MAIN key never gets a turn out of its
weight: a per-key fair turn would pay for more hotkeys, while pooled tickets are bounded by their shared pool.

Bounds: ``max_waiting`` in all; per key ``min(max_per_key, headroom(hotkey))`` (the quota's room for more live
sessions); one ticket per engagement. A waiter leaves at its deadline with a clear refusal. Everything runs on
the event loop (no lock needed)."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Hashable

from reliquary.sandbox.rl_capacity import (
    QUEUE_AGING_FORGET_S,
    QUEUE_AGING_S,
    QUEUE_FAIR_EVERY,
    QUEUE_HOLD_AFTER_S,
    QUEUE_MAX_PER_KEY,
    QUEUE_MAX_WAITING,
    QUEUE_WAIT_S,
)
from reliquary.sandbox.sessions import Refusal

logger = logging.getLogger(__name__)


class Ticket:
    __slots__ = ("active", "cls", "engagement", "event", "granted", "joined", "key", "pooled", "score", "tried")

    def __init__(self, key: str, score: float, cls: Hashable, engagement: str | None, joined: float,
                 pooled: bool = False) -> None:
        self.key, self.score, self.cls, self.engagement = key, float(score), cls, engagement
        self.pooled = bool(pooled)
        self.joined = float(joined)
        self.event = asyncio.Event()
        self.granted = False
        self.active = True
        self.tried = -1


class OpenQueue:
    def __init__(self, *, rank: Callable[[str], tuple[str, float, bool]], headroom: Callable[[str], int] | None = None,
                 wait_s: float = QUEUE_WAIT_S, max_waiting: int = QUEUE_MAX_WAITING,
                 max_per_key: int = QUEUE_MAX_PER_KEY, aging_s: float = QUEUE_AGING_S,
                 aging_forget_s: float = QUEUE_AGING_FORGET_S, fair_every: int = QUEUE_FAIR_EVERY,
                 hold_after_s: float | None = QUEUE_HOLD_AFTER_S,
                 clock: Callable[[], float] = time.monotonic,
                 on_wait: Callable[[float, Hashable], None] | None = None) -> None:
        self._rank, self._headroom = rank, headroom
        # Told how long each ticket waited, and its placement class, when it leaves (placed, refused, timed out or
        # cancelled): the metrics.
        self._on_wait = on_wait
        self._wait_s, self._aging_s, self._forget_s = float(wait_s), float(aging_s), float(aging_forget_s)
        self._max_waiting, self._max_per_key, self._fair_every = int(max_waiting), int(max_per_key), int(fair_every)
        self._hold_after_s = None if hold_after_s is None else float(hold_after_s)
        self._clock = clock
        self._waiting: list[Ticket] = []
        self._holder: Ticket | None = None
        self._sweep = 0
        self._dirty = False                      # a wake-up arrived during the running sweep
        self._room_missed: set[Hashable] = set()  # room keys that found no room in this sweep
        self._main_placed = 0                    # MAIN placements since the last pooled one
        self._hold: Ticket | None = None         # the head waiter a machine is held for
        self._hold_target: Hashable | None = None
        self._since: dict[str, float] = {}       # key -> when it started waiting (kept across retries)
        self._left: dict[str, float] = {}        # key -> when its last ticket left
        self._served: dict[str, float] = {}      # key -> its last placement

    @property
    def busy(self) -> bool:
        return bool(self._waiting) or self._holder is not None

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def _tickets(self) -> list[Ticket]:
        return self._waiting + ([self._holder] if self._holder is not None else [])

    def keys(self) -> list[str]:
        """The keys of the waiting tickets (the turn's holder excluded): who fills the queue."""
        return [ticket.key for ticket in self._waiting]

    def tickets(self) -> list[tuple[str, str | None]]:
        """``(key, engagement)`` of the waiting tickets (the turn's holder excluded)."""
        return [(ticket.key, ticket.engagement) for ticket in self._waiting]

    def waiting_in(self, cls: Hashable) -> bool:
        return any(ticket.cls == cls for ticket in self._tickets())

    def deadline(self) -> float:
        return self._clock() + self._wait_s

    def waited(self, ticket: Ticket) -> float:
        return self._clock() - ticket.joined

    def _retry_after(self) -> int:
        return max(1, math.ceil(self._wait_s))

    def _forget(self, now: float) -> None:
        present = {ticket.key for ticket in self._tickets()}
        for key, left in list(self._left.items()):
            if key in present:
                del self._left[key]
            elif now - left > self._forget_s:
                del self._left[key]
                self._since.pop(key, None)

    def join(self, hotkey: str, *, cls: Hashable, engagement: str | None = None) -> Ticket | Refusal:
        key, score, pooled = self._rank(hotkey)
        tickets = self._tickets()
        if len(self._waiting) >= self._max_waiting:
            return Refusal("open_queue_full", {"waiting": len(self._waiting), "max": self._max_waiting},
                           retry_after=self._retry_after())
        if engagement is not None and any(ticket.engagement == engagement for ticket in tickets):
            return Refusal("open_queued", {"engagement": engagement}, retry_after=self._retry_after())
        bound = self._max_per_key
        if self._headroom is not None:
            bound = min(bound, max(0, int(self._headroom(hotkey))))       # no free ticket at zero headroom
        mine = sum(1 for ticket in tickets if ticket.key == key)
        if mine >= bound:
            return Refusal("open_queue_key_full", {"waiting": mine, "max": bound}, retry_after=self._retry_after())
        now = self._clock()
        self._forget(now)
        self._since.setdefault(key, now)
        self._left.pop(key, None)
        ticket = Ticket(key, score, cls, engagement, now, pooled=pooled)
        if any(t.cls == cls and t.tried == self._sweep for t in self._waiting):
            # A waiter of its class already found no room in this sweep: the newcomer waits for the next
            # wake-up too (it would not fit either, and it must not jump the older waiter when room appears).
            ticket.tried = self._sweep
        self._waiting.append(ticket)
        if self._holder is None:
            self._grant_next(now)
        return ticket

    def _priority(self, ticket: Ticket, now: float) -> float:
        waited = max(0.0, now - self._since.get(ticket.key, ticket.joined))
        return ticket.score * (1.0 + waited / self._aging_s)

    def _new_sweep(self) -> None:
        self._sweep += 1
        self._dirty = False
        self._room_missed.clear()

    def _grant_next(self, now: float) -> None:
        if self._holder is not None:
            return
        candidates = [ticket for ticket in self._waiting if ticket.tried != self._sweep]
        if not candidates and self._dirty:
            self._new_sweep()                    # the coalesced wake-ups: one follow-up sweep
            candidates = list(self._waiting)
        if not candidates:
            return
        pooled = [ticket for ticket in candidates if ticket.pooled]
        if self._fair_every and self._main_placed >= self._fair_every - 1 and pooled:
            # The fair turn: pooled tickets only (their total is bounded by their shared pool), round-robin.
            best = min(pooled, key=lambda t: (self._served.get(t.key, -math.inf), t.joined))
        else:
            best = max(candidates, key=lambda t: (self._priority(t, now), -t.joined))
        self._waiting.remove(best)
        best.granted = True
        self._holder = best
        best.event.set()

    async def turn(self, ticket: Ticket, deadline: float) -> bool:
        """True once the ticket holds the turn; False at the deadline (the ticket has then left)."""
        if ticket.granted:
            return True
        remaining = deadline - self._clock()
        if remaining > 0:
            try:
                await asyncio.wait_for(ticket.event.wait(), remaining)
                return True
            except TimeoutError:
                pass
        if ticket.granted:
            return True
        self.done(ticket)
        return False

    def room_missed(self, room: Hashable) -> bool:
        """Whether a waiter of this room key already found no room in this sweep (the caller skips its attempt)."""
        return room in self._room_missed

    def should_hold(self, ticket: Ticket) -> bool:
        """Whether a machine should now be held for this ticket, which just found no room: it is the head waiter
        (the best priority of all tickets) and its key has waited ``hold_after_s``. A hold already held for it
        is renewed (its target may move)."""
        if self._hold_after_s is None or not ticket.active:
            return False
        if self._hold is not None and self._hold is not ticket and self._hold.active:
            return False
        now = self._clock()
        if now - self._since.get(ticket.key, ticket.joined) < self._hold_after_s:
            return False
        best = max(self._tickets(), key=lambda t: (self._priority(t, now), -t.joined))
        return best is ticket

    def hold(self, ticket: Ticket, target: Hashable | None) -> None:
        """Hold ``target`` (a machine) for ``ticket`` until it places or leaves (None: nothing to hold)."""
        if target is None:
            return
        self._hold, self._hold_target = ticket, target

    def held_for_others(self, ticket: Ticket | None) -> frozenset:
        """The machines an open of ``ticket`` (None: a direct newcomer) must not take."""
        hold = self._hold
        if hold is None or hold is ticket or not hold.active:
            return frozenset()
        return frozenset((self._hold_target,))

    def missed(self, ticket: Ticket, *, room: Hashable | None = None) -> None:
        """The turn found no room: the ticket waits again (tried in this sweep) and the next untried waiter of
        the sweep gets the turn at once. ``room``: its room key, passed over for the rest of the sweep."""
        if room is not None and ticket.tried != self._sweep:
            self._room_missed.add(room)
        if self._holder is ticket:
            self._holder = None
        ticket.granted = False
        ticket.tried = self._sweep
        ticket.event = asyncio.Event()
        if ticket.active and ticket not in self._waiting:
            self._waiting.append(ticket)
        self._grant_next(self._clock())

    def done(self, ticket: Ticket, *, placed: bool = False) -> None:
        """The ticket leaves (placed, refused, timed out or cancelled); a turn it held passes on. Idempotent."""
        if not ticket.active:
            return
        ticket.active = False
        now = self._clock()
        if self._on_wait is not None:
            try:
                self._on_wait(max(0.0, now - ticket.joined), ticket.cls)
            except Exception:
                logger.exception("the open queue's wait listener failed")
        if ticket in self._waiting:
            self._waiting.remove(ticket)
        held = self._holder is ticket
        if held:
            self._holder = None
        if self._hold is ticket:
            self._hold, self._hold_target = None, None
        others = any(t.key == ticket.key for t in self._tickets())
        if placed:
            self._served[ticket.key] = now
            self._main_placed = 0 if ticket.pooled else self._main_placed + 1
            if others:
                self._since[ticket.key] = now
            else:
                self._since.pop(ticket.key, None)
        if not others:
            self._left[ticket.key] = now
        if held:
            self._grant_next(now)

    def notify(self) -> None:
        """Room may have appeared: a new sweep, the best waiter first. While a sweep runs (a turn is out), the
        wake-up is coalesced into one follow-up sweep that starts when this one ends."""
        if self._holder is not None:
            self._dirty = True
            return
        self._new_sweep()
        self._grant_next(self._clock())


__all__ = ["OpenQueue", "Ticket"]
