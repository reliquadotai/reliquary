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
missed in this sweep waits for the next one too.

Priority = ``score x (1 + key_waited / aging_s)``: ``score`` = a MAIN hotkey's weight (useful volume x yield
multiplier) from ``rank``; the key's waiting time starts at its first ticket and is kept across retries (each
HTTP retry is a new ticket), forgotten ``aging_forget_s`` after its last ticket left, reset when it places. One
grant in ``fair_every`` goes to a POOLED waiter (``rank``'s third element: a newcomer's, a low-yield or a small
hotkey's, score 0), the one whose key was served least recently; with no pooled waiter it goes by priority. A
MAIN key never gets a turn out of its weight: a per-key fair turn would pay for more hotkeys, while pooled tickets
are bounded by their shared pool.

Bounds: ``max_waiting`` in all; per key ``min(max_per_key, headroom(hotkey))`` (the quota's room for more live
sessions); one ticket per engagement. A waiter leaves at its deadline with a clear refusal. Everything runs on
the event loop (no lock needed)."""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Hashable

from reliquary.sandbox.rl_capacity import (
    QUEUE_AGING_FORGET_S,
    QUEUE_AGING_S,
    QUEUE_FAIR_EVERY,
    QUEUE_MAX_PER_KEY,
    QUEUE_MAX_WAITING,
    QUEUE_WAIT_S,
)
from reliquary.sandbox.sessions import Refusal


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
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._rank, self._headroom = rank, headroom
        self._wait_s, self._aging_s, self._forget_s = float(wait_s), float(aging_s), float(aging_forget_s)
        self._max_waiting, self._max_per_key, self._fair_every = int(max_waiting), int(max_per_key), int(fair_every)
        self._clock = clock
        self._waiting: list[Ticket] = []
        self._holder: Ticket | None = None
        self._sweep = 0
        self._grants = 0
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

    def _grant_next(self, now: float) -> None:
        candidates = [ticket for ticket in self._waiting if ticket.tried != self._sweep]
        if not candidates or self._holder is not None:
            return
        self._grants += 1
        pooled = [ticket for ticket in candidates if ticket.pooled]
        if self._fair_every and self._grants % self._fair_every == 0 and pooled:
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

    def missed(self, ticket: Ticket) -> None:
        """The turn found no room: the ticket waits again (tried in this sweep) and the next untried waiter of
        the sweep gets the turn at once."""
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
        if ticket in self._waiting:
            self._waiting.remove(ticket)
        held = self._holder is ticket
        if held:
            self._holder = None
        others = any(t.key == ticket.key for t in self._tickets())
        if placed:
            self._served[ticket.key] = now
            if others:
                self._since[ticket.key] = now
            else:
                self._since.pop(ticket.key, None)
        if not others:
            self._left[ticket.key] = now
        if held:
            self._grant_next(now)

    def notify(self) -> None:
        """Room may have appeared: a new sweep; the best waiter gets the turn if none is outstanding."""
        self._sweep += 1
        if self._holder is None:
            self._grant_next(self._clock())


__all__ = ["OpenQueue", "Ticket"]
