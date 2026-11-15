"""Light leases on a draw job's slots: who is decoding which sample, so two miners do not decode the same one.

A draw job's sample ``(prompt, slot)`` is fixed by its public draw (``corpus.eval_draw``): a lease guards no
integrity, only work. A miner asks the job for a lease and gets a free slot until ``expires_at``; meanwhile a
submission of that slot by another hotkey is refused (``draw_slot_leased``). A lease ends when the slot is taken,
when its holder releases it, or when it expires (no penalty: the quota is the RL controller's, later; a job priced
by auction, ``corpus.token_auction``, charges it instead, lifts the per-hotkey cap and sets each lease's time). When no free
slot is left, a slot leased for more than half its time may be leased to a second miner, the first valid submission
taking it, so a job's last slots never wait for a slow or vanished miner. A miner that asks for no lease (an older
one) still submits any slot no one else holds.

The leases live in the process that serves the job's routes. With a ``state_path`` they are written there on every
change (a lease lost would refuse its miner's submission, one settled but restored would be charged unused) and read
back at start, so a control restart keeps every lease still running: a job priced by
auction refuses a submission without its lease (2026-11-14: three restarts in a day dropped every miner's leases in
flight). Without one, a restart drops them.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import random
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REASON_DRAW_SLOT_LEASED = "draw_slot_leased"
# Live leases one hotkey may hold on one job: a miner's batch of concurrent samples, never more than a share of the
# job's slots, and halved for each of its leases that expired unused within the last lease time (a hotkey that holds
# slots and never submits them ends with one).
MAX_LEASES_PER_HOTKEY = 64
MAX_JOB_SHARE_PER_HOTKEY = 0.25
# A second lessee on a slot whose lease has run this share of its time (the job's tail).
SECOND_LESSEE_AFTER = 0.5
# Lease time: a floor, plus the job's longest sample at a slow decode rate.
LEASE_FLOOR_SECONDS = 300.0
SLOW_TOKENS_PER_SECOND = 10.0
MAX_LEASE_SECONDS = 4 * 3600.0
# Over a signed episode's deadline: the lease is granted before the miner opens its session.
EPISODE_LEASE_MARGIN_SECONDS = 300.0


def episode_lease_floor(job: Any) -> float:
    """The shortest lease on a signed-episode job: the miner's own deadline for one episode
    (``signed_episode_deadline``) plus a margin, so a lease never ends while its episode may still finish.
    0 for any other job."""
    episode = getattr(job, "episode", None)
    sandbox = getattr(episode, "sandbox", None) if episode is not None else None
    if sandbox is None:
        return 0.0
    from reliquary.protocol.sandbox_session import signed_episode_deadline

    return signed_episode_deadline(sandbox.budgets.wall_s) + EPISODE_LEASE_MARGIN_SECONDS


def lease_seconds(job: Any) -> float:
    """How long a lease on this job lasts: long enough for a slow honest miner to decode its longest sample, and
    for a signed episode to reach its deadline."""
    episode = getattr(job, "episode", None)
    tokens = (getattr(episode, "max_total_tokens", None) if episode is not None else None) or \
        job.sampling.max_new_tokens * max(1, int(getattr(job.sampling, "n", 1)))
    decode = min(MAX_LEASE_SECONDS, LEASE_FLOOR_SECONDS + float(tokens) / SLOW_TOKENS_PER_SECOND)
    return max(decode, episode_lease_floor(job))


@dataclass(frozen=True, slots=True)
class Lease:
    lease_id: str
    hotkey: str
    prompt_index: int
    slot: int
    granted_at: float
    expires_at: float
    # A job priced by auction: what each token of this lease pays (pool share) and the round it was granted in.
    price: float | None = None
    round: int | None = None

    def to_wire(self) -> dict[str, Any]:
        return {"lease_id": self.lease_id, "prompt_index": self.prompt_index, "slot": self.slot,
                "expires_at": self.expires_at}


class DrawLeases:
    """The live leases of one draw job (``grant``, ``release``, ``refusal`` at submission, ``settle`` on accept)."""

    def __init__(self, job: Any, *, clock: Callable[[], float] = time.time, rng: random.Random | None = None,
                 max_per_hotkey: int = MAX_LEASES_PER_HOTKEY, seconds: float | None = None,
                 limit_per_hotkey: bool = True,
                 on_unused: Callable[[Lease, float], None] | None = None,
                 validator_fault: Callable[[Lease], tuple[str, str] | None] | None = None,
                 state_path: str | os.PathLike | None = None) -> None:
        """``limit_per_hotkey`` False (a job priced by auction): no per-hotkey cap, the price decides who works.
        ``on_unused(lease, held_fraction)`` hears of a lease that ended without its slot taken: expired (1) or
        released (the share of its time it was held). ``validator_fault(lease)`` names ``(cause, detail)`` when
        our side failed the lease's episode (``sandbox.corpus_lease_faults``, from the session book and the
        issuer's own refusals, never the miner's word): that lease ends uncharged, neither ``on_unused`` (no debt,
        no halved capacity) nor counted against the hotkey's unused leases."""
        self._job = job
        self._limit = bool(limit_per_hotkey)
        self._on_unused = on_unused
        self._validator_fault = validator_fault
        self._clock = clock
        self._rng = rng or random.Random(secrets.randbits(64))
        self._max = int(max_per_hotkey)
        # No lease, however set (``seconds``, an auction's per-hotkey time), ends before a signed episode's deadline.
        self.floor = episode_lease_floor(job)
        self.seconds = max(self.floor, float(seconds if seconds is not None else lease_seconds(job)))
        self._by_id: dict[str, Lease] = {}
        self._by_slot: dict[tuple[int, int], list[str]] = {}
        # Each hotkey's leases that expired unused, by when they expired.
        self._unused: dict[str, list[float]] = {}
        # The route grants off the event loop and refuses and settles on it.
        self._lock = threading.RLock()
        self._state_path = Path(state_path) if state_path is not None else None
        self._dirty = False
        self._load()

    # -- state ---------------------------------------------------------------

    def _load(self) -> None:
        """The leases still running in ``state_path``; one that expired meanwhile is dropped uncharged (it ran out
        while we were down)."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
            now = self._clock()
            for raw in state.get("leases", ()):
                lease = Lease(**raw)
                if lease.expires_at > now:
                    self._by_id[lease.lease_id] = lease
                    self._by_slot.setdefault((lease.prompt_index, lease.slot), []).append(lease.lease_id)
            self._unused = {str(h): [float(t) for t in ts] for h, ts in (state.get("unused") or {}).items()}
            logger.info("draw leases: %d lease(s) still running restored from %s", len(self._by_id),
                        self._state_path)
        except (OSError, ValueError, TypeError):
            logger.exception("draw leases: state %s unreadable; starting with no lease", self._state_path)

    def _changed(self) -> None:
        self._dirty = True
        self.save()

    def save(self) -> None:
        """Write the live leases to ``state_path`` if they changed."""
        if self._state_path is None:
            return
        with self._lock:
            if not self._dirty:
                return
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"leases": [asdict(lease) for lease in self._by_id.values()],
                                           "unused": self._unused}))
                os.replace(tmp, self._state_path)
                self._dirty = False
            except OSError:
                logger.exception("draw leases: state %s not written", self._state_path)

    # -- bookkeeping ---------------------------------------------------------

    def _log(self, event: str, lease: Lease, extra: str = "") -> None:
        logger.info("draw lease %s job=%s prompt=%d slot=%d hotkey=%s secs=%d%s", event,
                    getattr(self._job, "job_id", "?"), lease.prompt_index, lease.slot, lease.hotkey[:8],
                    round(lease.expires_at - lease.granted_at), extra)

    def _drop(self, lease_id: str) -> None:
        lease = self._by_id.pop(lease_id, None)
        if lease is None:
            return
        self._dirty = True
        key = (lease.prompt_index, lease.slot)
        held = [other for other in self._by_slot.get(key, ()) if other != lease_id]
        if held:
            self._by_slot[key] = held
        else:
            self._by_slot.pop(key, None)

    def expire(self) -> None:
        with self._lock:
            now = self._clock()
            for lease_id in [i for i, lease in self._by_id.items() if lease.expires_at <= now]:
                lease = self._by_id[lease_id]
                self._drop(lease_id)
                fault = self._fault(lease)
                if fault is not None:
                    self._log("expired", lease, f" ours={fault[0]} ({fault[1]}): not charged")
                    continue
                self._unused.setdefault(lease.hotkey, []).append(lease.expires_at)
                self._log("expired", lease)
                self._unused_lease(lease, 1.0)
            for hotkey in list(self._unused):
                recent = [t for t in self._unused[hotkey] if now - t < self.seconds]
                if recent:
                    self._unused[hotkey] = recent
                else:
                    del self._unused[hotkey]
            self.save()

    def _fault(self, lease: Lease) -> tuple[str, str] | None:
        if self._validator_fault is None:
            return None
        try:
            return self._validator_fault(lease)
        except Exception:                     # unjudged: charged as before, never a free pass we cannot justify
            logger.exception("draw leases: the validator-fault check of lease %s failed", lease.lease_id[:8])
            return None

    def _unused_lease(self, lease: Lease, held_fraction: float) -> None:
        if self._on_unused is not None:
            try:
                self._on_unused(lease, held_fraction)
            except Exception:                 # a charge never breaks the lease book
                import logging

                logging.getLogger(__name__).exception("draw leases: unused-lease hook failed")

    def holders(self, prompt_index: int, slot: int) -> list[Lease]:
        with self._lock:
            self.expire()
            return [self._by_id[i] for i in self._by_slot.get((int(prompt_index), int(slot)), ())]

    def held_by(self, hotkey: str) -> list[Lease]:
        with self._lock:
            self.expire()
            return [lease for lease in self._by_id.values() if lease.hotkey == hotkey]

    def cap(self, hotkey: str) -> int:
        """The live leases this hotkey may hold now (``MAX_LEASES_PER_HOTKEY``)."""
        with self._lock:
            self.expire()
            slots = int(self._job.prompt_count * int(self._job.slots_per_prompt))
            if not self._limit:
                return max(1, slots)
            share = int(MAX_JOB_SHARE_PER_HOTKEY * slots)
            cap = max(1, min(self._max, share))
            return max(1, cap >> len(self._unused.get(hotkey, ())))

    def __len__(self) -> int:
        with self._lock:
            self.expire()
            return len(self._by_id)

    def counts(self) -> dict[str, int]:
        """Live leases per hotkey."""
        with self._lock:
            self.expire()
            return dict(collections.Counter(lease.hotkey for lease in self._by_id.values()))

    # -- the miner's side ----------------------------------------------------

    def grant(self, hotkey: str, free_slots: Callable[[int], Iterable[tuple[int, int]]], *,
              price: float | None = None, round: int | None = None,
              seconds: float | None = None) -> Lease | str | None:
        """A lease for ``hotkey`` on a free slot: an unleased one, else (the job's tail) one leased to another hotkey
        for more than half its time. ``free_slots(start)`` yields the job's open ``(prompt, slot)``, prompts from
        ``start`` round. ``seconds``: this lease's time, at most the job's. None: no slot to lease; a string: why
        this hotkey gets none."""
        with self._lock:
            return self._grant(hotkey, free_slots, price, round, seconds)

    def _grant(self, hotkey, free_slots, price=None, round=None, seconds=None):
        self.expire()
        if len(self.held_by(hotkey)) >= self.cap(hotkey):
            return "too_many_leases"
        now = self._clock()
        span = self._job.prompt_count
        start = self._job.prompt_start + self._rng.randrange(span) if span else self._job.prompt_start
        tail = None
        for prompt_index, slot in free_slots(start):
            held = self._by_slot.get((prompt_index, slot), ())
            if not held:
                return self._add(hotkey, prompt_index, slot, now, price, round, seconds, "granted")
            leases = [self._by_id[i] for i in held]
            if (tail is None and len(leases) == 1 and leases[0].hotkey != hotkey
                    and now - leases[0].granted_at >= SECOND_LESSEE_AFTER * (leases[0].expires_at - leases[0].granted_at)):
                tail = (prompt_index, slot)
        if tail is not None:
            return self._add(hotkey, *tail, now, price, round, seconds, "granted_second")
        return None

    def _add(self, hotkey: str, prompt_index: int, slot: int, now: float, price: float | None = None,
             round: int | None = None, seconds: float | None = None, event: str = "granted") -> Lease:
        span = self.seconds if seconds is None else min(self.seconds, max(1.0, float(seconds), self.floor))
        lease = Lease(secrets.token_hex(16), hotkey, int(prompt_index), int(slot), now, now + span,
                      price, round)
        self._by_id[lease.lease_id] = lease
        self._by_slot.setdefault((lease.prompt_index, lease.slot), []).append(lease.lease_id)
        self._log(event, lease, "" if price is None else f" price={price:.3g}")
        self._changed()
        return lease

    def release(self, hotkey: str, lease_id: str) -> bool:
        with self._lock:
            lease = self._by_id.get(lease_id)
            if lease is None or lease.hotkey != hotkey:
                return False
            self._drop(lease_id)
            span = lease.expires_at - lease.granted_at
            held = (self._clock() - lease.granted_at) / span if span > 0 else 1.0
            fault = self._fault(lease)
            if fault is not None:
                self._log("released", lease, f" held={held:.2f} ours={fault[0]} ({fault[1]}): not charged")
                return True
            self._log("released", lease, f" held={held:.2f}")
            self._unused_lease(lease, held)
            self.save()
            return True

    # -- the submission's side -----------------------------------------------

    def refusal(self, hotkey: str, prompt_index: int, slot: int) -> str | None:
        """Why this hotkey may not submit this slot now: another hotkey alone holds its live lease."""
        holders = self.holders(prompt_index, slot)
        if holders and all(lease.hotkey != hotkey for lease in holders):
            return REASON_DRAW_SLOT_LEASED
        return None

    def held(self, hotkey: str, prompt_index: int, slot: int) -> Lease | None:
        """``hotkey``'s own live lease on this slot, or None."""
        for lease in self.holders(prompt_index, slot):
            if lease.hotkey == hotkey:
                return lease
        return None

    def settle(self, prompt_index: int, slot: int) -> None:
        """The slot was taken: every lease on it ends."""
        with self._lock:
            now = self._clock()
            for lease_id in list(self._by_slot.get((int(prompt_index), int(slot)), ())):
                lease = self._by_id[lease_id]
                self._drop(lease_id)
                self._log("taken", lease, f" after={round(now - lease.granted_at)}")
            self.save()


def free_draw_slots(job: Any, slots_ledger: Any, taken: Callable[[int, int], bool]):
    """``free(start)``: the job's open ``(prompt, slot)`` in prompt order from ``start`` round the range, skipping
    prompts that are complete or exhausted and slots out of attempts or ``taken``."""
    from reliquary.corpus.eval_draw import slot_count

    slots = slot_count(job)

    def free(start: int):
        first, count = job.prompt_start, job.prompt_count
        for offset in range(count):
            prompt = first + (start - first + offset) % count
            if slots_ledger.prompt_state(prompt) != "open":
                continue
            for slot in range(slots_ledger.draw_cursors(prompt, slots)):
                if not slots_ledger.slot_exhausted(prompt, slot) and not taken(prompt, slot):
                    yield prompt, slot

    return free


__all__ = ["DrawLeases", "Lease", "MAX_LEASES_PER_HOTKEY", "REASON_DRAW_SLOT_LEASED", "episode_lease_floor",
           "free_draw_slots", "lease_seconds"]
