"""Light leases on a draw job's slots: who is decoding which sample, so two miners do not decode the same one.

A draw job's sample ``(prompt, slot)`` is fixed by its public draw (``corpus.eval_draw``): a lease guards no
integrity, only work. A miner asks the job for a lease and gets a free slot until ``expires_at``; meanwhile a
submission of that slot by another hotkey is refused (``draw_slot_leased``). A lease ends when the slot is taken,
when its holder releases it, or when it expires (no penalty: the quota is the RL controller's, later). When no free
slot is left, a slot leased for more than half its time may be leased to a second miner, the first valid submission
taking it, so a job's last slots never wait for a slow or vanished miner. A miner that asks for no lease (an older
one) still submits any slot no one else holds.

The leases live in the process that serves the job's routes; a restart drops them, which only lets two miners decode
the same slot until the next leases are granted.
"""
from __future__ import annotations

import random
import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

REASON_DRAW_SLOT_LEASED = "draw_slot_leased"
# Live leases one hotkey may hold on one job: a miner's batch of concurrent samples, not the whole job.
MAX_LEASES_PER_HOTKEY = 64
# A second lessee on a slot whose lease has run this share of its time (the job's tail).
SECOND_LESSEE_AFTER = 0.5
# Lease time: a floor, plus the job's longest sample at a slow decode rate.
LEASE_FLOOR_SECONDS = 300.0
SLOW_TOKENS_PER_SECOND = 10.0
MAX_LEASE_SECONDS = 4 * 3600.0


def lease_seconds(job: Any) -> float:
    """How long a lease on this job lasts: long enough for a slow honest miner to decode its longest sample."""
    episode = getattr(job, "episode", None)
    tokens = (getattr(episode, "max_total_tokens", None) if episode is not None else None) or \
        job.sampling.max_new_tokens * max(1, int(getattr(job.sampling, "n", 1)))
    return min(MAX_LEASE_SECONDS, LEASE_FLOOR_SECONDS + float(tokens) / SLOW_TOKENS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class Lease:
    lease_id: str
    hotkey: str
    prompt_index: int
    slot: int
    granted_at: float
    expires_at: float

    def to_wire(self) -> dict[str, Any]:
        return {"lease_id": self.lease_id, "prompt_index": self.prompt_index, "slot": self.slot,
                "expires_at": self.expires_at}


class DrawLeases:
    """The live leases of one draw job (``grant``, ``release``, ``refusal`` at submission, ``settle`` on accept)."""

    def __init__(self, job: Any, *, clock: Callable[[], float] = time.time, rng: random.Random | None = None,
                 max_per_hotkey: int = MAX_LEASES_PER_HOTKEY, seconds: float | None = None) -> None:
        self._job = job
        self._clock = clock
        self._rng = rng or random.Random(secrets.randbits(64))
        self._max = int(max_per_hotkey)
        self.seconds = float(seconds if seconds is not None else lease_seconds(job))
        self._by_id: dict[str, Lease] = {}
        self._by_slot: dict[tuple[int, int], list[str]] = {}

    # -- bookkeeping ---------------------------------------------------------

    def _drop(self, lease_id: str) -> None:
        lease = self._by_id.pop(lease_id, None)
        if lease is None:
            return
        key = (lease.prompt_index, lease.slot)
        held = [other for other in self._by_slot.get(key, ()) if other != lease_id]
        if held:
            self._by_slot[key] = held
        else:
            self._by_slot.pop(key, None)

    def expire(self) -> None:
        now = self._clock()
        for lease_id in [i for i, lease in self._by_id.items() if lease.expires_at <= now]:
            self._drop(lease_id)

    def holders(self, prompt_index: int, slot: int) -> list[Lease]:
        self.expire()
        return [self._by_id[i] for i in self._by_slot.get((int(prompt_index), int(slot)), ())]

    def held_by(self, hotkey: str) -> list[Lease]:
        self.expire()
        return [lease for lease in self._by_id.values() if lease.hotkey == hotkey]

    def __len__(self) -> int:
        self.expire()
        return len(self._by_id)

    # -- the miner's side ----------------------------------------------------

    def grant(self, hotkey: str, free_slots: Callable[[int], Iterable[tuple[int, int]]]) -> Lease | str | None:
        """A lease for ``hotkey`` on a free slot: an unleased one, else (the job's tail) one leased to another hotkey
        for more than half its time. ``free_slots(start)`` yields the job's open ``(prompt, slot)``, prompts from
        ``start`` round. None: no slot to lease; a string: why this hotkey gets none."""
        self.expire()
        if len(self.held_by(hotkey)) >= self._max:
            return "too_many_leases"
        now = self._clock()
        span = self._job.prompt_count
        start = self._job.prompt_start + self._rng.randrange(span) if span else self._job.prompt_start
        tail = None
        for prompt_index, slot in free_slots(start):
            held = self._by_slot.get((prompt_index, slot), ())
            if not held:
                return self._add(hotkey, prompt_index, slot, now)
            leases = [self._by_id[i] for i in held]
            if (tail is None and len(leases) == 1 and leases[0].hotkey != hotkey
                    and now - leases[0].granted_at >= SECOND_LESSEE_AFTER * self.seconds):
                tail = (prompt_index, slot)
        if tail is not None:
            return self._add(hotkey, *tail, now)
        return None

    def _add(self, hotkey: str, prompt_index: int, slot: int, now: float) -> Lease:
        lease = Lease(secrets.token_hex(16), hotkey, int(prompt_index), int(slot), now, now + self.seconds)
        self._by_id[lease.lease_id] = lease
        self._by_slot.setdefault((lease.prompt_index, lease.slot), []).append(lease.lease_id)
        return lease

    def release(self, hotkey: str, lease_id: str) -> bool:
        lease = self._by_id.get(lease_id)
        if lease is None or lease.hotkey != hotkey:
            return False
        self._drop(lease_id)
        return True

    # -- the submission's side -----------------------------------------------

    def refusal(self, hotkey: str, prompt_index: int, slot: int) -> str | None:
        """Why this hotkey may not submit this slot now: another hotkey alone holds its live lease."""
        holders = self.holders(prompt_index, slot)
        if holders and all(lease.hotkey != hotkey for lease in holders):
            return REASON_DRAW_SLOT_LEASED
        return None

    def settle(self, prompt_index: int, slot: int) -> None:
        """The slot was taken: every lease on it ends."""
        for lease_id in list(self._by_slot.get((int(prompt_index), int(slot)), ())):
            self._drop(lease_id)


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


__all__ = ["DrawLeases", "Lease", "MAX_LEASES_PER_HOTKEY", "REASON_DRAW_SLOT_LEASED", "free_draw_slots",
           "lease_seconds"]
