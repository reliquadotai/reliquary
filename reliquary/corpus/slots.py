"""How many completions a prompt still accepts.

Sparse on purpose: a generated source declares up to ``1 << 31`` prompts, so
only the prompts actually consumed may cost memory.
"""

from __future__ import annotations

from collections.abc import Mapping


# An eval job reopens a prompt's slot when a submission fails its audit, up to
# this many attempts per prompt in all (as a multiple of its slots).
ATTEMPTS_PER_SLOT = 3


class SlotExhausted(Exception):
    """This prompt has no slot left; it is finished for the life of the job."""


class SlotLedger:
    """V slots per prompt, consumed permanently. There is no timer.

    Keyed by SOURCE index, over [prompt_start, prompt_start + prompt_count), so
    a snapshot of a job that starts at 0 is the same whether or not the job
    could have started elsewhere.

    ``failed`` (eval jobs only) names, per prompt, the submissions that failed
    their audit (the first 12 hex of each id, so recording one is idempotent);
    each reopens a slot, up to ``ATTEMPTS_PER_SLOT × V`` slots for the prompt
    in all. A job that never records a failure is exactly V slots.
    """

    __slots__ = (
        "_prompt_start", "_prompt_count", "_slots_per_prompt", "_consumed", "_filled",
        "_failed",
    )

    def __init__(
        self, prompt_count: int, slots_per_prompt: int, *, prompt_start: int = 0
    ) -> None:
        if prompt_count <= 0:
            raise ValueError(f"prompt_count must be positive, got {prompt_count}")
        if slots_per_prompt <= 0:
            raise ValueError(f"slots_per_prompt must be positive, got {slots_per_prompt}")
        if prompt_start < 0:
            raise ValueError(f"prompt_start must not be negative, got {prompt_start}")
        self._prompt_start = int(prompt_start)
        self._prompt_count = int(prompt_count)
        self._slots_per_prompt = int(slots_per_prompt)
        self._consumed: dict[int, int] = {}
        self._filled = 0
        self._failed: dict[int, list[str]] = {}

    def _check(self, index: int) -> int:
        position = int(index)
        if not self._owns(position):
            raise IndexError(
                f"prompt index {position} is outside {self._range()}"
            )
        return position

    def _owns(self, position: int) -> bool:
        return self._prompt_start <= position < self._prompt_start + self._prompt_count

    def _range(self) -> str:
        if self._prompt_start == 0:
            return f"a source of {self._prompt_count}"
        end = self._prompt_start + self._prompt_count
        return f"the job's rows [{self._prompt_start}, {end})"

    @property
    def attempt_limit(self) -> int:
        return ATTEMPTS_PER_SLOT * self._slots_per_prompt

    def capacity(self, index: int) -> int:
        """The slots this prompt has: V, plus one per failed submission, bounded."""
        position = self._check(index)
        return min(self._slots_per_prompt + len(self._failed.get(position, ())),
                   self.attempt_limit)

    def remaining(self, index: int) -> int:
        position = self._check(index)
        return self.capacity(position) - self._consumed.get(position, 0)

    def record_failure(self, index: int, submission_id: str) -> bool | None:
        """A submission for this prompt failed: True when that reopened a slot,
        False when the prompt had used every attempt (exhausted), None when it
        was already recorded."""
        position = self._check(index)
        key = str(submission_id)[:12]
        failed = self._failed.setdefault(position, [])
        if key in failed:
            return None
        before = self.capacity(position)
        failed.append(key)
        return self.capacity(position) > before

    def passing(self, index: int) -> int:
        """Slots holding work not known to have failed (passed or pending)."""
        position = self._check(index)
        return self._consumed.get(position, 0) - len(self._failed.get(position, ()))

    def prompt_state(self, index: int) -> str:
        """``complete`` (V slots of work not failed), ``exhausted`` (every
        attempt used, fewer than V not failed) or ``open``."""
        position = self._check(index)
        if self.passing(position) >= self._slots_per_prompt:
            return "complete"
        if self._consumed.get(position, 0) >= self.attempt_limit:
            return "exhausted"
        return "open"

    def prompt_counts(self) -> dict[str, int]:
        """How many of the job's prompts are complete, exhausted, open."""
        counts = {"complete": 0, "exhausted": 0, "open": 0}
        for offset in range(self._prompt_count):
            counts[self.prompt_state(self._prompt_start + offset)] += 1
        return counts

    def failed_snapshot(self) -> dict[int, list[str]]:
        return {index: sorted(ids) for index, ids in self._failed.items() if ids}

    def is_full(self, index: int) -> bool:
        return self.remaining(index) == 0

    def consume(self, index: int) -> int:
        """Take one slot and report what is left, or refuse if there is none."""
        position = self._check(index)
        taken = self._consumed.get(position, 0)
        capacity = self.capacity(position)
        if taken >= capacity:
            raise SlotExhausted(f"prompt {position} has no slot left")
        self._consumed[position] = taken + 1
        self._filled += 1
        return capacity - (taken + 1)

    @property
    def filled(self) -> int:
        return self._filled

    @property
    def total(self) -> int:
        extra = sum(min(len(failed), self.attempt_limit - self._slots_per_prompt)
                    for failed in self._failed.values())
        return self._prompt_count * self._slots_per_prompt + extra

    @property
    def is_complete(self) -> bool:
        return self._filled >= self.total

    def snapshot(self) -> dict[int, int]:
        """Only the prompts that were touched, so an archive stays small."""
        return dict(self._consumed)

    @classmethod
    def from_snapshot(
        cls,
        prompt_count: int,
        slots_per_prompt: int,
        snapshot: Mapping[int, int],
        *,
        prompt_start: int = 0,
        failed: Mapping[int, list[str]] | None = None,
    ) -> "SlotLedger":
        ledger = cls(prompt_count, slots_per_prompt, prompt_start=prompt_start)
        for index, ids in (failed or {}).items():
            position = int(index)
            if (not ledger._owns(position) or not isinstance(ids, list) or not ids
                    or len(set(ids)) != len(ids)
                    or not all(isinstance(i, str) and i for i in ids)):
                raise ValueError(f"failed names prompt {position} with {ids!r}")
            ledger._failed[position] = list(ids)
        filled = 0
        for index, taken in snapshot.items():
            position = int(index)
            if not ledger._owns(position):
                raise ValueError(
                    f"snapshot names prompt {position}, outside {ledger._range()}"
                )
            count = int(taken)
            if count <= 0 or count > ledger.capacity(position):
                raise ValueError(
                    f"snapshot gives prompt {position} {count} consumed slots, "
                    f"outside 1..{ledger.capacity(position)}"
                )
            ledger._consumed[position] = count
            filled += count
        ledger._filled = filled
        return ledger
