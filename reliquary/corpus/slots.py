"""How many completions a prompt still accepts.

Sparse on purpose: a generated source declares up to ``1 << 31`` prompts, so
only the prompts actually consumed may cost memory.
"""

from __future__ import annotations

from collections.abc import Mapping


class SlotExhausted(Exception):
    """This prompt has no slot left; it is finished for the life of the job."""


class SlotLedger:
    """V slots per prompt, consumed permanently. There is no timer.

    Keyed by SOURCE index, over [prompt_start, prompt_start + prompt_count), so
    a snapshot of a job that starts at 0 is the same whether or not the job
    could have started elsewhere.
    """

    __slots__ = (
        "_prompt_start", "_prompt_count", "_slots_per_prompt", "_consumed", "_filled",
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

    def remaining(self, index: int) -> int:
        position = self._check(index)
        return self._slots_per_prompt - self._consumed.get(position, 0)

    def is_full(self, index: int) -> bool:
        return self.remaining(index) == 0

    def consume(self, index: int) -> int:
        """Take one slot and report what is left, or refuse if there is none."""
        position = self._check(index)
        taken = self._consumed.get(position, 0)
        if taken >= self._slots_per_prompt:
            raise SlotExhausted(f"prompt {position} has no slot left")
        self._consumed[position] = taken + 1
        self._filled += 1
        return self._slots_per_prompt - (taken + 1)

    @property
    def filled(self) -> int:
        return self._filled

    @property
    def total(self) -> int:
        return self._prompt_count * self._slots_per_prompt

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
    ) -> "SlotLedger":
        ledger = cls(prompt_count, slots_per_prompt, prompt_start=prompt_start)
        filled = 0
        for index, taken in snapshot.items():
            position = int(index)
            if not ledger._owns(position):
                raise ValueError(
                    f"snapshot names prompt {position}, outside {ledger._range()}"
                )
            count = int(taken)
            if count <= 0 or count > slots_per_prompt:
                raise ValueError(
                    f"snapshot gives prompt {position} {count} consumed slots, "
                    f"outside 1..{slots_per_prompt}"
                )
            ledger._consumed[position] = count
            filled += count
        ledger._filled = filled
        return ledger
