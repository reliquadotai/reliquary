"""The order in which one miner consumes a job's prompt source.

Each miner walks its own hash order, which disperses collisions that a shared
heuristic would concentrate, and takes away any choice of WHICH prompt comes
next. It does not equalise the cost of skipping one against answering it: a
miner can always answer at the floor and step on. What closes that residue is
the audit tier, where junk completions fail token authenticity.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib


def walk_index(job_id: str, hotkey: str, cursor: int, prompt_count: int) -> int:
    """The prompt this miner's walk visits at ``cursor``."""
    if prompt_count <= 0:
        raise ValueError(f"prompt_count must be positive, got {prompt_count}")
    if cursor < 0:
        raise ValueError(f"cursor must not be negative, got {cursor}")
    digest = hashlib.sha256()
    digest.update(job_id.encode())
    digest.update(b"\x00")
    digest.update(hotkey.encode())
    digest.update(b"\x00")
    digest.update(int(cursor).to_bytes(8, "big", signed=False))
    # Modulo bias is negligible: no prompt source approaches 2**256.
    return int.from_bytes(digest.digest(), "big") % int(prompt_count)


class CursorLedger:
    """Where each miner is in its walk.

    Strictly sequential on purpose: a miner free to choose its cursor would
    grind cursors until one landed on a prompt it liked.
    """

    __slots__ = ("_cursors",)

    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}

    def expected(self, hotkey: str) -> int:
        return self._cursors.get(hotkey, 0)

    def advance(self, hotkey: str) -> int:
        """Move this miner one step on and report where it now stands."""
        moved = self._cursors.get(hotkey, 0) + 1
        self._cursors[hotkey] = moved
        return moved

    def snapshot(self) -> dict[str, int]:
        return dict(self._cursors)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, int]) -> "CursorLedger":
        ledger = cls()
        for hotkey, cursor in snapshot.items():
            if not isinstance(hotkey, str) or not hotkey:
                raise ValueError(f"unusable hotkey {hotkey!r} in cursor snapshot")
            # Not `int(cursor)`: that turns 1.5 into 1 and hands a miner a step
            # it never took. A cursor this binary cannot read exactly is named.
            if isinstance(cursor, bool) or not isinstance(cursor, int):
                raise ValueError(
                    f"cursor for {hotkey} is not a whole number: {cursor!r}"
                )
            position = cursor
            if position < 0:
                raise ValueError(f"cursor for {hotkey} is negative: {position}")
            ledger._cursors[hotkey] = position
        return ledger
