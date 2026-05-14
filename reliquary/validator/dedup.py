"""RolloutHashSet — per-rollout content dedup across a cooldown horizon.

A miner that re-submits a rollout whose token content matches one already
entered in a sealed batch within the retention window is rejected with
``RejectReason.HASH_DUPLICATE``. Mirrors the lifecycle of
``reliquary.validator.cooldown.CooldownMap``: in-memory set, rebuilt at
validator startup from the recent R2 archive payloads.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def compute_rollout_hash(tokens: Iterable[int]) -> bytes:
    """Return SHA256 digest of *tokens* packed as big-endian uint32.

    Deterministic over Python implementations: each int is serialised as a
    fixed 4-byte big-endian unsigned integer and concatenated before
    hashing. Rejects negative values (vocab token ids are always
    non-negative; a negative slipping in here means upstream corruption).
    """
    h = hashlib.sha256()
    for t in tokens:
        if t < 0:
            raise ValueError(f"compute_rollout_hash: negative token id {t}")
        h.update(int(t).to_bytes(4, "big", signed=False))
    return h.digest()


class RolloutHashSet:
    """Per-rollout content set with a sliding retention horizon.

    Membership tested via ``__contains__``. Entries older than
    ``retention_windows`` are dropped via ``prune``.
    """

    def __init__(self, retention_windows: int) -> None:
        if retention_windows < 0:
            raise ValueError("retention_windows must be non-negative")
        self._retention_windows = retention_windows
        self._entries: dict[bytes, int] = {}

    def add(self, h: bytes, window: int) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        # Keep the most recent window for any given hash.
        prev = self._entries.get(h, -1)
        if window > prev:
            self._entries[h] = window

    def __contains__(self, h: bytes) -> bool:
        return h in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def prune(self, current_window: int) -> None:
        """Drop entries whose window is older than the retention horizon.

        An entry at ``window=W`` survives while
        ``current_window - W < retention_windows``. At equality the entry
        is dropped — same half-open interval semantics as ``CooldownMap``.
        """
        if self._retention_windows == 0:
            self._entries.clear()
            return
        horizon = current_window - self._retention_windows
        self._entries = {
            h: w for h, w in self._entries.items() if w > horizon
        }
