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
