"""Frozen wire-v1 rollout root used for validator-side compatibility checks.

This module intentionally reproduces ``miner.engine._compute_merkle_root``
byte-for-byte. Do not improve its serialization or add fields: wire-v1 miners
already sign this exact root. A future canonical format belongs to a new wire
version.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Iterable


def _field(rollout: Any, name: str) -> Any:
    if isinstance(rollout, dict):
        return rollout[name]
    return getattr(rollout, name)


def compute_legacy_rollouts_merkle_root(rollouts: Iterable[Any]) -> str:
    """Return the exact Merkle root emitted by current wire-v1 miners."""
    leaves: list[bytes] = []
    for index, rollout in enumerate(rollouts):
        leaf = hashlib.sha256()
        leaf.update(index.to_bytes(8, "big"))
        leaf.update(
            json.dumps(
                _field(rollout, "tokens"), separators=(",", ":")
            ).encode()
        )
        leaf.update(json.dumps(_field(rollout, "reward")).encode())
        leaf.update(
            json.dumps(
                _field(rollout, "commit"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        leaves.append(leaf.digest())

    if not leaves:
        raise ValueError("cannot compute a legacy root for an empty group")

    while len(leaves) > 1:
        parents: list[bytes] = []
        for index in range(0, len(leaves), 2):
            left = leaves[index]
            right = leaves[index + 1] if index + 1 < len(leaves) else left
            parents.append(hashlib.sha256(left + right).digest())
        leaves = parents
    return leaves[0].hex()


def legacy_submission_merkle_matches(request: Any) -> tuple[bool, str]:
    """Return ``(matches, computed_root)`` for a parsed v1 request."""
    computed = compute_legacy_rollouts_merkle_root(request.rollouts)
    claimed = str(request.merkle_root).lower()
    return hmac.compare_digest(claimed, computed), computed


__all__ = [
    "compute_legacy_rollouts_merkle_root",
    "legacy_submission_merkle_matches",
]
