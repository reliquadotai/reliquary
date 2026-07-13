"""Canonical rollout-group Merkle binding shared by miners and validators."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _field(rollout: Any, name: str) -> Any:
    if isinstance(rollout, dict):
        return rollout[name]
    return getattr(rollout, name)


def compute_rollouts_merkle_root(rollouts: Iterable[Any]) -> str:
    """Return the canonical SHA-256 Merkle root for a rollout group.

    Leaves bind the rollout index, tokens, claimed reward, environment, and
    complete GRAIL commitment. Canonical JSON keeps the result stable across
    dict insertion order. An odd leaf is duplicated at each tree level.
    """
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
        env_name = (
            rollout.get("env_name", "")
            if isinstance(rollout, dict)
            else getattr(rollout, "env_name", "")
        )
        leaf.update(json.dumps(str(env_name)).encode())
        leaf.update(
            json.dumps(
                _field(rollout, "commit"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        leaves.append(leaf.digest())

    if not leaves:
        raise ValueError("cannot compute a Merkle root for an empty rollout group")

    while len(leaves) > 1:
        parents: list[bytes] = []
        for index in range(0, len(leaves), 2):
            left = leaves[index]
            right = leaves[index + 1] if index + 1 < len(leaves) else left
            parents.append(hashlib.sha256(left + right).digest())
        leaves = parents
    return leaves[0].hex()


def submission_merkle_matches(request: Any) -> bool:
    """Whether a request's claimed root matches its canonical rollout root."""
    try:
        expected = compute_rollouts_merkle_root(request.rollouts)
        claimed = str(request.merkle_root).lower()
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return claimed == expected
