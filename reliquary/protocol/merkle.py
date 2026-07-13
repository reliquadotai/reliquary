"""Canonical rollout-group Merkle binding shared by miners and validators."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


_LEAF_DOMAIN = b"reliquary-rollout-leaf-v2\x00"
_NODE_DOMAIN = b"reliquary-rollout-node-v2\x00"
_SELECTION_DOMAIN = b"reliquary-selection-stream-v1\x00"


def _field(rollout: Any, name: str) -> Any:
    if isinstance(rollout, dict):
        return rollout[name]
    return getattr(rollout, name)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def compute_rollouts_merkle_root(rollouts: Iterable[Any]) -> str:
    """Return the canonical SHA-256 Merkle root for a rollout group.

    Leaves bind the rollout index, tokens, claimed reward, environment, and
    complete GRAIL commitment. Canonical JSON keeps the result stable across
    dict insertion order. An odd leaf is duplicated at each tree level.
    """
    leaves: list[bytes] = []
    for index, rollout in enumerate(rollouts):
        env_name = (
            rollout.get("env_name", "")
            if isinstance(rollout, dict)
            else getattr(rollout, "env_name", "")
        )
        payload = _canonical_json(
            {
                "commit": _field(rollout, "commit"),
                "env_name": str(env_name),
                "index": index,
                "reward": _field(rollout, "reward"),
                "tokens": _field(rollout, "tokens"),
            }
        )
        leaves.append(hashlib.sha256(_LEAF_DOMAIN + payload).digest())

    if not leaves:
        raise ValueError("cannot compute a Merkle root for an empty rollout group")

    while len(leaves) > 1:
        parents: list[bytes] = []
        for index in range(0, len(leaves), 2):
            left = leaves[index]
            right = leaves[index + 1] if index + 1 < len(leaves) else left
            parents.append(
                hashlib.sha256(_NODE_DOMAIN + left + right).digest()
            )
        leaves = parents
    return leaves[0].hex()


def compute_rollouts_selection_digest(rollouts: Iterable[Any]) -> bytes:
    """Hash only generation-defining fields for canonical representative choice.

    The full Merkle root authenticates every submitted field, but several of
    those fields are intentionally tolerant or validator-overwritten. They are
    therefore unsuitable as a deterministic lottery input: a miner could vary
    harmless metadata and grind roots without generating a new rollout. The
    ordered token streams and environment are the immutable generation output.
    """
    streams = []
    for index, rollout in enumerate(rollouts):
        env_name = (
            rollout.get("env_name", "")
            if isinstance(rollout, dict)
            else getattr(rollout, "env_name", "")
        )
        streams.append(
            {
                "env_name": str(env_name),
                "index": index,
                "tokens": _field(rollout, "tokens"),
            }
        )
    if not streams:
        raise ValueError("cannot compute a selection digest for an empty group")
    return hashlib.sha256(_SELECTION_DOMAIN + _canonical_json(streams)).digest()


def submission_merkle_matches(request: Any) -> bool:
    """Whether a request's claimed root matches its canonical rollout root."""
    try:
        expected = compute_rollouts_merkle_root(request.rollouts)
        claimed = str(request.merkle_root).lower()
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return claimed == expected
