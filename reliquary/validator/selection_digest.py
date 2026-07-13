"""Validator-derived digest for deterministic representative selection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


_SELECTION_DOMAIN = b"reliquary-selection-stream-v1\x00"


def _field(rollout: Any, name: str) -> Any:
    if isinstance(rollout, dict):
        return rollout[name]
    return getattr(rollout, name)


def compute_rollouts_selection_digest(rollouts: Iterable[Any]) -> bytes:
    """Hash only immutable generation outputs used by the slot lottery."""
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
    payload = json.dumps(
        streams,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(_SELECTION_DOMAIN + payload).digest()
