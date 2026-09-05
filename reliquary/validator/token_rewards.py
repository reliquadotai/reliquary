"""Deterministic token-weighted accounting for the disabled fill experiment.

The caller supplies already eligible groups and their counted completion
tokens. This isolated function exists for replay and qualification; the
Reliquary 1 target retains selected-slot rewards and does not activate this
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AcceptedGroup:
    hotkey: str
    # Retained in the archived accounting record.
    operator_id: str
    eos_tokens: int


def split_environment_pool(
    groups: Sequence[AcceptedGroup],
    *,
    pool: float,
) -> dict[str, float]:
    """Return ``{hotkey: share}`` summing to ``pool`` over paying groups."""
    paying = [group for group in groups if group.eos_tokens > 0]
    if not paying:
        return {}

    total = sum(group.eos_tokens for group in paying)

    rewards: dict[str, float] = {}
    for group in paying:
        rewards[group.hotkey] = rewards.get(group.hotkey, 0.0) + (
            pool * group.eos_tokens / total
        )
    return rewards
