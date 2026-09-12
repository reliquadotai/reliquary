"""Deterministic accounting for fill-closed selected groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


EOS_TOKEN_PAYMENT_POLICY = "eos-tokens-per-batch/v1"
FIXED_GROUP_PAYMENT_POLICY = "fixed-selected-group/v1"


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


def split_fixed_environment_pool(
    groups: Sequence[AcceptedGroup],
    *,
    pool: float,
    slots: int,
) -> dict[str, float]:
    """Pay one fixed share per selected group; unfilled slot shares burn."""
    if type(slots) is not int or slots <= 0:
        raise ValueError("slots must be a positive integer")
    if len(groups) > slots:
        raise ValueError("groups exceed fixed payment slots")

    share = pool / slots
    rewards: dict[str, float] = {}
    for group in groups:
        rewards[group.hotkey] = rewards.get(group.hotkey, 0.0) + share
    return rewards
