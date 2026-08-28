"""Divide an environment's pool by EOS-terminated completion tokens.

Replaces ``slot_share = pool / B_BATCH``. Under the flat share a group costs
``16L/r`` rounds at rate ``r`` and length ``L`` and pays the same whatever
``L`` is, so revenue per GPU-second is proportional to ``1/L`` and halving
response length doubles income. Dividing by tokens removes that.

Only EOS-terminated rollouts contribute. The caller does that filtering; this
module simply never invents value for a token it was not given. That
restriction is load-bearing rather than cosmetic: the flat share is currently
one of four barriers against EOS suppression, and per-token payment removes
it. Paying only terminated tokens restores a strictly negative margin on
padding.

There is deliberately no per-operator cap. Every identity a cap could key on
-- operator, coldkey, hotkey -- costs one registration to multiply, so a cap
is bought around rather than respected. Per-token payment is itself the
concentration control precisely because it keys on nothing but tokens
produced: a registration count does not exist in this function's inputs, so
it cannot be split across coldkeys the way a slot count could.

Pure and deterministic: two validators replaying the same archive must reach
the same numbers bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AcceptedGroup:
    hotkey: str
    # Archived for audit; no longer drives payment (see module docstring).
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
