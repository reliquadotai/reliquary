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

Pure and deterministic: two validators replaying the same archive must reach
the same numbers bit for bit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AcceptedGroup:
    hotkey: str
    operator_id: str
    eos_tokens: int


def split_environment_pool(
    groups: Sequence[AcceptedGroup],
    *,
    pool: float,
    max_operator_share: float,
) -> dict[str, float]:
    """Return ``{hotkey: share}`` summing to ``pool`` over paying groups."""
    paying = [group for group in groups if group.eos_tokens > 0]
    if not paying:
        return {}

    by_operator: dict[str, int] = defaultdict(int)
    for group in paying:
        by_operator[group.operator_id] += group.eos_tokens
    total = sum(by_operator.values())

    # Clip operators over the cap, then reflow what they gave up across the
    # operators still under it, repeating until the split is stable. One pass
    # would leave the reflow itself pushing a second operator over the cap.
    ceiling = pool * max_operator_share
    weights = {
        operator: pool * tokens / total for operator, tokens in by_operator.items()
    }
    for _ in range(len(weights)):
        over = {op: w for op, w in weights.items() if w > ceiling + 1e-12}
        if not over:
            break
        spare = sum(w - ceiling for w in over.values())
        under = {op: w for op, w in weights.items() if op not in over}
        under_total = sum(under.values())
        for operator in over:
            weights[operator] = ceiling
        if under_total <= 0:
            break
        for operator, weight in under.items():
            weights[operator] = weight + spare * weight / under_total

    rewards: dict[str, float] = {}
    for group in paying:
        operator_tokens = by_operator[group.operator_id]
        rewards[group.hotkey] = rewards.get(group.hotkey, 0.0) + (
            weights[group.operator_id] * group.eos_tokens / operator_tokens
        )
    return rewards
