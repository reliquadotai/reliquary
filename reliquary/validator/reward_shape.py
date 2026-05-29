"""Group-level reward-shape checks for manufactured GRPO submissions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Sequence

from reliquary.constants import (
    REWARD_SHAPE_MIN_EXACT_ZERO_ROLLOUTS,
    REWARD_SHAPE_MIN_REPEATED_ZERO_ROLLOUTS,
    REWARD_SHAPE_ZERO_MODE_MIN_LENGTH,
    REWARD_SHAPE_ZERO_MODE_MIN_SHARE,
)


@dataclass(frozen=True)
class RewardShapeMetrics:
    suspicious: bool
    reward_vector: str
    ordered_prefix: bool
    positive_count: int
    zero_count: int
    zero_length_mode: int
    zero_length_mode_count: int
    zero_length_mode_share: float
    zero_mode_truncated_count: int
    repeated_truncated_losers: bool
    repeated_exact_losers: bool

    def to_log_dict(self) -> dict[str, bool | int | float | str]:
        return {
            "suspicious": self.suspicious,
            "reward_vector": self.reward_vector,
            "ordered_prefix": self.ordered_prefix,
            "positive_count": self.positive_count,
            "zero_count": self.zero_count,
            "zero_length_mode": self.zero_length_mode,
            "zero_length_mode_count": self.zero_length_mode_count,
            "zero_length_mode_share": round(self.zero_length_mode_share, 4),
            "zero_mode_truncated_count": self.zero_mode_truncated_count,
            "repeated_truncated_losers": self.repeated_truncated_losers,
            "repeated_exact_losers": self.repeated_exact_losers,
        }


def _reward_bits(rewards: Sequence[float]) -> list[str]:
    return ["1" if float(reward) >= 0.5 else "0" for reward in rewards]


def _is_ordered_prefix(bits: Sequence[str]) -> bool:
    if not bits or "1" not in bits or "0" not in bits:
        return False
    first_zero = bits.index("0")
    return all(bit == "1" for bit in bits[:first_zero]) and all(
        bit == "0" for bit in bits[first_zero:]
    )


def detect_reward_shape_manipulation(
    rewards: Sequence[float],
    completion_lengths: Sequence[int],
    truncated_flags: Sequence[bool] | None = None,
    *,
    min_zero_mode_length: int = REWARD_SHAPE_ZERO_MODE_MIN_LENGTH,
    min_mode_share: float = REWARD_SHAPE_ZERO_MODE_MIN_SHARE,
    min_repeated_zero_rollouts: int = REWARD_SHAPE_MIN_REPEATED_ZERO_ROLLOUTS,
    min_exact_zero_rollouts: int = REWARD_SHAPE_MIN_EXACT_ZERO_ROLLOUTS,
) -> RewardShapeMetrics:
    """Detect ordered reward vectors with manufactured loser-length caps.

    The target pattern is visible only across the eight rollouts: correct
    samples first, zero-reward samples last, and the zero-reward suffix cut to
    the same exact completion length. If those repeated loser slots also lack
    natural termination, it is high-confidence poison; if three or more loser
    slots share a non-trivial exact length, it is also suspicious even when
    they technically end with EOS.
    """

    n = min(len(rewards), len(completion_lengths))
    bits = _reward_bits(rewards[:n])
    reward_vector = "".join(bits)
    positive_count = bits.count("1")
    zero_count = bits.count("0")
    ordered_prefix = _is_ordered_prefix(bits)

    empty = RewardShapeMetrics(
        suspicious=False,
        reward_vector=reward_vector,
        ordered_prefix=ordered_prefix,
        positive_count=positive_count,
        zero_count=zero_count,
        zero_length_mode=0,
        zero_length_mode_count=0,
        zero_length_mode_share=0.0,
        zero_mode_truncated_count=0,
        repeated_truncated_losers=False,
        repeated_exact_losers=False,
    )

    if n < 4 or positive_count < 2 or zero_count < 2 or not ordered_prefix:
        return empty

    zero_indices = [idx for idx, bit in enumerate(bits) if bit == "0"]
    zero_lengths = [
        int(completion_lengths[idx])
        for idx in zero_indices
        if idx < len(completion_lengths)
    ]
    if not zero_lengths:
        return empty

    zero_length_mode, zero_length_mode_count = Counter(zero_lengths).most_common(1)[0]
    zero_length_mode_share = zero_length_mode_count / len(zero_lengths)
    min_mode_count = max(
        min_repeated_zero_rollouts,
        ceil(min_mode_share * len(zero_lengths)),
    )
    if (
        zero_length_mode < min_zero_mode_length
        or zero_length_mode_count < min_mode_count
    ):
        return RewardShapeMetrics(
            suspicious=False,
            reward_vector=reward_vector,
            ordered_prefix=ordered_prefix,
            positive_count=positive_count,
            zero_count=zero_count,
            zero_length_mode=zero_length_mode,
            zero_length_mode_count=zero_length_mode_count,
            zero_length_mode_share=zero_length_mode_share,
            zero_mode_truncated_count=0,
            repeated_truncated_losers=False,
            repeated_exact_losers=False,
        )

    flags = list(truncated_flags or [])
    zero_mode_truncated_count = sum(
        1
        for idx in zero_indices
        if idx < len(completion_lengths)
        and int(completion_lengths[idx]) == zero_length_mode
        and idx < len(flags)
        and bool(flags[idx])
    )
    repeated_truncated_losers = (
        zero_mode_truncated_count >= min_repeated_zero_rollouts
    )
    repeated_exact_losers = (
        zero_length_mode_count >= min_exact_zero_rollouts
    )

    return RewardShapeMetrics(
        suspicious=bool(repeated_truncated_losers or repeated_exact_losers),
        reward_vector=reward_vector,
        ordered_prefix=ordered_prefix,
        positive_count=positive_count,
        zero_count=zero_count,
        zero_length_mode=zero_length_mode,
        zero_length_mode_count=zero_length_mode_count,
        zero_length_mode_share=zero_length_mode_share,
        zero_mode_truncated_count=zero_mode_truncated_count,
        repeated_truncated_losers=repeated_truncated_losers,
        repeated_exact_losers=repeated_exact_losers,
    )
