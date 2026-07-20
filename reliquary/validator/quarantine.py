"""Training quarantine checks for selected GRPO windows.

These checks are deliberately proof-free and conservative. They do not decide
miner emissions; they only decide whether the current selected batch should be
allowed to mutate the validator's train model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    TRAINING_QUARANTINE_ENABLED,
    TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH,
    TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE,
    TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH,
    TRAINING_QUARANTINE_REJECT_SPIKE_MIN,
    TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS,
    TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS,
    TRAINING_QUARANTINE_LONG_ZERO_TAIL_MIN_LENGTH,
    TRAINING_QUARANTINE_REWARD_SHAPE_MIN_GROUPS,
    TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS,
)


HIGH_RISK_REJECT_REASONS = {
    "reward_distribution",
    "bad_termination",
    "distribution_suspicious",
    "tokens_mismatch",
    "reward_mismatch",
    "reward_shape_suspicious",
}


@dataclass(frozen=True)
class TrainingQuarantineDecision:
    quarantined: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_archive(self) -> dict[str, Any]:
        return {
            "quarantined": self.quarantined,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def _completion_length(rollout: Any) -> int:
    meta = (getattr(rollout, "commit", {}) or {}).get("rollout", {}) or {}
    if "completion_length" in meta:
        return int(meta.get("completion_length") or 0)
    tokens = list((getattr(rollout, "commit", {}) or {}).get("tokens", []))
    prompt_len = int(meta.get("prompt_length", 0) or 0)
    return max(0, len(tokens) - prompt_len)


def _binary_reward_vector(group: Any) -> str:
    bits: list[str] = []
    for rollout in getattr(group, "rollouts", []):
        reward = float(getattr(rollout, "reward", 0.0))
        bits.append("1" if reward >= 0.5 else "0")
    return "".join(bits)


def assess_training_batch(
    batch: list[Any],
    *,
    reject_counts: dict[str, int] | None = None,
    enabled: bool = TRAINING_QUARANTINE_ENABLED,
) -> TrainingQuarantineDecision:
    """Return whether a selected batch should be excluded from training.

    The validator still archives and credits quarantined windows. Quarantine is
    only a model-health gate: if the batch looks poisoned, do not run GRPO on
    it and do not publish a checkpoint from it.
    """

    n_groups = len(batch)
    if not enabled or n_groups == 0:
        return TrainingQuarantineDecision(
            quarantined=False,
            metrics={"n_groups": n_groups},
        )

    reasons: list[str] = []

    hotkey_counts = Counter(str(getattr(group, "hotkey", "")) for group in batch)
    max_hotkey_groups = max(hotkey_counts.values(), default=0)
    max_hotkey_share = max_hotkey_groups / n_groups
    # Hotkey dominance by itself is not a poison signature: if only one honest
    # miner is producing valid frontier work, training should still progress.
    # Keep the metric for operators/EMA policy, but quarantine on actual data
    # shape signals below (reward-vector dominance, cap/extreme length, reject
    # spikes) rather than identity concentration alone.

    reward_vectors = [_binary_reward_vector(group) for group in batch]
    vector_counts = Counter(reward_vectors)
    dominant_vector, dominant_vector_groups = vector_counts.most_common(1)[0]
    dominant_vector_share = dominant_vector_groups / n_groups
    if (
        dominant_vector_groups >= TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS
        and dominant_vector_share >= TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE
    ):
        # Reward-vector concentration is important operator telemetry, but it
        # is not by itself a high-confidence poison signal. During the current
        # OpenMath phase, frontier/cherry-picked submissions naturally cluster
        # around a small set of binary vectors; freezing training on that shape
        # alone lets the meta stall on one checkpoint without proving the text
        # is bad training data.
        pass

    completion_lengths: list[int] = []
    cap_length_groups = 0
    extreme_length_groups = 0
    reward_shape_groups = 0
    long_zero_tail_shape_groups = 0
    for group in batch:
        group_lengths = [
            _completion_length(rollout)
            for rollout in getattr(group, "rollouts", [])
        ]
        completion_lengths.extend(group_lengths)
        reward_shape = getattr(group, "reward_shape", {}) or {}
        if bool(reward_shape.get("suspicious", False)):
            reward_shape_groups += 1
            if int(reward_shape.get("zero_length_mode", 0) or 0) >= (
                TRAINING_QUARANTINE_LONG_ZERO_TAIL_MIN_LENGTH
            ):
                long_zero_tail_shape_groups += 1
        if any(length >= MAX_NEW_TOKENS_PROTOCOL_CAP for length in group_lengths):
            cap_length_groups += 1
        if any(
            length >= TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH
            for length in group_lengths
        ):
            extreme_length_groups += 1

    max_completion_length = max(completion_lengths, default=0)
    mean_completion_length = (
        sum(completion_lengths) / len(completion_lengths)
        if completion_lengths else 0.0
    )
    cap_length_rollouts = sum(
        1 for length in completion_lengths
        if length >= MAX_NEW_TOKENS_PROTOCOL_CAP
    )
    extreme_length_rollouts = sum(
        1 for length in completion_lengths
        if length >= TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH
    )
    if (
        cap_length_groups >= TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS
        or (
            cap_length_groups >= 2
            and cap_length_rollouts >= TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS
        )
    ):
        reasons.append("cap_length_density")
    elif (
        extreme_length_groups >= TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS
        or (
            extreme_length_groups >= 2
            and extreme_length_rollouts >= TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS
        )
    ):
        reasons.append("extreme_completion_length_density")
    if mean_completion_length >= TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH:
        reasons.append("mean_completion_length_high")
    if long_zero_tail_shape_groups > 0:
        reasons.append("long_zero_tail_reward_shape")
    elif reward_shape_groups >= TRAINING_QUARANTINE_REWARD_SHAPE_MIN_GROUPS:
        reasons.append("reward_shape_density")

    reject_counts = reject_counts or {}
    high_risk_rejects = sum(
        int(reject_counts.get(reason, 0))
        for reason in HIGH_RISK_REJECT_REASONS
    )
    if high_risk_rejects >= TRAINING_QUARANTINE_REJECT_SPIKE_MIN:
        reasons.append("high_risk_reject_spike")

    metrics = {
        "n_groups": n_groups,
        "n_rollouts": len(completion_lengths),
        "max_hotkey_groups": max_hotkey_groups,
        "max_hotkey_share": max_hotkey_share,
        "dominant_reward_vector": dominant_vector,
        "dominant_reward_vector_groups": dominant_vector_groups,
        "dominant_reward_vector_share": dominant_vector_share,
        "dominant_reward_vector_quarantine_threshold": (
            dominant_vector_groups >= TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS
            and dominant_vector_share >= TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE
        ),
        "mean_completion_length": mean_completion_length,
        "max_completion_length": max_completion_length,
        "cap_length_rollouts": cap_length_rollouts,
        "cap_length_groups": cap_length_groups,
        "extreme_length_rollouts": extreme_length_rollouts,
        "extreme_length_groups": extreme_length_groups,
        "reward_shape_groups": reward_shape_groups,
        "reward_shape_group_share": reward_shape_groups / n_groups,
        "long_zero_tail_shape_groups": long_zero_tail_shape_groups,
        "high_risk_rejects": high_risk_rejects,
    }
    return TrainingQuarantineDecision(
        quarantined=bool(reasons),
        reasons=reasons,
        metrics=metrics,
    )
