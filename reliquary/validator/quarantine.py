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
    TRAINING_QUARANTINE_MAX_HOTKEY_SHARE,
    TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH,
    TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE,
    TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH,
    TRAINING_QUARANTINE_REJECT_SPIKE_MIN,
    TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS,
)


HIGH_RISK_REJECT_REASONS = {
    "reward_distribution",
    "bad_termination",
    "distribution_suspicious",
    "tokens_mismatch",
    "reward_mismatch",
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
    if max_hotkey_share >= TRAINING_QUARANTINE_MAX_HOTKEY_SHARE:
        reasons.append("hotkey_batch_dominance")

    reward_vectors = [_binary_reward_vector(group) for group in batch]
    vector_counts = Counter(reward_vectors)
    dominant_vector, dominant_vector_groups = vector_counts.most_common(1)[0]
    dominant_vector_share = dominant_vector_groups / n_groups
    if (
        dominant_vector_groups >= TRAINING_QUARANTINE_REWARD_VECTOR_MIN_GROUPS
        and dominant_vector_share >= TRAINING_QUARANTINE_MAX_REWARD_VECTOR_SHARE
    ):
        reasons.append("reward_vector_dominance")

    completion_lengths = [
        _completion_length(rollout)
        for group in batch
        for rollout in getattr(group, "rollouts", [])
    ]
    max_completion_length = max(completion_lengths, default=0)
    mean_completion_length = (
        sum(completion_lengths) / len(completion_lengths)
        if completion_lengths else 0.0
    )
    cap_length_rollouts = sum(
        1 for length in completion_lengths
        if length >= MAX_NEW_TOKENS_PROTOCOL_CAP
    )
    if cap_length_rollouts:
        reasons.append("cap_length_rollout")
    elif max_completion_length >= TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH:
        reasons.append("extreme_completion_length")
    if mean_completion_length >= TRAINING_QUARANTINE_MAX_MEAN_COMPLETION_LENGTH:
        reasons.append("mean_completion_length_high")

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
        "mean_completion_length": mean_completion_length,
        "max_completion_length": max_completion_length,
        "cap_length_rollouts": cap_length_rollouts,
        "high_risk_rejects": high_risk_rejects,
    }
    return TrainingQuarantineDecision(
        quarantined=bool(reasons),
        reasons=reasons,
        metrics=metrics,
    )
