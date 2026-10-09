"""Plan 2C: which refusals of an episode group may be sent again, shared by the validator (it reserves no
(operator, prompt) identity for them) and the miner (it resends the same group in a new envelope). Imports
nothing: a legacy validator loads it without loading the episode modules."""
from __future__ import annotations

RETRYABLE_STAGES = frozenset({"episode_group_in_flight", "episode_session_busy", "episode_directory",
                              "episode_checker_busy", "episode_persist_failed", "episode_checks_in_flight"})
"""Episode refusals the miner may retry at once. ``episode_group_in_flight`` and
``episode_checks_in_flight`` are RATE_LIMITED (the miner's own load: retried after a back-off, not refunded);
the others are WORKER_DROPPED (the validator's capacity, refunded)."""

WORKER_DROPPED_FINAL_STAGES = frozenset({"code_grader_crash"})
"""WORKER_DROPPED stages that are not the validator's passing capacity: never resent."""


def group_resendable(reason: str | None, stage: str | None) -> bool:
    """Whether a refused group may be sent again (before its grading deadline, while the window is open).
    Every other refusal is final for the prompt this window: never resent after it."""
    return stage in RETRYABLE_STAGES or (reason == "worker_dropped" and stage not in WORKER_DROPPED_FINAL_STAGES)


__all__ = ["RETRYABLE_STAGES", "WORKER_DROPPED_FINAL_STAGES", "group_resendable"]
