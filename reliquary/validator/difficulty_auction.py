"""Observation-only difficulty-auction counterfactuals.

This module is intentionally pure. It does not admit submissions, run proofs,
mutate cooldown state, select the production batch, or distribute emission.
That separation lets the validator measure the proposed mechanism without
quietly changing consensus behavior under a "shadow" label.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from reliquary.validator.batch_selection import _within_slot_key
from reliquary.validator.cooldown import CooldownMap


class ScoredSubmission(Protocol):
    hotkey: str
    prompt_idx: int
    drand_round: int
    merkle_root: bytes
    selection_digest: bytes
    rollouts: list[Any]


@dataclass(frozen=True)
class DifficultyScore:
    value: float
    mean_reward: float
    reward_std: float
    reward_count: int


@dataclass(frozen=True)
class ShadowCandidate:
    submission: Any
    score: DifficultyScore
    rank: int | None
    eligible: bool
    selected: bool
    operator_id: str | None


@dataclass(frozen=True)
class ShadowAuctionResult:
    candidates: tuple[ShadowCandidate, ...]
    selected: tuple[Any, ...]
    eligible_count: int
    distinct_prompt_count: int
    operator_cap_requested: int | None
    operator_cap_applied: bool
    operator_mapping_complete: bool


def difficulty_score(
    rewards: Iterable[float],
    *,
    delta: float = 1.0,
) -> DifficultyScore:
    """Return ``std(rewards) * (1 - mean(rewards)) ** delta``.

    Validator rewards are required to be finite and inside ``[0, 1]``. A bad
    reward domain is a programming/configuration error, not a zero-value group:
    silently ranking it last would hide an invalid counterfactual.
    """
    values = tuple(float(reward) for reward in rewards)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("difficulty delta must be finite and non-negative")
    if any(not math.isfinite(reward) for reward in values):
        raise ValueError("difficulty rewards must be finite")
    if any(reward < 0.0 or reward > 1.0 for reward in values):
        raise ValueError("difficulty rewards must be in [0, 1]")

    count = len(values)
    if count == 0:
        return DifficultyScore(0.0, 0.0, 0.0, 0)

    mean_reward = sum(values) / count
    variance = sum(
        (reward - mean_reward) ** 2 for reward in values
    ) / count
    reward_std = variance**0.5
    value = reward_std * (1.0 - mean_reward) ** delta
    return DifficultyScore(value, mean_reward, reward_std, count)


def submission_score(
    submission: ScoredSubmission,
    *,
    delta: float,
) -> DifficultyScore:
    return difficulty_score(
        (float(rollout.reward) for rollout in submission.rollouts),
        delta=delta,
    )


def _rank_key(
    item: tuple[Any, DifficultyScore],
) -> tuple[float, int, bytes]:
    submission, score = item
    return (
        -score.value,
        int(submission.drand_round),
        _within_slot_key(submission),
    )


def select_shadow_auction(
    submissions: Iterable[Any],
    *,
    b: int,
    cooldown_map: CooldownMap,
    current_window: int,
    delta: float,
    max_slots_per_operator: int | None = None,
    operator_of: Callable[[str], str | None] | None = None,
) -> ShadowAuctionResult:
    """Rank a fully validated pool without mutating production state.

    A requested operator cap is applied only when every eligible hotkey maps to
    a non-empty operator identity. Identity fallback would make a Sybil guard
    look active while allowing the exact multi-hotkey bypass it is meant to
    measure, so incomplete mappings disable the capped counterfactual and are
    surfaced explicitly in the result.
    """
    if b < 0:
        raise ValueError("batch size must be non-negative")
    if max_slots_per_operator is not None and max_slots_per_operator <= 0:
        raise ValueError("operator slot cap must be positive")

    scored = [
        (submission, submission_score(submission, delta=delta))
        for submission in submissions
    ]
    eligible = [
        item
        for item in scored
        if item[1].value > 0.0
        and not cooldown_map.is_in_cooldown(
            item[0].prompt_idx, current_window
        )
    ]
    ranked = sorted(eligible, key=_rank_key)

    operator_ids: dict[int, str | None] = {}
    if operator_of is not None:
        for submission, _score in ranked:
            operator = operator_of(submission.hotkey)
            operator_ids[id(submission)] = (
                str(operator).strip() if operator is not None else None
            ) or None

    cap_requested = max_slots_per_operator is not None
    mapping_complete = bool(ranked) and operator_of is not None and all(
        operator_ids.get(id(submission)) is not None
        for submission, _score in ranked
    )
    cap_applied = cap_requested and mapping_complete

    selected: list[Any] = []
    selected_ids: set[int] = set()
    claimed_prompts: set[int] = set()
    slots_by_operator: Counter[str] = Counter()
    if b > 0:
        for submission, _score in ranked:
            if len(selected) >= b:
                break
            if submission.prompt_idx in claimed_prompts:
                continue
            operator = operator_ids.get(id(submission))
            if (
                cap_applied
                and operator is not None
                and slots_by_operator[operator] >= max_slots_per_operator
            ):
                continue
            selected.append(submission)
            selected_ids.add(id(submission))
            claimed_prompts.add(submission.prompt_idx)
            if operator is not None:
                slots_by_operator[operator] += 1

    ranks = {
        id(submission): rank
        for rank, (submission, _score) in enumerate(ranked, start=1)
    }
    candidates = tuple(
        ShadowCandidate(
            submission=submission,
            score=score,
            rank=ranks.get(id(submission)),
            eligible=id(submission) in ranks,
            selected=id(submission) in selected_ids,
            operator_id=operator_ids.get(id(submission)),
        )
        for submission, score in scored
    )
    return ShadowAuctionResult(
        candidates=candidates,
        selected=tuple(selected),
        eligible_count=len(ranked),
        distinct_prompt_count=len({
            submission.prompt_idx for submission, _score in ranked
        }),
        operator_cap_requested=max_slots_per_operator,
        operator_cap_applied=cap_applied,
        operator_mapping_complete=mapping_complete,
    )
