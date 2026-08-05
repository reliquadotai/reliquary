"""Observation-only difficulty-auction counterfactuals.

This module is intentionally pure. It does not admit submissions, run proofs,
mutate cooldown state, select the production batch, or distribute emission.
That separation lets the validator measure the proposed mechanism without
quietly changing consensus behavior under a "shadow" label.
"""

from __future__ import annotations

import functools
import math
from math import comb
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from reliquary.validator.batch_selection import _within_slot_key


@dataclass(frozen=True)
class ShadowSubmission:
    """Detached candidate values safe to expose to shadow experiments."""

    source_id: int
    hotkey: str
    prompt_idx: int
    drand_round: int
    merkle_root: bytes
    selection_digest: bytes
    rewards: tuple[float, ...]
    in_cooldown: bool = False
    arrival_drand_round: int | None = None


@dataclass(frozen=True)
class DifficultyScore:
    value: float
    mean_reward: float
    reward_std: float
    reward_count: int


@dataclass(frozen=True)
class ShadowCandidate:
    submission: ShadowSubmission
    score: DifficultyScore
    rank: int | None
    eligible: bool
    selected: bool
    operator_id: str | None


@dataclass(frozen=True)
class ShadowAuctionResult:
    candidates: tuple[ShadowCandidate, ...]
    selected: tuple[ShadowSubmission, ...]
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


def gated_difficulty_utility(
    rewards: Iterable[float],
    *,
    sigma_min: float,
    delta: float = 1.0,
) -> float:
    """Return auction difficulty only when the reward vector passes its gate.

    This mirrors the validator's population-sigma eligibility boundary without
    importing environment or bootstrap constants. Callers must supply the
    threshold that applies to the candidate being valued.
    """
    if not math.isfinite(sigma_min) or sigma_min < 0.0:
        raise ValueError("sigma minimum must be finite and non-negative")

    score = difficulty_score(rewards, delta=delta)
    if score.reward_std < 1e-8 or score.reward_std < sigma_min:
        return 0.0
    return score.value


def fractional_reward_lattice(total_tests: int) -> tuple[float, ...]:
    """Return every attainable ``passed / total_tests`` reward.

    ``total_tests=1`` is the Math lattice ``{0, 1}``; larger denominators model
    the fractional rewards emitted by the Code grader.
    """
    if (
        isinstance(total_tests, bool)
        or not isinstance(total_tests, int)
        or total_tests <= 0
    ):
        raise ValueError("total tests must be a positive integer")
    return tuple(passed / total_tests for passed in range(total_tests + 1))


ROBUST_UTILITY_MAX_OUTCOMES = 200_000


def robust_truncation_utility(
    rewards: Iterable[float],
    *,
    sigma_min: float,
    truncated_index: int | None = None,
    truncated_indices: Iterable[int] = (),
    attainable_rewards: Iterable[float] = (),
    delta: float = 1.0,
) -> float:
    """Return the least utility across all outcomes of one truncated rollout.

    A truncated rollout has an unknown reward. Replacing it with every value in
    its exact environment-specific lattice and minimizing the *gated* utility
    prevents a manufactured truncation from improving either sigma eligibility
    or auction difficulty. Several rollouts may be unknown; every joint
    assignment is priced, so the guarantee does not weaken as the per-
    environment truncation allowance rises.
    """
    values = tuple(float(reward) for reward in rewards)
    indices = tuple(truncated_indices)
    if truncated_index is not None:
        indices = (truncated_index, *indices)
    indices = tuple(dict.fromkeys(indices))
    if not indices:
        return gated_difficulty_utility(
            values,
            sigma_min=sigma_min,
            delta=delta,
        )
    for index in indices:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(values)
        ):
            raise ValueError("truncated index must identify one reward")

    lattice = tuple(dict.fromkeys(float(reward) for reward in attainable_rewards))
    if not lattice:
        raise ValueError("attainable rewards must not be empty")
    if any(not math.isfinite(reward) for reward in lattice):
        raise ValueError("attainable rewards must be finite")
    if any(reward < 0.0 or reward > 1.0 for reward in lattice):
        raise ValueError("attainable rewards must be in [0, 1]")

    # Validate the observed vector too, even though the unknown position will be
    # replaced below. A malformed validator reward should never be hidden.
    difficulty_score(values, delta=delta)

    # The score is symmetric in the rewards, so distinct OUTCOMES are multisets
    # of assignments, not ordered tuples: enumerate combinations with
    # replacement rather than the full product.
    from itertools import combinations_with_replacement

    outcome_count = comb(len(lattice) + len(indices) - 1, len(indices))
    if outcome_count > ROBUST_UTILITY_MAX_OUTCOMES:
        # Refuse to guess. Returning 0 makes the candidate ineligible, which is
        # the SAFE direction: a miner can never widen its own lattice to buy a
        # pass, only to be rejected.
        return 0.0

    utilities: list[float] = []
    for assignment in combinations_with_replacement(lattice, len(indices)):
        outcome = list(values)
        for index, attainable_reward in zip(indices, assignment):
            outcome[index] = attainable_reward
        utilities.append(
            gated_difficulty_utility(
                outcome,
                sigma_min=sigma_min,
                delta=delta,
            )
        )
    return min(utilities)


def conservative_difficulty_score(
    rewards: Iterable[float],
    *,
    truncated_count: int,
    delta: float = 1.0,
) -> DifficultyScore:
    """Difficulty score under the interpretation least favourable to the miner.

    A truncated rollout produced no gradeable answer, so it enters ``rewards``
    as 0. The validator cannot know whether it would have been correct — and the
    miner can *choose* to create one (suppressing EOS runs a would-be-correct
    rollout to the cap), which makes the prompt look harder and pays more. That
    manipulation is not reliably detectable, so it is priced out instead:
    every interpretation of the truncated rollouts is scored and the **minimum**
    is returned.

    Because the group's true outcome is always one of the interpretations, and
    we return the minimum over all of them, a manipulated group can never score
    above the honest group it came from — the gain is zero by construction, for
    any value function, with no threshold to calibrate and no detection.

    ``truncated_count`` truncated rollouts are graded 0, so the interpretations
    are generated by raising ``j`` of those zeros to 1.0 for ``j`` in
    ``0..truncated_count`` (the score depends only on the multiset of rewards,
    so *which* zeros are raised does not matter).
    """
    values = [float(reward) for reward in rewards]
    count = max(0, int(truncated_count))
    if count == 0 or not values:
        return difficulty_score(values, delta=delta)

    zero_positions = [i for i, value in enumerate(values) if value == 0.0][:count]
    best: DifficultyScore | None = None
    for j in range(len(zero_positions) + 1):
        candidate = list(values)
        for i in zero_positions[:j]:
            candidate[i] = 1.0
        score = difficulty_score(candidate, delta=delta)
        if best is None or score.value < best.value:
            best = score
    return best if best is not None else difficulty_score(values, delta=delta)


def flat_auction_value(value: float) -> float:
    """Collapse a positive auction value to 1.0 under flat valuation.

    Zero stays zero: degenerate / out-of-zone groups must keep ranking last.
    Identity when ``DIFFICULTY_AUCTION_FLAT_VALUE`` is off.
    """
    from reliquary.constants import DIFFICULTY_AUCTION_FLAT_VALUE

    if DIFFICULTY_AUCTION_FLAT_VALUE and value > 0.0:
        return 1.0
    return value


def auction_difficulty_score(
    rewards: Iterable[float],
    *,
    truncated_count: int = 0,
    delta: float | None = None,
) -> DifficultyScore:
    """The score the auction ranks and pays on — single source of truth.

    Plain difficulty score, except that when ``CONSERVATIVE_TRUNCATION_VALUE``
    is set a group carrying truncated rollouts is valued under the least
    favourable interpretation of them (see ``conservative_difficulty_score``),
    which removes any gain from manufacturing a truncated rollout.

    Under ``DIFFICULTY_AUCTION_FLAT_VALUE`` the ranked scalar collapses to
    the in-zone indicator (mean/std telemetry stays real); tie-breaks then
    order the equally-valued candidates.
    """
    from reliquary.constants import (
        CONSERVATIVE_TRUNCATION_VALUE,
        DIFFICULTY_AUCTION_DELTA,
    )

    resolved_delta = DIFFICULTY_AUCTION_DELTA if delta is None else delta
    if CONSERVATIVE_TRUNCATION_VALUE and truncated_count > 0:
        score = conservative_difficulty_score(
            rewards, truncated_count=truncated_count, delta=resolved_delta
        )
    else:
        score = difficulty_score(rewards, delta=resolved_delta)
    flat = flat_auction_value(score.value)
    if flat != score.value:
        return DifficultyScore(
            flat, score.mean_reward, score.reward_std, score.reward_count
        )
    return score


def auction_value(
    rewards: Iterable[float],
    *,
    truncated_count: int = 0,
    delta: float | None = None,
) -> float:
    """``auction_difficulty_score`` reduced to the ranked/paid scalar."""
    return auction_difficulty_score(
        rewards, truncated_count=truncated_count, delta=delta
    ).value

@functools.lru_cache(maxsize=None)
def max_difficulty_value(reward_count: int, *, delta: float = 1.0) -> float:
    """Exact float ceiling of ``difficulty_score`` over achievable rewards.

    For a fixed mean, std is maximized only by extremal (all 0/1) profiles,
    so the global maximum is attained on a binary profile; enumerating k is
    exhaustive. Computed through ``difficulty_score`` itself so the value
    compares bit-for-bit (``==``) with candidate scores.
    """
    if reward_count <= 0:
        return 0.0
    return max(
        difficulty_score(
            [1.0] * k + [0.0] * (reward_count - k), delta=delta
        ).value
        for k in range(reward_count + 1)
    )


def submission_score(
    submission: ShadowSubmission,
    *,
    delta: float,
) -> DifficultyScore:
    return difficulty_score(submission.rewards, delta=delta)


def _rank_key(
    item: tuple[ShadowSubmission, DifficultyScore],
) -> tuple[float, int, bytes]:
    """Mirror of the production ranking: score gates, validator-observed
    arrival breaks ties (miner-submitted round is the fallback), canonical
    hash orders within a tier for display only."""
    submission, score = item
    arrival = getattr(submission, "arrival_drand_round", None)
    chronological = (
        arrival if arrival is not None else submission.drand_round
    )
    return (
        -score.value,
        int(chronological),
        _within_slot_key(submission),
    )


def select_shadow_auction(
    submissions: Iterable[ShadowSubmission],
    *,
    b: int,
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

    materialized = tuple(submissions)
    source_ids = [submission.source_id for submission in materialized]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("shadow submissions must have unique source ids")

    scored = [
        (submission, submission_score(submission, delta=delta))
        for submission in materialized
    ]
    eligible = [
        item
        for item in scored
        if item[1].value > 0.0
        and not item[0].in_cooldown
    ]
    ranked = sorted(eligible, key=_rank_key)

    operator_ids: dict[int, str | None] = {}
    if operator_of is not None:
        for submission, _score in ranked:
            operator = operator_of(submission.hotkey)
            operator_ids[submission.source_id] = (
                str(operator).strip() if operator is not None else None
            ) or None

    cap_requested = max_slots_per_operator is not None
    mapping_complete = bool(ranked) and operator_of is not None and all(
        operator_ids.get(submission.source_id) is not None
        for submission, _score in ranked
    )
    cap_applied = cap_requested and mapping_complete

    selected: list[ShadowSubmission] = []
    selected_ids: set[int] = set()
    claimed_prompts: set[int] = set()
    slots_by_operator: Counter[str] = Counter()
    if b > 0:
        for submission, _score in ranked:
            if len(selected) >= b:
                break
            if submission.prompt_idx in claimed_prompts:
                continue
            operator = operator_ids.get(submission.source_id)
            if (
                cap_applied
                and operator is not None
                and slots_by_operator[operator] >= max_slots_per_operator
            ):
                continue
            selected.append(submission)
            selected_ids.add(submission.source_id)
            claimed_prompts.add(submission.prompt_idx)
            if operator is not None:
                slots_by_operator[operator] += 1

    ranks = {
        submission.source_id: rank
        for rank, (submission, _score) in enumerate(ranked, start=1)
    }
    candidates = tuple(
        ShadowCandidate(
            submission=submission,
            score=score,
            rank=ranks.get(submission.source_id),
            eligible=submission.source_id in ranks,
            selected=submission.source_id in selected_ids,
            operator_id=operator_ids.get(submission.source_id),
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
