"""Difficulty auction — the 8 HARDEST submissions win, speed only breaks ties.

Replaces ``batch_selection.select_batch_and_distribute``, whose rule was "the 8
submissions that arrived in the earliest drand rounds win". Measured over 855
live windows, that rule discarded 45 593 valid submissions as ``batch_filled``
against 13 414 accepted — roughly three in four thrown away for arriving late,
before anyone looked at what they contained — and it never priced difficulty at
all. The accepted batch drifted easy (mean k = 4.52 of 8), so the model spent its
gradient re-learning what it already knew.

Here the ranking is:

    1. value descending   — difficulty first (see ``verifier.submission_value``)
    2. drand_round ascending — the earlier 3-second bucket wins the tie
    3. canonical hash     — deterministic, so validators agree on weights

Step 2 is not a rare tie-break: ``v(k)`` takes only 7 distinct values for an
8-rollout binary group, so submissions tie at the top constantly. The mechanism's
real steady state is therefore **a speed race restricted to the hardest prompts**
— which is the intent. Speed keeps the one job it had, pressure on training
throughput, but it can no longer select FOR easy prompts, because an easy group
never reaches the top of the ranking to begin with.

Payout is flat: every filled slot pays ``pool / b``, regardless of value. The
score decides who gets in, not what a slot is worth. Paying proportionally to
value would pull miners toward ever-lower k — the lucky-guess and broken-label
region — whereas a flat-prize tournament already ratchets difficulty upward on
its own: to earn anything you must beat the b-th best submission.

Unfilled slots burn (no redistribution), as before.

See docs/superpowers/specs/2026-07-14-difficulty-auction-design.md
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from reliquary.validator.batch_selection import _within_slot_key
from reliquary.validator.cooldown import CooldownMap


class _ScoredSubmission(Protocol):
    """Duck-typed submission — ``value`` is set by the batcher after grading."""

    hotkey: str
    prompt_idx: int
    merkle_root: bytes
    selection_digest: bytes
    drand_round: int
    value: float


def _rank_key(sub: _ScoredSubmission) -> tuple[float, int, bytes]:
    """Total order over submissions. Hardest first; ties by speed; then hash."""
    return (-sub.value, sub.drand_round, _within_slot_key(sub))


def select_batch_auction(
    submissions: list[Any],
    *,
    b: int,
    cooldown_map: CooldownMap,
    current_window: int,
    pool: float = 1.0,
    max_slots_per_coldkey: int | None = None,
    coldkey_of: Callable[[str], str] | None = None,
) -> tuple[list[Any], dict[str, float]]:
    """Pick the training batch by difficulty and split the pool across winners.

    Args:
        submissions: every graded, validated submission collected this window.
            Each carries a ``value`` (``verifier.submission_value``).
        b: number of slots (= ``B_BATCH``), one per distinct prompt.
        cooldown_map: read-only; prompts in cooldown are skipped.
        pool: window emission budget. Each FILLED slot pays ``pool / b``;
            unfilled slots burn rather than redistribute.
        max_slots_per_coldkey: cap on slots one operator may win per window.
            ``None`` disables the cap. Today's 8-distinct rule is per PROMPT,
            not per operator, which is how one coldkey took 13.1% of emission
            by flooding distinct prompts across many hotkeys.
        coldkey_of: hotkey → coldkey. Defaults to identity (each hotkey is its
            own operator), which is the correct fallback when no metagraph
            mapping is available.

    Returns:
        ``(training_batch, rewards_by_hotkey)``. Does NOT mutate ``cooldown_map``
        — the caller records post-selection, as with the rule this replaces.
    """
    if b <= 0 or not submissions:
        return [], {}

    eligible = [
        sub for sub in submissions
        # value <= 0 means a unanimous group (k=0 or k=8): every GRPO advantage
        # cancels, so there is nothing to train on. This is what replaces the
        # SIGMA_MIN gate — and unlike SIGMA_MIN it does not also throw away the
        # genuinely hard k=1 groups, which were the ones worth the most.
        if sub.value > 0.0
        and not cooldown_map.is_in_cooldown(sub.prompt_idx, current_window)
    ]
    if not eligible:
        return [], {}

    identity: Callable[[str], str] = coldkey_of or (lambda hk: hk)

    training_batch: list[Any] = []
    claimed_prompts: set[int] = set()
    slots_per_coldkey: dict[str, int] = {}

    for sub in sorted(eligible, key=_rank_key):
        if len(training_batch) >= b:
            break
        if sub.prompt_idx in claimed_prompts:
            continue          # one slot per prompt; the highest-ranked took it
        coldkey = identity(sub.hotkey)
        if (
            max_slots_per_coldkey is not None
            and slots_per_coldkey.get(coldkey, 0) >= max_slots_per_coldkey
        ):
            continue          # operator at cap; the next-ranked miner is promoted
        training_batch.append(sub)
        claimed_prompts.add(sub.prompt_idx)
        slots_per_coldkey[coldkey] = slots_per_coldkey.get(coldkey, 0) + 1

    slot_share = pool / b
    rewards: dict[str, float] = {}
    for sub in training_batch:
        rewards[sub.hotkey] = rewards.get(sub.hotkey, 0.0) + slot_share

    return training_batch, rewards
