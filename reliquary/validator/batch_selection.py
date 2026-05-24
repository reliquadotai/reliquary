"""Drand-round-anchored batch selection + emission distribution (design A').

Called once per window at seal time. Two things happen here:

1. **Pick the B distinct prompts that go into the training step.** Order is
   driven by the drand round each miner attached at submit time. Submissions
   are bucketed into 3-second slots (one drand quicknet round per slot);
   earlier slots fill the batch first. Within a slot, distinct prompts are
   admitted in a canonical hash order so two validators with different
   in-memory orderings still agree.

2. **Compute the emission distribution.** Each filled slot is worth
   ``pool / B``. Within a slot, the K miners who submitted for that
   ``(round, prompt)`` split that share equally. So a miner sybiling the
   same prompt with N hotkeys in the same drand round earns N × (1/N) =
   identical total payout to a single hotkey — registration costs make
   sybiling strictly wasteful.

Unfilled slots burn their share (no redistribution). This keeps emission
math simple and stable; partial fills happen when miner activity is low,
and the burn appears in the on-chain weight setter as missing weight for
that window.

v2.3+: replaces the v2.2 pure-FIFO ``select_batch``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from reliquary.validator.cooldown import CooldownMap


class _SubmissionLike(Protocol):
    """Duck-typed submission — works with any class exposing these attrs."""

    hotkey: str
    prompt_idx: int
    merkle_root: bytes
    drand_round: int


def _within_slot_key(sub: _SubmissionLike) -> bytes:
    """Canonical sort key for tie-breaking within a (round, prompt) bucket.

    Two validators may receive submissions in different network order; both
    sort by this key to agree on which submission represents the slot when
    multiple prompts compete inside the same drand round.
    """
    h = hashlib.sha256()
    h.update(sub.hotkey.encode())
    h.update(sub.prompt_idx.to_bytes(8, "big", signed=False))
    h.update(sub.merkle_root)
    return h.digest()


def _prompt_canonical_key(prompt_idx: int) -> bytes:
    """Canonical order for prompts seen in the same drand round."""
    return hashlib.sha256(prompt_idx.to_bytes(8, "big", signed=False)).digest()


def select_batch_and_distribute(
    submissions: list[Any],
    *,
    b: int,
    cooldown_map: CooldownMap,
    current_window: int,
    pool: float = 1.0,
) -> tuple[list[Any], dict[str, float]]:
    """Pick the training batch and the reward distribution.

    Args:
        submissions: all GRAIL-validated submissions for the window, in
            arbitrary order. Each carries an attached ``drand_round`` that
            was already validated at accept time.
        b: training batch size (= B_BATCH). Equals the number of slots in
            the emission pool.
        cooldown_map: read-only view of which prompts are in cooldown.
            Prompts in cooldown are skipped entirely.
        current_window: needed to evaluate cooldown.
        pool: total emission budget for the window. Each filled slot pays
            ``pool / b`` (NOT ``pool / filled_slots`` — under-filled
            windows burn unused share rather than redistributing).

    Returns:
        (training_batch, rewards_by_hotkey)
        - training_batch: up to ``b`` submissions, one per filled slot. The
          canonical representative for each ``(round, prompt)`` slot is the
          submission with the smallest ``_within_slot_key`` (= deterministic
          across validators).
        - rewards_by_hotkey: dict hotkey → float. Sums to (slots_filled / b)
          × pool, with unfilled slots burned.

    Boundary-round fair split
    -------------------------
    When iterating rounds in ascending order, the round that first brings
    ``len(training_batch)`` past ``b`` is the "boundary round". Earlier
    code stopped the moment ``b`` was reached, which meant: within the
    boundary round, only the prompts with the smallest canonical hashes
    got slots, and every other miner in that same drand-3-s bucket
    earned zero. That's unfair — those miners arrived at the same
    chronological tier; the canonical hash is just a tiebreaker, not a
    priority.

    The boundary round now distributes its share fairly across ALL
    prompts in it:

      per_prompt_boundary = remaining_slots × slot_share / N_prompts_in_round
      per_miner_on_prompt = per_prompt_boundary / K_p

    Total emission paid in the boundary round equals
    ``remaining_slots × slot_share`` (= exactly what the boundary round
    is "worth"). The training batch still trains on
    ``remaining_slots`` prompts (canonical hash order) — that's just an
    implementation cap on training cost, not an economic statement.

    Sybil invariants preserved:
      * Same-prompt sybil neutral: K hotkeys on one prompt split the
        per_prompt share K-way, total payout per prompt is fixed
        independent of K.
      * Different-prompt sybil tax: K hotkeys on K different prompts in
        the boundary round each earn per_prompt_boundary, registration
        burn is the tax.
      * Conservation: total reward paid in boundary round equals what
        a full non-boundary round would have paid for the same slot
        count.

    Does NOT mutate ``cooldown_map`` — the caller records post-selection.
    """
    if b <= 0 or not submissions:
        return [], {}

    # Group: drand_round → prompt_idx → list[submission]
    by_round: dict[int, dict[int, list[Any]]] = {}
    for sub in submissions:
        if cooldown_map.is_in_cooldown(sub.prompt_idx, current_window):
            continue
        prompts = by_round.setdefault(sub.drand_round, {})
        prompts.setdefault(sub.prompt_idx, []).append(sub)
    if not by_round:
        return [], {}

    slot_share = pool / b
    training_batch: list[Any] = []
    rewards: dict[str, float] = {}
    claimed_prompts: set[int] = set()
    rounds_ascending = sorted(by_round)
    for round_no in rounds_ascending:
        remaining = b - len(training_batch)
        if remaining <= 0:
            break
        prompts_in_round = by_round[round_no]
        # Filter to prompts not yet claimed by an earlier round (a
        # prompt gets at most one slot across all rounds; the earliest
        # round wins).
        available = {
            p: subs
            for p, subs in prompts_in_round.items()
            if p not in claimed_prompts
        }
        if not available:
            continue
        num_prompts = len(available)

        if num_prompts <= remaining:
            # Full-inclusion case: every prompt in this round gets its
            # own slot. Same as the original algorithm.
            for prompt_idx in sorted(available, key=_prompt_canonical_key):
                slot_subs = available[prompt_idx]
                k_p = len(slot_subs)
                per_miner = slot_share / k_p
                for sub in slot_subs:
                    rewards[sub.hotkey] = rewards.get(sub.hotkey, 0.0) + per_miner
                representative = min(slot_subs, key=_within_slot_key)
                training_batch.append(representative)
                claimed_prompts.add(prompt_idx)
        else:
            # Boundary round: more candidate prompts than remaining
            # slots. Spread the remaining slot value across every prompt
            # in this round (fair within the drand bucket), still split
            # K-way within each prompt (same-prompt sybil neutral).
            per_prompt = remaining * slot_share / num_prompts
            for prompt_idx, slot_subs in available.items():
                k_p = len(slot_subs)
                per_miner = per_prompt / k_p
                for sub in slot_subs:
                    rewards[sub.hotkey] = rewards.get(sub.hotkey, 0.0) + per_miner
            # Training step still trains on at most ``remaining`` prompts.
            # Pick them by canonical hash order (deterministic across
            # validators); the unpicked prompts are paid out above but
            # don't go through the forward pass.
            chosen = sorted(available, key=_prompt_canonical_key)[:remaining]
            for prompt_idx in chosen:
                representative = min(available[prompt_idx], key=_within_slot_key)
                training_batch.append(representative)
                claimed_prompts.add(prompt_idx)
            # Boundary round consumes all remaining slot value, so any
            # later round contributes zero — stop iterating.
            break

    return training_batch, rewards


def explain_batch_selection(
    submissions: list[Any],
    *,
    b: int,
    cooldown_map: CooldownMap,
    current_window: int,
    pool: float = 1.0,
) -> dict[int, dict[str, Any]]:
    """Return per-submission final-selection metadata without mutating state.

    The traversal mirrors ``select_batch_and_distribute`` so operators can
    distinguish "accepted into the pool" from "selected for training" from
    "rewarded by emission sharing". Keys are ``id(submission)``.
    """
    meta: dict[int, dict[str, Any]] = {
        id(sub): {
            "accepted_into_pool": True,
            "selected_for_batch": False,
            "rewarded": False,
            "reward_amount": 0.0,
            "canonical_rank": None,
            "selection_reason": "not_reached_before_batch_filled",
        }
        for sub in submissions
    }
    if b <= 0 or not submissions:
        return meta

    by_round: dict[int, dict[int, list[Any]]] = {}
    for sub in submissions:
        if cooldown_map.is_in_cooldown(sub.prompt_idx, current_window):
            meta[id(sub)]["selection_reason"] = "prompt_in_cooldown_at_selection"
            continue
        prompts = by_round.setdefault(sub.drand_round, {})
        prompts.setdefault(sub.prompt_idx, []).append(sub)
    if not by_round:
        return meta

    slot_share = pool / b
    selected_count = 0
    claimed_prompts: set[int] = set()
    rank = 0

    for round_no in sorted(by_round):
        remaining = b - selected_count
        if remaining <= 0:
            break

        available = {
            p: subs
            for p, subs in by_round[round_no].items()
            if p not in claimed_prompts
        }
        if not available:
            continue

        sorted_prompts = sorted(available, key=_prompt_canonical_key)

        if len(sorted_prompts) <= remaining:
            for prompt_idx in sorted_prompts:
                rank += 1
                slot_subs = available[prompt_idx]
                representative = min(slot_subs, key=_within_slot_key)
                per_miner = slot_share / len(slot_subs)
                for sub in slot_subs:
                    sub_meta = meta[id(sub)]
                    sub_meta.update({
                        "rewarded": True,
                        "reward_amount": per_miner,
                        "canonical_rank": rank,
                        "selection_reason": (
                            "selected_for_batch"
                            if sub is representative
                            else "rewarded_same_prompt"
                        ),
                    })
                    if sub is representative:
                        sub_meta["selected_for_batch"] = True
                selected_count += 1
                claimed_prompts.add(prompt_idx)
        else:
            chosen = set(sorted_prompts[:remaining])
            per_prompt = remaining * slot_share / len(sorted_prompts)
            for prompt_idx in sorted_prompts:
                rank += 1
                slot_subs = available[prompt_idx]
                representative = min(slot_subs, key=_within_slot_key)
                per_miner = per_prompt / len(slot_subs)
                for sub in slot_subs:
                    reason = (
                        "selected_boundary_canonical"
                        if prompt_idx in chosen and sub is representative
                        else "rewarded_same_prompt"
                        if prompt_idx in chosen
                        else "same_trigger_round_lost_canonical_ordering"
                    )
                    sub_meta = meta[id(sub)]
                    sub_meta.update({
                        "rewarded": True,
                        "reward_amount": per_miner,
                        "canonical_rank": rank,
                        "selection_reason": reason,
                    })
                    if prompt_idx in chosen and sub is representative:
                        sub_meta["selected_for_batch"] = True
            break

    return meta
