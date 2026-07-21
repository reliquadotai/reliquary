"""Drand-round-anchored batch selection + emission distribution tests."""

from dataclasses import dataclass

from reliquary.validator.batch_selection import (
    _within_slot_key,
    select_batch_and_distribute,
)
from reliquary.validator.cooldown import CooldownMap


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    drand_round: int
    merkle_root: bytes = b"\x00" * 32
    selection_digest: bytes = b"\x00" * 32


def _sub(hotkey, prompt_idx, drand_round, merkle_root=None):
    digest = merkle_root or hotkey.encode().ljust(32, b"\x00")
    return FakeSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        drand_round=drand_round,
        merkle_root=digest,
        selection_digest=digest,
    )


def test_within_slot_key_uses_generation_digest_not_payload_root():
    sub = _sub("miner", prompt_idx=42, drand_round=7)
    original_key = _within_slot_key(sub)

    sub.merkle_root = b"\xff" * 32
    assert _within_slot_key(sub) == original_key

    sub.selection_digest = b"\x01" * 32
    assert _within_slot_key(sub) != original_key


def test_empty_pool_returns_empty():
    cd = CooldownMap(cooldown_windows=50)
    batch, rewards = select_batch_and_distribute(
        submissions=[], b=8, cooldown_map=cd, current_window=100,
    )
    assert batch == [] and rewards == {}


def test_round_1_prompts_win_over_round_2():
    """Earlier drand rounds fill the batch first — that's the whole point."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("late_a", prompt_idx=1, drand_round=2),
        _sub("late_b", prompt_idx=2, drand_round=2),
        _sub("early_a", prompt_idx=3, drand_round=1),
        _sub("early_b", prompt_idx=4, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    # All 4 fit (b=8, only 4 distinct (round, prompt)). Each is its own slot.
    assert len(batch) == 4
    assert all(abs(rewards[hk] - 1/8) < 1e-9 for hk in rewards)
    # Burned pool = 4 unfilled slots × 1/8 = 0.5. Distributed = 0.5.
    assert abs(sum(rewards.values()) - 0.5) < 1e-9


def test_fills_8_slots_chronologically():
    """Two rounds, 10 distinct prompts. First 8 by round order fill the batch."""
    cd = CooldownMap(cooldown_windows=50)
    subs = []
    # Round 1: prompts 0-4 (5 slots)
    for p in range(5):
        subs.append(_sub(f"r1_p{p}", prompt_idx=p, drand_round=1))
    # Round 2: prompts 5-9 (5 slots, only first 3 should fit)
    for p in range(5, 10):
        subs.append(_sub(f"r2_p{p}", prompt_idx=p, drand_round=2))
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 8
    # All round 1 prompts must win.
    batched_prompts = {s.prompt_idx for s in batch}
    assert {0, 1, 2, 3, 4}.issubset(batched_prompts)
    # Pool sums to 8 × 1/8 = 1.0 (full pool used).
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_within_slot_split_K_miners():
    """K miners on the same (round, prompt) split that slot's share."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("alice", prompt_idx=42, drand_round=1),
        _sub("bob", prompt_idx=42, drand_round=1),
        _sub("carol", prompt_idx=42, drand_round=1),
        _sub("dave", prompt_idx=7, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    # 2 distinct (round, prompt) slots, each worth 1/8.
    assert len(batch) == 2
    # Prompt 42 slot: 3 miners split 1/8 → 1/24 each.
    assert abs(rewards["alice"] - 1/24) < 1e-9
    assert abs(rewards["bob"] - 1/24) < 1e-9
    assert abs(rewards["carol"] - 1/24) < 1e-9
    # Prompt 7 slot: 1 miner takes 1/8.
    assert abs(rewards["dave"] - 1/8) < 1e-9


def test_same_prompt_different_rounds_first_wins():
    """When the same prompt appears in multiple rounds, only the earliest
    round's slot is claimed; later rounds' submissions earn nothing."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [
        _sub("early", prompt_idx=42, drand_round=1),
        _sub("late", prompt_idx=42, drand_round=2),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 1
    assert batch[0].hotkey == "early"
    assert "late" not in rewards
    assert abs(rewards["early"] - 1/8) < 1e-9


def test_cooldown_excludes_prompt():
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=100)
    subs = [
        _sub("a", prompt_idx=42, drand_round=1),  # cooldown'd
        _sub("b", prompt_idx=7, drand_round=1),
    ]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=110, pool=1.0,
    )
    assert [s.prompt_idx for s in batch] == [7]
    assert "a" not in rewards
    assert abs(rewards["b"] - 1/8) < 1e-9


def test_cooldown_not_mutated_by_select():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub("hk", prompt_idx=42, drand_round=1)]
    select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100,
    )
    assert cd.is_in_cooldown(42, 100) is False


def test_unfilled_slots_burn_share():
    """If only 5 slots fill, pool * 5/8 is distributed, 3/8 is burned."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i, drand_round=1) for i in range(5)]
    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 5
    total = sum(rewards.values())
    assert abs(total - 5/8) < 1e-9


def test_sybil_neutral_on_same_round_same_prompt():
    """N sybils on the same (round, prompt) earn the same total as 1 hotkey."""
    cd = CooldownMap(cooldown_windows=50)

    # Case A: lone attacker.
    lone = [
        _sub("attacker", prompt_idx=42, drand_round=1),
        _sub("honest", prompt_idx=7, drand_round=1),
    ]
    _, rewards_a = select_batch_and_distribute(
        submissions=lone, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    # Case B: 5 sybils on the same prompt-round.
    sybils = [
        _sub(f"sybil{i}", prompt_idx=42, drand_round=1) for i in range(5)
    ]
    sybils.append(_sub("honest", prompt_idx=7, drand_round=1))
    _, rewards_b = select_batch_and_distribute(
        submissions=sybils, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    attacker_a = rewards_a["attacker"]
    attacker_b_total = sum(v for k, v in rewards_b.items() if k.startswith("sybil"))
    assert abs(attacker_a - attacker_b_total) < 1e-9
    # The honest miner on a different prompt is unaffected.
    assert abs(rewards_a["honest"] - rewards_b["honest"]) < 1e-9


def test_canonical_representative_deterministic_across_input_order():
    """Validators with different network-arrival orders pick the same
    representative submission for each (round, prompt) slot."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=1, drand_round=1) for i in range(5)]

    batch_a, _ = select_batch_and_distribute(
        submissions=list(subs), b=8, cooldown_map=cd, current_window=100,
    )
    batch_b, _ = select_batch_and_distribute(
        submissions=list(reversed(subs)), b=8, cooldown_map=cd, current_window=100,
    )
    assert [s.hotkey for s in batch_a] == [s.hotkey for s in batch_b]


def test_pool_scaling():
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"hk{i}", prompt_idx=i, drand_round=1) for i in range(4)]

    _, rewards_1 = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    _, rewards_100 = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=100.0,
    )
    for hk in rewards_1:
        assert abs(rewards_100[hk] - rewards_1[hk] * 100.0) < 1e-7


def _merge_multi_env(pool_per_env, fills):
    """Mirror service._train_and_publish: distribute each env with its own
    pool and sum per-hotkey into ``combined_rewards`` across envs."""
    cd = CooldownMap(cooldown_windows=50)
    combined: dict[str, float] = {}
    for env_idx, n_fill in enumerate(fills):
        subs = [
            _sub(f"e{env_idx}_hk{i}", prompt_idx=i, drand_round=1)
            for i in range(n_fill)
        ]
        _, rewards = select_batch_and_distribute(
            submissions=subs, b=8, cooldown_map=cd, current_window=100,
            pool=pool_per_env,
        )
        for hk, r in rewards.items():
            combined[hk] = combined.get(hk, 0.0) + r
    return combined


def test_multi_env_combined_emission_stays_within_one_pool():
    """With N envs at pool=1/N each, the merged emission across envs stays
    <= 1.0 — so the weight setter's burn = max(0, 1 - total) is not defeated.
    Regression: pool=1.0 per env summed two full pools to ~2.0, zeroing burn.
    """
    combined = _merge_multi_env(pool_per_env=1.0 / 2, fills=(8, 8))
    total = sum(combined.values())
    assert total <= 1.0 + 1e-9
    assert abs(total - 1.0) < 1e-9  # both envs full -> exactly one pool, burn 0


def test_multi_env_partial_fill_leaves_positive_burn():
    """An under-filled env must leave room to burn: total < 1.0 so
    burn = 1 - total > 0. This is the whole point of the per-env split."""
    # env A full (8/8) -> 0.5 ; env B half (4/8) -> 0.5 * 4/8 = 0.25
    combined = _merge_multi_env(pool_per_env=1.0 / 2, fills=(8, 4))
    total = sum(combined.values())
    assert abs(total - 0.75) < 1e-9
    burn = max(0.0, 1.0 - total)
    assert burn > 0.0
    assert abs(burn - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# Boundary-round fair split
# ---------------------------------------------------------------------------

def test_boundary_round_fair_split_distinct_prompts():
    """One slot left at the boundary, 10 miners on 10 different prompts in
    that round. Old algorithm gave the slot to ONE prompt by canonical hash;
    the other 9 miners earned zero despite arriving in the same drand bucket.
    New behaviour: per_prompt = 1 × slot_share / 10 = slot_share/10, each
    miner gets slot_share/10 (K=1). Total paid in boundary round = slot_share.
    """
    cd = CooldownMap(cooldown_windows=50)
    subs = []
    # Round 1: 7 prompts, all win full slots → 7 slots consumed, 1 left.
    for p in range(7):
        subs.append(_sub(f"r1_p{p}", prompt_idx=p, drand_round=1))
    # Round 2 is the boundary: 10 miners on 10 distinct prompts.
    for p in range(10, 20):
        subs.append(_sub(f"r2_p{p}", prompt_idx=p, drand_round=2))

    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    slot_share = 1.0 / 8
    # Each round-1 miner: full slot share.
    for p in range(7):
        assert abs(rewards[f"r1_p{p}"] - slot_share) < 1e-9, (
            f"r1_p{p} should keep its full slot"
        )
    # Each round-2 miner: per_prompt × (1/K=1) = 1 × slot_share / 10.
    expected_per_r2 = 1 * slot_share / 10
    for p in range(10, 20):
        assert abs(rewards[f"r2_p{p}"] - expected_per_r2) < 1e-9, (
            f"r2_p{p} should earn fair share at boundary, got {rewards.get(f'r2_p{p}')}"
        )
    # Conservation: total reward = 7 × slot_share + 1 × slot_share = pool.
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_boundary_round_all_miners_same_prompt():
    """10 miners all on the same prompt in the boundary round. The K-way
    split keeps total per-prompt payout = per_prompt_boundary. With 1 slot
    left and 1 distinct prompt, per_prompt = slot_share, each miner gets
    slot_share/10 (sybil-neutral)."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(7)]
    # 10 miners on the same prompt 99 in round 2 — boundary round.
    subs.extend(
        _sub(f"sybil{i}", prompt_idx=99, drand_round=2) for i in range(10)
    )

    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    slot_share = 1.0 / 8
    # Each round-1 miner: full slot share.
    for p in range(7):
        assert abs(rewards[f"r1_p{p}"] - slot_share) < 1e-9
    # 10 sybils on one prompt: per_prompt = slot_share, K=10 split.
    expected_per_sybil = slot_share / 10
    for i in range(10):
        assert abs(rewards[f"sybil{i}"] - expected_per_sybil) < 1e-9
    # Total = pool (no burn at boundary if anything fits).
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_boundary_round_mixed_K():
    """Mix of prompts at boundary: some single-miner, some multi-miner.
    Each prompt earns per_prompt = remaining × slot_share / N_prompts;
    within a prompt, K-way split applies. Total still equals slot_share ×
    remaining."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(6)]
    # Boundary round: 4 prompts, with K = [1, 1, 3, 5] miners.
    subs.append(_sub("alice", prompt_idx=10, drand_round=2))
    subs.append(_sub("bob", prompt_idx=11, drand_round=2))
    for i in range(3):
        subs.append(_sub(f"trio{i}", prompt_idx=12, drand_round=2))
    for i in range(5):
        subs.append(_sub(f"quint{i}", prompt_idx=13, drand_round=2))

    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    slot_share = 1.0 / 8
    remaining = 8 - 6  # 2 slots left at boundary
    n_prompts = 4
    per_prompt = remaining * slot_share / n_prompts  # = 2/32 = 1/16

    assert abs(rewards["alice"] - per_prompt) < 1e-9  # K=1
    assert abs(rewards["bob"] - per_prompt) < 1e-9    # K=1
    for i in range(3):
        assert abs(rewards[f"trio{i}"] - per_prompt / 3) < 1e-9  # K=3
    for i in range(5):
        assert abs(rewards[f"quint{i}"] - per_prompt / 5) < 1e-9  # K=5

    # Sum per prompt independent of K (conservation):
    trio_total = sum(rewards[f"trio{i}"] for i in range(3))
    quint_total = sum(rewards[f"quint{i}"] for i in range(5))
    assert abs(trio_total - per_prompt) < 1e-9
    assert abs(quint_total - per_prompt) < 1e-9

    # Round 1 unchanged.
    for p in range(6):
        assert abs(rewards[f"r1_p{p}"] - slot_share) < 1e-9

    # Total: 6 slot_share + 4 × per_prompt = 6/8 + 4/16 = 6/8 + 2/8 = 8/8 = 1.
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_boundary_round_sybil_same_prompt_neutral():
    """Same-prompt sybil attack at the boundary round: K hotkeys on one
    prompt earn the same total as 1 hotkey alone on that prompt. Cancels
    the multi-prompt sybil-multiplier loophole at the boundary that the
    old algorithm would have created if naive per-miner splits were used.
    """
    cd = CooldownMap(cooldown_windows=50)

    # Round 1: 7 slots filled by r1_p0..r1_p6.
    base = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(7)]

    # Case A: lone honest miner takes the boundary prompt 99.
    case_a = base + [_sub("honest", prompt_idx=99, drand_round=2)]
    _, rewards_a = select_batch_and_distribute(
        submissions=case_a, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    # Case B: 5 sybils all on prompt 99 in the boundary round, no one else.
    case_b = base + [
        _sub(f"sybil{i}", prompt_idx=99, drand_round=2) for i in range(5)
    ]
    _, rewards_b = select_batch_and_distribute(
        submissions=case_b, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    honest_a = rewards_a["honest"]
    sybil_b_total = sum(v for k, v in rewards_b.items() if k.startswith("sybil"))
    assert abs(honest_a - sybil_b_total) < 1e-9, (
        f"sybil on same boundary prompt should split, total {sybil_b_total} "
        f"vs lone honest {honest_a}"
    )


def test_boundary_round_later_rounds_get_zero():
    """After the boundary round consumes all remaining slot value, any
    submission in a later round earns nothing — the boundary distribution
    is final."""
    cd = CooldownMap(cooldown_windows=50)
    subs = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(6)]
    # Round 2 boundary: 4 prompts.
    subs.extend(
        _sub(f"r2_p{p}", prompt_idx=10 + p, drand_round=2) for p in range(4)
    )
    # Round 3: a forlorn miner.
    subs.append(_sub("toolate", prompt_idx=999, drand_round=3))

    _, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert "toolate" not in rewards
    # Conservation across the rounds that did fire:
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_boundary_round_training_batch_capped_at_b():
    """Even when more prompts are rewarded than slots remaining, the
    training batch never exceeds B. The boundary distribution is purely
    economic — the training step always runs at most B forward passes."""
    cd = CooldownMap(cooldown_windows=50)
    # Round 1 takes 5 slots, boundary round has 20 distinct prompts.
    subs = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(5)]
    subs.extend(
        _sub(f"r2_p{p}", prompt_idx=100 + p, drand_round=2) for p in range(20)
    )

    batch, rewards = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert len(batch) == 8  # exactly B, not 5 + 20
    # But 25 miners earn (5 round-1 + 20 round-2):
    assert len(rewards) == 25
    # Conservation:
    assert abs(sum(rewards.values()) - 1.0) < 1e-9


def test_boundary_round_picks_canonical_prompts_for_training():
    """Training-batch prompts at the boundary are the lowest canonical-hash
    ones — deterministic across validators. Reward distribution is
    independent of which subset gets picked for training."""
    import hashlib
    cd = CooldownMap(cooldown_windows=50)

    candidate_prompts = [100, 101, 102, 103, 104, 105]
    # Compute canonical hash order of these prompts:
    expected_order = sorted(
        candidate_prompts,
        key=lambda p: hashlib.sha256(p.to_bytes(8, "big", signed=False)).digest(),
    )

    # Round 1 takes 6 slots, leaving 2 for the boundary with 6 prompts.
    subs = [_sub(f"r1_p{p}", prompt_idx=p, drand_round=1) for p in range(6)]
    for p in candidate_prompts:
        subs.append(_sub(f"r2_p{p}", prompt_idx=p, drand_round=2))

    batch, _ = select_batch_and_distribute(
        submissions=subs, b=8, cooldown_map=cd, current_window=100, pool=1.0,
    )

    trained_boundary = [s.prompt_idx for s in batch if s.drand_round == 2]
    # First 2 by canonical hash order.
    assert trained_boundary == expected_order[:2]


def test_legacy_selector_uses_drand_round():
    cd = CooldownMap(cooldown_windows=50)
    a = _sub("a", prompt_idx=1, drand_round=101)
    b = _sub("b", prompt_idx=2, drand_round=100)
    batch, _ = select_batch_and_distribute(
        submissions=[a, b], b=1, cooldown_map=cd, current_window=100, pool=1.0,
    )
    assert [s.hotkey for s in batch] == ["b"]   # earlier round wins the slot
