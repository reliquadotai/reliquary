"""Micro-batched train_step: token-budget packing + batched forward/backward,
numerically ~equivalent to the per-rollout path. Plus the atomic backward used
by the OOM split-retry must match the plain path exactly.
"""
import math

import pytest

try:
    import torch
    from transformers import AutoModelForCausalLM
    _check = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    del _check
    _HAS_TINY_GPT2 = True
except Exception:
    _HAS_TINY_GPT2 = False

import torch  # noqa: E402
from dataclasses import dataclass, field
from types import SimpleNamespace

from reliquary.constants import KL_BETA
from reliquary.validator.training import (
    _pack_by_token_budget, _accumulate_grouped_grads, _build_microbatch_items,
    _rollout_loss,
    _bft_training_metrics, _compute_advantages, _plan_from_batches,
    train_step, reset_training_state,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeRollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _FakeGroup:
    rollouts: list
    prompt_idx: int = 0


def _build_rollout(tokens, reward, prompt_length):
    n = len(tokens) - prompt_length
    return _FakeRollout(tokens=tokens, reward=reward, commit={
        "tokens": tokens,
        "rollout": {"prompt_length": prompt_length, "token_logprobs": [-1.0] * n},
    })


class _Base(torch.nn.Module):
    """Embedding-only base (no cross-token attention) — a padded batch forward
    gives per-row hidden states identical to single-sequence forwards."""

    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class _QwenLike(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.model = _Base(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, *a, **k):  # pragma: no cover - must not be called
        raise AssertionError("full-logits forward should not be used")


def _make_plan(groups):
    """Single-env plan: scale = 1/n_total per group, i.e. the global token-level
    normalization (one present batch reduces to the pre-fix DAPO denominator)."""
    surviving, n_total = [], 0
    for grp in groups:
        advs = _compute_advantages([r.reward for r in grp.rollouts])
        if all(a == 0.0 for a in advs):
            continue
        surviving.append((grp, advs))
        for r in grp.rollouts:
            n_total += len(r.commit["rollout"]["token_logprobs"])
    scale = 1.0 / n_total if n_total else 0.0
    plan = [(grp, advs, scale) for grp, advs in surviving]
    return plan, n_total


def _reference_grads(model, ref, plan, n_total, device):
    from reliquary.validator.training import _is_masked_from_loss

    for p in model.parameters():
        p.grad = None
    for grp, advs, _scale in plan:
        for r, adv in zip(grp.rollouts, advs):
            if _is_masked_from_loss(r):
                continue        # mirrors production: no PPO/KL, no forward
            ppo, kl, n = _rollout_loss(model, ref, r, adv, device=device)
            ((ppo + KL_BETA * kl) * n / n_total).backward()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def _grads_after(model, fn):
    for p in model.parameters():
        p.grad = None
    fn()
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def _rel_l2(a, b):
    num = sum(((x - y) ** 2).sum() for x, y in zip(a, b))
    den = sum((x ** 2).sum() for x in a)
    return (num.sqrt() / den.sqrt()).item()


def _frozen(model):
    import copy
    f = copy.deepcopy(model).eval()
    for p in f.parameters():
        p.requires_grad = False
    return f


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def test_pack_groups_short_and_isolates_long():
    bins = _pack_by_token_budget([10, 10, 10, 10, 100], budget=40)
    assert sorted(i for b in bins for i in b) == [0, 1, 2, 3, 4]
    assert [b for b in bins if 4 in b][0] == [4]  # long one alone


def test_pack_respects_budget():
    lengths = [7, 5, 5, 3, 9, 1]
    bins = _pack_by_token_budget(lengths, budget=12)
    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
    for b in bins:
        if len(b) > 1:
            assert len(b) * max(lengths[i] for i in b) <= 12


def test_microbatch_normalizes_protocol_full_sequence_logprobs():
    rollout = _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2)
    rollout.commit["rollout"]["token_logprobs"] = [-1.0] * 6
    plan = [(_FakeGroup([rollout]), [1.0], 1.0)]

    items = _build_microbatch_items(plan)

    assert len(items) == 1
    assert len(items[0][2]) == 4
    assert len(items[0][5]) == 4


# ---------------------------------------------------------------------------
# DAPO overlong filtering: a masked (advantage-0) rollout is skipped from the
# forward AND the N_e denominator — this is what makes the forced 16k rollouts
# memory-cheap instead of OOMing.
# ---------------------------------------------------------------------------

def test_masked_rollout_skipped_from_microbatch_forward(monkeypatch):
    monkeypatch.setattr("reliquary.constants.MASK_BUDGET_ENDED_FROM_LOSS", True)
    real = _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2)
    masked = _build_rollout([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.0, 2)  # the "16k" analogue
    masked.commit["rollout"]["forced"] = True
    plan = [(_FakeGroup([real, masked]), [1.0, 0.0], 1.0)]
    items = _build_microbatch_items(plan)
    assert len(items) == 1                       # forced rollout not forwarded
    assert tuple(items[0][0]) == tuple(real.tokens)


def test_natural_zero_advantage_rollout_is_still_forwarded():
    """REGRESSION: a non-forced rollout whose advantage is exactly 0 (its reward
    equals the group mean) must still be forwarded — it owes its KL term."""
    a = _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2)
    b = _build_rollout([1, 2, 3, 7, 8, 9], 0.5, 2)      # reward == group mean
    c = _build_rollout([1, 2, 3, 4, 5, 7], 0.0, 2)
    advs = _compute_advantages([1.0, 0.5, 0.0])
    assert advs[1] == 0.0
    items = _build_microbatch_items([(_FakeGroup([a, b, c]), advs, 1.0)])
    assert len(items) == 3                             # none dropped


def test_masked_forced_rollout_masked_and_skipped_end_to_end(monkeypatch):
    monkeypatch.setattr("reliquary.constants.SHAPE_PENALTY", 0.0)
    monkeypatch.setattr("reliquary.constants.MASK_BUDGET_ENDED_FROM_LOSS", True)
    correct = _build_rollout([1, 2, 3, 4, 5], 1.0, 2)
    wrong = _build_rollout([1, 2, 3, 6, 7], 0.0, 2)             # wrong, terminated
    forced = _build_rollout([1, 2, 3, 8, 9, 10, 11], 0.0, 2)    # wrong, forced (long/masked)
    forced.commit["rollout"]["forced"] = True
    plan, _ = _plan_from_batches([[_FakeGroup([correct, wrong, forced])]])
    advs = plan[0][1]
    assert advs[2] == 0.0                         # forced rollout masked to 0
    assert advs[0] != 0.0 and advs[1] != 0.0
    items = _build_microbatch_items(plan)
    assert len(items) == 2                        # forced (16k analogue) skipped
    assert all(tuple(it[0]) != tuple(forced.tokens) for it in items)


def test_masked_rollout_excluded_from_Ne(monkeypatch):
    # N_e must drop the masked rollout's tokens; same group with vs without the
    # would-be-masked forced rollout yields the same per-token scale.
    monkeypatch.setattr("reliquary.constants.SHAPE_PENALTY", 0.0)
    monkeypatch.setattr("reliquary.constants.MASK_BUDGET_ENDED_FROM_LOSS", True)

    def grp(with_forced):
        rs = [_build_rollout([1, 2, 3, 4, 5], 1.0, 2),
              _build_rollout([1, 2, 3, 6, 7], 0.0, 2)]
        if with_forced:
            f = _build_rollout([1, 2, 3, 8, 9, 10, 11], 0.0, 2)
            f.commit["rollout"]["forced"] = True
            rs.append(f)
        return _FakeGroup(rs)

    plan_with, _ = _plan_from_batches([[grp(True)]])
    plan_without, _ = _plan_from_batches([[grp(False)]])
    assert plan_with[0][2] == pytest.approx(plan_without[0][2])


# ---------------------------------------------------------------------------
# Equivalence (the Embedding base makes batched == per-rollout up to bf16 head)
# ---------------------------------------------------------------------------

def _sample_groups():
    return [
        _FakeGroup([
            _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2),
            _build_rollout([1, 2, 3, 7, 8], 0.0, 2),
            _build_rollout([1, 2, 9, 10, 11, 12, 13], 1.0, 2),
            _build_rollout([1, 2, 3], 0.0, 2),
        ]),
        _FakeGroup([
            _build_rollout([4, 5, 6, 7, 8, 9, 10], 1.0, 3),
            _build_rollout([4, 5, 6, 1], 0.0, 3),
        ]),
    ]


def test_batched_grads_match_per_rollout_qwenlike():
    import copy
    torch.manual_seed(0)
    model = _QwenLike()
    plan, n_total = _make_plan(_sample_groups())
    device = next(model.parameters()).device
    assert n_total > 0 and len(plan) == 2

    m_ref = copy.deepcopy(model)
    ref_grads = _reference_grads(m_ref, _frozen(m_ref), plan, n_total, device)
    m_bat = copy.deepcopy(model)
    bat_grads = _grads_after(m_bat, lambda: _accumulate_grouped_grads(
        m_bat, _frozen(m_bat), plan, device, budget=64, atomic=False))
    assert _rel_l2(bat_grads, ref_grads) < 5e-2


def test_behavior_model_overrides_untrusted_miner_logprobs():
    """Validator-recomputed pi_old makes the objective independent of the
    log-probability vector supplied in the miner commit."""
    torch.manual_seed(7)
    model = _QwenLike()
    ref = _frozen(model)
    behavior = _frozen(model)
    device = next(model.parameters()).device

    claimed_near = _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2)
    claimed_far = _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2)
    claimed_near.commit["rollout"]["token_logprobs"] = [-1.0] * 4
    claimed_far.commit["rollout"]["token_logprobs"] = [-8.0] * 4

    trusted_near, _, _ = _rollout_loss(
        model, ref, claimed_near, 1.0, device,
        behavior_model=behavior,
    )
    trusted_far, _, _ = _rollout_loss(
        model, ref, claimed_far, 1.0, device,
        behavior_model=behavior,
    )
    untrusted_near, _, _ = _rollout_loss(
        model, ref, claimed_near, 1.0, device,
    )
    untrusted_far, _, _ = _rollout_loss(
        model, ref, claimed_far, 1.0, device,
    )

    torch.testing.assert_close(trusted_near, trusted_far)
    assert not torch.isclose(untrusted_near, untrusted_far)


def test_microbatch_behavior_recompute_is_claim_invariant():
    import copy

    torch.manual_seed(8)
    base = _QwenLike()

    groups_a = _sample_groups()
    groups_b = copy.deepcopy(groups_a)
    for group in groups_b:
        for rollout in group.rollouts:
            n = len(rollout.commit["rollout"]["token_logprobs"])
            rollout.commit["rollout"]["token_logprobs"] = [-9.0] * n
    plan_a, _ = _make_plan(groups_a)
    plan_b, _ = _make_plan(groups_b)

    model_a = copy.deepcopy(base)
    grads_a = _grads_after(model_a, lambda: _accumulate_grouped_grads(
        model_a,
        _frozen(base),
        plan_a,
        next(model_a.parameters()).device,
        budget=64,
        atomic=False,
        behavior_model=_frozen(base),
    ))
    model_b = copy.deepcopy(base)
    grads_b = _grads_after(model_b, lambda: _accumulate_grouped_grads(
        model_b,
        _frozen(base),
        plan_b,
        next(model_b.parameters()).device,
        budget=64,
        atomic=False,
        behavior_model=_frozen(base),
    ))

    assert _rel_l2(grads_a, grads_b) < 1e-7


def test_batched_grads_mask_bft_force_span_like_per_rollout(monkeypatch):
    # Flag off (default): the forced rollout still trains, with its force span
    # masked at token level — the batched path must match per-rollout.
    monkeypatch.setattr("reliquary.constants.MASK_BUDGET_ENDED_FROM_LOSS", False)
    import copy

    torch.manual_seed(3)
    base = _QwenLike()
    device = next(base.parameters()).device
    forced = _build_rollout([1, 2, 3, 4, 5, 6, 7, 8], 1.0, 3)
    forced.commit["rollout"]["forced"] = True
    # Absolute span [5, 7) -> completion-relative positions 2 and 3.
    forced.commit["rollout"]["force_span"] = [5, 7]
    forced._validated_force_span = (5, 7)
    # A forced rollout is now masked out of the loss entirely, so the surviving
    # rollouts must carry the signal — otherwise both paths trivially produce
    # zero gradients and the comparison is vacuous.
    winner = _build_rollout([1, 2, 3, 12, 13, 14], 1.0, 3)
    loser = _build_rollout([1, 2, 3, 9, 10, 11], 0.0, 3)
    group = _FakeGroup([forced, winner, loser])
    plan, skipped = _plan_from_batches([[group]])
    assert skipped == 0
    assert len(plan) == 1
    assert plan[0][1][0] != 0.0          # flag off: the forced rollout still trains
    # One present env: scale is 1 / surviving trainable completion tokens.
    n_total = round(1.0 / plan[0][2])

    m_ref = copy.deepcopy(base)
    ref_grads = _reference_grads(m_ref, _frozen(m_ref), plan, n_total, device)
    m_bat = copy.deepcopy(base)
    bat_grads = _grads_after(m_bat, lambda: _accumulate_grouped_grads(
        m_bat, _frozen(m_bat), plan, device, budget=64, atomic=False))
    assert sum(float((g ** 2).sum()) for g in ref_grads) > 0.0   # non-vacuous
    assert _rel_l2(bat_grads, ref_grads) < 5e-2


def test_bft_training_metrics_measure_masked_and_weighted_exposure(
    monkeypatch,
):
    monkeypatch.setattr("reliquary.constants.SHAPE_PENALTY", 0.0)
    forced = _build_rollout([1, 2, 3, 4, 5, 6, 7, 8], 1.0, 3)
    forced._validated_force_span = (5, 7)
    forced._validated_termination_path = "forced_phase2_eos"
    natural = _build_rollout([1, 2, 3, 9, 10, 11], 0.0, 3)
    natural._validated_termination_path = "phase1_eos"
    plan, skipped = _plan_from_batches([[_FakeGroup([forced, natural])]])

    metrics = _bft_training_metrics(plan)

    assert skipped == 0
    assert metrics["bft/plan_rollouts"] == 2
    assert metrics["bft/forced_rollouts"] == 1
    assert metrics["bft/injected_tokens_masked"] == 2
    assert metrics["bft/trainable_completion_tokens"] == 6
    assert metrics["bft/forced_trainable_token_ratio"] == 0.5
    assert metrics["bft/forced_abs_adv_weighted_token_ratio"] == pytest.approx(
        0.5
    )
    assert metrics["bft/path/forced_phase2_eos/rollouts"] == 1
    assert metrics[
        "bft/path/forced_phase2_eos/abs_adv_weighted_tokens"
    ] > 0


def test_atomic_matches_nonatomic_qwenlike():
    import copy
    torch.manual_seed(1)
    base = _QwenLike()
    plan, n_total = _make_plan(_sample_groups())
    device = next(base.parameters()).device

    m1 = copy.deepcopy(base)
    g_plain = _grads_after(m1, lambda: _accumulate_grouped_grads(
        m1, _frozen(m1), plan, device, budget=32, atomic=False))
    m2 = copy.deepcopy(base)
    g_atomic = _grads_after(m2, lambda: _accumulate_grouped_grads(
        m2, _frozen(m2), plan, device, budget=32, atomic=True))
    assert _rel_l2(g_atomic, g_plain) < 1e-5


def test_kl_tail_stats_capture_weighted_objective_and_outliers():
    import copy

    from reliquary.validator.training import (
        _kl_telemetry_metrics,
        _new_kl_stats,
    )

    torch.manual_seed(4)
    model = _QwenLike()
    plan, _ = _make_plan(_sample_groups())
    device = next(model.parameters()).device
    stats = _new_kl_stats()

    _accumulate_grouped_grads(
        model,
        _frozen(copy.deepcopy(model)),
        plan,
        device,
        budget=64,
        atomic=False,
        kl_stats=stats,
    )

    assert stats["token_count"] > 0
    assert stats["nonfinite_count"] == 0
    assert stats["max"] >= 0.0
    assert math.isfinite(stats["weighted_ppo"])
    assert math.isfinite(stats["weighted_kl"])
    assert stats["ppo_token_count"] == stats["token_count"]
    assert stats["ppo_ratio_nonfinite_count"] == 0
    outside_count = (
        stats["ppo_ratio_below_clip_count"]
        + stats["ppo_ratio_above_clip_count"]
    )
    assert 0 <= outside_count <= stats["ppo_token_count"]
    metrics = _kl_telemetry_metrics(stats)
    assert metrics["train/ppo_ratio_outside_clip_ratio"] == pytest.approx(
        outside_count / stats["ppo_token_count"]
    )


def test_policy_health_stats_measure_recomputed_claim_error():
    import copy

    from reliquary.validator.training import (
        _kl_telemetry_metrics,
        _new_kl_stats,
    )

    torch.manual_seed(9)
    model = _QwenLike()
    groups = _sample_groups()
    for group in groups:
        for rollout in group.rollouts:
            n = len(rollout.commit["rollout"]["token_logprobs"])
            rollout.commit["rollout"]["token_logprobs"] = [-9.0] * n
    plan, _ = _make_plan(groups)
    stats = _new_kl_stats()

    _accumulate_grouped_grads(
        model,
        _frozen(copy.deepcopy(model)),
        plan,
        next(model.parameters()).device,
        budget=64,
        atomic=False,
        kl_stats=stats,
        behavior_model=_frozen(copy.deepcopy(model)),
    )
    metrics = _kl_telemetry_metrics(stats)

    assert metrics["train/pi_old_claim_token_count"] > 0
    assert metrics["train/pi_old_claim_abs_error_mean"] > 1.0
    assert metrics["train/pi_old_claim_abs_error_max"] > 1.0
    assert metrics["train/pi_old_claim_gt_1e_3_ratio"] == 1.0
    assert metrics["train/ppo_ratio_nonfinite_ratio"] == 0.0
    assert metrics["train/ppo_ratio_outside_clip_ratio"] == 0.0
    assert metrics["train/ppo_log_ratio_abs_gt_1_ratio"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end on a real (tiny) model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_TINY_GPT2, reason="tiny-gpt2 not available")
def test_train_step_microbatch_updates_params():
    import copy
    reset_training_state()
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    rollouts = [
        _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2),
        _build_rollout([1, 2, 3, 4, 5, 6], 0.0, 2),
        _build_rollout(list(range(1, 13)), 1.0, 2),
        _build_rollout(list(range(1, 13)), 0.0, 2),
    ]
    group = _FakeGroup(rollouts=rollouts)
    before = next(model.parameters()).detach().clone()
    train_step(model, [[group]], ref_model=ref)
    after = next(model.parameters()).detach().clone()
    assert (before - after).abs().max().item() > 0.0
