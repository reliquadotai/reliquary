"""pi_old from the seal-time verify pass.

The GRAIL verify pass already computes, on the frozen verify_model, the raw
(T=1) log-softmax of every completion token. Reusing those values as pi_old
removes the train-time behavior forward — and, with KL_BETA=0, the reference
forward too, leaving train_step with only the policy forward+backward.

Selection order is validator > behavior forward > miner-claimed; the
miner-claimed values are never promoted by this feature.
"""
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

import reliquary.validator.training as T
from reliquary.validator.training import (
    _microbatch_grad,
    _rollout_loss,
    _selected_logprobs_for_tokens,
    _validator_completion_logprobs,
)


class _CountingLM(torch.nn.Module):
    """Tiny deterministic LM that counts its forwards."""

    def __init__(self, seed=0, vocab=16, dim=8):
        super().__init__()
        torch.manual_seed(seed)
        base = torch.nn.Module()
        base.emb = torch.nn.Embedding(vocab, dim)
        outer = self

        class _Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = base.emb

            def forward(self, input_ids, use_cache=False, attention_mask=None):
                outer.calls += 1
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

        self.model = _Inner()
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.calls = 0

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        self.calls += 1
        h = self.model.emb(input_ids)
        return SimpleNamespace(logits=self.lm_head(h))


@dataclass
class _R:
    commit: dict
    reward: float = 1.0


def _mk_rollout(tokens, prompt_length, vold=None):
    n_c = len(tokens) - prompt_length
    r = _R(commit={
        "tokens": list(tokens),
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": n_c,
            "token_logprobs": [-1.0] * n_c,   # miner-claimed, deliberately wrong
        },
    })
    if vold is not None:
        r._validated_completion_logprobs = vold
    return r


def _true_verify_logprobs(verify, tokens, prompt_length):
    # Same autocast as the train-time behavior forward, so the equivalence
    # test compares like for like; residual differences are bf16 kernel noise.
    t = torch.tensor([tokens])
    with torch.no_grad():
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            lp = _selected_logprobs_for_tokens(verify, t, t[0, 1:])
    return [float(x) for x in lp[prompt_length - 1:]]


# ---------------------------------------------------------------------------
# helper validation
# ---------------------------------------------------------------------------

def test_helper_accepts_only_complete_finite_lists(monkeypatch):
    r = _mk_rollout([1, 2, 3, 4, 5], 2, vold=[-0.1, -0.2, -0.3])
    assert _validator_completion_logprobs(r, 3) == [-0.1, -0.2, -0.3]
    assert _validator_completion_logprobs(r, 4) is None          # wrong length
    r2 = _mk_rollout([1, 2, 3, 4, 5], 2, vold=[-0.1, float("nan"), -0.3])
    assert _validator_completion_logprobs(r2, 3) is None         # non-finite
    r3 = _mk_rollout([1, 2, 3, 4, 5], 2)                          # absent
    assert _validator_completion_logprobs(r3, 3) is None


def test_helper_respects_kill_switch(monkeypatch):
    import reliquary.constants as C
    monkeypatch.setattr(C, "PI_OLD_FROM_VERIFY_LOGPROBS", False)
    r = _mk_rollout([1, 2, 3, 4, 5], 2, vold=[-0.1, -0.2, -0.3])
    assert _validator_completion_logprobs(r, 3) is None


# ---------------------------------------------------------------------------
# micro-batched path
# ---------------------------------------------------------------------------

def _mb_items(rollouts, prompt_length):
    items = []
    for r in rollouts:
        tokens = r.commit["tokens"]
        n_c = len(tokens) - prompt_length
        vold = _validator_completion_logprobs(r, n_c)
        items.append((tokens, prompt_length, [-1.0] * n_c, 0.5, 1.0,
                      [True] * n_c, vold))
    return items


def test_microbatch_skips_all_nograd_forwards_when_validator_present(monkeypatch):
    monkeypatch.setattr(T, "KL_BETA", 0.0)
    policy, verify = _CountingLM(seed=1), _CountingLM(seed=2)
    tokens = [1, 2, 3, 4, 5, 6]
    rollouts = [
        _mk_rollout(tokens, 2, vold=_true_verify_logprobs(verify, tokens, 2))
        for _ in range(3)
    ]
    verify.calls = 0
    _microbatch_grad(policy, verify, _mb_items(rollouts, 2),
                     torch.device("cpu"), atomic=False, behavior_model=verify)
    assert verify.calls == 0      # neither ref nor behavior forward ran
    assert policy.calls >= 1


def test_microbatch_validator_old_matches_behavior_forward(monkeypatch):
    """Same pi_old values either way => same gradients (fp32 CPU: exact-ish)."""
    monkeypatch.setattr(T, "KL_BETA", 0.0)
    verify = _CountingLM(seed=2)
    tokens = [1, 2, 3, 4, 5, 6]
    vold = _true_verify_logprobs(verify, tokens, 2)

    grads = []
    for use_validator in (False, True):
        policy = _CountingLM(seed=1)
        rollouts = [_mk_rollout(tokens, 2, vold=vold if use_validator else None)]
        _microbatch_grad(policy, verify, _mb_items(rollouts, 2),
                         torch.device("cpu"), atomic=False,
                         behavior_model=verify)
        grads.append(policy.lm_head.weight.grad.detach().clone())
    # bf16 quantum on pi_old (~0.004 logprob) bounds the gradient delta; the
    # ratio perturbation is exp(±0.004) ≈ ±0.4%, 50× inside the clip band.
    torch.testing.assert_close(grads[0], grads[1], rtol=0.0, atol=2e-2)


def test_microbatch_falls_back_to_behavior_when_one_rollout_lacks_values(monkeypatch):
    monkeypatch.setattr(T, "KL_BETA", 0.0)
    policy, verify = _CountingLM(seed=1), _CountingLM(seed=2)
    tokens = [1, 2, 3, 4, 5, 6]
    rollouts = [
        _mk_rollout(tokens, 2, vold=_true_verify_logprobs(verify, tokens, 2)),
        _mk_rollout(tokens, 2),                     # no validator values
    ]
    verify.calls = 0
    _microbatch_grad(policy, verify, _mb_items(rollouts, 2),
                     torch.device("cpu"), atomic=False, behavior_model=verify)
    assert verify.calls >= 1      # behavior forward ran for the batch


def test_microbatch_ref_forward_runs_when_kl_beta_positive(monkeypatch):
    monkeypatch.setattr(T, "KL_BETA", 0.04)
    policy, verify = _CountingLM(seed=1), _CountingLM(seed=2)
    tokens = [1, 2, 3, 4, 5, 6]
    rollouts = [
        _mk_rollout(tokens, 2, vold=_true_verify_logprobs(verify, tokens, 2))
    ]
    verify.calls = 0
    _microbatch_grad(policy, verify, _mb_items(rollouts, 2),
                     torch.device("cpu"), atomic=False, behavior_model=verify)
    assert verify.calls >= 1      # KL needs the ref forward


# ---------------------------------------------------------------------------
# per-rollout path
# ---------------------------------------------------------------------------

def test_rollout_loss_uses_validator_values_and_skips_forwards(monkeypatch):
    monkeypatch.setattr(T, "KL_BETA", 0.0)
    policy, verify = _CountingLM(seed=1), _CountingLM(seed=2)
    tokens = [1, 2, 3, 4, 5, 6]
    r = _mk_rollout(tokens, 2, vold=_true_verify_logprobs(verify, tokens, 2))
    verify.calls = 0
    ppo, kl, n = _rollout_loss(policy, verify, r, 0.5, torch.device("cpu"),
                               behavior_model=verify)
    assert verify.calls == 0
    assert n == 4
    assert float(kl) == 0.0       # KL term skipped entirely


# ---------------------------------------------------------------------------
# producer side: batcher derivation from the proof's existing chosen probs
# ---------------------------------------------------------------------------

def test_batcher_derives_logprobs_at_unit_temperature(monkeypatch):
    import reliquary.constants as C
    from reliquary.validator.batcher import _verify_logprobs_for_training

    monkeypatch.setattr(C, "T_PROTO", 1.0)
    proof = SimpleNamespace(completion_chosen_probs=[0.5, 0.25, 1.0])
    got = _verify_logprobs_for_training(proof, 3)
    assert got == pytest.approx([math.log(0.5), math.log(0.25), 0.0])


def test_batcher_refuses_non_unit_temperature(monkeypatch):
    """log(chosen_prob) is pi_old ONLY when warp() is the identity: at any
    other T_PROTO the probs are temperature-scaled and reusing them would put
    the ratio in the wrong space — the exact v3 bug the v4 sampling fixed."""
    import reliquary.constants as C
    from reliquary.validator.batcher import _verify_logprobs_for_training

    monkeypatch.setattr(C, "T_PROTO", 0.6)
    proof = SimpleNamespace(completion_chosen_probs=[0.5, 0.25, 1.0])
    assert _verify_logprobs_for_training(proof, 3) is None


def test_batcher_refuses_partial_or_degenerate_coverage(monkeypatch):
    import reliquary.constants as C
    from reliquary.validator.batcher import _verify_logprobs_for_training

    monkeypatch.setattr(C, "T_PROTO", 1.0)
    proof = SimpleNamespace(completion_chosen_probs=[0.5, 0.25, 1.0])
    assert _verify_logprobs_for_training(proof, 4) is None    # partial coverage
    assert _verify_logprobs_for_training(proof, 0) is None
    zero = SimpleNamespace(completion_chosen_probs=[0.5, 0.0, 1.0])
    assert _verify_logprobs_for_training(zero, 3) is None     # log(0) undefined
    legacy = SimpleNamespace()                                # no field at all
    assert _verify_logprobs_for_training(legacy, 3) is None
