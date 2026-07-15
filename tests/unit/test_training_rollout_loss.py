"""_rollout_loss + train_step — uses sshleifer/tiny-gpt2 for CPU testing.

These tests require transformers but not a GPU. They verify the math
runs end-to-end on a real (tiny) model.
"""

import pytest

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Preload attempt to fail fast if no network / no cache
    _check = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    del _check
except Exception:
    pytest.skip("tiny-gpt2 not available", allow_module_level=True)

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

from reliquary.validator.training import (
    _rollout_loss, _compute_advantages, train_step, reset_training_state,
    _selected_logprobs, _selected_logprobs_for_tokens,
)


def _make_ref(model):
    import copy
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def test_selected_logprobs_matches_log_softmax_gather():
    """_selected_logprobs must equal log_softmax(x.float()).gather(idx) bit-for-bit
    (up to fp32 round-off) — the streaming implementation is a memory
    optimisation, not a math change.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    N, V = 200, 1024  # picks a non-multiple of chunk=64 to exercise the tail
    logits = torch.randn(N, V, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.randint(0, V, (N,))

    expected = F.log_softmax(logits.float(), dim=-1).gather(
        1, indices.unsqueeze(1)
    ).squeeze(1)
    got = _selected_logprobs(logits, indices)

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_selected_logprobs_backward_matches_reference():
    """Gradients through _selected_logprobs must match the reference
    log_softmax+gather path.
    """
    import torch.nn.functional as F
    torch.manual_seed(1)
    N, V = 130, 512
    logits_a = torch.randn(N, V, dtype=torch.float32, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    indices = torch.randint(0, V, (N,))

    F.log_softmax(logits_a, dim=-1).gather(
        1, indices.unsqueeze(1)
    ).squeeze(1).sum().backward()
    _selected_logprobs(logits_b, indices).sum().backward()

    torch.testing.assert_close(logits_a.grad, logits_b.grad, rtol=1e-5, atol=1e-5)


def test_qwen_like_selected_logprobs_uses_hidden_lm_head_path():
    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.called = False

        def forward(self, input_ids, use_cache=False):
            self.called = True
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    class _QwenLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Base()
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, *args, **kwargs):
            raise AssertionError("full logits forward should not be used")

    model = _QwenLike()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    next_tokens = tokens[0, 1:]

    logprobs = _selected_logprobs_for_tokens(model, tokens, next_tokens)
    assert model.model.called
    assert logprobs.shape == (4,)

    logprobs.sum().backward()
    assert model.model.emb.weight.grad is not None
    assert model.lm_head.weight.grad is not None


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Load sshleifer/tiny-gpt2 (≈ 500KB) for fast CPU tests."""
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    return model, tokenizer


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
    """Build a fake rollout with synthetic (but length-correct) old_logprobs."""
    n_completion = len(tokens) - prompt_length
    return _FakeRollout(
        tokens=tokens,
        reward=reward,
        commit={
            "tokens": tokens,
            "rollout": {
                "prompt_length": prompt_length,
                "token_logprobs": [-1.0] * n_completion,  # arbitrary baseline
            },
        },
    )


def test_rollout_loss_zero_advantage_gives_zero_ppo_loss(tiny_model_and_tokenizer):
    """With advantage=0, both surr1 and surr2 are 0 → ppo_loss = 0."""
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6],
        reward=0.5,
        prompt_length=2,
    )
    device = next(model.parameters()).device
    ppo_loss, kl, _ = _rollout_loss(model, ref, rollout, advantage=0.0, device=device)
    assert abs(ppo_loss.item()) < 1e-6
    # KL is advantage-independent; just check it's non-negative (by construction)
    assert kl.item() >= -1e-6


def test_rollout_loss_produces_finite_values(tiny_model_and_tokenizer):
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        reward=1.0,
        prompt_length=3,
    )
    device = next(model.parameters()).device
    ppo_loss, kl, _ = _rollout_loss(model, ref, rollout, advantage=1.0, device=device)
    assert torch.isfinite(ppo_loss)
    assert torch.isfinite(kl)


def test_rollout_loss_returns_completion_token_count(tiny_model_and_tokenizer):
    """The 3rd return value is the completion-token count (= len - prompt),
    used by train_step for DAPO token-level loss normalisation."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],  # 8 tokens
        reward=1.0,
        prompt_length=3,  # → 5 completion tokens
    )
    device = next(model.parameters()).device
    _ppo, _kl, n_tok = _rollout_loss(model, ref, rollout, advantage=1.0, device=device)
    assert n_tok == 5


def test_rollout_loss_accepts_protocol_full_sequence_logprobs(
    tiny_model_and_tokenizer,
):
    model, _ = tiny_model_and_tokenizer
    ref = _make_ref(model)
    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        reward=1.0,
        prompt_length=3,
    )
    rollout.commit["rollout"]["token_logprobs"] = [-1.0] * 8

    ppo, kl, n_tok = _rollout_loss(
        model, ref, rollout, advantage=1.0,
        device=next(model.parameters()).device,
    )

    assert n_tok == 5
    assert torch.isfinite(ppo)
    assert torch.isfinite(kl)


def test_rollout_loss_masks_force_span_from_loss(tiny_model_and_tokenizer):
    """BFT: a forced rollout's injected FORCE span is excluded from the loss —
    the trained-token count drops by the span length and the loss changes."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    ref = _make_ref(model)
    device = next(model.parameters()).device

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # prompt_length=3 → 5 completion tokens
    natural = _build_rollout(tokens=tokens, reward=1.0, prompt_length=3)
    forced = _build_rollout(tokens=tokens, reward=1.0, prompt_length=3)
    forced.commit["rollout"]["forced"] = True
    # absolute span [5,7) → completion-relative positions 2,3 (2 tokens)
    forced.commit["rollout"]["force_span"] = [5, 7]
    forced._validated_force_span = (5, 7)

    ppo_n, _kl_n, n_n = _rollout_loss(model, ref, natural, advantage=1.0, device=device)
    ppo_f, _kl_f, n_f = _rollout_loss(model, ref, forced, advantage=1.0, device=device)

    assert n_n == 5          # all completion tokens trained
    assert n_f == 3          # 2-token FORCE span masked out
    assert torch.isfinite(ppo_f)
    assert ppo_f.item() != ppo_n.item()   # masking the span changed the loss


def test_rollout_loss_ignores_unvalidated_wire_force_span(
    tiny_model_and_tokenizer,
):
    """A miner-declared span cannot suppress arbitrary GRPO loss tokens."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    ref = _make_ref(model)
    device = next(model.parameters()).device

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    natural = _build_rollout(tokens=tokens, reward=1.0, prompt_length=3)
    untrusted = _build_rollout(tokens=tokens, reward=1.0, prompt_length=3)
    untrusted.commit["rollout"]["forced"] = False
    untrusted.commit["rollout"]["force_span"] = [0, 10_000]

    ppo_n, _kl_n, n_n = _rollout_loss(
        model, ref, natural, advantage=1.0, device=device,
    )
    ppo_u, _kl_u, n_u = _rollout_loss(
        model, ref, untrusted, advantage=1.0, device=device,
    )

    assert n_n == n_u == 5
    torch.testing.assert_close(ppo_u, ppo_n)


def test_rollout_loss_uses_commit_tokens_as_source_of_truth(tiny_model_and_tokenizer):
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    ref = _make_ref(model)

    rollout = _build_rollout(
        tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        reward=1.0,
        prompt_length=3,
    )
    rollout.tokens = [1, 2]

    device = next(model.parameters()).device
    ppo_loss, kl, _ = _rollout_loss(model, ref, rollout, advantage=1.0, device=device)

    assert torch.isfinite(ppo_loss)
    assert torch.isfinite(kl)


def test_train_step_updates_optimizer(tiny_model_and_tokenizer):
    """train_step should take at least one optimizer step when there's signal."""
    reset_training_state()
    model, tokenizer = tiny_model_and_tokenizer
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], r, 2) for r in [1, 1, 0, 0]]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)

    # Take a snapshot of one parameter to verify it changed
    sample_param = next(model.parameters())
    before = sample_param.detach().clone()

    result = train_step(model, [[group]], ref_model=_make_ref(model))
    assert result is model

    # Parameter should have changed (tiny-gpt2 is tiny, but non-zero grad)
    after = next(model.parameters()).detach().clone()
    diff = (before - after).abs().max().item()
    assert diff > 0.0, "expected some parameter change after optimizer step"


def test_train_step_skips_optimizer_on_nonfinite_grad(tiny_model_and_tokenizer, monkeypatch):
    """Finite-guard: a non-finite grad_norm skips the optimizer step so a NaN
    never reaches the weights."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], r, 2) for r in [1, 1, 0, 0]]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)

    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_",
        lambda *a, **k: torch.tensor(float("nan")),
    )

    before = next(model.parameters()).detach().clone()
    result = train_step(model, [[group]], ref_model=_make_ref(model))
    assert result is model
    after = next(model.parameters()).detach().clone()
    assert torch.equal(before, after), "weights must not change on non-finite grad"


def test_train_step_token_level_handles_unequal_lengths(tiny_model_and_tokenizer):
    """Token-level normalisation must produce a finite update when rollouts
    have different completion lengths (the case where it diverges from the
    old per-sample mean)."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    # Mixed completion lengths: 4, 4, 10, 10 tokens (prompt_length=2).
    rollouts = [
        _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2),
        _build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2),
        _build_rollout(list(range(1, 13)), 0.0, 2),
        _build_rollout(list(range(1, 13)), 0.0, 2),
    ]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)

    before = next(model.parameters()).detach().clone()
    train_step(model, [[group]], ref_model=_make_ref(model))
    after = next(model.parameters()).detach().clone()
    assert torch.isfinite(after).all()
    assert (before - after).abs().max().item() > 0.0


def test_train_step_empty_batch_noop(tiny_model_and_tokenizer):
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    result = train_step(model, [[]], ref_model=_make_ref(model))
    assert result is model


def test_train_step_degenerate_groups_skipped(tiny_model_and_tokenizer):
    """All-same-reward groups contribute zero signal → no optimizer step."""
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    # All rewards identical → advantages all zero → no backward pass
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], 1.0, 2) for _ in range(4)]
    group = _FakeGroup(rollouts=rollouts)

    before = next(model.parameters()).detach().clone()
    train_step(model, [[group]], ref_model=_make_ref(model))
    after = next(model.parameters()).detach().clone()
    # No update should happen
    assert torch.equal(before, after)


def test_train_step_requires_ref_model_kwarg(tiny_model_and_tokenizer):
    """train_step must receive ref_model as an explicit kwarg — it no longer
    deep-copies internally."""
    import inspect
    sig = inspect.signature(train_step)
    assert "ref_model" in sig.parameters
    assert sig.parameters["ref_model"].kind == inspect.Parameter.KEYWORD_ONLY


def test_train_step_uses_caller_provided_ref(tiny_model_and_tokenizer):
    reset_training_state()
    model, _ = tiny_model_and_tokenizer
    import copy
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False
    rollouts = [_build_rollout([1, 2, 3, 4, 5, 6], r, 2) for r in [1, 1, 0, 0]]
    group = _FakeGroup(rollouts=rollouts, prompt_idx=0)
    before = next(model.parameters()).detach().clone()
    train_step(model, [[group]], ref_model=ref)
    after = next(model.parameters()).detach().clone()
    assert (before - after).abs().max().item() > 0.0
