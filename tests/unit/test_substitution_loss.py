"""The substitution loss (docs/rl-async-design.md 4.7): the policy term scores v*, the forced
Gumbel pick recomputed from the trainer's own logits with the rollout's public uniforms."""

import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.validator import training, verifier

ROOT = Path(__file__).resolve().parents[2]
GUMBEL_PROFILE = "teutonic-9b-reliquary-suite-v10-gumbel-dev1"
_VOCAB, _HIDDEN, _PROMPT = 48, 16, 3
_RANDOMNESS = "ab" * 32
_GUMBEL = verifier.DrawSettings(draw=fs.DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0,
                                stochastic_threshold=0.99, boundary_epsilon=0.05)


@dataclass
class _Rollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _Group:
    rollouts: list
    prompt_idx: int = 7


class _Base(torch.nn.Module):
    """Embedding-only base: the logits at a position depend on that position's token only."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(_VOCAB, _HIDDEN)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class _Model(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Base()
        self.lm_head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
        with torch.no_grad():
            self.lm_head.weight.mul_(0.3)  # most positions stochastic (max p < 0.99)


@pytest.fixture(autouse=True)
def _no_kl(monkeypatch):
    monkeypatch.setattr(training, "KL_BETA", 0.0)  # the run trains with KL 0; no reference model here


@pytest.fixture
def substitution(monkeypatch):
    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "substitution")
    monkeypatch.setattr(verifier, "draw_settings", lambda: _GUMBEL)


def _uniforms(rollout_index, n):
    return [fs.u_at(_RANDOMNESS, 7, "ckpt", rollout_index, j) for j in range(n)]


def _logits_at(model, token):
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        return model.lm_head(model.model.emb(torch.tensor([token])))


def _rollout(model, rollout_index, n, *, honest=True, reward=1.0, seed=0):
    """A rollout drawn with the forced Gumbel pick of ``model`` (honest), or random tokens."""
    gen = torch.Generator().manual_seed(seed + rollout_index)
    tokens = torch.randint(0, _VOCAB, (_PROMPT,), generator=gen).tolist()
    us = _uniforms(rollout_index, n)
    for j in range(n):
        if honest:
            tokens.append(int(fs.pick_rows(_logits_at(model, tokens[-1]), [us[j]],
                                           draw=fs.DRAW_GUMBEL_MAX)[0]))
        else:
            tokens.append(int(torch.randint(0, _VOCAB, (1,), generator=gen)))
    rollout = _Rollout(tokens=tokens, reward=reward, commit={
        "tokens": tokens,
        "rollout": {"prompt_length": _PROMPT, "completion_length": n, "token_logprobs": [-1.0] * n},
    })
    rollout._forced_draw_uniforms = us
    return rollout


def _grads(model, plan):
    for p in model.parameters():
        p.grad = None
    stats = training._new_kl_stats()
    training._accumulate_grouped_grads(model, None, plan, torch.device("cpu"),
                                       budget=10_000, atomic=False, kl_stats=stats)
    return [p.grad.clone() for p in model.parameters()], stats


def _plan(rollouts, advantages):
    n = sum(len(r.commit["rollout"]["token_logprobs"]) for r in rollouts)
    return [(_Group(rollouts), advantages, 1.0 / n)]


def test_default_profile_trains_the_current_loss():
    from reliquary import constants

    assert constants.FORCED_DRAW == "inverse_cdf"
    assert constants.TRAINING_LOSS == "current" == training.training_loss()


@pytest.mark.parametrize("override,expected", [("", "substitution"), ("current", "current")])
def test_gumbel_profile_defaults_to_substitution_and_the_guard_switches_it(override, expected):
    env = {**os.environ, "RELIQUARY_PROTOCOL_PROFILE": GUMBEL_PROFILE, "RELIQUARY_TRAINING_LOSS": override}
    out = subprocess.run(
        [sys.executable, "-c", "from reliquary import constants as c; print(c.TRAINING_LOSS)"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == expected


def test_an_unknown_training_loss_is_refused():
    env = {**os.environ, "RELIQUARY_TRAINING_LOSS": "mask"}
    out = subprocess.run([sys.executable, "-c", "import reliquary.constants"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "RELIQUARY_TRAINING_LOSS" in out.stderr


def test_substitution_equals_current_where_v_star_is_the_miner_token(monkeypatch, substitution):
    model = _Model()
    rollouts = [_rollout(model, i, 12, reward=float(i % 2)) for i in range(4)]
    plan = _plan(rollouts, [1.0, -1.0, 0.5, -0.5])
    sub_grads, stats = _grads(model, plan)
    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "current")
    cur_grads, _ = _grads(model, plan)
    assert all(torch.equal(a, b) for a, b in zip(sub_grads, cur_grads))
    metrics = training._kl_telemetry_metrics(stats)
    assert metrics["train/substitution_rate"] == 0.0
    assert metrics["train/draw_match_rate"] == 1.0
    assert metrics["train/draw_coverage_ratio"] == 1.0
    assert metrics["train/entropy"] > 0.0
    assert stats["draw_stochastic_count"] > 30


def test_gradient_flows_to_v_star_not_the_miner_token(monkeypatch, substitution):
    """One completion token, which the miner got wrong: the step equals the current loss
    on the rollout that carries v* in its place, and differs from training the miner's token."""
    model = _Model()
    honest = _rollout(model, 0, 1)
    v_star = honest.tokens[-1]
    wrong = copy.deepcopy(honest)
    wrong.tokens[-1] = (v_star + 1) % _VOCAB
    wrong.commit["tokens"] = wrong.tokens
    got, stats = _grads(model, _plan([wrong], [1.0]))
    assert stats["draw_substituted_count"] == 1

    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "current")
    # The substituted row's pi_old is the policy's own value (ratio 1): give the reference the same.
    lp = torch.log_softmax(_logits_at(model, honest.tokens[-2]).float(), -1)[0, v_star]
    honest.commit["rollout"]["token_logprobs"] = [float(lp)]
    want, _ = _grads(model, _plan([honest], [1.0]))
    miner, _ = _grads(model, _plan([wrong], [1.0]))
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(got, want))
    head = 1  # lm_head.weight
    assert got[head][v_star].abs().sum() > 0
    assert not torch.allclose(got[head], miner[head])


def test_rows_without_uniforms_keep_the_miner_token(monkeypatch, substitution):
    model = _Model()
    wrong = _rollout(model, 0, 8, honest=False)
    wrong._forced_draw_uniforms = None
    got, stats = _grads(model, _plan([wrong], [1.0]))
    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "current")
    want, _ = _grads(model, _plan([wrong], [1.0]))
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert training._kl_telemetry_metrics(stats)["train/draw_coverage_ratio"] == 0.0


def test_v_star_and_match_counts_are_the_verifiers(substitution):
    """The training forward's per-rollout (n_stochastic, n_match) is the proof's own, and an
    honest rollout's tokens are v* everywhere."""
    model = _Model().to(torch.bfloat16)
    rollouts = [_rollout(model, 0, 40), _rollout(model, 1, 40, honest=False)]
    _grads(model, _plan(rollouts, [1.0, -1.0]))
    for index, rollout in enumerate(rollouts):
        tokens = rollout.tokens
        with torch.no_grad():
            hidden = model.model.emb(torch.tensor([tokens]))
        head_model = SimpleNamespace(lm_head=model.lm_head, parameters=lambda: iter([torch.zeros(1)]))
        with patch("reliquary.shared.forward.forward_single_layer", return_value=(hidden, None)), \
             patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=_HIDDEN), \
             patch.object(verifier, "_eos_set_from_model", return_value={0}):
            proof = verifier.verify_commitment_proofs(
                {"tokens": tokens, "commitments": [{"sketch": 0}] * len(tokens),
                 "rollout": rollout.commit["rollout"]},
                head_model, _RANDOMNESS, seed_u_values=_uniforms(index, 40),
            )
        assert rollout._training_draw_counts == (proof.seed_n_stochastic, proof.seed_n_match)
    honest, random = rollouts[0]._training_draw_counts, rollouts[1]._training_draw_counts
    assert honest[0] > 20 and honest[1] == honest[0]
    assert random[1] < random[0] // 2


def test_chunking_gives_identical_picks(substitution):
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(37, _VOCAB, generator=gen)
    targets = torch.randint(0, _VOCAB, (37,), generator=gen)
    draws = [None if j % 5 == 0 else fs.u_at(_RANDOMNESS, 1, "c", 0, j) for j in range(37)]
    runs = [training._forced_draw_rows(lambda rows: rows, logits, targets, draws, chunk=c)
            for c in (1, 4, 37, 512)]
    for run in runs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(run, runs[0]))
    drawn = [j for j, u in enumerate(draws) if u is not None]
    whole = fs.pick_rows(logits[drawn], [draws[j] for j in drawn], draw=fs.DRAW_GUMBEL_MAX)
    assert torch.equal(runs[0].targets[drawn], whole)
    undrawn = [j for j, u in enumerate(draws) if u is None]
    assert torch.equal(runs[0].targets[undrawn], targets[undrawn])


def test_the_fused_fsdp2_head_substitutes_like_the_chunk_loop(substitution):
    from reliquary.trainer.distributed import FusedLogprobHead

    torch.manual_seed(0)
    plain = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
    fused = copy.deepcopy(plain)
    fused.__class__ = FusedLogprobHead
    hidden = torch.randn(700, _HIDDEN, requires_grad=True)
    targets = torch.randint(0, _VOCAB, (700,))
    draws = [fs.u_at(_RANDOMNESS, 2, "c", 0, j) for j in range(700)]
    a, rows_a = training._selected_logprobs_from_hidden(hidden, targets, plain, chunk=256, draws=draws)
    a.sum().backward()
    ga, gwa = hidden.grad.clone(), plain.weight.grad.clone()
    hidden.grad = None
    b, rows_b = training._selected_logprobs_from_hidden(hidden, targets, fused, chunk=256, draws=draws)
    b.sum().backward()
    # Same picks; the values agree to the projection kernel's ulp (the two heads already
    # differ by that at this shape without a draw).
    assert all(torch.equal(x, y) for x, y in zip(rows_a[:4], rows_b[:4]))
    assert torch.allclose(rows_a.entropy, rows_b.entropy, atol=1e-6)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(ga, hidden.grad, atol=1e-6) and torch.allclose(gwa, fused.weight.grad, atol=1e-6)
    assert not torch.equal(rows_a.targets, targets)


def test_fp32_head_draw_reads_bf16_operands_in_fp32_under_autocast(monkeypatch):
    monkeypatch.setattr(verifier, "lm_head_fp32", lambda: True)
    gen = torch.Generator().manual_seed(4)
    weight = torch.randn(_VOCAB, _HIDDEN, generator=gen)
    rows = torch.randn(5, _HIDDEN, generator=gen)
    project = training._draw_logits(None, weight)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = project(rows)
    assert got.dtype == torch.float32
    assert torch.equal(got, rows.to(torch.bfloat16).float() @ weight.to(torch.bfloat16).float().T)


def test_current_loss_ignores_uniforms_and_logs_no_draw_metrics():
    model = _Model()
    rollouts = [_rollout(model, i, 10, honest=False) for i in range(3)]
    plan = _plan(rollouts, [1.0, -0.5, -0.5])
    with_u, stats = _grads(model, plan)
    for rollout in rollouts:
        rollout._forced_draw_uniforms = None
    without_u, stats_without = _grads(model, plan)
    assert all(torch.equal(a, b) for a, b in zip(with_u, without_u))
    assert stats == stats_without and "draw_token_count" not in stats
    assert not any(k.startswith("train/draw") or k == "train/substitution_rate"
                   for k in training._kl_telemetry_metrics(stats))
    assert not hasattr(rollouts[0], "_training_draw_counts")

