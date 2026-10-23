"""The forced draw's single-pass stats (``training._draw_stats``) against the four-tensor computation they replace,
and the picks' independence from the stats pass (2026-10-23)."""
import random

import pytest
import torch

from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, DRAW_INVERSE_CDF
from reliquary.validator import training, verifier

THRESHOLD = 0.99


def _settings(draw):
    return verifier.DrawSettings(draw=draw, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=THRESHOLD,
                                 boundary_epsilon=0.0)


def _logits(n=96, vocab=4099, seed=0):
    gen = torch.Generator().manual_seed(seed)
    scale = torch.linspace(0.1, 40.0, n)[:, None]           # flat rows to near-deterministic ones
    x = torch.randn(n, vocab, generator=gen) * scale
    x[::7, ::5] = float("-inf")                                # some masked vocab entries
    return x


def _reference(logits_of, rows, targets, draws, chunk, settings):
    """The pre-2026-10-23 body of ``_forced_draw_rows``."""
    from reliquary.environment.forced_sampling import gumbel_gaps, pick_rows

    n = int(targets.shape[0])
    picked = targets.clone()
    stochastic = torch.zeros(n, dtype=torch.bool)
    entropy = torch.zeros(n, dtype=torch.float32)
    gap = torch.zeros(n, dtype=torch.float32)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        logits = logits_of(rows[start:end])
        logp = torch.log_softmax(logits.float(), dim=-1)
        probs = logp.exp()
        stochastic[start:end] = probs.max(dim=-1).values < settings.stochastic_threshold
        entropy[start:end] = -(probs * logp).nan_to_num(0.0).sum(dim=-1)
        local = [k for k in range(start, end) if draws[k] is not None]
        if local:
            index = torch.tensor(local, dtype=torch.long)
            rows_logits = logits.index_select(0, index - start)
            uniforms = [float(draws[k]) for k in local]
            if settings.draw == DRAW_GUMBEL_MAX:
                chosen, gaps = gumbel_gaps(rows_logits, uniforms, targets[index])
                gap[index] = gaps
            else:
                chosen = pick_rows(rows_logits, uniforms, draw=settings.draw, t=1.0, top_k=0, top_p=1.0)
            picked[index] = chosen
    return picked, stochastic, entropy, gap


@pytest.mark.parametrize("draw", [DRAW_GUMBEL_MAX, DRAW_INVERSE_CDF])
@pytest.mark.parametrize("holes", [False, True])
def test_the_single_pass_stats_match_the_four_tensor_ones_and_picks_are_identical(monkeypatch, draw, holes):
    monkeypatch.setattr(verifier, "draw_settings", lambda: _settings(draw))
    logits = _logits()
    rng = random.Random(1)
    draws = [None if holes and k % 3 == 0 else rng.random() for k in range(logits.shape[0])]
    targets = logits.argmax(dim=-1)
    ref_pick, ref_stoch, ref_ent, ref_gap = _reference(lambda r: r, logits, targets, draws, 32, _settings(draw))
    kept = logits.clone()
    got = training._forced_draw_rows(lambda r: r, logits, targets, draws, chunk=32)
    assert torch.equal(logits, kept)                             # an identity map: the caller's logits intact
    assert torch.equal(got.targets, ref_pick) and torch.equal(got.gap, ref_gap)
    maxp = torch.softmax(logits.double(), dim=-1).max(dim=-1).values
    clear = (maxp - THRESHOLD).abs() > 1e-4
    assert torch.equal(got.stochastic[clear], ref_stoch[clear])
    assert torch.allclose(got.entropy, ref_ent, rtol=1e-5, atol=1e-5)
    exact = -(torch.softmax(logits.double(), -1) * torch.log_softmax(logits.double(), -1)).nan_to_num(0.0).sum(-1)
    assert (got.entropy.double() - exact).abs().max() <= (ref_ent.double() - exact).abs().max() + 1e-5


def test_a_projection_map_may_be_overwritten_with_the_same_results(monkeypatch):
    monkeypatch.setattr(verifier, "draw_settings", lambda: _settings(DRAW_GUMBEL_MAX))
    gen = torch.Generator().manual_seed(3)
    hidden = torch.randn(70, 64, generator=gen)
    weight = torch.randn(3001, 64, generator=gen) * 3
    rng = random.Random(2)
    draws = [rng.random() for _ in range(70)]
    targets = torch.zeros(70, dtype=torch.long)
    project = lambda r: r @ weight.T                              # noqa: E731 - a fresh tensor per chunk
    a = training._forced_draw_rows(project, hidden, targets, draws, chunk=16)
    b = training._forced_draw_rows(project, hidden, targets, draws, chunk=16, scratch=True)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_bf16_logits_are_upcast_without_touching_them(monkeypatch):
    monkeypatch.setattr(verifier, "draw_settings", lambda: _settings(DRAW_GUMBEL_MAX))
    logits = _logits(n=20, vocab=1001).nan_to_num(neginf=-1e4).to(torch.bfloat16)
    kept = logits.clone()
    stochastic, entropy = training._draw_stats(logits, THRESHOLD, scratch=True)
    assert torch.equal(logits, kept)
    ref = torch.log_softmax(logits.float(), -1)
    assert torch.allclose(entropy, -(ref.exp() * ref).sum(-1), rtol=1e-5, atol=1e-5)


def test_the_fp32_head_weight_cache_is_off_on_cpu_and_by_switch(monkeypatch):
    w = torch.randn(8, 4).to(torch.bfloat16)
    assert verifier.fp32_head_weight(w) is None                  # not a CUDA weight: nothing to cache
    monkeypatch.setenv("RELIQUARY_FP32_HEAD_WEIGHT_CACHE", "0")
    assert verifier.fp32_head_weight(w) is None
    rows = torch.randn(3, 4).to(torch.bfloat16)
    assert torch.equal(verifier.fp32_head_matmul(rows, w, weight_fp32=w.float()), rows.float() @ w.float().T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_the_cached_fp32_weight_gives_the_same_logits_on_cuda():
    gen = torch.Generator(device="cuda").manual_seed(4)
    w = torch.randn(4096, 512, generator=gen, device="cuda").to(torch.bfloat16)
    rows = torch.randn(64, 512, generator=gen, device="cuda").to(torch.bfloat16)
    w32 = verifier.fp32_head_weight(w)
    if w32 is None:
        pytest.skip("this torch has mm(out_dtype): no cast to cache")
    assert torch.equal(verifier.fp32_head_matmul(rows, w, weight_fp32=w32), verifier.fp32_head_matmul(rows, w))
