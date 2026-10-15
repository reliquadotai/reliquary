"""The Gumbel-max forced draw (protocol ``sampling.draw = "gumbel_max"``).

pick_t = argmax_v (logit_v + G_v), G_v = -log(-log U_v), U_v a counter-based splitmix64 uniform of
(key_t, v), key_t the 64 bits of the public uniform u_t the inverse-CDF draw already uses.
"""

import math

import pytest
import torch

from reliquary.environment import forced_sampling as fs

MASK64 = (1 << 64) - 1


def _splitmix_reference(key: int, token: int) -> int:
    """The spec, in exact Python integers: the top 53 bits of the splitmix64 finaliser of
    (key * 0x9E3779B97F4A7C15 + token * 0xBF58476D1CE4E5B9) mod 2**64."""
    x = (key * 0x9E3779B97F4A7C15 + token * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x >> 11


def test_key_is_the_64_bit_image_of_the_public_uniform():
    u = fs.u_at("ab" * 16, 3, "sha:ck", 5, 17)
    key = fs.gumbel_key(u)
    unsigned = int(u * 2.0**64) & MASK64
    assert key == (unsigned - (1 << 64) if unsigned >= 1 << 63 else unsigned)
    assert -(1 << 63) <= key < (1 << 63)
    assert fs.gumbel_key(0.0) == 0
    assert fs.gumbel_key(0.5) == -(1 << 63)  # 2**63 wraps to the int64 minimum
    assert fs.gumbel_key(fs.u_at("ab" * 16, 3, "sha:ck", 5, 18)) != key


def test_uniform_integers_follow_the_exact_integer_spec():
    keys = [0, 1, -1, fs.gumbel_key(0.123456789), fs.gumbel_key(0.987654321), -(1 << 63)]
    ids = [0, 1, 2, 151643, 248319, 248320 - 2]
    got = fs.gumbel_uniform_bits(torch.tensor(keys), torch.tensor(ids))
    for row, key in enumerate(keys):
        for col, token in enumerate(ids):
            assert int(got[row, col]) == _splitmix_reference(key & MASK64, token)


def test_noise_is_a_function_of_key_and_token_only():
    keys = torch.tensor([fs.gumbel_key(fs.u_at("cd" * 16, 1, "c", r, 0)) for r in range(5)])
    ids = torch.arange(1000)
    full = fs.gumbel_noise(keys, ids)
    assert full.dtype == torch.float64
    assert torch.equal(full[2:4, 100:300], fs.gumbel_noise(keys[2:4], ids[100:300]))
    assert torch.isfinite(full).all()


def test_noise_is_standard_gumbel():
    keys = torch.tensor([fs.gumbel_key(fs.u_at("ef" * 16, 0, "c", 0, t)) for t in range(50)])
    g = fs.gumbel_noise(keys, torch.arange(4000)).flatten()
    euler_gamma = 0.5772156649
    assert abs(float(g.mean()) - euler_gamma) < 0.01
    assert abs(float(g.var()) - math.pi**2 / 6) < 0.03


def _honest_rows(n: int, vocab: int, *, scale: float = 3.0, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, vocab, generator=gen) * scale
    us = [fs.u_at("aa" * 16, 4, "sha:x", 2, t) for t in range(n)]
    return logits, us


def test_gumbel_pick_is_argmax_of_logits_plus_noise():
    logits, us = _honest_rows(40, 300)
    picks = fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX)
    keys = torch.tensor([fs.gumbel_key(u) for u in us])
    ref = (logits.double() + fs.gumbel_noise(keys, torch.arange(300))).argmax(-1)
    assert torch.equal(picks, ref)


def test_gumbel_pick_does_not_depend_on_chunking_or_dtype_upcast():
    logits, us = _honest_rows(70, 500)
    whole = fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX)
    parts = torch.cat([fs.pick_rows(logits[i:i + 7], us[i:i + 7], draw=fs.DRAW_GUMBEL_MAX)
                       for i in range(0, 70, 7)])
    assert torch.equal(whole, parts)
    bf16 = logits.to(torch.bfloat16)
    assert torch.equal(fs.pick_rows(bf16, us, draw=fs.DRAW_GUMBEL_MAX),
                       fs.pick_rows(bf16.float(), us, draw=fs.DRAW_GUMBEL_MAX))


def test_inverse_cdf_pick_rows_is_the_existing_pick():
    logits, us = _honest_rows(30, 64)
    picks = fs.pick_rows(logits, us, draw=fs.DRAW_INVERSE_CDF, t=1.0, top_k=0, top_p=1.0)
    for row in range(30):
        assert int(picks[row]) == fs.pick(fs.warp(logits[row], 1.0, 0, 1.0), us[row])


def test_gumbel_is_an_exact_sampler_chi_square():
    probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    logits = probs.log() + 7.0  # a constant shift changes nothing
    n = 30000
    us = [fs.u_at("99" * 16, 11, "sha:chi", 0, t) for t in range(n)]
    picks = fs.pick_rows(logits.expand(n, -1).contiguous(), us, draw=fs.DRAW_GUMBEL_MAX)
    counts = torch.bincount(picks, minlength=6).double()
    expected = probs.double() * n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 20.52  # df=5, alpha=0.001


def test_gumbel_rejects_a_warped_profile():
    logits, us = _honest_rows(2, 10)
    for kw in ({"t": 0.6}, {"top_k": 20}, {"top_p": 0.95}):
        with pytest.raises(ValueError, match="gumbel_max"):
            fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX, **kw)


def test_unknown_draw_is_refused():
    logits, us = _honest_rows(2, 10)
    with pytest.raises(ValueError, match="draw"):
        fs.pick_rows(logits, us, draw="top1")


def _diagnostics(logits, tokens, us, **kw):
    kw.setdefault("boundary_epsilon", 0.0)
    return fs.seed_consistency_diagnostics(
        logits, tokens, us, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99,
        draw=fs.DRAW_GUMBEL_MAX, **kw,
    )


def test_gumbel_diagnostics_accept_honest_picks_exactly():
    logits, us = _honest_rows(64, 200, scale=1.0)
    tokens = fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX).tolist()
    d = _diagnostics(logits, tokens, us)
    assert d.n_positions == 64
    assert d.n_stochastic > 50
    assert d.n_exact_match == d.n_stochastic
    assert d.n_hard_mismatch == 0 and d.max_cdf_miss == 0.0
    assert d.first_hard_mismatch_offset is None


def test_gumbel_diagnostics_catch_tokens_drawn_from_another_stream():
    logits, us = _honest_rows(64, 200, scale=1.0)
    other = [fs.u_at("bb" * 16, 4, "sha:x", 2, t) for t in range(64)]
    tokens = fs.pick_rows(logits, other, draw=fs.DRAW_GUMBEL_MAX).tolist()
    d = _diagnostics(logits, tokens, us)
    assert d.n_exact_match < 0.3 * d.n_stochastic
    assert d.n_hard_mismatch > 0 and 0.0 < d.max_cdf_miss < 1.0


def test_gumbel_miss_is_the_relative_score_shortfall():
    logits, us = _honest_rows(1, 50, scale=1.0)
    keys = torch.tensor([fs.gumbel_key(us[0])])
    scores = logits.double() + fs.gumbel_noise(keys, torch.arange(50))
    order = scores[0].argsort(descending=True)
    runner_up = int(order[1])
    gap = float(scores[0, order[0]] - scores[0, runner_up])
    exact, miss = fs.forced_pick_diagnostics(logits[0], runner_up, us[0], draw=fs.DRAW_GUMBEL_MAX)
    assert not exact
    assert miss == pytest.approx(-math.expm1(-gap), rel=1e-12)
    exact, miss = fs.forced_pick_diagnostics(logits[0], int(order[0]), us[0], draw=fs.DRAW_GUMBEL_MAX)
    assert exact and miss == 0.0
    # the batched diagnostics report the same distance and classify it against epsilon
    d = _diagnostics(logits, [runner_up], us, boundary_epsilon=-math.expm1(-gap) * 1.01)
    assert d.n_boundary_match == 1 and d.n_hard_mismatch == 0
    assert d.max_cdf_miss == pytest.approx(-math.expm1(-gap), rel=1e-6)
    d = _diagnostics(logits, [runner_up], us, boundary_epsilon=-math.expm1(-gap) * 0.99)
    assert d.n_boundary_match == 0 and d.n_hard_mismatch == 1


def test_gumbel_diagnostics_select_logit_rows_and_offsets():
    logits, us = _honest_rows(10, 80, scale=1.0)
    rows = [9, 3, 5]
    tokens = fs.pick_rows(logits[rows], [us[r] for r in rows], draw=fs.DRAW_GUMBEL_MAX).tolist()
    tokens[1] = (tokens[1] + 1) % 80
    d = _diagnostics(logits, tokens, [us[r] for r in rows], logit_positions=rows,
                     position_offsets=[40, 41, 42])
    assert d.n_positions == 3
    assert d.n_boundary_match == 2
    assert d.first_hard_mismatch_offset == 41


def test_inverse_cdf_diagnostics_are_unchanged_by_the_draw_argument():
    logits, us = _honest_rows(32, 100, scale=1.0)
    tokens = fs.pick_rows(logits, us, draw=fs.DRAW_INVERSE_CDF, t=1.0, top_k=0, top_p=1.0).tolist()
    kw = dict(t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99, boundary_epsilon=0.002)
    assert (fs.seed_consistency_diagnostics(logits, tokens, us, **kw)
            == fs.seed_consistency_diagnostics(logits, tokens, us, draw=fs.DRAW_INVERSE_CDF, **kw))


def test_inverse_cdf_single_position_diagnostics_keep_the_cdf_distance():
    probs = torch.tensor([0.5, 0.3, 0.2])
    logits = probs.log()
    exact, miss = fs.forced_pick_diagnostics(logits, 0, 0.6, draw=fs.DRAW_INVERSE_CDF,
                                             t=1.0, top_k=0, top_p=1.0)
    assert not exact and miss == pytest.approx(0.1, abs=1e-6)
    assert fs.forced_pick_diagnostics(logits, 1, 0.6, draw=fs.DRAW_INVERSE_CDF,
                                      t=1.0, top_k=0, top_p=1.0) == (True, 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_uniform_bits_are_bit_identical_to_cpu_and_picks_agree():
    keys = torch.tensor([fs.gumbel_key(fs.u_at("12" * 16, p, "c", r, t))
                         for p, r, t in [(0, 0, 0), (3, 7, 1000), (31, 15, 4095)]])
    ids = torch.arange(248320)
    cpu_bits = fs.gumbel_uniform_bits(keys, ids)
    gpu_bits = fs.gumbel_uniform_bits(keys.cuda(), ids.cuda()).cpu()
    assert torch.equal(cpu_bits, gpu_bits)
    cpu_g = fs.gumbel_noise(keys, ids)
    gpu_g = fs.gumbel_noise(keys.cuda(), ids.cuda()).cpu()
    # float64 log differs by at most an ulp or two between libm and libdevice
    assert float(((cpu_g - gpu_g).abs() / cpu_g.abs().clamp_min(1e-300)).max()) < 1e-14
    logits, us = _honest_rows(32, 248320)
    assert torch.equal(fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX),
                       fs.pick_rows(logits.cuda(), us, draw=fs.DRAW_GUMBEL_MAX).cpu())
