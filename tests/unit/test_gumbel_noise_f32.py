"""The float32 Gumbel noise of the ``gumbel_max`` draw: the torch reference is the definition
(module doc of ``forced_sampling``), the Triton kernel must reproduce its bits."""

import hashlib
import math
import struct

import pytest
import torch

from reliquary.environment import forced_sampling as fs

MASK64 = (1 << 64) - 1
GOLDEN_KEYS = [("golden", 0, 0), ("golden", 5, 77), ("golden", 31, 4095)]
GOLDEN_NOISE_SHA256 = "205817dcbd05a2ec036131dde98bb18c5c71c548634b0335b0343b55ee53f98a"


def _keys(spec):
    return torch.tensor([fs.gumbel_key(fs.u_at(r, p, "ck", 0, t)) for r, p, t in spec])


def _exact_gumbel(m: int) -> float:
    """G of the 53-bit integer m in float64 without the float64 (m + 0.5) / 2**53 rounding."""
    if m < 1 << 52:
        e = -math.log((2 * m + 1) / 2.0**54)
    else:
        e = -math.log1p(-((1 << 54) - 2 * m - 1) / 2.0**54)
    return -math.log(e)


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def test_broadcast_bits_are_the_outer_uniform_bits():
    keys = _keys([("b", p, t) for p in range(3) for t in range(4)])
    ids = torch.tensor([0, 1, 2, 151643, 248319])
    outer = fs.gumbel_uniform_bits(keys, ids)
    assert torch.equal(fs._splitmix_bits(keys[:, None], ids[None, :]), outer)
    paired = fs._splitmix_bits(keys[:5], ids)
    assert torch.equal(paired, outer[torch.arange(5), torch.arange(5)])


def test_logf_is_within_one_ulp_of_the_true_log():
    x = torch.cat([torch.logspace(-37, 38, 200_001), 1 - torch.rand(100_000) * 1e-3,
                   1 + torch.rand(100_000) * 1e-3, torch.rand(100_000) * 40]).float()
    x = x[(x > 0) & torch.isfinite(x)]
    got = fs._logf(x).double()
    true = torch.log(x.double())
    ulp = torch.nextafter(true.float().abs(), torch.tensor(math.inf)).double() - true.float().abs().double()
    assert float(((got - true).abs() / ulp.clamp_min(1e-45)).max()) < 1.0
    assert float(fs._logf(torch.tensor([1.0]))[0]) == 0.0


def test_noise_is_close_to_the_exact_gumbel_on_edge_and_random_bits():
    gen = torch.Generator().manual_seed(3)
    edge = [0, 1, 2, 3, (1 << 52) - 2, (1 << 52) - 1, 1 << 52, (1 << 52) + 1,
            (1 << 53) - 3, (1 << 53) - 2, (1 << 53) - 1]
    bits = torch.cat([torch.tensor(edge), torch.randint(0, 1 << 53, (20_000,), generator=gen)])
    got = fs.gumbel_noise_from_bits(bits)
    assert got.dtype == torch.float32 and bool(torch.isfinite(got).all())
    for m, g in zip(bits.tolist(), got.tolist()):
        assert abs(g - _exact_gumbel(m)) <= 4e-7 * max(1.0, abs(_exact_gumbel(m)))
    # the tail the float64 recipe rounded to U = 1 (G = inf) is finite and exact here
    assert got[len(edge) - 1] == pytest.approx(_exact_gumbel((1 << 53) - 1), rel=1e-6)
    assert float(got.max()) < 37.5 and float(got.min()) > -3.7


def test_noise_golden_bits():
    g = fs.gumbel_noise(_keys(GOLDEN_KEYS), torch.arange(248320))
    assert hashlib.sha256(g.numpy().tobytes()).hexdigest() == GOLDEN_NOISE_SHA256
    ext = torch.tensor([0, (1 << 52) - 1, 1 << 52, (1 << 53) - 2, (1 << 53) - 1])
    assert [v & 0xFFFFFFFF for v in fs.gumbel_noise_from_bits(ext).view(torch.int32).tolist()] == [
        0xC067D691, 0x3EBBA795, 0x3EBBA795, 0x4211534A, 0x4215B844]


def test_noise_is_deterministic_and_slice_invariant():
    keys = _keys([("d", p, t) for p in range(4) for t in range(3)])
    ids = torch.arange(5000)
    full = fs.gumbel_noise(keys, ids)
    assert torch.equal(full, fs.gumbel_noise(keys, ids))
    assert torch.equal(full[3:7, 1234:4321], fs.gumbel_noise(keys[3:7], ids[1234:4321]))


def test_reference_argmax_is_chunking_invariant(monkeypatch):
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(37, 3000, generator=gen) * 2
    keys = _keys([("c", 0, t) for t in range(37)])
    picks, best = fs.gumbel_argmax_reference(logits, keys)
    scores = logits + fs.gumbel_noise(keys, torch.arange(3000))
    assert torch.equal(picks, scores.argmax(-1)) and torch.equal(best, scores.max(-1).values)
    monkeypatch.setattr(fs, "_GUMBEL_WORKSPACE_BYTES", 3000 * 8 * 6 * 4)  # 4 rows per chunk
    assert fs._gumbel_chunk_rows(3000) == 4
    small = fs.gumbel_argmax_reference(logits, keys)
    assert torch.equal(small[0], picks) and torch.equal(small[1], best)


def test_first_index_wins_a_tie():
    keys = _keys([("t", 0, 0)])
    noise = fs.gumbel_noise(keys, torch.arange(64))[0]
    logits = (-noise).unsqueeze(0).clone()  # every score is exactly 0
    picks, best = fs.gumbel_argmax_reference(logits, keys)
    assert int(picks[0]) == 0 and float(best[0]) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_triton_kernel_matches_the_reference_bit_for_bit():
    pytest.importorskip("triton")
    from reliquary.environment.gumbel_kernel import gumbel_argmax_triton

    gen = torch.Generator().manual_seed(7)
    for vocab, dtype in ((248320, torch.float32), (248320, torch.bfloat16), (1000, torch.float32)):
        logits = (torch.randn(96, vocab, generator=gen) * 3).to(dtype)
        keys = _keys([("k", 1, t) for t in range(96)])
        ref = fs.gumbel_argmax_reference(logits, keys)
        got = gumbel_argmax_triton(logits.cuda(), keys.cuda())
        assert torch.equal(got[0].cpu(), ref[0]) and torch.equal(got[1].cpu(), ref[1])
    assert torch.equal(fs.pick_rows(logits.cuda(), [0.25] * 96, draw=fs.DRAW_GUMBEL_MAX).cpu(),
                       fs.pick_rows(logits, [0.25] * 96, draw=fs.DRAW_GUMBEL_MAX))
