"""Fused Triton Gumbel-max pick: ``forced_sampling.gumbel_argmax_reference`` in one pass.

Each program reads one ``BLOCK`` slice of one row of logits once, computes the splitmix64
bits, the float32 noise and ``logit + G`` in registers, and keeps the slice's best score
and first index; a second program per row reduces the slices (first index on a tie).
No [n, V] temporary is materialised. The arithmetic is the module doc's float32 recipe
operation for operation: uint64 wraparound with logical shifts, IEEE division
(``div_rn``) and no fused multiply-add (``enable_fp_fusion=False``), so the scores are
bit-identical to the torch reference on any device.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from reliquary.environment import forced_sampling as _fs

_BLOCK = 1024
_NUM_WARPS = 4

_GOLDEN = tl.constexpr(0x9E3779B97F4A7C15)
_MIX1 = tl.constexpr(0xBF58476D1CE4E5B9)
_MIX2 = tl.constexpr(0x94D049BB133111EB)
_LN2_HI = tl.constexpr(_fs.LOGF_LN2_HI)
_LN2_LO = tl.constexpr(_fs.LOGF_LN2_LO)
_LG1 = tl.constexpr(_fs.LOGF_LG1)
_LG2 = tl.constexpr(_fs.LOGF_LG2)
_LG3 = tl.constexpr(_fs.LOGF_LG3)
_LG4 = tl.constexpr(_fs.LOGF_LG4)
_SQRT_HALF_BITS = tl.constexpr(_fs.LOGF_SQRT_HALF_BITS)
_LIMB = tl.constexpr(_fs.LIMB_BITS)


@triton.jit
def _logf(x):
    ix = x.to(tl.int32, bitcast=True) + (0x3F800000 - _SQRT_HALF_BITS)
    k = ((ix >> 23) - 0x7F).to(tl.float32)
    f = ((ix & 0x007FFFFF) + _SQRT_HALF_BITS).to(tl.float32, bitcast=True) - 1.0
    s = tl.div_rn(f, f + 2.0)
    z = s * s
    w = z * z
    big_r = z * (w * _LG3 + _LG1) + w * (w * _LG4 + _LG2)
    hfsq = (f * 0.5) * f
    return (((s * (hfsq + big_r) + k * _LN2_LO) - hfsq) + f) + k * _LN2_HI


@triton.jit
def _bits(key, ids):
    x = key * _GOLDEN + ids * _MIX1
    x = (x ^ (x >> 30)) * _MIX1
    x = (x ^ (x >> 27)) * _MIX2
    x = x ^ (x >> 31)
    return (x >> 11).to(tl.int64, bitcast=True)


@triton.jit
def _noise_from_bits(m):
    low = m < (1 << 52)
    q = tl.where(low, 2 * m + 1, (1 << 54) - 2 * m - 1)
    mask = (1 << _LIMB) - 1
    hi = (q >> (2 * _LIMB)).to(tl.float32) * 68719476736.0  # 2**36
    mid = ((q >> _LIMB) & mask).to(tl.float32) * 262144.0  # 2**18
    t = ((hi + mid) + (q & mask).to(tl.float32)) * 5.551115123125783e-17  # 2**-54
    x1 = tl.where(low, t, 1.0 - t)
    c = tl.where(low, 0.0, (-t) - (x1 - 1.0))
    e = -(_logf(x1) + tl.div_rn(c, x1))
    return -_logf(e)


@triton.jit
def _partial_kernel(logits_ptr, stride, keys_ptr, score_ptr, index_ptr, vocab, n_parts,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    inside = offs < vocab
    logit = tl.load(logits_ptr + row.to(tl.int64) * stride + offs, mask=inside, other=0.0)
    key = tl.load(keys_ptr + row).to(tl.uint64, bitcast=True)
    score = logit.to(tl.float32) + _noise_from_bits(_bits(key, offs.to(tl.uint64)))
    score = tl.where(inside, score, float("-inf"))
    best = tl.max(score, axis=0)
    first = tl.min(tl.where(score == best, offs, 2147483647), axis=0)
    tl.store(score_ptr + row * n_parts + part, best)
    tl.store(index_ptr + row * n_parts + part, first)


@triton.jit
def _reduce_kernel(score_ptr, index_ptr, out_index_ptr, out_score_ptr, n_parts,
                   PARTS: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, PARTS)
    inside = offs < n_parts
    score = tl.load(score_ptr + row * n_parts + offs, mask=inside, other=float("-inf"))
    index = tl.load(index_ptr + row * n_parts + offs, mask=inside, other=2147483647)
    best = tl.max(score, axis=0)
    first = tl.min(tl.where(score == best, index, 2147483647), axis=0)
    tl.store(out_index_ptr + row, first.to(tl.int64))
    tl.store(out_score_ptr + row, best)


def gumbel_argmax_triton(logits: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(argmax_v (logit + G) int64 [n], its float32 score [n]) for CUDA ``logits`` [n, V]
    (float32/bfloat16/float16) and int64 ``keys`` [n]."""
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    n, vocab = int(logits.shape[0]), int(logits.shape[-1])
    keys = keys.to(device=logits.device, dtype=torch.int64).contiguous()
    n_parts = triton.cdiv(vocab, _BLOCK)
    part_score = torch.empty((n, n_parts), dtype=torch.float32, device=logits.device)
    part_index = torch.empty((n, n_parts), dtype=torch.int32, device=logits.device)
    _partial_kernel[(n, n_parts)](logits, logits.stride(0), keys, part_score, part_index, vocab,
                                  n_parts, BLOCK=_BLOCK, num_warps=_NUM_WARPS, enable_fp_fusion=False)
    picks = torch.empty(n, dtype=torch.int64, device=logits.device)
    best = torch.empty(n, dtype=torch.float32, device=logits.device)
    _reduce_kernel[(n,)](part_score, part_index, picks, best, n_parts,
                         PARTS=triton.next_power_of_2(n_parts), num_warps=1)
    return picks, best
