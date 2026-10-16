"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.

Two draws, chosen by the profile's ``sampling.draw``; both read the same public uniform
``u_t`` per completion position (``u_at``, or a public seed pool's ``uniform``), so the
stream identity, its plumbing and its domain are shared:

* ``inverse_cdf`` (every profile before v10): the first token whose cumulative warped
  probability, in vocabulary order, exceeds ``u_t``.
* ``gumbel_max``: ``argmax_v (logit_v + G_v)`` over the raw logits, an exact sample of
  ``softmax(logits)``. Only the identity warp (T=1, no top-k, no top-p) is allowed. The
  noise ``G_v = -log(-log U_v)`` comes from the 53-bit integer

      key_t = int(u_t * 2**64) mod 2**64              (as a two's-complement int64)
      x     = key_t * 0x9E3779B97F4A7C15 + v * 0xBF58476D1CE4E5B9      (mod 2**64)
      x     = (x ^ x >> 30) * 0xBF58476D1CE4E5B9;  x = (x ^ x >> 27) * 0x94D049BB133111EB
      x     = x ^ x >> 31                            (logical shifts, mod 2**64)
      m     = x >> 11                                (U_v = (m + 0.5) / 2**53, in (0, 1))

  and is computed in float32, every operation an IEEE round-to-nearest +, -, *, / (no
  fused multiply-add) so it has the same bits on every device, CPU included:

      q     = 2m + 1  if m < 2**52  else  2**54 - 2m - 1      (exact, odd, < 2**53)
      t     = (f(q >> 36) * 2**36 + f(q >> 18 & 0x3FFFF) * 2**18 + f(q & 0x3FFFF)) * 2**-54
              (each 18-bit limb exact in float32, summed high to low; t ~ U or 1 - U)
      E     = -logf(t)                             if m < 2**52     (-log U, U < 1/2)
      E     = -(logf(x) + c / x),  x = 1 - t,  c = (-t) - (x - 1)   otherwise
              (-log1p(-(1 - U)): c is x's exact rounding error, so E keeps full relative
              precision as U -> 1, where the winning tokens are decided)
      G_v   = -logf(E)

  with ``logf`` the musl/fdlibm float32 log written in those operations (``_logf``, < 1 ulp
  but not correctly rounded; its exact formula is the definition). ``u_t`` is a float64
  with at most 53 significant bits of the sha256 it came from, so ``key_t`` is that value
  scaled exactly and the key is a function of the public stream only. The score
  ``logit_v + G_v`` is a float32 sum (bf16/fp16 logits are upcast exactly) and the first
  index wins a tie. ``gumbel_argmax_reference`` is the chunked torch definition (CPU
  oracle); ``gumbel_kernel`` computes the same bits in one fused Triton pass and is what
  ``pick_rows`` runs on CUDA. Against the previous float64 noise the picks disagree on 1
  of 8.5 * 10**6 measured rows (H100, gaussian and 9B logits, 2026-10-16; a float64 score
  gap of 2e-7), and the float32 recipe is more exact in the tail (the float64 ``(m + 0.5) / 2**53`` rounded to
  1.0 at ``m = 2**53 - 1`` and gave ``G = inf``).

  Unlike the inverse CDF, the pick does not depend on a cumulative sum in vocabulary
  order: a logit error on token ``a`` can move it only if ``a`` is the winner or the
  runner-up (measured honest match 0.986 against 0.945 on four card types).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from reliquary.constants import FORCED_SEED_DOMAIN


# Vocabulary-wide float32 transforms create several temporary tensors per row
# (softmax, sort values/indices, CDF, and scatter output). Bound the source
# matrix for each pass so long completions cannot consume the entire GPU.
_DIAGNOSTIC_FLOAT_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class SeedConsistencyDiagnostics:
    n_positions: int = 0
    n_stochastic: int = 0
    n_exact_match: int = 0
    n_boundary_match: int = 0
    n_hard_mismatch: int = 0
    n_deterministic_hard_mismatch: int = 0
    n_miss_gt_0_01: int = 0
    n_miss_gt_0_05: int = 0
    n_miss_gt_0_10: int = 0
    max_cdf_miss: float = 0.0
    first_hard_mismatch_offset: int | None = None
    n_drawn_exact_match: int = 0      # exact picks over every position (``n_exact_match``: stochastic ones only)


def warp(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature -> top-k -> top-p, returned in canonical (token-id ascending) order."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.numel())
        kth = torch.topk(lg, k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))  # include crossing token
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum()


def pick(probs: torch.Tensor, u: float) -> int:
    """First token id whose cumulative probability exceeds u (inverse-CDF)."""
    cdf = torch.cumsum(probs, dim=-1)
    u_tensor = torch.tensor(float(u), device=cdf.device, dtype=cdf.dtype)
    idx = int(torch.searchsorted(cdf, u_tensor, right=True))
    return min(idx, probs.numel() - 1)


def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(2, "big") + b


def u_at(randomness: str, prompt_idx: int, checkpoint_hash: str,
         rollout_index: int, t: int) -> float:
    """Public uniform in [0, 1) for rollout `rollout_index`, completion position `t`.

    v2: the hotkey is deliberately NOT hashed. The forced group for a prompt is
    therefore identical for every miner in the window, so one operator's N hotkeys
    yield N copies of the same draw — variance farming (best-of-N distinct draws)
    is impossible. Anti-pregeneration still holds: `randomness` is unknown until
    the window opens. Keyed on the v2 domain so v1 and v2 streams never collide.
    """
    msg = (FORCED_SEED_DOMAIN.encode()
           + _lp(randomness.encode())
           + int(prompt_idx).to_bytes(8, "big")
           + _lp(checkpoint_hash.encode())
           + int(rollout_index).to_bytes(4, "big")
           + int(t).to_bytes(4, "big"))
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") / 2.0**64


DRAW_INVERSE_CDF = "inverse_cdf"
DRAW_GUMBEL_MAX = "gumbel_max"
DRAWS = (DRAW_INVERSE_CDF, DRAW_GUMBEL_MAX)


def check_draw(draw: str, *, t: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> None:
    """Refuse an unknown draw, and a Gumbel draw over a warped distribution: argmax(logit + G)
    samples softmax(logits), so anything but T=1 over the full vocabulary is not its distribution."""
    if draw not in DRAWS:
        raise ValueError(f"unknown forced draw {draw!r}; expected one of {DRAWS}")
    if draw == DRAW_GUMBEL_MAX and not (
        float(t) == 1.0 and not (top_k and int(top_k) > 0) and (top_p is None or float(top_p) >= 1.0)
    ):
        raise ValueError(
            f"the gumbel_max draw needs T=1 over the full vocabulary, got T={t} top_k={top_k} top_p={top_p}"
        )


def gumbel_key(u: float) -> int:
    """The int64 key of a position's Gumbel noise: the public uniform's 64-bit image."""
    unsigned = int(float(u) * 2.0**64) & _MASK64
    return unsigned - (1 << 64) if unsigned >= 1 << 63 else unsigned


_MASK64 = (1 << 64) - 1
# splitmix64 constants as int64 (torch has no uint64 arithmetic); products wrap mod 2**64.
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX2 = 0x94D049BB133111EB - (1 << 64)


def _lsr(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical right shift of int64 (torch's >> is arithmetic)."""
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def _splitmix_bits(keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The 53-bit integer of U (module doc) for broadcast int64 ``keys`` and ``ids``."""
    x = keys.to(torch.int64) * _GOLDEN + ids.to(torch.int64) * _MIX1
    x = (x ^ _lsr(x, 30)) * _MIX1
    x = (x ^ _lsr(x, 27)) * _MIX2
    x = x ^ _lsr(x, 31)
    return _lsr(x, 11)


def gumbel_uniform_bits(keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[n] keys x [V] token ids -> [n, V] int64 holding the 53-bit integer of U (see module doc)."""
    return _splitmix_bits(keys.to(torch.int64)[:, None], ids.to(torch.int64)[None, :])


# musl/fdlibm logf, every constant exactly a float32; see ``_logf``.
LOGF_LN2_HI = 0.6931381225585938          # 0x3f317180
LOGF_LN2_LO = 9.05800061445916e-06        # 0x3717f7d1
LOGF_LG1 = 0.6666666269302368             # 0xaaaaaa * 2**-24
LOGF_LG2 = 0.40000972151756287            # 0xccce13 * 2**-25
LOGF_LG3 = 0.2849878668785095             # 0x91e9ee * 2**-25
LOGF_LG4 = 0.24279078841209412            # 0xf89e26 * 2**-26
LOGF_SQRT_HALF_BITS = 0x3F3504F3          # float32 sqrt(0.5)
LIMB_BITS = 18                            # the 54-bit integer -> float32 split


def _logf(x: torch.Tensor) -> torch.Tensor:
    """Natural log of a positive normal float32 tensor from IEEE +, -, *, / alone (musl logf,
    < 1 ulp), so every device that rounds each operation to nearest gets the same bits:
    x = 2**k * r with r in [sqrt(1/2), sqrt(2)), f = r - 1, s = f / (2 + f) and
    log x = (((s * (hfsq + R) + k * LN2_LO) - hfsq) + f) + k * LN2_HI, hfsq = (0.5 * f) * f,
    R = z * (LG1 + w * LG3) + w * (LG2 + w * LG4), z = s * s, w = z * z (no fused multiply-add)."""
    ix = x.view(torch.int32) + (0x3F800000 - LOGF_SQRT_HALF_BITS)
    k = ((ix >> 23) - 0x7F).float()
    f = ((ix & 0x007FFFFF) + LOGF_SQRT_HALF_BITS).view(torch.float32) - 1.0
    s = f / (f + 2.0)
    z = s * s
    w = z * z
    big_r = z * (w * LOGF_LG3 + LOGF_LG1) + w * (w * LOGF_LG4 + LOGF_LG2)
    hfsq = (f * 0.5) * f
    return (((s * (hfsq + big_r) + k * LOGF_LN2_LO) - hfsq) + f) + k * LOGF_LN2_HI


def gumbel_noise_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """float32 standard Gumbel noise of the 53-bit integers ``bits`` (the module doc's U,
    computed as the module doc's float32 recipe): the CPU oracle of ``gumbel_kernel``."""
    m = bits.to(torch.int64)
    low = m < (1 << 52)
    q = torch.where(low, 2 * m + 1, (1 << 54) - 2 * m - 1)
    mask = (1 << LIMB_BITS) - 1
    hi = (q >> (2 * LIMB_BITS)).float() * float(1 << (2 * LIMB_BITS))
    mid = ((q >> LIMB_BITS) & mask).float() * float(1 << LIMB_BITS)
    t = ((hi + mid) + (q & mask).float()) * 2.0**-54
    x = torch.where(low, t, 1.0 - t)
    c = torch.where(low, torch.zeros_like(t), (-t) - (x - 1.0))
    e = -(_logf(x) + c / x)
    return -_logf(e)


def gumbel_noise(keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[n, V] float32 standard Gumbel noise for positions ``keys`` and tokens ``ids``."""
    return gumbel_noise_from_bits(gumbel_uniform_bits(keys, ids))


# int64/float32 [rows, vocab] temporaries of the torch noise (~8 bytes x 6 live at once).
_GUMBEL_WORKSPACE_BYTES = 256 * 1024 * 1024


def _gumbel_chunk_rows(vocab: int) -> int:
    return max(1, _GUMBEL_WORKSPACE_BYTES // (max(1, vocab) * 8 * 6))


def _gumbel_keys(draws: list[float], device) -> torch.Tensor:
    return torch.tensor([gumbel_key(u) for u in draws], dtype=torch.int64, device=device)


def _gumbel_scores(logits: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """logit + G in float32 for a chunk of rows, on the logits' device (torch reference)."""
    ids = torch.arange(logits.shape[-1], dtype=torch.int64, device=logits.device)
    return logits.float() + gumbel_noise(keys, ids)


_TRITON_ARGMAX = None


def _triton_argmax():
    global _TRITON_ARGMAX
    if _TRITON_ARGMAX is None:
        try:
            from reliquary.environment.gumbel_kernel import gumbel_argmax_triton
        except Exception:  # no triton: the torch reference is the same function
            gumbel_argmax_triton = False
        _TRITON_ARGMAX = gumbel_argmax_triton
    return _TRITON_ARGMAX


def gumbel_argmax_reference(logits: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(argmax_v (logit + G) as int64 [n], its float32 score [n]) in chunked torch ops; the first
    index wins a tie, like ``torch.argmax`` and the kernel."""
    n = int(logits.shape[0])
    picks = torch.empty(n, dtype=torch.int64, device=logits.device)
    best = torch.empty(n, dtype=torch.float32, device=logits.device)
    step = _gumbel_chunk_rows(int(logits.shape[-1]))
    for start in range(0, n, step):
        end = min(n, start + step)
        scores = _gumbel_scores(logits[start:end], keys[start:end])
        best[start:end] = scores.max(dim=-1).values
        picks[start:end] = scores.argmax(dim=-1)
    return picks, best


def gumbel_argmax(logits: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``gumbel_argmax_reference``, fused in one Triton pass on CUDA when triton imports (no
    [n, V] temporary; bit-identical scores, so identical picks)."""
    if logits.is_cuda and int(logits.shape[0]) > 0:
        kernel = _triton_argmax()
        if kernel:
            return kernel(logits, keys)
    return gumbel_argmax_reference(logits, keys)


def pick_rows(logits: torch.Tensor, draws: list[float], *, draw: str = DRAW_INVERSE_CDF,
              t: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """The forced pick of each row of ``logits`` [n, vocab] for its public uniform ``draws[i]``,
    as a [n] int64 tensor on the logits' device. The one batched pick the miner's processors
    and the validator's checks share."""
    check_draw(draw, t=t, top_k=top_k, top_p=top_p)
    if len(draws) != int(logits.shape[0]):
        raise ValueError("one public uniform per logits row is required")
    if draw == DRAW_GUMBEL_MAX:
        return gumbel_argmax(logits, _gumbel_keys(draws, logits.device))[0]
    probs = _warp_batch(logits, t=t, top_k=top_k, top_p=top_p)
    cdf = torch.cumsum(probs, dim=-1)
    uniforms = torch.tensor(draws, device=cdf.device, dtype=cdf.dtype).unsqueeze(1)
    picks = torch.searchsorted(cdf, uniforms, right=True).squeeze(1)
    return picks.clamp(max=probs.shape[-1] - 1)


def forced_pick_diagnostics(logits: torch.Tensor, token_id: int, u: float, *,
                            draw: str = DRAW_INVERSE_CDF, t: float = 1.0, top_k: int = 0,
                            top_p: float = 1.0) -> tuple[bool, float]:
    """One position: is ``token_id`` the forced pick, and how far is it from being it.

    inverse_cdf: the distance from ``u`` to the token's CDF interval. gumbel_max: the
    relative score shortfall ``1 - exp(-(s_win - s_token))`` with ``s = logit + G``, in
    [0, 1), 0 for the pick (telemetry, like the CDF distance)."""
    check_draw(draw, t=t, top_k=top_k, top_p=top_p)
    token = int(token_id)
    if draw == DRAW_GUMBEL_MAX:
        scores = _gumbel_scores(logits.reshape(1, -1), _gumbel_keys([u], logits.device))[0]
        if int(scores.argmax()) == token:
            return True, 0.0
        return False, float(-torch.expm1(-(scores.max() - scores[token])).item())
    probs = warp(logits.float(), t=t, top_k=top_k, top_p=top_p)
    if pick(probs, u) == token:
        return True, 0.0
    cdf = torch.cumsum(probs, dim=-1)
    upper = float(cdf[token].item())
    lower = upper - float(probs[token].item())
    u_value = float(torch.tensor(float(u), device=probs.device, dtype=probs.dtype).item())
    if u_value < lower:
        miss = lower - u_value
    elif u_value >= upper:
        miss = u_value - upper
    else:
        miss = 0.0
    return False, max(0.0, float(miss))


def _warp_batch(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Row-batched ``warp``: logits [n, vocab] -> probs [n, vocab], bit-identical
    per row to the 1-D ``warp`` (each op is independent along dim=-1) but with no
    per-row Python loop."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        k = min(top_k, lg.shape[-1])
        kth = torch.topk(lg, k, dim=-1).values[..., -1:]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum(dim=-1, keepdim=True)


def _interval_stats_dense(
    chunk_logits: torch.Tensor,
    toks: torch.Tensor,
    *,
    t: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference path: full-vocab warp + CDF. Returns (max_prob, lower, upper, mass)."""
    probs = _warp_batch(chunk_logits, t=t, top_k=top_k, top_p=top_p)
    cdf = torch.cumsum(probs, dim=-1)
    row = torch.arange(chunk_logits.shape[0], device=cdf.device)
    upper = cdf[row, toks]
    mass = probs[row, toks]
    return probs.max(dim=-1).values, upper - mass, upper, mass


# Extra topk slots beyond top_k so boundary ties at the kth value stay inside
# the window. Rows whose ties might extend past the window fall back to the
# dense path, so the sparse result is never a guess.
_SPARSE_TIE_MARGIN = 32


def _interval_stats_sparse(
    chunk_logits: torch.Tensor,
    toks: torch.Tensor,
    *,
    t: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same math as ``_interval_stats_dense`` restricted to the top-k window.

    Only ≤ ``top_k`` tokens (plus kth-value ties) survive the warp, so the
    per-position CDF interval needs a ~52-value window, not a 248k-vocab sort
    and cumsum. Candidate selection runs on the raw logits (bf16→fp32 and /t
    are monotone, so the survivor set is identical); all arithmetic then runs
    in fp32 on the gathered window, mirroring the dense path. Rows whose kth
    value still ties at the window edge are recomputed densely by the caller.
    """
    vocab = int(chunk_logits.shape[-1])
    window = min(vocab, max(1, int(top_k)) + _SPARSE_TIE_MARGIN)
    vals, idx = torch.topk(chunk_logits, window, dim=-1)
    lg = vals.float() / float(t)
    if top_k and top_k > 0 and top_k < vocab:
        kth = lg[:, min(top_k, window) - 1 : min(top_k, window)]
        tie_overflow = lg[:, -1:] >= kth
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    else:
        tie_overflow = torch.zeros_like(lg[:, -1:], dtype=torch.bool)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        # vals are already sorted descending, so probs are too — the top-p
        # walk needs no extra sort.
        cum = torch.cumsum(probs, dim=-1)
        probs = torch.where(
            (cum - probs) < top_p, probs, torch.zeros_like(probs)
        )
    probs = probs / probs.sum(dim=-1, keepdim=True)
    tok_col = toks.unsqueeze(1)
    mass = (probs * (idx == tok_col)).sum(dim=-1)
    upper = (probs * (idx <= tok_col)).sum(dim=-1)
    max_probs = probs.max(dim=-1).values
    # Poison overflow rows so the caller can spot and recompute them densely.
    overflow = tie_overflow.squeeze(1)
    nan = torch.full_like(mass, float("nan"))
    return (
        torch.where(overflow, nan, max_probs),
        torch.where(overflow, nan, upper - mass),
        torch.where(overflow, nan, upper),
        torch.where(overflow, nan, mass),
    )


def seed_consistency_diagnostics(
    logits: torch.Tensor,
    token_ids: list[int],
    u_values: list[float],
    *,
    t: float,
    top_k: int,
    top_p: float,
    stochastic_threshold: float,
    boundary_epsilon: float,
    position_offsets: list[int] | None = None,
    logit_positions: list[int] | None = None,
    draw: str = DRAW_INVERSE_CDF,
) -> SeedConsistencyDiagnostics:
    """Classify exact picks and numerical CDF-boundary misses on GPU.

    A submitted token is exact when the public uniform lies in its
    validator-recomputed inverse-CDF interval. A non-exact token is only
    boundary-compatible when the uniform is within ``boundary_epsilon`` of
    that interval. This distinguishes numerical drift near a real decision
    boundary from generic percentage forgiveness for arbitrary branch edits.

    Under ``gumbel_max`` a token is exact when it is ``argmax(logit + G)``, and
    its miss is the relative score shortfall ``1 - exp(-(s_win - s_token))``
    (``forced_pick_diagnostics``), classified against ``boundary_epsilon`` in
    those units; the ``max_cdf_miss`` and ``n_miss_gt_*`` fields then carry it.
    A position is stochastic by the same max-probability rule for both draws.
    """
    check_draw(draw, t=t, top_k=top_k, top_p=top_p)
    available_rows = (
        len(logit_positions)
        if logit_positions is not None
        else int(logits.shape[0])
    )
    n = min(len(token_ids), len(u_values), available_rows)
    if position_offsets is not None:
        n = min(n, len(position_offsets))
    if n == 0:
        return SeedConsistencyDiagnostics()
    if boundary_epsilon < 0:
        raise ValueError("boundary_epsilon must be non-negative")
    selected_positions = None
    if logit_positions is not None:
        selected_positions = [int(value) for value in logit_positions[:n]]
        row_count = int(logits.shape[0])
        if any(value < 0 or value >= row_count for value in selected_positions):
            raise ValueError("logit_positions contains an out-of-range row")

    vocab_size = max(1, int(logits.shape[-1]))
    # Full-support sampling (v4: top_k=0; verl uses -1) has no top-k window to
    # restrict to: the sparse path would renormalise a 33-token slice and its
    # tie-overflow fallback is unreachable at top_k<=0, so honest full-support
    # picks would be misclassified. Route straight to the dense intervals.
    full_support = not (top_k and 0 < int(top_k) < vocab_size)
    chunk_rows = max(
        1,
        min(n, _DIAGNOSTIC_FLOAT_WORKSPACE_BYTES // (vocab_size * 4)),
    )
    if draw == DRAW_GUMBEL_MAX:
        chunk_rows = min(chunk_rows, _gumbel_chunk_rows(vocab_size))
    counts = [0] * 9
    max_cdf_miss = 0.0
    first_hard_mismatch_offset: int | None = None
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        if selected_positions is None:
            chunk_logits = logits[start:end]
        else:
            row_indices = torch.tensor(
                selected_positions[start:end],
                device=logits.device,
                dtype=torch.long,
            )
            chunk_logits = logits.index_select(0, row_indices)

        toks = torch.tensor(
            [int(x) for x in token_ids[start:end]],
            device=chunk_logits.device,
            dtype=torch.long,
        )
        if draw == DRAW_GUMBEL_MAX:
            exact, miss, stochastic = _gumbel_chunk_stats(
                chunk_logits, toks, [float(x) for x in u_values[start:end]],
                stochastic_threshold,
            )
        else:
            exact, miss, stochastic = _cdf_chunk_stats(
                chunk_logits, toks, [float(x) for x in u_values[start:end]],
                t=t, top_k=top_k, top_p=top_p, full_support=full_support,
                stochastic_threshold=stochastic_threshold,
            )
        boundary_match = exact | (miss <= float(boundary_epsilon))
        hard_mismatch = ~boundary_match
        deterministic_hard = hard_mismatch & ~stochastic
        offsets = torch.tensor(
            (
                [int(value) for value in position_offsets[start:end]]
                if position_offsets is not None
                else list(range(start, end))
            ),
            device=miss.device,
            dtype=torch.long,
        )
        hard_offsets = offsets[hard_mismatch]

        chunk_counts = torch.stack(
            (
                stochastic.sum(),
                (stochastic & exact).sum(),
                boundary_match.sum(),
                hard_mismatch.sum(),
                deterministic_hard.sum(),
                (miss > 0.01).sum(),
                (miss > 0.05).sum(),
                (miss > 0.10).sum(),
                exact.sum(),
            )
        ).tolist()
        counts = [total + int(value) for total, value in zip(counts, chunk_counts)]
        max_cdf_miss = max(max_cdf_miss, float(miss.max().item()))
        if hard_offsets.numel():
            chunk_first = int(hard_offsets.min().item())
            first_hard_mismatch_offset = (
                chunk_first
                if first_hard_mismatch_offset is None
                else min(first_hard_mismatch_offset, chunk_first)
            )

    return SeedConsistencyDiagnostics(
        n_positions=n,
        n_stochastic=int(counts[0]),
        n_exact_match=int(counts[1]),
        n_boundary_match=int(counts[2]),
        n_hard_mismatch=int(counts[3]),
        n_deterministic_hard_mismatch=int(counts[4]),
        n_miss_gt_0_01=int(counts[5]),
        n_miss_gt_0_05=int(counts[6]),
        n_miss_gt_0_10=int(counts[7]),
        max_cdf_miss=max_cdf_miss,
        first_hard_mismatch_offset=first_hard_mismatch_offset,
        n_drawn_exact_match=int(counts[8]),
    )


def _gumbel_chunk_stats(chunk_logits, toks, draws, stochastic_threshold):
    """(exact, miss, stochastic) per row under the Gumbel draw."""
    max_probs = torch.softmax(chunk_logits.float(), dim=-1).max(dim=-1).values
    keys = _gumbel_keys(draws, chunk_logits.device)
    picks, best = gumbel_argmax(chunk_logits, keys)
    token_scores = (chunk_logits.gather(1, toks.unsqueeze(1)).squeeze(1).float()
                    + gumbel_noise_from_bits(_splitmix_bits(keys, toks)))
    exact = picks == toks
    miss = torch.where(exact, torch.zeros_like(best), -torch.expm1(-(best - token_scores)))
    return exact, miss, max_probs < stochastic_threshold


def _cdf_chunk_stats(chunk_logits, toks, draws, *, t, top_k, top_p, full_support,
                     stochastic_threshold):
    """(exact, miss, stochastic) per row under the inverse-CDF draw."""
    if full_support:
        max_probs, lower, upper, mass = _interval_stats_dense(
            chunk_logits, toks, t=t, top_k=top_k, top_p=top_p,
        )
        fallback = torch.zeros_like(mass, dtype=torch.bool)
    else:
        max_probs, lower, upper, mass = _interval_stats_sparse(
            chunk_logits, toks, t=t, top_k=top_k, top_p=top_p,
        )
        fallback = torch.isnan(mass)
    if bool(fallback.any()):
        # kth-value ties reached the window edge: recompute those rows on
        # the full vocab so the sparse path never changes a verdict.
        d_max, d_lower, d_upper, d_mass = _interval_stats_dense(
            chunk_logits[fallback], toks[fallback],
            t=t, top_k=top_k, top_p=top_p,
        )
        max_probs = max_probs.masked_scatter(fallback, d_max)
        lower = lower.masked_scatter(fallback, d_lower)
        upper = upper.masked_scatter(fallback, d_upper)
        mass = mass.masked_scatter(fallback, d_mass)
    stochastic = max_probs < stochastic_threshold
    u = torch.tensor(draws, device=upper.device, dtype=upper.dtype)
    exact = (u >= lower) & (u < upper)
    miss = torch.where(
        u < lower,
        lower - u,
        torch.where(u >= upper, u - upper, torch.zeros_like(u)),
    )
    return exact, miss, stochastic
