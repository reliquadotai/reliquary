"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.
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
) -> SeedConsistencyDiagnostics:
    """Classify exact picks and numerical CDF-boundary misses on GPU.

    A submitted token is exact when the public uniform lies in its
    validator-recomputed inverse-CDF interval. A non-exact token is only
    boundary-compatible when the uniform is within ``boundary_epsilon`` of
    that interval. This distinguishes numerical drift near a real decision
    boundary from generic percentage forgiveness for arbitrary branch edits.
    """
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
    chunk_rows = max(
        1,
        min(n, _DIAGNOSTIC_FLOAT_WORKSPACE_BYTES // (vocab_size * 4)),
    )
    counts = [0] * 8
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
        u = torch.tensor(
            [float(x) for x in u_values[start:end]],
            device=upper.device,
            dtype=upper.dtype,
        )

        exact = (u >= lower) & (u < upper)
        miss = torch.where(
            u < lower,
            lower - u,
            torch.where(u >= upper, u - upper, torch.zeros_like(u)),
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
            device=upper.device,
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
    )
