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


@dataclass(frozen=True)
class SeedConsistencyDiagnostics:
    n_positions: int = 0
    n_stochastic: int = 0
    n_exact_match: int = 0
    n_boundary_match: int = 0
    n_hard_mismatch: int = 0
    n_deterministic_hard_mismatch: int = 0
    max_cdf_miss: float = 0.0


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


def u_at(randomness: str, hotkey: str, prompt_idx: int, checkpoint_hash: str,
         rollout_index: int, t: int) -> float:
    """Public uniform in [0, 1) for rollout `rollout_index`, completion position `t`."""
    msg = (FORCED_SEED_DOMAIN.encode()
           + _lp(randomness.encode()) + _lp(hotkey.encode())
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


def seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *,
                     t: float, top_k: int, top_p: float,
                     stochastic_threshold: float) -> tuple[int, int]:
    """Teacher-forced check. logits is [n, vocab] predicting token_ids[i] at u_values[i].
    Counts stochastic positions (max_prob < threshold) and how many match the forced pick.

    Fully vectorized: one batched warp + one batched inverse-CDF over all n
    positions, with a single device sync for the two returned counts (the
    per-position loop used to force a GPU->CPU sync at every step, on the serial
    GRAIL-verify hot path)."""
    diagnostics = seed_consistency_diagnostics(
        logits,
        token_ids,
        u_values,
        t=t,
        top_k=top_k,
        top_p=top_p,
        stochastic_threshold=stochastic_threshold,
        boundary_epsilon=0.0,
    )
    return diagnostics.n_stochastic, diagnostics.n_exact_match


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
) -> SeedConsistencyDiagnostics:
    """Classify exact picks and numerical CDF-boundary misses on GPU.

    A submitted token is exact when the public uniform lies in its
    validator-recomputed inverse-CDF interval. A non-exact token is only
    boundary-compatible when the uniform is within ``boundary_epsilon`` of
    that interval. This distinguishes numerical drift near a real decision
    boundary from generic percentage forgiveness for arbitrary branch edits.
    """
    n = min(len(token_ids), len(u_values), int(logits.shape[0]))
    if n == 0:
        return SeedConsistencyDiagnostics()
    if boundary_epsilon < 0:
        raise ValueError("boundary_epsilon must be non-negative")

    probs = _warp_batch(logits[:n], t=t, top_k=top_k, top_p=top_p)
    max_probs = probs.max(dim=-1).values
    stochastic = max_probs < stochastic_threshold
    cdf = torch.cumsum(probs, dim=-1)
    toks = torch.tensor(
        [int(x) for x in token_ids[:n]],
        device=cdf.device,
        dtype=torch.long,
    )
    row = torch.arange(n, device=cdf.device)
    upper = cdf[row, toks]
    mass = probs[row, toks]
    lower = upper - mass
    u = torch.tensor(
        [float(x) for x in u_values[:n]],
        device=cdf.device,
        dtype=cdf.dtype,
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

    counts = torch.stack(
        (
            stochastic.sum(),
            (stochastic & exact).sum(),
            boundary_match.sum(),
            hard_mismatch.sum(),
            deterministic_hard.sum(),
        )
    ).tolist()
    return SeedConsistencyDiagnostics(
        n_positions=n,
        n_stochastic=int(counts[0]),
        n_exact_match=int(counts[1]),
        n_boundary_match=int(counts[2]),
        n_hard_mismatch=int(counts[3]),
        n_deterministic_hard_mismatch=int(counts[4]),
        max_cdf_miss=float(miss.max().item()),
    )
