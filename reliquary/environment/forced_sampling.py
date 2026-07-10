"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.
"""
from __future__ import annotations

import hashlib

import torch

from reliquary.constants import FORCED_SEED_DOMAIN


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
    n = min(len(token_ids), len(u_values), int(logits.shape[0]))
    if n == 0:
        return 0, 0
    probs = _warp_batch(logits[:n], t=t, top_k=top_k, top_p=top_p)         # [n, vocab]
    stochastic = probs.max(dim=-1).values < stochastic_threshold           # [n] bool
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.tensor([float(x) for x in u_values[:n]],
                     device=cdf.device, dtype=cdf.dtype).unsqueeze(-1)     # [n, 1]
    picks = torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp(max=probs.shape[-1] - 1)
    toks = torch.tensor([int(x) for x in token_ids[:n]],
                        device=picks.device, dtype=picks.dtype)            # [n]
    matched = stochastic & (picks == toks)
    return int(stochastic.sum().item()), int(matched.sum().item())
