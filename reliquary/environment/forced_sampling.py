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
    idx = int(torch.searchsorted(cdf, torch.tensor(float(u), dtype=cdf.dtype), right=True))
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


def seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *,
                     t: float, top_k: int, top_p: float,
                     stochastic_threshold: float) -> tuple[int, int]:
    """Teacher-forced check. logits is [n, vocab] predicting token_ids[i] at u_values[i].
    Counts stochastic positions (max_prob < threshold) and how many match the forced pick."""
    n_stoch = n_match = 0
    n = min(len(token_ids), len(u_values), logits.shape[0])
    for i in range(n):
        probs = warp(logits[i], t=t, top_k=top_k, top_p=top_p)
        if float(probs.max()) < stochastic_threshold:
            n_stoch += 1
            if pick(probs, u_values[i]) == int(token_ids[i]):
                n_match += 1
    return n_stoch, n_match
