#!/usr/bin/env python
"""Bit-identity of the Gumbel forced draw across devices.

Prints sha256 prefixes of the uniform integers, of G (float64) and of the picks on fixed
logits over the 9B vocabulary, for one device. Run it on every card type (and on CPU):
the ``bits`` hash must match everywhere; ``G`` and ``picks`` must match on every GPU (CPU
libm may differ from CUDA libdevice by an ulp in G, see forced_sampling).

    python scripts/gumbel_check.py [cuda|cpu]
"""
import hashlib
import sys

import torch

from reliquary.environment import forced_sampling as fs

VOCAB = 248320


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    us = [fs.u_at("gumbel-check", p, "ckpt", r, t)
          for p, r, t in [(0, 0, 0), (3, 7, 1000), (31, 15, 4095), (12, 2, 77)]]
    keys = torch.tensor([fs.gumbel_key(u) for u in us], dtype=torch.int64, device=device)
    ids = torch.arange(VOCAB, dtype=torch.int64, device=device)
    bits = fs.gumbel_uniform_bits(keys, ids)
    noise = fs.gumbel_noise(keys, ids)
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(64, VOCAB, generator=gen) * 3).to(device)
    draws = [fs.u_at("gumbel-check", 1, "ckpt", 0, t) for t in range(64)]
    picks = fs.pick_rows(logits, draws, draw=fs.DRAW_GUMBEL_MAX)
    name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"

    def digest(tensor: torch.Tensor) -> str:
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:16]

    print(f"{name} torch {torch.__version__} bits {digest(bits)} G {digest(noise)} picks {digest(picks)}")


if __name__ == "__main__":
    main()
