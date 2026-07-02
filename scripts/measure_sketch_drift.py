#!/usr/bin/env python3
"""Measure GRAIL sketch drift between published checkpoints.

For a fixed prompt, compute the GRAIL sketches at every token position using
each checkpoint's model weights. Then build the pairwise ``sketch_diff``
matrix and compare it against ``PROOF_SKETCH_TOLERANCE_BASE = 6000``.

If the drift between two adjacent checkpoints is well below the tolerance,
a miner staying on the older checkpoint would NOT be caught by the sketch
check — which is the weakness the live-testnet cheater test surfaced.

Usage on a GPU box with ``reliquary`` installed:
    python scripts/measure_sketch_drift.py \
        --hf-repo R0mAI/reliquary-math \
        --base-model Qwen/Qwen3.5-2B \
        --revisions de8490f58de7d256ea9d35e0d7715... 1e81f2751fc4... \
        --output /tmp/sketch_drift.json

The script loads models one at a time (keeps VRAM bounded) and caches
hidden states / sketches for pairwise comparison at the end.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from typing import Any

import torch

from reliquary.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    LAYER_INDEX,
    PRIME_Q,
    PROOF_NUM_BUCKETS,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
    PROOF_TOPK,
)
from reliquary.protocol.grail_verifier import (
    GRAILVerifier,
    adaptive_sketch_tolerance,
)
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

logger = logging.getLogger(__name__)


def load_model(source: str, revision: str | None = None):
    """Load a HF model (repo id or local path)."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    if revision:
        kwargs["revision"] = revision
    return (
        load_text_generation_model(source, **kwargs)
        .to("cuda:0")
        .eval()
    )


def compute_all_sketches(
    model,
    input_ids: torch.Tensor,
    randomness_hex: str,
) -> list[int]:
    """Forward pass the prompt through *model* and compute one sketch per token."""
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    with torch.inference_mode():
        h_batch, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)
    h_layer = h_batch[0]  # (seq_len, hidden_dim)

    r_vec_dev = r_vec.to(h_layer.device)
    commitments = verifier.create_commitments_batch(h_layer, r_vec_dev)
    return [c["sketch"] for c in commitments]


def sketch_mod_diff(a: int, b: int) -> int:
    """Modular distance in the sketch field (min(|a-b|, P-|a-b|))."""
    d = abs(a - b)
    return min(d, PRIME_Q - d)


def pairwise_stats(
    sketches_a: list[int],
    sketches_b: list[int],
    seq_len: int,
) -> dict:
    """Return diff statistics + per-position pass/fail vs tolerance."""
    assert len(sketches_a) == len(sketches_b) == seq_len
    diffs = [sketch_mod_diff(a, b) for a, b in zip(sketches_a, sketches_b)]
    tolerances = [adaptive_sketch_tolerance(p, seq_len) for p in range(seq_len)]
    passes = [d <= t for d, t in zip(diffs, tolerances)]

    # Summary
    n = len(diffs)
    sorted_diffs = sorted(diffs)
    mean = sum(diffs) / n
    p50 = sorted_diffs[n // 2]
    p95 = sorted_diffs[min(n - 1, int(n * 0.95))]
    maxv = sorted_diffs[-1]
    pass_frac = sum(passes) / n

    return {
        "n_positions": n,
        "mean_diff": round(mean, 1),
        "p50_diff": p50,
        "p95_diff": p95,
        "max_diff": maxv,
        "tolerance_at_last_pos": tolerances[-1],
        "pass_fraction": round(pass_frac, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--hf-repo", required=True, help="Published validator repo")
    parser.add_argument(
        "--revisions",
        nargs="+",
        required=True,
        help="HF commit SHAs (full or abbreviated) to compare against base",
    )
    parser.add_argument(
        "--prompt",
        default="Solve this problem step by step: if 3x + 5 = 20, what is x?",
        help="Fixed prompt used for all forward passes",
    )
    parser.add_argument(
        "--randomness",
        default="a" * 64,
        help="Randomness hex for r_vec (64 chars). Fixed = all runs use same sketch basis.",
    )
    parser.add_argument("--output", default="/tmp/sketch_drift.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Tokenize prompt once (tokenizer doesn't change across checkpoints).
    tok = load_tokenizer(args.base_model)
    tokens = tok.encode(args.prompt, add_special_tokens=False)
    seq_len = len(tokens)
    input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")

    logger.info(
        "Prompt tokenised to seq_len=%d. Tolerance at last pos = %d (base %d + growth %.1f × √%d).",
        seq_len,
        adaptive_sketch_tolerance(seq_len - 1, seq_len),
        PROOF_SKETCH_TOLERANCE_BASE,
        PROOF_SKETCH_TOLERANCE_GROWTH,
        seq_len - 1,
    )

    # Build the list of (label, source, revision) tuples.
    checkpoints: list[tuple[str, str, str | None]] = [
        ("base", args.base_model, None),
    ]
    for rev in args.revisions:
        checkpoints.append((f"rev_{rev[:8]}", args.hf_repo, rev))

    # Compute sketches for each checkpoint. Load / compute / free.
    all_sketches: dict[str, list[int]] = {}
    for label, src, rev in checkpoints:
        logger.info("Loading %s (%s @ %s)...", label, src, rev or "main")
        model = load_model(src, rev)
        logger.info("Computing %d sketches...", seq_len)
        all_sketches[label] = compute_all_sketches(model, input_ids, args.randomness)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("%s done. Example sketches[:3] = %s", label, all_sketches[label][:3])

    # Pairwise drift matrix.
    labels = list(all_sketches.keys())
    matrix: dict[str, dict[str, Any]] = {}
    for a in labels:
        matrix[a] = {}
        for b in labels:
            if a >= b:
                # Only compute upper triangle; mirror for symmetry.
                continue
            stats = pairwise_stats(all_sketches[a], all_sketches[b], seq_len)
            matrix[a][b] = stats
            logger.info(
                "%s ↔ %s  mean=%d  p95=%d  max=%d  pass_frac=%.3f  (tol at last pos = %d)",
                a, b,
                stats["mean_diff"], stats["p95_diff"], stats["max_diff"],
                stats["pass_fraction"], stats["tolerance_at_last_pos"],
            )

    output = {
        "prompt": args.prompt,
        "randomness": args.randomness,
        "seq_len": seq_len,
        "tolerance_base": PROOF_SKETCH_TOLERANCE_BASE,
        "tolerance_growth": PROOF_SKETCH_TOLERANCE_GROWTH,
        "challenge_k": CHALLENGE_K,
        "proof_topk": PROOF_TOPK,
        "proof_num_buckets": PROOF_NUM_BUCKETS,
        "layer_index": LAYER_INDEX,
        "checkpoints": [
            {"label": l, "source": s, "revision": r}
            for l, s, r in checkpoints
        ],
        "pairwise": matrix,
        "sketches_by_checkpoint": all_sketches,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote %s (%d bytes).", args.output, len(json.dumps(output)))


if __name__ == "__main__":
    main()
