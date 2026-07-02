#!/usr/bin/env python3
"""Offline test: would each candidate validator catch a base-model cheater?

Runs all four candidate checks on a single rollout pair without the
live-pipeline overhead (window timing, zone filter, network round-trips):

    1. GRAIL sketch     (what we already ship — known-weak on our testnet)
    2. Logprob IS-median (candidate port)
    3. Distribution q10  (candidate port)
    4. Termination       (candidate port)

Workflow per run:
    - Load the base model (= what the cheater uses).
    - Roll M rollouts on randomly-picked MATH prompts until one lands with
      base_model reward-sum in [1, M-1] (i.e. would pass the σ≥0.33 zone
      filter in the live batcher, so the sketch check would actually fire).
    - Capture base_model hidden states / logits / sketches / logprobs on
      the chosen rollout. This is exactly what a dumb cheater would submit.
    - Swap to the validator's published checkpoint; recompute the same
      quantities on the SAME token sequence.
    - Apply each of the four checks to the cheater-vs-validator pair.
    - Print PASS / FAIL / SKIP plus the scalar that drove the decision.

PASS on a check means "cheater slips through this check". FAIL means the
check would catch this cheater. SKIP = not enough data (typically too
short a completion).
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import statistics
from typing import Any

import numpy as np
import torch

from reliquary.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    PRIME_Q,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.environment import load_environment
from reliquary.protocol.crypto import indices_from_root, indices_from_root_in_range
from reliquary.protocol.grail_verifier import GRAILVerifier, adaptive_sketch_tolerance
from reliquary.protocol.tokens import encode_prompt
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size
from reliquary.shared.modeling import (
    first_eos_index,
    load_text_generation_model,
    load_tokenizer,
    resolve_eos_token_ids,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / env helpers
# ---------------------------------------------------------------------------

def load_model(source: str, revision: str | None = None) -> Any:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    if revision:
        kwargs["revision"] = revision
    return load_text_generation_model(source, **kwargs).to("cuda:0").eval()


def find_in_zone_rollout(
    base_model: Any,
    tokenizer: Any,
    env: Any,
    *,
    max_new_tokens: int,
    n_attempts: int,
) -> tuple[dict, list[int], list[int], float]:
    """Roll M_ROLLOUTS rollouts under base_model on random prompts until we
    land on one where sum(reward) ∈ [1, M-1] (would pass the zone filter).

    Returns (problem_dict, prompt_tokens, rollout_tokens, reward_of_chosen).
    """
    eos_ids = resolve_eos_token_ids(base_model, tokenizer)
    rng = np.random.default_rng()

    for attempt in range(n_attempts):
        idx = int(rng.integers(0, len(env)))
        problem = env.get_problem(idx)
        prompt_tokens = encode_prompt(tokenizer, problem["prompt"])
        input_ids = torch.tensor(
            [prompt_tokens] * M_ROLLOUTS, device="cuda:0",
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": T_PROTO,
                "top_p": TOP_P_PROTO,
                "top_k": TOP_K_PROTO,
                "pad_token_id": tokenizer.pad_token_id,
                "attention_mask": attention_mask,
            }
            if eos_ids:
                generate_kwargs["eos_token_id"] = sorted(eos_ids)
            out = base_model.generate(input_ids, **generate_kwargs)

        rewards: list[float] = []
        rollouts_tokens: list[list[int]] = []
        for i in range(M_ROLLOUTS):
            seq = out[i].tolist()
            gen = seq[len(prompt_tokens):]
            first_eos = first_eos_index(gen, eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            full = prompt_tokens + gen
            rollouts_tokens.append(full)
            rewards.append(env.compute_reward(problem, tokenizer.decode(gen)))

        k = int(sum(r > 0.5 for r in rewards))
        sigma = float(np.std(rewards))
        logger.info(
            "attempt %d: idx=%d  k=%d/%d  σ=%.3f",
            attempt, idx, k, M_ROLLOUTS, sigma,
        )
        if 1 <= k <= M_ROLLOUTS - 1:
            # Pick one of the rollouts — preferentially a "correct" one so
            # termination check is meaningful.
            chosen = next(
                (i for i, r in enumerate(rewards) if r > 0.5),
                0,
            )
            return problem, prompt_tokens, rollouts_tokens[chosen], rewards[chosen]

    raise RuntimeError(f"No in-zone rollout after {n_attempts} attempts")


# ---------------------------------------------------------------------------
# Per-model computation (sketches, logprobs, logits)
# ---------------------------------------------------------------------------

def compute_model_data(model: Any, tokens: list[int], randomness_hex: str) -> dict:
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    input_ids = torch.tensor([tokens], device="cuda:0")
    with torch.inference_mode():
        h_batch, logits_batch = forward_single_layer(model, input_ids, None, LAYER_INDEX)
    h = h_batch[0]
    logits = logits_batch[0]

    r_vec_dev = r_vec.to(h.device)
    commits = verifier.create_commitments_batch(h, r_vec_dev)
    sketches = [c["sketch"] for c in commits]

    log_probs_tensor = torch.log_softmax(logits.float(), dim=-1)
    token_lps: list[float] = [0.0]
    for i in range(1, len(tokens)):
        token_lps.append(float(log_probs_tensor[i - 1, tokens[i]].item()))

    return {
        "model": model,
        "sketches": sketches,
        "logits": logits.detach().to("cpu"),
        "token_logprobs": token_lps,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def sketch_mod_diff(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, PRIME_Q - d)


def check_sketch(cheater: dict, validator: dict, tokens: list[int], rand: str) -> tuple[bool, dict]:
    seq_len = len(tokens)
    idxs = indices_from_root(tokens, rand, seq_len, CHALLENGE_K)
    diffs: list[int] = []
    passes: list[bool] = []
    for i in idxs:
        d = sketch_mod_diff(cheater["sketches"][i], validator["sketches"][i])
        tol = adaptive_sketch_tolerance(i, seq_len)
        diffs.append(d)
        passes.append(d <= tol)
    return all(passes), {
        "mean_diff": round(statistics.mean(diffs), 1),
        "max_diff": max(diffs),
        "n_checked": len(diffs),
        "pass_fraction": round(sum(passes) / len(passes), 3),
    }


def check_logprob(
    cheater: dict, validator: dict, tokens: list[int], prompt_len: int, rand: str,
) -> tuple[bool | None, dict]:
    completion_len = len(tokens) - prompt_len
    if completion_len < CHALLENGE_K:
        return None, {"reason": "completion_too_short", "completion_len": completion_len}
    idxs = indices_from_root_in_range(tokens, rand, prompt_len, prompt_len + completion_len, CHALLENGE_K)
    devs: list[float] = []
    for abs_idx in idxs:
        miner_lp = cheater["token_logprobs"][abs_idx]
        val_lp = validator["token_logprobs"][abs_idx]
        devs.append(math.exp(abs(miner_lp - val_lp)) - 1.0)
    median_dev = float(statistics.median(devs))
    return median_dev <= 0.10, {
        "median_dev": round(median_dev, 4),
        "max_dev": round(max(devs), 4),
        "threshold": 0.10,
    }


def check_distribution(
    validator: dict, tokens: list[int], prompt_len: int,
) -> tuple[bool | None, dict]:
    completion_len = len(tokens) - prompt_len
    if completion_len < 30:
        return None, {"reason": "completion_too_short", "completion_len": completion_len}
    probs: list[float] = []
    logits = validator["logits"]
    for t in range(prompt_len, prompt_len + completion_len):
        if t == 0 or t - 1 >= logits.size(0) or t >= len(tokens):
            continue
        step = (logits[t - 1].float() / T_PROTO).softmax(dim=-1)
        probs.append(float(step[tokens[t]].item()))
    if len(probs) < 30:
        return None, {"reason": "too_few_probs"}
    x = np.asarray(probs, dtype=np.float64)
    median = float(np.median(x))
    q10 = float(np.quantile(x, 0.10))
    ok = median >= 0.30 and q10 >= 0.025
    return ok, {
        "mean": round(float(x.mean()), 4),
        "median": round(median, 4),
        "q10": round(q10, 4),
        "low_frac": round(float((x <= 0.10).mean()), 4),
        "high_frac": round(float((x >= 0.90).mean()), 4),
        "thresholds": {"median>=0.30": median >= 0.30, "q10>=0.025": q10 >= 0.025},
    }


def check_termination(
    validator: dict, tokens: list[int], tokenizer: Any, max_completion_length: int,
    prompt_len: int,
) -> tuple[bool, dict]:
    from reliquary.constants import MIN_EOS_PROBABILITY

    completion_len = len(tokens) - prompt_len
    if completion_len > max_completion_length:
        return False, {"reason": "exceeds_max"}
    if completion_len == max_completion_length:
        return True, {"reason": "max_length_natural"}

    eos_ids = resolve_eos_token_ids(validator.get("model"), tokenizer)
    if not eos_ids or int(tokens[-1]) not in eos_ids:
        return False, {"reason": "not_eos"}

    logits = validator["logits"]
    probs = logits[-2].float().softmax(dim=-1)
    eos_idx = torch.tensor(sorted(eos_ids), device=probs.device, dtype=torch.long)
    p_eos = float(probs[eos_idx].sum().item())
    return p_eos >= MIN_EOS_PROBABILITY, {
        "p_eos": round(p_eos, 4),
        "threshold": MIN_EOS_PROBABILITY,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--validator-repo", default="R0mAI/reliquary-math")
    p.add_argument("--validator-revision", required=True,
                   help="HF commit SHA of the validator's current published ckpt")
    p.add_argument("--randomness", default="0123456789abcdef" * 4)
    p.add_argument("--env", default="openmathinstruct")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--n-attempts", type=int, default=20,
                   help="Max prompts to try before giving up on finding an in-zone one")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    # We need MIN_EOS_PROBABILITY — if not yet in constants, define a default.
    import reliquary.constants as C
    if not hasattr(C, "MIN_EOS_PROBABILITY"):
        C.MIN_EOS_PROBABILITY = 0.02

    logger.info("Loading tokenizer + env")
    tokenizer = load_tokenizer(args.base_model)
    env = load_environment(args.env)

    logger.info("Loading BASE model (cheater) ...")
    base_model = load_model(args.base_model)

    logger.info("Searching for an in-zone rollout under base model ...")
    problem, prompt_tokens, rollout_tokens, reward = find_in_zone_rollout(
        base_model, tokenizer, env,
        max_new_tokens=args.max_tokens,
        n_attempts=args.n_attempts,
    )
    prompt_len = len(prompt_tokens)
    completion_len = len(rollout_tokens) - prompt_len
    logger.info("Selected rollout: prompt_len=%d completion_len=%d reward=%.2f",
                prompt_len, completion_len, reward)

    logger.info("Computing cheater (base) sketches / logprobs ...")
    cheater = compute_model_data(base_model, rollout_tokens, args.randomness)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading validator model (%s @ %s) ...",
                args.validator_repo, args.validator_revision)
    val_model = load_model(args.validator_repo, revision=args.validator_revision)

    logger.info("Computing validator-side data on same tokens ...")
    validator = compute_model_data(val_model, rollout_tokens, args.randomness)

    # Run the four checks
    print()
    print("=" * 78)
    print("CHEATER DETECTION — summary (PASS = cheater slips through)")
    print("=" * 78)
    sketch_ok, sketch_info = check_sketch(cheater, validator, rollout_tokens, args.randomness)
    print(f"SKETCH       : {'PASS' if sketch_ok else 'FAIL'}   {sketch_info}")

    lp_ok, lp_info = check_logprob(cheater, validator, rollout_tokens, prompt_len, args.randomness)
    lp_s = "SKIP" if lp_ok is None else ("PASS" if lp_ok else "FAIL")
    print(f"LOGPROB      : {lp_s}   {lp_info}")

    dist_ok, dist_info = check_distribution(validator, rollout_tokens, prompt_len)
    dist_s = "SKIP" if dist_ok is None else ("PASS" if dist_ok else "FAIL")
    print(f"DISTRIBUTION : {dist_s}   {dist_info}")

    term_ok, term_info = check_termination(
        validator, rollout_tokens, tokenizer, args.max_tokens, prompt_len,
    )
    print(f"TERMINATION  : {'PASS' if term_ok else 'FAIL'}   {term_info}")
    print("=" * 78)


if __name__ == "__main__":
    main()
