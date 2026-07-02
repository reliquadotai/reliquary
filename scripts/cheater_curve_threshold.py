#!/usr/bin/env python3
"""Cheater-curve threshold calibration.

For each run (independent fresh training trajectory):
  - Load base Qwen3.5-2B as the "validator_model" (will be trained).
  - Keep a frozen second copy as the "miner_model" (= the lazy cheater
    who never updates).
  - At step 1..N:
      * Generate a training batch with validator_model on MATH prompts,
        synthetic rewards, run real GRPO train_step → advance validator.
      * Measurement: pick a fresh prompt, miner_model generates a rollout,
        build the GRAIL commit (sketches + claimed_logprobs from base),
        run the validator's full filter pipeline (sketch + LP + dist) using
        validator_model.
      * Record (run, step, lp_dev, sk_max, dist_median, dist_q10,
        completion_length).

Repeat for N_RUNS runs.

Output: CSV with one row per measurement. Stdout summary table aggregates
mean/p95/max per step across all runs so we can read where each filter
crosses its threshold.

Usage:

    python scripts/cheater_curve_threshold.py \\
        --n-runs 10 --n-steps 30 --measurements-per-step 2 \\
        --output /tmp/cheater_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from statistics import median
from typing import Any

import torch

from reliquary.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    LOGPROB_IS_EPS,
    PROOF_SKETCH_TOLERANCE_BASE,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
from reliquary.protocol.crypto import indices_from_root
from reliquary.protocol.grail_verifier import GRAILVerifier
from reliquary.protocol.submission import RolloutSubmission
from reliquary.protocol.tokens import encode_prompt
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size
from reliquary.shared.modeling import (
    first_eos_index,
    load_text_generation_model,
    load_tokenizer,
    resolve_eos_token_ids,
)
from reliquary.validator import training as reliquary_training
from reliquary.validator.batcher import ValidSubmission
from reliquary.validator.verifier import (
    evaluate_token_distribution,
    verify_logprobs_claim,
)


logger = logging.getLogger("cheater_curve")


def _load_base(repo: str, device: torch.device, *, frozen: bool = False):
    m = load_text_generation_model(
        repo,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(device)
    if frozen:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def _generate_rollouts(
    model, tokenizer, prompt: str, m: int, max_tokens: int,
    device: torch.device, *, with_logprobs: bool = False,
):
    """Return list of dicts {tokens, prompt_length, completion_length,
    optional token_logprobs}."""
    prompt_tokens = encode_prompt(tokenizer, prompt)
    prompt_length = len(prompt_tokens)
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    inp = torch.tensor([prompt_tokens] * m, device=device)
    attention_mask = torch.ones_like(inp)
    with torch.no_grad():
        kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": T_PROTO,
            "top_p": TOP_P_PROTO,
            "top_k": TOP_K_PROTO,
            "pad_token_id": tokenizer.pad_token_id,
            "attention_mask": attention_mask,
            "return_dict_in_generate": True,
            "output_scores": with_logprobs,
        }
        if eos_ids:
            kwargs["eos_token_id"] = sorted(eos_ids)
        gen = model.generate(inp, **kwargs)
    rollouts = []
    sequences = gen.sequences
    for i in range(m):
        full = sequences[i].tolist()
        comp = full[prompt_length:]
        first_eos = first_eos_index(comp, eos_ids)
        if first_eos is not None:
            comp = comp[: first_eos + 1]
        if len(comp) == 0:
            continue
        rec = {
            "tokens": prompt_tokens + comp,
            "prompt_length": prompt_length,
            "completion_length": len(comp),
        }
        if with_logprobs:
            scores = gen.scores
            lps = []
            for t in range(len(comp)):
                step_logits = scores[t][i].float()
                lp = torch.log_softmax(step_logits, dim=-1)[comp[t]].item()
                lps.append(lp)
            rec["token_logprobs"] = lps
        rollouts.append(rec)
    return rollouts


def _miner_commit(model, randomness: str, all_tokens: list[int],
                 prompt_length: int):
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)
    inp = torch.tensor([all_tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        h, logits = forward_single_layer(model, inp, None, LAYER_INDEX)
    h = h[0]
    commitments = verifier.create_commitments_batch(h, r_vec)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    token_lps = [
        log_probs[i - 1, all_tokens[i]].item()
        for i in range(prompt_length, len(all_tokens))
    ]
    return {
        "tokens": all_tokens,
        "commitments": commitments,
        "proof_version": GRAIL_PROOF_VERSION,
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": len(all_tokens) - prompt_length,
            "token_logprobs": token_lps,
        },
    }


def _validator_check(model, commit: dict, randomness: str) -> dict:
    """Run sketch + LP + distribution filters; return metrics dict."""
    tokens = commit["tokens"]
    commitments = commit["commitments"]
    seq_len = len(tokens)

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    expected = min(CHALLENGE_K, seq_len)
    chal_idx = indices_from_root(tokens, randomness, seq_len, expected)

    inp = torch.tensor([tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        h, logits_b = forward_single_layer(model, inp, None, LAYER_INDEX)
    h = h[0]
    logits = logits_b[0].detach().to("cpu")

    sketch_diffs = []
    sketch_passed = 0
    for idx in chal_idx:
        if idx >= seq_len:
            continue
        ok, diag = verifier.verify_commitment(
            h[idx], commitments[idx], r_vec, seq_len, idx
        )
        sketch_diffs.append(diag["sketch_diff"])
        sketch_passed += int(ok)

    rollout = commit["rollout"]
    lp_ok, lp_dev = verify_logprobs_claim(
        tokens=tokens,
        prompt_length=rollout["prompt_length"],
        completion_length=rollout["completion_length"],
        claimed_logprobs=rollout["token_logprobs"],
        logits=logits,
        challenge_randomness=randomness,
    )
    dist_ok, dist_metrics = evaluate_token_distribution(
        tokens=tokens,
        prompt_length=rollout["prompt_length"],
        completion_length=rollout["completion_length"],
        logits=logits,
        temperature=T_PROTO,
    )
    return {
        "sketch_diff_med": median(sketch_diffs) if sketch_diffs else None,
        "sketch_diff_max": max(sketch_diffs) if sketch_diffs else None,
        "sketch_passed": sketch_passed,
        "sketch_checked": len(sketch_diffs),
        "lp_ok": bool(lp_ok),
        "lp_dev": float(lp_dev),
        "dist_ok": dist_ok,
        "dist_median": dist_metrics.get("median") if dist_metrics else None,
        "dist_q10": dist_metrics.get("q10") if dist_metrics else None,
    }


def _build_training_batch_from_rollouts(
    pool_groups: list[dict], group_indices: list[int],
):
    batch: list[ValidSubmission] = []
    for g in (pool_groups[i] for i in group_indices):
        rollouts = [
            RolloutSubmission(
                tokens=r["tokens"],
                reward=r["reward"],
                commit={
                    "tokens": r["tokens"],
                    "rollout": {
                        "prompt_length": r["prompt_length"],
                        "token_logprobs": r["token_logprobs"],
                    }
                },
                env_name="openmathinstruct",
            )
            for r in g["rollouts"]
        ]
        batch.append(
            ValidSubmission(
                hotkey="synthetic",
                prompt_idx=g["prompt_idx"],
                merkle_root_bytes=b"\x00" * 32,
                rollouts=rollouts,
            )
        )
    return batch


def _generate_training_pool(
    model, tokenizer, env, *, n_groups: int, m_rollouts: int,
    max_tokens: int, device: torch.device, prompt_offset: int = 0,
):
    """Generate a pool of training groups with synthetic rewards (half 1
    / half 0) so every group has non-zero reward std."""
    pool = []
    for g_idx in range(n_groups):
        problem = env.get_problem(prompt_offset + g_idx)
        prompt_tokens = encode_prompt(tokenizer, problem["prompt"])
        prompt_length = len(prompt_tokens)
        eos_ids = resolve_eos_token_ids(model, tokenizer)
        inp = torch.tensor([prompt_tokens] * m_rollouts, device=device)
        attention_mask = torch.ones_like(inp)
        with torch.no_grad():
            kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": T_PROTO,
                "top_p": TOP_P_PROTO,
                "top_k": TOP_K_PROTO,
                "pad_token_id": tokenizer.pad_token_id,
                "attention_mask": attention_mask,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if eos_ids:
                kwargs["eos_token_id"] = sorted(eos_ids)
            out = model.generate(inp, **kwargs)
        sequences = out.sequences
        scores = out.scores

        rollouts = []
        for i in range(m_rollouts):
            full = sequences[i].tolist()
            comp = full[prompt_length:]
            first_eos = first_eos_index(comp, eos_ids)
            if first_eos is not None:
                comp = comp[: first_eos + 1]
            if len(comp) == 0:
                continue
            lps = []
            for t in range(len(comp)):
                step_logits = scores[t][i].float()
                lp = torch.log_softmax(step_logits, dim=-1)[comp[t]].item()
                lps.append(lp)
            reward = 1.0 if len(rollouts) < m_rollouts // 2 else 0.0
            rollouts.append({
                "tokens": prompt_tokens + comp,
                "prompt_length": prompt_length,
                "token_logprobs": lps,
                "reward": reward,
            })
        if len(rollouts) < 2:
            continue
        # Force non-degenerate reward distribution (in case some rollouts dropped).
        if len({r["reward"] for r in rollouts}) < 2:
            for k, r in enumerate(rollouts):
                r["reward"] = 1.0 if k < len(rollouts) // 2 else 0.0
        pool.append({
            "rollouts": rollouts,
            "prompt_idx": prompt_offset + g_idx,
        })
    return pool


def _run_one(
    *, run_idx: int, n_steps: int, measurements_per_step: int,
    measure_max_tokens: int, train_max_tokens: int,
    train_pool_groups: int, train_b_batch: int,
    base_repo: str, device: torch.device, randomness: str, env, tokenizer,
    measure_prompt_offset: int, train_prompt_offset: int,
):
    logger.info("===== run %d / loading two model copies =====", run_idx)
    validator_model = _load_base(base_repo, device, frozen=False)
    miner_model = _load_base(base_repo, device, frozen=True)
    reliquary_training.reset_training_state()

    logger.info("run %d / generating training pool (%d groups)",
                run_idx, train_pool_groups)
    pool = _generate_training_pool(
        validator_model, tokenizer, env,
        n_groups=train_pool_groups, m_rollouts=4,
        max_tokens=train_max_tokens, device=device,
        prompt_offset=train_prompt_offset,
    )
    if len(pool) < train_b_batch:
        logger.error("Pool too small (%d < %d). Aborting run.",
                     len(pool), train_b_batch)
        del validator_model, miner_model
        torch.cuda.empty_cache()
        return []

    rng = random.Random(1000 + run_idx)

    rows = []
    # step=0 measurement: validator==miner==base, both untrained.
    rows.extend(_measure(
        run_idx=run_idx, step=0, miner_model=miner_model,
        validator_model=validator_model,
        env=env, tokenizer=tokenizer,
        randomness=randomness, n_meas=measurements_per_step,
        max_tokens=measure_max_tokens, device=device,
        prompt_offset=measure_prompt_offset,
    ))

    for step in range(1, n_steps + 1):
        idxs = rng.sample(range(len(pool)), train_b_batch)
        batch = _build_training_batch_from_rollouts(pool, idxs)
        reliquary_training.train_step(validator_model, [batch], ref_model=miner_model)

        rows.extend(_measure(
            run_idx=run_idx, step=step, miner_model=miner_model,
            validator_model=validator_model,
            env=env, tokenizer=tokenizer,
            randomness=randomness, n_meas=measurements_per_step,
            max_tokens=measure_max_tokens, device=device,
            prompt_offset=measure_prompt_offset + step * 17,
        ))
    del validator_model, miner_model
    reliquary_training.reset_training_state()
    torch.cuda.empty_cache()
    return rows


def _measure(*, run_idx, step, miner_model, validator_model, env, tokenizer,
            randomness, n_meas, max_tokens, device, prompt_offset):
    """Run n_meas measurements: miner (base) generates, validator filters."""
    rows = []
    for j in range(n_meas):
        problem = env.get_problem((prompt_offset + j) % len(env))
        # miner_model = frozen base
        miner_rollouts = _generate_rollouts(
            miner_model, tokenizer, problem["prompt"], 1,
            max_tokens, device,
        )
        if not miner_rollouts:
            continue
        ro = miner_rollouts[0]
        commit = _miner_commit(
            miner_model, randomness,
            ro["tokens"], ro["prompt_length"],
        )
        metrics = _validator_check(validator_model, commit, randomness)
        rows.append({
            "run": run_idx, "step": step, "meas_idx": j,
            "completion_length": ro["completion_length"],
            **metrics,
        })
        logger.info(
            "run=%d step=%d meas=%d sk_pass=%d/%d sk_max=%s lp_dev=%.6f"
            " dist_med=%s",
            run_idx, step, j,
            metrics["sketch_passed"], metrics["sketch_checked"],
            metrics["sketch_diff_max"], metrics["lp_dev"],
            f"{metrics['dist_median']:.3f}"
            if metrics['dist_median'] is not None else "n/a",
        )
    return rows


def _summarize(rows: list[dict]) -> None:
    print("\n=== summary by step (across runs) ===")
    print(f"thresholds: LP_EPS={LOGPROB_IS_EPS}  SK_BASE={PROOF_SKETCH_TOLERANCE_BASE}\n")
    print(f"{'step':>4} {'n':>4}"
          f" {'med(lp)':>10} {'p95(lp)':>10} {'max(lp)':>10}"
          f" {'med(sk)':>9} {'p95(sk)':>9} {'max(sk)':>9}"
          f" {'lp_catch%':>10} {'sk_catch%':>10}"
          f" {'med(distq10)':>13}")
    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)
    for step in sorted(by_step):
        rs = by_step[step]
        n = len(rs)
        lps = sorted(r["lp_dev"] for r in rs)
        sks = sorted((r["sketch_diff_max"] or 0) for r in rs)
        q10s = [r["dist_q10"] for r in rs
                if r["dist_q10"] is not None]
        lp_catch = 100.0 * sum(1 for r in rs if not r["lp_ok"]) / n
        sk_catch = 100.0 * sum(
            1 for r in rs
            if r["sketch_passed"] < r["sketch_checked"]
        ) / n
        print(f"{step:>4} {n:>4}"
              f" {median(lps):>10.5f}"
              f" {lps[int(0.95*(n-1))]:>10.5f}"
              f" {lps[-1]:>10.5f}"
              f" {median(sks):>9.0f}"
              f" {sks[int(0.95*(n-1))]:>9.0f}"
              f" {sks[-1]:>9.0f}"
              f" {lp_catch:>9.1f}% {sk_catch:>9.1f}%"
              f" {median(q10s) if q10s else float('nan'):>13.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--measurements-per-step", type=int, default=2)
    ap.add_argument("--measure-max-tokens", type=int, default=192)
    ap.add_argument("--train-max-tokens", type=int, default=192)
    ap.add_argument("--train-pool-groups", type=int, default=24)
    ap.add_argument("--train-b-batch", type=int, default=8)
    ap.add_argument("--randomness",
                    default=("c0ffee" * 11)[:64])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device("cuda:0")
    logger.info("Loading tokenizer + env")
    tokenizer = load_tokenizer(args.base_model)
    env = OpenMathInstructEnvironment()

    all_rows = []
    for run in range(args.n_runs):
        t0 = time.time()
        rows = _run_one(
            run_idx=run, n_steps=args.n_steps,
            measurements_per_step=args.measurements_per_step,
            measure_max_tokens=args.measure_max_tokens,
            train_max_tokens=args.train_max_tokens,
            train_pool_groups=args.train_pool_groups,
            train_b_batch=args.train_b_batch,
            base_repo=args.base_model,
            device=device, randomness=args.randomness,
            env=env, tokenizer=tokenizer,
            measure_prompt_offset=400 + run * 31,
            train_prompt_offset=run * 53,
        )
        all_rows.extend(rows)
        # Persist incrementally so partial results survive crashes.
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        logger.info("Run %d done in %.1fs (%d rows total)",
                    run, time.time() - t0, len(all_rows))

    logger.info("All runs done — wrote %d rows to %s",
                len(all_rows), args.output)
    _summarize(all_rows)


if __name__ == "__main__":
    main()
