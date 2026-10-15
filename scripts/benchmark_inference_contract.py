#!/usr/bin/env python3
"""Benchmark cached forced-seed generation against validator teacher forcing.

Run this script in a fresh process for each dependency/kernel profile. It emits
one JSON artifact containing the exact runtime fingerprint and per-rollout CDF,
termination, repetition, and throughput diagnostics. It does not change gates.

Service seeds (decision I, rulings R14/R16). ``--seed-source pool`` qualifies the
forced draw of the v2 service path. For every prompt it builds the public seed pool
exactly as the validator does (``SeedPool.from_contract`` on the order's contract, the
window beacon and the pool epoch): ``2 x M`` seeds, identical for every miner, renewed
every window, whose draw depends only on (pool digest, seed index, token position).
A group is ANY ``M`` of the pool seeds, so the operator picks the subset policy the
miners will really run:

* ``--pool-subset first``: generate seeds ``0..M-1`` and submit them;
* ``--pool-subset all``: generate all ``2M`` seeds (in two calls of ``M``, like the
  miner hook) and submit the ``M`` picked by ``--pool-pick``
  (``lowest-agreement`` is the adversarial case, ``highest-agreement`` the lenient one).

Example (GPU box, one fresh process per profile)::

    python scripts/benchmark_inference_contract.py --model MODEL --model-revision REV \
        --checkpoint-hash HASH --profile-label teutonic-h100 --replicate 0 \
        --prompts-jsonl prompts.jsonl --max-new-tokens 4096 --output report.json \
        --seed-source pool --contract order.json --pool-env ENV_ID \
        --randomness BEACON_HEX --pool-epoch WINDOW --pool-subset all \
        --pool-pick lowest-agreement

Scope notes: the top-level ``summary`` of the report (and the legacy per-rollout statistics)
covers EVERY generated rollout, including the non-submitted half in ``--pool-subset all``;
the ``forced_seed`` section covers only the submitted group of each prompt. In pool mode
``--randomness``, ``--pool-epoch`` and ``--checkpoint-hash`` are required (a default would
silently build a pool no window uses), and ``--pool-pick`` is only valid with
``--pool-subset all``.

The JSON report has a ``forced_seed`` section: per rollout (``n_stochastic``,
``n_exact_match``, agreement) and per group, scored with the validator's own
``_forced_seed_verdict`` / ``_forced_seed_rollout_reject`` at the floors of
``reliquary.constants``. The sha256 of that file is the runtime qualification's
``forced_seed_report_sha256``. Qualify on production hardware before the run;
nothing here runs a model unless you launch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


POOL_SUBSETS = ("first", "all")
POOL_PICKS = ("first", "last", "lowest-agreement", "highest-agreement")


def load_contract(path: Path):
    """The order's service contract from a JSON file (canonicalised like the validator's)."""
    from reliquary.protocol.service_contract import ServiceContract

    return ServiceContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_pool(args, *, prompt_idx: int):
    """The pool the validator builds for (contract, env, prompt, checkpoint, epoch, beacon)."""
    from reliquary.protocol.seed_pool import SeedPool

    contract = args.contract if not isinstance(args.contract, (str, Path)) else load_contract(args.contract)
    pool = SeedPool.from_contract(
        contract,
        environment=args.pool_env,
        prompt_idx=int(prompt_idx),
        checkpoint_hash=args.checkpoint_hash,
        pool_epoch=int(args.pool_epoch),
        randomness=args.randomness,
    )
    # R16 (pools renew every window) is enforced by the contract itself: v2 refuses
    # renewal_windows != 1 at parse time, so no pool reaches here with another value.
    return pool


def uniform_source(args, *, prompt_idx: int, pool=None):
    """``u(key, offset)``: the public uniform the validator recomputes for this run.

    Window source: ``key`` is the rollout index (``u_at``). Pool source: ``key`` is the
    SEED INDEX (``SeedPool.uniform``), whatever rank the seed has in the submitted group.
    """
    if args.seed_source == "pool":
        if pool is None:
            pool = build_pool(args, prompt_idx=prompt_idx)
        return pool.uniform
    from reliquary.environment.forced_sampling import u_at

    return lambda key, offset: u_at(
        args.randomness, int(prompt_idx), args.checkpoint_hash, int(key), int(offset),
    )


def seeds_to_generate(pool, subset: str) -> list[list[int]]:
    """Seed indices to generate, in calls of ``group_size`` (what a miner's calls look like)."""
    if subset not in POOL_SUBSETS:
        raise ValueError(f"pool subset must be one of {POOL_SUBSETS}")
    m = pool.group_size
    return [list(range(m))] if subset == "first" else [list(range(m)), list(range(m, pool.pool_seeds))]


def choose_seeds(pool, pick: str, agreement: dict[int, float]) -> tuple[int, ...]:
    """The ``group_size`` seeds submitted out of the generated ones, ascending."""
    if pick not in POOL_PICKS:
        raise ValueError(f"pool pick must be one of {POOL_PICKS}")
    seeds = sorted(agreement)
    if len(seeds) < pool.group_size:
        raise ValueError("fewer generated seeds than a group")
    if pick == "first":
        chosen = seeds[: pool.group_size]
    elif pick == "last":
        chosen = seeds[-pool.group_size:]
    else:
        ranked = sorted(seeds, key=lambda s: (agreement[s], s), reverse=(pick == "highest-agreement"))
        chosen = ranked[: pool.group_size]
    return tuple(sorted(chosen))


def score_group(rows: list[dict]) -> dict:
    """Score one group of rows with the validator's own forced-seed functions.

    Measured: the group ratio verdict, the per-rollout verdict and the service-path CDF
    hard-mismatch reject (``n_hard_mismatch`` summed over the group > 0, batcher.py
    ``cdf_reject``). NOT measured: the terminal-pick and EOS-padding checks; they need the
    submitted proofs, which this harness does not build.
    """
    from reliquary.constants import (
        FORCED_SEED_CONSISTENCY_FLOOR,
        FORCED_SEED_MIN_STOCH_POSITIONS,
        FORCED_SEED_ROLLOUT_FLOOR,
        FORCED_SEED_ROLLOUT_MIN_STOCH,
    )
    from reliquary.validator.batcher import (
        _forced_seed_rollout_reject,
        _forced_seed_verdict,
    )

    per_rollout = [(int(r["n_stochastic"]), int(r["n_exact_match"])) for r in rows]
    n_stoch = sum(n for n, _ in per_rollout)
    n_match = sum(m for _, m in per_rollout)
    group_reject = _forced_seed_verdict(n_stoch, n_match, True)
    rollout_reject = _forced_seed_rollout_reject(per_rollout, True)
    hard = sum(int(r["n_hard_mismatch"]) for r in rows)
    cdf_reject = hard > 0
    return {
        "n_hard_mismatch": hard,
        "cdf_rejected": cdf_reject,
        "n_stochastic": n_stoch,
        "n_exact_match": n_match,
        "agreement": n_match / n_stoch if n_stoch else None,
        "group_floor": FORCED_SEED_CONSISTENCY_FLOOR,
        "group_min_stochastic": FORCED_SEED_MIN_STOCH_POSITIONS,
        "group_abstained": n_stoch < FORCED_SEED_MIN_STOCH_POSITIONS,
        "group_rejected": bool(group_reject),
        "rollout_floor": FORCED_SEED_ROLLOUT_FLOOR,
        "rollout_min_stochastic": FORCED_SEED_ROLLOUT_MIN_STOCH,
        "rollouts_below_floor": sum(
            1 for n, m in per_rollout
            if n >= FORCED_SEED_ROLLOUT_MIN_STOCH and m / n < FORCED_SEED_ROLLOUT_FLOOR
        ),
        "rollout_rejected": bool(rollout_reject),
        "accepted": not (group_reject or rollout_reject or cdf_reject),
    }


def rollout_agreement(row: dict) -> float:
    n = int(row["n_stochastic"])
    return int(row["n_exact_match"]) / n if n else 1.0


def forced_seed_summary(groups: list[dict]) -> dict:
    """Pass rates over groups (an abstaining group is accepted, as on the validator)."""
    rows = [r for g in groups for r in g["rollouts"]]
    n = len(groups)
    from reliquary.constants import (
        FORCED_SEED_CONSISTENCY_FLOOR,
        FORCED_SEED_ROLLOUT_FLOOR,
        FORCED_SEED_ROLLOUT_MIN_STOCH,
    )

    judged = [r for r in rows if int(r["n_stochastic"]) >= FORCED_SEED_ROLLOUT_MIN_STOCH]
    rollouts_ok = [r for r in judged if rollout_agreement(r) >= FORCED_SEED_ROLLOUT_FLOOR]
    return {
        "groups": n,
        "group_floor": FORCED_SEED_CONSISTENCY_FLOOR,
        "rollout_floor": FORCED_SEED_ROLLOUT_FLOOR,
        "group_acceptance_rate": sum(g["score"]["accepted"] for g in groups) / n if n else None,
        "group_floor_pass_rate": (
            sum(not g["score"]["group_rejected"] for g in groups) / n if n else None
        ),
        "cdf_hard_mismatch_rate": (
            sum(g["score"]["cdf_rejected"] for g in groups) / n if n else None
        ),
        "rollouts_judged": len(judged),
        "rollout_floor_pass_rate": len(rollouts_ok) / len(judged) if judged else None,
        "mean_group_agreement": (
            sum(g["score"]["agreement"] for g in groups if g["score"]["agreement"] is not None)
            / max(1, sum(g["score"]["agreement"] is not None for g in groups))
            if n else None
        ),
    }


def build_group_record(prompt_idx: int, prompt_rows: list[dict], pool, subset: str, pick: str | None) -> dict:
    """The report record of one prompt: the submitted group, scored.

    Pool rows are keyed on ``seed_index``; ``group_rank`` is the rank of a submitted
    rollout inside the submitted (ascending) group.
    """
    if pool is not None:
        agreement = {int(r["seed_index"]): rollout_agreement(r) for r in prompt_rows}
        if subset == "first":
            chosen = tuple(sorted(agreement))
        else:
            chosen = choose_seeds(pool, pick, agreement)
        selection = pool.selection(chosen)
        chosen_set = set(chosen)
        group_rows = sorted(
            (r for r in prompt_rows if r["seed_index"] in chosen_set),
            key=lambda r: r["seed_index"],
        )
        group_rows = [dict(r, group_rank=rank) for rank, r in enumerate(group_rows)]
        group_meta = {
            "pool_sha256": pool.sha256,
            "pool": pool.to_dict(),
            "generated_seeds": sorted(agreement),
            "chosen_seeds": list(chosen),
            "selection_sha256": selection.sha256,
        }
    else:
        group_rows = prompt_rows
        group_meta = {}
    return {
        "prompt_idx": prompt_idx,
        **group_meta,
        "score": score_group(group_rows),
        "rollouts": [
            {
                key: row[key]
                for key in ("rollout_idx", "seed_index", "group_rank", "n_stochastic",
                            "n_exact_match", "n_hard_mismatch", "completion_length")
                if key in row
            } | {"agreement": rollout_agreement(row)}
            for row in group_rows
        ],
        "generated_rollouts": len(prompt_rows),
    }


def pool_arg_errors(args) -> list[str]:
    """Problems with the pool-mode arguments (empty when fine)."""
    errors = []
    if args.seed_source != "pool":
        if getattr(args, "pool_pick", None) is not None:
            errors.append("--pool-pick is only valid with --seed-source pool (it would be silently ignored)")
        return errors
    if args.contract is None or not args.pool_env:
        errors.append("--seed-source pool needs --contract and --pool-env")
    for flag, value in (("--randomness", args.randomness), ("--pool-epoch", args.pool_epoch),
                        ("--checkpoint-hash", args.checkpoint_hash)):
        if value is None or value == "":
            errors.append(f"{flag} is required in pool mode (a default builds a pool no window uses)")
    if args.pool_subset == "first" and args.pool_pick is not None:
        errors.append("--pool-pick needs --pool-subset all (subset first submits seeds 0..M-1)")
    return errors


def _dtype(torch, name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_prompts(path: Path | None, direct: list[str]) -> list[dict]:
    prompts = [
        {"prompt": prompt, "prompt_idx": index}
        for index, prompt in enumerate(direct)
    ]
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("prompt") if isinstance(row, dict) else row
            if not isinstance(value, str) or not value:
                raise ValueError("each JSONL row must contain a non-empty prompt")
            prompt_idx = row.get("prompt_idx") if isinstance(row, dict) else None
            if prompt_idx is None:
                prompt_idx = len(prompts)
            prompt_row = {"prompt": value, "prompt_idx": int(prompt_idx)}
            if isinstance(row, dict) and row.get("ground_truth") is not None:
                prompt_row["ground_truth"] = str(row["ground_truth"])
            prompts.append(prompt_row)
    if not prompts:
        raise ValueError("provide --prompt or --prompts-jsonl")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Immutable Hugging Face revision actually loaded for model/tokenizer.",
    )
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--bft-thinking-budget",
        type=int,
        default=0,
        help="Enable the real two-phase BFT path with this phase-1 budget.",
    )
    parser.add_argument(
        "--bft-answer-budget",
        type=int,
        default=0,
        help="Phase-2 answer budget when --bft-thinking-budget is enabled.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--verification-dtype",
        choices=("bfloat16", "float16", "float32"),
        help="Load a separate validator-style model at this dtype.",
    )
    parser.add_argument(
        "--verification-attn-implementation",
        help="Attention implementation for the separate verification model.",
    )
    parser.add_argument("--generation-use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic-algorithms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--randomness", help="Window beacon hex (required in pool mode).")
    parser.add_argument("--hotkey", default="benchmark-hotkey")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--seed-source", choices=("window", "pool"), default="window",
        help="window: legacy per-window u_at draw. pool: v2 service public seed pool.",
    )
    parser.add_argument("--contract", type=Path, help="service-contract/v2 JSON (pool mode).")
    parser.add_argument("--pool-env", help="Environment id in the contract (pool mode).")
    parser.add_argument("--pool-epoch", type=int, help="Pool epoch = the window (required in pool mode).")
    parser.add_argument(
        "--pool-subset", choices=POOL_SUBSETS, default="first",
        help="first: generate and submit seeds 0..M-1. all: generate all 2M seeds, submit M.",
    )
    parser.add_argument(
        "--pool-pick", choices=POOL_PICKS,
        help="With --pool-subset all, which M of the 2M seeds are submitted (default lowest-agreement).",
    )
    parser.add_argument(
        "--required-group-acceptance", type=float,
        help="Optional: write qualified=true only if the group acceptance rate reaches this.",
    )
    args = parser.parse_args()
    errors = pool_arg_errors(args)
    if errors:
        parser.error("; ".join(errors))
    if args.seed_source == "pool":
        args.contract = load_contract(args.contract)
        if args.pool_subset == "all" and args.pool_pick is None:
            args.pool_pick = "lowest-agreement"
    elif args.randomness is None:
        args.randomness = "42" * 32

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive")
    if (args.bft_thinking_budget > 0) != (args.bft_answer_budget > 0):
        raise ValueError("both BFT budgets must be positive, or both must be zero")

    import torch
    from transformers import AutoTokenizer

    from reliquary.constants import (
        FORCED_SEED_CDF_BOUNDARY_EPSILON,
        FORCED_SEED_STOCHASTIC_MAXPROB,
        LAYER_INDEX,
        T_PROTO,
        TOP_K_PROTO,
        TOP_P_PROTO,
    )
    from reliquary.environment.forced_sampling import seed_consistency_diagnostics
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    from reliquary.miner.forced_seed_sampler import (
        ForcedSeedLogitsProcessor,
        forced_seed_generate_kwargs,
    )
    from reliquary.miner.engine import _bft_assemble_rollouts
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.modeling import (
        first_eos_index,
        force_close_token_ids,
        load_text_generation_model,
        resolve_eos_token_ids,
        think_close_token_ids,
    )
    from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint
    from reliquary.validator.rollout_telemetry import (
        classify_bft_termination,
        token_degeneracy_metrics,
    )

    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    model = load_text_generation_model(
        args.model,
        revision=args.model_revision,
        torch_dtype=_dtype(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    verification_dtype = args.verification_dtype or args.dtype
    verification_attention = (
        args.verification_attn_implementation or args.attn_implementation
    )
    if (
        verification_dtype == args.dtype
        and verification_attention == args.attn_implementation
    ):
        verification_model = model
    else:
        verification_model = load_text_generation_model(
            args.model,
            revision=args.model_revision,
            torch_dtype=_dtype(torch, verification_dtype),
            attn_implementation=verification_attention,
        ).to(device).eval()
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = min(eos_ids) if eos_ids else 0

    prompts = _load_prompts(args.prompts_jsonl, args.prompt)
    prompts_sha256 = None
    if args.prompts_jsonl is not None:
        prompts_sha256 = hashlib.sha256(args.prompts_jsonl.read_bytes()).hexdigest()
    rows: list[dict] = []
    group_records: list[dict] = []
    generated_tokens = 0
    generation_seconds_total = 0.0
    teacher_force_seconds_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for prompt_row in prompts:
        prompt = prompt_row["prompt"]
        prompt_idx = int(prompt_row["prompt_idx"])
        ground_truth = prompt_row.get("ground_truth")
        prompt_tokens = encode_prompt(tokenizer, prompt)
        prompt_length = len(prompt_tokens)
        pool = (
            build_pool(args, prompt_idx=prompt_idx)
            if args.seed_source == "pool" else None
        )
        u = uniform_source(args, prompt_idx=prompt_idx, pool=pool)
        chunks = (
            seeds_to_generate(pool, args.pool_subset)
            if pool is not None else [list(range(args.batch_size))]
        )
        prompt_rows: list[dict] = []
        for chunk in chunks:
            n_rows = len(chunk)
            input_ids = torch.tensor(
                [prompt_tokens] * n_rows,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            processor = ForcedSeedLogitsProcessor(
                randomness=args.randomness,
                hotkey=args.hotkey,
                prompt_idx=prompt_idx,
                checkpoint_hash=args.checkpoint_hash,
                rollout_indices=list(range(n_rows)),
                base_offsets=[0] * n_rows,
                start_len=prompt_length,
                seed_pool=pool,
                seeds=list(chunk) if pool is not None else None,
            )
            generation_args = {
                "attention_mask": attention_mask,
                "max_new_tokens": (
                    args.bft_thinking_budget
                    if args.bft_thinking_budget > 0
                    else args.max_new_tokens
                ),
                "pad_token_id": pad_token_id,
                "use_cache": args.generation_use_cache,
            }
            if eos_ids:
                generation_args["eos_token_id"] = sorted(eos_ids)
            batch_started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    **forced_seed_generate_kwargs(generation_args, processor),
                )
                if args.bft_thinking_budget > 0:
                    phase2_kwargs = {
                        "pad_token_id": pad_token_id,
                        "use_cache": args.generation_use_cache,
                    }
                    if eos_ids:
                        phase2_kwargs["eos_token_id"] = sorted(eos_ids)
                    generated_rows = _bft_assemble_rollouts(
                        model=model,
                        phase1_tensor=generated,
                        prompt_tokens=prompt_tokens,
                        think_close_ids=set(think_close_token_ids(tokenizer)),
                        force_ids=force_close_token_ids(tokenizer),
                        eos_ids=eos_ids,
                        answer_budget=args.bft_answer_budget,
                        randomness=args.randomness,
                        hotkey=args.hotkey,
                        prompt_idx=prompt_idx,
                        checkpoint_hash=args.checkpoint_hash,
                        gen_kwargs=phase2_kwargs,
                        seed_pool=pool,
                        seeds=list(chunk) if pool is not None else None,
                    )
                else:
                    generated_rows = [
                        {
                            "tokens": generated[index].tolist(),
                            "prompt_length": prompt_length,
                            "forced": False,
                        }
                        for index in range(n_rows)
                    ]
            if device.type == "cuda":
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - batch_started
            generation_seconds_total += generation_seconds

            for rollout_idx in range(n_rows):
                draw_key = chunk[rollout_idx]
                generation = generated_rows[rollout_idx]
                completion = generation["tokens"][prompt_length:]
                eos_offset = first_eos_index(completion, eos_ids)
                if eos_offset is not None:
                    completion = completion[:eos_offset + 1]
                sequence = prompt_tokens + completion
                generated_tokens += len(completion)
                full_ids = torch.tensor([sequence], dtype=torch.long, device=device)
                verify_started = time.perf_counter()
                with torch.no_grad():
                    _, logits = forward_single_layer(
                        verification_model, full_ids, None, LAYER_INDEX,
                    )
                logits_slice = logits[
                    0,
                    prompt_length - 1:prompt_length + len(completion) - 1,
                ]
                all_uniforms = [
                    u(draw_key, offset) for offset in range(len(completion))
                ]
                force_span = generation.get("force_span")
                if force_span is None:
                    sampled_offsets = list(range(len(completion)))
                else:
                    force_start = int(force_span[0]) - prompt_length
                    force_end = int(force_span[1]) - prompt_length
                    sampled_offsets = [
                        offset for offset in range(len(completion))
                        if not (force_start <= offset < force_end)
                    ]
                selected_logits = logits_slice[sampled_offsets]
                selected_tokens = [completion[offset] for offset in sampled_offsets]
                uniforms = [all_uniforms[offset] for offset in sampled_offsets]
                diagnostics = seed_consistency_diagnostics(
                    selected_logits,
                    selected_tokens,
                    uniforms,
                    t=T_PROTO,
                    top_k=TOP_K_PROTO,
                    top_p=TOP_P_PROTO,
                    stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB,
                    boundary_epsilon=FORCED_SEED_CDF_BOUNDARY_EPSILON,
                    position_offsets=sampled_offsets,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                verify_seconds = time.perf_counter() - verify_started
                teacher_force_seconds_total += verify_seconds
                row = {
                    "prompt_idx": prompt_idx,
                    # pool mode: unique across chunks (the seed index); window mode: batch row
                    "rollout_idx": draw_key if pool is not None else rollout_idx,
                    **({"seed_index": draw_key} if pool is not None else {}),
                    "prompt_length": prompt_length,
                    "completion_length": len(completion),
                    "completion_sha256": hashlib.sha256(
                        b"".join(
                            int(token).to_bytes(4, "big", signed=False)
                            for token in completion
                        )
                    ).hexdigest(),
                    "ended_eos": eos_offset is not None,
                    "forced": bool(generation.get("forced", False)),
                    "force_span_length": (
                        int(force_span[1]) - int(force_span[0])
                        if force_span is not None
                        else 0
                    ),
                    "bft_termination_path": (
                        classify_bft_termination(
                            sequence,
                            prompt_length=prompt_length,
                            completion_length=len(completion),
                            eos_ids=eos_ids,
                            think_close_ids=set(think_close_token_ids(tokenizer)),
                            validated_force_span=(
                                (int(force_span[0]), int(force_span[1]))
                                if force_span is not None
                                else None
                            ),
                            thinking_budget=args.bft_thinking_budget,
                            answer_budget=args.bft_answer_budget,
                        )
                        if args.bft_thinking_budget > 0
                        else None
                    ),
                    "generation_batch_seconds": generation_seconds,
                    "teacher_force_seconds": verify_seconds,
                    "n_positions": diagnostics.n_positions,
                    "n_stochastic": diagnostics.n_stochastic,
                    "n_exact_match": diagnostics.n_exact_match,
                    "n_boundary_match": diagnostics.n_boundary_match,
                    "n_hard_mismatch": diagnostics.n_hard_mismatch,
                    "n_deterministic_hard_mismatch": (
                        diagnostics.n_deterministic_hard_mismatch
                    ),
                    "max_cdf_miss": diagnostics.max_cdf_miss,
                    "first_hard_mismatch_offset": (
                        diagnostics.first_hard_mismatch_offset
                    ),
                    **token_degeneracy_metrics(completion),
                }
                completion_text = None
                if args.include_text or ground_truth is not None:
                    completion_text = tokenizer.decode(completion)
                if ground_truth is not None:
                    row["reward"] = _compute_omi_reward(
                        {"ground_truth": ground_truth}, completion_text or "",
                    )
                if args.include_text:
                    row["completion_text"] = completion_text
                rows.append(row)
                prompt_rows.append(row)
        group_records.append(build_group_record(
            prompt_idx, prompt_rows, pool, args.pool_subset, args.pool_pick,
        ))

    elapsed = time.perf_counter() - started
    positions = sum(int(row["n_positions"]) for row in rows)
    hard = sum(int(row["n_hard_mismatch"]) for row in rows)
    stochastic = sum(int(row["n_stochastic"]) for row in rows)
    exact = sum(int(row["n_exact_match"]) for row in rows)
    artifact = {
        "schema_version": 1,
        "created_unix": time.time(),
        "process_id": os.getpid(),
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "checkpoint_hash": args.checkpoint_hash,
        "prompts_sha256": prompts_sha256,
        "profile_label": args.profile_label,
        "replicate": args.replicate,
        "config": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "bft_thinking_budget": args.bft_thinking_budget,
            "bft_answer_budget": args.bft_answer_budget,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "verification_dtype": verification_dtype,
            "verification_attn_implementation": verification_attention,
            "generation_use_cache": args.generation_use_cache,
            "deterministic_algorithms": args.deterministic_algorithms,
            "cudnn_benchmark": args.cudnn_benchmark,
            "allow_tf32": args.allow_tf32,
        },
        "runtime_profile": collect_runtime_fingerprint(
            model, verification_model,
        ),
        "summary": {
            "prompts": len(prompts),
            "rollouts": len(rows),
            "elapsed_seconds": elapsed,
            "generation_seconds": generation_seconds_total,
            "teacher_force_seconds": teacher_force_seconds_total,
            "generated_tokens": generated_tokens,
            "generation_tokens_per_second": (
                generated_tokens / generation_seconds_total
                if generation_seconds_total
                else 0.0
            ),
            "pipeline_tokens_per_second": (
                generated_tokens / elapsed if elapsed else 0.0
            ),
            "generated_tokens_per_second": (
                generated_tokens / elapsed if elapsed else 0.0
            ),
            "n_positions": positions,
            "n_hard_mismatch": hard,
            "hard_mismatch_rate": hard / positions if positions else None,
            "stochastic_agreement": exact / stochastic if stochastic else None,
            "ended_eos_rate": (
                sum(bool(row["ended_eos"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
        },
        "forced_seed": {
            "seed_source": args.seed_source,
            "pool_subset": args.pool_subset if args.seed_source == "pool" else None,
            "pool_pick": (
                args.pool_pick
                if args.seed_source == "pool" and args.pool_subset == "all" else None
            ),
            "scoring": (
                "reliquary.validator.batcher._forced_seed_verdict, "
                "_forced_seed_rollout_reject (floors from reliquary.constants)"
            ),
            "summary": forced_seed_summary(group_records),
            "groups": group_records,
        },
        "rollouts": rows,
    }
    if args.required_group_acceptance is not None:
        rate = artifact["forced_seed"]["summary"]["group_acceptance_rate"]
        artifact["forced_seed"]["required_group_acceptance"] = args.required_group_acceptance
        artifact["forced_seed"]["qualified"] = bool(
            rate is not None and rate >= args.required_group_acceptance
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
