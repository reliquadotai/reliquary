#!/usr/bin/env python3
"""Benchmark cached forced-seed generation against validator teacher forcing.

Run this script in a fresh process for each dependency/kernel profile. It emits
one JSON artifact containing the exact runtime fingerprint and per-rollout CDF,
termination, repetition, and throughput diagnostics. It does not change gates.
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
            prompts.append({"prompt": value, "prompt_idx": int(prompt_idx)})
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
    parser.add_argument("--randomness", default="42" * 32)
    parser.add_argument("--hotkey", default="benchmark-hotkey")
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()

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
    from reliquary.environment.forced_sampling import (
        seed_consistency_diagnostics,
        u_at,
    )
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
    generated_tokens = 0
    generation_seconds_total = 0.0
    teacher_force_seconds_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for prompt_row in prompts:
        prompt = prompt_row["prompt"]
        prompt_idx = int(prompt_row["prompt_idx"])
        prompt_tokens = encode_prompt(tokenizer, prompt)
        prompt_length = len(prompt_tokens)
        input_ids = torch.tensor(
            [prompt_tokens] * args.batch_size,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        processor = ForcedSeedLogitsProcessor(
            randomness=args.randomness,
            hotkey=args.hotkey,
            prompt_idx=prompt_idx,
            checkpoint_hash=args.checkpoint_hash,
            rollout_indices=list(range(args.batch_size)),
            base_offsets=[0] * args.batch_size,
            start_len=prompt_length,
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
                )
            else:
                generated_rows = [
                    {
                        "tokens": generated[index].tolist(),
                        "prompt_length": prompt_length,
                        "forced": False,
                    }
                    for index in range(args.batch_size)
                ]
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - batch_started
        generation_seconds_total += generation_seconds

        for rollout_idx in range(args.batch_size):
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
                u_at(
                    args.randomness,
                    args.hotkey,
                    prompt_idx,
                    args.checkpoint_hash,
                    rollout_idx,
                    offset,
                )
                for offset in range(len(completion))
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
                "rollout_idx": rollout_idx,
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
            if args.include_text:
                row["completion_text"] = tokenizer.decode(completion)
            rows.append(row)

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
        "rollouts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
