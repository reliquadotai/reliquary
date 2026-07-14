#!/usr/bin/env python3
"""Benchmark cached forced-seed generation against validator teacher forcing.

Run this script in a fresh process for each dependency/kernel profile. It emits
one JSON artifact containing the exact runtime fingerprint and per-rollout CDF,
termination, repetition, and throughput diagnostics. It does not change gates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _dtype(torch, name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_prompts(path: Path | None, direct: list[str]) -> list[str]:
    prompts = list(direct)
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("prompt") if isinstance(row, dict) else row
            if not isinstance(value, str) or not value:
                raise ValueError("each JSONL row must contain a non-empty prompt")
            prompts.append(value)
    if not prompts:
        raise ValueError("provide --prompt or --prompts-jsonl")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
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
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.modeling import (
        first_eos_index,
        load_text_generation_model,
        resolve_eos_token_ids,
    )
    from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint
    from reliquary.validator.rollout_telemetry import token_degeneracy_metrics

    torch.use_deterministic_algorithms(args.deterministic_algorithms)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_text_generation_model(
        args.model,
        torch_dtype=_dtype(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(device).eval()
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = min(eos_ids) if eos_ids else 0

    prompts = _load_prompts(args.prompts_jsonl, args.prompt)
    rows: list[dict] = []
    generated_tokens = 0
    started = time.perf_counter()

    for prompt_idx, prompt in enumerate(prompts):
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
            "max_new_tokens": args.max_new_tokens,
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - batch_started

        for rollout_idx in range(args.batch_size):
            completion = generated[rollout_idx, prompt_length:].tolist()
            eos_offset = first_eos_index(completion, eos_ids)
            if eos_offset is not None:
                completion = completion[:eos_offset + 1]
            sequence = prompt_tokens + completion
            generated_tokens += len(completion)
            full_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            verify_started = time.perf_counter()
            with torch.no_grad():
                _, logits = forward_single_layer(
                    model, full_ids, None, LAYER_INDEX,
                )
            logits_slice = logits[
                0,
                prompt_length - 1:prompt_length + len(completion) - 1,
            ]
            uniforms = [
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
            diagnostics = seed_consistency_diagnostics(
                logits_slice,
                completion,
                uniforms,
                t=T_PROTO,
                top_k=TOP_K_PROTO,
                top_p=TOP_P_PROTO,
                stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB,
                boundary_epsilon=FORCED_SEED_CDF_BOUNDARY_EPSILON,
                position_offsets=list(range(len(completion))),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            verify_seconds = time.perf_counter() - verify_started
            row = {
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
                "prompt_length": prompt_length,
                "completion_length": len(completion),
                "ended_eos": eos_offset is not None,
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
        "model": args.model,
        "checkpoint_hash": args.checkpoint_hash,
        "config": {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "generation_use_cache": args.generation_use_cache,
            "deterministic_algorithms": args.deterministic_algorithms,
            "cudnn_benchmark": args.cudnn_benchmark,
            "allow_tf32": args.allow_tf32,
        },
        "runtime_profile": collect_runtime_fingerprint(model, model),
        "summary": {
            "prompts": len(prompts),
            "rollouts": len(rows),
            "elapsed_seconds": elapsed,
            "generated_tokens": generated_tokens,
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
