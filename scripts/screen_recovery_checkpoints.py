#!/usr/bin/env python3
"""Protocol-parity recovery screen for immutable checkpoints.

The screen uses common forced inverse-CDF draws across models, the production
2048/512 BFT path, and Reliquary's authoritative environment grader. It is a
paired recovery benchmark, not a replacement for the sealed private
evaluation dashboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


ANSWER_INSTRUCTION = "\n\nPut your final answer within \\boxed{}."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(repository: Path | None = None) -> str | None:
    """Best-effort identity of the mounted source tree used for the screen."""
    repository = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else repository.resolve()
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "rev-parse",
                "HEAD",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = completed.stdout.strip().lower()
        if len(revision) == 40:
            return revision
    except (OSError, subprocess.CalledProcessError):
        pass
    revision = os.environ.get("RELIQUARY_BUILD_REVISION", "").strip().lower()
    return revision or None


def select_tasks(
    path: Path,
    *,
    n_prompts: int,
    dataset_revision: str,
    prompt_offset: int = 0,
) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if prompt_offset < 0:
        raise ValueError("prompt_offset must be non-negative")
    if n_prompts <= 0 or prompt_offset + n_prompts > len(rows):
        raise ValueError(
            "prompt range must be within dataset bounds: "
            f"offset={prompt_offset}, n_prompts={n_prompts}, rows={len(rows)}"
        )

    def rank(row: dict[str, Any]) -> bytes:
        task_id = str(row.get("unique_id") or row.get("task_id") or "")
        if not task_id:
            raise ValueError("every benchmark row must have a stable task id")
        return hashlib.sha256(
            f"{dataset_revision}\0{task_id}".encode("utf-8")
        ).digest()

    selected = sorted(rows, key=rank)[
        prompt_offset:prompt_offset + n_prompts
    ]
    return [
        {
            "task_id": str(row.get("unique_id") or row.get("task_id")),
            "prompt": str(row["problem"]),
            "ground_truth": str(row["answer"]),
            "subject": str(row.get("subject") or "unknown"),
            "level": str(row.get("level") or "unknown"),
        }
        for row in selected
    ]


def select_code_tasks(
    environment: Any,
    *,
    n_prompts: int,
    dataset_revision: str,
    prompt_offset: int = 0,
) -> list[dict[str, str]]:
    """Select a revision-bound code holdout without downloading every row."""
    dataset_size = len(environment)
    if prompt_offset < 0:
        raise ValueError("prompt_offset must be non-negative")
    if n_prompts <= 0 or prompt_offset + n_prompts > dataset_size:
        raise ValueError(
            "prompt range must be within dataset bounds: "
            f"offset={prompt_offset}, n_prompts={n_prompts}, "
            f"rows={dataset_size}"
        )

    ranked_indices = sorted(
        range(dataset_size),
        key=lambda index: hashlib.sha256(
            f"{dataset_revision}\0{index}".encode("utf-8")
        ).digest(),
    )[prompt_offset:prompt_offset + n_prompts]
    tasks = []
    for index in ranked_indices:
        problem = environment.get_problem(index)
        tasks.append({
            "task_id": str(problem["id"]),
            "prompt": str(problem["prompt"]),
            "ground_truth": str(problem["ground_truth"]),
            "subject": "code",
            "level": "unknown",
        })
    return tasks


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _token_repetition(tokens: list[int], ngram: int = 4) -> tuple[float, int]:
    if len(tokens) < ngram:
        repeated_ratio = 0.0
    else:
        grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
        repeated_ratio = 1.0 - (len(set(grams)) / len(grams))
    max_run = 0
    current_run = 0
    previous = None
    for token in tokens:
        if token == previous:
            current_run += 1
        else:
            previous = token
            current_run = 1
        max_run = max(max_run, current_run)
    return repeated_ratio, max_run


def resolve_model_source(
    *,
    model_repo: str | None,
    model_revision: str | None,
    model_path: Path | None,
) -> tuple[str, dict[str, str], dict[str, str | None]]:
    """Resolve either an immutable Hub checkpoint or a local candidate."""
    if model_path is not None:
        if model_repo is not None or model_revision is not None:
            raise ValueError(
                "--model-path cannot be combined with Hub model arguments"
            )
        resolved = model_path.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"local model path is not a directory: {resolved}")
        return str(resolved), {}, {
            "kind": "local",
            "repo": None,
            "revision": None,
            "path": str(resolved),
        }
    if not model_repo or not model_revision:
        raise ValueError(
            "Hub checkpoints require --model-repo and --model-revision"
        )
    return model_repo, {"revision": model_revision}, {
        "kind": "hub",
        "repo": model_repo,
        "revision": model_revision,
        "path": None,
    }


def summarize(samples: list[dict[str, Any]], n_prompts: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_task.setdefault(str(sample["task_id"]), []).append(sample)
    if len(by_task) != n_prompts:
        raise ValueError(
            f"expected {n_prompts} task groups, observed {len(by_task)}"
        )
    first_rewards = []
    best_rewards = []
    all_rewards = []
    lengths = []
    for task_samples in by_task.values():
        ordered = sorted(task_samples, key=lambda row: int(row["sample_index"]))
        first_rewards.append(float(ordered[0]["reward"]))
        best_rewards.append(max(float(row["reward"]) for row in ordered))
        all_rewards.extend(float(row["reward"]) for row in ordered)
        lengths.extend(float(row["completion_length"]) for row in ordered)
    denominator = max(1, len(samples))
    return {
        "prompts": len(by_task),
        "samples": len(samples),
        "pass_at_1": statistics.fmean(first_rewards),
        "pass_at_k": statistics.fmean(best_rewards),
        "pass_average": statistics.fmean(all_rewards),
        "termination_rate": sum(bool(row["terminated"]) for row in samples)
        / denominator,
        "forced_rate": sum(bool(row["forced"]) for row in samples)
        / denominator,
        "boxed_rate": sum(bool(row["boxed"]) for row in samples)
        / denominator,
        "rambling_proxy_rate": sum(
            bool(row["rambling_proxy"]) for row in samples
        ) / denominator,
        "mean_completion_length": statistics.fmean(lengths),
        "p50_completion_length": statistics.median(lengths),
        "p95_completion_length": _quantile(lengths, 0.95),
        "max_completion_length": max(lengths),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-repo")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument(
        "--environment",
        choices=("openmathinstruct", "opencodeinstruct"),
        default="openmathinstruct",
    )
    parser.add_argument("--tokenizer-repo", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--tokenizer-revision",
        default="15852e8c16360a2fea060d615a32b45270f8a8fc",
    )
    parser.add_argument("--math-jsonl", type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--code-repo", default="R0mAI/opencodeinstruct-curated"
    )
    parser.add_argument(
        "--code-revision",
        default="d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7",
    )
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=2048)
    parser.add_argument("--answer-budget", type=int, default=512)
    parser.add_argument(
        "--seed-domain", default="reliquary-recovery-common-draws-v1"
    )
    parser.add_argument(
        "--attention-implementation", default="flash_attention_2"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    # Capture provenance before any long-running model work. The checkout may be
    # fast-forwarded while a screen is running; late resolution would mislabel
    # already-imported code with the newer revision.
    reliquary_revision = _source_revision()
    screen_script_sha256 = _sha256(Path(__file__).resolve())

    model_source, model_kwargs, model_identity = resolve_model_source(
        model_repo=args.model_repo,
        model_revision=args.model_revision,
        model_path=args.model_path,
    )

    import torch

    from reliquary.miner.engine import _bft_assemble_rollouts
    from reliquary.miner.forced_seed_sampler import (
        ForcedSeedLogitsProcessor,
        forced_seed_generate_kwargs,
    )
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index,
        force_close_token_ids,
        load_text_generation_model,
        load_tokenizer,
        resolve_eos_token_ids,
        think_close_token_ids,
    )

    if args.environment == "openmathinstruct":
        if args.math_jsonl is None:
            raise ValueError(
                "--math-jsonl is required for openmathinstruct"
            )
        from reliquary.environment.openmathinstruct import _compute_omi_reward

        tasks = select_tasks(
            args.math_jsonl,
            n_prompts=args.n_prompts,
            dataset_revision=args.dataset_revision,
            prompt_offset=args.prompt_offset,
        )
        reward_fn = lambda task, completion: _compute_omi_reward(  # noqa: E731
            {"ground_truth": task["ground_truth"]}, completion
        )
        answer_instruction = ANSWER_INSTRUCTION
        dataset_identity = {
            "path": str(args.math_jsonl.resolve()),
            "sha256": _sha256(args.math_jsonl),
            "repo": None,
            "revision": args.dataset_revision,
        }
    else:
        from reliquary.cli.main import (
            _ensure_grader_running,
            _grader_is_running,
        )
        from reliquary.constants import GRADER_SOCKET_PATH
        from reliquary.environment.opencodeinstruct import (
            OpenCodeInstructEnvironment,
        )

        if args.dataset_revision != args.code_revision:
            raise ValueError(
                "--dataset-revision must match --code-revision for code screens"
            )
        os.environ["RELIQUARY_OCI_REPO"] = args.code_repo
        os.environ["RELIQUARY_OCI_REVISION"] = args.code_revision
        _ensure_grader_running(use_runsc=True)
        if not _grader_is_running(GRADER_SOCKET_PATH):
            raise RuntimeError("the gVisor code grader did not become ready")
        code_environment = OpenCodeInstructEnvironment()
        tasks = select_code_tasks(
            code_environment,
            n_prompts=args.n_prompts,
            dataset_revision=args.dataset_revision,
            prompt_offset=args.prompt_offset,
        )
        reward_fn = code_environment.compute_reward
        answer_instruction = ""
        dataset_identity = {
            "path": None,
            "sha256": None,
            "repo": args.code_repo,
            "revision": args.code_revision,
        }
    tokenizer = load_tokenizer(
        args.tokenizer_repo,
        revision=args.tokenizer_revision,
    )
    model = load_text_generation_model(
        model_source,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation,
        **model_kwargs,
    ).to("cuda:0").eval()
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    think_close_ids = set(think_close_token_ids(tokenizer))
    force_ids = force_close_token_ids(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = min(eos_ids)

    randomness = hashlib.sha256(args.seed_domain.encode("utf-8")).hexdigest()
    eval_checkpoint_hash = hashlib.sha256(
        f"{args.seed_domain}\0checkpoint".encode("utf-8")
    ).hexdigest()
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for task_number, task in enumerate(tasks, start=1):
        prompt = task["prompt"] + answer_instruction
        prompt_tokens = encode_prompt(tokenizer, prompt)
        prompt_length = len(prompt_tokens)
        input_tensor = torch.tensor(
            [prompt_tokens] * args.samples_per_prompt,
            dtype=torch.long,
            device="cuda:0",
        )
        attention_mask = torch.ones_like(input_tensor)
        prompt_idx = int.from_bytes(
            hashlib.sha256(task["task_id"].encode("utf-8")).digest()[:8],
            "big",
        )
        phase1_kwargs: dict[str, Any] = {
            "max_new_tokens": args.thinking_budget,
            "pad_token_id": pad_token_id,
            "attention_mask": attention_mask,
        }
        if eos_ids:
            phase1_kwargs["eos_token_id"] = sorted(eos_ids)
        processor = ForcedSeedLogitsProcessor(
            randomness=randomness,
            hotkey="reliquary-recovery-eval",
            prompt_idx=prompt_idx,
            checkpoint_hash=eval_checkpoint_hash,
            rollout_indices=list(range(args.samples_per_prompt)),
            base_offsets=[0] * args.samples_per_prompt,
            start_len=prompt_length,
        )
        with torch.inference_mode():
            phase1 = model.generate(
                input_tensor,
                **forced_seed_generate_kwargs(phase1_kwargs, processor),
            )
            phase2_kwargs: dict[str, Any] = {"pad_token_id": pad_token_id}
            if eos_ids:
                phase2_kwargs["eos_token_id"] = sorted(eos_ids)
            rollouts = _bft_assemble_rollouts(
                model=model,
                phase1_tensor=phase1,
                prompt_tokens=prompt_tokens,
                think_close_ids=think_close_ids,
                force_ids=force_ids,
                eos_ids=eos_ids,
                answer_budget=args.answer_budget,
                randomness=randomness,
                hotkey="reliquary-recovery-eval",
                prompt_idx=prompt_idx,
                checkpoint_hash=eval_checkpoint_hash,
                gen_kwargs=phase2_kwargs,
            )

        for sample_index, rollout in enumerate(rollouts):
            completion_tokens = rollout["tokens"][prompt_length:]
            completion_text = tokenizer.decode(completion_tokens)
            repeated_ratio, max_token_run = _token_repetition(
                completion_tokens
            )
            terminated = first_eos_index(completion_tokens, eos_ids) is not None
            forced = bool(rollout.get("forced", False))
            samples.append({
                "task_id": task["task_id"],
                "subject": task["subject"],
                "level": task["level"],
                "sample_index": sample_index,
                "reward": float(reward_fn(task, completion_text)),
                "completion_length": len(completion_tokens),
                "terminated": terminated,
                "forced": forced,
                "boxed": (
                    "\\boxed{" in completion_text
                    or "\\fbox{" in completion_text
                ),
                "think_closed": any(
                    int(token) in think_close_ids
                    for token in completion_tokens
                ),
                "repeated_4gram_ratio": repeated_ratio,
                "max_token_run": max_token_run,
                "rambling_proxy": (
                    repeated_ratio >= 0.50 or max_token_run >= 8
                ),
                "completion_sha256": hashlib.sha256(
                    completion_text.encode("utf-8")
                ).hexdigest(),
            })
        print(
            f"checkpoint={args.checkpoint_label} "
            f"task={task_number}/{len(tasks)} id={task['task_id']}",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "checkpoint_label": args.checkpoint_label,
        "environment": args.environment,
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "model_path": model_identity["path"],
        "model_source_kind": model_identity["kind"],
        "tokenizer_repo": args.tokenizer_repo,
        "tokenizer_revision": args.tokenizer_revision,
        "dataset_path": dataset_identity["path"],
        "dataset_sha256": dataset_identity["sha256"],
        "dataset_repo": dataset_identity["repo"],
        "dataset_revision": args.dataset_revision,
        "n_prompts": args.n_prompts,
        "prompt_offset": args.prompt_offset,
        "samples_per_prompt": args.samples_per_prompt,
        "thinking_budget": args.thinking_budget,
        "answer_budget": args.answer_budget,
        "seed_domain": args.seed_domain,
        "attention_implementation": args.attention_implementation,
        "reliquary_revision": reliquary_revision,
        "screen_script_sha256": screen_script_sha256,
        "summary": summarize(samples, args.n_prompts),
        "samples": samples,
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
