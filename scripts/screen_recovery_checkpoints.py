#!/usr/bin/env python3
"""Protocol-parity recovery screen for immutable checkpoints.

The screen uses the selected production generation profile, its exact canonical
prompts, and Reliquary's authoritative environment grader. It can optionally
hold forced inverse-CDF draws common across profiles for an explicitly labelled
offline ablation. It is not a replacement for the sealed private evaluation
dashboard.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from typing import Any


LEGACY_ANSWER_INSTRUCTION = "\n\nPut your final answer within \\boxed{}."
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")

_RUNTIME_PACKAGES = {
    "transformers_version": "transformers",
    "flash_linear_attention_version": "flash-linear-attention",
    "flash_attn_version": "flash-attn",
    "causal_conv1d_version": "causal-conv1d",
    "bitsandbytes_version": "bitsandbytes",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_snapshot_sha256(directory: Path) -> str:
    """Bind every relative path and byte in a local model snapshot."""
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint(torch: Any) -> dict[str, Any]:
    """Record the numerical stack that can alter forced-token generation."""
    packages: dict[str, str | None] = {}
    for field, distribution in _RUNTIME_PACKAGES.items():
        try:
            packages[field] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[field] = None
    return {
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        **packages,
    }


def _source_identity(
    repository: Path | None = None,
) -> tuple[str | None, str | None]:
    """Resolve source identity without mistaking the image for a bind mount."""
    repository = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else repository.resolve()
    )
    explicit = os.environ.get("RELIQUARY_SOURCE_REVISION", "").strip().lower()
    if explicit and _IMMUTABLE_REVISION.fullmatch(explicit) is None:
        raise ValueError(
            "RELIQUARY_SOURCE_REVISION must be a full 40-character SHA"
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
        if _IMMUTABLE_REVISION.fullmatch(revision) is not None:
            if explicit and explicit != revision:
                raise ValueError(
                    "RELIQUARY_SOURCE_REVISION does not match checkout HEAD"
                )
            return revision, "git"
    except (OSError, subprocess.CalledProcessError):
        pass
    if explicit:
        return explicit, "explicit"
    revision = os.environ.get("RELIQUARY_BUILD_REVISION", "").strip().lower()
    if (
        repository == Path("/opt/reliquary").resolve()
        and _IMMUTABLE_REVISION.fullmatch(revision) is not None
    ):
        return revision, "image"
    return None, None


def _source_revision(repository: Path | None = None) -> str | None:
    """Return the immutable source revision used for the screen."""
    return _source_identity(repository)[0]


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


def render_math_task_prompt(profile: Any, problem: str) -> str:
    """Render the screen prompt through the selected generation contract."""

    environment = profile.environments["openmathinstruct"]
    template = environment.prompt_template
    if template is None:
        # Historical v2-v4 screen compatibility. The production v5 profile
        # takes the explicit-template branch above.
        return problem + LEGACY_ANSWER_INSTRUCTION
    return template.render(problem=problem)


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
            "snapshot_sha256": _directory_snapshot_sha256(resolved),
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
        "snapshot_sha256": None,
    }


def _single_phase_rollouts(
    outputs: Any,
    *,
    prompt_tokens: list[int],
    eos_ids: set[int],
) -> list[dict[str, Any]]:
    """Mirror the non-BFT miner path used by the code environment."""
    prompt_length = len(prompt_tokens)
    rollouts = []
    for row in outputs:
        sequence = row.tolist() if hasattr(row, "tolist") else list(row)
        completion = sequence[prompt_length:]
        first_eos = next(
            (
                index
                for index, token in enumerate(completion)
                if int(token) in eos_ids
            ),
            None,
        )
        if first_eos is not None:
            completion = completion[:first_eos + 1]
        rollouts.append({
            "tokens": prompt_tokens + completion,
            "prompt_length": prompt_length,
            "forced": False,
        })
    return rollouts


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
    parser.add_argument(
        "--protocol-profile",
        default=None,
        help=(
            "Exact generation profile to screen. May instead be supplied via "
            "RELIQUARY_PROTOCOL_PROFILE; one of the two is required."
        ),
    )
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
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
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--answer-budget", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional assertion of the selected profile's environment cap; "
            "a different value is rejected."
        ),
    )
    parser.add_argument(
        "--seed-domain", default="reliquary-recovery-common-draws-v1"
    )
    parser.add_argument(
        "--forced-stream-domain",
        default=None,
        help=(
            "Offline paired-ablation override for the protocol's internal "
            "forced-seed domain. Omit for a deployment-parity screen."
        ),
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
    if args.forced_stream_domain is not None and not args.forced_stream_domain:
        raise ValueError("--forced-stream-domain must not be empty")
    profile_id = args.protocol_profile or os.environ.get(
        "RELIQUARY_PROTOCOL_PROFILE", ""
    ).strip()
    if not profile_id:
        raise ValueError(
            "--protocol-profile or RELIQUARY_PROTOCOL_PROFILE is required"
        )
    inherited_profile = os.environ.get("RELIQUARY_PROTOCOL_PROFILE", "").strip()
    if inherited_profile and inherited_profile != profile_id:
        raise ValueError(
            "--protocol-profile disagrees with RELIQUARY_PROTOCOL_PROFILE"
        )
    # Select before importing any module that snapshots protocol constants.
    os.environ["RELIQUARY_PROTOCOL_PROFILE"] = profile_id
    from reliquary.protocol.profiles import (
        ACTIVE_PROTOCOL_PROFILE,
        to_generation_contract,
    )

    if ACTIVE_PROTOCOL_PROFILE.profile_id != profile_id:
        raise RuntimeError(
            "protocol modules were imported before the screen profile was "
            "selected"
        )
    try:
        environment_profile = ACTIVE_PROTOCOL_PROFILE.environments[
            args.environment
        ]
    except KeyError as exc:
        raise ValueError(
            f"profile {profile_id!r} does not support {args.environment!r}"
        ) from exc
    tokenizer_repo = args.tokenizer_repo or ACTIVE_PROTOCOL_PROFILE.model_id
    tokenizer_revision = (
        args.tokenizer_revision or ACTIVE_PROTOCOL_PROFILE.model_revision
    )
    if tokenizer_repo != ACTIVE_PROTOCOL_PROFILE.model_id:
        raise ValueError(
            "--tokenizer-repo must match the selected protocol model"
        )
    if tokenizer_revision != ACTIVE_PROTOCOL_PROFILE.model_revision:
        raise ValueError(
            "--tokenizer-revision must match the selected protocol revision"
        )
    max_new_tokens = (
        environment_profile.max_new_tokens
        if args.max_new_tokens is None
        else args.max_new_tokens
    )
    if max_new_tokens != environment_profile.max_new_tokens:
        raise ValueError(
            "--max-new-tokens must match the selected environment profile"
        )
    bft_profile = environment_profile.bft
    bft_applicable = bft_profile is not None
    thinking_budget = bft_profile.thinking_budget if bft_profile else 0
    answer_budget = bft_profile.answer_budget if bft_profile else 0
    if args.thinking_budget is not None and args.thinking_budget != thinking_budget:
        raise ValueError(
            "--thinking-budget must match the selected environment profile"
        )
    if args.answer_budget is not None and args.answer_budget != answer_budget:
        raise ValueError(
            "--answer-budget must match the selected environment profile"
        )
    # Capture provenance before any long-running model work. The checkout may be
    # fast-forwarded while a screen is running; late resolution would mislabel
    # already-imported code with the newer revision.
    reliquary_revision, reliquary_revision_source = _source_identity()
    if reliquary_revision is None:
        raise RuntimeError(
            "source revision unavailable; set RELIQUARY_SOURCE_REVISION when "
            "running from a bind-mounted checkout"
        )
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
    from reliquary.environment import forced_sampling as forced_sampling_module
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index,
        force_close_token_ids,
        load_text_generation_model,
        load_tokenizer,
        resolve_eos_token_ids,
        think_close_token_ids,
    )

    protocol_forced_stream_domain = forced_sampling_module.FORCED_SEED_DOMAIN
    forced_stream_domain = (
        args.forced_stream_domain or protocol_forced_stream_domain
    )
    if args.forced_stream_domain is not None:
        forced_sampling_module.FORCED_SEED_DOMAIN = args.forced_stream_domain

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
        for task in tasks:
            task["prompt"] = render_math_task_prompt(
                ACTIVE_PROTOCOL_PROFILE,
                task["prompt"],
            )
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
        dataset_identity = {
            "path": None,
            "sha256": None,
            "repo": args.code_repo,
            "revision": args.code_revision,
        }
    tokenizer = load_tokenizer(
        tokenizer_repo,
        revision=tokenizer_revision,
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
        prompt = task["prompt"]
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
            "max_new_tokens": (
                thinking_budget if bft_applicable
                else max_new_tokens
            ),
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
            if bft_applicable:
                phase2_kwargs: dict[str, Any] = {
                    "pad_token_id": pad_token_id
                }
                if eos_ids:
                    phase2_kwargs["eos_token_id"] = sorted(eos_ids)
                rollouts = _bft_assemble_rollouts(
                    model=model,
                    phase1_tensor=phase1,
                    prompt_tokens=prompt_tokens,
                    think_close_ids=think_close_ids,
                    force_ids=force_ids,
                    eos_ids=eos_ids,
                    answer_budget=answer_budget,
                    randomness=randomness,
                    hotkey="reliquary-recovery-eval",
                    prompt_idx=prompt_idx,
                    checkpoint_hash=eval_checkpoint_hash,
                    gen_kwargs=phase2_kwargs,
                )
            else:
                rollouts = _single_phase_rollouts(
                    phase1,
                    prompt_tokens=prompt_tokens,
                    eos_ids=eos_ids,
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
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_length": prompt_length,
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
        "schema_version": 2,
        "checkpoint_label": args.checkpoint_label,
        "environment": args.environment,
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "model_path": model_identity["path"],
        "model_snapshot_sha256": model_identity["snapshot_sha256"],
        "model_source_kind": model_identity["kind"],
        "protocol_profile_id": ACTIVE_PROTOCOL_PROFILE.profile_id,
        "protocol_version": ACTIVE_PROTOCOL_PROFILE.protocol_version,
        "generation_contract": to_generation_contract(
            ACTIVE_PROTOCOL_PROFILE
        ),
        "tokenizer_repo": tokenizer_repo,
        "tokenizer_revision": tokenizer_revision,
        "dataset_path": dataset_identity["path"],
        "dataset_sha256": dataset_identity["sha256"],
        "dataset_repo": dataset_identity["repo"],
        "dataset_revision": args.dataset_revision,
        "n_prompts": args.n_prompts,
        "prompt_offset": args.prompt_offset,
        "samples_per_prompt": args.samples_per_prompt,
        "thinking_budget": thinking_budget,
        "answer_budget": answer_budget,
        "generation_mode": (
            "bft" if bft_applicable else "single_phase"
        ),
        "max_new_tokens": max_new_tokens,
        "seed_domain": args.seed_domain,
        "forced_stream_domain": forced_stream_domain,
        "protocol_forced_stream_domain": protocol_forced_stream_domain,
        "deployment_parity": args.forced_stream_domain is None,
        "attention_implementation": args.attention_implementation,
        "reliquary_revision": reliquary_revision,
        "reliquary_revision_source": reliquary_revision_source,
        "screen_script_sha256": screen_script_sha256,
        "summary": summarize(samples, args.n_prompts),
        "samples": samples,
        "runtime": {
            **_runtime_fingerprint(torch),
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
