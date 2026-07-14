"""GPU checkpoint runner for locked math and code holdouts."""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from reliquary.eval_dashboard.config import (
    build_effective_config,
    config_hash,
    sha256_bytes,
)
from reliquary.eval_dashboard.metrics import summarize_domain
from reliquary.eval_dashboard.models import (
    CheckpointResult,
    CodeTask,
    ContaminationReview,
    EvalConfig,
    MathTask,
    SampleResult,
    TaskResult,
)


logger = logging.getLogger(__name__)


def derive_sample_seed(
    seed_salt: str,
    effective_config_hash: str,
    task_id: str,
    sample_index: int,
) -> int:
    material = (
        f"reliquary-eval-v1\0{seed_salt}\0{effective_config_hash}\0"
        f"{task_id}\0{sample_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


@contextmanager
def _torch_seed(torch, seed: int, device: str) -> Iterator[None]:
    devices: list[int] = []
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        devices = [index if index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _generation_kwargs(
    config: EvalConfig,
    *,
    max_new_tokens: int,
    eos_ids: set[int],
    pad_token_id: int | None,
) -> dict[str, Any]:
    generation = config.generation
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": generation.repetition_penalty,
        # Checkpoint-local generation_config values must not mutate the
        # canonical benchmark policy. These mirror the protocol miner's inert
        # logits-processor overrides while preserving stochastic eval draws.
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "min_length": 0,
        "min_new_tokens": 0,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "bad_words_ids": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "sequence_bias": None,
    }
    if eos_ids:
        kwargs["eos_token_id"] = sorted(eos_ids)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    domain: str,
    seed: int,
    config: EvalConfig,
    device: str,
) -> tuple[str, int, bool, bool]:
    import torch

    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        first_eos_index,
        has_think_close,
        resolve_eos_token_ids,
        think_close_token_ids,
    )

    prompt_tokens = encode_prompt(tokenizer, prompt)
    prompt_length = len(prompt_tokens)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = min(eos_ids)

    is_bft = domain == "math" and config.generation.math_bft_enabled
    phase1_budget = (
        config.generation.math_thinking_budget
        if is_bft
        else (
            config.generation.math_max_new_tokens
            if domain == "math"
            else config.generation.code_max_new_tokens
        )
    )
    kwargs = _generation_kwargs(
        config,
        max_new_tokens=phase1_budget,
        eos_ids=eos_ids,
        pad_token_id=pad_token_id,
    )
    kwargs["attention_mask"] = attention_mask

    with _torch_seed(torch, seed, device), torch.inference_mode():
        output = model.generate(input_ids, **kwargs)[0].tolist()
        completion = output[prompt_length:]
        eos_index = first_eos_index(completion, eos_ids)
        if eos_index is not None:
            completion = completion[: eos_index + 1]
            text = tokenizer.decode(completion)
            return text, len(completion), True, False

        if not is_bft:
            text = tokenizer.decode(completion)
            return text, len(completion), False, False

        close_ids = set(think_close_token_ids(tokenizer))
        forced = not has_think_close(completion, close_ids)
        primed = output
        if forced:
            force_template = config.generation.math_force_template
            if not force_template.startswith("</think>"):
                raise ValueError("math_force_template must begin with </think>")
            close_tokens = list(think_close_token_ids(tokenizer))
            tail = force_template[len("</think>") :]
            force_tokens = close_tokens + list(
                tokenizer.encode(tail, add_special_tokens=False)
            )
            primed = primed + [int(token) for token in force_tokens]

        phase2_ids = torch.tensor([primed], dtype=torch.long, device=device)
        phase2_mask = torch.ones_like(phase2_ids)
        phase2_kwargs = _generation_kwargs(
            config,
            max_new_tokens=config.generation.math_answer_budget,
            eos_ids=eos_ids,
            pad_token_id=pad_token_id,
        )
        phase2_kwargs["attention_mask"] = phase2_mask
        phase2 = model.generate(phase2_ids, **phase2_kwargs)[0].tolist()
        tail = phase2[len(primed) :]
        eos_index = first_eos_index(tail, eos_ids)
        terminated = eos_index is not None
        if eos_index is not None:
            tail = tail[: eos_index + 1]
        full_completion = primed[prompt_length:] + tail
        text = tokenizer.decode(full_completion)
        return text, len(full_completion), terminated, forced


def _score_math(task: MathTask, completion: str) -> float:
    from reliquary.environment.openmathinstruct import _compute_omi_reward

    return float(
        _compute_omi_reward(
            {"ground_truth": task.ground_truth},
            completion,
        )
    )


def _score_code(task: CodeTask, completion: str, grader: Any) -> float:
    from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS
    from reliquary.environment.opencodeinstruct import _extract_python

    code = _extract_python(completion)
    return float(
        grader.evaluate_cases_strict(
            code,
            task.cases,
            timeout_s=GRADER_EVAL_TIMEOUT_SECONDS,
        )
    )


def _run_domain(
    *,
    domain: str,
    tasks: Sequence[MathTask] | Sequence[CodeTask],
    model: Any,
    tokenizer: Any,
    grader: Any,
    config: EvalConfig,
    effective_hash: str,
    device: str,
) -> list[TaskResult]:
    output: list[TaskResult] = []
    for task_n, task in enumerate(tasks, start=1):
        logger.info(
            "eval domain=%s task=%d/%d id=%s", domain, task_n, len(tasks), task.task_id
        )
        samples: list[SampleResult] = []
        for sample_index in range(config.generation.samples_per_prompt):
            seed = derive_sample_seed(
                config.generation.seed_salt,
                effective_hash,
                task.task_id,
                sample_index,
            )
            started = time.perf_counter()
            completion, length, terminated, forced = _generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=task.prompt,
                domain=domain,
                seed=seed,
                config=config,
                device=device,
            )
            if domain == "math":
                reward = _score_math(task, completion)  # type: ignore[arg-type]
            else:
                reward = _score_code(task, completion, grader)  # type: ignore[arg-type]
            samples.append(
                SampleResult(
                    sample_index=sample_index,
                    seed=seed,
                    reward=max(0.0, min(1.0, reward)),
                    completion_length=length,
                    terminated=terminated,
                    forced=forced,
                    completion_sha256=sha256_bytes(completion.encode("utf-8")),
                    duration_seconds=time.perf_counter() - started,
                )
            )
        output.append(
            TaskResult(
                task_id=task.task_id,
                prompt_sha256=sha256_bytes(task.prompt.encode("utf-8")),
                samples=samples,
            )
        )
    return output


def hardware_metadata(torch, device: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = index if index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        data.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return data


def load_tokenizer(repo_id: str, revision: str):
    from reliquary.shared.modeling import load_tokenizer as shared_load_tokenizer

    return shared_load_tokenizer(repo_id, revision=revision, trust_remote_code=False)


def run_checkpoint(
    *,
    config: EvalConfig,
    model_repo_id: str,
    model_revision: str,
    checkpoint_n: int,
    observed_window: int,
    math_tasks: Sequence[MathTask],
    code_tasks: Sequence[CodeTask],
    tokenizer: Any | None = None,
    device: str = "cuda:0",
    attention_implementation: str = "flash_attention_2",
    contamination_reviews: dict[str, ContaminationReview],
) -> tuple[CheckpointResult, dict[str, Any]]:
    """Evaluate one immutable checkpoint and return evidence plus its contract."""
    import torch

    from reliquary.environment.grader_client import GraderClient
    from reliquary.shared.modeling import load_text_generation_model

    if not model_revision or len(model_revision) < 40:
        raise ValueError("model_revision must be an immutable revision")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    if tokenizer is None:
        tokenizer = load_tokenizer(
            config.tokenizer_source.repo_id,
            config.tokenizer_source.revision,
        )
    hardware = hardware_metadata(torch, device)
    effective = build_effective_config(
        config,
        tokenizer=tokenizer,
        attention_implementation=attention_implementation,
        hardware=hardware,
        contamination_reviews=contamination_reviews,
    )
    effective_hash = config_hash(effective)

    grader = GraderClient()
    preflight_cases = [
        {
            "entry": {"kind": "function", "name": "reliquary_eval_preflight"},
            "args": [41],
            "kwargs": {},
            "expected": 42,
            "compare": "exact",
        }
    ]
    preflight_score = grader.evaluate_cases_strict(
        "def reliquary_eval_preflight(value):\n    return value + 1\n",
        preflight_cases,
        timeout_s=5.0,
    )
    if preflight_score != 1.0:
        raise RuntimeError("code grader preflight failed")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    started_at = time.time()
    model = (
        load_text_generation_model(
            model_repo_id,
            revision=model_revision,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation=attention_implementation,
        )
        .to(device)
        .eval()
    )
    try:
        math_results = _run_domain(
            domain="math",
            tasks=math_tasks,
            model=model,
            tokenizer=tokenizer,
            grader=grader,
            config=config,
            effective_hash=effective_hash,
            device=device,
        )
        code_results = _run_domain(
            domain="code",
            tasks=code_tasks,
            model=model,
            tokenizer=tokenizer,
            grader=grader,
            config=config,
            effective_hash=effective_hash,
            device=device,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed_at = time.time()
    result = CheckpointResult(
        config_hash=effective_hash,
        config_sha256=effective_hash,
        lineage_id=config.lineage.lineage_id,
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        checkpoint_n=checkpoint_n,
        observed_window=observed_window,
        started_at=started_at,
        completed_at=completed_at,
        completed_at_iso=_iso(completed_at),
        duration_seconds=completed_at - started_at,
        hardware=hardware,
        runtime={
            **effective["runtime"],
            "tokenizer": effective["tokenizer"],
        },
        domains={
            "math": summarize_domain("math", math_results),
            "code": summarize_domain("code", code_results),
        },
    )
    return result, effective
