#!/usr/bin/env python3
"""Qualify Episode v1 admission-to-optimizer flow on a real CUDA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROFILE = "qwen3-4b-reliquary-episode-v7-dev1"


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(getattr(encoded, "ids", encoded))


def _episode_metadata(trace: Any) -> dict[str, Any]:
    assert trace.reward is not None
    return {
        "schema_version": trace.schema,
        "renderer_id": "reliquary-jsonl-tools-v1",
        "task_id": trace.task_id,
        "seed": trace.seed,
        "actions": [action.to_wire() for action in trace.actions],
        "assistant_spans": [list(span) for span in trace.assistant_spans],
        "observation_digests": list(trace.observation_digests),
        "termination_reason": trace.termination_reason,
        "state_digest": trace.reward.state_digest,
        "trace_digest": trace.trace_digest,
    }


def _model_weight_sha256(model: Any) -> str:
    """Hash the complete in-memory parameter state without one giant copy."""

    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
            digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--environment", default="reliquary_stateful_tools_v1"
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    # constants.py resolves the active signed profile at import time.
    os.environ["RELIQUARY_PROTOCOL_PROFILE"] = args.profile

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("this qualification requires an available CUDA device")

    from scripts.qualify_episode_suite import (
        _artifact_digest,
        _git_identity,
        _runtime_identity,
    )
    from reliquary.constants import ATTN_IMPLEMENTATION, M_ROLLOUTS
    from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
    from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
    from reliquary.environment.agentic.types import AssistantAction
    from reliquary.environment.registry import get_environment_spec
    from reliquary.protocol.submission import (
        BatchSubmissionRequest,
        RolloutSubmission,
    )
    from reliquary.shared.modeling import (
        load_text_generation_model,
        load_tokenizer,
    )
    from reliquary.shared.training_payload import (
        decode_training_payload,
        encode_training_payload,
    )
    from reliquary.validator import admission, telemetry
    from reliquary.validator.admission import (
        AdmissionContext,
        AdmissionRuntimeMaterials,
        ParsedSubmission,
        score_and_finalize_submission,
    )
    from reliquary.validator import training
    from reliquary.validator.training import (
        _policy_token_positions,
        _selected_logprobs_for_tokens,
        reset_training_state,
        train_step,
    )

    started = time.perf_counter()
    artifact = _artifact_digest(args.model_path, args.model_revision)
    report: dict[str, Any] = {
        "schema": "reliquary/episode-training-gpu-qualification/v1",
        "passed": False,
        "profile": args.profile,
        "environment": args.environment,
        "task_index": args.task_index,
        "model_artifact": artifact,
        "git": _git_identity(),
        "runtime": _runtime_identity(),
        "device": {
            "requested": args.device,
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(args.device).total_memory
            ),
        },
        "attention_implementation": ATTN_IMPLEMENTATION,
        "error": None,
    }
    try:
        if not artifact.get("revision_verified"):
            raise RuntimeError("local model revision receipt did not verify")

        tokenizer = load_tokenizer(
            args.model_path,
            revision=args.model_revision,
            local_files_only=True,
        )
        load_kwargs = {
            "revision": args.model_revision,
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": ATTN_IMPLEMENTATION,
        }
        model = load_text_generation_model(args.model_path, **load_kwargs).to(
            args.device
        )
        reference = load_text_generation_model(
            args.model_path, **load_kwargs
        ).to(args.device).eval()
        for parameter in reference.parameters():
            parameter.requires_grad = False
        model.gradient_checkpointing_enable()

        spec = get_environment_spec(args.environment)
        task = spec.create().get_task(args.task_index)
        rollouts = []
        traces = []
        for index in range(M_ROLLOUTS):
            actions = (
                task.private["reference_actions"]
                if index % 2 == 0
                else [AssistantAction.final("incorrect")]
            )
            trace = EpisodeRunner(
                renderer=CanonicalEpisodeRenderer(
                    lambda text: _encode(tokenizer, text)
                )
            ).run(
                spec.create(),
                task,
                seed=index,
                policy=ScriptedPolicy(actions),
            )
            traces.append(trace)
            assistant_tokens = sum(
                end - start for start, end in trace.assistant_spans
            )
            prompt_length = trace.assistant_spans[0][0]
            rollouts.append(
                RolloutSubmission(
                    tokens=list(trace.tokens),
                    reward=0.0,
                    env_name=args.environment,
                    commit={
                        "tokens": list(trace.tokens),
                        "rollout": {
                            "prompt_length": prompt_length,
                            "completion_length": len(trace.tokens) - prompt_length,
                            "success": False,
                            "total_reward": 0.0,
                            "advantage": 0.0,
                            "token_logprobs": [0.0] * assistant_tokens,
                            "episode": _episode_metadata(trace),
                        },
                    },
                )
            )

        request = BatchSubmissionRequest(
            miner_hotkey="episode-gpu-qualification",
            prompt_idx=args.task_index,
            window_start=1,
            merkle_root="0" * 64,
            rollouts=rollouts,
            checkpoint_hash=args.model_revision,
        )
        admission._WORKER_TOKENIZER = tokenizer
        admission_result = score_and_finalize_submission(
            ParsedSubmission(
                request=request,
                rollout_hashes=[bytes([index]) for index in range(M_ROLLOUTS)],
                selection_digest=b"episode-gpu-qualification",
            ),
            AdmissionRuntimeMaterials(
                canonical_prompt_tokens=list(
                    traces[0].tokens[: traces[0].assistant_spans[0][0]]
                ),
                problem=spec.create().get_problem(args.task_index),
                completion_texts=[""] * M_ROLLOUTS,
            ),
            AdmissionContext(
                randomness="aa",
                environment=args.environment,
                vocab_size=int(model.config.vocab_size),
                max_sequence_length=20000,
                eos_token_ids=(),
                canonical_force_ids=(),
                think_close_ids=(),
                bootstrap=False,
                enforce_envelope_signature=False,
                enforce_legacy_merkle=False,
            ),
            time.monotonic() + 60,
        )
        if admission_result.reject_reason is not None:
            raise RuntimeError(
                f"admission rejected: {admission_result.reject_reason}"
            )

        # Seal-time behavior logprobs over validator-admitted assistant spans.
        with torch.no_grad():
            for rollout in request.rollouts:
                tokens = torch.tensor(
                    [rollout.commit["tokens"]], device=args.device
                )
                selected = _selected_logprobs_for_tokens(
                    reference, tokens, tokens[0, 1:]
                )
                positions = _policy_token_positions(rollout)
                values = [
                    float(selected[position - 1]) for position in positions
                ]
                rollout.commit["rollout"]["token_logprobs"] = values
                rollout._validated_completion_logprobs = values

        group = SimpleNamespace(rollouts=request.rollouts, prompt_idx=args.task_index)
        payload = encode_training_payload(
            {args.environment: [group]},
            window_start=1,
            checkpoint_revision=args.model_revision,
            env_order=[args.environment],
            env_targets={args.environment: M_ROLLOUTS},
            window_quarantine={"quarantined": False, "reasons": []},
        )
        decoded = decode_training_payload(payload).batches()[args.environment]
        if not all(
            rollout._validated_assistant_spans is not None
            for rollout in decoded[0].rollouts
        ):
            raise RuntimeError("assistant spans were lost in training payload")

        captured_metrics: dict[str, Any] = {}

        def capture_metrics(values, **_kwargs):
            captured_metrics.update(values)

        telemetry.log_training_step = capture_metrics
        reset_training_state()
        torch.cuda.reset_peak_memory_stats(args.device)
        before_sha256 = _model_weight_sha256(model)
        train_step(model, [decoded], ref_model=reference, window_index=1)
        torch.cuda.synchronize(args.device)
        after_sha256 = _model_weight_sha256(model)

        report.update(
            {
                "rollouts": M_ROLLOUTS,
                "admitted_rewards": [row.reward for row in request.rollouts],
                "assistant_tokens": sum(
                    sum(end - start for start, end in trace.assistant_spans)
                    for trace in traces
                ),
                "payload_bytes": len(payload),
                "weight_sha256_before": before_sha256,
                "weight_sha256_after": after_sha256,
                "weights_changed": before_sha256 != after_sha256,
                "training_metrics": captured_metrics,
                "optimizer_steps": int(training._scheduler.last_epoch),
                "peak_vram_bytes": int(
                    torch.cuda.max_memory_allocated(args.device)
                ),
            }
        )
        report["passed"] = bool(
            before_sha256 != after_sha256
            and captured_metrics.get("train/rollouts_processed") == M_ROLLOUTS
            and training._scheduler.last_epoch >= 1
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
