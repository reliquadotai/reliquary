#!/usr/bin/env python3
"""Qualify Reliquary Episode v1 on a CPU host or with a real local model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer  # noqa: E402
from reliquary.environment.agentic.replay import replay_tokenized_episode  # noqa: E402
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy  # noqa: E402
from reliquary.environment.agentic.suite import BUILTIN_EPISODE_ENVIRONMENTS  # noqa: E402
from reliquary.environment.registry import (  # noqa: E402
    environment_manifest_sha256,
    get_environment_spec,
)


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _byte_decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


def qualify_cpu(tasks_per_env: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in BUILTIN_EPISODE_ENVIRONMENTS:
        spec = get_environment_spec(name)
        cases = []
        for index in range(tasks_per_env):
            env = spec.create()
            task = env.get_task(index)
            started = time.perf_counter()
            trace = EpisodeRunner(
                renderer=CanonicalEpisodeRenderer(_byte_encode)
            ).run(
                env,
                task,
                seed=2026,
                policy=ScriptedPolicy(task.private["reference_actions"]),
            )
            replay = replay_tokenized_episode(
                spec.create(),
                task_index=index,
                seed=2026,
                tokens=list(trace.tokens),
                assistant_spans=trace.assistant_spans,
                decode=_byte_decode,
                encode=_byte_encode,
            )
            elapsed = time.perf_counter() - started
            cases.append({
                "task_id": task.id,
                "reward": trace.reward.reward if trace.reward else None,
                "success": bool(trace.reward and trace.reward.success),
                "turns": len(trace.actions),
                "trace_digest": trace.trace_digest,
                "exact_replay": (
                    replay.tokens == trace.tokens
                    and replay.trace_digest == trace.trace_digest
                ),
                "elapsed_ms": round(elapsed * 1000.0, 3),
            })
        results[name] = {
            "passed": all(
                case["success"] and case["reward"] == 1.0 and case["exact_replay"]
                for case in cases
            ),
            "p50_ms": statistics.median(case["elapsed_ms"] for case in cases),
            "cases": cases,
        }
    return results


def qualify_model(
    model_path: str,
    *,
    device: str,
    tasks_per_env: int,
    rollouts: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from reliquary.environment.agentic.runner import EpisodeRunner
    from reliquary.miner.episode_policy import HFEpisodePolicy
    from reliquary.shared.modeling import load_text_generation_model

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_text_generation_model(
        model_path,
        torch_dtype=dtype,
    ).to(device).eval()

    def encode(text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    output: dict[str, Any] = {}
    checkpoint_hash = hashlib.sha256(model_path.encode("utf-8")).hexdigest()
    for name in BUILTIN_EPISODE_ENVIRONMENTS:
        env = get_environment_spec(name).create()
        rows = []
        for task_index in range(tasks_per_env):
            task = env.get_task(task_index)
            for rollout_index in range(rollouts):
                started = time.perf_counter()
                trace = EpisodeRunner(
                    renderer=CanonicalEpisodeRenderer(encode),
                    max_turns=env.max_turns,
                ).run(
                    env,
                    task,
                    seed=rollout_index,
                    policy=HFEpisodePolicy(
                        model=model,
                        tokenizer=tokenizer,
                        randomness="42" * 32,
                        hotkey="qualification-host",
                        prompt_idx=task_index,
                        checkpoint_hash=checkpoint_hash,
                        rollout_index=rollout_index,
                        max_action_tokens=1024,
                        max_episode_tokens=16384,
                    ),
                )
                rows.append({
                    "task_id": task.id,
                    "reward": trace.reward.reward if trace.reward else None,
                    "success": bool(trace.reward and trace.reward.success),
                    "turns": len(trace.actions),
                    "assistant_tokens": sum(
                        end - start for start, end in trace.assistant_spans
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                })
        output[name] = {"rollouts": rows}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-per-env", type=int, default=8)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.tasks_per_env <= 0 or args.rollouts <= 0:
        parser.error("task and rollout counts must be positive")

    report: dict[str, Any] = {
        "schema": "reliquary/episode-qualification/v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment_manifest_sha256": environment_manifest_sha256(
            BUILTIN_EPISODE_ENVIRONMENTS
        ),
        "cpu": qualify_cpu(args.tasks_per_env),
    }
    if args.model_path:
        report["model"] = qualify_model(
            args.model_path,
            device=args.device,
            tasks_per_env=args.tasks_per_env,
            rollouts=args.rollouts,
        )
    report["passed"] = all(
        value["passed"] for value in report["cpu"].values()
    )
    rendered = json.dumps(report, sort_keys=True, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
