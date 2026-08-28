#!/usr/bin/env python3
"""Fail-closed CPU/GPU qualification for Reliquary Episode v1."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer  # noqa: E402
from reliquary.environment.agentic.replay import replay_tokenized_episode  # noqa: E402
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy  # noqa: E402
from reliquary.environment.agentic.suite import BUILTIN_EPISODE_ENVIRONMENTS  # noqa: E402
from reliquary.environment.agentic.types import AssistantAction  # noqa: E402
from reliquary.environment.registry import (  # noqa: E402
    environment_manifest_sha256,
    get_environment_spec,
)
from reliquary.protocol.profiles import resolve_protocol_profile  # noqa: E402


DEFAULT_PROFILE = "qwen3-4b-reliquary-episode-v7-dev1"


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _byte_decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[max(0, index)])


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "revision": run("rev-parse", "HEAD"),
            "tracked_dirty": bool(
                run("status", "--porcelain", "--untracked-files=no")
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tracked_dirty": None}


def _artifact_digest(source: str, revision: str) -> dict[str, Any]:
    path = Path(source).expanduser()
    if not path.exists():
        identity = f"{source}@{revision}"
        return {
            "kind": "immutable_remote_revision",
            "source": source,
            "revision": revision,
            "sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "files": None,
            "bytes": None,
        }
    files = [path] if path.is_file() else sorted(
        value for value in path.rglob("*") if value.is_file()
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.name if path.is_file() else str(
            file_path.relative_to(path)
        )
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                file_digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return {
        "kind": "local_artifact_tree",
        "source": str(path.resolve()),
        "revision": revision,
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "safetensors")
        },
    }


def qualify_cpu(names: tuple[str, ...], tasks_per_env: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in names:
        spec = get_environment_spec(name)
        cases = []
        for index in range(tasks_per_env):
            started = time.perf_counter()
            try:
                env = spec.create()
                task = env.get_task(index)
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
                exact_replay = (
                    replay.tokens == trace.tokens
                    and replay.trace_digest == trace.trace_digest
                )
                reward = trace.reward
                case = {
                    "task_id": task.id,
                    "reward": None if reward is None else reward.reward,
                    "success": bool(reward and reward.success),
                    "turns": len(trace.actions),
                    "trace_digest": trace.trace_digest,
                    "exact_replay": exact_replay,
                    "error": None,
                }
            except Exception as exc:  # qualification reports every failure
                case = {
                    "task_id": None,
                    "reward": None,
                    "success": False,
                    "turns": 0,
                    "trace_digest": None,
                    "exact_replay": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            case["elapsed_ms"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            cases.append(case)
        latencies = [float(case["elapsed_ms"]) for case in cases]
        results[name] = {
            "passed": all(
                case["success"]
                and case["reward"] == 1.0
                and case["exact_replay"]
                and case["error"] is None
                for case in cases
            ),
            "p50_ms": statistics.median(latencies),
            "p95_ms": _percentile(latencies, 0.95),
            "cases": cases,
        }
    return results


def qualify_adversarial(names: tuple[str, ...]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in names:
        spec = get_environment_spec(name)
        env = spec.create()
        task = env.get_task(0)
        trace = EpisodeRunner(
            renderer=CanonicalEpisodeRenderer(_byte_encode)
        ).run(
            env,
            task,
            seed=2026,
            policy=ScriptedPolicy(task.private["reference_actions"]),
        )
        tampered = list(trace.tokens)
        first_observation = trace.assistant_spans[0][1] + 1
        tamper_detected = first_observation < len(tampered)
        if tamper_detected:
            tampered[first_observation] ^= 1
            replay = replay_tokenized_episode(
                spec.create(),
                task_index=0,
                seed=2026,
                tokens=tampered,
                assistant_spans=trace.assistant_spans,
                decode=_byte_decode,
                encode=_byte_encode,
            )
            tamper_detected = tuple(tampered) != replay.tokens
        rejected = EpisodeRunner().run(
            spec.create(),
            task,
            seed=2026,
            policy=ScriptedPolicy([AssistantAction.final("incorrect")]),
        )
        wrong_action_rejected = bool(
            rejected.reward
            and rejected.reward.reward == 0.0
            and not rejected.reward.success
        )
        results[name] = {
            "passed": tamper_detected and wrong_action_rejected,
            "tampered_observation_rejected": tamper_detected,
            "wrong_action_reward_zero": wrong_action_rejected,
        }
    return results


def summarize_model_environment(
    rows: list[dict[str, Any]],
    reward_groups: dict[int, list[float]],
    *,
    sigma_min: float,
) -> dict[str, Any]:
    eligible_groups = 0
    mixed_groups = 0
    for rewards in reward_groups.values():
        if len(set(rewards)) > 1:
            mixed_groups += 1
        if len(rewards) > 1 and statistics.pstdev(rewards) >= sigma_min:
            eligible_groups += 1
    successes = sum(row["reward"] == 1.0 for row in rows)
    failures = sum(row["reward"] == 0.0 for row in rows)
    latencies = [
        float(row["elapsed_seconds"])
        for row in rows
        if row["error"] is None
    ]
    return {
        "passed": bool(
            rows
            and all(
                row["error"] is None and row["exact_replay"]
                for row in rows
            )
            and successes > 0
            and failures > 0
            and eligible_groups > 0
        ),
        "successes": successes,
        "failures": failures,
        "mixed_groups": mixed_groups,
        "grpo_eligible_groups": eligible_groups,
        "sigma_min": sigma_min,
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "p50_seconds": statistics.median(latencies) if latencies else None,
        "p95_seconds": _percentile(latencies, 0.95),
        "rollouts": rows,
    }


def qualify_model(
    model_path: str,
    *,
    model_revision: str,
    profile_id: str,
    names: tuple[str, ...],
    device: str,
    tasks_per_env: int,
    rollouts: int,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    import torch

    from reliquary.miner.episode_policy import HFEpisodePolicy
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    profile = resolve_protocol_profile(profile_id)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = load_tokenizer(model_path, revision=model_revision)
    model = load_text_generation_model(
        model_path,
        revision=model_revision,
        torch_dtype=dtype,
    ).to(device).eval()
    resolved_commit = getattr(model.config, "_commit_hash", None)
    revision_verified = bool(
        artifact["kind"] == "local_artifact_tree"
        or resolved_commit == model_revision
    )

    def encode(text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    def decode(tokens: list[int]) -> str:
        return str(tokenizer.decode(tokens, skip_special_tokens=False))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    output: dict[str, Any] = {
        "artifact": artifact,
        "resolved_commit": resolved_commit,
        "revision_verified": revision_verified,
        "environments": {},
    }
    sigma_min = 0.24 if profile.protocol_version >= 4 else 0.43
    for name in names:
        episode_profile = profile.environments[name].episode
        if episode_profile is None:
            raise ValueError(f"profile has no Episode v1 limits for {name}")
        rows = []
        reward_groups: dict[int, list[float]] = {}
        for task_index in range(tasks_per_env):
            for rollout_index in range(rollouts):
                started = time.perf_counter()
                try:
                    env = get_environment_spec(name).create()
                    task = env.get_task(task_index)
                    trace = EpisodeRunner(
                        renderer=CanonicalEpisodeRenderer(encode),
                        max_turns=min(env.max_turns, episode_profile.max_turns),
                        max_episode_tokens=episode_profile.max_episode_tokens,
                        max_observation_bytes=(
                            episode_profile.max_observation_bytes
                        ),
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
                            checkpoint_hash=artifact["sha256"],
                            rollout_index=rollout_index,
                            max_action_tokens=(
                                episode_profile.max_action_tokens
                            ),
                            max_episode_tokens=(
                                episode_profile.max_episode_tokens
                            ),
                        ),
                    )
                    replay = replay_tokenized_episode(
                        get_environment_spec(name).create(),
                        task_index=task_index,
                        seed=rollout_index,
                        tokens=list(trace.tokens),
                        assistant_spans=trace.assistant_spans,
                        decode=decode,
                        encode=encode,
                        max_episode_tokens=episode_profile.max_episode_tokens,
                        max_observation_bytes=(
                            episode_profile.max_observation_bytes
                        ),
                    )
                    reward = None if trace.reward is None else trace.reward.reward
                    if reward is not None:
                        reward_groups.setdefault(task_index, []).append(reward)
                    row = {
                        "task_id": task.id,
                        "task_index": task_index,
                        "rollout_index": rollout_index,
                        "reward": reward,
                        "success": bool(trace.reward and trace.reward.success),
                        "turns": len(trace.actions),
                        "assistant_tokens": sum(
                            end - start for start, end in trace.assistant_spans
                        ),
                        "total_tokens": len(trace.tokens),
                        "invalid_actions": sum(
                            action.tool == "__invalid_action__"
                            for action in trace.actions
                        ),
                        "termination_reason": trace.termination_reason,
                        "trace_digest": trace.trace_digest,
                        "exact_replay": (
                            replay.tokens == trace.tokens
                            and replay.trace_digest == trace.trace_digest
                        ),
                        "error": None,
                    }
                except Exception as exc:
                    row = {
                        "task_id": None,
                        "task_index": task_index,
                        "rollout_index": rollout_index,
                        "reward": None,
                        "success": False,
                        "turns": 0,
                        "assistant_tokens": 0,
                        "total_tokens": 0,
                        "invalid_actions": 0,
                        "termination_reason": "error",
                        "trace_digest": None,
                        "exact_replay": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                row["elapsed_seconds"] = round(
                    time.perf_counter() - started, 3
                )
                rows.append(row)
        output["environments"][name] = summarize_model_environment(
            rows,
            reward_groups,
            sigma_min=sigma_min,
        )
    if device.startswith("cuda"):
        output["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    output["passed"] = revision_verified and all(
        value["passed"] for value in output["environments"].values()
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-per-env", type=int, default=8)
    parser.add_argument("--environment", action="append", dest="environments")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model-path")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.tasks_per_env <= 0 or args.rollouts <= 0:
        parser.error("task and rollout counts must be positive")
    names = tuple(args.environments or BUILTIN_EPISODE_ENVIRONMENTS)
    if not names or len(set(names)) != len(names):
        parser.error("environment selection must be non-empty and unique")
    unknown = set(names) - set(BUILTIN_EPISODE_ENVIRONMENTS)
    if unknown:
        parser.error(f"unknown Episode v1 environments: {sorted(unknown)}")
    profile = resolve_protocol_profile(args.profile)
    inactive = set(names) - set(profile.environments)
    if inactive:
        parser.error(f"profile does not declare environments: {sorted(inactive)}")
    if args.model_revision and not args.model_path:
        parser.error("--model-revision requires --model-path")
    model_revision = args.model_revision or profile.model_revision

    cpu = qualify_cpu(names, args.tasks_per_env)
    adversarial = qualify_adversarial(names)
    report: dict[str, Any] = {
        "schema": "reliquary/episode-qualification/v2",
        "git": _git_identity(),
        "runtime": _runtime_identity(),
        "profile_id": profile.profile_id,
        "generation_contract_sha256": hashlib.sha256(
            json.dumps(
                profile.to_generation_contract(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "environments": list(names),
        "environment_manifest_sha256": environment_manifest_sha256(names),
        "cpu": cpu,
        "adversarial": adversarial,
    }
    if args.model_path:
        report["model"] = qualify_model(
            args.model_path,
            model_revision=model_revision,
            profile_id=profile.profile_id,
            names=names,
            device=args.device,
            tasks_per_env=args.tasks_per_env,
            rollouts=args.rollouts,
            artifact=_artifact_digest(args.model_path, model_revision),
        )
    report["passed"] = bool(
        all(value["passed"] for value in cpu.values())
        and all(value["passed"] for value in adversarial.values())
        and ("model" not in report or report["model"]["passed"])
    )
    rendered = json.dumps(report, sort_keys=True, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
