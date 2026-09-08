#!/usr/bin/env python3
"""CPU conformance for two pinned native Verifiers Tasksets and core adapters.

Wheels must already be installed in a dedicated Python 3.12 environment. Hash
both wheel files before importing their code. No model, service or profile is
started, and no package is installed by this command.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import zipfile

from reliquary.environment.agentic.adapters.prime_v1 import (
    VERIFIERS_COMMIT,
    actions_from_prime_v1_trace,
    native_prime_v1_trace,
    pinned_verifiers_v1,
)
from reliquary.environment.agentic.external import (
    ExternalAnswerEnvironment,
    ExternalEpisodeEnvironment,
    load_external_backend,
)
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction, canonical_json
from reliquary.environment.registry import get_environment_spec

LOGIC_WHEEL_SHA256 = "d12e4258fa9b190f29a33a055cee11ae632d813aed5f51b28a20a3727212b527"
STATEFUL_WHEEL_SHA256 = "f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155"


def checked_goldens(path: Path, expected: str, environment: str) -> list[dict]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("qualification wheel exceeds size limit")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError("qualification wheel SHA-256 mismatch; no environment imported")
    spec = get_environment_spec(environment)
    with zipfile.ZipFile(path) as archive:
        artifact = json.loads(archive.read(spec.external_artifact_resource))
        if hashlib.sha256(canonical_json(artifact).encode()).hexdigest() != spec.environment_manifest_sha256:
            raise ValueError("wheel artifact does not match the catalog")
        resource = spec.external_artifact_resource.rsplit("/", 1)[0]
        return [json.loads(line) for line in archive.read(
            resource + "/goldens/reference.jsonl"
        ).splitlines() if line]


def _score_roundtrip(vf, task, trace, expected: float) -> object:
    asyncio.run(task.score(trace))
    if trace.reward != expected:
        raise AssertionError(f"native Verifiers reward mismatch for {task.key}")
    wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
    # Recompute; never accept a reward simply copied through the wire.
    wire.rewards.clear()
    asyncio.run(task.score(wire))
    if wire.reward != expected:
        raise AssertionError(f"wire Verifiers reward mismatch for {task.key}")
    return wire


def qualify(logic_wheel: Path, stateful_wheel: Path) -> dict:
    goldens = {
        "reliquary_logic_v2": checked_goldens(logic_wheel, LOGIC_WHEEL_SHA256, "reliquary_logic_v2"),
        "reliquary_stateful_tools_v2": checked_goldens(stateful_wheel, STATEFUL_WHEEL_SHA256, "reliquary_stateful_tools_v2"),
    }
    vf = pinned_verifiers_v1()
    results = []
    for name, taskset_id, family_count in (
        ("reliquary_logic_v2", "reliquary-logic", 12),
        ("reliquary_stateful_tools_v2", "reliquary-stateful-tools", 3),
    ):
        spec = get_environment_spec(name)
        split_ids = []
        for split in ("train", "eval", "qualification"):
            backend = load_external_backend(spec, split=split)
            environment = (ExternalAnswerEnvironment(backend, spec) if family_count == 12
                           else ExternalEpisodeEnvironment(backend, spec))
            config = vf.taskset_config_type(taskset_id)(id=taskset_id, split=split)
            tasks = list(vf.load_taskset(config).head(128))
            families = {}
            for index, task in enumerate(tasks):
                source = backend.task(index)
                if task.key != source["id"] or task.data.prompt != source["prompt"]:
                    raise AssertionError("native Taskset prompt/identity differs from replay ABI")
                families.setdefault(source["metadata"]["family"], index)
            if len(families) != family_count:
                raise AssertionError("qualification did not cover every declared task family")
            split_ids.append({task.key for task in tasks})
            indices = sorted(set(families.values()) | {
                row["index"] for row in goldens[name] if row["split"] == split
            })
            cases = 0
            for index in indices:
                task = tasks[index]
                if not asyncio.run(task.validate(None)):
                    raise AssertionError("native Task.validate failed")
                if family_count == 12:
                    problem = environment.get_problem(index)
                    good = backend.reference_completion(index)
                    completions = (
                        (good, 1.0),
                        ("Reasoning:\n```python\nx=1\n```\n" + good, 1.0),
                        ('```json\n{"result":"__wrong__"}\n```', 0.0),
                        ('```json\n{"result":NaN}\n```', 0.0),
                        ('```json\n{"result":1,"result":2}\n```', 0.0),
                        ('```json\n{"result":1,"extra":2}\n```', 0.0),
                        ('```json\n[1,2]\n```', 0.0), ("", 0.0),
                        ("x" * (16 * 1024 + 1), 0.0),
                    )
                    for completion, expected in completions:
                        if environment.compute_reward(problem, completion) != expected:
                            raise AssertionError("core answer reward differs from native Taskset")
                        wire = _score_roundtrip(vf, task, native_prime_v1_trace(
                            task, completion=completion), expected)
                        if environment.compute_reward(problem, wire.last_reply or "") != expected:
                            raise AssertionError("answer wire roundtrip changed the reward")
                        cases += 1
                    if environment.get_problem(index + len(environment)) != problem:
                        raise AssertionError("answer task index wrapping changed identity")
                    if environment.compute_reward({**problem, "id": "forged"}, good) != 0.0:
                        raise AssertionError("answer scorer accepted a forged task identity")
                else:
                    module = importlib.import_module("reliquary_stateful_tools.taskset")
                    actions = module._build_task(index, split)["private"]["reference_actions"]
                    for action_list, expected in ((actions, 1.0), ([{"final": "incorrect"}], 0.0),
                                                 ([{"tool": "unknown", "arguments": {}}], 0.0)):
                        episode = EpisodeRunner().run(environment, environment.get_task(index), seed=0,
                            policy=ScriptedPolicy([AssistantAction.from_wire(action) for action in action_list]))
                        if episode.reward.reward != expected:
                            raise AssertionError("core episode reward differs from native Taskset")
                        wire = _score_roundtrip(vf, task, native_prime_v1_trace(task, episode=episode), expected)
                        recovered = actions_from_prime_v1_trace(wire)
                        replay = EpisodeRunner().run(environment, environment.get_task(index), seed=0,
                                                     policy=ScriptedPolicy(recovered))
                        if replay.reward != episode.reward or replay.actions != episode.actions:
                            raise AssertionError("episode wire roundtrip changed actions or authoritative outcome")
                        cases += 1
                for golden in goldens[name]:
                    if golden["split"] == split and golden["index"] == index:
                        source = backend.task(index)
                        if source["id"] != golden["task_id"]:
                            raise AssertionError("published golden task identity changed")
                        if "prompt_sha256" in golden and hashlib.sha256(source["prompt"].encode()).hexdigest() != golden["prompt_sha256"]:
                            raise AssertionError("published golden prompt changed")
                        reward = (backend.grade(index, backend.reference_completion(index)) if family_count == 12
                                  else backend.replay(index, module._build_task(index, split)["private"]["reference_actions"])["reward"])
                        if reward["state_digest"] != golden["state_digest"]:
                            raise AssertionError("published golden state digest changed")
            results.append({"environment": name, "split": split, "families": sorted(families),
                            "tasks": len(indices), "cases": cases,
                            "artifact_sha256": spec.environment_manifest_sha256})
        if any(split_ids[a] & split_ids[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise AssertionError("Taskset splits overlap")
    return {"qualification": "CPU native Taskset/Trace/WireTrace conformance only",
            "verifiers_commit": VERIFIERS_COMMIT, "profile_activated": False,
            "wheel_sha256": {"logic": LOGIC_WHEEL_SHA256, "stateful_tools": STATEFUL_WHEEL_SHA256},
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logic-wheel", type=Path, required=True)
    parser.add_argument("--stateful-wheel", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.logic_wheel, args.stateful_wheel), indent=2))


if __name__ == "__main__":
    main()
