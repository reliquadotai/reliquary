#!/usr/bin/env python3
"""Download/hash the pinned wheel, then replay its installed adapter on CPU.

Install the exact Verifiers revision and verified wheel in a dedicated Python
3.12 environment first (see docs/standalone-environment-qualification.md).
This check never installs packages, enables profiles, or exercises a GPU.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import zipfile

import requests

REPOSITORY = "https://github.com/reliquadotai/reliquary-environments"
RELEASE = "v0.1.0a1"
RELEASE_COMMIT = "cdb998b355ee41355823f2a15cdd21ab19c4718a"
INTEGRATION_COMMIT = "73285b192f0359ed75635d6d4ff1694ab3a1b106"
VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
WHEEL_NAME = "reliquary_stateful_tools-0.1.0a1-py3-none-any.whl"
WHEEL_URL = f"{REPOSITORY}/releases/download/{RELEASE}/{WHEEL_NAME}"
WHEEL_SHA256 = "f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155"


def download_pinned_wheel(directory: Path) -> Path:
    """Check bytes before publishing a local wheel or executing package code."""
    digest = hashlib.sha256()
    descriptor, filename = tempfile.mkstemp(prefix="external-wheel-", dir=directory)
    partial = Path(filename)
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as stream, requests.get(WHEEL_URL, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > 16 * 1024 * 1024:
                    raise ValueError("external wheel exceeds size limit")
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != WHEEL_SHA256:
            raise ValueError("external wheel SHA-256 mismatch; no code imported")
        wheel = directory / WHEEL_NAME
        if wheel.exists():
            raise FileExistsError(wheel)
        os.link(partial, wheel)
        return wheel
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
    from reliquary.environment.agentic.types import AssistantAction
    from reliquary.environment.registry import get_environment_spec
    from reliquary.protocol.profiles import PROFILES

    installed = importlib.metadata.distribution("verifiers")
    direct_url = json.loads(installed.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != VERIFIERS_COMMIT:
        raise ValueError("external replay requires the exact pinned Verifiers commit")
    with tempfile.TemporaryDirectory(prefix="reliquary-external-wheel-") as temporary:
        wheel = download_pinned_wheel(Path(temporary))
        with zipfile.ZipFile(wheel) as archive:
            goldens = [json.loads(line) for line in archive.read(
                "reliquary_stateful_tools/goldens/reference.jsonl"
            ).splitlines() if line]
        spec = get_environment_spec("reliquary_stateful_tools_v2")
        assert all(spec.name not in profile.environments for profile in PROFILES.values())
        environment = spec.create()
        taskset = importlib.import_module("reliquary_stateful_tools.taskset")
        results = []
        for index, golden in enumerate(goldens):
            task = environment.get_task(index)
            actions = taskset._build_task(index, "train")["private"]["reference_actions"]
            trace = EpisodeRunner().run(environment, task, seed=0, policy=ScriptedPolicy([AssistantAction.from_wire(action) for action in actions]))
            assert trace.reward is not None and trace.reward.reward == 1.0
            assert trace.task_id == golden["task_id"]
            assert trace.reward.state_digest == golden["state_digest"]
            bad = EpisodeRunner().run(environment, task, seed=0,
                                      policy=ScriptedPolicy([AssistantAction.final("incorrect")]))
            assert bad.reward is not None and bad.reward.reward == 0.0
            results.append({"index": index, "task_id": trace.task_id,
                            "reward": trace.reward.reward, "state_digest": trace.reward.state_digest,
                            "golden": golden})
        print(json.dumps({"qualification": "CPU replay only", "repository": REPOSITORY,
                          "release": RELEASE, "release_commit": RELEASE_COMMIT,
                          "integration_commit": INTEGRATION_COMMIT,
                          "wheel_sha256": WHEEL_SHA256,
                          "artifact_sha256": spec.environment_manifest_sha256,
                          "profile_active": False, "results": results}, indent=2))


if __name__ == "__main__":
    main()
