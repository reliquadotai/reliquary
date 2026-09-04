from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
import sys

import pytest

from reliquary.environment.agentic.external import verify_external_artifact
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction, canonical_json
from reliquary.environment.registry import EnvironmentSpec, _build_catalog


_PACKAGE = """
class FakeEnvironment:
    name = "fake_external_v2"
    validator_authoritative_reward = True
    max_turns = 2

    def __len__(self):
        return 1

    def task(self, index):
        return {
            "id": f"fake-{index}",
            "prompt": "Add two.",
            "tools": [{
                "name": "add",
                "description": "Add a value.",
                "parameters": {"type": "object"},
            }],
            "metadata": {"family": "fake", "generator_version": "v2"},
        }

    def reset(self, index, seed):
        return {"state": {"value": index + seed}, "events": []}

    def step(self, index, state, action):
        if "tool" in action:
            state["value"] += action["arguments"]["value"]
            return {
                "state": state,
                "events": [{"role": "tool", "name": "add", "content": str(state["value"])}],
                "done": False,
                "termination_reason": None,
            }
        state["final"] = action["final"]
        return {
            "state": state,
            "events": [],
            "done": True,
            "termination_reason": "final",
        }

    def grade(self, index, state, actions):
        passed = state.get("final") == str(state["value"])
        return {
            "reward": float(passed),
            "success": passed,
            "checks": [{"name": "answer", "passed": passed, "weight": 1.0, "detail": ""}],
            "state_digest": "0" * 64,
            "environment_error": None,
        }

    def close(self, state):
        pass
"""


def test_external_wheel_adapter_verifies_then_replays_and_fails_on_drift(
    tmp_path, monkeypatch
):
    package = tmp_path / "fake_external_env"
    package.mkdir()
    implementation = package / "__init__.py"
    implementation.write_text(_PACKAGE, encoding="utf-8")
    dist_info = tmp_path / "reliquary_fake-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: reliquary-fake\nVersion: 1.0\n",
        encoding="utf-8",
    )

    artifact = {
        "schema": "reliquary/environment-artifact/v1",
        "environment": "fake_external_v2",
        "contract": "fake-external-v2",
        "distribution": {"name": "reliquary-fake", "version": "1.0"},
        "entrypoints": {
            "taskset": "fake_external_env:FakeTaskset",
            "replay": "fake_external_env:FakeEnvironment",
        },
        "source_manifest_sha256": "1" * 64,
        "files": {
            "fake_external_env/__init__.py": hashlib.sha256(
                _PACKAGE.encode("utf-8")
            ).hexdigest()
        },
    }
    (package / "artifact.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    (dist_info / "RECORD").write_text(
        "fake_external_env/__init__.py,,\n"
        "fake_external_env/artifact.json,,\n"
        "reliquary_fake-1.0.dist-info/METADATA,,\n"
        "reliquary_fake-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    spec = EnvironmentSpec(
        name="fake_external_v2",
        factory_path="fake_external_env:FakeEnvironment",
        scorer_path=(
            "reliquary.environment.agentic.suite:episode_score_many_not_supported"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="fake-external-v2",
        interaction_mode="episode",
        episode_replay_path="reliquary.environment.agentic.suite:replay_submission",
        renderer_id="reliquary-jsonl-tools-v1",
        environment_manifest_sha256=hashlib.sha256(
            canonical_json(artifact).encode("utf-8")
        ).hexdigest(),
        external_distribution="reliquary-fake",
        external_artifact_resource="fake_external_env/artifact.json",
    )

    # Catalog construction and an inactive profile do not require the wheel.
    assert tuple(_build_catalog((spec,))) == ("fake_external_v2",)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_external_env", raising=False)
    importlib.invalidate_caches()
    verify_external_artifact.cache_clear()

    env = spec.create()
    task = env.get_task(0)
    trace = EpisodeRunner().run(
        env,
        task,
        seed=0,
        policy=ScriptedPolicy(
            [
                AssistantAction.tool_call("add", value=2),
                AssistantAction.final("2"),
            ]
        ),
    )
    assert trace.reward is not None and trace.reward.reward == 1.0

    verify_external_artifact.cache_clear()
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        replace(spec, environment_manifest_sha256="0" * 64).create()
    implementation.write_text(_PACKAGE + "\n# drift\n", encoding="utf-8")
    verify_external_artifact.cache_clear()
    with pytest.raises(ValueError, match="file digest mismatch"):
        spec.create()
