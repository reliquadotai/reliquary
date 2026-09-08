from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from reliquary.environment.abi import (
    EnvironmentManifest,
    EnvironmentRegistry,
    TaskEnvelope,
    TrajectoryEnvelope,
    TrajectoryEvent,
)


def _manifest(**updates) -> EnvironmentManifest:
    values = {
        "environment_id": "example_tools_v1",
        "revision": "example-tools-2026-09-01",
        "interaction_mode": "episode",
        "task_schema": "example/task/v1",
        "trajectory_schema": "reliquary/trajectory/v1",
        "renderer_id": "chat-events/v1",
        "verifier_id": "deterministic-replay/v1",
        "reward_policy_id": "weighted-checks/v1",
        "implementation_sha256": "11" * 32,
        "task_source_sha256": "22" * 32,
        "adapter_ids": ("prime-verifiers/v1", "native/v1"),
    }
    values.update(updates)
    return EnvironmentManifest(**values)


def _task(manifest: EnvironmentManifest, **updates) -> TaskEnvelope:
    values = {
        "environment_id": manifest.environment_id,
        "environment_manifest_sha256": manifest.sha256,
        "task_id": "task-7",
        "task_index": 7,
        "task_seed": "33" * 32,
        "payload": {
            "prompt": "Use the tools, then answer.",
            "tools": [{"name": "lookup", "arguments": {"type": "object"}}],
        },
    }
    values.update(updates)
    return TaskEnvelope(**values)


def test_environment_manifest_is_canonical_strict_and_hash_stable():
    manifest = _manifest()
    reordered = _manifest(adapter_ids=("native/v1", "prime-verifiers/v1"))

    assert manifest.canonical_bytes == reordered.canonical_bytes
    assert manifest.sha256 == reordered.sha256
    assert EnvironmentManifest.parse(manifest.canonical_bytes) == manifest

    value = manifest.to_dict()
    value["local_import_path"] = "untrusted.module:Environment"
    with pytest.raises(ValueError, match="extra=local_import_path"):
        EnvironmentManifest.parse(json.dumps(value))


def test_environment_manifest_rejects_duplicate_keys_and_nonfinite_values():
    raw = _manifest().canonical_bytes.decode("utf-8")
    duplicate = raw[:-1] + ',"schema":"reliquary/environment-manifest/v1"}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        EnvironmentManifest.parse(duplicate)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _manifest(implementation_sha256="AA" * 32)


def test_task_envelope_is_deeply_immutable_and_binds_manifest_and_seed():
    manifest = _manifest()
    payload = {"prompt": "hello", "nested": {"values": [1, 2]}}
    task = _task(manifest, payload=payload)
    payload["prompt"] = "mutated after construction"

    assert task.payload["prompt"] == "hello"
    assert tuple(task.payload["nested"]["values"]) == (1, 2)
    with pytest.raises(TypeError):
        task.payload["prompt"] = "cannot mutate"
    with pytest.raises(FrozenInstanceError):
        task.task_id = "changed"
    assert TaskEnvelope.parse(task.canonical_bytes) == task

    changed_seed = _task(manifest, task_seed="44" * 32)
    changed_manifest = _task(_manifest(revision="example-tools-2026-09-02"))
    assert changed_seed.sha256 != task.sha256
    assert changed_manifest.sha256 != task.sha256


def test_trajectory_round_trip_binds_semantics_without_runtime_fields():
    manifest = _manifest()
    task = _task(manifest)
    trajectory = TrajectoryEnvelope(
        environment_id=manifest.environment_id,
        environment_manifest_sha256=manifest.sha256,
        task_sha256=task.sha256,
        events=(
            TrajectoryEvent(role="user", content="Find the record."),
            TrajectoryEvent(role="assistant", content='{"query":"record"}'),
            TrajectoryEvent(role="tool", name="lookup", content="record=7"),
            TrajectoryEvent(role="assistant", content="The answer is 7."),
        ),
        termination_reason="final",
        metadata={"turns": 2},
    )

    assert TrajectoryEnvelope.parse(trajectory.canonical_bytes) == trajectory
    changed = TrajectoryEnvelope(
        environment_id=trajectory.environment_id,
        environment_manifest_sha256=trajectory.environment_manifest_sha256,
        task_sha256=trajectory.task_sha256,
        events=trajectory.events[:-1]
        + (TrajectoryEvent(role="assistant", content="The answer is 8."),),
        termination_reason=trajectory.termination_reason,
        metadata=trajectory.metadata,
    )
    assert changed.sha256 != trajectory.sha256


def test_registry_requires_both_release_allowlist_and_exact_installed_manifest():
    manifest = _manifest()

    class Adapter:
        def __init__(self, bound: EnvironmentManifest) -> None:
            self._manifest = bound

        @property
        def manifest(self) -> EnvironmentManifest:
            return self._manifest

        def task(self, *, index: int, seed: str) -> TaskEnvelope:
            return _task(self._manifest, task_index=index, task_seed=seed)

        def replay(self, trajectory: TrajectoryEnvelope):
            return trajectory.sha256

    registry = EnvironmentRegistry({manifest.environment_id: manifest.sha256})
    adapter = Adapter(manifest)
    registry.register(adapter)

    assert registry.resolve(manifest.environment_id) is adapter
    assert dict(registry.allowed) == {manifest.environment_id: manifest.sha256}
    with pytest.raises(ValueError, match="already registered"):
        registry.register(adapter)
    with pytest.raises(ValueError, match="not allowed"):
        EnvironmentRegistry({manifest.environment_id: "00" * 32}).register(
            Adapter(manifest)
        )
