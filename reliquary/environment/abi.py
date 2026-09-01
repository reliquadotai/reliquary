"""Dependency-light environment interchange contracts.

This module defines the small consensus boundary shared by validators, miners,
trainers, and optional external environment adapters.  It deliberately does
not import or execute environment packages.  Runtime loading is an operator
decision; protocol eligibility is an exact manifest-hash decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable


ENVIRONMENT_MANIFEST_SCHEMA = "reliquary/environment-manifest/v1"
TASK_ENVELOPE_SCHEMA = "reliquary/task/v1"
TRAJECTORY_ENVELOPE_SCHEMA = "reliquary/trajectory/v1"

InteractionMode = Literal["single_turn", "episode"]
EventRole = Literal["system", "user", "assistant", "tool"]

_HEX = frozenset("0123456789abcdef")
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_EVENT_BYTES = 64 * 1024


def _require_text(name: str, value: Any, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
    ):
        raise ValueError(f"{name} must be non-empty, trimmed text")
    return value


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze_json(value: Any, *, path: str = "value") -> Any:
    """Validate JSON-native data and return a recursively immutable copy."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            if key in frozen:
                raise ValueError(f"{path} contains a duplicate object key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} is not JSON-native")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding used by environment contracts."""
    encoded = json.dumps(
        _thaw_json(_freeze_json(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_DOCUMENT_BYTES:
        raise ValueError("canonical environment document exceeds the byte limit")
    return encoded


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _parse_object(raw: bytes | str) -> dict[str, Any]:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if len(encoded) > _MAX_DOCUMENT_BYTES:
        raise ValueError("environment document exceeds the byte limit")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        encoded,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("environment document must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise ValueError(f"document fields mismatch; missing={missing}; extra={extra}")


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    """Consensus identity for an installed environment implementation.

    Local import paths, worker counts, credentials, and inference runtimes are
    intentionally absent.  A release contract may allow this environment only
    by binding :attr:`sha256`.
    """

    environment_id: str
    revision: str
    interaction_mode: InteractionMode
    task_schema: str
    trajectory_schema: str
    renderer_id: str
    verifier_id: str
    reward_policy_id: str
    implementation_sha256: str
    task_source_sha256: str
    adapter_ids: tuple[str, ...] = ()
    schema: str = ENVIRONMENT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported environment manifest schema: {self.schema}")
        _require_text("environment_id", self.environment_id)
        _require_text("revision", self.revision)
        if self.interaction_mode not in ("single_turn", "episode"):
            raise ValueError("unsupported interaction mode")
        _require_text("task_schema", self.task_schema)
        _require_text("trajectory_schema", self.trajectory_schema)
        _require_text("renderer_id", self.renderer_id)
        _require_text("verifier_id", self.verifier_id)
        _require_text("reward_policy_id", self.reward_policy_id)
        _require_sha256("implementation_sha256", self.implementation_sha256)
        _require_sha256("task_source_sha256", self.task_source_sha256)
        adapters = tuple(self.adapter_ids)
        if len(adapters) != len(set(adapters)):
            raise ValueError("adapter ids must be unique")
        for adapter_id in adapters:
            _require_text("adapter_id", adapter_id)
        object.__setattr__(self, "adapter_ids", tuple(sorted(adapters)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "environment_id": self.environment_id,
            "revision": self.revision,
            "interaction_mode": self.interaction_mode,
            "task_schema": self.task_schema,
            "trajectory_schema": self.trajectory_schema,
            "renderer_id": self.renderer_id,
            "verifier_id": self.verifier_id,
            "reward_policy_id": self.reward_policy_id,
            "implementation_sha256": self.implementation_sha256,
            "task_source_sha256": self.task_source_sha256,
            "adapter_ids": list(self.adapter_ids),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def parse(cls, raw: bytes | str) -> "EnvironmentManifest":
        value = _parse_object(raw)
        _exact_keys(
            value,
            {
                "schema",
                "environment_id",
                "revision",
                "interaction_mode",
                "task_schema",
                "trajectory_schema",
                "renderer_id",
                "verifier_id",
                "reward_policy_id",
                "implementation_sha256",
                "task_source_sha256",
                "adapter_ids",
            },
        )
        adapters = value["adapter_ids"]
        if not isinstance(adapters, list) or not all(
            isinstance(item, str) for item in adapters
        ):
            raise ValueError("adapter_ids must be an array of strings")
        return cls(
            schema=value["schema"],
            environment_id=value["environment_id"],
            revision=value["revision"],
            interaction_mode=value["interaction_mode"],
            task_schema=value["task_schema"],
            trajectory_schema=value["trajectory_schema"],
            renderer_id=value["renderer_id"],
            verifier_id=value["verifier_id"],
            reward_policy_id=value["reward_policy_id"],
            implementation_sha256=value["implementation_sha256"],
            task_source_sha256=value["task_source_sha256"],
            adapter_ids=tuple(adapters),
        )


@dataclass(frozen=True, slots=True)
class TaskEnvelope:
    """Public deterministic task material passed to a generation runtime."""

    environment_id: str
    environment_manifest_sha256: str
    task_id: str
    task_index: int
    task_seed: str
    payload: Mapping[str, Any] = field(repr=False)
    schema: str = TASK_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TASK_ENVELOPE_SCHEMA:
            raise ValueError(f"unsupported task schema: {self.schema}")
        _require_text("environment_id", self.environment_id)
        _require_sha256(
            "environment_manifest_sha256", self.environment_manifest_sha256
        )
        _require_text("task_id", self.task_id)
        if not isinstance(self.task_index, int) or isinstance(self.task_index, bool):
            raise TypeError("task_index must be an integer")
        if self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        _require_sha256("task_seed", self.task_seed)
        object.__setattr__(self, "payload", _freeze_json(self.payload, path="payload"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "environment_id": self.environment_id,
            "environment_manifest_sha256": self.environment_manifest_sha256,
            "task_id": self.task_id,
            "task_index": self.task_index,
            "task_seed": self.task_seed,
            "payload": _thaw_json(self.payload),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def parse(cls, raw: bytes | str) -> "TaskEnvelope":
        value = _parse_object(raw)
        _exact_keys(
            value,
            {
                "schema",
                "environment_id",
                "environment_manifest_sha256",
                "task_id",
                "task_index",
                "task_seed",
                "payload",
            },
        )
        if not isinstance(value["payload"], dict):
            raise ValueError("task payload must be an object")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    role: EventRole
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("unsupported trajectory event role")
        if not isinstance(self.content, str):
            raise TypeError("event content must be text")
        if len(self.content.encode("utf-8")) > _MAX_EVENT_BYTES:
            raise ValueError("event content exceeds the byte limit")
        if self.role == "tool" and not self.name:
            raise ValueError("tool events require a name")
        if self.name is not None:
            _require_text("event name", self.name)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            value["name"] = self.name
        return value


@dataclass(frozen=True, slots=True)
class TrajectoryEnvelope:
    """Portable semantic trace; token-authenticity data remains separate."""

    environment_id: str
    environment_manifest_sha256: str
    task_sha256: str
    events: tuple[TrajectoryEvent, ...]
    termination_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    schema: str = TRAJECTORY_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAJECTORY_ENVELOPE_SCHEMA:
            raise ValueError(f"unsupported trajectory schema: {self.schema}")
        _require_text("environment_id", self.environment_id)
        _require_sha256(
            "environment_manifest_sha256", self.environment_manifest_sha256
        )
        _require_sha256("task_sha256", self.task_sha256)
        _require_text("termination_reason", self.termination_reason)
        events = tuple(self.events)
        if not events or not all(isinstance(event, TrajectoryEvent) for event in events):
            raise ValueError("trajectory events must be non-empty TrajectoryEvent values")
        object.__setattr__(self, "events", events)
        object.__setattr__(
            self, "metadata", _freeze_json(self.metadata, path="metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "environment_id": self.environment_id,
            "environment_manifest_sha256": self.environment_manifest_sha256,
            "task_sha256": self.task_sha256,
            "events": [event.to_dict() for event in self.events],
            "termination_reason": self.termination_reason,
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def parse(cls, raw: bytes | str) -> "TrajectoryEnvelope":
        value = _parse_object(raw)
        _exact_keys(
            value,
            {
                "schema",
                "environment_id",
                "environment_manifest_sha256",
                "task_sha256",
                "events",
                "termination_reason",
                "metadata",
            },
        )
        raw_events = value["events"]
        if not isinstance(raw_events, list):
            raise ValueError("events must be an array")
        events: list[TrajectoryEvent] = []
        for item in raw_events:
            if not isinstance(item, dict) or set(item) not in (
                {"role", "content"},
                {"role", "content", "name"},
            ):
                raise ValueError("invalid trajectory event")
            events.append(TrajectoryEvent(**item))
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("trajectory metadata must be an object")
        return cls(
            schema=value["schema"],
            environment_id=value["environment_id"],
            environment_manifest_sha256=value["environment_manifest_sha256"],
            task_sha256=value["task_sha256"],
            events=tuple(events),
            termination_reason=value["termination_reason"],
            metadata=metadata,
        )


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Optional local adapter behind an exact, pre-allowed manifest."""

    @property
    def manifest(self) -> EnvironmentManifest: ...

    def task(self, *, index: int, seed: str) -> TaskEnvelope: ...

    def replay(self, trajectory: TrajectoryEnvelope) -> Any: ...


class EnvironmentRegistry:
    """Two-key registry: release allowlist plus locally installed adapter."""

    def __init__(self, allowed: Mapping[str, str]) -> None:
        checked: dict[str, str] = {}
        for environment_id, digest in allowed.items():
            _require_text("environment_id", environment_id)
            checked[environment_id] = _require_sha256("manifest sha256", digest)
        self._allowed = MappingProxyType(checked)
        self._adapters: dict[str, EnvironmentAdapter] = {}

    @property
    def allowed(self) -> Mapping[str, str]:
        return self._allowed

    def register(self, adapter: EnvironmentAdapter) -> None:
        if not isinstance(adapter, EnvironmentAdapter):
            raise TypeError("adapter does not implement EnvironmentAdapter")
        manifest = adapter.manifest
        expected = self._allowed.get(manifest.environment_id)
        if expected is None or manifest.sha256 != expected:
            raise ValueError("environment manifest is not allowed by this release")
        if manifest.environment_id in self._adapters:
            raise ValueError("environment adapter is already registered")
        self._adapters[manifest.environment_id] = adapter

    def resolve(self, environment_id: str) -> EnvironmentAdapter:
        try:
            return self._adapters[environment_id]
        except KeyError as exc:
            raise KeyError(f"environment adapter is not installed: {environment_id}") from exc


__all__ = [
    "ENVIRONMENT_MANIFEST_SCHEMA",
    "TASK_ENVELOPE_SCHEMA",
    "TRAJECTORY_ENVELOPE_SCHEMA",
    "EnvironmentAdapter",
    "EnvironmentManifest",
    "EnvironmentRegistry",
    "TaskEnvelope",
    "TrajectoryEnvelope",
    "TrajectoryEvent",
    "canonical_json_bytes",
    "canonical_sha256",
]
