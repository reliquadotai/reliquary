"""Strict bridge from a pinned standalone wheel to Episode v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import importlib.metadata
import importlib.abc
import importlib.util
import sys
import threading
import json
from pathlib import Path, PurePosixPath
from typing import Any, TYPE_CHECKING

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardCheck,
    RewardReport,
    StepResult,
    ToolSpec,
    canonical_json,
)

if TYPE_CHECKING:
    from reliquary.environment.registry import EnvironmentSpec


ARTIFACT_SCHEMA = "reliquary/environment-artifact/v1"


def _object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    canonical_json(dict(value))
    keys = set(value)
    if not all(isinstance(key, str) for key in keys):
        raise TypeError(f"{name} fields must be strings")
    if not required <= keys or not keys <= required | optional:
        raise ValueError(f"invalid {name} fields: {sorted(keys)}")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _number(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate artifact key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite artifact value: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("environment artifact must be an object")
    return value


def verify_external_artifact(spec: EnvironmentSpec) -> Mapping[str, Any]:
    """Verify one installed distribution before any of its code is imported."""

    distribution_name = spec.external_distribution
    artifact_resource = spec.external_artifact_resource
    if not distribution_name or not artifact_resource:
        raise ValueError(f"environment {spec.name!r} has no external artifact")
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(
            f"external environment distribution {distribution_name!r} is not installed"
        ) from exc

    root = Path(distribution.locate_file("")).resolve()
    resource = PurePosixPath(artifact_resource)
    if resource.is_absolute() or ".." in resource.parts or len(resource.parts) < 2:
        raise ValueError("external artifact resource must be package-relative")
    located_artifact = Path(distribution.locate_file(str(resource)))
    if located_artifact.is_symlink():
        raise ValueError("external artifact resource must not be a symlink")
    artifact_path = located_artifact.resolve()
    if root not in artifact_path.parents:
        raise ValueError("external artifact resource escapes its distribution")
    if not artifact_path.is_file():
        raise ValueError(
            f"external environment artifact is missing: {artifact_resource}"
        )

    artifact = _load_json_object(artifact_path)
    _object(
        artifact,
        name="artifact",
        required={
            "schema",
            "environment",
            "contract",
            "distribution",
            "entrypoints",
            "source_manifest_sha256",
            "files",
        },
    )
    if artifact["schema"] != ARTIFACT_SCHEMA:
        raise ValueError("unsupported external environment artifact schema")
    if artifact["environment"] != spec.name:
        raise ValueError("external environment artifact name mismatch")
    if artifact["contract"] != spec.contract_version:
        raise ValueError("external environment artifact contract mismatch")
    actual_manifest_digest = _sha256(canonical_json(artifact).encode("utf-8"))
    if actual_manifest_digest != spec.environment_manifest_sha256:
        raise ValueError("external environment artifact digest mismatch")
    _digest(artifact["source_manifest_sha256"], name="source manifest digest")

    identity = _object(
        artifact["distribution"],
        name="artifact distribution",
        required={"name", "version"},
    )
    if identity["name"] != distribution_name:
        raise ValueError("external environment distribution name mismatch")
    if distribution.metadata["Name"] != distribution_name:
        raise ValueError("installed external distribution name mismatch")
    if identity["version"] != distribution.version:
        raise ValueError("external environment distribution version mismatch")

    entrypoints = _object(
        artifact["entrypoints"],
        name="artifact entrypoints",
        required={"taskset", "replay"},
    )
    if entrypoints["replay"] != spec.factory_path:
        raise ValueError("external environment replay entrypoint mismatch")
    for name, value in entrypoints.items():
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError(f"invalid external {name} entrypoint")

    files = artifact["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("external artifact files must be a non-empty object")
    package = resource.parts[0]
    bound_paths: set[str] = set()
    for relative, expected in files.items():
        if not isinstance(relative, str):
            raise ValueError("external artifact file path must be a string")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != package
            or relative_path == resource
        ):
            raise ValueError(f"unsafe external artifact file: {relative}")
        expected_digest = _digest(expected, name=f"digest for {relative}")
        located = Path(distribution.locate_file(relative))
        if located.is_symlink():
            raise ValueError(
                f"external artifact file must not be a symlink: {relative}"
            )
        installed = located.resolve()
        if root not in installed.parents or not installed.is_file():
            raise ValueError(f"external artifact file is missing or unsafe: {relative}")
        if _sha256(installed.read_bytes()) != expected_digest:
            raise ValueError(f"external artifact file digest mismatch: {relative}")
        bound_paths.add(relative_path.as_posix())

    recorded_paths = {
        PurePosixPath(str(path)).as_posix() for path in (distribution.files or ())
    }
    if not ({resource.as_posix()} | bound_paths) <= recorded_paths:
        raise ValueError(
            "external artifact contains files not owned by its distribution"
        )

    for entrypoint in entrypoints.values():
        module_name = entrypoint.partition(":")[0].replace(".", "/")
        if not ({f"{module_name}.py", f"{module_name}/__init__.py"} & bound_paths):
            raise ValueError(
                f"external entrypoint module is not artifact-bound: {entrypoint}"
            )
    return artifact


def _events(value: Any) -> tuple[EpisodeEvent, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("external events must be a list")
    events = []
    for raw in value:
        event = _object(
            raw,
            name="external event",
            required={"role", "content"},
            optional={"name", "action_index"},
        )
        role = event["role"]
        content = event["content"]
        name = event.get("name")
        action_index = event.get("action_index")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("external event role and content must be strings")
        if name is not None and not isinstance(name, str):
            raise TypeError("external event name must be a string")
        if action_index is not None and (
            not isinstance(action_index, int) or isinstance(action_index, bool)
        ):
            raise TypeError("external event action_index must be an integer")
        events.append(
            EpisodeEvent(
                role=role,
                content=content,
                name=name,
                action_index=action_index,
            )
        )
    return tuple(events)


class ExternalEpisodeEnvironment:
    """Convert one small JSON/stdlib wheel ABI into Reliquary Episode types."""

    def __init__(self, backend: Any, spec: EnvironmentSpec) -> None:
        for method in ("__len__", "task", "reset", "step", "grade", "close"):
            if not callable(getattr(backend, method, None)):
                raise TypeError(f"external environment is missing {method}()")
        if getattr(backend, "name", None) != spec.name:
            raise ValueError("external environment runtime name mismatch")
        if getattr(backend, "validator_authoritative_reward", None) is not (
            spec.validator_authoritative_reward
        ):
            raise ValueError("external environment reward authority mismatch")
        max_turns = getattr(backend, "max_turns", None)
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns <= 0
        ):
            raise ValueError("external environment max_turns must be positive")
        self._backend = backend
        self.name = spec.name
        self.validator_authoritative_reward = spec.validator_authoritative_reward
        self.max_turns = max_turns

    def __len__(self) -> int:
        value = self._backend.__len__()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("external environment length must be non-negative")
        return value

    def get_task(self, index: int) -> EpisodeTask:
        task_index = int(index)
        raw = _object(
            self._backend.task(task_index),
            name="external task",
            required={"id", "prompt", "tools", "metadata"},
        )
        if not isinstance(raw["id"], str) or not isinstance(raw["prompt"], str):
            raise TypeError("external task id and prompt must be strings")
        if not isinstance(raw["metadata"], Mapping):
            raise TypeError("external task metadata must be an object")
        if not isinstance(raw["tools"], Sequence) or isinstance(
            raw["tools"], (str, bytes)
        ):
            raise TypeError("external task tools must be a list")
        tools = []
        for raw_tool in raw["tools"]:
            tool = _object(
                raw_tool,
                name="external tool",
                required={"name", "description", "parameters"},
            )
            if not isinstance(tool["name"], str) or not isinstance(
                tool["description"], str
            ):
                raise TypeError("external tool name and description must be strings")
            if not isinstance(tool["parameters"], Mapping):
                raise TypeError("external tool parameters must be an object")
            tools.append(
                ToolSpec(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=dict(tool["parameters"]),
                )
            )
        metadata = dict(raw["metadata"])
        return EpisodeTask(
            id=raw["id"],
            prompt=raw["prompt"],
            tools=tuple(tools),
            metadata=metadata,
            private={"external_task_index": task_index},
        )

    def get_problem(self, index: int) -> dict[str, Any]:
        from reliquary.environment.agentic.compat import episode_problem

        return episode_problem(self, index)

    @staticmethod
    def _task_index(task: EpisodeTask) -> int:
        value = task.private.get("external_task_index")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("external task has no valid source index")
        return value

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        raw = _object(
            self._backend.reset(self._task_index(task), int(seed)),
            name="external reset",
            required={"state", "events"},
        )
        return ResetResult(state=raw["state"], events=_events(raw["events"]))

    def step(
        self,
        task: EpisodeTask,
        state: Any,
        action: AssistantAction,
    ) -> StepResult:
        raw = _object(
            self._backend.step(self._task_index(task), state, action.to_wire()),
            name="external step",
            required={"state", "events", "done", "termination_reason"},
        )
        done = raw["done"]
        reason = raw["termination_reason"]
        if not isinstance(done, bool):
            raise TypeError("external step done must be boolean")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("external termination reason must be a string")
        return StepResult(
            state=raw["state"],
            events=_events(raw["events"]),
            done=done,
            termination_reason=reason,
        )

    def grade(
        self,
        task: EpisodeTask,
        state: Any,
        trace: EpisodeTrace,
    ) -> RewardReport:
        raw = _object(
            self._backend.grade(
                self._task_index(task),
                state,
                [action.to_wire() for action in trace.actions],
            ),
            name="external reward",
            required={
                "reward",
                "success",
                "checks",
                "state_digest",
                "environment_error",
            },
        )
        if not isinstance(raw["success"], bool):
            raise TypeError("external reward success must be boolean")
        if raw["environment_error"] is not None and not isinstance(
            raw["environment_error"], str
        ):
            raise TypeError("external environment_error must be a string")
        if not isinstance(raw["checks"], Sequence) or isinstance(
            raw["checks"], (str, bytes)
        ):
            raise TypeError("external reward checks must be a list")
        checks = []
        for raw_check in raw["checks"]:
            check = _object(
                raw_check,
                name="external reward check",
                required={"name", "passed", "weight", "detail"},
            )
            if not isinstance(check["name"], str) or not isinstance(
                check["detail"], str
            ):
                raise TypeError("external reward check text must be strings")
            if not isinstance(check["passed"], bool):
                raise TypeError("external reward check passed must be boolean")
            checks.append(
                RewardCheck(
                    name=check["name"],
                    passed=check["passed"],
                    weight=_number(
                        check["weight"], name="external reward check weight"
                    ),
                    detail=check["detail"],
                )
            )
        return RewardReport(
            reward=_number(raw["reward"], name="external reward"),
            success=raw["success"],
            checks=tuple(checks),
            state_digest=_digest(raw["state_digest"], name="state digest"),
            environment_error=raw["environment_error"],
        )

    def close(self, state: Any) -> None:
        self._backend.close(state)


_IMPORT_LOCK = threading.RLock()
_VERIFIED_MODULES: dict[str, tuple[str, Any]] = {}


class _ArtifactSourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute only a snapshot of hash-bound source, never sys.path or pyc code."""

    def __init__(self, package: str, sources: dict, digest: str) -> None:
        self.package, self.sources, self.digest = package, sources, digest

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package and not fullname.startswith(self.package + "."):
            return None
        if fullname not in self.sources:
            raise ImportError(f"external module is not artifact-bound: {fullname}")
        filename, _, package = self.sources[fullname]
        return importlib.util.spec_from_file_location(
            fullname, filename, loader=self,
            submodule_search_locations=[str(Path(filename).parent)] if package else None,
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename, source, _ = self.sources[module.__name__]
        exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
        _VERIFIED_MODULES[module.__name__] = (self.digest, module)


def load_external_episode_environment(
    spec: EnvironmentSpec,
) -> ExternalEpisodeEnvironment:
    with _IMPORT_LOCK:
        artifact = verify_external_artifact(spec)
        distribution = importlib.metadata.distribution(spec.external_distribution or "")
        package = PurePosixPath(spec.external_artifact_resource or "").parts[0]
        sources = {}
        for relative, digest in artifact["files"].items():
            if not relative.endswith(".py"):
                continue
            path = Path(distribution.locate_file(relative)).resolve()
            source = path.read_bytes()
            if _sha256(source) != digest:
                raise ValueError(f"external artifact file digest mismatch: {relative}")
            is_package = relative.endswith("/__init__.py")
            module_name = relative[:-12] if is_package else relative[:-3]
            sources[module_name.replace("/", ".")] = (str(path), source, is_package)
        # Already imported modules must originate from this verified source
        # loader. A matching __file__ alone is forgeable and does not bind code.
        for name, module in tuple(sys.modules.items()):
            if name == package or name.startswith(package + "."):
                if _VERIFIED_MODULES.get(name) != (spec.environment_manifest_sha256, module):
                    raise ValueError(f"external module was imported without verification: {name}")
        loader = _ArtifactSourceLoader(package, sources, spec.environment_manifest_sha256)
        entrypoint = artifact["entrypoints"]["replay"]
        module_name, _, attribute_name = entrypoint.partition(":")
        sys.meta_path.insert(0, loader)
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.meta_path.remove(loader)
        backend_type = getattr(module, attribute_name, None)
        if not callable(backend_type):
            raise TypeError(f"external replay entrypoint is not callable: {entrypoint}")
        return ExternalEpisodeEnvironment(backend_type(), spec)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ExternalEpisodeEnvironment",
    "load_external_episode_environment",
    "verify_external_artifact",
]
