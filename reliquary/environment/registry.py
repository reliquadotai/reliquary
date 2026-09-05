"""Immutable runtime catalog for Reliquary environments.

Registration, protocol eligibility, and operator activation are intentionally
separate layers:

* this module describes known runtime artifacts (external wheels are optional
  until selected);
* ``ProtocolProfile.environments`` describes code allowed by the signed wire
  contract; and
* the CLI environment list selects a non-empty subset for this process.

The catalog stores import paths instead of imported classes/functions so
listing environments never downloads a dataset or starts a grader.  Runtime
adapters remain module-level and therefore process-picklable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from reliquary.environment.base import Environment
from reliquary.environment.agentic.base import EpisodeEnvironment


AdmissionResourceClass = Literal["cpu", "sandbox"]
TerminationPolicy = Literal["eos_or_cap", "math_bft"]
FinalAnswerPolicy = Literal["boxed", "fenced_python", "json"]
InteractionMode = Literal["single_turn", "episode"]


def _import_attribute(path: str) -> Any:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"invalid import path {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise ValueError(f"import path {path!r} has no such attribute") from exc


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Known runtime behavior for one environment.

    ``contract_version`` and the policy identifiers are consensus-facing.
    Import paths and resource-class settings are deliberately omitted from the
    canonical manifest: callable representations, worker counts, and local
    queue choices must never become accidental wire fields.
    """

    name: str
    factory_path: str
    scorer_path: str
    validator_authoritative_reward: bool
    admission_resource_class: AdmissionResourceClass
    termination_policy: TerminationPolicy
    final_answer_policy: FinalAnswerPolicy
    reward_lattice_policy: str
    attainable_rewards: tuple[float, ...]
    contract_version: str
    interaction_mode: InteractionMode = "single_turn"
    episode_replay_path: str | None = None
    renderer_id: str | None = None
    environment_manifest_sha256: str | None = None
    reward_materializer_method: str | None = None
    external_distribution: str | None = None
    external_artifact_resource: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("environment name must be non-empty and trimmed")
        if not self.contract_version:
            raise ValueError("environment contract version must be non-empty")
        if not self.reward_lattice_policy:
            raise ValueError("reward lattice policy must be non-empty")
        digest = self.environment_manifest_sha256
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("environment manifest sha256 must be lowercase hex")
        if self.admission_resource_class not in ("cpu", "sandbox"):
            raise ValueError("unknown admission resource class")
        if self.termination_policy not in ("eos_or_cap", "math_bft"):
            raise ValueError("unknown termination policy")
        if self.final_answer_policy not in (
            "boxed",
            "fenced_python",
            "json",
        ):
            raise ValueError("unknown final-answer policy")
        if self.interaction_mode not in ("single_turn", "episode"):
            raise ValueError("unknown interaction mode")
        if self.interaction_mode == "episode":
            if not self.episode_replay_path or not self.renderer_id:
                raise ValueError(
                    "episode environments require replay path and renderer id"
                )
            _validate_import_path(self.episode_replay_path)
        elif self.episode_replay_path is not None or self.renderer_id is not None:
            raise ValueError(
                "single-turn environments cannot declare episode runtime fields"
            )
        values = tuple(float(value) for value in self.attainable_rewards)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("attainable rewards must be within [0, 1]")
        if values != tuple(sorted(set(values))):
            raise ValueError("attainable rewards must be unique and sorted")
        _validate_import_path(self.factory_path)
        _validate_import_path(self.scorer_path)
        if bool(self.external_distribution) != bool(self.external_artifact_resource):
            raise ValueError(
                "external distribution and artifact resource must be set together"
            )
        if self.external_distribution is not None:
            if self.interaction_mode != "episode" or digest is None:
                raise ValueError(
                    "external environments require episode mode and a manifest digest"
                )

    def create(self) -> Environment | EpisodeEnvironment:
        if self.external_distribution is not None:
            from reliquary.environment.agentic.external import (
                load_external_episode_environment,
            )

            environment = load_external_episode_environment(self)
        else:
            factory = _import_attribute(self.factory_path)
            if not callable(factory):
                raise TypeError(f"environment factory {self.factory_path!r} is not callable")
            environment = factory()
        expected_protocol = (
            EpisodeEnvironment
            if self.interaction_mode == "episode"
            else Environment
        )
        if not isinstance(environment, expected_protocol):
            raise TypeError(
                f"factory for {self.name!r} returned an invalid "
                f"{self.interaction_mode} environment"
            )
        if getattr(environment, "name", None) != self.name:
            raise ValueError(
                f"environment factory name mismatch: expected {self.name!r}, "
                f"got {getattr(environment, 'name', None)!r}"
            )
        return environment

    def score_many(
        self,
        problem: dict[str, Any],
        completion_texts: list[str],
        reward_materials: Any = None,
    ) -> list[float]:
        if self.interaction_mode == "episode":
            raise TypeError("episode environments must be scored by replay")
        scorer = _import_attribute(self.scorer_path)
        if not callable(scorer):
            raise TypeError(f"environment scorer {self.scorer_path!r} is not callable")
        values = scorer(problem, completion_texts, reward_materials)
        rewards = [float(value) for value in values]
        if len(rewards) != len(completion_texts):
            raise ValueError(
                f"environment scorer for {self.name!r} returned "
                f"{len(rewards)} rewards for {len(completion_texts)} completions"
            )
        return rewards

    def consensus_manifest(self) -> dict[str, Any]:
        manifest = {
            "name": self.name,
            "contract_version": self.contract_version,
            "validator_authoritative_reward": self.validator_authoritative_reward,
            "termination_policy": self.termination_policy,
            "final_answer_policy": self.final_answer_policy,
            "reward_lattice_policy": self.reward_lattice_policy,
            "attainable_rewards": list(self.attainable_rewards),
        }
        if self.environment_manifest_sha256 is not None:
            manifest["environment_manifest_sha256"] = (
                self.environment_manifest_sha256
            )
        # Historical single-turn manifests remain byte-for-byte stable.
        if self.interaction_mode == "episode":
            manifest.update({
                "interaction_mode": "episode",
                "episode_schema": "reliquary/episode/v1",
                "renderer_id": self.renderer_id,
            })
        return manifest

    def replay(
        self,
        *,
        task_index: int,
        seed: int,
        actions: Sequence[dict[str, Any]],
    ) -> Any:
        if self.interaction_mode != "episode" or not self.episode_replay_path:
            raise TypeError(f"environment {self.name!r} is not episode-based")
        replay = _import_attribute(self.episode_replay_path)
        if not callable(replay):
            raise TypeError(f"episode replay {self.episode_replay_path!r} is not callable")
        return replay(
            self.create(),
            task_index=int(task_index),
            seed=int(seed),
            actions=list(actions),
        )


def _validate_import_path(path: str) -> None:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"invalid import path {path!r}")


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_document_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = (_REPOSITORY_ROOT / relative).resolve()
        if _REPOSITORY_ROOT not in path.parents:
            raise ValueError(f"implementation path escapes repository: {relative}")
        file_digest = _file_sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_installed_manifest(spec: EnvironmentSpec) -> None:
    expected_digest = spec.environment_manifest_sha256
    if expected_digest is None:
        return
    path = _MANIFEST_ROOT / f"{spec.name}.json"
    if not path.is_file():
        raise ValueError(f"environment manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual_digest = _canonical_document_sha256(manifest)
    if actual_digest != expected_digest:
        raise ValueError(
            f"environment manifest digest mismatch for {spec.name}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    if manifest.get("environment") != spec.name:
        if manifest.get("environment_id") != spec.name:
            raise ValueError(f"environment manifest name mismatch for {spec.name}")
    implementation_files = manifest.get("implementation_files")
    implementation_digest = manifest.get("implementation_sha256")
    if implementation_files is not None or implementation_digest is not None:
        if not isinstance(implementation_files, list) or not all(
            isinstance(value, str) and value for value in implementation_files
        ):
            raise ValueError(f"invalid implementation file list for {spec.name}")
        if _implementation_sha256(implementation_files) != implementation_digest:
            raise ValueError(f"implementation digest mismatch for {spec.name}")
    for path_key, digest_key in (
        ("golden_fixture", "golden_fixture_sha256"),
        ("episode_schema", "episode_schema_sha256"),
    ):
        relative = manifest.get(path_key)
        bound_digest = manifest.get(digest_key)
        # Historical generated manifests bind a fixture digest without
        # carrying its repository-relative path. Their dedicated qualifier
        # verifies that legacy shape; new Episode v1 manifests bind both.
        if relative is None:
            continue
        if not isinstance(bound_digest, str):
            raise ValueError(f"incomplete {path_key} binding for {spec.name}")
        bound_path = (_REPOSITORY_ROOT / relative).resolve()
        if _REPOSITORY_ROOT not in bound_path.parents:
            raise ValueError(f"{path_key} escapes repository for {spec.name}")
        if _file_sha256(bound_path) != bound_digest:
            raise ValueError(f"{path_key} digest mismatch for {spec.name}")


_SPEC_VALUES = (
    EnvironmentSpec(
        name="openmathinstruct",
        factory_path=(
            "reliquary.environment.openmathinstruct:"
            "OpenMathInstructEnvironment"
        ),
        scorer_path=(
            "reliquary.validator.admission:_score_openmath_adapter"
        ),
        validator_authoritative_reward=False,
        admission_resource_class="cpu",
        termination_policy="math_bft",
        final_answer_policy="boxed",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="openmathinstruct-runtime-v1",
    ),
    EnvironmentSpec(
        name="opencodeinstruct",
        factory_path=(
            "reliquary.environment.opencodeinstruct:"
            "OpenCodeInstructEnvironment"
        ),
        scorer_path=(
            "reliquary.validator.admission:_score_opencode_adapter"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="sandbox",
        termination_policy="eos_or_cap",
        final_answer_policy="fenced_python",
        reward_lattice_policy="fractional-by-case-count-v1",
        attainable_rewards=(),
        contract_version="opencodeinstruct-runtime-v1",
        reward_materializer_method="admission_reward_cases",
    ),
    EnvironmentSpec(
        name="reliquaryverifiable_v1",
        factory_path=(
            "reliquary.environment.reliquaryverifiable:"
            "ReliquaryVerifiableEnvironment"
        ),
        scorer_path=(
            "reliquary.environment.reliquaryverifiable:"
            "score_reliquaryverifiable"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary-records-v1",
        environment_manifest_sha256=(
            "d0d5d838e40b383d1c95a62d1cdded8"
            "458f4a7b62df621c87c9435b62207929b"
        ),
    ),
    EnvironmentSpec(
        name="reliquarylogic_v1",
        factory_path=(
            "reliquary.environment.reliquarylogic:"
            "ReliquaryLogicEnvironment"
        ),
        scorer_path=(
            "reliquary.environment.reliquarylogic:"
            "score_reliquarylogic"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary-logic-v1",
        environment_manifest_sha256=(
            "9cb29e487321b2e6c005f2a1a89ccff"
            "ecf01b1c09bfd03337094e338ab912ca9"
        ),
    ),
    EnvironmentSpec(
        name="reliquary_stateful_tools_v1",
        factory_path=(
            "reliquary.environment.agentic.envs.stateful_tools_v1:"
            "StatefulToolsEnvironment"
        ),
        scorer_path=(
            "reliquary.environment.agentic.suite:"
            "episode_score_many_not_supported"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary-stateful-tools-v1",
        interaction_mode="episode",
        episode_replay_path="reliquary.environment.agentic.suite:replay_submission",
        renderer_id="reliquary-jsonl-tools-v1",
        environment_manifest_sha256=(
            "3725a5ec6186702d3f387c2a8cb174ff"
            "ce672dc3efe9b877460a8454e775db2e"
        ),
    ),
    EnvironmentSpec(
        name="reliquary_stateful_tools_v2",
        factory_path="reliquary_stateful_tools:StatefulToolsEnvironment",
        scorer_path=(
            "reliquary.environment.agentic.suite:"
            "episode_score_many_not_supported"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary/episode-json/v1",
        interaction_mode="episode",
        episode_replay_path="reliquary.environment.agentic.suite:replay_submission",
        renderer_id="reliquary-jsonl-tools-v1",
        environment_manifest_sha256=(
            "9779a3151575687e823567c6c9c5459d"
            "93794bab4f27f4bcde987c8414cb50f6"
        ),
        external_distribution="reliquary-stateful-tools",
        external_artifact_resource="reliquary_stateful_tools/artifact.json",
    ),
    EnvironmentSpec(
        name="reliquary_retrieval_tools_v1",
        factory_path=(
            "reliquary.environment.agentic.envs.retrieval_tools_v1:"
            "RetrievalToolsEnvironment"
        ),
        scorer_path=(
            "reliquary.environment.agentic.suite:"
            "episode_score_many_not_supported"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="cpu",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary-retrieval-tools-v1",
        interaction_mode="episode",
        episode_replay_path="reliquary.environment.agentic.suite:replay_submission",
        renderer_id="reliquary-jsonl-tools-v1",
        environment_manifest_sha256=(
            "94095ba52ae58895f19b99bc9d605d8"
            "a3b6cdea55118af1b857ed42d484072c0"
        ),
    ),
    EnvironmentSpec(
        name="reliquary_workspace_tools_v1",
        factory_path=(
            "reliquary.environment.agentic.envs.workspace_tools_v1:"
            "WorkspaceToolsEnvironment"
        ),
        scorer_path=(
            "reliquary.environment.agentic.suite:"
            "episode_score_many_not_supported"
        ),
        validator_authoritative_reward=True,
        admission_resource_class="sandbox",
        termination_policy="eos_or_cap",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary-workspace-tools-v1",
        interaction_mode="episode",
        episode_replay_path="reliquary.environment.agentic.suite:replay_submission",
        renderer_id="reliquary-jsonl-tools-v1",
        environment_manifest_sha256=(
            "9f888c49e5d1775f0f83314a0177ee5"
            "562c9e4858b8e4e401af8ef9ff7e0f4a7"
        ),
    ),
)


def _build_catalog(
    specs: Sequence[EnvironmentSpec],
) -> Mapping[str, EnvironmentSpec]:
    catalog: dict[str, EnvironmentSpec] = {}
    for spec in specs:
        if spec.name in catalog:
            raise ValueError(f"duplicate environment registration: {spec.name}")
        # External wheels are optional until selected; their artifact is verified
        # offline before first use instead of making the base install depend on it.
        if spec.external_distribution is None:
            _validate_installed_manifest(spec)
        catalog[spec.name] = spec
    return MappingProxyType(catalog)


ENVIRONMENT_SPECS: Mapping[str, EnvironmentSpec] = _build_catalog(_SPEC_VALUES)


def environment_catalog() -> Mapping[str, EnvironmentSpec]:
    """Return the immutable installed-environment catalog."""

    return ENVIRONMENT_SPECS


def get_environment_spec(name: str) -> EnvironmentSpec:
    try:
        return ENVIRONMENT_SPECS[name]
    except KeyError as exc:
        available = ", ".join(ENVIRONMENT_SPECS)
        raise ValueError(
            f"Unknown environment: {name}; expected one of: {available}"
        ) from exc


def environment_manifest(
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = tuple(ENVIRONMENT_SPECS) if names is None else tuple(names)
    if len(set(selected)) != len(selected):
        raise ValueError("environment manifest names must be unique")
    return {
        "schema": "reliquary/environment-manifest/v1",
        "environments": [
            get_environment_spec(name).consensus_manifest()
            for name in selected
        ],
    }


def environment_manifest_sha256(
    names: Sequence[str] | None = None,
) -> str:
    payload = json.dumps(
        environment_manifest(names),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_environment_mix(
    requested_names: Sequence[str],
    *,
    profile_environments: Mapping[str, Any],
    default_batch_target: int,
) -> list[tuple[str, int]]:
    """Resolve an explicit runtime subset without ever falling back.

    Every name must be installed and declared by the active signed profile.
    Targets come from an optional ``EnvironmentProfile.batch_target`` and use
    the caller's legacy target only when the profile omits that field.
    """

    names = [str(name).strip() for name in requested_names]
    if not names or any(not name for name in names):
        raise ValueError("at least one environment must be selected")
    if len(set(names)) != len(names):
        raise ValueError("environment selection contains duplicate names")
    if int(default_batch_target) <= 0:
        raise ValueError("default batch target must be positive")

    mix: list[tuple[str, int]] = []
    for name in names:
        spec = get_environment_spec(name)
        try:
            environment_profile = profile_environments[name]
        except KeyError as exc:
            raise ValueError(
                f"environment {name!r} is not declared by the active protocol profile"
            ) from exc
        profile_contract = getattr(
            environment_profile, "environment_contract_id", None
        )
        profile_manifest = getattr(
            environment_profile, "environment_manifest_sha256", None
        )
        if profile_contract is not None and profile_contract != spec.contract_version:
            raise ValueError(
                f"environment {name!r} contract does not match installed code"
            )
        if (
            profile_manifest is not None
            and profile_manifest != spec.environment_manifest_sha256
        ):
            raise ValueError(
                f"environment {name!r} manifest does not match installed code"
            )
        if spec.external_distribution is not None:
            from reliquary.environment.agentic.external import verify_external_artifact

            verify_external_artifact(spec)
        configured = getattr(environment_profile, "batch_target", None)
        target = int(
            default_batch_target if configured is None else configured
        )
        if target <= 0:
            raise ValueError(f"environment {name!r} has a non-positive target")
        mix.append((name, target))
    return mix


__all__ = [
    "ENVIRONMENT_SPECS",
    "EnvironmentSpec",
    "InteractionMode",
    "environment_catalog",
    "environment_manifest",
    "environment_manifest_sha256",
    "get_environment_spec",
    "resolve_environment_mix",
]
