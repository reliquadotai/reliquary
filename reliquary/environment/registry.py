"""Immutable runtime catalog for Reliquary environments.

Registration, protocol eligibility, and operator activation are intentionally
separate layers:

* this module describes code that is installed;
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
from types import MappingProxyType
from typing import Any, Literal

from reliquary.environment.base import Environment


AdmissionResourceClass = Literal["cpu", "sandbox"]
TerminationPolicy = Literal["eos_or_cap", "math_bft"]
FinalAnswerPolicy = Literal["boxed", "fenced_python", "json"]


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
    """Installed runtime behavior for one environment.

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
    environment_manifest_sha256: str | None = None
    reward_materializer_method: str | None = None

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
        values = tuple(float(value) for value in self.attainable_rewards)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("attainable rewards must be within [0, 1]")
        if values != tuple(sorted(set(values))):
            raise ValueError("attainable rewards must be unique and sorted")
        _validate_import_path(self.factory_path)
        _validate_import_path(self.scorer_path)

    def create(self) -> Environment:
        factory = _import_attribute(self.factory_path)
        if not callable(factory):
            raise TypeError(f"environment factory {self.factory_path!r} is not callable")
        environment = factory()
        if not isinstance(environment, Environment):
            raise TypeError(f"factory for {self.name!r} returned an invalid environment")
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
        return manifest


def _validate_import_path(path: str) -> None:
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"invalid import path {path!r}")


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
)


def _build_catalog(
    specs: Sequence[EnvironmentSpec],
) -> Mapping[str, EnvironmentSpec]:
    catalog: dict[str, EnvironmentSpec] = {}
    for spec in specs:
        if spec.name in catalog:
            raise ValueError(f"duplicate environment registration: {spec.name}")
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
        get_environment_spec(name)
        try:
            environment_profile = profile_environments[name]
        except KeyError as exc:
            raise ValueError(
                f"environment {name!r} is not declared by the active protocol profile"
            ) from exc
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
    "environment_catalog",
    "environment_manifest",
    "environment_manifest_sha256",
    "get_environment_spec",
    "resolve_environment_mix",
]
