"""Canonical, capability-addressed release contracts.

This module is intentionally dependency-light.  It describes which immutable
wire, generation, market, verification, training, and environment contracts a
peer implements without using an integer version as feature detection.

It does not activate a release.  Activation, rollout, and compatibility policy
belong to the validator/miner control plane.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


RELEASE_CONTRACT_SCHEMA = "reliquary.release-contract/v1"
RELEASE_CONTRACT_DOMAIN = "reliquary.release/v1"

CAP_CHECKPOINT_ADOPTION_GATE = "checkpoint.publication-adoption-gate/v1"
CAP_DURABLE_LANE_JOURNAL = "training.durable-lane-journal/v1"
CAP_ENVIRONMENT_EPISODE_ABI = "environment.episode-abi/v1"
CAP_FRESH_POST_SEAL_ORDERING = "selection.fresh-post-seal/v1"
CAP_MANIFEST_16_LANES = "generation.manifest-16-lanes/v1"
CAP_MINER_SELECTED_INTENTS = "market.miner-selected-intents/v1"
CAP_SELECTED_SLOT_REWARD = "reward.selected-slot/v1"
CAP_STREAMING_TICKET_VALIDATION = "verification.streaming-ticketed/v1"
CAP_TICKETED_PACED_EPOCH = "market.ticketed-paced-epoch/v1"
CAP_TRAINER_PACED_LANES = "training.trainer-paced-lanes/v1"

_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,126}[a-z0-9]$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class ReleaseContractError(ValueError):
    """The release contract is malformed or not canonical."""


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("canonical JSON does not permit non-finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON keys must be strings")
            result[key] = _json_native(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    raise TypeError(f"value is not JSON-native: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-native data with Reliquary's canonical JSON profile."""

    return json.dumps(
        _json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_identifier(label: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ReleaseContractError(f"{label} is not a canonical identifier")
    return value


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReleaseContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_object(label: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{label} must be an object")
    return value


def _require_exact_keys(
    label: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ReleaseContractError(
            f"{label} keys differ; missing={missing}, unknown={unknown}"
        )


@dataclass(frozen=True, slots=True)
class ContractComponent:
    """An immutable reference to canonical component bytes."""

    component_id: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        _require_identifier("component_id", self.component_id)
        _require_sha256("canonical_sha256", self.canonical_sha256)

    @classmethod
    def bind(cls, component_id: str, canonical_payload: Any) -> ContractComponent:
        return cls(
            component_id=component_id,
            canonical_sha256=canonical_sha256(canonical_payload),
        )

    def matches(self, canonical_payload: Any) -> bool:
        """Return whether payload bytes reproduce this component binding."""

        return canonical_sha256(canonical_payload) == self.canonical_sha256


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    """One named environment ABI/configuration contract."""

    environment_id: str
    component: ContractComponent

    def __post_init__(self) -> None:
        _require_identifier("environment_id", self.environment_id)
        if not isinstance(self.component, ContractComponent):
            raise TypeError("component must be ContractComponent")


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    """Explicit feature membership; numeric feature dispatch is unnecessary."""

    bundle_id: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("bundle_id", self.bundle_id)
        if not isinstance(self.capabilities, tuple):
            raise TypeError("capabilities must be a tuple")
        for capability in self.capabilities:
            _require_identifier("capability", capability)
        if tuple(sorted(set(self.capabilities))) != self.capabilities:
            raise ReleaseContractError(
                "capabilities must be sorted and contain no duplicates"
            )

    @classmethod
    def from_iterable(
        cls,
        bundle_id: str,
        capabilities: Iterable[str],
    ) -> CapabilityBundle:
        return cls(bundle_id, tuple(sorted(set(capabilities))))

    def supports(self, capability_id: str) -> bool:
        _require_identifier("capability_id", capability_id)
        return capability_id in self.capabilities


RELIQUARY_1_CAPABILITIES = CapabilityBundle.from_iterable(
    "reliquary-1-ticketed-paced/v1",
    (
        CAP_CHECKPOINT_ADOPTION_GATE,
        CAP_DURABLE_LANE_JOURNAL,
        CAP_ENVIRONMENT_EPISODE_ABI,
        CAP_FRESH_POST_SEAL_ORDERING,
        CAP_MANIFEST_16_LANES,
        CAP_MINER_SELECTED_INTENTS,
        CAP_SELECTED_SLOT_REWARD,
        CAP_STREAMING_TICKET_VALIDATION,
        CAP_TICKETED_PACED_EPOCH,
        CAP_TRAINER_PACED_LANES,
    ),
)


@dataclass(frozen=True, slots=True)
class ReleaseContract:
    """Composition root for a coordinated Reliquary release."""

    release_id: str
    wire: ContractComponent
    generation: ContractComponent
    market: ContractComponent
    verification: ContractComponent
    training: ContractComponent
    environments: tuple[EnvironmentContract, ...]
    capabilities: CapabilityBundle
    schema: str = RELEASE_CONTRACT_SCHEMA
    domain: str = RELEASE_CONTRACT_DOMAIN

    def __post_init__(self) -> None:
        _require_identifier("release_id", self.release_id)
        if self.schema != RELEASE_CONTRACT_SCHEMA:
            raise ReleaseContractError("unsupported release contract schema")
        if self.domain != RELEASE_CONTRACT_DOMAIN:
            raise ReleaseContractError("unsupported release contract domain")
        for name in ("wire", "generation", "market", "verification", "training"):
            if not isinstance(getattr(self, name), ContractComponent):
                raise TypeError(f"{name} must be ContractComponent")
        if not isinstance(self.environments, tuple):
            raise TypeError("environments must be a tuple")
        if not self.environments:
            raise ReleaseContractError("at least one environment is required")
        if any(
            not isinstance(environment, EnvironmentContract)
            for environment in self.environments
        ):
            raise TypeError("environments must contain EnvironmentContract")
        environment_ids = tuple(
            environment.environment_id for environment in self.environments
        )
        if tuple(sorted(set(environment_ids))) != environment_ids:
            raise ReleaseContractError(
                "environments must be sorted and contain no duplicate IDs"
            )
        component_ids = (
            self.wire.component_id,
            self.generation.component_id,
            self.market.component_id,
            self.verification.component_id,
            self.training.component_id,
            *(environment.component.component_id for environment in self.environments),
        )
        if len(set(component_ids)) != len(component_ids):
            raise ReleaseContractError("component IDs must be globally unique")
        if not isinstance(self.capabilities, CapabilityBundle):
            raise TypeError("capabilities must be CapabilityBundle")

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def supports(self, capability_id: str) -> bool:
        return self.capabilities.supports(capability_id)

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(release_contract_to_dict(self))


def _component_to_dict(component: ContractComponent) -> dict[str, str]:
    return {
        "canonical_sha256": component.canonical_sha256,
        "component_id": component.component_id,
    }


def release_contract_to_dict(contract: ReleaseContract) -> dict[str, Any]:
    return {
        "capabilities": {
            "bundle_id": contract.capabilities.bundle_id,
            "values": list(contract.capabilities.capabilities),
        },
        "components": {
            "environments": [
                {
                    "canonical_sha256": environment.component.canonical_sha256,
                    "component_id": environment.component.component_id,
                    "environment_id": environment.environment_id,
                }
                for environment in contract.environments
            ],
            "generation": _component_to_dict(contract.generation),
            "market": _component_to_dict(contract.market),
            "training": _component_to_dict(contract.training),
            "verification": _component_to_dict(contract.verification),
            "wire": _component_to_dict(contract.wire),
        },
        "domain": contract.domain,
        "release_id": contract.release_id,
        "schema": contract.schema,
    }


def _parse_component(label: str, value: object) -> ContractComponent:
    raw = _require_object(label, value)
    _require_exact_keys(label, raw, {"component_id", "canonical_sha256"})
    return ContractComponent(
        component_id=_require_identifier(f"{label}.component_id", raw["component_id"]),
        canonical_sha256=_require_sha256(
            f"{label}.canonical_sha256", raw["canonical_sha256"]
        ),
    )


def _decode_json_strict(raw: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReleaseContractError(f"non-finite JSON number: {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError("release contract is not valid UTF-8 JSON") from exc
    return _require_object("release contract", decoded)


def parse_release_contract(raw: bytes) -> ReleaseContract:
    """Parse exact canonical bytes, rejecting defaults and unknown fields."""

    if not isinstance(raw, bytes):
        raise TypeError("raw release contract must be bytes")
    value = _decode_json_strict(raw)
    _require_exact_keys(
        "release contract",
        value,
        {"schema", "domain", "release_id", "components", "capabilities"},
    )
    if value["schema"] != RELEASE_CONTRACT_SCHEMA:
        raise ReleaseContractError("unsupported release contract schema")
    if value["domain"] != RELEASE_CONTRACT_DOMAIN:
        raise ReleaseContractError("unsupported release contract domain")

    components = _require_object("components", value["components"])
    _require_exact_keys(
        "components",
        components,
        {
            "wire",
            "generation",
            "market",
            "verification",
            "training",
            "environments",
        },
    )
    raw_environments = components["environments"]
    if not isinstance(raw_environments, list):
        raise ReleaseContractError("components.environments must be an array")
    environments: list[EnvironmentContract] = []
    for index, item in enumerate(raw_environments):
        label = f"components.environments[{index}]"
        environment = _require_object(label, item)
        _require_exact_keys(
            label,
            environment,
            {"environment_id", "component_id", "canonical_sha256"},
        )
        environments.append(
            EnvironmentContract(
                environment_id=_require_identifier(
                    f"{label}.environment_id", environment["environment_id"]
                ),
                component=ContractComponent(
                    component_id=_require_identifier(
                        f"{label}.component_id", environment["component_id"]
                    ),
                    canonical_sha256=_require_sha256(
                        f"{label}.canonical_sha256",
                        environment["canonical_sha256"],
                    ),
                ),
            )
        )

    raw_capabilities = _require_object("capabilities", value["capabilities"])
    _require_exact_keys("capabilities", raw_capabilities, {"bundle_id", "values"})
    capability_values = raw_capabilities["values"]
    if not isinstance(capability_values, list) or any(
        not isinstance(item, str) for item in capability_values
    ):
        raise ReleaseContractError("capabilities.values must be an array of strings")

    contract = ReleaseContract(
        release_id=_require_identifier("release_id", value["release_id"]),
        wire=_parse_component("components.wire", components["wire"]),
        generation=_parse_component("components.generation", components["generation"]),
        market=_parse_component("components.market", components["market"]),
        verification=_parse_component(
            "components.verification", components["verification"]
        ),
        training=_parse_component("components.training", components["training"]),
        environments=tuple(environments),
        capabilities=CapabilityBundle(
            bundle_id=_require_identifier(
                "capabilities.bundle_id", raw_capabilities["bundle_id"]
            ),
            capabilities=tuple(capability_values),
        ),
    )
    if raw != contract.to_bytes():
        raise ReleaseContractError("release contract bytes are not canonical")
    return contract


__all__ = [
    "CAP_CHECKPOINT_ADOPTION_GATE",
    "CAP_DURABLE_LANE_JOURNAL",
    "CAP_ENVIRONMENT_EPISODE_ABI",
    "CAP_FRESH_POST_SEAL_ORDERING",
    "CAP_MANIFEST_16_LANES",
    "CAP_MINER_SELECTED_INTENTS",
    "CAP_SELECTED_SLOT_REWARD",
    "CAP_STREAMING_TICKET_VALIDATION",
    "CAP_TICKETED_PACED_EPOCH",
    "CAP_TRAINER_PACED_LANES",
    "CapabilityBundle",
    "ContractComponent",
    "EnvironmentContract",
    "RELEASE_CONTRACT_DOMAIN",
    "RELEASE_CONTRACT_SCHEMA",
    "RELIQUARY_1_CAPABILITIES",
    "ReleaseContract",
    "ReleaseContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "parse_release_contract",
    "release_contract_to_dict",
]
