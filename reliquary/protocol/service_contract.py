"""Immutable service orders, separate from the historical generation contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256

SCHEMA = "service-contract/v1"
MAX_SAFE_INTEGER = 2**53 - 1
SERVICE_KINDS = {"dataset_mapping", "dataset_curation", "adaptive_training"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_POLICIES = {
    "sampling": {
        "legacy/v1": {},
        "public-group-pool/v1": {"group_size": (2, 64), "pool_groups": (2, 1024), "renewal_windows": (1, 1000000)},
        "public-draw-pool/v1": {"group_size": (2, 64), "pool_draws": (3, 65536), "renewal_windows": (1, 1000000)},
    },
    "eligibility": {
        "all/v1": {},
        "dataset-epoch/v1": {"coverage_bps": (1, 10000), "refresh_windows": (1, 1000000), "max_epoch_windows": (1, 1000000)},
    },
    "cooldown": {
        "static/v1": {"windows": (0, 1000000)},
        "adaptive-rotation/v1": {
            "min_windows": (0, 1000000), "max_windows": (1, 1000000),
            "margin_bps": (1, 10000), "min_panel_groups": (1, 1000000),
            "coverage_bps": (1, 10000), "freshness_windows": (1, 1000000),
            "smoothing_bps": (1, 10000), "max_change_windows": (1, 1000000),
            "hysteresis_windows": (0, 1000000),
            "fallback_windows": (0, 1000000),
        },
    },
    "checkpoint": {
        "frozen/v1": {},
        "trainer-driven/v1": {"task_scoped": (0, 1)},
    },
    "reward": {
        "legacy/v1": {},
        "exploration-discount/v1": {
            "divisor": (2, 1000000), "budget_bps": (1, 10000),
            "refresh_windows": (1, 1000000), "max_tokens_per_group": (1, MAX_SAFE_INTEGER),
        },
    },
}


class ServiceContractError(ValueError):
    pass


def _object(value: Any, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ServiceContractError(f"{name}: expected fields {sorted(fields)}")
    return value


def _integer(value: Any, name: str, low: int = 1, high: int = MAX_SAFE_INTEGER) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ServiceContractError(f"{name}: integer in [{low}, {high}] required")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or ".." in value:
        raise ServiceContractError(f"{name}: canonical identifier required")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ServiceContractError(f"{name}: lowercase SHA-256 required")
    return value


def validate_service_contract(value: dict) -> None:
    _object(value, {"schema", "service_kind", "revision_id", "dataset", "checkpoint", "environment",
                    "generation_contract_sha256", "scoring", "policies", "limits", "visibility"}, "contract")
    if value["schema"] != SCHEMA or not isinstance(value["service_kind"], str) or value["service_kind"] not in SERVICE_KINDS:
        raise ServiceContractError("unknown service schema or kind")
    _identifier(value["revision_id"], "revision_id")
    _object(value["dataset"], {"id", "sha256"}, "dataset")
    _identifier(value["dataset"]["id"], "dataset.id")
    _sha(value["dataset"]["sha256"], "dataset.sha256")
    _object(value["checkpoint"], {"repo", "revision", "sha256"}, "checkpoint")
    for field in ("repo", "revision"):
        _identifier(value["checkpoint"][field], f"checkpoint.{field}")
    if not re.fullmatch(r"[0-9a-f]{40}", value["checkpoint"]["revision"]):
        raise ServiceContractError("checkpoint.revision must pin an immutable 40-hex commit")
    _sha(value["checkpoint"]["sha256"], "checkpoint.sha256")
    _object(value["environment"], {"id", "version"}, "environment")
    for field in ("id", "version"):
        _identifier(value["environment"][field], f"environment.{field}")
    _sha(value["generation_contract_sha256"], "generation_contract_sha256")
    score = _object(value["scoring"], {"kind", "sigma_min_bps", "weights_bps"}, "scoring")
    if not isinstance(score["kind"], str) or score["kind"] not in {"environment-reward/v1", "weighted-reward/v1"}:
        raise ServiceContractError("unknown scoring policy")
    _integer(score["sigma_min_bps"], "sigma_min_bps", 0, 10000)
    weights = score["weights_bps"]
    if not isinstance(weights, dict) or not 1 <= len(weights) <= 16:
        raise ServiceContractError("weights_bps: 1..16 metrics required")
    for name, weight in weights.items():
        _identifier(name, "metric")
        _integer(weight, f"weight.{name}", 1, 10000)
    if sum(weights.values()) != 10000:
        raise ServiceContractError("weights_bps must sum to 10000")
    if score["kind"] == "environment-reward/v1" and weights != {"reward": 10000}:
        raise ServiceContractError("environment-reward/v1 uses the environment reward unchanged")
    policies = _object(value["policies"], set(_POLICIES), "policies")
    for lane, variants in _POLICIES.items():
        policy = policies[lane]
        if not isinstance(policy, dict) or not isinstance(policy.get("kind"), str) or policy.get("kind") not in variants:
            raise ServiceContractError(f"unknown {lane} policy")
        params = variants[policy["kind"]]
        _object(policy, {"kind", *params}, lane)
        for name, bounds in params.items():
            _integer(policy[name], f"{lane}.{name}", *bounds)
    sampling = policies["sampling"]
    if sampling["kind"] == "public-draw-pool/v1" and sampling["pool_draws"] <= sampling["group_size"]:
        raise ServiceContractError("pool_draws must exceed group_size")
    cooldown = policies["cooldown"]
    if cooldown["kind"] == "adaptive-rotation/v1":
        if cooldown["min_windows"] > cooldown["max_windows"] or not cooldown["min_windows"] <= cooldown["fallback_windows"] <= cooldown["max_windows"]:
            raise ServiceContractError("invalid cooldown bounds/fallback")
        if policies["eligibility"]["kind"] != "dataset-epoch/v1":
            raise ServiceContractError("adaptive cooldown needs dataset-epoch eligibility")
    if value["service_kind"] != "adaptive_training":
        if policies["checkpoint"]["kind"] != "frozen/v1" or policies["reward"]["kind"] != "legacy/v1" or policies["cooldown"]["kind"] != "static/v1":
            raise ServiceContractError("mapping/curation requires frozen checkpoint and no adaptive incentives")
    elif policies["checkpoint"]["kind"] != "trainer-driven/v1":
        raise ServiceContractError("adaptive training needs trainer-driven checkpoints")
    limits = _object(value["limits"], {"max_groups", "max_tokens", "deadline_seconds"}, "limits")
    for name, amount in limits.items():
        _integer(amount, f"limits.{name}")
    if not isinstance(value["visibility"], str) or value["visibility"] not in {"private", "task"}:
        raise ServiceContractError("unknown visibility")


@dataclass(frozen=True, slots=True)
class ServiceContract:
    """Canonical bytes prevent a caller mutating an already ordered contract."""

    canonical: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical, bytes) or len(self.canonical) > 65536:
            raise ServiceContractError("contract must be at most 64 KiB")
        value = _decode(self.canonical)
        validate_service_contract(value)
        if canonical_json_bytes(value) != self.canonical:
            raise ServiceContractError("contract bytes must be canonical")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServiceContract:
        try:
            return cls(canonical_json_bytes(value))
        except (TypeError, OverflowError) as exc:
            raise ServiceContractError(str(exc)) from exc

    def to_dict(self) -> dict:
        return json.loads(self.canonical)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def context_sha256(self) -> str:
        value = self.to_dict()
        return canonical_sha256({name: value[name] for name in
                                 ("dataset", "checkpoint", "environment", "generation_contract_sha256", "scoring")})

    def require_capabilities(self, supported: set[str]) -> None:
        value = self.to_dict()
        required = {value["scoring"]["kind"], *(p["kind"] for p in value["policies"].values())}
        if value["policies"]["checkpoint"].get("task_scoped") == 1:
            required.add("task-scoped/v1")
        missing = required - supported
        if missing:
            raise ServiceContractError(f"runtime lacks capabilities: {sorted(missing)}")


def _decode(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ServiceContractError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ServiceContractError("invalid contract JSON") from exc


def parse_service_contract(raw: bytes) -> ServiceContract:
    return ServiceContract(raw)
