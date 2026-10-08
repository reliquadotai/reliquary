"""Bounded public candidate groups with immutable sampling identities."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_contract import ServiceContract

POOL_SCHEMA = "public-group-pool/v2"
SELECTION_SCHEMA = "public-group-selection/v1"
ROLLOUT_SCHEMA = "public-group-rollout/v1"
CAPABILITY = "public-group-pool/v1"
DRAW_DOMAIN = b"public-group-draw/v2"
PROOF_VERSION = "public-group-proof/v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class SeedPoolError(ValueError):
    pass


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise SeedPoolError(f"{name}: integer in [{low}, {high}] required")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise SeedPoolError(f"{name}: lowercase SHA-256 required")
    return value


@dataclass(frozen=True, slots=True)
class PoolSelection:
    pool_sha256: str
    candidate_id: int

    def __post_init__(self) -> None:
        _sha(self.pool_sha256, "pool_sha256")
        _integer(self.candidate_id, "candidate_id", 0, 1023)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PoolSelection:
        if not isinstance(value, dict) or set(value) != {"schema", "pool_sha256", "candidate_id"} or value["schema"] != SELECTION_SCHEMA:
            raise SeedPoolError("unknown public group selection")
        return cls(value["pool_sha256"], value["candidate_id"])

    def to_dict(self) -> dict:
        return {"schema": SELECTION_SCHEMA, "pool_sha256": self.pool_sha256, "candidate_id": self.candidate_id}

    def rollout_binding(self, rollout_index: int) -> dict:
        _integer(rollout_index, "rollout_index", 0, 63)
        return {"schema": ROLLOUT_SCHEMA, "pool_sha256": self.pool_sha256,
                "candidate_id": self.candidate_id, "rollout_index": rollout_index}


def parse_rollout_binding(value: Any) -> tuple[PoolSelection, int]:
    if not isinstance(value, dict) or set(value) != {"schema", "pool_sha256", "candidate_id", "rollout_index"} or value["schema"] != ROLLOUT_SCHEMA:
        raise SeedPoolError("unknown public group rollout binding")
    selection = PoolSelection(value["pool_sha256"], value["candidate_id"])
    index = _integer(value["rollout_index"], "rollout_index", 0, 63)
    return selection, index


_ANNOUNCEMENT_FIELDS = {"contract", "schedule", "checkpoint", "supported_capabilities", "pool_epoch", "pool_randomness"}


def pool_from_service_policy(value: Any, *, environment: str, prompt_idx: int,
                             checkpoint_hash: str) -> SeedPool | None:
    """Resolve the exact operator announcement for one env, refusing unsupported policies."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict) or set(value) != _ANNOUNCEMENT_FIELDS:
        raise SeedPoolError("invalid service policy announcement")
    contract = ServiceContract.from_dict(value["contract"])
    if contract.version != 2:
        raise SeedPoolError("only service-contract/v2 announces public pools")
    supported = value["supported_capabilities"]
    if not isinstance(supported, list) or not 1 <= len(supported) <= 32 or any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in supported):
        raise SeedPoolError("bounded service capabilities required")
    contract.require_capabilities(set(supported))
    policy = contract.environment(environment)["sampling"]
    if policy["kind"] == "legacy/v1":
        return None
    if policy["kind"] != CAPABILITY:
        raise SeedPoolError("sampling policy is not implemented by this miner")
    return SeedPool.from_contract(contract, environment=environment, prompt_idx=prompt_idx,
                                  checkpoint_hash=checkpoint_hash, pool_epoch=value["pool_epoch"],
                                  randomness=value["pool_randomness"])


@dataclass(frozen=True, slots=True)
class SeedPool:
    """An operator-published pool; the miner only chooses a candidate ID.

    The beacon and epoch belong to the authoritative pool announcement. A new
    window does not silently change an existing pool's beacon. Publication and
    renewal are the caller's responsibility; generation accepts this exact
    immutable manifest and validation compares its digest to the active one.
    """
    service_contract_sha256: str
    environment: str
    prompt_idx: int
    checkpoint_hash: str
    pool_epoch: int
    randomness: str
    group_size: int
    pool_groups: int
    renewal_windows: int
    _digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _sha(self.service_contract_sha256, "service_contract_sha256")
        if not isinstance(self.environment, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,63}", self.environment):
            raise SeedPoolError("environment: canonical environment id required")
        _integer(self.prompt_idx, "prompt_idx", 0, 2**53 - 1)
        if not isinstance(self.checkpoint_hash, str) or not 1 <= len(self.checkpoint_hash.encode()) <= 256:
            raise SeedPoolError("checkpoint_hash: nonempty bounded identity required")
        _integer(self.pool_epoch, "pool_epoch", 0, 2**53 - 1)
        _sha(self.randomness, "randomness")
        _integer(self.group_size, "group_size", 2, 64)
        _integer(self.pool_groups, "pool_groups", 2, 1024)
        _integer(self.renewal_windows, "renewal_windows", 1, 1000000)
        object.__setattr__(self, "_digest", hashlib.sha256(canonical_json_bytes(self.to_dict())).digest())

    @classmethod
    def from_contract(cls, contract: ServiceContract, *, environment: str, prompt_idx: int,
                      checkpoint_hash: str, pool_epoch: int, randomness: str) -> SeedPool:
        policy = contract.environment(environment)["sampling"]
        if policy["kind"] != CAPABILITY:
            raise SeedPoolError("only public-group-pool/v1 is supported")
        return cls(contract.sha256, environment, prompt_idx, checkpoint_hash, pool_epoch, randomness,
                   policy["group_size"], policy["pool_groups"], policy["renewal_windows"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SeedPool:
        names = {"service_contract_sha256", "environment", "prompt_idx", "checkpoint_hash",
                 "pool_epoch", "randomness", "group_size", "pool_groups", "renewal_windows"}
        if not isinstance(value, dict) or set(value) != {"schema", *names} or value["schema"] != POOL_SCHEMA:
            raise SeedPoolError("unknown public group pool")
        return cls(**{key: value[key] for key in names})

    def to_dict(self) -> dict:
        return {"schema": POOL_SCHEMA, "service_contract_sha256": self.service_contract_sha256,
                "environment": self.environment, "prompt_idx": self.prompt_idx,
                "checkpoint_hash": self.checkpoint_hash, "pool_epoch": self.pool_epoch,
                "randomness": self.randomness, "group_size": self.group_size,
                "pool_groups": self.pool_groups, "renewal_windows": self.renewal_windows}

    @property
    def sha256(self) -> str:
        return self._digest.hex()

    @property
    def exploration_rollouts(self) -> int:
        return self.pool_groups * self.group_size

    def selection(self, candidate_id: int) -> PoolSelection:
        _integer(candidate_id, "candidate_id", 0, self.pool_groups - 1)
        return PoolSelection(self.sha256, candidate_id)

    def validate_selection(self, selection: PoolSelection, *, rollout_count: int) -> None:
        if not isinstance(selection, PoolSelection) or selection.pool_sha256 != self.sha256:
            raise SeedPoolError("selection is not bound to the active pool")
        _integer(selection.candidate_id, "candidate_id", 0, self.pool_groups - 1)
        if type(rollout_count) is not int or rollout_count != self.group_size:
            raise SeedPoolError("candidate must contain one complete canonical group")

    def validate_rollout_binding(self, binding: Any, selection: PoolSelection, rollout_index: int) -> None:
        self.validate_selection(selection, rollout_count=self.group_size)
        _integer(rollout_index, "rollout_index", 0, self.group_size - 1)
        if not isinstance(binding, dict) or binding != selection.rollout_binding(rollout_index):
            raise SeedPoolError("rollout identity differs from the selected public group")
        # Dict equality considers True == 1; exact canonical bytes do not.
        if canonical_json_bytes(binding) != canonical_json_bytes(selection.rollout_binding(rollout_index)):
            raise SeedPoolError("rollout identity must use canonical integer fields")

    def uniform(self, candidate_id: int, rollout_index: int, position: int) -> float:
        _integer(candidate_id, "candidate_id", 0, self.pool_groups - 1)
        _integer(rollout_index, "rollout_index", 0, self.group_size - 1)
        _integer(position, "position", 0, 2**32 - 1)
        message = (DRAW_DOMAIN + self._digest + candidate_id.to_bytes(4, "big")
                   + rollout_index.to_bytes(4, "big") + position.to_bytes(4, "big"))
        # 53 bits ensure the floating-point result can never round up to 1.0.
        bits = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") >> 11
        return bits / 2**53


def validate_rollout_selection(pool: SeedPool, selection: PoolSelection,
                               commits: list[dict]) -> None:
    pool.validate_selection(selection, rollout_count=len(commits))
    for index, commit in enumerate(commits):
        metadata = commit.get("rollout")
        if not isinstance(metadata, dict) or metadata.get("episode") is not None:
            raise SeedPoolError("public group pool requires single-turn rollouts")
        pool.validate_rollout_binding(metadata.get("seed_pool"), selection, index)
