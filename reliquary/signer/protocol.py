"""Versioned semantic contract shared by signer clients and the signer API."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SIGNER_PROTOCOL_VERSION = 1
_DOMAIN = b"reliquary-signer-v1\x00"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def checkpoint_payload(checkpoint_n: int, revision: str) -> bytes:
    """Preserve the checkpoint bytes used by production before extraction."""
    return f"{int(checkpoint_n)}|{revision}".encode("utf-8")


def operation_id(kind: str, payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        _DOMAIN + kind.encode("ascii") + b"\x00" + canonical
    ).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CheckpointSignRequest(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    netuid: int = Field(ge=0, le=65535)
    checkpoint_n: int = Field(ge=0)
    repo_id: str = Field(min_length=3, max_length=200)
    revision: str = Field(min_length=40, max_length=40)

    @model_validator(mode="after")
    def validate_claim(self) -> "CheckpointSignRequest":
        if not _REPO_ID.fullmatch(self.repo_id):
            raise ValueError("repo_id must be an owner/repository identifier")
        if not _REVISION.fullmatch(self.revision):
            raise ValueError("revision must be a lowercase 40-hex commit")
        if self.operation_id != checkpoint_operation_id(
            netuid=self.netuid,
            checkpoint_n=self.checkpoint_n,
            repo_id=self.repo_id,
            revision=self.revision,
        ):
            raise ValueError("operation_id does not bind this checkpoint claim")
        return self


class CheckpointSignResponse(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signer_hotkey: str = Field(min_length=10, max_length=100)
    signature_hex: str = Field(pattern=r"^[0-9a-f]+$", min_length=128, max_length=256)
    cached: bool = False


class SetWeightsRequest(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    netuid: int = Field(ge=0, le=65535)
    epoch_id: int = Field(ge=0)
    uids: list[int] = Field(min_length=1, max_length=4096)
    weights: list[float] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_vector(self) -> "SetWeightsRequest":
        if len(self.uids) != len(self.weights):
            raise ValueError("uids and weights must have equal length")
        if self.uids != sorted(self.uids) or len(self.uids) != len(set(self.uids)):
            raise ValueError("uids must be unique and sorted")
        if any(uid < 0 or uid > 65535 for uid in self.uids):
            raise ValueError("uid is outside uint16 range")
        if any(
            not math.isfinite(weight) or weight < 0.0 or weight > 1.0
            for weight in self.weights
        ):
            raise ValueError("weights must be finite values in [0, 1]")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("weight vector must conserve unit mass")
        if self.operation_id != weights_operation_id(
            netuid=self.netuid,
            epoch_id=self.epoch_id,
            uids=self.uids,
            weights=self.weights,
        ):
            raise ValueError("operation_id does not bind this weight vector")
        return self


class SetWeightsResponse(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signer_hotkey: str = Field(min_length=10, max_length=100)
    accepted: bool
    message: str = Field(max_length=500)
    cached: bool = False


class ServeAxonRequest(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    netuid: int = Field(ge=0, le=65535)
    ip: str = Field(min_length=2, max_length=64)
    port: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "ServeAxonRequest":
        try:
            ipaddress.ip_address(self.ip)
        except ValueError as exc:
            raise ValueError("ip must be a literal IPv4 or IPv6 address") from exc
        if self.operation_id != axon_operation_id(
            netuid=self.netuid,
            ip=self.ip,
            port=self.port,
        ):
            raise ValueError("operation_id does not bind this axon endpoint")
        return self


class ServeAxonResponse(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signer_hotkey: str = Field(min_length=10, max_length=100)
    accepted: bool
    message: str = Field(max_length=500)
    cached: bool = False


class SignerHealth(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    status: Literal["ok"] = "ok"
    signer_hotkey: str = Field(min_length=10, max_length=100)
    network: str = Field(min_length=1, max_length=200)
    netuid: int = Field(ge=0, le=65535)
    repo_id: str = Field(min_length=3, max_length=200)
    axon_ip: str | None = None
    axon_port: int | None = None


def checkpoint_operation_id(
    *, netuid: int, checkpoint_n: int, repo_id: str, revision: str
) -> str:
    return operation_id(
        "checkpoint-sign",
        {
            "checkpoint_n": int(checkpoint_n),
            "netuid": int(netuid),
            "repo_id": str(repo_id),
            "revision": str(revision),
        },
    )


def weights_operation_id(
    *, netuid: int, epoch_id: int, uids: list[int], weights: list[float]
) -> str:
    return operation_id(
        "set-weights",
        {
            "epoch_id": int(epoch_id),
            "netuid": int(netuid),
            "uids": [int(uid) for uid in uids],
            "weights": [float(weight) for weight in weights],
        },
    )


def axon_operation_id(*, netuid: int, ip: str, port: int) -> str:
    return operation_id(
        "serve-axon",
        {"ip": str(ip), "netuid": int(netuid), "port": int(port)},
    )
