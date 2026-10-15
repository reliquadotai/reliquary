"""Versioned semantic contract shared by signer clients and the signer API."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.shared.ckpt_index import index_signing_payload, run_prefix

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


_SHA256 = r"^[0-9a-f]{64}$"
_KEY = r"^rl/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+$"
_RUN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MAX_INDEX_ANCHORS = 256
MAX_INDEX_DELTAS = 4096
MAX_INDEX_FILES = 1024


class _IndexObject(_StrictModel):
    key: str = Field(pattern=_KEY, max_length=512)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256)


class _IndexModelFile(_IndexObject):
    name: str = Field(min_length=1, max_length=128)


class _IndexVersion(_StrictModel):
    version: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256)


class _IndexAnchor(_IndexVersion):
    files: list[_IndexObject] = Field(min_length=1, max_length=MAX_INDEX_FILES)
    model_files: list[_IndexModelFile] | None = Field(default=None, max_length=MAX_INDEX_FILES)


class _IndexDelta(_IndexObject):
    from_: int = Field(alias="from", ge=0)
    to: int = Field(ge=0)
    target_sha256: str = Field(pattern=_SHA256)


class RlCheckpointIndex(_StrictModel):
    """The trainer's checkpoint index (``shared/ckpt_index``), field by field: the only shape the signer
    signs under the index domain. The signed bytes are rebuilt here, never taken from the client."""

    format: Literal["reliquary-ckpt-index/1"]
    run: str = Field(min_length=1, max_length=128)
    latest: _IndexVersion
    anchors: list[_IndexAnchor] = Field(max_length=MAX_INDEX_ANCHORS)
    deltas: list[_IndexDelta] = Field(max_length=MAX_INDEX_DELTAS)
    published_at: float | int | None = None

    @model_validator(mode="after")
    def validate_index(self) -> "RlCheckpointIndex":
        if not _RUN.fullmatch(self.run) or self.run in {".", ".."}:
            raise ValueError("run must be a single path component")
        prefix = run_prefix(self.run) + "/"
        latest = self.latest.version
        if any(a.version > latest for a in self.anchors):
            raise ValueError("an anchor is newer than the latest version")
        if any(not d.from_ < d.to <= latest for d in self.deltas):
            raise ValueError("a delta must go forwards, up to the latest version")
        if self.published_at is not None and not math.isfinite(float(self.published_at)):
            raise ValueError("published_at must be finite")
        objects = [
            *(f for a in self.anchors for f in a.files),
            *(f for a in self.anchors for f in (a.model_files or ())),
            *self.deltas,
        ]
        if any(not o.key.startswith(prefix) or ".." in o.key.split("/") for o in objects):
            raise ValueError("every object key must live under the run's prefix")
        return self


def rl_index_digest(index: dict) -> str:
    return hashlib.sha256(index_signing_payload(index)).hexdigest()


class RlIndexSignRequest(_StrictModel):
    """Sign one checkpoint index (served on ``GET /rl/checkpoint``). The payload is the index domain
    prefix and the canonical index JSON (``index_signing_payload``), rebuilt by the signer after validation."""

    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    netuid: int = Field(ge=0, le=65535)
    index: dict

    @model_validator(mode="after")
    def validate_index(self) -> "RlIndexSignRequest":
        RlCheckpointIndex.model_validate(self.index)
        if self.operation_id != rl_index_operation_id(netuid=self.netuid, index=self.index):
            raise ValueError("operation_id does not bind this checkpoint index")
        return self

    @property
    def run(self) -> str:
        return str(self.index["run"])

    @property
    def version(self) -> int:
        return int(self.index["latest"]["version"])


class RlIndexSignResponse(CheckpointSignResponse):
    pass


class SignerHealth(_StrictModel):
    protocol_version: Literal[1] = SIGNER_PROTOCOL_VERSION
    status: Literal["ok"] = "ok"
    signer_hotkey: str = Field(min_length=10, max_length=100)
    network: str = Field(min_length=1, max_length=200)
    netuid: int = Field(ge=0, le=65535)
    repo_id: str = Field(min_length=3, max_length=200)
    axon_ip: str | None = None
    axon_port: int | None = None


class WeightSubmissionStatus(_StrictModel):
    signer_hotkey: str = Field(min_length=10, max_length=100)
    last_attempt_epoch: int | None = Field(default=None, ge=0)


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


def rl_index_operation_id(*, netuid: int, index: dict) -> str:
    return operation_id(
        "rl-index-sign",
        {"index_sha256": rl_index_digest(index), "netuid": int(netuid)},
    )
