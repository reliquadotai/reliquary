"""Binary per-window training payload for the detached trainer.

One npz object per sealed window under ``reliquary/training/``. Carries
everything train_step consumes — tokens, miner token_logprobs, seal-time
verify logprobs (pi_old), reward, forced/truncated — so the trainer never
recomputes a forward. pi_old is fp32 log-space; encoding gates on
T_PROTO == 1.0 exactly like _verify_logprobs_for_training.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from reliquary.shared.strict_json import strict_json_loads

PAYLOAD_SCHEMA_VERSION = 2
TOMBSTONE_SCHEMA_VERSION = 2
CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION = 3
_SUPPORTED_PAYLOAD_SCHEMA_VERSIONS = {
    1,
    PAYLOAD_SCHEMA_VERSION,
    CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION,
}
_SUPPORTED_TOMBSTONE_SCHEMA_VERSIONS = {
    1,
    TOMBSTONE_SCHEMA_VERSION,
    CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION,
}
_TRAINING_IDENTITY_KEYS = (
    "protocol_profile_id",
    "protocol_version",
    "training_run_id",
    "generation_contract_sha256",
)


class TrainingPayloadProtocolMismatch(RuntimeError):
    """A detached-training artifact belongs to another protocol/run."""


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointEpochTrainingBinding:
    """Exact epoch/lane identity carried through the detached journal."""

    epoch_id: str
    manifest_sha256: str
    training_run_id: str
    training_mode: str
    first_window: int
    lane_offset: int
    window_count: int
    target_groups_per_environment_lane: int

    @property
    def final_lane(self) -> bool:
        return self.lane_offset == self.window_count - 1

    @property
    def publication_units(self) -> int:
        if self.training_mode == "aggregate_one_step" and self.final_lane:
            return self.window_count
        return 1


def _checkpoint_epoch_binding(
    value: CheckpointEpochTrainingBinding | Mapping[str, Any],
    *,
    window_start: int,
) -> CheckpointEpochTrainingBinding:
    window_start = _nonnegative_int(
        window_start,
        field="checkpoint epoch window",
    )
    if isinstance(value, CheckpointEpochTrainingBinding):
        binding = value
    elif isinstance(value, Mapping) and set(value) == {
        "epoch_id",
        "manifest_sha256",
        "training_run_id",
        "training_mode",
        "first_window",
        "lane_offset",
        "window_count",
        "target_groups_per_environment_lane",
    }:
        binding = CheckpointEpochTrainingBinding(**dict(value))
    else:
        raise ValueError("invalid checkpoint epoch training binding")
    for name, digest in (
        ("epoch_id", binding.epoch_id),
        ("manifest_sha256", binding.manifest_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid checkpoint epoch {name}")
    if (
        not isinstance(binding.training_run_id, str)
        or not binding.training_run_id
        or len(binding.training_run_id) > 256
    ):
        raise ValueError("invalid checkpoint epoch training run")
    if binding.training_mode not in {
        "aggregate_one_step",
        "sequential_steps",
    }:
        raise ValueError("invalid checkpoint epoch training mode")
    integer_fields = (
        binding.first_window,
        binding.lane_offset,
        binding.window_count,
        binding.target_groups_per_environment_lane,
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) for item in integer_fields
    ):
        raise ValueError("checkpoint epoch training integers are invalid")
    if (
        binding.first_window < 0
        or binding.window_count < 1
        or binding.lane_offset < 0
        or binding.lane_offset >= binding.window_count
        or binding.target_groups_per_environment_lane < 1
        or binding.first_window + binding.lane_offset != window_start
    ):
        raise ValueError("checkpoint epoch training range is invalid")
    return binding


def active_training_identity() -> dict[str, Any]:
    """Return the protocol identity shared by payloads and manifests.

    Imports are lazy because this codec is also used by tooling that selects a
    protocol profile before importing ``reliquary.constants``.
    """

    from reliquary.constants import (
        PROTOCOL_GENERATION_CONTRACT,
        PROTOCOL_PROFILE_ID,
        PROTOCOL_VERSION,
        TRAINING_RUN_ID,
    )

    contract = json.dumps(
        PROTOCOL_GENERATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "protocol_profile_id": PROTOCOL_PROFILE_ID,
        "protocol_version": int(PROTOCOL_VERSION),
        "training_run_id": TRAINING_RUN_ID,
        "generation_contract_sha256": hashlib.sha256(contract).hexdigest(),
    }


def _artifact_protocol_header(*, latest_schema_version: int) -> dict[str, Any]:
    """Return a wire header that remains readable by deployed v2-v4 workers.

    Protocol v5 is the first version that requires detached artifacts to carry
    run identity.  Older workers only accept schema 1, so legacy profiles keep
    emitting the original schema and header shape during a rolling image
    upgrade.  V5 emits the latest schema and fails closed on missing identity.
    """

    identity = active_training_identity()
    if int(identity["protocol_version"]) < 5:
        return {"schema_version": 1}
    return {
        "schema_version": int(latest_schema_version),
        **identity,
    }


def validate_training_identity(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    artifact: str,
) -> None:
    """Fail closed when a detached artifact crosses a run boundary."""

    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise TrainingPayloadProtocolMismatch(
                f"{artifact} identity mismatch for {key}: "
                f"expected {expected_value!r}, got {actual.get(key)!r}"
            )


def _pi_old_for_encode(rollout: Any, completion_length: int) -> list[float] | None:
    """Same acceptance ladder as _verify_logprobs_for_training: T=1.0 only,
    full coverage only, finite only. Lazy import so tests can monkeypatch
    reliquary.constants."""
    from reliquary.constants import T_PROTO

    if float(T_PROTO) != 1.0:
        return None
    values = getattr(rollout, "_validated_completion_logprobs", None)
    if not isinstance(values, list) or len(values) != completion_length:
        return None
    if completion_length == 0:
        return None
    try:
        floats = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in floats):
        return None
    return floats


def encode_training_payload(
    window_batches: dict[str, list],
    *,
    window_start: int,
    checkpoint_revision: str,
    env_order: list[str],
    window_quarantine: dict,
    checkpoint_epoch: (
        CheckpointEpochTrainingBinding | Mapping[str, Any] | None
    ) = None,
) -> bytes:
    groups_meta: list[dict[str, Any]] = []
    rollout_meta: list[dict[str, Any]] = []
    # Validator-DERIVED per-rollout state (never wire metadata): the BFT
    # force span and termination path drive loss masking (PR #167) and
    # must survive the hop as the private attrs training reads.
    validated_spans: list[list[int] | None] = []
    termination_paths: list[str | None] = []
    rewards: list[float] = []
    env_names: list[str] = []
    tokens_flat: list[int] = []
    tokens_off: list[int] = [0]
    miner_lp_flat: list[float] = []
    miner_lp_off: list[int] = [0]
    pi_old_flat: list[float] = []
    pi_old_off: list[int] = [0]
    has_pi_old: list[bool] = []

    for env in env_order:
        for group in window_batches.get(env, []):
            groups_meta.append({
                "env": env,
                "prompt_idx": int(getattr(group, "prompt_idx", 0) or 0),
                "n_rollouts": len(group.rollouts),
            })
            for rollout in group.rollouts:
                commit = rollout.commit or {}
                meta = dict(commit.get("rollout") or {})
                miner_lp = list(meta.pop("token_logprobs", []) or [])
                completion_length = int(meta.get("completion_length", 0) or 0)
                rollout_meta.append(meta)
                span = getattr(rollout, "_validated_force_span", None)
                validated_spans.append(
                    [int(span[0]), int(span[1])] if span else None
                )
                term = getattr(rollout, "_validated_termination_path", None)
                termination_paths.append(str(term) if term else None)
                rewards.append(float(getattr(rollout, "reward", 0.0)))
                env_names.append(str(getattr(rollout, "env_name", env)))
                tokens_flat.extend(int(t) for t in commit.get("tokens", []))
                tokens_off.append(len(tokens_flat))
                miner_lp_flat.extend(float(v) for v in miner_lp)
                miner_lp_off.append(len(miner_lp_flat))
                pi_old = _pi_old_for_encode(rollout, completion_length)
                if pi_old is None:
                    has_pi_old.append(False)
                else:
                    has_pi_old.append(True)
                    pi_old_flat.extend(pi_old)
                pi_old_off.append(len(pi_old_flat))

    epoch_binding = (
        _checkpoint_epoch_binding(checkpoint_epoch, window_start=window_start)
        if checkpoint_epoch is not None
        else None
    )
    artifact_schema = (
        CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION
        if epoch_binding is not None
        else PAYLOAD_SCHEMA_VERSION
    )
    protocol_header = _artifact_protocol_header(
        latest_schema_version=artifact_schema,
    )
    if epoch_binding is not None and protocol_header["schema_version"] != (
        CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint epoch journal requires protocol v5+")
    header = {
        **protocol_header,
        "window_start": int(window_start),
        "checkpoint_revision": str(checkpoint_revision),
        "env_order": list(env_order),
        "window_quarantine": window_quarantine,
        "groups": groups_meta,
        "rollout_meta": rollout_meta,
        "validated_spans": validated_spans,
        "termination_paths": termination_paths,
    }
    if epoch_binding is not None:
        header["checkpoint_epoch"] = asdict(epoch_binding)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        header=np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8),
        rewards=np.asarray(rewards, dtype=np.float32),
        env_names=np.asarray(env_names, dtype=np.str_),
        tokens_flat=np.asarray(tokens_flat, dtype=np.int32),
        tokens_off=np.asarray(tokens_off, dtype=np.int64),
        miner_lp_flat=np.asarray(miner_lp_flat, dtype=np.float32),
        miner_lp_off=np.asarray(miner_lp_off, dtype=np.int64),
        pi_old_flat=np.asarray(pi_old_flat, dtype=np.float32),
        pi_old_off=np.asarray(pi_old_off, dtype=np.int64),
        has_pi_old=np.asarray(has_pi_old, dtype=np.bool_),
    )
    return buf.getvalue()


class DecodedPayload:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        header = strict_json_loads(bytes(arrays["header"]))
        if not isinstance(header, dict):
            raise ValueError("training payload header must be an object")
        schema_version = header.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version not in _SUPPORTED_PAYLOAD_SCHEMA_VERSIONS
        ):
            raise ValueError(f"unsupported payload schema {schema_version}")
        self.schema_version = schema_version
        self.training_identity = {
            key: header.get(key) for key in _TRAINING_IDENTITY_KEYS
        }
        self.window_start = _nonnegative_int(
            header.get("window_start"),
            field="training payload window",
        )
        checkpoint_revision = header.get("checkpoint_revision")
        if (
            not isinstance(checkpoint_revision, str)
            or not checkpoint_revision
            or checkpoint_revision.strip() != checkpoint_revision
        ):
            raise ValueError(
                "training payload checkpoint revision must be canonical text"
            )
        self.checkpoint_revision = checkpoint_revision
        self.env_order = list(header["env_order"])
        self.window_quarantine = dict(header["window_quarantine"])
        raw_epoch = header.get("checkpoint_epoch")
        if self.schema_version == CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION:
            if raw_epoch is None:
                raise ValueError("epoch payload omitted checkpoint epoch binding")
            self.checkpoint_epoch = _checkpoint_epoch_binding(
                raw_epoch,
                window_start=self.window_start,
            )
        else:
            if raw_epoch is not None:
                raise ValueError("legacy payload carries checkpoint epoch binding")
            self.checkpoint_epoch = None
        self._groups_meta = header["groups"]
        self._rollout_meta = header["rollout_meta"]
        self._validated_spans = header.get("validated_spans") or []
        self._termination_paths = header.get("termination_paths") or []
        self._arrays = arrays

    def batches(self) -> dict[str, list]:
        a = self._arrays
        out: dict[str, list] = {env: [] for env in self.env_order}
        cursor = 0
        for gm in self._groups_meta:
            rollouts = []
            for _ in range(gm["n_rollouts"]):
                i = cursor
                t0, t1 = int(a["tokens_off"][i]), int(a["tokens_off"][i + 1])
                m0, m1 = int(a["miner_lp_off"][i]), int(a["miner_lp_off"][i + 1])
                meta = dict(self._rollout_meta[i])
                meta["token_logprobs"] = [
                    float(v) for v in a["miner_lp_flat"][m0:m1]
                ]
                rollout = SimpleNamespace(
                    reward=float(a["rewards"][i]),
                    env_name=str(a["env_names"][i]),
                    commit={
                        "tokens": [int(t) for t in a["tokens_flat"][t0:t1]],
                        "rollout": meta,
                    },
                )
                if bool(a["has_pi_old"][i]):
                    p0, p1 = int(a["pi_old_off"][i]), int(a["pi_old_off"][i + 1])
                    rollout._validated_completion_logprobs = [
                        float(v) for v in a["pi_old_flat"][p0:p1]
                    ]
                if i < len(self._validated_spans):
                    span = self._validated_spans[i]
                    if span:
                        rollout._validated_force_span = (
                            int(span[0]), int(span[1]),
                        )
                    term = self._termination_paths[i]
                    if term:
                        rollout._validated_termination_path = str(term)
                rollouts.append(rollout)
                cursor += 1
            out[gm["env"]].append(
                SimpleNamespace(rollouts=rollouts, prompt_idx=gm["prompt_idx"])
            )
        return out


def decode_training_payload(data: bytes) -> DecodedPayload:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return DecodedPayload(arrays)


def encode_tombstone(
    *,
    window_start: int,
    failure_stage: str,
    failure_type: str,
    checkpoint_epoch: (
        CheckpointEpochTrainingBinding | Mapping[str, Any] | None
    ) = None,
) -> bytes:
    epoch_binding = (
        _checkpoint_epoch_binding(checkpoint_epoch, window_start=window_start)
        if checkpoint_epoch is not None
        else None
    )
    artifact_schema = (
        CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION
        if epoch_binding is not None
        else TOMBSTONE_SCHEMA_VERSION
    )
    protocol_header = _artifact_protocol_header(
        latest_schema_version=artifact_schema,
    )
    if epoch_binding is not None and protocol_header["schema_version"] != (
        CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint epoch journal requires protocol v5+")
    doc = {
        **protocol_header,
        "window_start": int(window_start),
        "failure_stage": str(failure_stage),
        "failure_type": str(failure_type),
    }
    if epoch_binding is not None:
        doc["checkpoint_epoch"] = asdict(epoch_binding)
    return json.dumps(doc).encode("utf-8")


def decode_tombstone(data: bytes) -> dict[str, Any]:
    doc = strict_json_loads(data)
    if not isinstance(doc, dict):
        raise ValueError("training tombstone must be an object")
    schema_version = doc.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version not in _SUPPORTED_TOMBSTONE_SCHEMA_VERSIONS
    ):
        raise ValueError("unsupported tombstone schema")
    window_start = _nonnegative_int(
        doc.get("window_start"),
        field="training tombstone window",
    )
    raw_epoch = doc.get("checkpoint_epoch")
    if schema_version == CHECKPOINT_EPOCH_ARTIFACT_SCHEMA_VERSION:
        if raw_epoch is None:
            raise ValueError("epoch tombstone omitted checkpoint epoch binding")
        binding = _checkpoint_epoch_binding(
            raw_epoch,
            window_start=window_start,
        )
        doc["checkpoint_epoch"] = binding
    elif raw_epoch is not None:
        raise ValueError("legacy tombstone carries checkpoint epoch binding")
    return doc


def encode_checkpoint_epoch_marker(
    checkpoint_epoch: CheckpointEpochTrainingBinding | Mapping[str, Any],
    *,
    status: str,
) -> bytes:
    binding = _checkpoint_epoch_binding(
        checkpoint_epoch,
        window_start=(
            checkpoint_epoch.first_window + checkpoint_epoch.lane_offset
            if isinstance(checkpoint_epoch, CheckpointEpochTrainingBinding)
            else int(checkpoint_epoch["first_window"])
            + int(checkpoint_epoch["lane_offset"])
        ),
    )
    if binding.lane_offset != 0:
        raise ValueError("checkpoint epoch marker must bind lane zero")
    if status not in {"completed", "aborted"}:
        raise ValueError("invalid checkpoint epoch marker status")
    return json.dumps(
        {
            "schema_version": 1,
            "status": status,
            "checkpoint_epoch": asdict(binding),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_checkpoint_epoch_marker(data: bytes) -> dict[str, Any]:
    try:
        doc = strict_json_loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid checkpoint epoch marker") from exc
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema_version", "status", "checkpoint_epoch"}
        or type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
        or doc["status"] not in {"completed", "aborted"}
    ):
        raise ValueError("invalid checkpoint epoch marker")
    raw_binding = doc["checkpoint_epoch"]
    if not isinstance(raw_binding, dict):
        raise ValueError("invalid checkpoint epoch marker binding")
    first_window = _nonnegative_int(
        raw_binding.get("first_window"),
        field="checkpoint epoch first window",
    )
    lane_offset = _nonnegative_int(
        raw_binding.get("lane_offset"),
        field="checkpoint epoch lane offset",
    )
    binding = _checkpoint_epoch_binding(
        raw_binding,
        window_start=first_window + lane_offset,
    )
    if binding.lane_offset != 0:
        raise ValueError("checkpoint epoch marker must bind lane zero")
    canonical = encode_checkpoint_epoch_marker(
        binding,
        status=doc["status"],
    )
    if data != canonical:
        raise ValueError("checkpoint epoch marker is not canonical")
    doc["checkpoint_epoch"] = binding
    return doc
