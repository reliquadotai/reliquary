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
from types import SimpleNamespace
from typing import Any

import numpy as np

from reliquary.shared.strict_json import strict_json_loads

PAYLOAD_SCHEMA_VERSION = 4
EPISODE_PAYLOAD_SCHEMA_VERSION = PAYLOAD_SCHEMA_VERSION
LEGACY_PAYLOAD_SCHEMA_VERSION = 2
TOMBSTONE_SCHEMA_VERSION = 2
# Schemas 3 and 5 only ever carried the retired checkpoint-epoch binding, and
# that regime never ran outside tests -- no archive holds one.
_SUPPORTED_PAYLOAD_SCHEMA_VERSIONS = {
    1,
    PAYLOAD_SCHEMA_VERSION,
    LEGACY_PAYLOAD_SCHEMA_VERSION,
}
_SUPPORTED_TOMBSTONE_SCHEMA_VERSIONS = {
    1,
    TOMBSTONE_SCHEMA_VERSION,
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
    env_targets: dict[str, int] | None = None,
    window_quarantine: dict,
) -> bytes:
    groups_meta: list[dict[str, Any]] = []
    rollout_meta: list[dict[str, Any]] = []
    # Validator-DERIVED per-rollout state (never wire metadata): the BFT
    # force span and termination path drive loss masking (PR #167) and
    # must survive the hop as the private attrs training reads.
    validated_spans: list[list[int] | None] = []
    assistant_spans: list[list[list[int]] | None] = []
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
                episode_spans = getattr(
                    rollout, "_validated_assistant_spans", None
                )
                assistant_spans.append(
                    (
                        [
                            [int(item[0]), int(item[1])]
                            for item in episode_spans
                        ]
                        if episode_spans is not None
                        else None
                    )
                )
                term = getattr(rollout, "_validated_termination_path", None)
                termination_paths.append(str(term) if term else None)
                rewards.append(float(getattr(rollout, "reward", 0.0)))
                env_names.append(str(getattr(rollout, "env_name", env)))
                tokens_flat.extend(int(t) for t in commit.get("tokens", []))
                tokens_off.append(len(tokens_flat))
                miner_lp_flat.extend(float(v) for v in miner_lp)
                miner_lp_off.append(len(miner_lp_flat))
                policy_length = (
                    sum(end - start for start, end in episode_spans)
                    if episode_spans is not None
                    else completion_length
                )
                pi_old = _pi_old_for_encode(rollout, policy_length)
                if pi_old is None:
                    has_pi_old.append(False)
                else:
                    has_pi_old.append(True)
                    pi_old_flat.extend(pi_old)
                pi_old_off.append(len(pi_old_flat))

    extended = env_targets is not None or any(item is not None for item in assistant_spans)
    artifact_schema = (
        EPISODE_PAYLOAD_SCHEMA_VERSION
        if extended
        else LEGACY_PAYLOAD_SCHEMA_VERSION
    )
    protocol_header = _artifact_protocol_header(
        latest_schema_version=artifact_schema,
    )
    if any(item is not None for item in assistant_spans) and protocol_header["schema_version"] != EPISODE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("episode payload requires protocol v5+ and schema 4")
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
    if protocol_header["schema_version"] == EPISODE_PAYLOAD_SCHEMA_VERSION:
        header["assistant_spans"] = assistant_spans
        resolved_targets = dict(env_targets or {})
        if not env_order or len(set(env_order)) != len(env_order):
            raise ValueError(
                "schema-v2 training payload env_order must be non-empty and unique"
            )
        if set(resolved_targets) != set(env_order):
            raise ValueError(
                "schema-v2 training payload targets must match env_order"
            )
        if any(type(value) is not int or value <= 0 for value in resolved_targets.values()):
            raise ValueError("training payload targets must be positive")
        header["env_targets"] = {
            environment: int(resolved_targets[environment])
            for environment in env_order
        }
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
        has_episode_metadata = "assistant_spans" in header
        if header.get("checkpoint_epoch") is not None:
            raise ValueError("payload carries a retired checkpoint epoch binding")
        requires_targets = self.schema_version >= EPISODE_PAYLOAD_SCHEMA_VERSION or (self.schema_version == 3 and has_episode_metadata)
        self.env_targets = dict(header.get("env_targets") or {})
        if requires_targets or self.env_targets:
            if (
                not self.env_order or len(set(self.env_order)) != len(self.env_order)
                or set(self.env_targets) != set(self.env_order)
            ):
                raise ValueError("training payload targets must match unique env_order")
            if any(type(target) is not int or target <= 0 for target in self.env_targets.values()):
                raise ValueError("training payload targets must be positive integers")
        self.window_quarantine = dict(header["window_quarantine"])
        self._groups_meta = header["groups"]
        self._rollout_meta = header["rollout_meta"]
        self._validated_spans = header.get("validated_spans") or []
        self._assistant_spans = header.get("assistant_spans") or []
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
                if i < len(self._assistant_spans):
                    spans = self._assistant_spans[i]
                    if spans is not None:
                        rollout._validated_assistant_spans = tuple(
                            (int(span[0]), int(span[1]))
                            for span in spans
                        )
                if i < len(self._termination_paths):
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
) -> bytes:
    protocol_header = _artifact_protocol_header(
        latest_schema_version=TOMBSTONE_SCHEMA_VERSION,
    )
    doc = {
        **protocol_header,
        "window_start": int(window_start),
        "failure_stage": str(failure_stage),
        "failure_type": str(failure_type),
    }
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
    _nonnegative_int(
        doc.get("window_start"),
        field="training tombstone window",
    )
    if doc.get("checkpoint_epoch") is not None:
        raise ValueError("tombstone carries a retired checkpoint epoch binding")
    return doc
