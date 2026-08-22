"""Binary per-window training payload for the detached trainer.

One npz object per sealed window under ``reliquary/training/``. Carries
everything train_step consumes — tokens, miner token_logprobs, seal-time
verify logprobs (pi_old), reward, forced/truncated — so the trainer never
recomputes a forward. pi_old is fp32 log-space; encoding gates on
T_PROTO == 1.0 exactly like _verify_logprobs_for_training.
"""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

PAYLOAD_SCHEMA_VERSION = 1
TOMBSTONE_SCHEMA_VERSION = 1


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

    header = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "window_start": int(window_start),
        "checkpoint_revision": str(checkpoint_revision),
        "env_order": list(env_order),
        "window_quarantine": window_quarantine,
        "groups": groups_meta,
        "rollout_meta": rollout_meta,
        "validated_spans": validated_spans,
        "termination_paths": termination_paths,
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        header=np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8
        ),
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
        header = json.loads(bytes(arrays["header"]).decode("utf-8"))
        if header["schema_version"] != PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported payload schema {header['schema_version']}"
            )
        self.schema_version = header["schema_version"]
        self.window_start = int(header["window_start"])
        self.checkpoint_revision = str(header["checkpoint_revision"])
        self.env_order = list(header["env_order"])
        self.window_quarantine = dict(header["window_quarantine"])
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
    *, window_start: int, failure_stage: str, failure_type: str
) -> bytes:
    return json.dumps({
        "schema_version": TOMBSTONE_SCHEMA_VERSION,
        "window_start": int(window_start),
        "failure_stage": str(failure_stage),
        "failure_type": str(failure_type),
    }).encode("utf-8")


def decode_tombstone(data: bytes) -> dict[str, Any]:
    doc = json.loads(data.decode("utf-8"))
    if doc.get("schema_version") != TOMBSTONE_SCHEMA_VERSION:
        raise ValueError("unsupported tombstone schema")
    return doc
