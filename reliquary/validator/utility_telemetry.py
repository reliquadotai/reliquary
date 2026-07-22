"""Private, fail-open utility telemetry for future batch selection research."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_FALSE_VALUES = {"0", "false", "no", "off"}
_SCHEMA_VERSION = 1


def utility_telemetry_enabled() -> bool:
    value = os.environ.get("RELIQUARY_UTILITY_TELEMETRY_ENABLED", "1")
    return value.strip().lower() not in _FALSE_VALUES


def _retention_windows() -> int:
    value = os.environ.get(
        "RELIQUARY_UTILITY_TELEMETRY_RETENTION_WINDOWS", "2048"
    )
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning("Invalid utility telemetry retention %r; using 2048", value)
        return 2048


class UtilityTelemetryWriter:
    """Write one private gzip bundle per completed window."""

    def __init__(self, state_dir: str | Path | None = None) -> None:
        root = Path(
            state_dir
            or os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
        )
        self.directory = root / "utility_telemetry"
        self.retention_windows = _retention_windows()
        self.enabled = utility_telemetry_enabled()
        self.writes_total = 0
        self.failures_total = 0
        self.last_window: int | None = None
        self.last_write_ts: float | None = None
        self.last_error_type: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "schema_version": _SCHEMA_VERSION,
            "retention_windows": self.retention_windows,
            "writes_total": self.writes_total,
            "failures_total": self.failures_total,
            "last_window": self.last_window,
            "last_write_ts": self.last_write_ts,
            "last_error_type": self.last_error_type,
        }

    def _secret(self) -> bytes:
        self.directory.mkdir(parents=True, exist_ok=True)
        key_path = self.directory / ".hmac-key"
        try:
            secret = key_path.read_bytes()
        except FileNotFoundError:
            secret = os.urandom(32)
            try:
                fd = os.open(
                    key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                secret = key_path.read_bytes()
            else:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(secret)
                    handle.flush()
                    os.fsync(handle.fileno())
        if len(secret) != 32:
            raise ValueError("utility telemetry HMAC key must be 32 bytes")
        os.chmod(key_path, 0o600)
        return secret

    @staticmethod
    def _candidate_id(
        window: int,
        env_name: str,
        submission: Any,
        operator_pseudonym: str,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"reliquary/utility-candidate/v1\0")
        hasher.update(int(window).to_bytes(8, "big", signed=False))
        hasher.update(str(env_name).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(bytes.fromhex(operator_pseudonym))
        hasher.update(int(submission.prompt_idx).to_bytes(8, "big"))
        hasher.update(str(submission.prompt_content_sha256).encode("ascii"))
        hasher.update(bytes(submission.selection_digest))
        return hasher.hexdigest()

    @staticmethod
    def _rollouts(submission: Any) -> list[dict[str, Any]]:
        metrics = {
            int(row.get("rollout_idx", index)): dict(row)
            for index, row in enumerate(
                list(getattr(submission, "utility_rollouts", []) or [])
            )
        }
        rows: list[dict[str, Any]] = []
        for index, rollout in enumerate(submission.rollouts):
            commit = rollout.commit or {}
            meta = commit.get("rollout", {}) or {}
            row = {
                "rollout_idx": index,
                "tokens": [int(token) for token in commit.get("tokens", [])],
                "reward": float(getattr(rollout, "reward", 0.0) or 0.0),
                "prompt_length": int(meta.get("prompt_length", 0) or 0),
                "completion_length": int(
                    meta.get("completion_length", 0) or 0
                ),
            }
            row.update(metrics.get(index, {}))
            rows.append(row)
        return rows

    @staticmethod
    def _group_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        rewards = [float(row.get("reward", 0.0) or 0.0) for row in rollouts]
        shifts = [
            float(value)
            for row in rollouts
            if (value := row.get("representation_shift_l2")) is not None
            and math.isfinite(float(value))
        ]

        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        positive_shifts = [
            shift
            for row in rollouts
            if float(row.get("reward", 0.0) or 0.0) > 0.0
            and row.get("representation_shift_l2") is not None
            and math.isfinite(
                shift := float(row["representation_shift_l2"])
            )
        ]
        nonpositive_shifts = [
            shift
            for row in rollouts
            if float(row.get("reward", 0.0) or 0.0) <= 0.0
            and row.get("representation_shift_l2") is not None
            and math.isfinite(
                shift := float(row["representation_shift_l2"])
            )
        ]
        return {
            "rollout_count": len(rollouts),
            "positive_reward_count": sum(reward > 0.0 for reward in rewards),
            "reward_mean": _mean(rewards),
            "completion_length_mean": _mean([
                float(row.get("completion_length", 0) or 0)
                for row in rollouts
            ]),
            "natural_eos_rate": _mean([
                float(bool(row.get("natural_eos", False))) for row in rollouts
            ]),
            "representation_shift_l2_mean": _mean(shifts),
            "positive_reward_shift_l2_mean": _mean(positive_shifts),
            "nonpositive_reward_shift_l2_mean": _mean(nonpositive_shifts),
        }

    def _group(
        self,
        *,
        secret: bytes,
        window: int,
        env_name: str,
        checkpoint_revision: str,
        role: str,
        submission: Any,
        operator_by_hotkey: Mapping[str, str],
    ) -> dict[str, Any]:
        operator = str(
            operator_by_hotkey.get(submission.hotkey, submission.hotkey)
        )
        operator_pseudonym = hmac.new(
            secret,
            operator.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        rollouts = self._rollouts(submission)
        return {
            "candidate_id": self._candidate_id(
                window, env_name, submission, operator_pseudonym
            ),
            "operator_pseudonym": operator_pseudonym,
            "role": role,
            "environment": env_name,
            "checkpoint_revision": checkpoint_revision,
            "prompt_idx": int(submission.prompt_idx),
            "prompt_content_sha256": submission.prompt_content_sha256,
            "target_content_sha256": submission.target_content_sha256,
            "selection_digest": bytes(submission.selection_digest).hex(),
            "group_summary": self._group_summary(rollouts),
            "rollouts": rollouts,
        }

    def _prune(self, current_window: int) -> None:
        minimum = current_window - self.retention_windows + 1
        for path in self.directory.glob("window-*.json.gz"):
            try:
                window = int(path.name.removeprefix("window-").removesuffix(
                    ".json.gz"
                ))
            except ValueError:
                continue
            if window < minimum:
                path.unlink(missing_ok=True)

    def write_window(
        self,
        *,
        window: int,
        checkpoint_revision: str,
        batchers: Mapping[str, Any],
        selected_by_environment: Mapping[str, list[Any]],
    ) -> bool:
        if not self.enabled:
            return False
        try:
            secret = self._secret()
            environments: dict[str, list[dict[str, Any]]] = {}
            for env_name, batcher in batchers.items():
                operator_by_hotkey = dict(
                    getattr(batcher, "_operator_by_hotkey", {}) or {}
                )
                groups = [
                    self._group(
                        secret=secret,
                        window=window,
                        env_name=env_name,
                        checkpoint_revision=checkpoint_revision,
                        role="winner",
                        submission=submission,
                        operator_by_hotkey=operator_by_hotkey,
                    )
                    for submission in selected_by_environment.get(env_name, [])
                ]
                for forensic in list(
                    getattr(batcher, "forensic_sample", []) or []
                ):
                    submission = getattr(forensic, "submission", None)
                    if submission is None:
                        continue
                    groups.append(self._group(
                        secret=secret,
                        window=window,
                        env_name=env_name,
                        checkpoint_revision=checkpoint_revision,
                        role="forensic",
                        submission=submission,
                        operator_by_hotkey=operator_by_hotkey,
                    ))
                environments[env_name] = groups

            payload = {
                "schema_version": _SCHEMA_VERSION,
                "window": int(window),
                "checkpoint_revision": checkpoint_revision,
                "created_at": time.time(),
                "utility_contract": {
                    "entropy": (
                        "full_policy_pre_warp_at_t_proto_stratified_max_64"
                    ),
                    "hidden_anchor": "last_prompt_token",
                    "hidden_delta": "last_trainable_non_eos_minus_anchor",
                },
                "environments": environments,
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            destination = self.directory / f"window-{int(window)}.json.gz"
            fd, temporary = tempfile.mkstemp(
                prefix=".utility.", suffix=".json.gz", dir=self.directory
            )
            try:
                with os.fdopen(fd, "wb") as raw:
                    with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
                        compressed.write(json.dumps(
                            payload,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8"))
                    raw.flush()
                    os.fsync(raw.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
            except Exception:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
            self._prune(int(window))
            self.writes_total += 1
            self.last_window = int(window)
            self.last_write_ts = time.time()
            self.last_error_type = None
            return True
        except Exception as exc:
            self.failures_total += 1
            self.last_error_type = type(exc).__name__
            logger.exception("Private utility telemetry write failed open")
            return False
