"""Durable, local-only planner for experimental checkpoint epochs.

The planner has no submission transport. Generation is supplied through one
callback, so any conforming backend can prepare work. Release is only an atomic
move into a local directory after the live validator state matches every bound
field and the exact target window is OPEN.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Collection, Literal, Mapping, Sequence

from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CHECKPOINT_EPOCH_CAPABILITY_ID,
    EpochPlan,
    canonical_manifest_bytes,
    generation_contract_sha256,
    manifest_sha256,
    parse_epoch_plan,
    validate_epoch_plan,
)


CHECKPOINT_EPOCH_ENDPOINT = "/checkpoint-epoch"
_ACTIVE = frozenset({"planned", "generating", "prepared", "releasing"})
_TERMINAL = frozenset({"released", "quarantined"})
_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_RANDOMNESS_KEYS = frozenset({
    "auction_randomness",
    "post_seal_randomness",
    "seal_beacon",
    "seal_randomness",
    "selection_randomness",
    "tie_break_randomness",
})


class ShadowPlannerError(RuntimeError):
    pass


class ShadowPlannerDisabled(ShadowPlannerError):
    pass


class EpochPlanMismatch(ShadowPlannerError):
    pass


@dataclass(frozen=True, slots=True)
class ShadowPlannerConfig:
    spool_root: Path
    enabled: bool = False
    max_queue_groups: int = 64
    max_queue_bytes: int = 512 * 1024 * 1024
    max_groups_per_environment_window: int = 8
    target_groups_per_environment_window: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "spool_root", Path(self.spool_root))
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        for name in (
            "max_queue_groups",
            "max_queue_bytes",
            "max_groups_per_environment_window",
            "target_groups_per_environment_window",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ShadowWorkSpec:
    window_offset: int
    environment: str
    prompt_idx: int
    prompt_content_sha256: str
    estimated_payload_bytes: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_offset, bool)
            or not isinstance(self.window_offset, int)
            or self.window_offset < 0
            or isinstance(self.prompt_idx, bool)
            or not isinstance(self.prompt_idx, int)
            or self.prompt_idx < 0
        ):
            raise ValueError("window_offset and prompt_idx must be non-negative")
        if not isinstance(self.environment, str) or not self.environment:
            raise ValueError("environment must be non-empty")
        _require_digest("prompt_content_sha256", self.prompt_content_sha256)
        if (
            isinstance(self.estimated_payload_bytes, bool)
            or not isinstance(self.estimated_payload_bytes, int)
            or self.estimated_payload_bytes < 1
        ):
            raise ValueError("estimated_payload_bytes must be positive")


@dataclass(frozen=True, slots=True)
class PreparedGroup:
    payload: Mapping[str, Any]
    gpu_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.gpu_seconds, bool)
            or not isinstance(self.gpu_seconds, (int, float))
            or not math.isfinite(self.gpu_seconds)
            or self.gpu_seconds < 0
        ):
            raise ValueError("gpu_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EpochWorkBinding:
    manifest_sha256: str
    epoch_id: str
    profile_id: str
    protocol_version: int
    generation_contract_sha256: str
    training_mode: str
    checkpoint_number: int
    checkpoint_repo_id: str
    checkpoint_revision: str
    window_offset: int
    window_number: int
    generation_randomness: str
    environment: str
    prompt_slice_start: int
    prompt_slice_stop: int
    prompt_idx: int
    prompt_content_sha256: str


@dataclass(frozen=True, slots=True)
class ShadowRecord:
    identity: str
    status: str
    created_at: float
    updated_at: float
    binding: EpochWorkBinding
    reservation_bytes: int
    payload: Mapping[str, Any] | None = None
    actual_gpu_seconds: float | None = None
    prepared_at: float | None = None
    released_at: float | None = None
    quarantine_reason: str | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _get(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _state_text(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


class EpochShadowPlanner:
    """Bounded spool with an intentionally backend-neutral prepare callback."""

    def __init__(
        self,
        config: ShadowPlannerConfig,
        *,
        clock: Callable[[], float] = time.time,
        beacon_verifier: Callable[[BeaconBinding], bool] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock
        self._beacon_verifier = beacon_verifier
        if config.enabled:
            self._create_directories()
            self._recover()

    @property
    def queue_dir(self) -> Path:
        return self.config.spool_root / "queue"

    @property
    def released_dir(self) -> Path:
        return self.config.spool_root / "released"

    @property
    def quarantine_dir(self) -> Path:
        return self.config.spool_root / "quarantine"

    @property
    def active_plan_path(self) -> Path:
        return self.config.spool_root / "active-plan.json"

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise ShadowPlannerDisabled("checkpoint epoch planner is disabled")

    def _create_directories(self) -> None:
        self.config.spool_root.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(exist_ok=True)
        self.released_dir.mkdir(exist_ok=True)
        self.quarantine_dir.mkdir(exist_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write(self, path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(_canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

    def _move(self, source: Path, destination: Path) -> None:
        os.replace(source, destination)
        self._fsync_directory(source.parent)
        if source.parent != destination.parent:
            self._fsync_directory(destination.parent)

    @staticmethod
    def _record_dict(record: ShadowRecord) -> dict[str, Any]:
        value = asdict(record)
        value["schema_version"] = 1
        return value

    @staticmethod
    def _record_from_dict(value: Mapping[str, Any]) -> ShadowRecord:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported shadow record schema")
        binding = EpochWorkBinding(**dict(value["binding"]))
        fields = {
            key: item
            for key, item in value.items()
            if key not in {"schema_version", "binding"}
        }
        record = ShadowRecord(binding=binding, **fields)
        EpochShadowPlanner._validate_record(record)
        return record

    @staticmethod
    def _identity(binding: EpochWorkBinding) -> str:
        return hashlib.sha256(_canonical_json(asdict(binding))).hexdigest()

    @staticmethod
    def _validate_record(record: ShadowRecord) -> None:
        if record.status not in _ACTIVE | _TERMINAL:
            raise ValueError("unknown shadow status")
        _require_digest("identity", record.identity)
        binding = record.binding
        for name in (
            "manifest_sha256",
            "generation_contract_sha256",
            "generation_randomness",
            "prompt_content_sha256",
        ):
            _require_digest(name, getattr(binding, name))
        if record.identity != EpochShadowPlanner._identity(binding):
            raise ValueError("shadow identity does not bind the record")
        if not (
            binding.prompt_slice_start
            <= binding.prompt_idx
            < binding.prompt_slice_stop
        ):
            raise ValueError("bound prompt is outside its slice")
        if record.reservation_bytes < 1:
            raise ValueError("reservation_bytes must be positive")

    def _write_record(self, directory: Path, record: ShadowRecord) -> Path:
        path = directory / f"{record.identity}.json"
        self._write(path, self._record_dict(record))
        return path

    def _read_record(self, path: Path) -> ShadowRecord:
        encoded = path.read_bytes()
        value = json.loads(encoded)
        if not isinstance(value, dict) or encoded != _canonical_json(value):
            raise ValueError("shadow record is not canonical")
        record = self._record_from_dict(value)
        if path.stem != record.identity:
            raise ValueError("shadow filename differs from identity")
        return record

    def _records(self, directory: Path) -> list[tuple[Path, ShadowRecord]]:
        result: list[tuple[Path, ShadowRecord]] = []
        for path in sorted(directory.glob("*.json")):
            try:
                result.append((path, self._read_record(path)))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                if directory != self.quarantine_dir:
                    self._move(path, self.quarantine_dir / f"corrupt-{path.name}")
        return result

    def _quarantine(
        self,
        path: Path,
        record: ShadowRecord,
        reason: str,
    ) -> None:
        quarantined = replace(
            record,
            status="quarantined",
            updated_at=self._clock(),
            quarantine_reason=reason,
        )
        self._write(path, self._record_dict(quarantined))
        self._move(path, self.quarantine_dir / path.name)

    def _recover(self) -> None:
        for path, record in self._records(self.queue_dir):
            if record.status in {"generating", "releasing"}:
                self._quarantine(path, record, "ambiguous_restart")
            elif record.status in _TERMINAL:
                self._quarantine(path, record, "ambiguous_terminal_location")
        for path, record in self._records(self.released_dir):
            if record.status == "releasing":
                self._write_record(
                    self.released_dir,
                    replace(record, status="released", updated_at=self._clock()),
                )
            elif record.status != "released":
                self._quarantine(path, record, "ambiguous_terminal_location")

    def adopt_plan(self, plan: EpochPlan, expected_manifest_sha256: str) -> str:
        self._require_enabled()
        validate_epoch_plan(plan)
        if plan.experimental_capability_id != CHECKPOINT_EPOCH_CAPABILITY_ID:
            raise EpochPlanMismatch("unsupported checkpoint epoch capability")
        digest = manifest_sha256(plan)
        if digest != expected_manifest_sha256:
            raise EpochPlanMismatch("checkpoint epoch manifest hash mismatch")
        if self._beacon_verifier is None:
            raise EpochPlanMismatch("public epoch beacon verifier is required")
        if self._beacon_verifier(plan.epoch_beacon) is not True:
            raise EpochPlanMismatch("public epoch beacon verification failed")

        if self.active_plan_path.exists():
            current = json.loads(self.active_plan_path.read_bytes())
            if current.get("manifest_sha256") != digest:
                self.invalidate_all("plan_replaced")
        self._write(
            self.active_plan_path,
            {
                "schema_version": 1,
                "manifest_sha256": digest,
                "canonical_manifest": json.loads(canonical_manifest_bytes(plan)),
            },
        )
        return digest

    def fetch_and_adopt_plan(
        self,
        fetch_read_only: Callable[[str], bytes | str],
        *,
        expected_manifest_sha256: str | None = None,
    ) -> EpochPlan:
        self._require_enabled()
        raw = fetch_read_only(CHECKPOINT_EPOCH_ENDPOINT)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        expected = expected_manifest_sha256 or hashlib.sha256(encoded).hexdigest()
        plan = parse_epoch_plan(
            encoded,
            expected_manifest_sha256=expected,
        )
        self.adopt_plan(plan, expected)
        return plan

    @staticmethod
    def _window_and_slice(plan: EpochPlan, spec: ShadowWorkSpec):
        if spec.window_offset >= len(plan.windows):
            raise EpochPlanMismatch("work offset is outside the plan")
        window = plan.windows[spec.window_offset]
        slices = [
            prompt_slice
            for prompt_slice in window.prompt_slices
            if prompt_slice.environment == spec.environment
        ]
        if len(slices) != 1:
            raise EpochPlanMismatch("environment slice is missing or ambiguous")
        prompt_slice = slices[0]
        if not prompt_slice.start <= spec.prompt_idx < prompt_slice.stop:
            raise EpochPlanMismatch("work prompt is outside the plan slice")
        return window, prompt_slice

    def enqueue(
        self,
        plan: EpochPlan,
        *,
        expected_manifest_sha256: str,
        specs: Sequence[ShadowWorkSpec],
    ) -> list[ShadowRecord]:
        digest = self.adopt_plan(plan, expected_manifest_sha256)
        queued = self.records("queue")
        count = sum(record.status in _ACTIVE for record in queued)
        reserved = sum(
            record.reservation_bytes
            for record in queued
            if record.status in _ACTIVE
        )
        per_window: dict[tuple[str, int], int] = defaultdict(int)
        for record in queued:
            if record.status in _ACTIVE:
                per_window[
                    (record.binding.environment, record.binding.window_number)
                ] += 1

        accepted: list[ShadowRecord] = []
        for spec in sorted(
            specs,
            key=lambda item: (
                item.window_offset,
                item.prompt_idx,
                item.environment,
            ),
        ):
            window, prompt_slice = self._window_and_slice(plan, spec)
            binding = EpochWorkBinding(
                manifest_sha256=digest,
                epoch_id=plan.epoch_id,
                profile_id=plan.protocol.profile_id,
                protocol_version=plan.protocol.protocol_version,
                generation_contract_sha256=(
                    plan.protocol.generation_contract_sha256
                ),
                training_mode=plan.training_mode,
                checkpoint_number=plan.checkpoint.number,
                checkpoint_repo_id=plan.checkpoint.repo_id,
                checkpoint_revision=plan.checkpoint.revision,
                window_offset=window.offset,
                window_number=window.window_number,
                generation_randomness=window.generation_randomness,
                environment=spec.environment,
                prompt_slice_start=prompt_slice.start,
                prompt_slice_stop=prompt_slice.stop,
                prompt_idx=spec.prompt_idx,
                prompt_content_sha256=spec.prompt_content_sha256,
            )
            identity = self._identity(binding)
            paths = (
                self.queue_dir / f"{identity}.json",
                self.released_dir / f"{identity}.json",
                self.quarantine_dir / f"{identity}.json",
            )
            key = (binding.environment, binding.window_number)
            if any(path.exists() for path in paths):
                continue
            if (
                count >= self.config.max_queue_groups
                or reserved + spec.estimated_payload_bytes
                > self.config.max_queue_bytes
                or per_window[key]
                >= self.config.max_groups_per_environment_window
            ):
                continue
            now = self._clock()
            record = ShadowRecord(
                identity=identity,
                status="planned",
                created_at=now,
                updated_at=now,
                binding=binding,
                reservation_bytes=spec.estimated_payload_bytes,
            )
            self._write_record(self.queue_dir, record)
            accepted.append(record)
            count += 1
            reserved += record.reservation_bytes
            per_window[key] += 1
        return accepted

    @staticmethod
    def _validate_payload(record: ShadowRecord, payload: Mapping[str, Any]) -> None:
        pending: list[Any] = [payload]
        visited: set[int] = set()
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if isinstance(value, Mapping):
                if _FORBIDDEN_RANDOMNESS_KEYS.intersection(value):
                    raise ShadowPlannerError(
                        "prepared payload contains final selection randomness"
                    )
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)

        binding = record.binding
        expected = {
            "prompt_idx": binding.prompt_idx,
            "window_start": binding.window_number,
            "checkpoint_hash": binding.checkpoint_revision,
            "protocol_version": binding.protocol_version,
            "generation_profile_id": binding.profile_id,
            "generation_randomness": binding.generation_randomness,
        }
        for key, expected_value in expected.items():
            if key in payload and payload[key] != expected_value:
                raise ShadowPlannerError(f"prepared payload has wrong {key}")
        rollouts = payload.get("rollouts")
        if not isinstance(rollouts, list) or not rollouts:
            raise ShadowPlannerError("prepared payload must contain rollouts")
        for rollout in rollouts:
            if not isinstance(rollout, Mapping):
                raise ShadowPlannerError("prepared rollout must be an object")
            if rollout.get("env_name") != binding.environment:
                raise ShadowPlannerError("prepared rollout has wrong environment")
            commit = rollout.get("commit")
            if not isinstance(commit, Mapping):
                raise ShadowPlannerError("prepared rollout has no commit")
            beacon = commit.get("beacon", {})
            if beacon.get("randomness") != binding.generation_randomness:
                raise ShadowPlannerError("prepared rollout has wrong randomness")

    def prepare_next(
        self,
        prepare: Callable[[ShadowRecord], PreparedGroup],
    ) -> ShadowRecord | None:
        self._require_enabled()
        planned = [
            (path, record)
            for path, record in self._records(self.queue_dir)
            if record.status == "planned"
        ]
        if not planned:
            return None
        path, record = min(
            planned,
            key=lambda item: (
                item[1].binding.window_offset,
                item[1].binding.prompt_idx,
                item[1].binding.environment,
            ),
        )
        generating = replace(
            record,
            status="generating",
            updated_at=self._clock(),
        )
        self._write(path, self._record_dict(generating))
        try:
            result = prepare(generating)
            if not isinstance(result, PreparedGroup):
                raise TypeError("prepare callback must return PreparedGroup")
            payload = dict(result.payload)
            self._validate_payload(generating, payload)
        except Exception:
            self._quarantine(path, generating, "generation_failed_ambiguous")
            raise

        reservation = max(
            generating.reservation_bytes,
            len(_canonical_json(payload)),
        )
        current_bytes = sum(
            item.reservation_bytes
            for item in self.records("queue")
            if item.status in _ACTIVE and item.identity != record.identity
        )
        if current_bytes + reservation > self.config.max_queue_bytes:
            self._quarantine(path, generating, "queue_bytes_exceeded")
            return None
        prepared = replace(
            generating,
            status="prepared",
            updated_at=self._clock(),
            payload=payload,
            actual_gpu_seconds=float(result.gpu_seconds),
            prepared_at=self._clock(),
            reservation_bytes=reservation,
        )
        self._write(path, self._record_dict(prepared))
        return prepared

    def invalidate_all(self, reason: str) -> int:
        self._require_enabled()
        count = 0
        for path, record in self._records(self.queue_dir):
            if record.status in _ACTIVE:
                self._quarantine(path, record, reason)
                count += 1
        return count

    @staticmethod
    def _contract_hash(state: Mapping[str, Any] | Any) -> str | None:
        contract = _get(state, "generation_contract")
        if isinstance(contract, Mapping):
            return generation_contract_sha256(contract)
        return None

    @staticmethod
    def _global_mismatch(record: ShadowRecord, state) -> str | None:
        binding = record.binding
        expected = (
            ("checkpoint_epoch_id", binding.epoch_id, "epoch_changed"),
            (
                "checkpoint_epoch_manifest_sha256",
                binding.manifest_sha256,
                "manifest_changed",
            ),
            ("generation_profile_id", binding.profile_id, "profile_changed"),
            ("protocol_version", binding.protocol_version, "protocol_changed"),
            ("checkpoint_n", binding.checkpoint_number, "checkpoint_changed"),
            (
                "checkpoint_repo_id",
                binding.checkpoint_repo_id,
                "checkpoint_changed",
            ),
            (
                "checkpoint_revision",
                binding.checkpoint_revision,
                "checkpoint_changed",
            ),
        )
        for field, value, reason in expected:
            if _get(state, field) != value:
                return reason
        if EpochShadowPlanner._contract_hash(state) != (
            binding.generation_contract_sha256
        ):
            return "contract_changed"
        return None

    def release_ready(
        self,
        *,
        live_state: Mapping[str, Any] | Any,
        cooldown_prompts_by_environment: Mapping[str, Collection[int]],
        prompt_sha256: Callable[[str, int], str],
    ) -> list[Path]:
        """Move exact-OPEN, fully revalidated work to the local release spool."""
        self._require_enabled()
        queued = self._records(self.queue_dir)
        if not queued:
            return []
        mismatch = self._global_mismatch(queued[0][1], live_state)
        if mismatch is not None:
            self.invalidate_all(mismatch)
            return []
        try:
            live_window = int(_get(live_state, "window_n"))
        except (TypeError, ValueError):
            return []
        live_open = _state_text(_get(live_state, "state", "")) == "open"
        released: list[Path] = []
        for path, record in self._records(self.queue_dir):
            binding = record.binding
            if record.status != "prepared":
                continue
            if live_window != binding.window_number or not live_open:
                continue
            if _get(live_state, "randomness") != binding.generation_randomness:
                self.invalidate_all("generation_randomness_mismatch")
                return []
            cooldown = cooldown_prompts_by_environment.get(binding.environment)
            if cooldown is None:
                continue
            if binding.prompt_idx in cooldown:
                self._quarantine(path, record, "prompt_in_cooldown")
                continue
            try:
                current_prompt_hash = prompt_sha256(
                    binding.environment,
                    binding.prompt_idx,
                )
            except Exception:
                continue
            if current_prompt_hash != binding.prompt_content_sha256:
                self._quarantine(path, record, "prompt_content_changed")
                continue
            if record.payload is None:
                self._quarantine(path, record, "prepared_payload_missing")
                continue
            try:
                self._validate_payload(record, record.payload)
            except ShadowPlannerError:
                self._quarantine(path, record, "payload_binding_changed")
                continue

            releasing = replace(
                record,
                status="releasing",
                updated_at=self._clock(),
                released_at=self._clock(),
            )
            self._write(path, self._record_dict(releasing))
            destination = self.released_dir / path.name
            self._move(path, destination)
            self._write(
                destination,
                self._record_dict(
                    replace(
                        releasing,
                        status="released",
                        updated_at=self._clock(),
                    )
                ),
            )
            released.append(destination)
        return released

    def release_epoch_ready(
        self,
        *,
        live_states: Mapping[tuple[str, int], Mapping[str, Any] | Any],
        prompt_sha256: Callable[[str, int], str],
    ) -> list[Path]:
        """Release work after polling each exact concurrently open lane."""
        self._require_enabled()
        released: list[Path] = []
        for (environment, window_number), state in sorted(
            live_states.items(),
            key=lambda item: (item[0][1], item[0][0]),
        ):
            try:
                state_window = int(_get(state, "window_n"))
            except (TypeError, ValueError):
                continue
            if state_window != int(window_number):
                continue
            cooldown = _get(state, "cooldown_prompts", ())
            if not isinstance(cooldown, Collection) or isinstance(
                cooldown, (str, bytes)
            ):
                continue
            released.extend(self.release_ready(
                live_state=state,
                cooldown_prompts_by_environment={environment: cooldown},
                prompt_sha256=prompt_sha256,
            ))
        return released

    def records(
        self,
        location: Literal["queue", "released", "quarantine"] = "queue",
    ) -> list[ShadowRecord]:
        directory = {
            "queue": self.queue_dir,
            "released": self.released_dir,
            "quarantine": self.quarantine_dir,
        }[location]
        if not directory.exists():
            return []
        return [record for _, record in self._records(directory)]

    def metrics(self, *, now: float | None = None) -> dict[str, Any]:
        observation = self._clock() if now is None else float(now)
        queue = self.records("queue") if self.queue_dir.exists() else []
        released = self.records("released") if self.released_dir.exists() else []
        quarantined = (
            self.records("quarantine") if self.quarantine_dir.exists() else []
        )
        all_records = queue + released + quarantined
        coverage: dict[str, dict[str, dict[str, int]]] = {}
        for record in all_records:
            windows = coverage.setdefault(record.binding.environment, {})
            counts = windows.setdefault(
                str(record.binding.window_number),
                {"planned": 0, "prepared": 0, "released": 0, "discarded": 0},
            )
            counts["planned"] += 1
            if record.status in {"prepared", "released"}:
                counts["prepared"] += 1
            if record.status == "released":
                counts["released"] += 1
            if record.status == "quarantined":
                counts["discarded"] += 1
        underfill = {
            environment: {
                window: max(
                    self.config.target_groups_per_environment_window
                    - counts["prepared"],
                    0,
                )
                for window, counts in windows.items()
            }
            for environment, windows in coverage.items()
        }
        ages = [max(0.0, observation - record.created_at) for record in queue]
        return {
            "experimental_enabled": self.config.enabled,
            "network_send_capable": False,
            "gpu_seconds_generated": sum(
                record.actual_gpu_seconds or 0.0 for record in all_records
            ),
            "valid_groups_prepared_local": sum(
                record.status in {"prepared", "released"}
                for record in all_records
            ),
            "valid_groups_released_local": len(released),
            "discarded_stale_work": len(quarantined),
            "queue_groups": len(queue),
            "queue_bytes": sum(record.reservation_bytes for record in queue),
            "queue_age_seconds": max(ages) if ages else None,
            "coverage_by_environment_window": coverage,
            "underfill_opportunity_by_environment_window": underfill,
        }


__all__ = [
    "CHECKPOINT_EPOCH_ENDPOINT",
    "EpochPlanMismatch",
    "EpochShadowPlanner",
    "EpochWorkBinding",
    "PreparedGroup",
    "ShadowPlannerConfig",
    "ShadowPlannerDisabled",
    "ShadowPlannerError",
    "ShadowRecord",
    "ShadowWorkSpec",
]
