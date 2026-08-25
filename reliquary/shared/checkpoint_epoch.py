"""Shared checkpoint-epoch manifest and deterministic derivation.

One verified drand beacon, obtained after an immutable checkpoint is installed,
expands into the generation seed and prompt slice for each window in the next
checkpoint interval. Final auction randomness is deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


CHECKPOINT_EPOCH_SCHEMA_VERSION = 2
CHECKPOINT_EPOCH_CAPABILITY_ID = "checkpoint-epoch-scheduling-v2"
CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT = 16
CHECKPOINT_EPOCH_SCHEDULE_MODE = "concurrent_checkpoint_epoch"
CHECKPOINT_EPOCH_TRAINING_MODES = frozenset({
    "aggregate_one_step",
    "sequential_steps",
})

_ID_DOMAIN = b"reliquary/checkpoint-epoch/id/v1"
_ROOT_DOMAIN = b"reliquary/checkpoint-epoch/root/v1"
_WINDOW_DOMAIN = b"reliquary/checkpoint-epoch/window/v1"
_SLICE_DOMAIN = b"reliquary/checkpoint-epoch/slice/v1"


@dataclass(frozen=True, slots=True)
class ProtocolBinding:
    profile_id: str
    protocol_version: int
    generation_contract_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    number: int
    repo_id: str
    revision: str
    commit_observed_round: int


@dataclass(frozen=True, slots=True)
class BeaconBinding:
    source: str
    chain: str
    chain_hash: str
    round: int
    randomness: str


@dataclass(frozen=True, slots=True)
class WindowSchedule:
    mode: str
    collection_seconds: float
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class PromptSlice:
    environment: str
    universe_size: int
    start: int
    stop: int
    policy: str
    cycle: int


@dataclass(frozen=True, slots=True)
class EpochWindow:
    offset: int
    window_number: int
    generation_randomness: str
    prompt_slices: tuple[PromptSlice, ...]


@dataclass(frozen=True, slots=True)
class EpochPlan:
    schema_version: int
    experimental_capability_id: str
    protocol: ProtocolBinding
    checkpoint: CheckpointBinding
    first_window: int
    window_count: int
    beacon_delay_rounds: int
    epoch_beacon: BeaconBinding
    warmup_rounds: int
    activation_not_before_round: int
    window_schedule: WindowSchedule
    training_mode: str
    prompt_range_size: int
    epoch_id: str
    epoch_seed: str
    windows: tuple[EpochWindow, ...]


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON rejects non-finite floats")
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
    return json.dumps(
        _json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def generation_contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


def _frame_hash(domain: bytes, *parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _protocol_dict(value: ProtocolBinding) -> dict[str, Any]:
    return {
        "profile_id": value.profile_id,
        "protocol_version": value.protocol_version,
        "generation_contract_sha256": value.generation_contract_sha256,
    }


def _checkpoint_dict(value: CheckpointBinding) -> dict[str, Any]:
    return {
        "number": value.number,
        "repo_id": value.repo_id,
        "revision": value.revision,
        "commit_observed_round": value.commit_observed_round,
    }


def _beacon_dict(value: BeaconBinding) -> dict[str, Any]:
    return {
        "source": value.source,
        "chain": value.chain,
        "chain_hash": value.chain_hash,
        "round": value.round,
        "randomness": value.randomness,
    }


def _schedule_dict(value: WindowSchedule) -> dict[str, Any]:
    return {
        "mode": value.mode,
        "collection_seconds": value.collection_seconds,
        "timeout_seconds": value.timeout_seconds,
    }


def _intent_dict(
    *,
    experimental_capability_id: str,
    protocol: ProtocolBinding,
    checkpoint: CheckpointBinding,
    first_window: int,
    window_count: int,
    beacon_delay_rounds: int,
    epoch_beacon: BeaconBinding,
    warmup_rounds: int,
    window_schedule: WindowSchedule,
    training_mode: str,
    prompt_range_size: int,
    environment_universes: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_EPOCH_SCHEMA_VERSION,
        "experimental_capability_id": experimental_capability_id,
        "protocol": _protocol_dict(protocol),
        "checkpoint": _checkpoint_dict(checkpoint),
        "first_window": first_window,
        "window_count": window_count,
        "beacon_delay_rounds": beacon_delay_rounds,
        "epoch_beacon_target": {
            "source": epoch_beacon.source,
            "chain": epoch_beacon.chain,
            "chain_hash": epoch_beacon.chain_hash,
            "round": epoch_beacon.round,
        },
        "warmup_rounds": warmup_rounds,
        "activation_not_before_round": epoch_beacon.round + warmup_rounds,
        "window_schedule": _schedule_dict(window_schedule),
        "training_mode": training_mode,
        "prompt_range_size": prompt_range_size,
        "environment_universes": {
            name: int(environment_universes[name])
            for name in sorted(environment_universes)
        },
    }


def derive_epoch_id(
    *,
    experimental_capability_id: str,
    protocol: ProtocolBinding,
    checkpoint: CheckpointBinding,
    first_window: int,
    window_count: int,
    beacon_delay_rounds: int,
    epoch_beacon: BeaconBinding,
    warmup_rounds: int,
    window_schedule: WindowSchedule,
    training_mode: str,
    prompt_range_size: int,
    environment_universes: Mapping[str, int],
) -> str:
    intent = _intent_dict(
        experimental_capability_id=experimental_capability_id,
        protocol=protocol,
        checkpoint=checkpoint,
        first_window=first_window,
        window_count=window_count,
        beacon_delay_rounds=beacon_delay_rounds,
        epoch_beacon=epoch_beacon,
        warmup_rounds=warmup_rounds,
        window_schedule=window_schedule,
        training_mode=training_mode,
        prompt_range_size=prompt_range_size,
        environment_universes=environment_universes,
    )
    return _frame_hash(_ID_DOMAIN, canonical_json_bytes(intent))


def derive_epoch_seed(*, epoch_id: str, epoch_beacon: BeaconBinding) -> str:
    _require_hex64("epoch_id", epoch_id)
    _validate_beacon(epoch_beacon)
    return _frame_hash(
        _ROOT_DOMAIN,
        bytes.fromhex(epoch_id),
        canonical_json_bytes(_beacon_dict(epoch_beacon)),
    )


def derive_window_seed(
    epoch_seed: str,
    *,
    offset: int,
    window_number: int,
) -> str:
    _require_hex64("epoch_seed", epoch_seed)
    offset = _require_int("offset", offset, minimum=0)
    window_number = _require_int("window_number", window_number, minimum=0)
    return _frame_hash(
        _WINDOW_DOMAIN,
        bytes.fromhex(epoch_seed),
        offset.to_bytes(8, "big"),
        window_number.to_bytes(8, "big"),
    )


def _environment_prompt_slices(
    epoch_seed: str,
    *,
    environment: str,
    universe_size: int,
    prompt_range_size: int,
    window_count: int,
) -> tuple[PromptSlice, ...]:
    _require_text("environment", environment)
    universe_size = _require_int("universe_size", universe_size, minimum=1)
    prompt_range_size = _require_int(
        "prompt_range_size", prompt_range_size, minimum=1
    )
    window_count = _require_int("window_count", window_count, minimum=1)

    width = min(prompt_range_size, universe_size)
    if width == universe_size:
        return tuple(
            PromptSlice(
                environment=environment,
                universe_size=universe_size,
                start=0,
                stop=universe_size,
                policy="full_universe",
                cycle=offset,
            )
            for offset in range(window_count)
        )

    slot_count = max(1, universe_size // width)
    placement = _frame_hash(
        _SLICE_DOMAIN,
        bytes.fromhex(epoch_seed),
        environment.encode("utf-8"),
        universe_size.to_bytes(8, "big"),
        width.to_bytes(8, "big"),
    )
    base_slot = int(placement, 16) % slot_count
    policy = "disjoint" if slot_count >= window_count else "deterministic_cycle"
    slices: list[PromptSlice] = []
    for offset in range(window_count):
        slot = (base_slot + offset) % slot_count
        start = slot * width
        slices.append(
            PromptSlice(
                environment=environment,
                universe_size=universe_size,
                start=start,
                stop=min(start + width, universe_size),
                policy=policy,
                cycle=offset // slot_count,
            )
        )
    return tuple(slices)


def derive_prompt_slices(
    epoch_seed: str,
    *,
    environment_universes: Mapping[str, int],
    prompt_range_size: int,
    window_count: int,
) -> tuple[tuple[PromptSlice, ...], ...]:
    if not environment_universes:
        raise ValueError("environment_universes must not be empty")
    by_environment = {
        environment: _environment_prompt_slices(
            epoch_seed,
            environment=environment,
            universe_size=universe_size,
            prompt_range_size=prompt_range_size,
            window_count=window_count,
        )
        for environment, universe_size in sorted(environment_universes.items())
    }
    return tuple(
        tuple(by_environment[name][offset] for name in sorted(by_environment))
        for offset in range(window_count)
    )


def build_epoch_plan(
    *,
    protocol: ProtocolBinding,
    checkpoint: CheckpointBinding,
    first_window: int,
    window_count: int,
    epoch_beacon: BeaconBinding,
    beacon_delay_rounds: int,
    warmup_rounds: int,
    window_schedule: WindowSchedule,
    training_mode: str,
    prompt_range_size: int,
    environment_universes: Mapping[str, int],
    experimental_capability_id: str = CHECKPOINT_EPOCH_CAPABILITY_ID,
) -> EpochPlan:
    _validate_protocol(protocol)
    _validate_checkpoint(checkpoint)
    _validate_beacon(epoch_beacon)
    _require_text("experimental_capability_id", experimental_capability_id)
    first_window = _require_int("first_window", first_window, minimum=0)
    window_count = _require_int("window_count", window_count, minimum=1)
    beacon_delay_rounds = _require_int(
        "beacon_delay_rounds", beacon_delay_rounds, minimum=1
    )
    if beacon_delay_rounds != 1:
        raise ValueError("checkpoint epoch uses the first post-commit beacon")
    if epoch_beacon.round != checkpoint.commit_observed_round + 1:
        raise ValueError(
            "epoch beacon must be the first round after checkpoint commitment"
        )
    warmup_rounds = _require_int("warmup_rounds", warmup_rounds, minimum=1)
    _validate_window_schedule(window_schedule)
    _validate_training_mode(training_mode)
    prompt_range_size = _require_int(
        "prompt_range_size", prompt_range_size, minimum=1
    )
    universes = _validate_environment_universes(environment_universes)

    epoch_id = derive_epoch_id(
        experimental_capability_id=experimental_capability_id,
        protocol=protocol,
        checkpoint=checkpoint,
        first_window=first_window,
        window_count=window_count,
        beacon_delay_rounds=beacon_delay_rounds,
        epoch_beacon=epoch_beacon,
        warmup_rounds=warmup_rounds,
        window_schedule=window_schedule,
        training_mode=training_mode,
        prompt_range_size=prompt_range_size,
        environment_universes=universes,
    )
    epoch_seed = derive_epoch_seed(epoch_id=epoch_id, epoch_beacon=epoch_beacon)
    seeds = tuple(
        derive_window_seed(
            epoch_seed,
            offset=offset,
            window_number=first_window + offset,
        )
        for offset in range(window_count)
    )
    if len(set(seeds)) != window_count:
        raise ValueError("derived generation seeds are not unique")
    prompt_slices = derive_prompt_slices(
        epoch_seed,
        environment_universes=universes,
        prompt_range_size=prompt_range_size,
        window_count=window_count,
    )
    windows = tuple(
        EpochWindow(
            offset=offset,
            window_number=first_window + offset,
            generation_randomness=seeds[offset],
            prompt_slices=prompt_slices[offset],
        )
        for offset in range(window_count)
    )
    return EpochPlan(
        schema_version=CHECKPOINT_EPOCH_SCHEMA_VERSION,
        experimental_capability_id=experimental_capability_id,
        protocol=protocol,
        checkpoint=checkpoint,
        first_window=first_window,
        window_count=window_count,
        beacon_delay_rounds=beacon_delay_rounds,
        epoch_beacon=epoch_beacon,
        warmup_rounds=warmup_rounds,
        activation_not_before_round=epoch_beacon.round + warmup_rounds,
        window_schedule=window_schedule,
        training_mode=training_mode,
        prompt_range_size=prompt_range_size,
        epoch_id=epoch_id,
        epoch_seed=epoch_seed,
        windows=windows,
    )


def epoch_plan_to_dict(plan: EpochPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "experimental_capability_id": plan.experimental_capability_id,
        "protocol": _protocol_dict(plan.protocol),
        "checkpoint": _checkpoint_dict(plan.checkpoint),
        "first_window": plan.first_window,
        "window_count": plan.window_count,
        "beacon_delay_rounds": plan.beacon_delay_rounds,
        "epoch_beacon": _beacon_dict(plan.epoch_beacon),
        "warmup_rounds": plan.warmup_rounds,
        "activation_not_before_round": plan.activation_not_before_round,
        "window_schedule": _schedule_dict(plan.window_schedule),
        "training_mode": plan.training_mode,
        "prompt_range_size": plan.prompt_range_size,
        "epoch_id": plan.epoch_id,
        "epoch_seed": plan.epoch_seed,
        "windows": [
            {
                "offset": window.offset,
                "window_number": window.window_number,
                "generation_randomness": window.generation_randomness,
                "prompt_slices": [
                    {
                        "environment": prompt_slice.environment,
                        "universe_size": prompt_slice.universe_size,
                        "start": prompt_slice.start,
                        "stop": prompt_slice.stop,
                        "policy": prompt_slice.policy,
                        "cycle": prompt_slice.cycle,
                    }
                    for prompt_slice in window.prompt_slices
                ],
            }
            for window in plan.windows
        ],
    }


def canonical_manifest_bytes(plan: EpochPlan) -> bytes:
    validate_epoch_plan(plan)
    return canonical_json_bytes(epoch_plan_to_dict(plan))


def manifest_sha256(plan: EpochPlan) -> str:
    return hashlib.sha256(canonical_manifest_bytes(plan)).hexdigest()


def parse_epoch_plan(
    raw: bytes | str,
    *,
    expected_manifest_sha256: str | None = None,
) -> EpochPlan:
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if expected_manifest_sha256 is not None:
        _require_hex64("expected_manifest_sha256", expected_manifest_sha256)
        actual = hashlib.sha256(raw_bytes).hexdigest()
        if actual != expected_manifest_sha256:
            raise ValueError("manifest SHA-256 mismatch")
    obj = _strict_json_loads(raw_bytes)
    _exact_keys(
        obj,
        {
            "schema_version",
            "experimental_capability_id",
            "protocol",
            "checkpoint",
            "first_window",
            "window_count",
            "beacon_delay_rounds",
            "epoch_beacon",
            "warmup_rounds",
            "activation_not_before_round",
            "window_schedule",
            "training_mode",
            "prompt_range_size",
            "epoch_id",
            "epoch_seed",
            "windows",
        },
        "manifest",
    )

    protocol_obj = _mapping(obj["protocol"], "protocol")
    _exact_keys(
        protocol_obj,
        {"profile_id", "protocol_version", "generation_contract_sha256"},
        "protocol",
    )
    checkpoint_obj = _mapping(obj["checkpoint"], "checkpoint")
    _exact_keys(
        checkpoint_obj,
        {"number", "repo_id", "revision", "commit_observed_round"},
        "checkpoint",
    )
    beacon_obj = _mapping(obj["epoch_beacon"], "epoch_beacon")
    _exact_keys(
        beacon_obj,
        {"source", "chain", "chain_hash", "round", "randomness"},
        "epoch_beacon",
    )
    schedule_obj = _mapping(obj["window_schedule"], "window_schedule")
    _exact_keys(
        schedule_obj,
        {"mode", "collection_seconds", "timeout_seconds"},
        "window_schedule",
    )

    windows_obj = _list(obj["windows"], "windows")
    windows: list[EpochWindow] = []
    universes: dict[str, int] = {}
    for index, item in enumerate(windows_obj):
        window_obj = _mapping(item, f"windows[{index}]")
        _exact_keys(
            window_obj,
            {
                "offset",
                "window_number",
                "generation_randomness",
                "prompt_slices",
            },
            f"windows[{index}]",
        )
        slices: list[PromptSlice] = []
        for slice_index, slice_item in enumerate(
            _list(window_obj["prompt_slices"], f"windows[{index}].prompt_slices")
        ):
            slice_obj = _mapping(
                slice_item, f"windows[{index}].prompt_slices[{slice_index}]"
            )
            _exact_keys(
                slice_obj,
                {
                    "environment",
                    "universe_size",
                    "start",
                    "stop",
                    "policy",
                    "cycle",
                },
                f"windows[{index}].prompt_slices[{slice_index}]",
            )
            prompt_slice = PromptSlice(
                environment=slice_obj["environment"],
                universe_size=slice_obj["universe_size"],
                start=slice_obj["start"],
                stop=slice_obj["stop"],
                policy=slice_obj["policy"],
                cycle=slice_obj["cycle"],
            )
            slices.append(prompt_slice)
            if prompt_slice.environment in universes and (
                universes[prompt_slice.environment] != prompt_slice.universe_size
            ):
                raise ValueError("environment universe changes within manifest")
            universes[prompt_slice.environment] = prompt_slice.universe_size
        windows.append(
            EpochWindow(
                offset=window_obj["offset"],
                window_number=window_obj["window_number"],
                generation_randomness=window_obj["generation_randomness"],
                prompt_slices=tuple(slices),
            )
        )

    plan = EpochPlan(
        schema_version=obj["schema_version"],
        experimental_capability_id=obj["experimental_capability_id"],
        protocol=ProtocolBinding(**protocol_obj),
        checkpoint=CheckpointBinding(**checkpoint_obj),
        first_window=obj["first_window"],
        window_count=obj["window_count"],
        beacon_delay_rounds=obj["beacon_delay_rounds"],
        epoch_beacon=BeaconBinding(**beacon_obj),
        warmup_rounds=obj["warmup_rounds"],
        activation_not_before_round=obj["activation_not_before_round"],
        window_schedule=WindowSchedule(**schedule_obj),
        training_mode=obj["training_mode"],
        prompt_range_size=obj["prompt_range_size"],
        epoch_id=obj["epoch_id"],
        epoch_seed=obj["epoch_seed"],
        windows=tuple(windows),
    )
    validate_epoch_plan(plan, environment_universes=universes)
    if raw_bytes != canonical_json_bytes(epoch_plan_to_dict(plan)):
        raise ValueError("manifest is not canonical JSON")
    return plan


def validate_epoch_plan(
    plan: EpochPlan,
    *,
    environment_universes: Mapping[str, int] | None = None,
) -> None:
    if not isinstance(plan, EpochPlan):
        raise TypeError("plan must be an EpochPlan")
    if plan.schema_version != CHECKPOINT_EPOCH_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint epoch schema")
    if environment_universes is None:
        environment_universes = {}
        for window in plan.windows:
            for prompt_slice in window.prompt_slices:
                previous = environment_universes.get(prompt_slice.environment)
                if previous is not None and previous != prompt_slice.universe_size:
                    raise ValueError("environment universe changes within manifest")
                environment_universes[prompt_slice.environment] = (
                    prompt_slice.universe_size
                )
    expected = build_epoch_plan(
        protocol=plan.protocol,
        checkpoint=plan.checkpoint,
        first_window=plan.first_window,
        window_count=plan.window_count,
        epoch_beacon=plan.epoch_beacon,
        beacon_delay_rounds=plan.beacon_delay_rounds,
        warmup_rounds=plan.warmup_rounds,
        window_schedule=plan.window_schedule,
        training_mode=plan.training_mode,
        prompt_range_size=plan.prompt_range_size,
        environment_universes=environment_universes,
        experimental_capability_id=plan.experimental_capability_id,
    )
    if plan != expected:
        raise ValueError("checkpoint epoch manifest does not match derivation")


def _strict_json_loads(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be UTF-8") from exc

    def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid manifest JSON") from exc
    return dict(_mapping(value, "manifest"))


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{path} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_number(name: str, value: Any, *, minimum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or float(value) < minimum
    ):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def _require_hex64(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _validate_protocol(value: ProtocolBinding) -> None:
    if not isinstance(value, ProtocolBinding):
        raise ValueError("protocol must be a ProtocolBinding")
    _require_text("protocol.profile_id", value.profile_id)
    _require_int("protocol.protocol_version", value.protocol_version, minimum=1)
    _require_hex64(
        "protocol.generation_contract_sha256",
        value.generation_contract_sha256,
    )


def _validate_checkpoint(value: CheckpointBinding) -> None:
    if not isinstance(value, CheckpointBinding):
        raise ValueError("checkpoint must be a CheckpointBinding")
    _require_int("checkpoint.number", value.number, minimum=0)
    _require_text("checkpoint.repo_id", value.repo_id)
    _require_text("checkpoint.revision", value.revision)
    _require_int(
        "checkpoint.commit_observed_round",
        value.commit_observed_round,
        minimum=1,
    )


def _validate_beacon(value: BeaconBinding) -> None:
    if not isinstance(value, BeaconBinding):
        raise ValueError("epoch_beacon must be a BeaconBinding")
    if value.source != "drand":
        raise ValueError("epoch_beacon.source must be 'drand'")
    _require_text("epoch_beacon.chain", value.chain)
    _require_hex64("epoch_beacon.chain_hash", value.chain_hash)
    _require_int("epoch_beacon.round", value.round, minimum=1)
    _require_hex64("epoch_beacon.randomness", value.randomness)


def _validate_window_schedule(value: WindowSchedule) -> None:
    if not isinstance(value, WindowSchedule):
        raise ValueError("window_schedule must be a WindowSchedule")
    if value.mode != CHECKPOINT_EPOCH_SCHEDULE_MODE:
        raise ValueError("unsupported checkpoint epoch window schedule")
    collection = _require_number(
        "window_schedule.collection_seconds",
        value.collection_seconds,
        minimum=0.001,
    )
    timeout = _require_number(
        "window_schedule.timeout_seconds",
        value.timeout_seconds,
        minimum=collection,
    )
    if timeout < collection:
        raise ValueError("window timeout must not precede collection close")


def _validate_training_mode(value: str) -> None:
    if value not in CHECKPOINT_EPOCH_TRAINING_MODES:
        raise ValueError("unsupported checkpoint epoch training mode")


def _validate_environment_universes(
    value: Mapping[str, int],
) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("environment_universes must not be empty")
    result: dict[str, int] = {}
    for environment, size in value.items():
        _require_text("environment", environment)
        result[environment] = _require_int(
            f"environment_universes[{environment!r}]",
            size,
            minimum=1,
        )
    return result


__all__ = [
    "BeaconBinding",
    "CHECKPOINT_EPOCH_CAPABILITY_ID",
    "CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT",
    "CHECKPOINT_EPOCH_SCHEDULE_MODE",
    "CHECKPOINT_EPOCH_SCHEMA_VERSION",
    "CHECKPOINT_EPOCH_TRAINING_MODES",
    "CheckpointBinding",
    "EpochPlan",
    "EpochWindow",
    "PromptSlice",
    "ProtocolBinding",
    "WindowSchedule",
    "build_epoch_plan",
    "canonical_json_bytes",
    "canonical_manifest_bytes",
    "derive_epoch_id",
    "derive_epoch_seed",
    "derive_prompt_slices",
    "derive_window_seed",
    "epoch_plan_to_dict",
    "generation_contract_sha256",
    "manifest_sha256",
    "parse_epoch_plan",
    "validate_epoch_plan",
]
