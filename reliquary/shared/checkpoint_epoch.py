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


CHECKPOINT_EPOCH_SCHEMA_VERSION = 8
CHECKPOINT_EPOCH_CAPABILITY_ID = "checkpoint-epoch-scheduling-v8"
CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT = 16
CHECKPOINT_EPOCH_SCHEDULE_MODE = "concurrent_checkpoint_epoch"
CHECKPOINT_EPOCH_ADMISSION_POLICY = "self_selected_intent_operator_rounds_v1"
CHECKPOINT_EPOCH_VALUATION_POLICY = "balanced_advantage_portfolio_v1"
CHECKPOINT_EPOCH_RANKING_POLICY = (
    "strata_operator_rounds_post_seal_beacon_v1"
)
CHECKPOINT_EPOCH_REWARD_POLICY = "selected_slot_v1"
CHECKPOINT_EPOCH_FINALIZATION_POLICY = (
    "streamed_validation_ordered_lanes_atomic_epoch_v1"
)
CHECKPOINT_EPOCH_COMMITMENT_SET_SCHEMA_VERSION = 1
CHECKPOINT_EPOCH_TRAINING_MODES = frozenset(
    {
        "aggregate_one_step",
        "sequential_steps",
    }
)

_ID_DOMAIN = b"reliquary/checkpoint-epoch/id/v1"
_ROOT_DOMAIN = b"reliquary/checkpoint-epoch/root/v1"
_WINDOW_DOMAIN = b"reliquary/checkpoint-epoch/window/v1"
_SLICE_DOMAIN = b"reliquary/checkpoint-epoch/slice/v1"
_ADMISSION_OPERATOR_DOMAIN = b"reliquary/checkpoint-epoch/admission-operator/v1"
_ADMISSION_COMMITMENT_DOMAIN = b"reliquary/checkpoint-epoch/admission-commitment/v1"
_COMMITMENT_SET_ROOT_DOMAIN = b"reliquary/checkpoint-epoch/commitment-set-root/v1"
_COMMITMENT_SET_SIGNING_DOMAIN = b"reliquary/checkpoint-epoch/commitment-set-signing/v1"


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
    target_groups_per_environment_lane: int
    candidate_limit_per_environment_lane: int
    admission_policy: str
    valuation_policy: str
    ranking_policy: str
    reward_policy: str
    finalization_policy: str
    commitments_per_operator_per_environment_lane: int
    intent_seconds: float
    backup_activation_fractions: tuple[float, ...]
    reveal_seconds: float
    epoch_id: str
    epoch_seed: str
    windows: tuple[EpochWindow, ...]


@dataclass(frozen=True, slots=True)
class EpochAdmissionCommitment:
    """Public, compact input to post-commit reveal selection."""

    commitment_id: str
    operator_id: str
    window_number: int
    environment: str
    prompt_idx: int
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class EpochCommitmentRecord:
    """One accepted compact commitment in the public frozen set."""

    receipt_id: str
    commitment_id: str
    operator_id: str
    miner_hotkey: str
    window_number: int
    environment: str
    prompt_idx: int
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class EpochCommitmentSet:
    """Canonical set frozen and persisted before the admission beacon."""

    schema_version: int
    epoch_id: str
    manifest_sha256: str
    commitment_close_round: int
    validator_hotkey: str
    commitment_root: str
    commitments: tuple[EpochCommitmentRecord, ...]


@dataclass(frozen=True, slots=True)
class SignedEpochCommitmentSet:
    """Public validator attestation over one immutable commitment set."""

    commitment_set: EpochCommitmentSet
    commitment_set_sha256: str
    validator_signature: str


def select_epoch_reveals(
    commitments: Sequence[EpochAdmissionCommitment],
    *,
    admission_randomness: str,
    epoch_id: str,
    manifest_sha256_hex: str,
    commitment_set_sha256_hex: str,
    limit: int,
    per_prompt_limit: int,
) -> tuple[str, ...]:
    """Select a bounded reveal cohort without using commitment arrival time.

    Operators are shuffled from the post-commit beacon, then visited in rounds.
    Each round can contribute at most one commitment per operator. Commitments
    within an operator are independently shuffled from the same beacon.
    """
    _require_hex64("admission_randomness", admission_randomness)
    _require_hex64("epoch_id", epoch_id)
    _require_hex64("manifest_sha256_hex", manifest_sha256_hex)
    _require_hex64("commitment_set_sha256_hex", commitment_set_sha256_hex)
    limit = _require_int("limit", limit, minimum=1)
    per_prompt_limit = _require_int("per_prompt_limit", per_prompt_limit, minimum=1)
    by_operator: dict[str, list[EpochAdmissionCommitment]] = {}
    seen: set[str] = set()
    lane: tuple[int, str] | None = None
    for commitment in commitments:
        if not isinstance(commitment, EpochAdmissionCommitment):
            raise TypeError("commitments must contain EpochAdmissionCommitment")
        _require_text("commitment_id", commitment.commitment_id)
        _require_text("operator_id", commitment.operator_id)
        _require_int("window_number", commitment.window_number, minimum=0)
        _require_text("environment", commitment.environment)
        _require_int("prompt_idx", commitment.prompt_idx, minimum=0)
        _require_hex64("payload_sha256", commitment.payload_sha256)
        if commitment.commitment_id in seen:
            raise ValueError("duplicate epoch admission commitment")
        commitment_lane = (commitment.window_number, commitment.environment)
        if lane is None:
            lane = commitment_lane
        elif commitment_lane != lane:
            raise ValueError("epoch admission commitments must share one lane")
        seen.add(commitment.commitment_id)
        by_operator.setdefault(commitment.operator_id, []).append(commitment)

    beacon = bytes.fromhex(admission_randomness)
    epoch = bytes.fromhex(epoch_id)
    manifest = bytes.fromhex(manifest_sha256_hex)
    commitment_set_digest = bytes.fromhex(commitment_set_sha256_hex)

    def operator_key(operator_id: str) -> str:
        if lane is None:
            raise AssertionError("operator key requires a lane")
        return _frame_hash(
            _ADMISSION_OPERATOR_DOMAIN,
            beacon,
            epoch,
            manifest,
            commitment_set_digest,
            canonical_json_bytes(
                {
                    "environment": lane[1],
                    "window_number": lane[0],
                }
            ),
            operator_id.encode("utf-8"),
        )

    def commitment_key(commitment: EpochAdmissionCommitment) -> str:
        return _frame_hash(
            _ADMISSION_COMMITMENT_DOMAIN,
            beacon,
            epoch,
            manifest,
            commitment_set_digest,
            canonical_json_bytes(
                {
                    "commitment_id": commitment.commitment_id,
                    "environment": commitment.environment,
                    "operator_id": commitment.operator_id,
                    "payload_sha256": commitment.payload_sha256,
                    "prompt_idx": commitment.prompt_idx,
                    "window_number": commitment.window_number,
                }
            ),
        )

    operators = sorted(by_operator, key=lambda value: (operator_key(value), value))
    queues = {
        operator: sorted(
            by_operator[operator],
            key=lambda value: (commitment_key(value), value.commitment_id),
        )
        for operator in operators
    }
    selected: list[str] = []
    prompt_counts: dict[tuple[int, str, int], int] = {}
    selected_operator_prompts: set[tuple[str, int]] = set()
    round_index = 0
    while len(selected) < limit:
        added = False
        for operator in operators:
            queue = queues[operator]
            while round_index < len(queue):
                candidate = queue[round_index]
                prompt_key = (
                    candidate.window_number,
                    candidate.environment,
                    candidate.prompt_idx,
                )
                operator_prompt = (
                    candidate.operator_id,
                    candidate.prompt_idx,
                )
                if (
                    prompt_counts.get(prompt_key, 0) >= per_prompt_limit
                    or operator_prompt in selected_operator_prompts
                ):
                    queue.pop(round_index)
                    continue
                selected.append(candidate.commitment_id)
                prompt_counts[prompt_key] = prompt_counts.get(prompt_key, 0) + 1
                selected_operator_prompts.add(operator_prompt)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break
        round_index += 1
    return tuple(selected)


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


def _commitment_record_dict(
    value: EpochCommitmentRecord,
) -> dict[str, Any]:
    return {
        "receipt_id": value.receipt_id,
        "commitment_id": value.commitment_id,
        "operator_id": value.operator_id,
        "miner_hotkey": value.miner_hotkey,
        "window_number": value.window_number,
        "environment": value.environment,
        "prompt_idx": value.prompt_idx,
        "payload_sha256": value.payload_sha256,
    }


def _validate_commitment_record(value: EpochCommitmentRecord) -> None:
    if not isinstance(value, EpochCommitmentRecord):
        raise TypeError("commitments must contain EpochCommitmentRecord")
    _require_text("receipt_id", value.receipt_id)
    _require_text("commitment_id", value.commitment_id)
    _require_text("operator_id", value.operator_id)
    _require_text("miner_hotkey", value.miner_hotkey)
    _require_int("window_number", value.window_number, minimum=0)
    _require_text("environment", value.environment)
    _require_int("prompt_idx", value.prompt_idx, minimum=0)
    _require_hex64("payload_sha256", value.payload_sha256)


def _canonical_commitment_records(
    commitments: Sequence[EpochCommitmentRecord],
) -> tuple[EpochCommitmentRecord, ...]:
    records = tuple(commitments)
    for record in records:
        _validate_commitment_record(record)
    ordered = tuple(
        sorted(
            records,
            key=lambda record: canonical_json_bytes(_commitment_record_dict(record)),
        )
    )
    if records != ordered:
        raise ValueError("checkpoint epoch commitments are not canonical")
    receipt_ids = {record.receipt_id for record in records}
    commitment_ids = {record.commitment_id for record in records}
    if len(receipt_ids) != len(records):
        raise ValueError("duplicate checkpoint epoch receipt")
    if len(commitment_ids) != len(records):
        raise ValueError("duplicate checkpoint epoch commitment")
    return records


def _commitment_root(
    commitments: Sequence[EpochCommitmentRecord],
) -> str:
    records = [_commitment_record_dict(item) for item in commitments]
    return _frame_hash(
        _COMMITMENT_SET_ROOT_DOMAIN,
        canonical_json_bytes(records),
    )


def commitment_set_to_dict(value: EpochCommitmentSet) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "epoch_id": value.epoch_id,
        "manifest_sha256": value.manifest_sha256,
        "commitment_close_round": value.commitment_close_round,
        "validator_hotkey": value.validator_hotkey,
        "commitment_root": value.commitment_root,
        "commitments": [
            _commitment_record_dict(record) for record in value.commitments
        ],
    }


def validate_commitment_set(value: EpochCommitmentSet) -> None:
    if not isinstance(value, EpochCommitmentSet):
        raise TypeError("commitment_set must be an EpochCommitmentSet")
    if value.schema_version != CHECKPOINT_EPOCH_COMMITMENT_SET_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint epoch commitment-set schema")
    _require_hex64("epoch_id", value.epoch_id)
    _require_hex64("manifest_sha256", value.manifest_sha256)
    _require_int("commitment_close_round", value.commitment_close_round, minimum=1)
    _require_text("validator_hotkey", value.validator_hotkey)
    _require_hex64("commitment_root", value.commitment_root)
    records = _canonical_commitment_records(value.commitments)
    if value.commitment_root != _commitment_root(records):
        raise ValueError("checkpoint epoch commitment root differs")


def build_commitment_set(
    commitments: Sequence[EpochCommitmentRecord],
    *,
    epoch_id: str,
    manifest_sha256_hex: str,
    commitment_close_round: int,
    validator_hotkey: str,
) -> EpochCommitmentSet:
    _require_hex64("epoch_id", epoch_id)
    _require_hex64("manifest_sha256_hex", manifest_sha256_hex)
    _require_int("commitment_close_round", commitment_close_round, minimum=1)
    _require_text("validator_hotkey", validator_hotkey)
    supplied = tuple(commitments)
    for record in supplied:
        _validate_commitment_record(record)
    records = tuple(
        sorted(
            supplied,
            key=lambda record: canonical_json_bytes(_commitment_record_dict(record)),
        )
    )
    value = EpochCommitmentSet(
        schema_version=CHECKPOINT_EPOCH_COMMITMENT_SET_SCHEMA_VERSION,
        epoch_id=epoch_id,
        manifest_sha256=manifest_sha256_hex,
        commitment_close_round=int(commitment_close_round),
        validator_hotkey=validator_hotkey,
        commitment_root=_commitment_root(records),
        commitments=records,
    )
    validate_commitment_set(value)
    return value


def canonical_commitment_set_bytes(value: EpochCommitmentSet) -> bytes:
    validate_commitment_set(value)
    return canonical_json_bytes(commitment_set_to_dict(value))


def commitment_set_sha256(value: EpochCommitmentSet) -> str:
    return hashlib.sha256(canonical_commitment_set_bytes(value)).hexdigest()


def commitment_set_signing_bytes(value: EpochCommitmentSet) -> bytes:
    """Return the fixed-size message signed by the validator hotkey."""
    return bytes.fromhex(
        _frame_hash(
            _COMMITMENT_SET_SIGNING_DOMAIN,
            canonical_commitment_set_bytes(value),
        )
    )


def signed_commitment_set_to_dict(
    value: SignedEpochCommitmentSet,
) -> dict[str, Any]:
    return {
        "commitment_set": commitment_set_to_dict(value.commitment_set),
        "commitment_set_sha256": value.commitment_set_sha256,
        "validator_signature": value.validator_signature,
    }


def validate_signed_commitment_set(
    value: SignedEpochCommitmentSet,
) -> None:
    if not isinstance(value, SignedEpochCommitmentSet):
        raise TypeError("publication must be a SignedEpochCommitmentSet")
    validate_commitment_set(value.commitment_set)
    _require_hex64("commitment_set_sha256", value.commitment_set_sha256)
    if value.commitment_set_sha256 != commitment_set_sha256(value.commitment_set):
        raise ValueError("checkpoint epoch commitment-set SHA-256 differs")
    signature = value.validator_signature
    if (
        not isinstance(signature, str)
        or not signature
        or len(signature) > 256
        or len(signature) % 2
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("validator signature must be lowercase hexadecimal")


def validate_commitment_set_for_plan(
    value: EpochCommitmentSet,
    plan: EpochPlan,
) -> None:
    """Validate every frozen record against its immutable epoch lane."""
    validate_commitment_set(value)
    validate_epoch_plan(plan)
    if (
        value.epoch_id != plan.epoch_id
        or value.manifest_sha256 != manifest_sha256(plan)
    ):
        raise ValueError("checkpoint epoch commitment set differs from plan")

    windows = {window.window_number: window for window in plan.windows}
    operator_lane_counts: dict[tuple[str, int, str], int] = {}
    for record in value.commitments:
        window = windows.get(record.window_number)
        if window is None:
            raise ValueError("checkpoint epoch commitment window is outside plan")
        slices = {
            prompt_slice.environment: prompt_slice
            for prompt_slice in window.prompt_slices
        }
        prompt_slice = slices.get(record.environment)
        if prompt_slice is None or not (
            prompt_slice.start <= record.prompt_idx < prompt_slice.stop
        ):
            raise ValueError("checkpoint epoch commitment prompt is outside plan")
        count_key = (
            record.operator_id,
            record.window_number,
            record.environment,
        )
        operator_lane_counts[count_key] = operator_lane_counts.get(count_key, 0) + 1
        if operator_lane_counts[count_key] > (
            plan.commitments_per_operator_per_environment_lane
        ):
            raise ValueError("checkpoint epoch operator commitment bound exceeded")


def canonical_signed_commitment_set_bytes(
    value: SignedEpochCommitmentSet,
) -> bytes:
    validate_signed_commitment_set(value)
    return canonical_json_bytes(signed_commitment_set_to_dict(value))


def parse_signed_commitment_set(
    raw: bytes | str,
) -> SignedEpochCommitmentSet:
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    obj = _strict_json_loads(raw_bytes)
    _exact_keys(
        obj,
        {"commitment_set", "commitment_set_sha256", "validator_signature"},
        "signed commitment set",
    )
    set_obj = _mapping(obj["commitment_set"], "commitment_set")
    _exact_keys(
        set_obj,
        {
            "schema_version",
            "epoch_id",
            "manifest_sha256",
            "commitment_close_round",
            "validator_hotkey",
            "commitment_root",
            "commitments",
        },
        "commitment_set",
    )
    records: list[EpochCommitmentRecord] = []
    for index, item in enumerate(
        _list(set_obj["commitments"], "commitment_set.commitments")
    ):
        record_obj = _mapping(item, f"commitment_set.commitments[{index}]")
        _exact_keys(
            record_obj,
            {
                "receipt_id",
                "commitment_id",
                "operator_id",
                "miner_hotkey",
                "window_number",
                "environment",
                "prompt_idx",
                "payload_sha256",
            },
            f"commitment_set.commitments[{index}]",
        )
        records.append(EpochCommitmentRecord(**record_obj))
    commitment_set = EpochCommitmentSet(
        schema_version=set_obj["schema_version"],
        epoch_id=set_obj["epoch_id"],
        manifest_sha256=set_obj["manifest_sha256"],
        commitment_close_round=set_obj["commitment_close_round"],
        validator_hotkey=set_obj["validator_hotkey"],
        commitment_root=set_obj["commitment_root"],
        commitments=tuple(records),
    )
    publication = SignedEpochCommitmentSet(
        commitment_set=commitment_set,
        commitment_set_sha256=obj["commitment_set_sha256"],
        validator_signature=obj["validator_signature"],
    )
    validate_signed_commitment_set(publication)
    if raw_bytes != canonical_signed_commitment_set_bytes(publication):
        raise ValueError("signed commitment set is not canonical JSON")
    return publication


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
    target_groups_per_environment_lane: int,
    candidate_limit_per_environment_lane: int,
    admission_policy: str,
    valuation_policy: str,
    ranking_policy: str,
    reward_policy: str,
    finalization_policy: str,
    commitments_per_operator_per_environment_lane: int,
    intent_seconds: float,
    backup_activation_fractions: Sequence[float],
    reveal_seconds: float,
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
        "target_groups_per_environment_lane": (
            target_groups_per_environment_lane
        ),
        "candidate_limit_per_environment_lane": (
            candidate_limit_per_environment_lane
        ),
        "admission_policy": admission_policy,
        "valuation_policy": valuation_policy,
        "ranking_policy": ranking_policy,
        "reward_policy": reward_policy,
        "finalization_policy": finalization_policy,
        "commitments_per_operator_per_environment_lane": (
            commitments_per_operator_per_environment_lane
        ),
        "intent_seconds": intent_seconds,
        "backup_activation_fractions": list(backup_activation_fractions),
        "reveal_seconds": reveal_seconds,
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
    target_groups_per_environment_lane: int,
    candidate_limit_per_environment_lane: int,
    admission_policy: str = CHECKPOINT_EPOCH_ADMISSION_POLICY,
    valuation_policy: str = CHECKPOINT_EPOCH_VALUATION_POLICY,
    ranking_policy: str = CHECKPOINT_EPOCH_RANKING_POLICY,
    reward_policy: str = CHECKPOINT_EPOCH_REWARD_POLICY,
    finalization_policy: str = CHECKPOINT_EPOCH_FINALIZATION_POLICY,
    commitments_per_operator_per_environment_lane: int = 16,
    intent_seconds: float = 60.0,
    backup_activation_fractions: Sequence[float] = (0.5, 0.75),
    reveal_seconds: float = 60.0,
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
        target_groups_per_environment_lane=(
            target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=(
            candidate_limit_per_environment_lane
        ),
        admission_policy=admission_policy,
        valuation_policy=valuation_policy,
        ranking_policy=ranking_policy,
        reward_policy=reward_policy,
        finalization_policy=finalization_policy,
        commitments_per_operator_per_environment_lane=(
            commitments_per_operator_per_environment_lane
        ),
        intent_seconds=intent_seconds,
        backup_activation_fractions=backup_activation_fractions,
        reveal_seconds=reveal_seconds,
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
    target_groups_per_environment_lane: int,
    candidate_limit_per_environment_lane: int,
    environment_universes: Mapping[str, int],
    admission_policy: str = CHECKPOINT_EPOCH_ADMISSION_POLICY,
    valuation_policy: str = CHECKPOINT_EPOCH_VALUATION_POLICY,
    ranking_policy: str = CHECKPOINT_EPOCH_RANKING_POLICY,
    reward_policy: str = CHECKPOINT_EPOCH_REWARD_POLICY,
    finalization_policy: str = CHECKPOINT_EPOCH_FINALIZATION_POLICY,
    commitments_per_operator_per_environment_lane: int = 16,
    intent_seconds: float = 60.0,
    backup_activation_fractions: Sequence[float] = (0.5, 0.75),
    reveal_seconds: float = 60.0,
    experimental_capability_id: str = CHECKPOINT_EPOCH_CAPABILITY_ID,
) -> EpochPlan:
    _validate_protocol(protocol)
    validate_checkpoint_binding(checkpoint)
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
    target_groups_per_environment_lane = _require_int(
        "target_groups_per_environment_lane",
        target_groups_per_environment_lane,
        minimum=1,
    )
    candidate_limit_per_environment_lane = _require_int(
        "candidate_limit_per_environment_lane",
        candidate_limit_per_environment_lane,
        minimum=target_groups_per_environment_lane,
    )
    if admission_policy != CHECKPOINT_EPOCH_ADMISSION_POLICY:
        raise ValueError("unsupported checkpoint epoch admission policy")
    if valuation_policy != CHECKPOINT_EPOCH_VALUATION_POLICY:
        raise ValueError("unsupported checkpoint epoch valuation policy")
    if ranking_policy != CHECKPOINT_EPOCH_RANKING_POLICY:
        raise ValueError("unsupported checkpoint epoch ranking policy")
    if reward_policy != CHECKPOINT_EPOCH_REWARD_POLICY:
        raise ValueError("unsupported checkpoint epoch reward policy")
    if finalization_policy != CHECKPOINT_EPOCH_FINALIZATION_POLICY:
        raise ValueError("unsupported checkpoint epoch finalization policy")
    commitments_per_operator_per_environment_lane = _require_int(
        "commitments_per_operator_per_environment_lane",
        commitments_per_operator_per_environment_lane,
        minimum=1,
    )
    intent_seconds = _require_number(
        "intent_seconds", intent_seconds, minimum=0.001
    )
    backup_activation_fractions = _validate_backup_activation_fractions(
        backup_activation_fractions
    )
    reveal_seconds = _require_number(
        "reveal_seconds", reveal_seconds, minimum=0.001
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
        target_groups_per_environment_lane=(
            target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=(
            candidate_limit_per_environment_lane
        ),
        admission_policy=admission_policy,
        valuation_policy=valuation_policy,
        ranking_policy=ranking_policy,
        reward_policy=reward_policy,
        finalization_policy=finalization_policy,
        commitments_per_operator_per_environment_lane=(
            commitments_per_operator_per_environment_lane
        ),
        intent_seconds=intent_seconds,
        backup_activation_fractions=backup_activation_fractions,
        reveal_seconds=reveal_seconds,
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
        target_groups_per_environment_lane=(
            target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=(
            candidate_limit_per_environment_lane
        ),
        admission_policy=admission_policy,
        valuation_policy=valuation_policy,
        ranking_policy=ranking_policy,
        reward_policy=reward_policy,
        finalization_policy=finalization_policy,
        commitments_per_operator_per_environment_lane=(
            commitments_per_operator_per_environment_lane
        ),
        intent_seconds=intent_seconds,
        backup_activation_fractions=backup_activation_fractions,
        reveal_seconds=reveal_seconds,
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
        "target_groups_per_environment_lane": (
            plan.target_groups_per_environment_lane
        ),
        "candidate_limit_per_environment_lane": (
            plan.candidate_limit_per_environment_lane
        ),
        "admission_policy": plan.admission_policy,
        "valuation_policy": plan.valuation_policy,
        "ranking_policy": plan.ranking_policy,
        "reward_policy": plan.reward_policy,
        "finalization_policy": plan.finalization_policy,
        "commitments_per_operator_per_environment_lane": (
            plan.commitments_per_operator_per_environment_lane
        ),
        "intent_seconds": plan.intent_seconds,
        "backup_activation_fractions": list(
            plan.backup_activation_fractions
        ),
        "reveal_seconds": plan.reveal_seconds,
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
            "target_groups_per_environment_lane",
            "candidate_limit_per_environment_lane",
            "admission_policy",
            "valuation_policy",
            "ranking_policy",
            "reward_policy",
            "finalization_policy",
            "commitments_per_operator_per_environment_lane",
            "intent_seconds",
            "backup_activation_fractions",
            "reveal_seconds",
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
        target_groups_per_environment_lane=(
            obj["target_groups_per_environment_lane"]
        ),
        candidate_limit_per_environment_lane=(
            obj["candidate_limit_per_environment_lane"]
        ),
        admission_policy=obj["admission_policy"],
        valuation_policy=obj["valuation_policy"],
        ranking_policy=obj["ranking_policy"],
        reward_policy=obj["reward_policy"],
        finalization_policy=obj["finalization_policy"],
        commitments_per_operator_per_environment_lane=(
            obj["commitments_per_operator_per_environment_lane"]
        ),
        intent_seconds=obj["intent_seconds"],
        backup_activation_fractions=tuple(
            _list(
                obj["backup_activation_fractions"],
                "backup_activation_fractions",
            )
        ),
        reveal_seconds=obj["reveal_seconds"],
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
        target_groups_per_environment_lane=(
            plan.target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=(
            plan.candidate_limit_per_environment_lane
        ),
        admission_policy=plan.admission_policy,
        valuation_policy=plan.valuation_policy,
        ranking_policy=plan.ranking_policy,
        reward_policy=plan.reward_policy,
        finalization_policy=plan.finalization_policy,
        commitments_per_operator_per_environment_lane=(
            plan.commitments_per_operator_per_environment_lane
        ),
        intent_seconds=plan.intent_seconds,
        backup_activation_fractions=plan.backup_activation_fractions,
        reveal_seconds=plan.reveal_seconds,
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


def _validate_backup_activation_fractions(
    value: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("backup_activation_fractions must be a sequence")
    result = tuple(
        _require_number(
            f"backup_activation_fractions[{index}]",
            item,
            minimum=0.001,
        )
        for index, item in enumerate(value)
    )
    if any(item >= 1.0 for item in result):
        raise ValueError("backup activation fractions must be below one")
    if tuple(sorted(set(result))) != result:
        raise ValueError(
            "backup activation fractions must be unique and increasing"
        )
    return result


def _require_hex64(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _require_hex40(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a 40-character immutable commit OID")
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


def validate_checkpoint_binding(value: CheckpointBinding) -> None:
    if not isinstance(value, CheckpointBinding):
        raise ValueError("checkpoint must be a CheckpointBinding")
    _require_int("checkpoint.number", value.number, minimum=0)
    _require_text("checkpoint.repo_id", value.repo_id)
    _require_hex40("checkpoint.revision", value.revision)
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
    "CHECKPOINT_EPOCH_ADMISSION_POLICY",
    "CHECKPOINT_EPOCH_COMMITMENT_SET_SCHEMA_VERSION",
    "CHECKPOINT_EPOCH_FINALIZATION_POLICY",
    "CHECKPOINT_EPOCH_VALUATION_POLICY",
    "CHECKPOINT_EPOCH_RANKING_POLICY",
    "CHECKPOINT_EPOCH_REQUIRED_WINDOW_COUNT",
    "CHECKPOINT_EPOCH_REWARD_POLICY",
    "CHECKPOINT_EPOCH_SCHEDULE_MODE",
    "CHECKPOINT_EPOCH_SCHEMA_VERSION",
    "CHECKPOINT_EPOCH_TRAINING_MODES",
    "CheckpointBinding",
    "EpochAdmissionCommitment",
    "EpochCommitmentRecord",
    "EpochCommitmentSet",
    "EpochPlan",
    "EpochWindow",
    "PromptSlice",
    "ProtocolBinding",
    "SignedEpochCommitmentSet",
    "WindowSchedule",
    "build_epoch_plan",
    "build_commitment_set",
    "canonical_commitment_set_bytes",
    "canonical_json_bytes",
    "canonical_manifest_bytes",
    "canonical_signed_commitment_set_bytes",
    "commitment_set_sha256",
    "commitment_set_signing_bytes",
    "commitment_set_to_dict",
    "derive_epoch_id",
    "derive_epoch_seed",
    "derive_prompt_slices",
    "derive_window_seed",
    "epoch_plan_to_dict",
    "generation_contract_sha256",
    "manifest_sha256",
    "parse_epoch_plan",
    "parse_signed_commitment_set",
    "select_epoch_reveals",
    "signed_commitment_set_to_dict",
    "validate_commitment_set",
    "validate_commitment_set_for_plan",
    "validate_epoch_plan",
    "validate_signed_commitment_set",
]
