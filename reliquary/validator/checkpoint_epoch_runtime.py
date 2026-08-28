"""Small durable runtime for checkpoint-epoch plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CHECKPOINT_EPOCH_ADMISSION_POLICY,
    CHECKPOINT_EPOCH_CAPABILITY_ID,
    CHECKPOINT_EPOCH_FINALIZATION_POLICY,
    CHECKPOINT_EPOCH_RANKING_POLICY,
    CHECKPOINT_EPOCH_REWARD_POLICY,
    CHECKPOINT_EPOCH_SCHEDULE_MODE,
    CHECKPOINT_EPOCH_SCHEMA_VERSION,
    CHECKPOINT_EPOCH_TRAINING_MODES,
    CHECKPOINT_EPOCH_VALUATION_POLICY,
    CheckpointBinding,
    EpochPlan,
    ProtocolBinding,
    SignedEpochCommitmentSet,
    WindowSchedule,
    build_epoch_plan,
    canonical_json_bytes,
    canonical_manifest_bytes,
    canonical_signed_commitment_set_bytes,
    manifest_sha256,
    parse_epoch_plan,
    parse_signed_commitment_set,
    validate_commitment_set_for_plan,
)
from reliquary.shared.checkpoint_epoch_market import (
    SignedGenerationIntentSet,
    canonical_signed_generation_intent_set_bytes,
    parse_signed_generation_intent_set,
)


class EpochStoreError(RuntimeError):
    pass


class EpochEquivocationError(EpochStoreError):
    pass


@dataclass(frozen=True, slots=True)
class EpochCommitIntent:
    protocol: ProtocolBinding
    checkpoint: CheckpointBinding
    first_window: int
    window_count: int
    beacon_source: str
    beacon_chain: str
    beacon_chain_hash: str
    beacon_target_round: int
    warmup_rounds: int
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
    environment_universes: tuple[tuple[str, int], ...]

    @property
    def intent_id(self) -> str:
        return hashlib.sha256(canonical_intent_bytes(self)).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedEpochIntent:
    intent: EpochCommitIntent
    intent_sha256: str
    validator_hotkey: str
    validator_signature: str


def _intent_dict(intent: EpochCommitIntent) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_EPOCH_SCHEMA_VERSION,
        "experimental_capability_id": CHECKPOINT_EPOCH_CAPABILITY_ID,
        "protocol": {
            "profile_id": intent.protocol.profile_id,
            "protocol_version": intent.protocol.protocol_version,
            "generation_contract_sha256": (
                intent.protocol.generation_contract_sha256
            ),
        },
        "checkpoint": {
            "number": intent.checkpoint.number,
            "repo_id": intent.checkpoint.repo_id,
            "revision": intent.checkpoint.revision,
            "commit_observed_round": (
                intent.checkpoint.commit_observed_round
            ),
        },
        "first_window": intent.first_window,
        "window_count": intent.window_count,
        "beacon_target": {
            "source": intent.beacon_source,
            "chain": intent.beacon_chain,
            "chain_hash": intent.beacon_chain_hash,
            "round": intent.beacon_target_round,
        },
        "warmup_rounds": intent.warmup_rounds,
        "window_schedule": {
            "mode": intent.window_schedule.mode,
            "collection_seconds": intent.window_schedule.collection_seconds,
            "timeout_seconds": intent.window_schedule.timeout_seconds,
        },
        "training_mode": intent.training_mode,
        "prompt_range_size": intent.prompt_range_size,
        "target_groups_per_environment_lane": (
            intent.target_groups_per_environment_lane
        ),
        "candidate_limit_per_environment_lane": (
            intent.candidate_limit_per_environment_lane
        ),
        "admission_policy": intent.admission_policy,
        "valuation_policy": intent.valuation_policy,
        "ranking_policy": intent.ranking_policy,
        "reward_policy": intent.reward_policy,
        "finalization_policy": intent.finalization_policy,
        "commitments_per_operator_per_environment_lane": (
            intent.commitments_per_operator_per_environment_lane
        ),
        "intent_seconds": intent.intent_seconds,
        "backup_activation_fractions": list(
            intent.backup_activation_fractions
        ),
        "reveal_seconds": intent.reveal_seconds,
        "environment_universes": {
            name: size for name, size in intent.environment_universes
        },
    }


def canonical_intent_bytes(intent: EpochCommitIntent) -> bytes:
    return canonical_json_bytes(_intent_dict(intent))


def intent_signing_bytes(intent: EpochCommitIntent) -> bytes:
    domain = b"reliquary/checkpoint-epoch/intent-signing/v1"
    raw = canonical_intent_bytes(intent)
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.digest()


def canonical_signed_intent_bytes(value: SignedEpochIntent) -> bytes:
    if not isinstance(value, SignedEpochIntent):
        raise TypeError("value must be SignedEpochIntent")
    if value.intent_sha256 != value.intent.intent_id:
        raise ValueError("signed epoch intent hash differs")
    if not value.validator_hotkey:
        raise ValueError("signed epoch intent validator hotkey is empty")
    signature = value.validator_signature
    if (
        not signature
        or len(signature) > 256
        or len(signature) % 2
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("signed epoch intent signature is invalid")
    return canonical_json_bytes(
        {
            "intent": _intent_dict(value.intent),
            "intent_sha256": value.intent_sha256,
            "validator_hotkey": value.validator_hotkey,
            "validator_signature": value.validator_signature,
        }
    )


def parse_signed_epoch_intent(raw: bytes | str) -> SignedEpochIntent:
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid signed checkpoint epoch intent") from exc
    if not isinstance(value, dict) or set(value) != {
        "intent",
        "intent_sha256",
        "validator_hotkey",
        "validator_signature",
    }:
        raise ValueError("signed checkpoint epoch intent keys differ")
    intent_raw = canonical_json_bytes(value["intent"])
    intent = parse_epoch_intent(intent_raw)
    publication = SignedEpochIntent(
        intent=intent,
        intent_sha256=value["intent_sha256"],
        validator_hotkey=value["validator_hotkey"],
        validator_signature=value["validator_signature"],
    )
    if raw_bytes != canonical_signed_intent_bytes(publication):
        raise ValueError("signed checkpoint epoch intent is not canonical")
    return publication


def build_epoch_intent(
    *,
    protocol: ProtocolBinding,
    checkpoint_number: int,
    checkpoint_repo_id: str,
    checkpoint_revision: str,
    commit_observed_round: int,
    first_window: int,
    window_count: int,
    beacon_chain: str,
    beacon_chain_hash: str,
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
    backup_activation_fractions: tuple[float, ...] = (0.5, 0.75),
    reveal_seconds: float = 60.0,
    environment_universes: Mapping[str, int],
) -> EpochCommitIntent:
    if isinstance(commit_observed_round, bool) or commit_observed_round < 1:
        raise ValueError("commit_observed_round must be positive")
    checkpoint = CheckpointBinding(
        number=int(checkpoint_number),
        repo_id=str(checkpoint_repo_id),
        revision=str(checkpoint_revision),
        commit_observed_round=int(commit_observed_round),
    )
    universes = tuple(
        (str(name), int(environment_universes[name]))
        for name in sorted(environment_universes)
    )
    if not universes or any(not name or size < 1 for name, size in universes):
        raise ValueError("environment universes must be non-empty and positive")
    if window_schedule.mode != CHECKPOINT_EPOCH_SCHEDULE_MODE:
        raise ValueError("unsupported checkpoint epoch window schedule")
    if training_mode not in CHECKPOINT_EPOCH_TRAINING_MODES:
        raise ValueError("unsupported checkpoint epoch training mode")
    intent = EpochCommitIntent(
        protocol=protocol,
        checkpoint=checkpoint,
        first_window=int(first_window),
        window_count=int(window_count),
        beacon_source="drand",
        beacon_chain=str(beacon_chain),
        beacon_chain_hash=str(beacon_chain_hash),
        beacon_target_round=int(commit_observed_round) + 1,
        warmup_rounds=int(warmup_rounds),
        window_schedule=window_schedule,
        training_mode=str(training_mode),
        prompt_range_size=int(prompt_range_size),
        target_groups_per_environment_lane=int(
            target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=int(
            candidate_limit_per_environment_lane
        ),
        admission_policy=str(admission_policy),
        valuation_policy=str(valuation_policy),
        ranking_policy=str(ranking_policy),
        reward_policy=str(reward_policy),
        finalization_policy=str(finalization_policy),
        commitments_per_operator_per_environment_lane=int(
            commitments_per_operator_per_environment_lane
        ),
        intent_seconds=float(intent_seconds),
        backup_activation_fractions=tuple(
            float(value) for value in backup_activation_fractions
        ),
        reveal_seconds=float(reveal_seconds),
        environment_universes=universes,
    )
    if intent.first_window < 0 or intent.window_count < 1:
        raise ValueError("invalid checkpoint epoch window range")
    if intent.warmup_rounds < 1 or intent.prompt_range_size < 1:
        raise ValueError("invalid checkpoint epoch warm-up or prompt range")
    if (
        intent.target_groups_per_environment_lane < 1
        or intent.candidate_limit_per_environment_lane
        < intent.target_groups_per_environment_lane
        or intent.admission_policy != CHECKPOINT_EPOCH_ADMISSION_POLICY
        or intent.valuation_policy != CHECKPOINT_EPOCH_VALUATION_POLICY
        or intent.ranking_policy != CHECKPOINT_EPOCH_RANKING_POLICY
        or intent.reward_policy != CHECKPOINT_EPOCH_REWARD_POLICY
        or intent.finalization_policy
        != CHECKPOINT_EPOCH_FINALIZATION_POLICY
        or intent.commitments_per_operator_per_environment_lane < 1
        or not math.isfinite(intent.intent_seconds)
        or intent.intent_seconds <= 0
        or not intent.backup_activation_fractions
        or tuple(sorted(set(intent.backup_activation_fractions)))
        != intent.backup_activation_fractions
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in intent.backup_activation_fractions
        )
        or not math.isfinite(intent.reveal_seconds)
        or intent.reveal_seconds <= 0
    ):
        raise ValueError("invalid checkpoint epoch admission bounds")
    return intent


def parse_epoch_intent(raw: bytes) -> EpochCommitIntent:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint epoch intent") from exc
    if not isinstance(value, dict):
        raise ValueError("checkpoint epoch intent must be an object")
    if set(value) != {
        "schema_version",
        "experimental_capability_id",
        "protocol",
        "checkpoint",
        "first_window",
        "window_count",
        "beacon_target",
        "warmup_rounds",
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
        "environment_universes",
    }:
        raise ValueError("checkpoint epoch intent keys differ")
    if (
        value["schema_version"] != CHECKPOINT_EPOCH_SCHEMA_VERSION
        or value["experimental_capability_id"]
        != CHECKPOINT_EPOCH_CAPABILITY_ID
    ):
        raise ValueError("checkpoint epoch intent version differs")
    protocol = value["protocol"]
    checkpoint = value["checkpoint"]
    beacon = value["beacon_target"]
    schedule = value["window_schedule"]
    universes = value["environment_universes"]
    if not all(
        isinstance(item, dict)
        for item in (protocol, checkpoint, beacon, schedule, universes)
    ):
        raise ValueError("checkpoint epoch intent objects are malformed")
    intent = EpochCommitIntent(
        protocol=ProtocolBinding(**protocol),
        checkpoint=CheckpointBinding(**checkpoint),
        first_window=value["first_window"],
        window_count=value["window_count"],
        beacon_source=beacon["source"],
        beacon_chain=beacon["chain"],
        beacon_chain_hash=beacon["chain_hash"],
        beacon_target_round=beacon["round"],
        warmup_rounds=value["warmup_rounds"],
        window_schedule=WindowSchedule(**schedule),
        training_mode=value["training_mode"],
        prompt_range_size=value["prompt_range_size"],
        target_groups_per_environment_lane=(
            value["target_groups_per_environment_lane"]
        ),
        candidate_limit_per_environment_lane=(
            value["candidate_limit_per_environment_lane"]
        ),
        admission_policy=value["admission_policy"],
        valuation_policy=value["valuation_policy"],
        ranking_policy=value["ranking_policy"],
        reward_policy=value["reward_policy"],
        finalization_policy=value["finalization_policy"],
        commitments_per_operator_per_environment_lane=(
            value["commitments_per_operator_per_environment_lane"]
        ),
        intent_seconds=value["intent_seconds"],
        backup_activation_fractions=tuple(
            value["backup_activation_fractions"]
        ),
        reveal_seconds=value["reveal_seconds"],
        environment_universes=tuple(
            (str(name), int(size)) for name, size in sorted(universes.items())
        ),
    )
    if raw != canonical_intent_bytes(intent):
        raise ValueError("checkpoint epoch intent is not canonical")
    if intent.window_schedule.mode != CHECKPOINT_EPOCH_SCHEDULE_MODE:
        raise ValueError("unsupported checkpoint epoch window schedule")
    if intent.training_mode not in CHECKPOINT_EPOCH_TRAINING_MODES:
        raise ValueError("unsupported checkpoint epoch training mode")
    if (
        intent.target_groups_per_environment_lane < 1
        or intent.candidate_limit_per_environment_lane
        < intent.target_groups_per_environment_lane
        or intent.admission_policy != CHECKPOINT_EPOCH_ADMISSION_POLICY
        or intent.valuation_policy != CHECKPOINT_EPOCH_VALUATION_POLICY
        or intent.ranking_policy != CHECKPOINT_EPOCH_RANKING_POLICY
        or intent.reward_policy != CHECKPOINT_EPOCH_REWARD_POLICY
        or intent.finalization_policy
        != CHECKPOINT_EPOCH_FINALIZATION_POLICY
        or intent.commitments_per_operator_per_environment_lane < 1
        or not math.isfinite(intent.intent_seconds)
        or intent.intent_seconds <= 0
        or not intent.backup_activation_fractions
        or tuple(sorted(set(intent.backup_activation_fractions)))
        != intent.backup_activation_fractions
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in intent.backup_activation_fractions
        )
        or not math.isfinite(intent.reveal_seconds)
        or intent.reveal_seconds <= 0
    ):
        raise ValueError("invalid checkpoint epoch admission bounds")
    if intent.beacon_target_round != intent.checkpoint.commit_observed_round + 1:
        raise ValueError("intent does not target the first post-commit beacon")
    return intent


def plan_from_intent(
    intent: EpochCommitIntent,
    *,
    beacon: BeaconBinding,
) -> EpochPlan:
    if (
        beacon.source != intent.beacon_source
        or beacon.chain != intent.beacon_chain
        or beacon.chain_hash != intent.beacon_chain_hash
        or beacon.round != intent.beacon_target_round
    ):
        raise ValueError("beacon does not match checkpoint epoch intent")
    return build_epoch_plan(
        protocol=intent.protocol,
        checkpoint=intent.checkpoint,
        first_window=intent.first_window,
        window_count=intent.window_count,
        epoch_beacon=beacon,
        beacon_delay_rounds=1,
        warmup_rounds=intent.warmup_rounds,
        window_schedule=intent.window_schedule,
        training_mode=intent.training_mode,
        prompt_range_size=intent.prompt_range_size,
        target_groups_per_environment_lane=(
            intent.target_groups_per_environment_lane
        ),
        candidate_limit_per_environment_lane=(
            intent.candidate_limit_per_environment_lane
        ),
        admission_policy=intent.admission_policy,
        valuation_policy=intent.valuation_policy,
        ranking_policy=intent.ranking_policy,
        reward_policy=intent.reward_policy,
        finalization_policy=intent.finalization_policy,
        commitments_per_operator_per_environment_lane=(
            intent.commitments_per_operator_per_environment_lane
        ),
        intent_seconds=intent.intent_seconds,
        backup_activation_fractions=intent.backup_activation_fractions,
        reveal_seconds=intent.reveal_seconds,
        environment_universes=dict(intent.environment_universes),
    )


class EpochStore:
    """Create-only intent and manifest persistence with current pointers."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def install_intent(self, intent: EpochCommitIntent) -> bytes:
        raw = canonical_intent_bytes(intent)
        self._install_create_only(
            self.root / f"intent-{intent.intent_id}.json",
            raw,
            "intent",
        )
        self._write_pointer("current-intent", intent.intent_id)
        return raw

    def load_current_intent(self) -> EpochCommitIntent | None:
        identifier = self._read_pointer("current-intent")
        if identifier is None:
            return None
        path = self.root / f"intent-{identifier}.json"
        intent = parse_epoch_intent(path.read_bytes())
        if intent.intent_id != identifier:
            raise EpochStoreError("intent pointer/hash mismatch")
        return intent

    def confirm_before_beacon(
        self,
        intent: EpochCommitIntent,
        *,
        observed_round: int,
    ) -> None:
        if (
            isinstance(observed_round, bool)
            or observed_round < 1
            or observed_round >= intent.beacon_target_round
        ):
            raise EpochStoreError(
                "intent was not durably confirmed before beacon availability"
            )
        if self.load_signed_intent(intent) is None:
            raise EpochStoreError(
                "signed epoch intent was not durable before confirmation"
            )
        raw = canonical_json_bytes(
            {
                "intent_id": intent.intent_id,
                "observed_round": int(observed_round),
                "beacon_target_round": intent.beacon_target_round,
            }
        )
        self._install_create_only(
            self.root / f"confirmed-{intent.intent_id}.json",
            raw,
            "intent confirmation",
        )

    def is_confirmed(self, intent: EpochCommitIntent) -> bool:
        path = self.root / f"confirmed-{intent.intent_id}.json"
        if not path.exists():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return (
                value["intent_id"] == intent.intent_id
                and value["beacon_target_round"] == intent.beacon_target_round
                and 1 <= int(value["observed_round"])
                < intent.beacon_target_round
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def install_plan(
        self,
        intent: EpochCommitIntent,
        plan: EpochPlan,
    ) -> bytes:
        if not self.is_confirmed(intent):
            raise EpochStoreError("checkpoint epoch intent is not confirmed")
        if plan.epoch_beacon.round != intent.beacon_target_round:
            raise EpochStoreError("plan does not match confirmed intent")
        expected = plan_from_intent(intent, beacon=plan.epoch_beacon)
        if plan != expected:
            raise EpochStoreError("plan differs from confirmed intent")
        raw = canonical_manifest_bytes(plan)
        self._install_create_only(
            self.root / f"plan-{plan.epoch_id}.json",
            raw,
            "manifest",
        )
        self._write_pointer("current-plan", plan.epoch_id)
        return raw

    def install_signed_intent(
        self,
        publication: SignedEpochIntent,
    ) -> bytes:
        current = self.load_current_intent()
        if current is None or current != publication.intent:
            raise EpochStoreError("signed epoch intent is not current")
        raw = canonical_signed_intent_bytes(publication)
        self._install_create_only(
            self.root / f"signed-intent-{publication.intent.intent_id}.json",
            raw,
            "signed intent",
        )
        return raw

    def load_signed_intent(
        self,
        intent: EpochCommitIntent,
    ) -> SignedEpochIntent | None:
        path = self.root / f"signed-intent-{intent.intent_id}.json"
        if not path.exists():
            return None
        try:
            publication = parse_signed_epoch_intent(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise EpochStoreError("invalid signed epoch intent") from exc
        if publication.intent != intent:
            raise EpochStoreError("signed epoch intent differs from intent")
        return publication

    def load_current_plan(self) -> EpochPlan | None:
        identifier = self._read_pointer("current-plan")
        if identifier is None:
            return None
        path = self.root / f"plan-{identifier}.json"
        plan = parse_epoch_plan(path.read_bytes())
        if plan.epoch_id != identifier:
            raise EpochStoreError("plan pointer/epoch mismatch")
        return plan

    def mark_activated(self, plan: EpochPlan) -> None:
        """Durably forbid reopening a collection after its routes are exposed."""
        self._install_create_only(
            self.root / f"activated-{plan.epoch_id}.json",
            canonical_json_bytes({
                "epoch_id": plan.epoch_id,
                "manifest_sha256": manifest_sha256(plan),
            }),
            "epoch activation",
        )

    def is_activated(self, plan: EpochPlan) -> bool:
        path = self.root / f"activated-{plan.epoch_id}.json"
        if not path.exists():
            return False
        expected = canonical_json_bytes({
            "epoch_id": plan.epoch_id,
            "manifest_sha256": manifest_sha256(plan),
        })
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise EpochStoreError("cannot read epoch activation") from exc
        if actual != expected:
            raise EpochStoreError("epoch activation does not match plan")
        return True

    def mark_terminal(self, plan: EpochPlan, *, status: str) -> None:
        """Commit the one terminal outcome for an activated epoch."""
        if status not in {"completed", "aborted"}:
            raise ValueError("epoch terminal status must be completed or aborted")
        if not self.is_activated(plan):
            raise EpochStoreError("checkpoint epoch was not activated")
        self._install_create_only(
            self.root / f"terminal-{plan.epoch_id}.json",
            canonical_json_bytes({
                "epoch_id": plan.epoch_id,
                "manifest_sha256": manifest_sha256(plan),
                "status": status,
            }),
            "epoch terminal outcome",
        )

    def terminal_status(self, plan: EpochPlan) -> str | None:
        path = self.root / f"terminal-{plan.epoch_id}.json"
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EpochStoreError("invalid epoch terminal outcome") from exc
        if (
            value.get("epoch_id") != plan.epoch_id
            or value.get("manifest_sha256") != manifest_sha256(plan)
            or value.get("status") not in {"completed", "aborted"}
            or raw != canonical_json_bytes(value)
        ):
            raise EpochStoreError("epoch terminal outcome does not match plan")
        return str(value["status"])

    def install_commitment_set(
        self,
        plan: EpochPlan,
        publication: SignedEpochCommitmentSet,
    ) -> bytes:
        """Persist the signed frozen set before admission randomness exists."""
        if not self.is_activated(plan):
            raise EpochStoreError("checkpoint epoch was not activated")
        try:
            validate_commitment_set_for_plan(publication.commitment_set, plan)
        except (TypeError, ValueError) as exc:
            raise EpochStoreError("commitment set does not match epoch plan") from exc
        raw = canonical_signed_commitment_set_bytes(publication)
        self._install_create_only(
            self.root / f"commitments-{plan.epoch_id}.json",
            raw,
            "signed commitment set",
        )
        return raw

    def load_commitment_set(
        self,
        plan: EpochPlan,
    ) -> SignedEpochCommitmentSet | None:
        path = self.root / f"commitments-{plan.epoch_id}.json"
        if not path.exists():
            return None
        try:
            publication = parse_signed_commitment_set(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise EpochStoreError("invalid signed commitment set") from exc
        try:
            validate_commitment_set_for_plan(publication.commitment_set, plan)
        except (TypeError, ValueError) as exc:
            raise EpochStoreError("stored commitment set does not match plan") from exc
        return publication

    def install_generation_intent_set(
        self,
        plan: EpochPlan,
        publication: SignedGenerationIntentSet,
    ) -> bytes:
        """Persist the exact miner intent population before beacon A."""
        if not self.is_activated(plan):
            raise EpochStoreError("checkpoint epoch was not activated")
        intent_set = publication.intent_set
        if (
            intent_set.epoch_id != plan.epoch_id
            or intent_set.manifest_sha256 != manifest_sha256(plan)
            or intent_set.intent_close_round < plan.epoch_beacon.round
        ):
            raise EpochStoreError("generation intent set does not match epoch plan")
        raw = canonical_signed_generation_intent_set_bytes(publication)
        self._install_create_only(
            self.root / f"generation-intents-{plan.epoch_id}.json",
            raw,
            "signed generation intent set",
        )
        return raw

    def load_generation_intent_set(
        self,
        plan: EpochPlan,
    ) -> SignedGenerationIntentSet | None:
        path = self.root / f"generation-intents-{plan.epoch_id}.json"
        if not path.exists():
            return None
        try:
            publication = parse_signed_generation_intent_set(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise EpochStoreError("invalid signed generation intent set") from exc
        intent_set = publication.intent_set
        if (
            intent_set.epoch_id != plan.epoch_id
            or intent_set.manifest_sha256 != manifest_sha256(plan)
            or intent_set.intent_close_round < plan.epoch_beacon.round
        ):
            raise EpochStoreError("stored generation intent set differs from plan")
        return publication

    def _install_create_only(
        self,
        path: Path,
        raw: bytes,
        label: str,
    ) -> None:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise EpochEquivocationError(
                    f"{label} already exists with different bytes"
                )
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_directory()

    def _write_pointer(self, name: str, identifier: str) -> None:
        temporary = self.root / f".{name}.{os.getpid()}.tmp"
        with temporary.open("w", encoding="ascii") as handle:
            handle.write(identifier + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.root / name)
        self._fsync_directory()

    def _read_pointer(self, name: str) -> str | None:
        path = self.root / name
        if not path.exists():
            return None
        value = path.read_text(encoding="ascii").strip()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise EpochStoreError(f"invalid {name} pointer")
        return value

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "EpochCommitIntent",
    "EpochEquivocationError",
    "EpochStore",
    "EpochStoreError",
    "SignedEpochIntent",
    "build_epoch_intent",
    "canonical_intent_bytes",
    "canonical_signed_intent_bytes",
    "intent_signing_bytes",
    "parse_epoch_intent",
    "parse_signed_epoch_intent",
    "plan_from_intent",
]
