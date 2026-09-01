"""Fail-closed staging seam for future checkpoint-epoch proof streaming.

It does not submit proofs.  Live streaming remains off until the scheduler can
recover results durably and isolate all concurrent lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import threading
from typing import Any, Mapping, Sequence

from reliquary.shared.checkpoint_epoch import canonical_json_bytes
from reliquary.validator.proof_scheduler import ProofSchedulerCapabilities


class EpochProofStagingError(RuntimeError):
    pass


class EpochProofStreamingUnsupported(EpochProofStagingError):
    pass


class StagedProofState(str, Enum):
    STAGED = "staged"
    DISPATCHED = "dispatched"
    PASSED = "passed"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class EpochProofBinding:
    epoch_id: str
    manifest_sha256: str
    checkpoint_revision: str
    protocol_profile_id: str
    generation_contract_sha256: str
    first_window: int
    window_count: int
    environments: tuple[str, ...]
    generation_randomness_by_offset: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "epoch_id",
            "manifest_sha256",
            "generation_contract_sha256",
        ):
            _hex64(name, getattr(self, name))
        if not self.checkpoint_revision or not self.protocol_profile_id:
            raise ValueError("checkpoint revision and profile must be non-empty")
        if self.first_window < 0 or self.window_count < 1:
            raise ValueError("invalid epoch window range")
        if not self.environments or len(set(self.environments)) != len(
            self.environments
        ):
            raise ValueError("environments must be non-empty and unique")
        if len(self.generation_randomness_by_offset) != self.window_count:
            raise ValueError("generation randomness must cover every window")
        for randomness in self.generation_randomness_by_offset:
            _hex64("generation_randomness", randomness)

    def randomness_for(self, window_number: int) -> str:
        offset = window_number - self.first_window
        if offset < 0 or offset >= self.window_count:
            raise EpochProofStagingError("window is outside the epoch")
        return self.generation_randomness_by_offset[offset]


@dataclass(frozen=True)
class TicketedProofCandidate:
    intent_id: str
    operator_id: str
    window_number: int
    environment: str
    prompt_idx: int
    payload_sha256: str
    generation_randomness: str
    selection_rank: int
    ticket_state: str

    def __post_init__(self) -> None:
        if not self.intent_id or not self.operator_id or not self.environment:
            raise ValueError("ticket identities must be non-empty")
        if self.window_number < 0 or self.prompt_idx < 0 or self.selection_rank < 0:
            raise ValueError("ticket numeric fields must be non-negative")
        _hex64("payload_sha256", self.payload_sha256)
        _hex64("generation_randomness", self.generation_randomness)
        if self.ticket_state not in {"primary", "active_backup"}:
            raise ValueError("ticket_state is not activated for generation")

    @property
    def lane(self) -> tuple[int, str]:
        return self.window_number, self.environment


@dataclass
class _Record:
    candidate: TicketedProofCandidate
    state: StagedProofState = StagedProofState.STAGED
    result_sha256: str | None = None


class TicketedEpochProofCoordinator:
    """Journalable ticket-to-proof state machine.

    Persist the snapshot after ``mark_dispatched`` and before invoking a worker.
    Recovery quarantines persisted in-flight work, so it cannot be re-proved.
    """

    SCHEMA_VERSION = 1

    def __init__(self, binding: EpochProofBinding) -> None:
        self.binding = binding
        self._records: dict[str, _Record] = {}
        self._payload_owner: dict[str, str] = {}
        self._frozen = False
        self._population_sha256: str | None = None
        self._claims: dict[tuple[int, str], tuple[str, tuple[str, ...]]] = {}
        self._lock = threading.RLock()

    def stage(self, candidate: TicketedProofCandidate) -> bool:
        if candidate.environment not in self.binding.environments:
            raise EpochProofStagingError("unknown epoch environment")
        if candidate.generation_randomness != self.binding.randomness_for(
            candidate.window_number
        ):
            raise EpochProofStagingError("proof generation randomness changed")
        with self._lock:
            if self._frozen:
                raise EpochProofStagingError("proof population is frozen")
            existing = self._records.get(candidate.intent_id)
            if existing:
                if existing.candidate != candidate:
                    raise EpochProofStagingError("intent_id was rebound")
                return False
            if candidate.payload_sha256 in self._payload_owner:
                raise EpochProofStagingError("payload is already staged")
            self._records[candidate.intent_id] = _Record(candidate)
            self._payload_owner[candidate.payload_sha256] = candidate.intent_id
            return True

    def freeze(self) -> str:
        with self._lock:
            digest = hashlib.sha256(
                canonical_json_bytes(
                    [_candidate_dict(record.candidate) for record in self._ordered()]
                )
            ).hexdigest()
            if self._frozen and digest != self._population_sha256:
                raise EpochProofStagingError("proof population equivocated")
            self._frozen = True
            self._population_sha256 = digest
            return digest

    def dispatch_order(
        self, lane: tuple[int, str]
    ) -> tuple[TicketedProofCandidate, ...]:
        with self._lock:
            self._require_frozen()
            self._validate_lane(lane)
            return tuple(
                record.candidate
                for record in self._ordered()
                if record.candidate.lane == lane
            )

    def mark_dispatched(self, intent_id: str) -> bool:
        with self._lock:
            self._require_frozen()
            record = self._record(intent_id)
            if record.state is StagedProofState.DISPATCHED:
                return False
            if record.state is not StagedProofState.STAGED:
                raise EpochProofStagingError(
                    f"proof in state {record.state.value!r} cannot dispatch"
                )
            for earlier in self._ordered():
                if earlier.candidate.lane != record.candidate.lane:
                    continue
                if earlier.candidate.intent_id == intent_id:
                    break
                if earlier.state is StagedProofState.STAGED:
                    raise EpochProofStagingError(
                        "dispatch would make arrival order override ticket order"
                    )
            record.state = StagedProofState.DISPATCHED
            return True

    def record_terminal(
        self, intent_id: str, *, passed: bool, result_sha256: str
    ) -> bool:
        _hex64("result_sha256", result_sha256)
        state = StagedProofState.PASSED if passed else StagedProofState.REJECTED
        with self._lock:
            record = self._record(intent_id)
            if record.state is state and record.result_sha256 == result_sha256:
                return False
            if record.state is not StagedProofState.DISPATCHED:
                raise EpochProofStagingError(
                    f"proof in state {record.state.value!r} cannot become terminal"
                )
            record.state = state
            record.result_sha256 = result_sha256
            return True

    def assert_lane_terminal(self, lane: tuple[int, str]) -> None:
        with self._lock:
            self._require_frozen()
            self._validate_lane(lane)
            blockers = sorted(
                intent_id
                for intent_id, record in self._records.items()
                if record.candidate.lane == lane
                and record.state not in {
                    StagedProofState.PASSED,
                    StagedProofState.REJECTED,
                }
            )
            if blockers:
                raise EpochProofStagingError(
                    "lane has non-terminal or ambiguous proofs: "
                    + ",".join(blockers)
                )

    def claim_lane_finalization(
        self, lane: tuple[int, str], winner_intent_ids: Sequence[str]
    ) -> tuple[str, bool]:
        """Return ``(claim_hash, should_emit)``; exact replay never emits twice."""

        with self._lock:
            self.assert_lane_terminal(lane)
            winners = tuple(winner_intent_ids)
            if len(winners) != len(set(winners)):
                raise EpochProofStagingError("winner intent ids must be unique")
            for intent_id in winners:
                record = self._record(intent_id)
                if (
                    record.candidate.lane != lane
                    or record.state is not StagedProofState.PASSED
                ):
                    raise EpochProofStagingError("winner is not a passed lane proof")
            claim = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "epoch_id": self.binding.epoch_id,
                        "manifest_sha256": self.binding.manifest_sha256,
                        "population_sha256": self._population_sha256,
                        "lane": list(lane),
                        "winner_intent_ids": list(winners),
                    }
                )
            ).hexdigest()
            existing = self._claims.get(lane)
            if existing:
                if existing != (claim, winners):
                    raise EpochProofStagingError("lane was already claimed differently")
                return claim, False
            self._claims[lane] = claim, winners
            return claim, True

    def snapshot_bytes(self) -> bytes:
        with self._lock:
            return canonical_json_bytes(self._snapshot())

    @classmethod
    def from_snapshot_bytes(
        cls, raw: bytes, *, quarantine_dispatched: bool = True
    ) -> TicketedEpochProofCoordinator:
        try:
            body = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("epoch proof snapshot is not valid JSON") from exc
        if raw != canonical_json_bytes(body):
            raise ValueError("epoch proof snapshot is not canonical")
        if not isinstance(body, Mapping) or set(body) != {
            "binding",
            "claims",
            "frozen",
            "population_sha256",
            "records",
            "schema_version",
        }:
            raise ValueError("epoch proof snapshot keys differ")
        if body["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported epoch proof snapshot schema")
        binding = EpochProofBinding(**{
            **body["binding"],
            "environments": tuple(body["binding"]["environments"]),
            "generation_randomness_by_offset": tuple(
                body["binding"]["generation_randomness_by_offset"]
            ),
        })
        coordinator = cls(binding)
        for value in body["records"]:
            candidate = TicketedProofCandidate(**value["candidate"])
            coordinator.stage(candidate)
            record = coordinator._records[candidate.intent_id]
            state = StagedProofState(value["state"])
            record.state = (
                StagedProofState.QUARANTINED
                if quarantine_dispatched and state is StagedProofState.DISPATCHED
                else state
            )
            record.result_sha256 = value["result_sha256"]
            if record.state in {StagedProofState.PASSED, StagedProofState.REJECTED}:
                _hex64("result_sha256", record.result_sha256)
            elif record.result_sha256 is not None:
                raise ValueError("non-terminal proof has a result digest")
        coordinator._frozen = bool(body["frozen"])
        coordinator._population_sha256 = body["population_sha256"]
        if coordinator._frozen:
            _hex64("population_sha256", coordinator._population_sha256)
            if coordinator.freeze() != body["population_sha256"]:
                raise ValueError("epoch proof population digest differs")
        elif body["population_sha256"] is not None:
            raise ValueError("unfrozen snapshot has a population hash")
        for value in body["claims"]:
            lane = value["window_number"], value["environment"]
            coordinator._validate_lane(lane)
            _hex64("claim_sha256", value["claim_sha256"])
            coordinator._claims[lane] = (
                value["claim_sha256"],
                tuple(value["winner_intent_ids"]),
            )
        return coordinator

    def quarantined_intent_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                intent_id
                for intent_id, record in self._records.items()
                if record.state is StagedProofState.QUARANTINED
            ))

    def _ordered(self) -> list[_Record]:
        return sorted(
            self._records.values(),
            key=lambda record: (
                record.candidate.window_number,
                record.candidate.environment,
                record.candidate.selection_rank,
                record.candidate.intent_id,
            ),
        )

    def _record(self, intent_id: str) -> _Record:
        try:
            return self._records[intent_id]
        except KeyError as exc:
            raise EpochProofStagingError("unknown proof intent") from exc

    def _require_frozen(self) -> None:
        if not self._frozen or self._population_sha256 is None:
            raise EpochProofStagingError("proof population is not frozen")

    def _validate_lane(self, lane: tuple[int, str]) -> None:
        self.binding.randomness_for(lane[0])
        if lane[1] not in self.binding.environments:
            raise EpochProofStagingError("unknown epoch environment")

    def _snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "binding": {
                "epoch_id": self.binding.epoch_id,
                "manifest_sha256": self.binding.manifest_sha256,
                "checkpoint_revision": self.binding.checkpoint_revision,
                "protocol_profile_id": self.binding.protocol_profile_id,
                "generation_contract_sha256": self.binding.generation_contract_sha256,
                "first_window": self.binding.first_window,
                "window_count": self.binding.window_count,
                "environments": list(self.binding.environments),
                "generation_randomness_by_offset": list(
                    self.binding.generation_randomness_by_offset
                ),
            },
            "frozen": self._frozen,
            "population_sha256": self._population_sha256,
            "records": [
                {
                    "candidate": _candidate_dict(record.candidate),
                    "state": record.state.value,
                    "result_sha256": record.result_sha256,
                }
                for record in self._ordered()
            ],
            "claims": [
                {
                    "window_number": lane[0],
                    "environment": lane[1],
                    "claim_sha256": claim,
                    "winner_intent_ids": list(winners),
                }
                for lane, (claim, winners) in sorted(self._claims.items())
            ],
        }


def streaming_runtime_blockers(
    capabilities: ProofSchedulerCapabilities, *, concurrent_lanes: int
) -> tuple[str, ...]:
    if concurrent_lanes < 1:
        raise ValueError("concurrent_lanes must be positive")
    checks = (
        (
            capabilities.max_live_plans_per_environment < concurrent_lanes,
            "insufficient_lane_isolation",
        ),
        (not capabilities.durable_result_recovery, "proof_results_are_not_durable"),
        (
            not capabilities.supports_predeclared_candidates,
            "ticket_slots_cannot_be_predeclared",
        ),
        (
            not capabilities.supports_rank_independent_extension,
            "plan_extension_is_append_ordered",
        ),
    )
    return tuple(name for blocked, name in checks if blocked)


def require_streaming_runtime(
    capabilities: ProofSchedulerCapabilities, *, concurrent_lanes: int
) -> None:
    blockers = streaming_runtime_blockers(
        capabilities, concurrent_lanes=concurrent_lanes
    )
    if blockers:
        raise EpochProofStreamingUnsupported(
            "checkpoint-epoch proof streaming is unavailable: "
            + ",".join(blockers)
        )


def _candidate_dict(candidate: TicketedProofCandidate) -> dict[str, Any]:
    return {
        "intent_id": candidate.intent_id,
        "operator_id": candidate.operator_id,
        "window_number": candidate.window_number,
        "environment": candidate.environment,
        "prompt_idx": candidate.prompt_idx,
        "payload_sha256": candidate.payload_sha256,
        "generation_randomness": candidate.generation_randomness,
        "selection_rank": candidate.selection_rank,
        "ticket_state": candidate.ticket_state,
    }


def _hex64(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


__all__ = [
    "EpochProofBinding",
    "EpochProofStagingError",
    "EpochProofStreamingUnsupported",
    "StagedProofState",
    "TicketedEpochProofCoordinator",
    "TicketedProofCandidate",
    "require_streaming_runtime",
    "streaming_runtime_blockers",
]
