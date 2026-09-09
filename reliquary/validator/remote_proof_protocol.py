"""Versioned JSON boundary for the existing GRAIL proof kernel.

Only sparse, finite proof outputs cross this boundary. Model tensors, Python
callbacks and the local multiprocessing transport never do.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from reliquary.shared.checkpoint_identity import canonical_checkpoint_identity
from reliquary.shared.strict_json import strict_json_loads

PROOF_PROTOCOL = "reliquary.remote-proof/v1"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_TOKENS = 65536
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OID = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Name = Annotated[str, Field(min_length=1, max_length=192, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+#-]*$")]
Count = Annotated[int, Field(ge=0, le=2**63-1)]
Probability = Annotated[float, Field(ge=0, le=1)]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def transport_hash() -> str:
    """Pin the adapter as well as the unchanged kernel capacity contract."""
    root = Path(__file__).parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("remote_proof_protocol.py", "remote_proof.py",
                                "remote_proof_server.py", "proof_worker.py",
                                "proof_measurements.py", "batcher.py", "service.py")})


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @classmethod
    def read(cls, body: bytes, *, limit: int = MAX_REQUEST_BYTES):
        if len(body) > limit:
            raise ValueError("remote proof message is too large")
        return cls.model_validate(strict_json_loads(body))


class CheckpointBinding(WireModel):
    profile_id: Name
    generation_contract_sha256: Hash
    training_run_id: Name
    checkpoint_n: Count
    repo_id: Name
    revision: OID

    @model_validator(mode="after")
    def canonical_identity(self):
        canonical_checkpoint_identity(self.checkpoint_n, self.repo_id, self.revision)
        return self


class RemoteProofMeasurement(WireModel):
    protocol: Literal[PROOF_PROTOCOL] = PROOF_PROTOCOL
    worker_id: Name
    transport_sha256: Hash
    measurement_scope: Literal["validator-end-to-end-mtls"] = "validator-end-to-end-mtls"


class Sketch(WireModel):
    sketch: Annotated[int, Field(ge=0, le=2**63-1)]


class ProofInput(WireModel):
    tokens: Annotated[list[Annotated[int, Field(ge=0, le=2**31-1)]], Field(min_length=1, max_length=MAX_TOKENS)]
    commitments: Annotated[list[Sketch], Field(min_length=1, max_length=MAX_TOKENS)]
    # Existing, versioned rollout metadata includes the episode transcript.
    # JSON values preserve that contract without admitting Python objects.
    rollout: dict[str, JsonValue]
    randomness: Annotated[str, Field(min_length=2, max_length=256, pattern=r"^(?:0x)?[0-9a-fA-F]+$")]
    seed_u_values: Annotated[list[Probability], Field(max_length=MAX_TOKENS)] | None

    @model_validator(mode="after")
    def aligned(self):
        if len(self.tokens) != len(self.commitments):
            raise ValueError("one sketch per token is required")
        return self

    def commit(self) -> dict:
        return {"tokens": self.tokens,
                "commitments": [x.model_dump() for x in self.commitments],
                "rollout": self.rollout}


class ProofRequest(WireModel):
    protocol: Literal[PROOF_PROTOCOL] = PROOF_PROTOCOL
    job_id: Name
    attempt: Annotated[int, Field(ge=0, le=16)]
    worker_id: Name
    session_id: Name
    device_id: Name
    runtime_hash: Hash
    checkpoint: CheckpointBinding
    window: Count
    environment: Name
    expires_at_ms: Count
    content_sha256: Hash
    payload: ProofInput

    @model_validator(mode="after")
    def bound_content(self):
        if self.content_sha256 != digest(self.payload.model_dump()):
            raise ValueError("proof content digest mismatch")
        return self


class ProofValues(WireModel):
    all_passed: bool
    passed: Count
    checked: Count
    sketch_diff_max: Count
    has_sparse_outputs: Literal[True]
    p_stop: Probability | None
    challenge_lp_indices: list[Count]
    challenge_lp_values: list[float]
    completion_chosen_probs: list[Probability]
    completion_argmax_probs: list[Probability]
    completion_argmax_ids: list[Count]
    completion_entropies: list[Annotated[float, Field(ge=0)]]
    hidden_start_f16_b64: str | None
    hidden_delta_f16_b64: str | None
    hidden_dim: Count
    hidden_end_completion_offset: Count | None
    representation_shift_l2: Annotated[float, Field(ge=0)] | None
    seed_n_stochastic: Count
    seed_n_match: Count
    seed_n_positions: Count
    seed_n_boundary_match: Count
    seed_n_hard_mismatch: Count
    seed_n_deterministic_hard_mismatch: Count
    seed_n_miss_gt_0_01: Count
    seed_n_miss_gt_0_05: Count
    seed_n_miss_gt_0_10: Count
    seed_max_cdf_miss: Probability
    seed_first_hard_mismatch_offset: Count | None
    terminal_pick_ok: bool | None
    terminal_pick_cdf_miss: Probability | None
    natural_close_pick_ok: bool | None
    natural_close_pick_cdf_miss: Probability | None

    @model_validator(mode="after")
    def aligned(self):
        if self.passed > self.checked or self.all_passed != (0 < self.checked == self.passed):
            raise ValueError("inconsistent GRAIL counters")
        if len(self.challenge_lp_indices) != len(self.challenge_lp_values):
            raise ValueError("logprob challenges are not aligned")
        if not (len(self.completion_chosen_probs) == len(self.completion_argmax_probs)
                == len(self.completion_argmax_ids)):
            raise ValueError("token authenticity vectors are not aligned")
        return self

    @classmethod
    def from_kernel(cls, result):
        return cls.model_validate({name: getattr(result, name)
                                   for name in cls.model_fields})

    def to_kernel(self):
        from reliquary.validator.verifier import ProofResult
        return ProofResult(**self.model_dump())

    def validate_input_coverage(self, payload: ProofInput) -> None:
        from reliquary.validator.verifier import policy_token_positions, proof_challenge_indices
        positions = policy_token_positions(payload.tokens, payload.rollout)
        challenges = proof_challenge_indices(payload.tokens, payload.rollout, payload.randomness)
        if self.checked != len(challenges) or len(self.completion_chosen_probs) != len(positions):
            raise ValueError("remote proof did not cover every required token/challenge")
        if self.completion_entropies and len(self.completion_entropies) != len(positions):
            raise ValueError("remote proof entropy vector is incomplete")
        if any(index not in positions for index in self.challenge_lp_indices):
            raise ValueError("remote logprob result refers to another policy span")


class ProofResponse(WireModel):
    protocol: Literal[PROOF_PROTOCOL] = PROOF_PROTOCOL
    request_sha256: Hash
    job_id: Name
    attempt: Count
    worker_id: Name
    session_id: Name
    device_id: Name
    runtime_hash: Hash
    checkpoint: CheckpointBinding
    window: Count
    environment: Name
    content_sha256: Hash
    result: ProofValues


class AdoptionRequest(WireModel):
    protocol: Literal[PROOF_PROTOCOL] = PROOF_PROTOCOL
    worker_id: Name
    session_id: Name
    checkpoint: CheckpointBinding


class SlotState(WireModel):
    device_id: Name
    physical_device: Name
    hardware_class: Annotated[str, Field(min_length=1, max_length=256)]
    device_uuid: Name
    revision: OID | None
    runtime: dict[str, JsonValue]


class ProofHealth(WireModel):
    protocol: Literal[PROOF_PROTOCOL] = PROOF_PROTOCOL
    worker_id: Name
    session_id: Name
    profile_id: Name
    generation_contract_sha256: Hash
    training_run_id: Name
    repo_id: Name
    software_revision: OID
    proof_path_hash: Hash
    transport_sha256: Hash
    checkpoint: CheckpointBinding | None
    slots: Annotated[list[SlotState], Field(min_length=1, max_length=64)]
    config: dict[str, JsonValue]
    generation_config: dict[str, JsonValue]
