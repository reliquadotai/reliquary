import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    RuntimeFingerprint,
)
from reliquary.shared.runtime_fingerprint import (
    bind_runtime_profile_nonce,
    collect_runtime_fingerprint,
    runtime_profile_hash,
)
from reliquary.validator.server import ValidatorServer


def _runtime() -> RuntimeFingerprint:
    return RuntimeFingerprint.model_validate(collect_runtime_fingerprint())


def _request(runtime: RuntimeFingerprint, nonce: str) -> BatchSubmissionRequest:
    rollout = RolloutSubmission(
        tokens=[1],
        reward=0.0,
        commit={"tokens": [1]},
        env_name="openmathinstruct",
    )
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=1,
        window_start=2,
        merkle_root="00" * 32,
        rollouts=[rollout.model_copy(deep=True) for _ in range(8)],
        checkpoint_hash="sha256:test",
        protocol_version=2,
        nonce=nonce,
        runtime_fingerprint=runtime,
    )


def test_collected_runtime_profile_hash_is_canonical():
    runtime = _runtime()
    assert runtime.profile_hash == runtime_profile_hash(
        runtime.model_dump(exclude={"profile_hash"})
    )


def test_runtime_profile_rejects_tampered_field():
    profile = collect_runtime_fingerprint()
    profile["torch_version"] = "tampered"
    with pytest.raises(ValidationError, match="profile_hash"):
        RuntimeFingerprint.model_validate(profile)


def test_runtime_profile_must_be_bound_to_signed_nonce():
    runtime = _runtime()
    with pytest.raises(ValidationError, match="bound to nonce"):
        _request(runtime, "unbound")

    nonce = bind_runtime_profile_nonce("fresh", runtime.profile_hash)
    request = _request(runtime, nonce)
    assert request.runtime_fingerprint == runtime
    assert len(nonce) <= 128


def test_validator_exposes_runtime_contract_outside_legacy_state():
    client = TestClient(ValidatorServer().app)

    contract = client.get("/runtime-contract")
    health = client.get("/health")

    assert contract.status_code == 200
    assert contract.json()["telemetry_version"] == 2
    assert "fla_core_version" in contract.json()["validator_profile"]
    assert len(contract.json()["validator_profile"]["profile_hash"]) == 64
    assert health.json()["runtime_fingerprint"]["profile_hash"] == (
        contract.json()["validator_profile"]["profile_hash"]
    )
