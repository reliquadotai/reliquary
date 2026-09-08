from __future__ import annotations

import hashlib
import hmac

from reliquary.environment.evidence import (
    RUN_MANIFEST_DOMAIN,
    RUN_MANIFEST_SCHEMA,
    build_training_run_manifest_binding,
    sign_training_run_manifest,
    training_run_manifest_sha256,
    verify_signed_training_run_manifest,
)


_KEYS = {"validator-a": b"validator-a-secret", "validator-b": b"other"}


class _Hotkey:
    ss58_address = "validator-a"

    @staticmethod
    def sign(data: bytes) -> bytes:
        return hmac.new(_KEYS["validator-a"], data, hashlib.sha256).digest()


class _Wallet:
    hotkey = _Hotkey()


def _manifest():
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "status": "candidate",
        "run_id": "records-canary-001",
        "base_checkpoint": "Qwen/Qwen3-4B-Base@base",
        "tokenizer_revision": "tokenizer-rev",
        "protocol_profile_id": "qwen3-4b-reliquary-verifiable-v6-dev1",
        "generation_contract_sha256": "1" * 64,
        "environment_manifest_sha256": "2" * 64,
        "golden_fixture_sha256": "3" * 64,
        "software_revision": "4" * 40,
        "image_digest": "5" * 64,
        "environments": ["reliquaryverifiable_v1"],
        "windows": {"first": 100, "last": 109},
        "selected_payload_roots": ["6" * 64, "7" * 64],
        "optimizer": {"name": "adamw", "learning_rate": "1e-6"},
        "candidate_checkpoint": "org/reliquary-records@candidate",
    }


def _verify(signer: str, message: bytes, signature: bytes) -> bool:
    key = _KEYS.get(signer)
    if key is None:
        return False
    expected = hmac.new(key, message, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)


def test_run_manifest_signature_round_trip_and_digest_stability():
    manifest = _manifest()
    envelope = sign_training_run_manifest(manifest, wallet=_Wallet())
    assert verify_signed_training_run_manifest(envelope, verifier=_verify)
    assert envelope["attestation"]["manifest_sha256"] == (
        training_run_manifest_sha256(manifest)
    )
    assert build_training_run_manifest_binding(manifest).startswith(
        RUN_MANIFEST_DOMAIN.encode() + b"\0"
    )


def test_run_manifest_signature_rejects_payload_signer_and_domain_changes():
    envelope = sign_training_run_manifest(_manifest(), wallet=_Wallet())

    tampered_payload = {
        **envelope,
        "manifest": {**envelope["manifest"], "candidate_checkpoint": "other"},
    }
    assert not verify_signed_training_run_manifest(
        tampered_payload, verifier=_verify
    )

    wrong_signer = {
        **envelope,
        "attestation": {**envelope["attestation"], "signer": "validator-b"},
    }
    assert not verify_signed_training_run_manifest(wrong_signer, verifier=_verify)

    wrong_domain = {
        **envelope,
        "attestation": {**envelope["attestation"], "domain": "other-v1"},
    }
    assert not verify_signed_training_run_manifest(wrong_domain, verifier=_verify)
