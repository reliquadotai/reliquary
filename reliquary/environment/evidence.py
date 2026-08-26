"""Canonical, domain-separated evidence for environment training runs.

This module does not publish or promote a checkpoint.  It only creates a
portable attestation envelope that an operator can store beside evaluation
artifacts and verify before making a training or model-improvement claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from typing import Any, Protocol


RUN_MANIFEST_SCHEMA = "reliquary/training-run-manifest/v1"
RUN_MANIFEST_DOMAIN = "reliquary-training-run-manifest-v1"
_HEX_64_FIELDS = (
    "generation_contract_sha256",
    "environment_manifest_sha256",
    "golden_fixture_sha256",
    "image_digest",
)


class _HotkeyLike(Protocol):
    ss58_address: str

    def sign(self, data: bytes) -> bytes: ...


class _WalletLike(Protocol):
    hotkey: _HotkeyLike


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Validate and serialize a run manifest with stable JSON bytes."""

    validate_training_run_manifest(manifest)
    return json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def training_run_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def validate_training_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject incomplete or ambiguous candidate-lineage evidence."""

    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported training run manifest schema")
    required_strings = (
        "run_id",
        "base_checkpoint",
        "tokenizer_revision",
        "protocol_profile_id",
        "candidate_checkpoint",
    )
    for field in required_strings:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"training run manifest requires {field}")
    status = manifest.get("status")
    if status not in ("candidate", "promoted", "rejected"):
        raise ValueError("training run manifest has invalid status")
    for field in _HEX_64_FIELDS:
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"training run manifest requires 64-hex {field}")
    software_revision = manifest.get("software_revision")
    if (
        not isinstance(software_revision, str)
        or len(software_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in software_revision
        )
    ):
        raise ValueError("training run manifest requires 40-hex software_revision")
    windows = manifest.get("windows")
    if not isinstance(windows, Mapping):
        raise ValueError("training run manifest requires windows")
    try:
        first = int(windows["first"])
        last = int(windows["last"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("training run manifest has invalid window range") from exc
    if first < 0 or last < first:
        raise ValueError("training run manifest has invalid window range")
    roots = manifest.get("selected_payload_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("training run manifest requires selected payload roots")
    if any(
        not isinstance(root, str)
        or len(root) != 64
        or any(character not in "0123456789abcdef" for character in root)
        for root in roots
    ):
        raise ValueError("selected payload roots must be lowercase 64-hex")
    optimizer = manifest.get("optimizer")
    if not isinstance(optimizer, Mapping) or not optimizer:
        raise ValueError("training run manifest requires optimizer settings")
    environments = manifest.get("environments")
    if not isinstance(environments, list) or not environments:
        raise ValueError("training run manifest requires environments")
    if len(set(environments)) != len(environments) or any(
        not isinstance(name, str) or not name for name in environments
    ):
        raise ValueError("training run manifest environments must be unique names")


def build_training_run_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    domain: str = RUN_MANIFEST_DOMAIN,
) -> bytes:
    if domain != RUN_MANIFEST_DOMAIN:
        raise ValueError("unsupported training run manifest signature domain")
    return domain.encode("ascii") + b"\0" + canonical_manifest_bytes(manifest)


def sign_training_run_manifest(
    manifest: Mapping[str, Any],
    *,
    wallet: _WalletLike,
) -> dict[str, Any]:
    """Return a self-contained signed envelope without mutating the input."""

    hotkey = getattr(wallet, "hotkey", None)
    signer = getattr(hotkey, "ss58_address", "")
    sign = getattr(hotkey, "sign", None)
    if not signer or not callable(sign):
        raise TypeError("wallet must provide hotkey.ss58_address and hotkey.sign")
    message = build_training_run_manifest_binding(manifest)
    signature = bytes(sign(message)).hex()
    return {
        "manifest": dict(manifest),
        "attestation": {
            "domain": RUN_MANIFEST_DOMAIN,
            "signer": str(signer),
            "signature": signature,
            "manifest_sha256": training_run_manifest_sha256(manifest),
        },
    }


def verify_signed_training_run_manifest(
    envelope: Mapping[str, Any],
    *,
    verifier: Callable[[str, bytes, bytes], bool] | None = None,
) -> bool:
    """Verify domain, digest, signer and payload; fail closed on any error."""

    try:
        manifest = envelope["manifest"]
        attestation = envelope["attestation"]
        if not isinstance(manifest, Mapping) or not isinstance(
            attestation, Mapping
        ):
            return False
        if attestation.get("domain") != RUN_MANIFEST_DOMAIN:
            return False
        signer = str(attestation["signer"])
        signature = bytes.fromhex(str(attestation["signature"]))
        if not signer or not signature:
            return False
        if attestation.get("manifest_sha256") != training_run_manifest_sha256(
            manifest
        ):
            return False
        message = build_training_run_manifest_binding(manifest)
        if verifier is not None:
            return bool(verifier(signer, message, signature))
        try:
            import bittensor as bt
        except ImportError:
            return False
        return bool(
            bt.Keypair(ss58_address=signer).verify(
                data=message,
                signature=signature,
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


__all__ = [
    "RUN_MANIFEST_DOMAIN",
    "RUN_MANIFEST_SCHEMA",
    "build_training_run_manifest_binding",
    "canonical_manifest_bytes",
    "sign_training_run_manifest",
    "training_run_manifest_sha256",
    "validate_training_run_manifest",
    "verify_signed_training_run_manifest",
]
