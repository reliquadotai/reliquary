"""Wallet-free wire binding for RL GPU admission. It is NOT a GPU signing key.

The hotkey signs the complete request; NVIDIA signs the detached EAT. Their common
nonce binds a fresh hardware report to this validator, run and hotkey. It does not
bind subsequent model execution to an enclave or prevent an attestation relay.
"""
from __future__ import annotations

import hashlib
import json

DOMAIN = b"reliquary/rl-gpu-admission/v1\x00"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def gpu_request_binding(payload: dict, *, validator_hotkey: str, run: str, path: str) -> bytes:
    """Sign every field except the signature, including declarations and the EAT."""
    return hashlib.sha256(DOMAIN + canonical({"validator": validator_hotkey, "run": run, "path": path,
                                             "request": {k: v for k, v in payload.items() if k != "signature"}})).digest()


def challenge_nonce(*, challenge_id: str, secret: str, hotkey: str, validator: str, run: str) -> str:
    return hashlib.sha256(DOMAIN + canonical({"id": challenge_id, "secret": secret, "hotkey": hotkey,
                                             "validator": validator, "run": run})).hexdigest()
