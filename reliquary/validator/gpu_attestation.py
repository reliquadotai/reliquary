"""Strict NVIDIA NRAS detached EAT verification; no client-selected trust roots.

Supported wire profile: a single-GPU NRAS bundle, ES384/P-384, claims v3/v4.
The SDK's HS256 outer wrapper is deliberately NOT accepted. Keys come only from
the fixed production NRAS HTTPS JWKS, never jku/jwk/x5u in a miner's token.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

ISSUER = "https://nras.attestation.nvidia.com"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
MAX_BUNDLE_BYTES = 128 * 1024
MAX_JWT_BYTES = 64 * 1024
HEX_NONCE = re.compile(r"[0-9a-f]{64}\Z")


class AttestationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class VerifierUnavailable(RuntimeError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AttestationError("duplicate_json_key")
        result[key] = value
    return result


def strict_json(raw: bytes):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(AttestationError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AttestationError("malformed_json") from exc


class NvidiaKeys:
    """Bounded HTTPS fetch, short cache, throttled unknown-kid refresh, no stale use.

    One instance per validator process. The lock coalesces concurrent JWKS fetches;
    unknown key IDs cannot induce an unbounded stream of NVIDIA HTTP requests.
    """
    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic, client=None):
        import jwt  # optional extra, required only when admission is enabled

        self._jwt, self._clock, self._client = jwt, monotonic, client
        self._keys, self._expires, self._last_attempt = {}, 0.0, float("-inf")
        self._lock = threading.Lock()

    def __call__(self, kid: str):
        import httpx

        with self._lock:
            now = self._clock()
            if now < self._expires and kid in self._keys:
                return self._keys[kid]
            if now - self._last_attempt < 30:
                if now >= self._expires:
                    raise VerifierUnavailable("nvidia_keys_unavailable")
                raise AttestationError("unknown_signing_key")
            self._last_attempt = now
            try:
                def load(client):
                    with client.stream("GET", JWKS_URL) as response:
                        response.raise_for_status()
                        raw = bytearray()
                        for part in response.iter_bytes():
                            raw.extend(part)
                            if len(raw) > 2 * 1024 * 1024:
                                raise ValueError("JWKS too large")
                    data = strict_json(bytes(raw))
                    keys = data["keys"]
                    if not isinstance(keys, list) or not 1 <= len(keys) <= 512:
                        raise ValueError("invalid JWKS")
                    accepted = {}
                    for item in keys:
                        name = item.get("kid")
                        if (not isinstance(name, str) or len(name) > 256 or name in accepted or
                                item.get("kty") != "EC" or item.get("crv") != "P-384" or
                                item.get("alg", "ES384") != "ES384" or item.get("use", "sig") != "sig" or
                                "d" in item):
                            raise ValueError("invalid JWKS key")
                        accepted[name] = self._jwt.PyJWK.from_dict(item, algorithm="ES384").key
                    return accepted

                if self._client is None:
                    with httpx.Client(timeout=5.0, follow_redirects=False, trust_env=False) as client:
                        keys = load(client)
                else:
                    keys = load(self._client)
                self._keys, self._expires = keys, now + 300
            except Exception as exc:
                raise VerifierUnavailable("nvidia_keys_unavailable") from exc
            if kid not in self._keys:
                raise AttestationError("unknown_signing_key")
            return self._keys[kid]


@dataclass(frozen=True)
class VerifiedGpu:
    ueid: str
    hardware_model: str
    driver: str
    vbios: str
    claims_version: str
    issued_at: int
    expires_at: int
    proof_sha256: str


# NRAS v3/v4, strict booleans: strings like "true" do not pass.
REQUIRED_TRUE = (
    "x-nvidia-gpu-arch-check", "x-nvidia-gpu-attestation-report-parsed",
    "x-nvidia-gpu-attestation-report-signature-verified", "x-nvidia-gpu-attestation-report-nonce-match",
    "x-nvidia-gpu-attestation-report-cert-chain-fwid-match",
    *tuple(f"x-nvidia-gpu-{part}-rim-{check}" for part in ("driver", "vbios") for check in
           ("fetched", "schema-validated", "signature-verified", "measurements-available", "version-match")),
    "x-nvidia-gpu-vbios-index-no-conflict",
)
CERT_CLAIMS = tuple(f"x-nvidia-gpu-{part}" for part in
                    ("attestation-report-cert-chain", "driver-rim-cert-chain", "vbios-rim-cert-chain"))


class NvidiaVerifier:
    def __init__(self, *, keys=None):
        import jwt

        self.jwt, self.keys = jwt, keys if keys is not None else NvidiaKeys()

    def _decode(self, token, *, nonce: str, now: float, issued_after: float) -> dict:
        if not isinstance(token, str) or len(token) > MAX_JWT_BYTES:
            raise AttestationError("malformed_token")
        try:
            parts = token.split(".")
            if len(parts) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in parts):
                raise AttestationError("malformed_token")
            header, claims = [strict_json(base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))) for x in parts[:2]]
            if not isinstance(header, dict) or not isinstance(claims, dict):
                raise AttestationError("malformed_token")
            if (header.get("alg") != "ES384" or set(header) - {"alg", "kid", "typ"} or
                    not isinstance(header.get("kid"), str) or not 1 <= len(header["kid"]) <= 256):
                raise AttestationError("unsupported_token_header")
            key = self.keys(header["kid"])
            verified = self.jwt.decode(token, key=key, algorithms=["ES384"], issuer=ISSUER,
                                       options={"require": ["iss", "iat", "nbf", "exp", "eat_nonce"],
                                                "verify_exp": False, "verify_iat": False,
                                                "verify_nbf": False, "verify_aud": False})
            if verified != claims:
                raise AttestationError("malformed_token")
            if any(type(claims[k]) is not int for k in ("iat", "nbf", "exp")):
                raise AttestationError("invalid_token_time")
            if (claims["exp"] <= now or claims["nbf"] > now + 5 or claims["iat"] > now + 5 or
                    claims["iat"] < issued_after - 5 or not 0 < claims["exp"] - claims["iat"] <= 3600 or
                    claims["nbf"] > claims["exp"]):
                raise AttestationError("stale_token")
            if not isinstance(claims["eat_nonce"], str) or not hmac.compare_digest(claims["eat_nonce"], nonce):
                raise AttestationError("nonce_mismatch")
            return claims
        except (AttestationError, VerifierUnavailable):
            raise
        except Exception as exc:
            raise AttestationError("bad_token_signature_or_claims") from exc

    def verify(self, bundle, *, nonce: str, now: float, issued_after: float) -> VerifiedGpu:
        from reliquary.protocol.gpu_attestation import canonical

        if not HEX_NONCE.fullmatch(nonce):
            raise AttestationError("invalid_challenge_nonce")
        if len(canonical(bundle)) > MAX_BUNDLE_BYTES:
            raise AttestationError("bundle_too_large")
        if (not isinstance(bundle, list) or len(bundle) != 2 or not isinstance(bundle[0], list) or
                len(bundle[0]) != 2 or bundle[0][0] != "JWT" or not isinstance(bundle[1], dict) or
                set(bundle[1]) != {"GPU-0"}):
            raise AttestationError("single_gpu_nras_bundle_required")
        overall = self._decode(bundle[0][1], nonce=nonce, now=now, issued_after=issued_after)
        device_token = bundle[1]["GPU-0"]
        device = self._decode(device_token, nonce=nonce, now=now, issued_after=issued_after)
        if overall.get("x-nvidia-overall-att-result") is not True or overall.get("x-nvidia-ver") not in ("3.0", "4.0"):
            raise AttestationError("attestation_failed")
        digest = hashlib.sha256(device_token.encode("ascii")).hexdigest()
        if overall.get("submods") != {"GPU-0": ["DIGEST", ["SHA-256", digest]]}:
            raise AttestationError("detached_token_digest_mismatch")
        version = device.get("x-nvidia-gpu-claims-version", overall["x-nvidia-ver"])
        if version != overall["x-nvidia-ver"] or device.get("x-nvidia-device-type", "gpu") != "gpu":
            raise AttestationError("unsupported_device_claims")
        if (device.get("secboot") is not True or device.get("dbgstat") != "disabled" or
                device.get("measres") != "success" or any(device.get(k) is not True for k in REQUIRED_TRUE) or
                device.get("x-nvidia-attestation-warning") not in (None, False) or
                device.get("x-nvidia-error-details") or device.get("x-nvidia-mismatch-opaque-data-records")):
            raise AttestationError("gpu_security_policy_failed")
        for field in CERT_CLAIMS:
            cert = device.get(field)
            if (not isinstance(cert, dict) or cert.get("x-nvidia-cert-status") != "valid" or
                    cert.get("x-nvidia-cert-ocsp-status") != "good" or cert.get("x-nvidia-cert-revocation-reason") is not None):
                raise AttestationError("gpu_certificate_policy_failed")
            if any(k in cert and cert[k] is not True for k in
                   ("x-nvidia-cert-ocsp-nonce-matches", "x-nvidia-cert-ocsp-response-valid")):
                raise AttestationError("gpu_certificate_policy_failed")
            try:
                expiry = datetime.fromisoformat(cert["x-nvidia-cert-expiration-date"].replace("Z", "+00:00"))
                if expiry.tzinfo is None or expiry.timestamp() <= now:
                    raise ValueError("expired")
            except (KeyError, TypeError, ValueError) as exc:
                raise AttestationError("gpu_certificate_policy_failed") from exc
        values = [device.get(k) for k in ("ueid", "hwmodel", "x-nvidia-gpu-driver-version", "x-nvidia-gpu-vbios-version")]
        if any(not isinstance(v, str) or not 1 <= len(v) <= 256 or not v.isprintable() for v in values):
            raise AttestationError("missing_hardware_identity")
        return VerifiedGpu(*values, version, max(overall["iat"], device["iat"]),
                           min(overall["exp"], device["exp"]), hashlib.sha256(canonical(bundle)).hexdigest())
