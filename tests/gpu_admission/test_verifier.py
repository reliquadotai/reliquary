import base64
import copy
import hashlib
import json

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from reliquary.validator.gpu_attestation import (
    CERT_CLAIMS, JWKS_URL, REQUIRED_TRUE, AttestationError, NvidiaKeys, VerifierUnavailable, strict_json,
)
from .conftest import NOW

NONCE = "ab" * 32


def verify(authority, bundle):
    return authority.verifier.verify(bundle, nonce=NONCE, now=NOW, issued_after=NOW)


def test_accepts_real_ecdsa_signatures_and_extracts_only_attested_fields(authority):
    proof = authority.bundle(NONCE)
    result = verify(authority, proof)
    assert result.ueid == "synthetic-h100-identity-1"
    assert result.driver == "580.105.08" and result.hardware_model == "GH100 TEST ONLY"
    assert result.expires_at == NOW + 3600


@pytest.mark.parametrize("field", REQUIRED_TRUE)
@pytest.mark.parametrize("value", [False, "true", None])
def test_every_critical_measurement_is_fail_closed(authority, field, value):
    with pytest.raises(AttestationError, match="gpu_security_policy_failed"):
        verify(authority, authority.bundle(NONCE, device={field: value}))


@pytest.mark.parametrize("device", [
    {"secboot": False}, {"secboot": "true"}, {"dbgstat": "enabled"}, {"dbgstat": None},
    {"measres": "fail"}, {"x-nvidia-attestation-warning": "CERT_HOLD"},
    {"x-nvidia-error-details": {"error": "failed"}}, {"x-nvidia-mismatch-opaque-data-records": [1]},
    {"ueid": None}, {"hwmodel": ""}, {"x-nvidia-gpu-driver-version": None},
    {"x-nvidia-device-type": "nvswitch"}, {"x-nvidia-gpu-claims-version": "9.0"},
])
def test_unsafe_or_incomplete_hardware_is_refused(authority, device):
    with pytest.raises(AttestationError):
        verify(authority, authority.bundle(NONCE, device=device))


@pytest.mark.parametrize("field", CERT_CLAIMS)
@pytest.mark.parametrize("change", [
    {"x-nvidia-cert-ocsp-status": "revoked"}, {"x-nvidia-cert-status": "unknown"},
    {"x-nvidia-cert-revocation-reason": "CERT_HOLD"},
    {"x-nvidia-cert-expiration-date": "2000-01-01T00:00:00Z"},
    {"x-nvidia-cert-expiration-date": "2099-01-01T00:00:00"},
    {"x-nvidia-cert-ocsp-nonce-matches": False}, {"x-nvidia-cert-ocsp-response-valid": False},
])
def test_revoked_unknown_or_expired_certificates_fail(authority, field, change):
    cert = {"x-nvidia-cert-status": "valid", "x-nvidia-cert-ocsp-status": "good",
            "x-nvidia-cert-expiration-date": "2099-01-01T00:00:00Z", **change}
    with pytest.raises(AttestationError, match="certificate"):
        verify(authority, authority.bundle(NONCE, device={field: cert}))


@pytest.mark.parametrize("change", [
    {"eat_nonce": "cd" * 32}, {"exp": NOW}, {"iat": NOW + 30}, {"iat": NOW - 300},
    {"nbf": NOW + 30}, {"exp": NOW + 7200}, {"iat": True},
    {"iss": "https://attacker.example"},
])
@pytest.mark.parametrize("target", ["device", "overall"])
def test_both_tokens_require_freshness_nonce_and_nvidia_issuer(authority, change, target):
    with pytest.raises(AttestationError):
        verify(authority, authority.bundle(NONCE, **{target: change}))


def test_forged_and_edited_tokens_are_rejected(authority):
    other = ec.generate_private_key(ec.SECP384R1())
    with pytest.raises(AttestationError):
        verify(authority, authority.bundle(NONCE, key=other))
    with pytest.raises(AttestationError):
        verify(authority, authority.bundle(NONCE, overall={"x-nvidia-overall-att-result": False}))
    bundle = authority.bundle(NONCE)
    bundle[0][1] = bundle[0][1][:-10] + "aaaaaaaaaa"
    with pytest.raises(AttestationError):
        verify(authority, bundle)


def test_detached_gpu_cannot_be_swapped_even_with_valid_signature(authority):
    bundle = authority.bundle(NONCE)
    bundle[1] = authority.bundle(NONCE, device={"ueid": "other-device"})[1]
    with pytest.raises(AttestationError, match="digest"):
        verify(authority, bundle)


@pytest.mark.parametrize("header", [{"jku": "https://evil.test/keys"}, {"jwk": {}}, {"crit": ["b64"]},
                                    {"x5u": "file:///etc/passwd"}])
def test_client_cannot_choose_trust_roots(authority, header):
    with pytest.raises(AttestationError, match="header"):
        verify(authority, authority.bundle(NONCE, headers=header))


def test_unsigned_hmac_and_sdk_outer_wrappers_do_not_pass(authority):
    bundle = authority.bundle(NONCE)
    for alg, key in (("none", None), ("HS256", "test-only-sdk-wrapper-key-never-trusted")):
        bad = copy.deepcopy(bundle)
        bad[0][1] = jwt.encode({"iss": "NV-Attestation-SDK"}, key, algorithm=alg, headers={"kid": "test-only"})
        with pytest.raises(AttestationError):
            verify(authority, bad)
    with pytest.raises(AttestationError):
        verify(authority, [bundle[0], {"REMOTE_GPU_CLAIMS": bundle}])
    with pytest.raises(AttestationError):
        verify(authority, {"verified": True, "gpu": "H100"})


def test_multi_gpu_bundles_are_refused_in_single_gpu_protocol(authority):
    bundle = authority.bundle(NONCE)
    bundle[1]["GPU-1"] = bundle[1]["GPU-0"]
    with pytest.raises(AttestationError):
        verify(authority, bundle)


def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(authority):
    for raw in (b'{"secboot":false,"secboot":true}', b'{"value":NaN}'):
        with pytest.raises(AttestationError):
            strict_json(raw)
    bundle = authority.bundle(NONCE)
    token = bundle[1]["GPU-0"]
    h, p, _ = token.split(".")
    raw = base64.urlsafe_b64decode(p + "===")
    payload = b'{"secboot":false,' + raw[1:]
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    message = f"{h}.{encoded}".encode()
    signature = jwt.get_algorithm_by_name("ES384").sign(message, authority.key)
    bundle[1]["GPU-0"] = message.decode() + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    with pytest.raises(AttestationError, match="duplicate"):
        verify(authority, bundle)


def jwk(authority):
    value = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(authority.key.public_key()))
    return {**value, "kid": "test-only"}


def test_production_jwks_url_cache_rotation_and_unknown_kid_limit(authority):
    calls, clock = [], [0.0]
    def serve(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"keys": [jwk(authority)]})
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        keys = NvidiaKeys(monotonic=lambda: clock[0], client=client)
        assert keys("test-only") is keys("test-only")
        for _ in range(8):
            with pytest.raises(AttestationError):
                keys("attacker-controlled-kid")
        assert calls == [JWKS_URL]
        clock[0] = 301
        keys("test-only")
        assert calls == [JWKS_URL, JWKS_URL]


@pytest.mark.parametrize("response", [
    httpx.Response(503), httpx.Response(302, headers={"location": "https://evil.test"}),
    httpx.Response(200, json={"keys": []}), httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1)),
])
def test_jwks_outage_redirect_and_oversize_never_fail_open(response):
    with httpx.Client(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(VerifierUnavailable):
            NvidiaKeys(client=client)("test")


def test_private_or_wrong_curve_keys_are_rejected(authority):
    bad = jwk(authority)
    bad["d"] = "private-key-material-is-never-a-trust-anchor"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"keys": [bad]}))) as client:
        with pytest.raises(VerifierUnavailable):
            NvidiaKeys(client=client)("test-only")


def test_expired_cached_keys_are_not_used_during_nvidia_outage(authority):
    clock, available, calls = [0.0], [True], []
    def serve(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"keys": [jwk(authority)]}) if available[0] else httpx.Response(503)
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        keys = NvidiaKeys(monotonic=lambda: clock[0], client=client)
        keys("test-only")
        available[0] = False
        clock[0] = 299
        keys("test-only")       # still within the documented five-minute cache
        clock[0] = 301
        with pytest.raises(VerifierUnavailable):
            keys("test-only")
        with pytest.raises(VerifierUnavailable):
            keys("test-only")   # no fallback, no retry storm
        assert len(calls) == 2
