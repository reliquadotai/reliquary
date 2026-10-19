"""Synthetic identities in RAM only. No real wallet, NVIDIA secrets or GPU needed."""
import hashlib

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from reliquary.validator.gpu_attestation import CERT_CLAIMS, ISSUER, REQUIRED_TRUE, NvidiaVerifier

NOW = 1_800_000_000


@pytest.fixture
def now():
    return [float(NOW)]


@pytest.fixture
def authority():
    class Authority:
        key = ec.generate_private_key(ec.SECP384R1())

        def token(self, claims, *, key=None, headers=None):
            return jwt.encode(claims, self.key if key is None else key, algorithm="ES384",
                              headers={"kid": "test-only", **(headers or {})})

        def bundle(self, nonce, *, device=None, overall=None, key=None, headers=None, **kwargs):
            times = {"iss": ISSUER, "iat": NOW, "nbf": NOW, "exp": NOW + 3600, "eat_nonce": nonce}
            claims = {**times, **{k: True for k in REQUIRED_TRUE},
                      **{k: {"x-nvidia-cert-status": "valid", "x-nvidia-cert-ocsp-status": "good",
                             "x-nvidia-cert-expiration-date": "2099-01-01T00:00:00Z",
                             "x-nvidia-cert-revocation-reason": None} for k in CERT_CLAIMS},
                      "secboot": True, "dbgstat": "disabled", "measres": "success",
                      "ueid": "synthetic-h100-identity-1", "hwmodel": "GH100 TEST ONLY",
                      "x-nvidia-gpu-driver-version": "580.105.08", "x-nvidia-gpu-vbios-version": "test-vbios",
                      **(device or {})}
            dev = self.token(claims, key=key, headers=headers)
            summary = {**times, "sub": "NVIDIA-PLATFORM-ATTESTATION", "x-nvidia-ver": "3.0",
                       "x-nvidia-overall-att-result": True,
                       "submods": {"GPU-0": ["DIGEST", ["SHA-256", hashlib.sha256(dev.encode()).hexdigest()]]},
                       **(overall or {})}
            return [["JWT", self.token(summary, **kwargs)], {"GPU-0": dev}]

        @property
        def verifier(self):
            return NvidiaVerifier(keys=lambda kid: self.key.public_key())

    return Authority()
