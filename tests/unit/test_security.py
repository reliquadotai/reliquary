"""Security tests — verify that known attack vectors are blocked.

Each test documents a specific attack scenario, explains why it would
work on unpatched code, and asserts that the patched code prevents it.

No GPU required — models are mocked where needed.
"""

import hashlib
import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

from reliquary.constants import CHALLENGE_K


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

VALIDATOR_RANDOMNESS = "aa" * 32
MINER_RANDOMNESS = "bb" * 32
HIDDEN_DIM = 128


def _make_mock_model():
    param = torch.zeros(1)
    model = MagicMock()
    model.parameters.return_value = iter([param])
    return model


# ══════════════════════════════════════════════════════════════════════
# FAILLE #1 — Miner cannot control the randomness used by the verifier
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #2 — Commitment count must match token count
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #3 — Minimum challenge positions enforced
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #4 — Prompt verification is mandatory
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Note: prompt verification moved into the WindowBatcher in the GRPO
# refactor. The validator now derives prompts deterministically from the
# beacon, so the "miner uses a crafted prompt" attack surface is gone —
# the slot's expected prompt is owned by the validator, not declared by
# the miner. See tests/unit/test_batcher.py::test_invalid_prompt_rejected.
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #7 — PRF uses SHAKE256 only, no dual path
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #11 — PRF length-prefix prevents collisions
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #3+4 — Drand fallback defaults to False
# ══════════════════════════════════════════════════════════════════════


class TestDrandFallbackDisabled:
    """Attack: use_fallback=True would silently substitute mock randomness
    on network failure. Fix: default is False.
    """

    def test_get_drand_beacon_default(self):
        from reliquary.infrastructure.drand import get_drand_beacon
        assert inspect.signature(get_drand_beacon).parameters["use_fallback"].default is False

    def test_get_beacon_default(self):
        from reliquary.infrastructure.drand import get_beacon
        assert inspect.signature(get_beacon).parameters["use_fallback"].default is False


# ══════════════════════════════════════════════════════════════════════
# FAILLE #4 — Beacon signature verification
# ══════════════════════════════════════════════════════════════════════


class TestBeaconSignatureVerification:
    """Attack: MITM injects fake beacon without valid BLS signature.
    Fix: verify_beacon_signature rejects missing/invalid signatures.
    Without a BLS library (blst), verification must fail-closed (return False).
    """

    def test_rejects_none_signature(self):
        from reliquary.infrastructure.drand import verify_beacon_signature
        assert verify_beacon_signature("abc", 1, "dead" * 8, None) is False

    def test_rejects_empty_signature(self):
        from reliquary.infrastructure.drand import verify_beacon_signature
        assert verify_beacon_signature("abc", 1, "dead" * 8, "") is False

    def test_rejects_wrong_randomness(self):
        from reliquary.infrastructure.drand import verify_beacon_signature
        assert verify_beacon_signature("abc", 1, "bb" * 32, "aa" * 48) is False

    def test_no_bls_library_fails_closed(self):
        """Without blst, verification must return False — not fall through
        to a hash-based check that any attacker can satisfy."""
        from reliquary.infrastructure.drand import verify_beacon_signature
        sig_bytes = bytes.fromhex("cc" * 48)
        crafted_rand = hashlib.sha256(sig_bytes).hexdigest()
        # Even though SHA256(sig) == randomness, this MUST reject without a
        # matching cross-check from the independent bittensor-drand relay.
        assert verify_beacon_signature("abc", 1, crafted_rand, "cc" * 48) is False

    def test_correct_dst_for_quicknet(self):
        """The DST must use BLS12381G1, not BN254G1."""
        import reliquary.infrastructure.drand as drand_mod
        source = inspect.getsource(drand_mod.verify_beacon_signature)
        assert "BLS12381G1" in source or "BLS_SIG_BLS12381G1" in source
        assert "BN254G1" not in source


# ══════════════════════════════════════════════════════════════════════
# Sketch forgery resistance (statistical)
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# FAILLE #6 — Proof version consistency in signature verification
# ══════════════════════════════════════════════════════════════════════


class TestProofVersionConsistency:
    """verify_commit_signature must only accept the current GRAIL_PROOF_VERSION."""

    def test_rejects_v4(self):
        from reliquary.protocol.signatures import verify_commit_signature
        commit = {
            "tokens": [1, 2, 3],
            "commitments": [{"sketch": 0}],
            "proof_version": "v4",
            "signature": "aa" * 64,
            "beacon": {"randomness": "bb" * 32},
            "model": {"name": "test", "layer_index": -1},
        }
        assert verify_commit_signature(commit, "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY") is False

    def test_rejects_unknown_version(self):
        from reliquary.protocol.signatures import verify_commit_signature
        commit = {
            "tokens": [1, 2, 3],
            "commitments": [{"sketch": 0}],
            "proof_version": "v99",
            "signature": "aa" * 64,
            "beacon": {"randomness": "bb" * 32},
            "model": {"name": "test", "layer_index": -1},
        }
        assert verify_commit_signature(commit, "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY") is False


# ══════════════════════════════════════════════════════════════════════
# FAILLE #8 — Token sequence length check
# ══════════════════════════════════════════════════════════════════════


