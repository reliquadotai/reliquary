"""Security tests — verify that known attack vectors are blocked.

Each test documents a specific attack scenario, explains why it would
work on unpatched code, and asserts that the patched code prevents it.

No GPU required — models are mocked where needed.
"""

import hashlib
import hmac as hmac_mod
import inspect
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from reliquary.constants import CHALLENGE_K, PRIME_Q
from reliquary.protocol.crypto import indices_from_root, prf
from reliquary.protocol.grail_verifier import GRAILVerifier


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


class TestMinerCannotControlRandomness:
    """Attack: miner embeds their own beacon.randomness inside the commit
    so the verifier uses it for index selection (predicting challenged
    positions). Fix: verifier always uses window_randomness.
    """

    @patch("reliquary.shared.forward.forward_single_layer")
    @patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM)
    def test_verifier_ignores_commit_beacon_randomness(self, _rhs, mock_fwd):
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        tokens = list(range(seq_len))
        fake_hidden = torch.randn(1, seq_len, HIDDEN_DIM)
        mock_fwd.return_value = (fake_hidden, torch.zeros(1, seq_len, 10))

        commitments = [{"sketch": 0}] * seq_len
        commit = {
            "tokens": tokens,
            "commitments": commitments,
            "beacon": {"randomness": MINER_RANDOMNESS},
        }

        # Spy on indices_from_root to see which randomness was used
        with patch(
            "reliquary.protocol.crypto.indices_from_root",
            wraps=indices_from_root,
        ) as spy:
            verify_commitment_proofs(commit, _make_mock_model(), VALIDATOR_RANDOMNESS)
            spy.assert_called_once()
            used_randomness = spy.call_args[0][1]
            assert used_randomness == VALIDATOR_RANDOMNESS
            assert used_randomness != MINER_RANDOMNESS

    def test_different_randomness_produces_different_r_vecs(self):
        """The r_vec derived from miner's randomness must differ from the
        validator's r_vec — this is the core of the defence. If they're
        different, the sketches computed by validator won't match."""
        verifier = GRAILVerifier(hidden_dim=HIDDEN_DIM)
        r_miner = verifier.generate_r_vec(MINER_RANDOMNESS)
        r_validator = verifier.generate_r_vec(VALIDATOR_RANDOMNESS)
        assert not torch.equal(r_miner, r_validator), (
            "Different randomness must produce different r_vecs"
        )

    def test_different_randomness_produces_different_indices(self):
        """Different randomness must select different challenge positions."""
        tokens = list(range(64))
        k = min(CHALLENGE_K, 64)
        idx_miner = indices_from_root(tokens, MINER_RANDOMNESS, 64, k)
        idx_validator = indices_from_root(tokens, VALIDATOR_RANDOMNESS, 64, k)
        assert idx_miner != idx_validator, (
            "Different randomness must select different challenge indices"
        )


# ══════════════════════════════════════════════════════════════════════
# FAILLE #2 — Commitment count must match token count
# ══════════════════════════════════════════════════════════════════════


class TestCommitmentCountMustMatchTokens:
    """Attack: miner submits fewer commitments than tokens. Challenged
    positions beyond the array are silently skipped.
    Fix: len(commitments) != len(tokens) → CommitModel raises ValidationError
    (now enforced by _commitments_len_matches_tokens Pydantic validator on
    CommitModel, before verify_commitment_proofs is called).
    """

    def _make_base_commit(self, tokens, commitments):
        """Build a minimal commit dict with the given tokens and commitments."""
        from reliquary.constants import CHALLENGE_K
        seq_len = len(tokens)
        completion_length = max(seq_len - 4, 1)
        return {
            "tokens": tokens,
            "commitments": commitments,
            "proof_version": "v6",
            "model": {"name": "test-model", "layer_index": 6},
            "signature": "ab" * 32,
            "beacon": {"randomness": "cd" * 16},
            "rollout": {
                "prompt_length": min(4, seq_len - 1),
                "completion_length": completion_length,
                "success": False,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": [0.0] * seq_len,
            },
        }

    def test_fewer_commitments_rejected(self):
        from pydantic import ValidationError
        from reliquary.protocol.submission import CommitModel

        commit = self._make_base_commit(list(range(100)), [{"sketch": 0}] * 10)
        with pytest.raises(ValidationError):
            CommitModel.model_validate(commit)

    def test_more_commitments_rejected(self):
        from pydantic import ValidationError
        from reliquary.protocol.submission import CommitModel

        commit = self._make_base_commit(list(range(10)), [{"sketch": 0}] * 100)
        with pytest.raises(ValidationError):
            CommitModel.model_validate(commit)

    def test_empty_commitments_rejected(self):
        from pydantic import ValidationError
        from reliquary.protocol.submission import CommitModel

        commit = self._make_base_commit(list(range(50)), [])
        with pytest.raises(ValidationError):
            CommitModel.model_validate(commit)

    @patch("reliquary.shared.forward.forward_single_layer")
    @patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM)
    def test_matching_count_proceeds_to_verification(self, _rhs, mock_fwd):
        """Equal counts should reach the forward pass (not short-circuit)."""
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        mock_fwd.return_value = (torch.randn(1, seq_len, HIDDEN_DIM), torch.zeros(1, seq_len, 10))
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}
        verify_commitment_proofs(commit, _make_mock_model(), "aabb")
        mock_fwd.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# FAILLE #3 — Minimum challenge positions enforced
# ══════════════════════════════════════════════════════════════════════


class TestMinimumChallengesEnforced:
    """Attack: if only 1 position is checked and it passes, proof was
    accepted. Fix: checked must equal expected_challenges.
    """

    @patch("reliquary.shared.forward.forward_single_layer")
    @patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM)
    def test_too_few_checks_fails_even_if_all_pass(self, _rhs, mock_fwd):
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        mock_fwd.return_value = (torch.randn(1, seq_len, HIDDEN_DIM), torch.zeros(1, seq_len, 10))
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}

        with patch("reliquary.protocol.crypto.indices_from_root", return_value=[0, 1]), \
             patch("reliquary.protocol.grail_verifier.GRAILVerifier.verify_commitment", return_value=(True, {})):
            res = verify_commitment_proofs(
                commit, _make_mock_model(), VALIDATOR_RANDOMNESS
            )
            assert res.checked == 2
            assert res.passed == 2
            assert res.all_passed is False  # 2 < min(32, 64) = 32

    @patch("reliquary.shared.forward.forward_single_layer")
    @patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM)
    def test_full_checks_can_pass(self, _rhs, mock_fwd):
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        mock_fwd.return_value = (torch.randn(1, seq_len, HIDDEN_DIM), torch.zeros(1, seq_len, 10))
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}
        expected = min(CHALLENGE_K, seq_len)

        with patch("reliquary.protocol.crypto.indices_from_root", return_value=list(range(expected))), \
             patch("reliquary.protocol.grail_verifier.GRAILVerifier.verify_commitment", return_value=(True, {})):
            res = verify_commitment_proofs(
                commit, _make_mock_model(), VALIDATOR_RANDOMNESS
            )
            assert res.checked == expected
            assert res.all_passed is True

    @patch("reliquary.shared.forward.forward_single_layer")
    @patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM)
    def test_one_failure_rejects(self, _rhs, mock_fwd):
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        mock_fwd.return_value = (torch.randn(1, seq_len, HIDDEN_DIM), torch.zeros(1, seq_len, 10))
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}
        expected = min(CHALLENGE_K, seq_len)

        call_count = [0]

        def verify_side(*a, **kw):
            call_count[0] += 1
            return (False, {}) if call_count[0] == expected else (True, {})

        with patch("reliquary.protocol.crypto.indices_from_root", return_value=list(range(expected))), \
             patch("reliquary.protocol.grail_verifier.GRAILVerifier.verify_commitment", side_effect=verify_side):
            res = verify_commitment_proofs(
                commit, _make_mock_model(), VALIDATOR_RANDOMNESS
            )
            assert res.all_passed is False
            assert res.passed == expected - 1


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


class TestPrfSinglePath:
    """Attack: SHAKE256 and SHA256-counter produce different outputs.
    If miner and validator take different paths, proofs diverge.
    Fix: SHAKE256 only, no fallback.
    """

    def test_matches_manual_shake256(self):
        label, part = b"test", b"data"
        shake = hashlib.shake_256()
        shake.update(len(label).to_bytes(4, "big"))
        shake.update(label)
        shake.update(len(part).to_bytes(4, "big"))
        shake.update(part)
        expected = shake.digest(32)
        assert prf(b"test", b"data", out_bytes=32) == expected

    def test_deterministic(self):
        for _ in range(100):
            assert prf(b"a", b"b", out_bytes=32) == prf(b"a", b"b", out_bytes=32)


# ══════════════════════════════════════════════════════════════════════
# FAILLE #11 — PRF length-prefix prevents collisions
# ══════════════════════════════════════════════════════════════════════


class TestPrfDomainSeparation:
    """Attack: with || separator, different part splits could collide.
    Fix: length-prefixed encoding.
    """

    def test_separator_in_data_no_collision(self):
        assert prf(b"L", b"a||b", out_bytes=32) != prf(b"L", b"a", b"", b"b", out_bytes=32)

    def test_different_split_points(self):
        assert prf(b"L", b"abc", b"def", out_bytes=32) != prf(b"L", b"ab", b"cdef", out_bytes=32)

    def test_single_vs_multi_part(self):
        assert prf(b"L", b"abcdef", out_bytes=32) != prf(b"L", b"abc", b"def", out_bytes=32)

    def test_empty_part_matters(self):
        assert prf(b"L", b"data", out_bytes=32) != prf(b"L", b"data", b"", out_bytes=32)


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
        with patch("reliquary.infrastructure.drand._fetch_chain_pubkey", return_value=b"\x00" * 48):
            assert verify_beacon_signature("abc", 1, "bb" * 32, "aa" * 48) is False

    def test_no_bls_library_fails_closed(self):
        """Without blst, verification must return False — not fall through
        to a hash-based check that any attacker can satisfy."""
        from reliquary.infrastructure.drand import verify_beacon_signature
        sig_bytes = bytes.fromhex("cc" * 48)
        crafted_rand = hashlib.sha256(sig_bytes).hexdigest()
        with patch("reliquary.infrastructure.drand._fetch_chain_pubkey", return_value=b"\x01" * 48):
            # Even though SHA256(sig) == randomness, this MUST reject
            assert verify_beacon_signature("abc", 1, crafted_rand, "cc" * 48) is False

    def test_correct_dst_for_quicknet(self):
        """The DST must use BLS12381G1, not BN254G1."""
        import reliquary.infrastructure.drand as drand_mod
        source = inspect.getsource(drand_mod.verify_beacon_signature)
        assert "BLS12381G1" in source or "BLS_SIG_BLS12381G1" in source
        assert "BN254G1" not in source


# ══════════════════════════════════════════════════════════════════════
# FAILLE #10 — Storage HMAC integrity
# ══════════════════════════════════════════════════════════════════════


class TestStorageHmacIntegrity:
    """Attack: entity with S3 write access modifies used_indices.json.gz.
    Fix: HMAC keyed to R2_SECRET_ACCESS_KEY detects tampering.
    """

    @patch.dict(os.environ, {"R2_SECRET_ACCESS_KEY": "test-secret"})
    def test_hmac_deterministic(self):
        from reliquary.infrastructure.storage import _compute_state_hmac
        assert _compute_state_hmac(b"data") == _compute_state_hmac(b"data")

    @patch.dict(os.environ, {"R2_SECRET_ACCESS_KEY": "test-secret"})
    def test_hmac_changes_with_data(self):
        from reliquary.infrastructure.storage import _compute_state_hmac
        assert _compute_state_hmac(b"data1") != _compute_state_hmac(b"data2")

    def test_hmac_changes_with_secret(self):
        from reliquary.infrastructure.storage import _compute_state_hmac
        with patch.dict(os.environ, {"R2_SECRET_ACCESS_KEY": "secret_A"}):
            a = _compute_state_hmac(b"payload")
        with patch.dict(os.environ, {"R2_SECRET_ACCESS_KEY": "secret_B"}):
            b = _compute_state_hmac(b"payload")
        assert a != b

    @patch.dict(os.environ, {"R2_SECRET_ACCESS_KEY": "known"})
    def test_key_derivation_uses_domain_separator(self):
        from reliquary.infrastructure.storage import _state_hmac_key
        expected = hashlib.sha256(b"reliquary-state-hmac|" + b"known").digest()
        assert _state_hmac_key() == expected


# ══════════════════════════════════════════════════════════════════════
# FAILLE #12 — Validator state isolation
# ══════════════════════════════════════════════════════════════════════


class TestValidatorStateIsolation:
    """Attack: multiple validators overwrite each other's state file.
    Fix: state files are keyed by validator hotkey.
    """

    def test_load_accepts_validator_id(self):
        from reliquary.infrastructure.storage import load_used_indices
        assert "validator_id" in inspect.signature(load_used_indices).parameters

    def test_save_accepts_validator_id(self):
        from reliquary.infrastructure.storage import save_used_indices
        assert "validator_id" in inspect.signature(save_used_indices).parameters


# ══════════════════════════════════════════════════════════════════════
# Sketch forgery resistance (statistical)
# ══════════════════════════════════════════════════════════════════════


class TestSketchForgeryResistance:
    """Random sketch values must fail verification with overwhelming
    probability. Tolerance of 6000 vs PRIME_Q ~2^31 gives ~0.0006%
    per position.
    """

    def test_random_sketches_fail(self):
        import random
        torch.manual_seed(42)
        hidden = torch.randn(64, 256)
        verifier = GRAILVerifier(hidden_dim=256)
        r_vec = verifier.generate_r_vec("aabbccdd")

        rng = random.Random(99)
        failures = sum(
            1 for i in range(64)
            if not verifier.verify_commitment(
                hidden[i], {"sketch": rng.randint(0, PRIME_Q - 1)}, r_vec, 64, i
            )[0]
        )
        assert failures >= 60, f"Only {failures}/64 random sketches failed"

    def test_one_corrupted_position_rejects_proof(self):
        """Corrupting a single challenged commitment must reject the whole proof."""
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        torch.manual_seed(42)
        hidden = torch.randn(1, seq_len, HIDDEN_DIM)
        randomness = "aabbccddee112233"
        tokens = list(range(seq_len))

        verifier = GRAILVerifier(hidden_dim=HIDDEN_DIM)
        r_vec = verifier.generate_r_vec(randomness)
        commitments = verifier.create_commitments_batch(hidden[0], r_vec)

        # Corrupt one challenged position
        challenged = indices_from_root(tokens, randomness, seq_len, min(CHALLENGE_K, seq_len))
        idx = challenged[0]
        commitments[idx] = {"sketch": (commitments[idx]["sketch"] + PRIME_Q // 2) % PRIME_Q}

        commit = {"tokens": tokens, "commitments": commitments}
        with patch("reliquary.shared.forward.forward_single_layer", return_value=(hidden, torch.zeros(1, seq_len, 10))), \
             patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM):
            res = verify_commitment_proofs(
                commit, _make_mock_model(), randomness
            )
        assert not res.all_passed
        assert res.passed < res.checked


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


class TestTokenSequenceLengthCheck:
    """Rollouts with excessively long token sequences must be rejected
    before the forward pass to prevent GPU OOM.
    Fix: enforced by verify_tokens (_validate_sequence_length), which checks
    len(tokens) against the model's max_position_embeddings before GRAIL
    compute is triggered."""

    def test_rejects_oversized_sequence(self):
        from reliquary.protocol.tokens import verify_tokens
        from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

        # Maximum token sequence length: prompt (4096) + completion (8192)
        max_tokens_per_rollout = MAX_NEW_TOKENS_PROTOCOL_CAP + 4096
        tokens = list(range(max_tokens_per_rollout + 1))

        class _OverflowConfig:
            vocab_size = max_tokens_per_rollout + 10
            max_position_embeddings = max_tokens_per_rollout

        assert verify_tokens(tokens, _OverflowConfig()) is False

    def test_accepts_valid_length(self):
        from reliquary.validator.verifier import verify_commitment_proofs

        seq_len = 64
        commit = {"tokens": list(range(seq_len)), "commitments": [{"sketch": 0}] * seq_len}
        # Will fail on proof verification but should not short-circuit on length
        with patch("reliquary.shared.forward.forward_single_layer",
                   return_value=(torch.randn(1, seq_len, HIDDEN_DIM), torch.zeros(1, seq_len, 10))), \
             patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=HIDDEN_DIM):
            res = verify_commitment_proofs(
                commit, _make_mock_model(), "aabb"
            )
            assert res.checked > 0  # Got past the length check
