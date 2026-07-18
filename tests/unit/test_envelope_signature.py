"""Envelope-signature gate on /submit.

Closes the hotkey-spoof DoS where a third party could spam 8 unsigned
requests claiming a victim's ``miner_hotkey`` and exhaust the victim's
``_per_window_counts`` slot before any honest submission arrived. The
gate runs BEFORE the rate-limit increment so a flood of forged
signatures costs O(1) ed25519 verifies per request and never touches a
victim's quota.

Tests cover:

  1. ``build_envelope_binding`` is deterministic and includes every
     field the validator routes on.
  2. ``sign_envelope`` round-trips through ``verify_envelope_signature``.
  3. Tampering with any envelope field invalidates the signature.
  4. /submit with a valid sig is accepted (regression test).
  5. /submit with NO sig is rejected as BAD_ENVELOPE_SIGNATURE
     when ``ENFORCE_ENVELOPE_SIGNATURE`` is on.
  6. /submit with a sig signed by a DIFFERENT hotkey is rejected.
  7. The per-hotkey rate-limit counter is NOT incremented when the
     signature is rejected (the core DoS defence — if the counter
     moved on bad sigs an attacker could still exhaust quotas).
  8. With the enforcement flag OFF, an empty signature is silently
     accepted (back-compat rollout path).
"""

from __future__ import annotations

import os

import bittensor as bt
import pytest
from fastapi.testclient import TestClient

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.signatures import (
    build_envelope_binding,
    build_precommit_binding,
    sign_envelope,
    sign_precommit,
    verify_envelope_signature,
    verify_precommit_signature,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    RejectReason,
    WindowState,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.server import ValidatorServer


# ---------------------------------------------------------------------------
# Reusable helpers — minimal copies of what test_validator_server.py uses.
# ---------------------------------------------------------------------------


class _FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c):
        return 1.0 if "CORRECT" in c else 0.0


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


class _ModelStub:
    class config:
        vocab_size = 10000
        max_position_embeddings = 4096


def _make_commit(success=False, total_reward=0.0):
    prompt_length = 4
    seq_len = CHALLENGE_K + prompt_length
    tokens = list(range(seq_len))
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v7",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def _new_keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _batcher(window_start=500, randomness="cd" * 16):
    batcher = GrpoWindowBatcher(
        window_start=window_start,
        env=_FakeEnv(),
        model=_ModelStub(),
        cooldown_map=None,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "CORRECT" if r.reward > 0.5 else "wrong",
        drand_round_check_enabled=False,
    )
    batcher.current_checkpoint_hash = "sha256:test"
    batcher.randomness = randomness
    return batcher


def _signed_request(
    keypair: bt.Keypair,
    *,
    prompt_idx=42,
    window_start=500,
    k=4,
    checkpoint_hash="sha256:test",
    randomness="cd" * 16,
    drand_round=0,
    nonce: str | None = None,
) -> BatchSubmissionRequest:
    rollouts = []
    for i in range(M_ROLLOUTS):
        success = i < k
        reward = 1.0 if success else 0.0
        commit = _make_commit(success=success, total_reward=reward)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=reward,
                commit=commit,
                env_name=_FakeEnv.name,
            )
        )
    if nonce is None:
        nonce = os.urandom(12).hex()
    merkle_root = "00" * 32

    # Build a wallet-like object exposing .hotkey.sign(...).
    class _W:
        class hotkey:
            ss58_address = keypair.ss58_address
            @staticmethod
            def sign(msg: bytes) -> bytes:
                return keypair.sign(msg)
    sig = sign_envelope(
        wallet=_W,
        miner_hotkey=keypair.ss58_address,
        window_start=window_start,
        prompt_idx=prompt_idx,
        merkle_root=merkle_root,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        randomness=randomness,
        nonce=nonce,
    ).hex()
    return BatchSubmissionRequest(
        miner_hotkey=keypair.ss58_address,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root=merkle_root,
        rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        protocol_version=2,
        nonce=nonce,
        envelope_signature=sig,
    )


# ---------------------------------------------------------------------------
# 1. Canonical message determinism + field coverage
# ---------------------------------------------------------------------------


def test_envelope_binding_is_deterministic():
    """Same inputs must always produce the same bytes."""
    common = dict(
        miner_hotkey="5HotKey",
        window_start=10,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        drand_round=12345,
        randomness="cd" * 16,
        nonce="abcd1234",
    )
    a = build_envelope_binding(**common)
    b = build_envelope_binding(**common)
    assert a == b
    assert len(a) == 32  # SHA-256


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("miner_hotkey", "5DifferentKey"),
        ("window_start", 11),
        ("prompt_idx", 43),
        ("merkle_root", "ff" * 32),
        ("checkpoint_hash", "sha256:other"),
        ("drand_round", 12346),
        ("randomness", "ef" * 16),
        ("nonce", "abcd1235"),
    ],
)
def test_envelope_binding_changes_on_each_field(field, bad_value):
    """Mutating ANY input field must change the digest — otherwise the
    signature wouldn't cover that field and would be replayable for a
    subtly-different submission."""
    common = dict(
        miner_hotkey="5HotKey",
        window_start=10,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        drand_round=12345,
        randomness="cd" * 16,
        nonce="abcd1234",
    )
    original = build_envelope_binding(**common)
    mutated = build_envelope_binding(**{**common, field: bad_value})
    assert original != mutated, f"binding does NOT change when {field} mutates"


# ---------------------------------------------------------------------------
# 2. Sign / verify round-trip
# ---------------------------------------------------------------------------


def test_sign_envelope_round_trips_through_verify():
    kp = _new_keypair()

    class _W:
        class hotkey:
            ss58_address = kp.ss58_address
            @staticmethod
            def sign(msg):
                return kp.sign(msg)

    common = dict(
        miner_hotkey=kp.ss58_address,
        window_start=500,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        drand_round=12345,
        randomness="cd" * 16,
        nonce="nonce-1",
    )
    sig = sign_envelope(wallet=_W, **common).hex()
    assert verify_envelope_signature(**common, envelope_signature=sig)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("window_start", 501),
        ("prompt_idx", 43),
        ("merkle_root", "00" * 32),
        ("checkpoint_hash", "sha256:other"),
        ("environment", "opencodeinstruct"),
        ("payload_bytes", 654322),
        ("payload_sha256", "00" * 32),
        ("drand_round", 12346),
        ("randomness", "ef" * 16),
        ("protocol_version", 3),
        ("nonce", "nonce-2"),
    ],
)
def test_precommit_signature_binds_routing_size_and_freshness(field, bad_value):
    kp = _new_keypair()

    class _W:
        class hotkey:
            ss58_address = kp.ss58_address

            @staticmethod
            def sign(msg):
                return kp.sign(msg)

    common = dict(
        miner_hotkey=kp.ss58_address,
        window_start=500,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        environment="openmathinstruct",
        payload_bytes=654321,
        payload_sha256="ab" * 32,
        drand_round=12345,
        randomness="cd" * 16,
        protocol_version=2,
        nonce="nonce-1",
    )
    assert build_precommit_binding(**common) == build_precommit_binding(**common)
    signature = sign_precommit(wallet=_W, **common).hex()
    assert verify_precommit_signature(
        **common,
        precommit_signature=signature,
    )
    assert not verify_precommit_signature(
        **{**common, field: bad_value},
        precommit_signature=signature,
    )


# ---------------------------------------------------------------------------
# 3. Tampering breaks the signature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tamper_field,tamper_value",
    [
        ("window_start", 999),
        ("prompt_idx", 999),
        ("merkle_root", "00" * 32),
        ("drand_round", 99999),
        ("randomness", "00" * 16),
        ("nonce", "tampered-nonce"),
    ],
)
def test_tampering_invalidates_signature(tamper_field, tamper_value):
    kp = _new_keypair()
    class _W:
        class hotkey:
            ss58_address = kp.ss58_address
            @staticmethod
            def sign(msg):
                return kp.sign(msg)
    common = dict(
        miner_hotkey=kp.ss58_address,
        window_start=500,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        drand_round=12345,
        randomness="cd" * 16,
        nonce="nonce-1",
    )
    sig = sign_envelope(wallet=_W, **common).hex()
    # Original verifies
    assert verify_envelope_signature(**common, envelope_signature=sig)
    # Tampered version does NOT
    bad = {**common, tamper_field: tamper_value}
    assert not verify_envelope_signature(**bad, envelope_signature=sig)


def test_signature_under_different_hotkey_rejected():
    """A signature is bound to a specific ss58. Re-attributing it to a
    different ``miner_hotkey`` field must fail verification."""
    kp_a = _new_keypair()
    kp_b = _new_keypair()
    class _WA:
        class hotkey:
            ss58_address = kp_a.ss58_address
            @staticmethod
            def sign(msg):
                return kp_a.sign(msg)

    common = dict(
        window_start=500,
        prompt_idx=42,
        merkle_root="ab" * 32,
        checkpoint_hash="sha256:ckpt",
        drand_round=12345,
        randomness="cd" * 16,
        nonce="nonce-1",
    )
    # A signs for themselves
    sig = sign_envelope(
        wallet=_WA, miner_hotkey=kp_a.ss58_address, **common,
    ).hex()
    # Attempting to attribute to B's hotkey must not verify.
    assert not verify_envelope_signature(
        miner_hotkey=kp_b.ss58_address,
        **common,
        envelope_signature=sig,
    )


# ---------------------------------------------------------------------------
# 4. /submit accepts a valid envelope (regression)
# ---------------------------------------------------------------------------


def test_submit_accepts_valid_envelope(enforce_envelope):
    kp = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    req = _signed_request(kp, window_start=500)
    resp = client.post("/submit", json=req.model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True, body
    assert body["reason"] == RejectReason.ACCEPTED.value


# ---------------------------------------------------------------------------
# 5. Unsigned + 6. wrong-signer requests are rejected with BAD_ENVELOPE_SIGNATURE
# ---------------------------------------------------------------------------


def test_submit_rejects_unsigned_envelope(enforce_envelope):
    kp = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    req = _signed_request(kp, window_start=500)
    # Clear the signature → pretend the miner-side helper didn't run.
    payload = req.model_dump(mode="json")
    payload["envelope_signature"] = ""
    resp = client.post("/submit", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value


def test_submit_rejects_spoofed_hotkey(enforce_envelope):
    """The attack scenario: caller fills ``miner_hotkey`` with a victim's
    address but signs with their own key. Validator must reject as
    BAD_ENVELOPE_SIGNATURE — the binding includes the hotkey so the sig
    from key A can't verify against key B's ss58."""
    kp_victim = _new_keypair()
    kp_attacker = _new_keypair()

    class _WAttacker:
        class hotkey:
            ss58_address = kp_attacker.ss58_address
            @staticmethod
            def sign(msg):
                return kp_attacker.sign(msg)

    # Attacker signs an envelope claiming the victim's hotkey.
    sig_inputs = dict(
        window_start=500,
        prompt_idx=42,
        merkle_root="00" * 32,
        checkpoint_hash="sha256:test",
        drand_round=0,
        randomness="cd" * 16,
        nonce="attack",
    )
    spoofed_sig = sign_envelope(
        wallet=_WAttacker,
        miner_hotkey=kp_victim.ss58_address,  # ← lies about who signed
        **sig_inputs,
    ).hex()
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=(i < 4), total_reward=float(i < 4))
        rollouts.append(RolloutSubmission(
            tokens=commit["tokens"],
            reward=float(i < 4),
            commit=commit,
            env_name=_FakeEnv.name,
        ))
    # ``randomness`` is signed over but not a wire field — drop it before
    # constructing the request body.
    req_inputs = {k: v for k, v in sig_inputs.items() if k != "randomness"}
    req = BatchSubmissionRequest(
        miner_hotkey=kp_victim.ss58_address,
        rollouts=rollouts,
        envelope_signature=spoofed_sig,
        **req_inputs,
    )

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=req.model_dump(mode="json"))
    body = resp.json()
    assert body["accepted"] is False, body
    assert body["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value


# ---------------------------------------------------------------------------
# 7. The DoS-defence invariant: bad-sig rejects do NOT bump the counter
# ---------------------------------------------------------------------------


def test_bad_signature_does_not_consume_victim_quota(enforce_envelope):
    """THE critical regression test for the spoofed-hotkey DoS.

    Attacker sends MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW invalid-sig
    requests claiming the victim's hotkey. Then the legitimate victim
    sends a single valid submission. Pre-fix, the victim's submission
    returned RATE_LIMITED because the counter was already at the cap.
    Post-fix, the counter is untouched by bad-sig requests and the
    victim's valid submission is queued normally.
    """
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
    kp_victim = _new_keypair()

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # Attacker spam — every request lies about miner_hotkey + bad sig.
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=(i < 4), total_reward=float(i < 4))
        rollouts.append(RolloutSubmission(
            tokens=commit["tokens"],
            reward=float(i < 4),
            commit=commit,
            env_name=_FakeEnv.name,
        ))
    bad_req = BatchSubmissionRequest(
        miner_hotkey=kp_victim.ss58_address,
        prompt_idx=42,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        drand_round=0,
        nonce="attack",
        envelope_signature="",  # unsigned
    )
    for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW):
        resp = client.post("/submit", json=bad_req.model_dump(mode="json"))
        assert resp.json()["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value

    # The victim's quota counter must NOT have moved.
    assert server._per_window_counts.get(kp_victim.ss58_address, 0) == 0

    # And the victim's first legitimate submission now goes through.
    good = _signed_request(kp_victim, window_start=500, prompt_idx=42)
    resp = client.post("/submit", json=good.model_dump(mode="json"))
    body = resp.json()
    assert body["accepted"] is True, body
    assert body["reason"] == RejectReason.ACCEPTED.value


# ---------------------------------------------------------------------------
# 8. Back-compat: enforcement flag off accepts empty signatures
# ---------------------------------------------------------------------------


def test_enforcement_off_accepts_unsigned(monkeypatch):
    """During a rolling miner upgrade, operators may temporarily set
    ``ENFORCE_ENVELOPE_SIGNATURE = False`` so that pre-PR miners aren't
    locked out. Verify the gate is fully bypassed in that mode."""
    import reliquary.validator.server as server_mod

    monkeypatch.setattr(server_mod, "ENFORCE_ENVELOPE_SIGNATURE", False)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    # Construct a request with empty signature — pre-PR clients.
    kp = _new_keypair()
    req = _signed_request(kp, window_start=500)
    payload = req.model_dump(mode="json")
    payload["envelope_signature"] = ""
    payload["nonce"] = ""
    resp = client.post("/submit", json=payload)
    body = resp.json()
    # Reaches the batcher; should be accepted by the test fixtures.
    assert body["reason"] != RejectReason.BAD_ENVELOPE_SIGNATURE.value
