"""Validator HTTP server — v2 GRPO market endpoints."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
import bittensor as bt
import pytest
from fastapi.testclient import TestClient

from reliquary.constants import (
    CHALLENGE_K,
    FORCED_SEED_PROTOCOL_VERSION,
    M_ROLLOUTS,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
    REGISTERED_HOTKEY_CACHE_TTL_SECONDS,
    REGISTERED_HOTKEY_STALE_GRACE_SECONDS,
)
from reliquary.environment.virtual_parquet import PromptSourceUnavailable
from reliquary.protocol.legacy_merkle import (
    compute_legacy_rollouts_merkle_root,
)
from reliquary.protocol.signatures import sign_envelope, sign_precommit
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    FillClosedWindowState,
    GrpoBatchState,
    RolloutSubmission,
    RejectReason,
    SubmissionPrecommitRequest,
)
from reliquary.validator.admission import PreparedSubmission
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.observability import DrandRoundObservation, SubmitTelemetry
from reliquary.validator.selection_digest import compute_rollouts_selection_digest
from reliquary.validator.server import (
    _SubmissionBodyLimitMiddleware,
    _SubmissionHttpProtocol,
    ValidatorServer,
    _QueuedAuctionSubmission,
    _UploadPrecommitReceipt,
)


# Persistent test keypair so every ``_request()`` carries a verifiable
# envelope signature. Built once per module — generating a fresh
# keypair each call is fine but slows tests.
_TEST_KEYPAIR = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


class _TestWallet:
    """Wallet-like shim around ``_TEST_KEYPAIR`` for ``sign_envelope``."""

    class hotkey:
        ss58_address = _TEST_KEYPAIR.ss58_address

        @staticmethod
        def sign(msg: bytes) -> bytes:
            return _TEST_KEYPAIR.sign(msg)


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


class _UnavailableEnv(FakeEnv):
    def __init__(self, fail_at):
        self.fail_at = fail_at

    def __len__(self):
        if self.fail_at == "length":
            raise PromptSourceUnavailable("source unavailable")
        return super().__len__()

    def get_problem(self, idx):
        if self.fail_at == "row":
            raise PromptSourceUnavailable("source unavailable")
        return super().get_problem(idx)

    def source_health(self):
        return {
            "status": "degraded",
            "repo": "owner/repo",
            "revision": "rev",
            "last_error_type": "PromptSourceUnavailable",
        }


class _FlakyCodeEnv(FakeEnv):
    name = "opencodeinstruct"
    validator_authoritative_reward = True

    def __init__(self):
        self.unavailable = True

    def compute_reward(self, problem, completion):
        if self.unavailable:
            from reliquary.environment.grader_client import (
                GraderInfrastructureError,
            )

            raise GraderInfrastructureError("unreachable")
        return super().compute_reward(problem, completion)


class _CrashingCodeEnv(_FlakyCodeEnv):
    def compute_reward(self, problem, completion):
        from reliquary.environment.grader_client import GraderInfrastructureError

        raise GraderInfrastructureError("crash")


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _make_commit(success: bool = False, total_reward: float = 0.0) -> dict:
    """Build a schema-compliant commit dict for server tests."""
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


class _ModelStub:
    """Minimal stub for TokenValidator tests."""
    class config:
        vocab_size = 10000
        max_position_embeddings = 4096


def _batcher(
    window_start=500,
    cooldown_map=None,
    env=None,
    operator_by_hotkey=None,
):
    batcher = GrpoWindowBatcher(
        window_start=window_start,
        env=env or FakeEnv(),
        model=_ModelStub(),
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "CORRECT" if r.reward > 0.5 else "wrong",
        # Server tests post legacy requests without drand_round; disable the gate.
        drand_round_check_enabled=False,
        operator_by_hotkey=operator_by_hotkey,
    )
    batcher.current_checkpoint_hash = "sha256:test"
    # Match the per-window randomness used by ``_make_commit`` so the
    # randomness-binding check accepts the test request. Production sets
    # this in service.py's ``_set_window_randomness`` step before the
    # window opens for submissions.
    batcher.randomness = "cd" * 16
    return batcher


def _request(
    prompt_idx=42,
    window_start=500,
    k=4,
    checkpoint_hash="sha256:test",
    randomness="cd" * 16,
    valid_merkle=False,
):
    """Build a fully-signed envelope so tests pass the validator's
    ``ENFORCE_ENVELOPE_SIGNATURE`` gate. ``miner_hotkey`` is bound to
    ``_TEST_KEYPAIR.ss58_address`` because the signature covers it."""
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
                env_name=FakeEnv.name,
            )
        )
    merkle_root = (
        compute_legacy_rollouts_merkle_root(rollouts)
        if valid_merkle
        else "00" * 32
    )
    drand_round = 0
    protocol_version = PROTOCOL_VERSION
    generation_profile_id = (
        PROTOCOL_PROFILE_ID if protocol_version >= 3 else ""
    )
    nonce = os.urandom(8).hex()
    sig = sign_envelope(
        wallet=_TestWallet,
        miner_hotkey=_TEST_KEYPAIR.ss58_address,
        window_start=window_start,
        prompt_idx=prompt_idx,
        merkle_root=merkle_root,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        randomness=randomness,
        nonce=nonce,
        protocol_version=protocol_version,
        generation_profile_id=generation_profile_id,
    ).hex()
    return BatchSubmissionRequest(
        miner_hotkey=_TEST_KEYPAIR.ss58_address,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root=merkle_root,
        rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        protocol_version=protocol_version,
        generation_profile_id=generation_profile_id,
        nonce=nonce,
        envelope_signature=sig,
    )


def _precommit_for(
    request: BatchSubmissionRequest,
    *,
    payload_bytes: int,
    payload_sha256: str | None = None,
) -> SubmissionPrecommitRequest:
    if payload_sha256 is None:
        payload = request.model_dump_json().encode("utf-8")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
    fields = {
        "miner_hotkey": request.miner_hotkey,
        "window_start": request.window_start,
        "prompt_idx": request.prompt_idx,
        "merkle_root": request.merkle_root,
        "checkpoint_hash": request.checkpoint_hash,
        "environment": FakeEnv.name,
        "payload_bytes": payload_bytes,
        "payload_sha256": payload_sha256,
        "drand_round": request.drand_round,
        "randomness": "cd" * 16,
        "protocol_version": request.protocol_version,
        "generation_profile_id": request.generation_profile_id,
        "nonce": request.nonce,
    }
    signature = sign_precommit(wallet=_TestWallet, **fields).hex()
    fields.pop("randomness")
    return SubmissionPrecommitRequest(
        **fields,
        precommit_signature=signature,
    )


def test_submit_openapi_keeps_batch_submission_request_contract():
    schema = ValidatorServer().app.openapi()
    request_body = schema["paths"]["/submit"]["post"]["requestBody"]
    submission_schema = request_body["content"]["application/json"]["schema"]

    assert request_body["required"] is True
    assert submission_schema["title"] == "BatchSubmissionRequest"
    assert "miner_hotkey" in submission_schema["properties"]
    assert "rollouts" in submission_schema["properties"]


def test_auction_submit_requires_precommit_before_body_parse(monkeypatch):
    from reliquary.protocol.submission import WindowState

    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server = ValidatorServer()
    server.set_active_batchers({"openmathinstruct": batcher})
    server.set_current_state(WindowState.OPEN)
    server._auction_admission_enabled = True

    def _unexpected_parse(*_args, **_kwargs):
        raise AssertionError("large body parser must not run without receipt")

    monkeypatch.setattr(
        BatchSubmissionRequest,
        "model_validate_json",
        _unexpected_parse,
    )
    response = TestClient(server.app).post(
        "/submit",
        content=b'{"untrusted":"large body"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.PRECOMMIT_REQUIRED.value,
    }
    assert response.headers["connection"] == "close"


def test_claiming_a_precommit_stamps_its_receipt_id_onto_the_request():
    """v6 only. The arrival proof path looks up the rate a precommit
    registered off ``pending.request._precommit_receipt_id``. This is the
    exact point the body is matched to its precommit, so it is where the
    id must land -- a replay or a mismatch must never stamp anything."""
    batcher = _batcher(window_start=500)
    server = ValidatorServer()
    request = _request(valid_merkle=True)
    raw_body = request.model_dump_json().encode()
    payload_sha256 = hashlib.sha256(raw_body).hexdigest()
    receipt_id = "rate-linking-receipt"
    receipt = _UploadPrecommitReceipt(
        receipt_id=receipt_id,
        precommit_signature="signed",
        miner_hotkey=request.miner_hotkey,
        prompt_idx=request.prompt_idx,
        window_start=request.window_start,
        merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash,
        environment=FakeEnv.name,
        payload_bytes=len(raw_body),
        payload_sha256=payload_sha256,
        drand_round=request.drand_round,
        protocol_version=request.protocol_version,
        nonce=request.nonce,
        expires_at_wall=time.time() + 30.0,
        precommit_arrival_ts=time.time(),
        drand_observation=DrandRoundObservation(
            submitted_drand_round=request.drand_round,
            arrival_drand_round=request.drand_round,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        ),
        batcher=batcher,
        body_completed_at_wall=time.time(),
    )
    server._upload_precommit_receipts[receipt_id] = receipt

    assert getattr(request, "_precommit_receipt_id", "") == ""

    status, claimed = server._claim_upload_precommit(
        receipt_id,
        request,
        batcher=batcher,
        environment=FakeEnv.name,
        payload_bytes=len(raw_body),
        payload_sha256=payload_sha256,
        upload_started_at=time.time(),
        body_completed_at=time.time(),
    )

    assert status == "valid"
    assert claimed is receipt
    assert request._precommit_receipt_id == receipt_id


@pytest.mark.asyncio
async def test_grader_crash_prepared_result_retains_operator_prompt_claim():
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server = ValidatorServer()
    server.set_active_batchers({FakeEnv.name: batcher})
    request = _request(valid_merkle=True)
    raw_body = request.model_dump_json().encode()
    receipt_id = "crash-receipt"
    accepted, reason, _deadline = batcher.try_register_upload_precommit(
        receipt_id,
        request.miner_hotkey,
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=len(raw_body),
    )
    assert accepted is True
    assert reason is None
    assert batcher.mark_upload_precommit_revealed(receipt_id) is True

    receipt = _UploadPrecommitReceipt(
        receipt_id=receipt_id,
        precommit_signature="signed",
        miner_hotkey=request.miner_hotkey,
        prompt_idx=request.prompt_idx,
        window_start=request.window_start,
        merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash,
        environment=FakeEnv.name,
        payload_bytes=len(raw_body),
        payload_sha256=hashlib.sha256(raw_body).hexdigest(),
        drand_round=request.drand_round,
        protocol_version=request.protocol_version,
        nonce=request.nonce,
        expires_at_wall=time.time() + 30.0,
        precommit_arrival_ts=time.time(),
        drand_observation=DrandRoundObservation(
            submitted_drand_round=request.drand_round,
            arrival_drand_round=request.drand_round,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        ),
        batcher=batcher,
        consumed=True,
    )
    telemetry = SubmitTelemetry.from_request(
        request,
        t_arrival=time.time(),
        payload_bytes=len(raw_body),
        payload_sha256=receipt.payload_sha256,
    )
    rollout_hashes = [bytes([index]) * 32 for index in range(M_ROLLOUTS)]
    prepared = PreparedSubmission(
        request=request,
        completion_texts=[],
        rewards=[],
        rollout_hashes=rollout_hashes,
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
        reject_reason=RejectReason.WORKER_DROPPED,
        reject_stage="code_grader",
        grader_failure_reason="crash",
    )
    server._admission_materialization_pool = ThreadPoolExecutor(max_workers=1)
    server._run_admission_process = AsyncMock(return_value=prepared)
    item = _QueuedAuctionSubmission(
        raw_body=raw_body,
        receipt=receipt,
        batcher=batcher,
        telemetry=telemetry,
        enqueued_monotonic=time.monotonic(),
    )

    try:
        await server._process_auction_submission(item, asyncio.Queue())
    finally:
        server._admission_materialization_pool.shutdown(wait=True)

    assert receipt.terminal is True
    assert receipt.outcome == BatchSubmissionResponse(
        accepted=False, reason=RejectReason.WORKER_DROPPED
    )
    assert batcher.logical_group_reservation_count == 1
    assert batcher.grader_failures == {"crash": 1}
    duplicate = batcher.reserve_prepared_identity(request, rollout_hashes)
    assert duplicate == (
        False,
        RejectReason.HASH_DUPLICATE,
        "logical_dedup",
    )
    conservation = batcher.upload_precommit_conservation()
    assert conservation["terminal_decisions"] == 1
    assert conservation["conserved"] is True


@pytest.mark.asyncio
async def test_prepared_prompt_mismatch_releases_received_bytes(monkeypatch):
    """Reproduce the production ordering missed by the grading-refund tests.

    Prompt binding rejects in the isolated worker before
    ``start_revealed_admission``.  Resolving that terminal receipt must recycle
    its precommit reservation while leaving the miner's cumulative quota spent.
    """
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(batcher_module, "MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW", 1)

    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server = ValidatorServer()
    server.set_active_batchers({FakeEnv.name: batcher})
    request = _request(valid_merkle=True)
    raw_body = request.model_dump_json().encode()
    receipt_id = "prompt-mismatch-receipt"
    accepted, reason, _deadline = batcher.try_register_upload_precommit(
        receipt_id,
        request.miner_hotkey,
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=len(raw_body),
    )
    assert accepted is True
    assert reason is None
    assert batcher.account_upload_precommit_bytes(
        receipt_id,
        t_arrival_wall=batcher.window_opened_wall_ts,
        chunk_bytes=len(raw_body),
    ) == (True, None)
    assert batcher.mark_upload_precommit_transport_complete(
        receipt_id,
        completed_at_wall=batcher.window_opened_wall_ts,
    ) == (True, None)
    assert batcher.mark_upload_precommit_revealed(receipt_id) is True

    receipt = _UploadPrecommitReceipt(
        receipt_id=receipt_id,
        precommit_signature="signed-prompt-mismatch",
        miner_hotkey=request.miner_hotkey,
        prompt_idx=request.prompt_idx,
        window_start=request.window_start,
        merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash,
        environment=FakeEnv.name,
        payload_bytes=len(raw_body),
        payload_sha256=hashlib.sha256(raw_body).hexdigest(),
        drand_round=request.drand_round,
        protocol_version=request.protocol_version,
        nonce=request.nonce,
        expires_at_wall=time.time() + 30.0,
        precommit_arrival_ts=time.time(),
        drand_observation=DrandRoundObservation(
            submitted_drand_round=request.drand_round,
            arrival_drand_round=request.drand_round,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        ),
        batcher=batcher,
        consumed=True,
    )
    telemetry = SubmitTelemetry.from_request(
        request,
        t_arrival=time.time(),
        payload_bytes=len(raw_body),
        payload_sha256=receipt.payload_sha256,
    )
    prepared = PreparedSubmission(
        request=request,
        completion_texts=[],
        rewards=[],
        rollout_hashes=[bytes([index]) * 32 for index in range(M_ROLLOUTS)],
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
        reject_reason=RejectReason.PROMPT_MISMATCH,
        reject_stage="prompt_binding",
    )
    server._per_window_counts[request.miner_hotkey] = 1
    server._admission_materialization_pool = ThreadPoolExecutor(max_workers=1)
    server._run_admission_process = AsyncMock(return_value=prepared)
    item = _QueuedAuctionSubmission(
        raw_body=raw_body,
        receipt=receipt,
        batcher=batcher,
        telemetry=telemetry,
        enqueued_monotonic=time.monotonic(),
    )

    try:
        await server._process_auction_submission(item, asyncio.Queue())
    finally:
        server._admission_materialization_pool.shutdown(wait=True)

    assert receipt.outcome == BatchSubmissionResponse(
        accepted=False,
        reason=RejectReason.PROMPT_MISMATCH,
    )
    assert batcher.proof_grading_attempts == 0
    assert server._per_window_counts[request.miner_hotkey] == 1
    conservation = batcher.upload_precommit_conservation()
    assert conservation["accepted_receipts"] == 1
    assert conservation["pending"] == 0
    assert conservation["capacity_reserved"] == 0
    assert conservation["productive_capacity_used"] == 0
    assert conservation["capacity_conserved"] is True
    circuit_health = server._prompt_mismatch_circuit.health_snapshot(
        current_window=request.window_start
    )
    assert circuit_health["mismatches_total"] == 1
    assert circuit_health["partial_strike_entries"] == 1

    accepted, reason, _deadline = batcher.try_register_upload_precommit(
        "honest-replacement",
        request.miner_hotkey,
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=len(raw_body),
    )
    assert accepted is True
    assert reason is None
    assert batcher.upload_precommit_conservation()["accepted_receipts"] == 2


def test_prompt_mismatch_circuit_rejects_before_receipt_registration():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    # Bootstrap intentionally relaxes the request generation-profile gate. The
    # circuit namespace must still be validator-owned so rotating this signed,
    # miner-controlled field cannot bypass an armed identity.
    batcher.current_checkpoint_hash = ""
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True).model_copy(
        update={"generation_profile_id": "attacker-rotated-profile"}
    )
    operator = "operator-coldkey"
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: operator},
    )
    identities = {
        "hotkey": request.miner_hotkey,
        "operator": operator,
    }
    for index in range(3):
        server._prompt_mismatch_circuit.record_mismatch(
            environment=FakeEnv.name,
            identities=identities,
            window=request.window_start,
            precommit_signature=f"terminal-mismatch-{index}",
            precommit_arrival_ts=float(index + 1),
        )

    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    with TestClient(server.app) as client:
        response = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.RATE_LIMITED.value,
        "receipt_id": None,
        "upload_deadline_ts": None,
    }
    assert server._upload_precommit_receipts == {}
    assert server._per_window_counts.get(request.miner_hotkey, 0) == 0
    assert batcher.pending_upload_precommits == 0


def test_no_reveal_circuit_rejects_operator_before_receipt_registration():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = ""
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    operator = "operator-coldkey"
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: operator},
    )
    for index in range(4):
        server._no_reveal_circuit.record_no_reveal(
            environment=FakeEnv.name,
            operator=operator,
            window=request.window_start,
            precommit_signature=f"expired-receipt-{index}",
            precommit_arrival_ts=float(index + 1),
        )

    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    with TestClient(server.app) as client:
        response = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.json()["reason"] == RejectReason.RATE_LIMITED.value
    assert server._upload_precommit_receipts == {}
    assert server._per_window_counts.get(request.miner_hotkey, 0) == 0
    assert batcher.pending_upload_precommits == 0


@pytest.mark.asyncio
async def test_upload_start_is_stamped_on_first_body_byte():
    starts = []

    async def downstream(scope, receive, _send):
        state = scope.setdefault("state", {})
        state["upload_start_callback"] = lambda started_at: (
            starts.append(started_at) or (True, None)
        )
        assert (await receive())["body"] == b""
        assert starts == []
        assert (await receive())["body"] == b"payload"

    app = _SubmissionBodyLimitMiddleware(downstream, max_bytes=100)
    chunks = [
        {"type": "http.request", "body": b"", "more_body": True},
        {"type": "http.request", "body": b"payload", "more_body": False},
    ]

    async def receive():
        return chunks.pop(0)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/submit",
        "headers": [],
    }
    await app(scope, receive, lambda _message: None)

    assert len(starts) == 1
    assert scope["state"]["body_receive_started_at"] == starts[0]
    assert scope["state"]["upload_start_result"] == (True, None)


@pytest.mark.asyncio
async def test_wire_timestamps_override_delayed_application_receive():
    wire_started_at = time.time() - 2.0
    wire_completed_at = time.time() - 1.0
    starts = []

    async def downstream(scope, receive, _send):
        state = scope.setdefault("state", {})
        state["upload_start_callback"] = lambda started_at: (
            starts.append(started_at) or (True, None)
        )
        assert (await receive())["body"] == b"payload"

    app = _SubmissionBodyLimitMiddleware(downstream, max_bytes=100)

    async def receive():
        return {
            "type": "http.request",
            "body": b"payload",
            "more_body": False,
        }

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/submit",
        "headers": [],
        "state": {
            "wire_body_started_at": wire_started_at,
            "wire_body_completed_at": wire_completed_at,
        },
    }
    await app(scope, receive, lambda _message: None)

    assert starts == [wire_started_at]
    assert scope["state"]["body_receive_started_at"] == wire_started_at
    assert scope["state"]["body_completed_at"] == wire_completed_at


@pytest.mark.asyncio
async def test_body_completed_on_wire_before_deadline_survives_app_delay():
    wire_completed_at = time.time() - 1.0
    deadline = wire_completed_at + 0.1
    sent = []

    async def downstream(_scope, receive, send):
        assert (await receive())["body"] == b"payload"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    app = _SubmissionBodyLimitMiddleware(
        downstream,
        max_bytes=100,
        deadline_resolver=lambda _scope, _path, _started: deadline,
    )

    async def receive():
        return {
            "type": "http.request",
            "body": b"payload",
            "more_body": False,
        }

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/submit",
        "headers": [],
        "state": {"wire_body_completed_at": wire_completed_at},
    }
    await app(scope, receive, send)

    assert sent[0]["status"] == 200
    assert scope["state"].get("body_read_timed_out") is None


@pytest.mark.asyncio
async def test_duplicate_receipt_headers_are_rejected_before_body_read():
    downstream_called = False

    async def downstream(_scope, _receive, _send):
        nonlocal downstream_called
        downstream_called = True

    app = _SubmissionBodyLimitMiddleware(downstream, max_bytes=100)
    sent = []

    async def receive():
        raise AssertionError("ambiguous request body must not be read")

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/submit",
        "headers": [
            (b"x-reliquary-precommit", b"first"),
            (b"x-reliquary-precommit", b"second"),
        ],
    }
    await app(scope, receive, send)

    assert downstream_called is False
    assert sent[0]["status"] == 400
    assert (b"connection", b"close") in sent[0]["headers"]


def test_submission_protocol_closes_incomplete_headers_and_bodies():
    closed = []
    protocol = object.__new__(_SubmissionHttpProtocol)
    protocol.transport = SimpleNamespace(
        is_closing=lambda: False,
        close=lambda: closed.append(True),
    )
    protocol.scope = {"state": {}}
    protocol._header_read_timeout_handle = None
    protocol._body_read_timeout_handle = None

    protocol._header_read_timed_out()
    protocol._body_read_timed_out()

    assert closed == [True, True]
    assert protocol.scope["state"]["wire_body_timed_out"] is True


@pytest.mark.asyncio
async def test_submission_protocol_stamps_wire_ingress():
    import uvicorn
    from uvicorn.server import ServerState

    observed = {}

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        observed.update(scope["state"])
        observed["body"] = body
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    config = uvicorn.Config(
        app,
        lifespan="off",
        access_log=False,
        log_level="error",
        http=_SubmissionHttpProtocol,
    )
    config.load()
    loop = asyncio.get_running_loop()
    server_state = ServerState()
    transport_server = await loop.create_server(
        lambda: _SubmissionHttpProtocol(
            config,
            server_state,
            {},
            loop,
        ),
        "127.0.0.1",
        0,
    )
    try:
        port = transport_server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"POST /submit HTTP/1.1\r\n"
            b"Host: validator.test\r\n"
            b"Content-Length: 7\r\n"
            b"Connection: close\r\n\r\n"
            b"payload"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
    finally:
        transport_server.close()
        await transport_server.wait_closed()

    assert response.startswith(b"HTTP/1.1 200 OK")
    assert observed["body"] == b"payload"
    assert observed["wire_headers_completed_at"] <= observed["wire_body_started_at"]
    assert observed["wire_body_started_at"] <= observed["wire_body_completed_at"]


@pytest.mark.asyncio
async def test_submission_protocol_closes_stalled_fresh_connection(monkeypatch):
    import uvicorn
    from uvicorn.server import ServerState
    from reliquary.validator import server as server_module

    monkeypatch.setattr(
        server_module,
        "SUBMISSION_HEADER_READ_TIMEOUT_SECONDS",
        0.02,
    )

    async def app(_scope, _receive, _send):
        raise AssertionError("incomplete headers must not reach ASGI")

    config = uvicorn.Config(
        app,
        lifespan="off",
        access_log=False,
        log_level="error",
        http=_SubmissionHttpProtocol,
    )
    config.load()
    loop = asyncio.get_running_loop()
    server_state = ServerState()
    transport_server = await loop.create_server(
        lambda: _SubmissionHttpProtocol(
            config,
            server_state,
            {},
            loop,
        ),
        "127.0.0.1",
        0,
    )
    try:
        port = transport_server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        closed = await asyncio.wait_for(reader.read(), timeout=0.5)
        writer.close()
        await writer.wait_closed()
    finally:
        transport_server.close()
        await transport_server.wait_closed()

    assert closed == b""


@pytest.mark.asyncio
async def test_incomplete_precommit_body_times_out_and_closes_connection():
    async def downstream(_scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send(
            {"type": "http.response.body", "body": b"", "more_body": False}
        )

    app = _SubmissionBodyLimitMiddleware(
        downstream,
        max_bytes=100,
        precommit_timeout_seconds=0.02,
    )
    sent = []

    async def stalled_receive():
        await asyncio.sleep(1.0)
        return {"type": "http.request", "body": b"late", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/submit/precommit",
        "headers": [],
    }
    await app(scope, stalled_receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 408
    assert (b"connection", b"close") in sent[0]["headers"]
    assert scope["state"]["body_read_timed_out"] is True


def test_prompt_mismatch_persistence_error_degrades_validator_health(tmp_path):
    server = ValidatorServer(
        prompt_mismatch_state_path=tmp_path / "prompt-mismatch.json"
    )
    server._prompt_mismatch_circuit._last_persistence_error = (
        "OSError: disk unavailable"
    )

    health = server._health_payload()

    assert health.status == "degraded"
    assert health.prompt_mismatch_circuit_breaker["status"] == "degraded"


def test_endpoint_latency_does_not_index_arbitrary_paths():
    server = ValidatorServer()
    client = TestClient(server.app)

    for index in range(10):
        assert client.get(f"/unknown-{index}").status_code == 404

    assert server._endpoint_latency_samples_ms == {}


def test_inactive_signed_precommit_does_not_extend_collection():
    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-coldkey"},
    )
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        assert committed.status_code == 200
        receipt = committed.json()
        assert receipt["accepted"] is True
        assert batcher.pending_upload_precommits == 1
        assert server._per_window_counts[request.miner_hotkey] == 1

        # A receipt without a body request in progress cannot keep the auction
        # open after the generation cutoff.
        batcher.window_opened_at -= WINDOW_COLLECTION_SECONDS + 1.0
        batcher.window_opened_wall_ts -= WINDOW_COLLECTION_SECONDS + 1.0
        assert batcher.collection_closed() is True
        assert batcher.poll_deadline() is True

        revealed = client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": receipt["receipt_id"],
            },
        )
        assert revealed.status_code == 200
        assert revealed.json() == {
            "accepted": False,
            "reason": RejectReason.PRECOMMIT_EXPIRED.value,
        }
        assert batcher.pending_upload_precommits == 0
        assert server._per_window_counts[request.miner_hotkey] == 1
        assert batcher.poll_deadline() is True

        replayed = client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": receipt["receipt_id"],
            },
        )
        assert replayed.json() == {
            "accepted": False,
            "reason": RejectReason.PRECOMMIT_EXPIRED.value,
        }

    circuit = server._no_reveal_circuit.health_snapshot(current_window=500)
    assert circuit["no_reveals_total"] == 1
    assert circuit["partial_strike_entries"] == 1


def test_abandoned_started_upload_is_terminal_and_releases_bytes():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    operator = "operator-coldkey"
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: operator},
    )
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))

    committed = (
        TestClient(server.app)
        .post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        .json()
    )
    receipt_id = committed["receipt_id"]
    scope = {"state": {"wire_receipt_id": receipt_id}}
    started_at = time.time()

    assert server._wire_upload_event("chunk", scope, started_at, 1) == (
        True,
        None,
    )
    assert batcher.upload_precommit_payload_bytes == 1
    assert server._wire_upload_event("aborted", scope, started_at + 0.1, 0) == (
        False,
        "upload_body_incomplete",
    )

    receipt = server._upload_precommit_receipts[receipt_id]
    assert receipt.terminal is True
    assert receipt.outcome == BatchSubmissionResponse(
        accepted=False,
        reason=RejectReason.PRECOMMIT_INVALID,
    )
    assert batcher.pending_upload_precommits == 0
    assert batcher.upload_precommit_payload_bytes == 0
    circuit = server._no_reveal_circuit.health_snapshot(current_window=500)
    assert circuit["no_reveals_total"] == 1
    assert circuit["valid_reveals_total"] == 0


def test_wire_complete_reveal_is_not_expired_during_application_delay():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    committed = TestClient(server.app).post(
        "/submit/precommit",
        content=precommit.model_dump_json(),
        headers={"Content-Type": "application/json"},
    ).json()
    receipt_id = committed["receipt_id"]
    receipt = server._upload_precommit_receipts[receipt_id]
    scope = {"state": {"wire_receipt_id": receipt_id}}

    assert server._wire_upload_event(
        "chunk",
        scope,
        receipt.expires_at_wall - 2.0,
        len(payload),
    ) == (True, None)
    assert server._wire_upload_event(
        "complete",
        scope,
        receipt.expires_at_wall - 1.0,
        0,
    ) == (True, None)

    server._prune_upload_precommits(now=receipt.expires_at_wall + 1.0)

    assert server._upload_precommit_receipts[receipt_id] is receipt
    assert batcher.pending_upload_precommits == 1
    circuit = server._no_reveal_circuit.health_snapshot(current_window=500)
    assert circuit["no_reveals_total"] == 0


@pytest.mark.asyncio
async def test_exact_malformed_body_is_not_counted_as_valid_reveal():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server._auction_admission_enabled = True
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    operator = "operator-coldkey"
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: operator},
    )
    malformed = b"x"
    precommit = _precommit_for(
        request,
        payload_bytes=len(malformed),
        payload_sha256=hashlib.sha256(malformed).hexdigest(),
    )

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()
        submitted = client.post(
            "/submit",
            content=malformed,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert submitted.json() == {
        "accepted": True,
        "reason": RejectReason.SUBMITTED.value,
    }
    queued = server._submit_queue.get_nowait()
    prepared = PreparedSubmission(
        request=None,
        completion_texts=[],
        rewards=[],
        rollout_hashes=[],
        selection_digest=None,
        reject_reason=RejectReason.BAD_SCHEMA,
        reject_stage="body_parse",
    )
    server._admission_materialization_pool = ThreadPoolExecutor(max_workers=1)
    server._run_admission_process = AsyncMock(return_value=prepared)
    try:
        await server._process_auction_submission(
            queued,
            server._submit_queue,
        )
    finally:
        server._admission_materialization_pool.shutdown(wait=True)

    receipt = server._upload_precommit_receipts[committed["receipt_id"]]
    assert receipt.terminal is True
    assert receipt.outcome == BatchSubmissionResponse(
        accepted=False,
        reason=RejectReason.BAD_SCHEMA,
    )
    circuit = server._no_reveal_circuit.health_snapshot(current_window=500)
    assert circuit["no_reveals_total"] == 1
    assert circuit["valid_reveals_total"] == 0


def test_precommit_headers_cannot_prime_before_window_open():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    batcher.window_opened_wall_ts = time.time() + 10.0
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))

    response = TestClient(server.app).post(
        "/submit/precommit",
        content=precommit.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.WINDOW_NOT_ACTIVE.value,
        "receipt_id": None,
        "upload_deadline_ts": None,
    }
    assert batcher.pending_upload_precommits == 0


def test_precommit_signature_saturation_fails_before_executor():
    from reliquary.constants import MAX_PENDING_PROOF_QUEUE_DEPTH
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    server._precommit_signature_pool = object()
    server._precommit_signature_inflight = MAX_PENDING_PROOF_QUEUE_DEPTH

    try:
        response = TestClient(server.app).post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
    finally:
        server._precommit_signature_pool = None

    assert response.status_code == 200
    assert response.json()["reason"] == RejectReason.BATCH_FILLED.value
    assert batcher.pending_upload_precommits == 0


@pytest.mark.asyncio
async def test_concurrent_identical_precommits_publish_one_receipt():
    import httpx

    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    arrivals = 0
    both_waiting = asyncio.Event()

    async def _registration_reject_reason(_hotkey):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            both_waiting.set()
        await asyncio.wait_for(both_waiting.wait(), timeout=1.0)

    server._registration_reject_reason = _registration_reject_reason
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        first, second = await asyncio.gather(
            client.post(
                "/submit/precommit",
                content=precommit.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ),
            client.post(
                "/submit/precommit",
                content=precommit.model_dump_json(),
                headers={"Content-Type": "application/json"},
            ),
        )

    assert first.json()["accepted"] is True
    assert second.json()["accepted"] is True
    assert first.json()["receipt_id"] == second.json()["receipt_id"]
    assert len(server._upload_precommit_receipts) == 1
    assert server._per_window_counts[request.miner_hotkey] == 1
    conservation = batcher.upload_precommit_conservation()
    assert conservation["accepted_receipts"] == 1
    assert conservation["pending"] == 1
    assert conservation["capacity_reserved"] == 0
    assert conservation["conserved"] is True


@pytest.mark.asyncio
async def test_complete_precommit_body_time_is_authoritative_for_drand_reveal(
    monkeypatch,
):
    import httpx

    from reliquary.protocol.submission import WindowState
    from reliquary.validator.observability import DrandRoundObservation

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    batcher.drand_round_check_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    precommit_body = precommit.model_dump_json().encode("utf-8")
    observations = []
    final_chunk_sent_at = []

    def observe(_self, drand_round, *, t_arrival=None):
        observations.append(t_arrival)
        return DrandRoundObservation(
            submitted_drand_round=drand_round,
            arrival_drand_round=drand_round,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        )

    monkeypatch.setattr(GrpoWindowBatcher, "observe_drand_round", observe)

    async def delayed_precommit_body():
        midpoint = len(precommit_body) // 2
        yield precommit_body[:midpoint]
        await asyncio.sleep(0.03)
        final_chunk_sent_at.append(time.time())
        yield precommit_body[midpoint:]

    header_arrival_started = time.time()
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        committed_response = await client.post(
            "/submit/precommit",
            content=delayed_precommit_body(),
            headers={"Content-Type": "application/json"},
        )
        committed = committed_response.json()
        revealed = await client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert revealed.json()["accepted"] is True
    assert len(observations) == 1
    assert observations[0] >= final_chunk_sent_at[0]
    assert observations[0] - header_arrival_started >= 0.02
    receipt = server._upload_precommit_receipts[committed["receipt_id"]]
    assert receipt.precommit_arrival_ts == observations[0]


@pytest.mark.asyncio
async def test_complete_direct_body_time_is_authoritative_for_drand(monkeypatch):
    import httpx

    from reliquary.protocol.submission import WindowState
    from reliquary.validator.observability import DrandRoundObservation

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.drand_round_check_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    body = request.model_dump_json().encode("utf-8")
    observations = []
    final_chunk_sent_at = []

    def observe(_self, drand_round, *, t_arrival=None):
        observations.append(t_arrival)
        return DrandRoundObservation(
            submitted_drand_round=drand_round,
            arrival_drand_round=drand_round,
            drand_delta=0,
            drand_tolerance=0,
            drand_status="current",
            reject_reason=None,
        )

    monkeypatch.setattr(GrpoWindowBatcher, "observe_drand_round", observe)

    async def delayed_body():
        midpoint = len(body) // 2
        yield body[:midpoint]
        await asyncio.sleep(0.03)
        final_chunk_sent_at.append(time.time())
        yield body[midpoint:]

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        response = await client.post(
            "/submit",
            content=delayed_body(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert len(observations) == 1
    assert observations[0] >= final_chunk_sent_at[0]


@pytest.mark.asyncio
async def test_precommit_body_crossing_collection_cutoff_is_rejected():
    import httpx

    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    precommit_body = precommit.model_dump_json().encode("utf-8")
    batcher.window_opened_wall_ts = (
        time.time() - WINDOW_COLLECTION_SECONDS + 0.03
    )

    async def late_body():
        await asyncio.sleep(0.08)
        yield precommit_body

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        response = await client.post(
            "/submit/precommit",
            content=late_body(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 408
    assert response.json() == {"detail": "submission_body_timeout"}
    assert response.headers["connection"] == "close"
    assert server._upload_precommit_receipts == {}
    assert batcher.pending_upload_precommits == 0


@pytest.mark.asyncio
async def test_started_reveal_body_is_closed_at_receipt_deadline():
    import httpx

    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server._auction_admission_enabled = True
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-coldkey"},
    )
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    transport = httpx.ASGITransport(app=server.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        committed = (
            await client.post(
                "/submit/precommit",
                content=precommit.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        ).json()
        receipt = server._upload_precommit_receipts[committed["receipt_id"]]
        receipt.expires_at_wall = time.time() + 0.05

        async def stalled_reveal():
            yield payload[:1]
            await asyncio.sleep(1.0)
            yield payload[1:]

        started = time.monotonic()
        response = await client.post(
            "/submit",
            content=stalled_reveal(),
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert time.monotonic() - started < 0.5
    assert response.status_code == 408
    assert response.json() == {"detail": "submission_body_timeout"}
    assert response.headers["connection"] == "close"
    assert batcher.pending_upload_precommits == 0


@pytest.mark.asyncio
async def test_first_reveal_byte_after_cutoff_aborts_remaining_body():
    import httpx

    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server._auction_admission_enabled = True
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    server.set_registered_hotkeys(
        {request.miner_hotkey},
        operator_by_hotkey={request.miner_hotkey: "operator-coldkey"},
    )
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload))
    transport = httpx.ASGITransport(app=server.app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://validator.test",
    ) as client:
        committed = (
            await client.post(
                "/submit/precommit",
                content=precommit.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        ).json()
        batcher.window_opened_wall_ts = (
            time.time() - WINDOW_COLLECTION_SECONDS + 0.05
        )

        async def late_reveal():
            await asyncio.sleep(0.08)
            yield payload[:1]
            await asyncio.sleep(1.0)
            yield payload[1:]

        started = time.monotonic()
        response = await client.post(
            "/submit",
            content=late_reveal(),
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert time.monotonic() - started < 0.5
    assert response.status_code == 200
    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.PRECOMMIT_EXPIRED.value,
    }
    assert response.headers["connection"] == "close"
    assert batcher.pending_upload_precommits == 0


def test_precommit_rejects_body_with_different_serialized_size():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    precommit = _precommit_for(request, payload_bytes=len(payload) + 1)

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()
        response = client.post(
            "/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )
    assert response.json()["reason"] == RejectReason.PRECOMMIT_INVALID.value


def test_precommit_rejects_same_size_body_substitution():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request(valid_merkle=True)
    payload = request.model_dump_json().encode("utf-8")
    tampered = payload.replace(b'"reward":1.0', b'"reward":0.0', 1)
    assert tampered != payload
    assert len(tampered) == len(payload)
    precommit = _precommit_for(request, payload_bytes=len(payload))

    with TestClient(server.app) as client:
        committed = client.post(
            "/submit/precommit",
            content=precommit.model_dump_json(),
            headers={"Content-Type": "application/json"},
        ).json()
        response = client.post(
            "/submit",
            content=tampered,
            headers={
                "Content-Type": "application/json",
                "X-Reliquary-Precommit": committed["receipt_id"],
            },
        )

    assert response.json()["reason"] == RejectReason.PRECOMMIT_INVALID.value


def test_legacy_merkle_shadow_accepts_mismatch_and_exposes_telemetry():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=False).model_dump(mode="json")
    )
    health = server._health_payload()

    assert response.json()["accepted"] is True
    assert health.legacy_merkle_root_enforced is False
    assert health.legacy_merkle_checks_total == 1
    assert health.legacy_merkle_matches == 0
    assert health.legacy_merkle_mismatches == 1
    assert health.legacy_merkle_errors == 0
    assert health.legacy_merkle_distinct_hotkeys == 1
    assert health.legacy_merkle_environments == [FakeEnv.name]
    assert health.legacy_merkle_protocol_versions == {
        str(FORCED_SEED_PROTOCOL_VERSION): 1
    }
    assert health.legacy_merkle_last_mismatch_ts is not None


def test_legacy_merkle_shadow_records_current_miner_match():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=True).model_dump(mode="json")
    )
    health = server._health_payload()

    assert response.json()["accepted"] is True
    assert health.legacy_merkle_matches == 1
    assert health.legacy_merkle_mismatches == 0
    assert health.legacy_merkle_last_mismatch_ts is None


def test_old_forced_seed_protocol_is_rejected_before_quota_or_proof():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    request = _request()
    request.protocol_version = FORCED_SEED_PROTOCOL_VERSION - 1

    response = TestClient(server.app).post(
        "/submit",
        json=request.model_dump(mode="json"),
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.SEED_MISMATCH.value,
    }
    assert server._per_window_counts == {}
    assert server.submit_queue_depth == 0
    assert batcher.proof_admission_count == 0
    assert batcher.proof_grading_attempts == 0


def test_legacy_merkle_enforcement_rejects_before_quota(monkeypatch):
    from reliquary.protocol.submission import WindowState
    from reliquary.validator import server as server_module

    monkeypatch.setattr(server_module, "LEGACY_MERKLE_ROOT_ENFORCE", True)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=False).model_dump(mode="json")
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.MERKLE_ROOT_MISMATCH.value,
    }
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_mixed_environment_is_rejected_before_quota():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    payload = _request(valid_merkle=True).model_dump(mode="json")
    payload["rollouts"][1]["env_name"] = "opencodeinstruct"

    response = TestClient(server.app).post("/submit", json=payload)

    assert response.json()["reason"] == RejectReason.BAD_SCHEMA.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts
    assert server._health_payload().legacy_merkle_checks_total == 0


def test_unknown_environment_never_falls_back_to_first_batcher():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    payload = _request(valid_merkle=True).model_dump(mode="json")
    for rollout in payload["rollouts"]:
        rollout["env_name"] = "opencodeinstruct"

    response = TestClient(server.app).post("/submit", json=payload)

    assert response.json()["reason"] == RejectReason.BAD_SCHEMA.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_submit_returns_queued_on_active_window():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == RejectReason.ACCEPTED.value


def test_logical_group_duplicate_is_rejected_quota_neutral_before_proof():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    first = client.post("/submit", json=_request(k=4).model_dump(mode="json"))
    duplicate = client.post(
        "/submit", json=_request(k=5).model_dump(mode="json")
    )

    assert first.json()["accepted"] is True
    assert duplicate.json() == {
        "accepted": False,
        "reason": RejectReason.HASH_DUPLICATE.value,
    }
    assert server._per_window_counts == {_TEST_KEYPAIR.ss58_address: 1}
    assert batcher.proof_grading_attempts == 1
    assert batcher.logical_group_reservation_count == 1
    assert batcher.logical_group_duplicate_rejects == 1
    # Deferred-proof mode admits the first candidate into the pending pool;
    # it does not become valid until the ranked proof pass at seal.
    assert len(batcher.pending_submissions()) == 1
    assert len(batcher.valid_submissions()) == 0
    health = client.get("/health").json()
    assert health["logical_group_reservations"] == 1
    assert health["logical_group_duplicate_rejects"] == 1
    assert health["logical_group_dedup_by_environment"] == {
        FakeEnv.name: {"reservations": 1, "duplicate_rejects": 1}
    }


def test_logical_group_digest_runs_off_event_loop(monkeypatch):
    from reliquary.protocol.submission import WindowState
    from reliquary.validator import server as server_module

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    original_to_thread = server_module.asyncio.to_thread
    offloaded = []

    async def tracking_to_thread(func, *args, **kwargs):
        if func is GrpoWindowBatcher.try_reserve_logical_group:
            offloaded.append(func)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(server_module.asyncio, "to_thread", tracking_to_thread)

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json")
    )

    assert response.json()["accepted"] is True
    assert offloaded == [GrpoWindowBatcher.try_reserve_logical_group]


def test_auction_missing_operator_mapping_rejects_prequeue_quota_neutral():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(
        window_start=500,
        operator_by_hotkey={"some-other-hotkey": "operator-a"},
    )
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json")
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.REGISTRATION_UNAVAILABLE.value,
    }
    assert server._per_window_counts == {}
    assert batcher.proof_grading_attempts == 0
    assert batcher.logical_group_reservation_count == 0


def test_code_grader_outage_refunds_quota_and_surfaces_health():
    from reliquary.protocol.submission import WindowState

    env = _FlakyCodeEnv()
    server = ValidatorServer()
    batcher = _batcher(
        window_start=500,
        env=env,
        operator_by_hotkey={_TEST_KEYPAIR.ss58_address: "operator-a"},
    )
    batcher.difficulty_auction_enabled = True
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    request = _request()
    for rollout in request.rollouts:
        rollout.env_name = env.name

    outage = client.post("/submit", json=request.model_dump(mode="json"))

    assert outage.json() == {
        "accepted": False,
        "reason": RejectReason.WORKER_DROPPED.value,
    }
    assert server._per_window_counts == {}
    assert batcher.logical_group_reservation_count == 0
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["grader_failures_by_environment"] == {
        env.name: {"unreachable": 1}
    }

    env.unavailable = False
    retry = _request()
    for rollout in retry.rollouts:
        rollout.env_name = env.name
    recovered = client.post("/submit", json=retry.model_dump(mode="json"))
    assert recovered.json()["accepted"] is True
    assert server._per_window_counts == {_TEST_KEYPAIR.ss58_address: 1}


@pytest.mark.parametrize("auction_enabled", [False, True])
def test_code_grader_crash_consumes_quota_without_protocol_fault(
    auction_enabled,
):
    from reliquary.protocol.submission import WindowState

    env = _CrashingCodeEnv()
    server = ValidatorServer()
    batcher = _batcher(
        window_start=500,
        env=env,
        operator_by_hotkey={_TEST_KEYPAIR.ss58_address: "operator-a"},
    )
    batcher.difficulty_auction_enabled = auction_enabled
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    request = _request()
    for rollout in request.rollouts:
        rollout.env_name = env.name

    crashed = client.post("/submit", json=request.model_dump(mode="json"))

    assert crashed.json() == {
        "accepted": False,
        "reason": RejectReason.WORKER_DROPPED.value,
    }
    assert server._per_window_counts == {_TEST_KEYPAIR.ss58_address: 1}
    assert batcher.logical_group_reservation_count == 1
    health = client.get("/health").json()
    assert health["grader_failures_by_environment"] == {
        env.name: {"crash": 1}
    }


@pytest.mark.parametrize("fail_at", ["length", "row"])
def test_prompt_source_outage_is_retryable_quota_neutral_and_visible(fail_at):
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.env = _UnavailableEnv(fail_at)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json")
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "prompt_source_unavailable"
    assert response.headers["retry-after"] == "30"
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts
    health = TestClient(server.app).get("/health").json()
    assert health["status"] == "degraded"
    assert health["prompt_source_unavailable_total"] == 1
    assert health["prompt_sources"][FakeEnv.name]["status"] == "degraded"


def _registration_server():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    server.configure_registration_gate()
    return server


def test_registered_hotkey_passes_without_refresh():
    server = _registration_server()
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        operator_by_hotkey={_TEST_KEYPAIR.ss58_address: "operator-a"},
    )
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["accepted"] is True


def test_registration_cache_miss_never_schedules_rpc_or_spends_quota():
    server = _registration_server()
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.REGISTRATION_UNAVAILABLE.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_unregistered_hotkey_is_rejected_without_spending_quota():
    server = _registration_server()
    server.set_registered_hotkeys({"some-other-hotkey"})
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.HOTKEY_NOT_REGISTERED.value,
    }
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_missing_registration_snapshot_fails_closed():
    server = _registration_server()
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.REGISTRATION_UNAVAILABLE.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_recent_last_known_good_registration_survives_refresh_failure():
    server = _registration_server()
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        refreshed_at=time.time() - 400,
    )
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["accepted"] is True


def test_expired_registration_snapshot_fails_closed():
    server = _registration_server()
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        refreshed_at=time.time() - REGISTERED_HOTKEY_STALE_GRACE_SECONDS - 1,
    )
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.REGISTRATION_UNAVAILABLE.value


def test_health_reports_registration_gate_state():
    server = _registration_server()
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        operator_by_hotkey={_TEST_KEYPAIR.ss58_address: "operator-a"},
    )

    health = TestClient(server.app).get("/health").json()

    assert health["registration_gate_enforced"] is True
    assert health["registered_hotkey_count"] == 1
    assert health["registered_operator_mapping_count"] == 1
    assert health["registered_operator_mapping_complete"] is True
    assert health["registration_cache_stale"] is False
    assert health["registration_cache_usable"] is True
    assert health["registration_cache_age_seconds"] >= 0
    assert health["registration_cache_next_refresh_ts"] > time.time()
    assert health["chain_client_fingerprint"]["bittensor_version"]
    assert health["chain_client_fingerprint"][
        "async_substrate_interface_version"
    ]


def test_health_surfaces_process_lifecycle_and_degrades_on_pid_pressure():
    server = ValidatorServer()
    server._process_health_snapshot = {
        "status": "critical",
        "collected_at": 1234.0,
        "zombie_processes": 900,
        "cgroup_pids_current": 600,
        "cgroup_pids_max": 1000,
        "restart_recommended": True,
        "grader": {
            "workers_spawned_total": 128,
            "worker_recycles_total": 96,
            "worker_reaped_total": 96,
        },
    }

    body = TestClient(server.app).get("/health").json()

    assert body["status"] == "degraded"
    assert body["process_lifecycle"]["zombie_processes"] == 900
    assert body["process_lifecycle"]["restart_recommended"] is True
    assert body["process_lifecycle"]["grader"][
        "worker_reaped_total"
    ] == 96


def test_health_preserves_registration_refresh_failure_after_recovery():
    server = ValidatorServer()
    server.record_registration_cache_refresh(
        success=False,
        reason="proactive",
        failure_type="TimeoutError",
    )
    server.record_registration_cache_refresh(
        success=True,
        reason="window_boundary",
    )

    health = TestClient(server.app).get("/health").json()

    assert health["registration_cache_refresh_attempts_total"] == 2
    assert health["registration_cache_refresh_successes_total"] == 1
    assert health["registration_cache_refresh_failures_total"] == 1
    assert health["registration_cache_last_refresh_failure_type"] == "TimeoutError"
    assert health["registration_cache_last_refresh_failure_reason"] == "proactive"
    assert health["registration_cache_last_refresh_reason"] == "window_boundary"


def test_health_degrades_before_registration_cache_becomes_unusable():
    server = _registration_server()
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        refreshed_at=(
            time.time() - REGISTERED_HOTKEY_CACHE_TTL_SECONDS - 1
        ),
    )

    health = TestClient(server.app).get("/health").json()

    assert health["status"] == "degraded"
    assert health["registration_cache_stale"] is True
    assert health["registration_cache_usable"] is True


def test_health_degrades_when_registration_cache_is_unavailable():
    server = _registration_server()

    health = TestClient(server.app).get("/health").json()

    assert health["status"] == "degraded"
    assert health["registration_cache_stale"] is None
    assert health["registration_cache_usable"] is False


def test_health_degrades_on_proof_plane_runtime_signal():
    server = ValidatorServer()
    server.configure_proof_scheduler_health(lambda: {
        "required": True,
        "state": "running",
        "degraded_reasons": ["active_proof_over_wall"],
    })

    health = TestClient(server.app).get("/health").json()

    assert health["status"] == "degraded"
    assert health["proof_scheduler"]["degraded_reasons"] == [
        "active_proof_over_wall"
    ]


def test_submit_503_when_no_active_batcher():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 503


def test_submit_409_on_window_mismatch():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request(window_start=999).model_dump(mode="json"))
    assert resp.status_code == 409


def test_state_endpoint_returns_grpo_batch_state():
    from reliquary.protocol.submission import WindowState
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    payload = resp.json()
    assert "protocol_version" not in payload
    assert "generation_profile_id" not in payload
    assert "generation_contract" not in payload
    assert "fill_closed" not in payload
    state = GrpoBatchState(**payload)
    assert state.window_n == 500
    assert 42 in state.cooldown_prompts


def test_fill_closed_state_advertises_cutoff_progress_and_budget(monkeypatch):
    import reliquary.validator.server as server_module
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.fill_window import FillState

    monkeypatch.setattr(server_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(server_module, "FILL_CLOSED_MAX_SECONDS", 300.0)
    now = [1_000.0]
    batcher = _batcher(window_start=500)
    batcher._time_fn = lambda: now[0]
    batcher.collection_seconds = 200.0
    batcher.mark_window_opened(
        monotonic_time=1_000.0,
        wall_time=10_000.0,
    )
    batcher.window_open_drand_chain = "quicknet"
    batcher.window_open_drand_chain_hash = "ab" * 32
    batcher.window_open_drand_round = 123
    batcher.fill_state = FillState(
        budgets={"openmathinstruct": 32},
        picks_target=2,
    )
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    first = client.get("/state").json()["fill_closed"]
    assert first == {
        "phase": "collecting",
        "generation_beacon_chain": "quicknet",
        "generation_beacon_chain_hash": "ab" * 32,
        "generation_beacon_round": 123,
        "precommit_cutoff_ts": 10_200.0,
        "precommit_seconds": 200.0,
        "max_window_seconds": 300.0,
        "picks_emitted": 0,
        "picks_target": 2,
        "picks_by_environment": {"openmathinstruct": 0},
        "admission_budgets": {"openmathinstruct": 32},
        "admitted": {"openmathinstruct": 0},
        "proven": {"openmathinstruct": 0},
        "in_flight": {"openmathinstruct": 0},
        "remaining": {"openmathinstruct": 32},
    }
    with pytest.raises(ValueError, match="provenance must be complete"):
        FillClosedWindowState.model_validate(
            {**first, "generation_beacon_chain_hash": None}
        )

    with batcher.fill_state.lock:
        batcher.fill_state.reserve("openmathinstruct")
        batcher.fill_state.record_proven("openmathinstruct")
        batcher.fill_state.record_pick("openmathinstruct")
    updated = client.get("/state").json()["fill_closed"]
    assert updated["admitted"] == {"openmathinstruct": 1}
    assert updated["proven"] == {"openmathinstruct": 1}
    assert updated["in_flight"] == {"openmathinstruct": 0}
    assert updated["remaining"] == {"openmathinstruct": 31}
    assert updated["picks_emitted"] == 1

    now[0] = 1_201.0
    assert client.get("/state").json()["fill_closed"]["phase"] == "draining"
    batcher.force_seal("test")
    assert client.get("/state").json()["fill_closed"]["phase"] == "sealed"


def test_disabled_fill_closed_keeps_legacy_state_bytes(monkeypatch):
    import reliquary.validator.server as server_module
    from reliquary.protocol.submission import WindowState

    monkeypatch.setattr(server_module, "FILL_CLOSED_ENABLED", False)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    body = TestClient(server.app).get("/state").content

    assert body == (
        b'{"state":"open","window_n":500,"anchor_block":500,'
        b'"cooldown_prompts":[],"valid_submissions":0,"checkpoint_n":0,'
        b'"checkpoint_repo_id":null,"checkpoint_revision":null,'
        b'"randomness":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"}'
    )


def test_v3_state_advertises_exact_generation_contract(monkeypatch):
    import reliquary.validator.server as server_module
    from reliquary.protocol.profiles import PROFILES
    from reliquary.protocol.submission import WindowState

    profile = PROFILES["qwen35-4b-auction-v3"]
    contract = profile.to_generation_contract()
    monkeypatch.setattr(server_module, "PROTOCOL_VERSION", 3)
    monkeypatch.setattr(
        server_module,
        "PROTOCOL_PROFILE_ID",
        profile.profile_id,
    )
    monkeypatch.setattr(
        server_module,
        "PROTOCOL_GENERATION_CONTRACT",
        contract,
    )
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    payload = TestClient(server.app).get("/state").json()

    assert payload["protocol_version"] == 3
    assert payload["generation_profile_id"] == profile.profile_id
    assert payload["generation_contract"] == contract


def test_v3_protocol_contract_rejects_wrong_version_or_profile(monkeypatch):
    import reliquary.validator.server as server_module

    monkeypatch.setattr(server_module, "PROTOCOL_VERSION", 3)
    monkeypatch.setattr(server_module, "FORCED_SEED_PROTOCOL_VERSION", 3)
    monkeypatch.setattr(
        server_module,
        "PROTOCOL_PROFILE_ID",
        "qwen35-4b-auction-v3",
    )
    monkeypatch.setattr(server_module, "FORCED_SEED_ENFORCE", False)
    check = server_module._protocol_contract_reject_reason

    assert check(
        protocol_version=2,
        generation_profile_id="",
        checkpoint_bound=True,
    ) is RejectReason.PROTOCOL_MISMATCH
    assert check(
        protocol_version=3,
        generation_profile_id="",
        checkpoint_bound=True,
    ) is RejectReason.GENERATION_CONTRACT_MISMATCH
    assert check(
        protocol_version=3,
        generation_profile_id="qwen35-4b-auction-v3",
        checkpoint_bound=True,
    ) is None


def test_state_endpoint_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "no_active_window"


def test_state_endpoint_per_env_cooldown_via_query_param():
    """Multi-env: ``/state?env=<name>`` returns THAT env's cooldown set.

    Regression for miners only ever seeing the first active batcher's
    cooldown (math) and applying it to every env, so the code env's real
    cooldown was never communicated.
    """
    from reliquary.protocol.submission import WindowState

    class _Env:
        def __init__(self, name): self.name = name
        def __len__(self): return 1000
        def get_problem(self, idx):
            return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
        def compute_reward(self, p, c): return 0.0

    def _env_batcher(env, cd):
        b = GrpoWindowBatcher(
            window_start=500, env=env, model=_ModelStub(), cooldown_map=cd,
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            completion_text_fn=lambda r: "wrong",
            drand_round_check_enabled=False,
        )
        b.randomness = "cd" * 16
        return b

    math_cd = CooldownMap(cooldown_windows=50)
    math_cd.record_batched(11, 490)
    code_cd = CooldownMap(cooldown_windows=50)
    code_cd.record_batched(22, 490)
    server = ValidatorServer()
    # Dict order = math first, so the no-arg /state reflects math (the bug).
    server.set_active_batchers({
        "openmathinstruct": _env_batcher(_Env("openmathinstruct"), math_cd),
        "opencode": _env_batcher(_Env("opencode"), code_cd),
    })
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # No-arg: unchanged — first (math) env.
    base = client.get("/state").json()
    assert 11 in base["cooldown_prompts"] and 22 not in base["cooldown_prompts"]

    # Per-env: each env's own cooldown.
    code = client.get("/state", params={"env": "opencode"}).json()
    assert 22 in code["cooldown_prompts"] and 11 not in code["cooldown_prompts"]
    math = client.get("/state", params={"env": "openmathinstruct"}).json()
    assert 11 in math["cooldown_prompts"] and 22 not in math["cooldown_prompts"]

    # Unknown env while a window is active → 404.
    unk = client.get("/state", params={"env": "nope"})
    assert unk.status_code == 404
    assert unk.json()["detail"] == "unknown_env"


def test_health_exposes_each_environment_window_independently():
    math = _batcher(window_start=500)
    code = _batcher(window_start=500)
    math.valid_count = 4
    math._valid = [SimpleNamespace(prompt_idx=i) for i in range(4)]
    math._proof_admission_count = 4
    math._proof_grading_attempts = 4
    math._proof_grading_charged = 3
    code.valid_count = 8
    code._valid = [SimpleNamespace(prompt_idx=i) for i in range(8)]
    code._proof_admission_count = 8
    code._proof_grading_attempts = 9
    code.force_seal("batch_filled")

    server = ValidatorServer()
    server.set_active_batchers({
        "openmathinstruct": math,
        "opencodeinstruct": code,
    })
    health = TestClient(server.app).get("/health").json()

    assert health["valid_submissions_count"] == 4
    assert health["difficulty_auction_productive_candidate_limit"] == 96
    assert health["difficulty_auction_primary_candidate_target"] == 64
    assert health["difficulty_auction_challenger_capacity"] == 32
    assert health["auction_early_close_mode"] == "shadow"
    assert health["auction_early_close_min_seconds"] == 60.0
    assert health["auction_collection_ceiling_seconds"] == 100.0
    assert health["window_environments"]["openmathinstruct"] == {
        "window_n": 500,
        "sealed": False,
        "force_seal_reason": None,
        "valid_submissions_count": 4,
        "distinct_valid_prompt_count": 4,
        "last_valid_submission_ts": None,
        "seconds_since_last_valid_submission": None,
        "proof_admission_count": 4,
        "proof_grading_attempts": 4,
        "proof_grading_charged": 3,
        "pending_proof_reservations": 0,
        "inflight_proof_reservations": 0,
        "reserved_payload_bytes": 0,
        "pending_payload_bytes": 0,
        "inflight_payload_bytes": 0,
        "retained_payload_bytes": 0,
        "pending_upload_precommits": 0,
        "upload_precommit_payload_bytes": 0,
        "upload_precommit_conservation": {
                "early_close": {
                    "mode": "shadow",
                    "strategy": "adaptive_gpu_quiet",
                    "minimum_collection_seconds": 60.0,
                    "maximum_collection_seconds": 100.0,
                    "quiet_seconds": 3.0,
                    "primary_candidate_target": 64,
                    "challenger_capacity": 32,
                    "pipeline_ready": False,
                    "pipeline_ready_offset_seconds": None,
                    "last_blocker": "minimum_collection",
                    "eligible_offset_seconds": None,
                    "sealed_early": False,
                    "sealed_offset_seconds": None,
                    "refusing_precommits": False,
            },
            "graded_batch_fill_offset_seconds": None,
            "graded_prefix_fill_offset_seconds": None,
            "fill_state": None,
            # v6.1 (R32): zero off the fill-closed path -- an auction
            # window has no pick pool and therefore nothing to burn.
            "fill_closed_burned_groups": 0,
            "fill_closed_burned_eos_tokens": 0,
            "accepted_receipts": 0,
            "revealed": 0,
            "revealed_terminal": 0,
            "expired": 0,
            "terminal_decisions": 0,
            "pending": 0,
            "conserved": True,
            "pending_limit": None,
            "total_limit": None,
            "peak_pending": 0,
            "capacity_reserved": 0,
            "productive_capacity_used": 3,
            "productive_capacity_limit": 96,
            "capacity_conserved": True,
            "inactive_receipts": 0,
            "capacity_rejections": {},
        },
        "collection_closed": False,
        "difficulty_auction_enabled": False,
        "difficulty_auction_proof_wall_elapsed_seconds": 0.0,
        "difficulty_auction_proof_wall_exhausted": False,
        "forensic_proof_errors_by_type": {},
        "post_trigger_proof_admission_count": 0,
        "expensive_proof_failures_by_hotkey": {},
        "expensive_proof_failures_by_operator": {},
        "grader_failures": {},
        "auction_seal_drain": {},
    }
    code_health = health["window_environments"]["opencodeinstruct"]
    assert code_health["valid_submissions_count"] == 8
    assert code_health["sealed"] is True
    assert code_health["force_seal_reason"] == "batch_filled"


def test_state_endpoint_does_not_block_on_batcher_lock():
    """Regression for the 2026-05-12 miner-timeout outage.

    The /state handler must NOT acquire batcher._lock. The submit worker
    holds that lock for the entire GRAIL verify (~5-25s per submission).
    When /state acquired the same lock synchronously on the asyncio event
    loop, 12+ miners polling at once serialised through it and timed out
    at the 60s HTTP budget — and the lock-wait starved the event loop so
    /submit, /health, /checkpoint all backed up behind it too.
    """
    import threading
    import time

    from reliquary.protocol.submission import WindowState
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # Hold batcher._lock from a thread for 2s — simulating a long GRAIL
    # verify in progress. /state must still return promptly.
    lock_held = threading.Event()
    release_lock = threading.Event()

    def _hold_lock():
        with batcher._lock:
            lock_held.set()
            release_lock.wait(timeout=5)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=1), "background thread failed to grab _lock"

    try:
        start = time.monotonic()
        resp = client.get("/state")
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert elapsed < 0.5, (
            f"/state took {elapsed:.2f}s while batcher._lock was held — "
            f"it's still on the lock hot path"
        )
        state = GrpoBatchState(**resp.json())
        assert state.window_n == 500
        assert 42 in state.cooldown_prompts
        assert state.valid_submissions == 0
    finally:
        release_lock.set()
        holder.join(timeout=1)


# --- v2.1: state-aware endpoints ---

def test_submit_rejects_when_state_not_open():
    """When state != OPEN, /submit returns a non-accepted response."""
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.TRAINING)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "window_not_active"


def test_submit_accepted_when_state_open():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True


def test_state_endpoint_returns_window_state_enum():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "open"
    assert body["window_n"] == 500
    assert body["checkpoint_n"] == 0  # no checkpoint published yet
    assert body["checkpoint_repo_id"] is None
    assert body["checkpoint_revision"] is None


def test_state_endpoint_exposes_checkpoint_when_set():
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    revision = "7" * 40
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=7,
        repo_id="aivolutionedge/reliquary-sn",
        revision=revision,
        signature="ed25519:sig",
    ))
    client = TestClient(server.app)
    resp = client.get("/state")
    body = resp.json()
    assert body["checkpoint_n"] == 7
    assert body["checkpoint_repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["checkpoint_revision"] == revision


def test_checkpoint_endpoint_404_when_none_published():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 404


def test_checkpoint_endpoint_returns_manifest_when_set():
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    revision = "a" * 40
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=42,
        repo_id="aivolutionedge/reliquary-sn",
        revision=revision,
        signature="ed25519:sig_42",
    ))
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_n"] == 42
    assert body["repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["revision"] == revision
    assert body["signature"] == "ed25519:sig_42"


def test_server_refuses_to_advertise_mutable_checkpoint_revision():
    from reliquary.validator.checkpoint import ManifestEntry

    server = ValidatorServer()
    with pytest.raises(ValueError, match="40-character commit OID"):
        server.set_current_checkpoint(
            ManifestEntry(
                checkpoint_n=42,
                repo_id="aivolutionedge/reliquary-sn",
                revision="main",
                signature="ed25519:sig_42",
            )
        )


# --- provisional response semantics ---

def test_submit_returns_submitted_under_worker_path():
    """Under uvicorn (worker path), /submit must enqueue and return the
    ``SUBMITTED`` sentinel rather than ``ACCEPTED``. The miner needs a way
    to tell "queued, validation pending" from "fully validated and in
    _valid" — the latter is reserved for the inline sync path used by
    tests.
    """
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    # Pretend a worker is running so the prod enqueue branch is taken.
    server._worker_task = object()

    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == "submitted"


def test_code_submit_routes_to_the_isolated_code_queue():
    from reliquary.protocol.submission import WindowState
    from tests.unit.test_grpo_window_batcher import PrivateRewardFakeEnv

    server = ValidatorServer()
    batcher = _batcher(
        window_start=500,
        env=PrivateRewardFakeEnv(),
    )
    server.set_active_batchers({"opencodeinstruct": batcher})
    server.set_current_state(WindowState.OPEN)
    server._worker_task = object()
    request = _request()
    for rollout in request.rollouts:
        rollout.env_name = "opencodeinstruct"

    response = TestClient(server.app).post(
        "/submit",
        json=request.model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.SUBMITTED.value
    assert server._submit_queue.qsize() == 0
    assert server._code_submit_queue.qsize() == 1


def test_full_server_queue_returns_pending_proof_reservation():
    import asyncio
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    server._worker_task = object()
    server._submit_queue = asyncio.Queue(maxsize=1)
    server._submit_queue.put_nowait((object(), object()))

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.BATCH_FILLED.value
    assert batcher.pending_proof_reservations == 0
    assert batcher.proof_grading_attempts == 0
    assert batcher.reserved_payload_bytes == 0


# --- worker drops items whose batcher is no longer active ---

@pytest.mark.asyncio
async def test_worker_drops_late_items_for_stale_batcher():
    """The worker must skip queue items whose batcher is no longer the
    server's active_batcher — those items were enqueued when the previous
    window was OPEN and would otherwise burn GRAIL compute on a sealed
    _valid that will never be archived.
    """
    import asyncio

    server = ValidatorServer()
    old_batcher = _batcher(window_start=500)
    new_batcher = _batcher(window_start=501)
    accept_calls: list[int] = []

    def _spy_accept(req):
        accept_calls.append(req.prompt_idx)
        from reliquary.protocol.submission import BatchSubmissionResponse
        return BatchSubmissionResponse(accepted=True, reason=RejectReason.ACCEPTED)

    old_batcher.accept_submission = _spy_accept
    new_batcher.accept_submission = _spy_accept

    # Active is the new batcher. The queue has a leftover stale item plus
    # one for the new batcher.
    server.set_active_batcher(new_batcher)
    old_request = _request(prompt_idx=11)
    assert old_batcher.try_reserve_proof_admission(old_request) == (True, None)
    await server._submit_queue.put((old_request, old_batcher))
    await server._submit_queue.put((_request(prompt_idx=22), new_batcher))

    # Run the worker just long enough to drain both items.
    worker_task = asyncio.create_task(server._submit_worker())
    # Queue emptiness only means the active item was dequeued. Wait until its
    # mocked accept call completes before cancelling the worker task.
    for _ in range(50):
        if server._submit_queue.empty() and accept_calls:
            break
        await asyncio.sleep(0.01)
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # Only the active-batcher item should have hit accept_submission.
    assert accept_calls == [22], (
        f"expected only prompt 22 to be processed, got {accept_calls}"
    )
    assert old_batcher.pending_proof_reservations == 0
    assert old_batcher.proof_grading_attempts == 0
