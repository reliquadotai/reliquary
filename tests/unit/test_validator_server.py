"""Validator HTTP server — v2 GRPO market endpoints."""

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    RejectReason,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


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
        "proof_version": "v5",
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


def _batcher(window_start=500, cooldown_map=None):
    batcher = GrpoWindowBatcher(
        window_start=window_start,
        env=FakeEnv(),
        model=_ModelStub(),
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "CORRECT" if r.reward > 0.5 else "wrong",
    )
    batcher.current_checkpoint_hash = "sha256:test"
    return batcher


def _request(prompt_idx=42, window_start=500, k=4, checkpoint_hash="sha256:test"):
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
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
    )


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
    state = GrpoBatchState(**resp.json())
    assert state.window_n == 500
    assert 42 in state.cooldown_prompts


def test_state_endpoint_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "no_active_window"


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
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=7,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_007",
        signature="ed25519:sig",
    ))
    client = TestClient(server.app)
    resp = client.get("/state")
    body = resp.json()
    assert body["checkpoint_n"] == 7
    assert body["checkpoint_repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["checkpoint_revision"] == "rev_sha_007"


def test_checkpoint_endpoint_404_when_none_published():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 404


def test_checkpoint_endpoint_returns_manifest_when_set():
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=42,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_042",
        signature="ed25519:sig_42",
    ))
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_n"] == 42
    assert body["repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["revision"] == "rev_sha_042"
    assert body["signature"] == "ed25519:sig_42"


# --- queue backpressure (bounded depth) ---

def test_submit_rejects_window_busy_when_queue_full():
    """When the submit queue is at or above MAX_SUBMIT_QUEUE_DEPTH, /submit
    must reject new requests with WINDOW_BUSY instead of the silent
    over-accept that used to drown miners post-seal.

    The bound only fires under the worker path (real prod): the TestClient
    sync path runs accept_submission inline. Mark ``_worker_task`` as
    non-None so the prod branch is exercised, then pre-fill the queue.
    """
    import asyncio
    from reliquary.constants import MAX_SUBMIT_QUEUE_DEPTH
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    # Pretend a worker is running so the prod enqueue branch is taken.
    server._worker_task = object()  # any non-None sentinel
    # Saturate the queue.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        for _ in range(MAX_SUBMIT_QUEUE_DEPTH):
            server._submit_queue.put_nowait((_request(), batcher))
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "window_busy"


def test_submit_accepts_when_queue_has_room():
    """Below MAX_SUBMIT_QUEUE_DEPTH, /submit returns a provisional
    accepted=True under the worker path."""
    import asyncio
    from reliquary.constants import MAX_SUBMIT_QUEUE_DEPTH
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    server._worker_task = object()
    # One item already queued, well below MAX.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        server._submit_queue.put_nowait((_request(), batcher))
        assert server._submit_queue.qsize() < MAX_SUBMIT_QUEUE_DEPTH
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == "accepted"


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
    await server._submit_queue.put((_request(prompt_idx=11), old_batcher))
    await server._submit_queue.put((_request(prompt_idx=22), new_batcher))

    # Run the worker just long enough to drain both items.
    worker_task = asyncio.create_task(server._submit_worker())
    # Yield until queue is empty.
    for _ in range(50):
        if server._submit_queue.empty():
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
