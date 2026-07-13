"""Tests for the BATCH_FILLED early-cutoff at HTTP + worker layers.

Once a batcher has sealed (B_BATCH distinct non-cooldown valid submissions
accepted), ``select_batch`` will pick those by ``arrived_at``. Further
submissions land strictly later in arrival order and therefore cannot
displace any of the already-selected entries.

Pre-fix, those late arrivals went through the full pipeline:
  * HTTP /submit queued them
  * Submit worker ran ~5–25 s of GRAIL forward pass per item
  * Result added to ``self._valid`` but never made it into the final batch

In production this caused the queue to grow into the TRAINING phase,
inflated the OPEN→TRAIN transition by 5–15 minutes on busy windows, and
burned ~30× the necessary GPU cycles when spammer hotkeys were active.

Post-fix:
  * HTTP /submit returns ``BATCH_FILLED`` the instant the batcher seals
  * The submit worker drops queue items it picks up after the seal,
    also as ``BATCH_FILLED``
  * Both paths invoke the existing late-drop callback so operators can
    measure the savings in R2 archives
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import (
    B_BATCH,
    FORCED_SEED_PROTOCOL_VERSION,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest, RejectReason, RolloutSubmission, WindowState,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.server import ValidatorServer


# ---- batcher-level is_sealed() ------------------------------------------


def _fake_batcher_marked_sealed(window_start: int = 500) -> GrpoWindowBatcher:
    """Build a minimal batcher with sufficient stubs to allow construction
    without dragging in torch/env/tokenizer, then manually flip the
    seal_flag. The HTTP early-cutoff doesn't need a real verifier — it
    only reads ``is_sealed()``.
    """
    return GrpoWindowBatcher(
        window_start=window_start,
        env=MagicMock(),
        model=MagicMock(),
        bootstrap=False,
        verify_commitment_proofs_fn=lambda **kw: True,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "",
        hash_set=None,
    )


def test_is_sealed_false_at_construction():
    b = _fake_batcher_marked_sealed()
    assert b.is_sealed() is False


def test_is_sealed_true_after_seal_flag_set():
    b = _fake_batcher_marked_sealed()
    b._seal_flag.set()
    assert b.is_sealed() is True


def test_is_sealed_does_not_create_asyncio_event():
    """is_sealed must not touch the lazy asyncio.Event — it has to be
    callable from any thread, including the submit worker's thread."""
    b = _fake_batcher_marked_sealed()
    assert b._seal_event is None
    _ = b.is_sealed()
    assert b._seal_event is None  # still lazy


# ---- HTTP /submit early-cutoff ------------------------------------------


def _submission(hotkey: str = "hkX", window_start: int = 500) -> dict:
    """Minimal JSON payload accepted by FastAPI; matches test_late_drop_server."""
    commit = {
        "tokens": list(range(36)),
        "commitments": [{"sketch": 0} for _ in range(36)],
        "proof_version": "v7",
        "model": {"name": "test", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": 4, "completion_length": 32,
            "success": True, "total_reward": 1.0, "advantage": 0.0,
            "token_logprobs": [0.0] * 36,
        },
    }
    return {
        "miner_hotkey": hotkey,
        "prompt_idx": 42,
        "window_start": window_start,
        "merkle_root": "00" * 32,
        "rollouts": [{"tokens": list(range(36)), "reward": 1.0, "commit": commit, "env_name": "openmathinstruct"}] * 8,
        "checkpoint_hash": "sha256:test",
        "protocol_version": FORCED_SEED_PROTOCOL_VERSION,
    }


def _server_with_sealed_batcher() -> tuple[ValidatorServer, GrpoWindowBatcher]:
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _fake_batcher_marked_sealed(window_start=500)
    batcher._seal_flag.set()  # pretend the batch already filled
    s.set_active_batcher(batcher)
    return s, batcher


def test_submit_returns_batch_filled_when_sealed():
    """HTTP /submit must return BATCH_FILLED reject the instant the batcher
    has sealed, even though state is still OPEN."""
    s, _ = _server_with_sealed_batcher()
    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission(hotkey="hkA"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == RejectReason.BATCH_FILLED.value


def test_submit_batch_filled_invokes_callback():
    """Late-drop callback fires with reason 'batch_filled'."""
    s, _ = _server_with_sealed_batcher()
    captured: list[tuple[str, str]] = []
    s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))
    with TestClient(s.app) as client:
        client.post("/submit", json=_submission(hotkey="hkA"))
    assert captured == [("hkA", "batch_filled")]


def test_submit_batch_filled_before_state_check():
    """When the batcher is sealed AND state is still OPEN, the response
    is BATCH_FILLED (not WINDOW_NOT_ACTIVE). The early-cutoff is a
    distinct, earlier-firing reason."""
    s, _ = _server_with_sealed_batcher()
    # state is still OPEN at this point.
    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission(hotkey="hkA"))
    assert resp.json()["reason"] == RejectReason.BATCH_FILLED.value


def test_submit_rate_limit_still_fires_first_when_sealed():
    """Rate-limit is the cheapest reject and fires before the sealed
    check. A spammer over their hotkey cap gets RATE_LIMITED even if the
    batcher has sealed — matches the existing rate-limit-before-state
    ordering for predictability."""
    s, _ = _server_with_sealed_batcher()
    with TestClient(s.app) as client:
        # First MAX hits get BATCH_FILLED (sealed reject is the first
        # one to match for an in-cap hotkey).
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW):
            r = client.post("/submit", json=_submission(hotkey="hkA"))
            assert r.json()["reason"] == RejectReason.BATCH_FILLED.value
        # N+1 — counter is at cap, rate-limit beats batch-filled.
        r = client.post("/submit", json=_submission(hotkey="hkA"))
        assert r.json()["reason"] == RejectReason.RATE_LIMITED.value


def test_submit_when_not_sealed_still_succeeds():
    """Regression: when the batcher has NOT sealed, /submit takes the
    normal path. With no worker task it runs the synchronous accept
    pipeline (TestClient mode)."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    # No seal flag set.
    batcher = _fake_batcher_marked_sealed(window_start=500)
    s.set_active_batcher(batcher)
    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission(hotkey="hkA"))
    # The submission goes through to the verification pipeline; the
    # GRAIL stub returns True so this should be ACCEPTED (or some
    # post-GRAIL reject — anything except BATCH_FILLED).
    body = resp.json()
    assert body["reason"] != RejectReason.BATCH_FILLED.value


# ---- Worker pre-GRAIL drain ---------------------------------------------


@pytest.mark.asyncio
async def test_worker_drops_post_seal_items_without_grail():
    """When the worker picks up an item from the queue and the batcher
    has sealed, it must drop the item WITHOUT calling
    ``batcher.accept_submission`` (which would run ~5-25s of GRAIL)."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _fake_batcher_marked_sealed(window_start=500)
    s.set_active_batcher(batcher)

    # Track whether the verifier was called — it must NOT be.
    grail_calls: list[BatchSubmissionRequest] = []
    original_accept = batcher.accept_submission

    def trace_accept(req):
        grail_calls.append(req)
        return original_accept(req)

    batcher.accept_submission = trace_accept  # type: ignore[method-assign]

    captured: list[tuple[str, str]] = []
    s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))

    # Build a real submission object and put it on the queue manually
    # (simulating an item that arrived BEFORE seal and was queued, then
    # the batcher sealed while the worker was busy on a prior item).
    rollouts = [
        RolloutSubmission(
            tokens=list(range(36)), reward=1.0,
            commit={
                "tokens": list(range(36)),
                "rollout": {"prompt_length": 4, "completion_length": 32},
            },
            env_name="openmathinstruct",
        ) for _ in range(8)
    ]
    req = BatchSubmissionRequest(
        miner_hotkey="hkLate", prompt_idx=999, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
    )
    # Manually flip the seal flag — the queued item is now post-seal.
    batcher._seal_flag.set()
    await s._submit_queue.put((req, batcher))

    # Run one iteration of the worker.
    worker = asyncio.create_task(s._submit_worker())
    try:
        # Wait for the queue to be processed.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if s._submit_queue.empty():
                break
        # Verifier MUST NOT have been called for the late item.
        assert grail_calls == [], (
            f"worker ran GRAIL on a post-seal item: {len(grail_calls)} calls"
        )
        # Late-drop callback fired with batch_filled.
        assert ("hkLate", "batch_filled") in captured
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_worker_processes_normally_when_not_sealed():
    """Regression: worker still runs accept_submission on pre-seal items."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _fake_batcher_marked_sealed(window_start=500)
    s.set_active_batcher(batcher)

    grail_calls: list[BatchSubmissionRequest] = []
    original_accept = batcher.accept_submission

    def trace_accept(req):
        grail_calls.append(req)
        return original_accept(req)

    batcher.accept_submission = trace_accept  # type: ignore[method-assign]

    rollouts = [
        RolloutSubmission(
            tokens=list(range(36)), reward=1.0,
            commit={
                "tokens": list(range(36)),
                "rollout": {"prompt_length": 4, "completion_length": 32},
            },
            env_name="openmathinstruct",
        ) for _ in range(8)
    ]
    req = BatchSubmissionRequest(
        miner_hotkey="hkEarly", prompt_idx=1, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
    )
    # Seal flag NOT set — worker must process normally.
    assert not batcher.is_sealed()
    await s._submit_queue.put((req, batcher))

    worker = asyncio.create_task(s._submit_worker())
    try:
        for _ in range(50):
            await asyncio.sleep(0.01)
            if s._submit_queue.empty():
                break
        # Worker DID call the verifier.
        assert len(grail_calls) == 1
        assert grail_calls[0].prompt_idx == 1
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_worker_seal_check_after_batcher_swap_check():
    """If batcher was swapped (next window opened), drop reason stays
    ``worker_dropped`` — the swap check fires first because it's the
    older, more general drain path. ``batch_filled`` only applies when
    the same batcher is still active."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    old_batcher = _fake_batcher_marked_sealed(window_start=500)
    new_batcher = _fake_batcher_marked_sealed(window_start=501)
    # Old batcher is sealed AND no longer active — swap check should win.
    old_batcher._seal_flag.set()
    s.set_active_batcher(new_batcher)

    captured: list[tuple[str, str]] = []
    s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))

    rollouts = [
        RolloutSubmission(
            tokens=list(range(36)), reward=1.0,
            commit={
                "tokens": list(range(36)),
                "rollout": {"prompt_length": 4, "completion_length": 32},
            },
            env_name="openmathinstruct",
        ) for _ in range(8)
    ]
    req = BatchSubmissionRequest(
        miner_hotkey="hkX", prompt_idx=1, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
    )
    await s._submit_queue.put((req, old_batcher))

    worker = asyncio.create_task(s._submit_worker())
    try:
        for _ in range(50):
            await asyncio.sleep(0.01)
            if s._submit_queue.empty():
                break
        assert ("hkX", "worker_dropped") in captured
        assert ("hkX", "batch_filled") not in captured
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


# ---- RejectReason enum --------------------------------------------------


def test_batch_filled_reject_reason_exists():
    assert RejectReason.BATCH_FILLED.value == "batch_filled"
