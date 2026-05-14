"""Late-drop callback wiring on ValidatorServer."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from reliquary.protocol.submission import (
    BatchSubmissionRequest, RejectReason, RolloutSubmission, WindowState,
)
from reliquary.validator.server import ValidatorServer


def _submission(hotkey="hkX", window_start=500) -> dict:
    """Return a JSON-serialisable submission payload (8 minimal rollouts)."""
    commit = {
        "tokens": list(range(36)),
        "commitments": [{"sketch": 0} for _ in range(36)],
        "proof_version": "v5",
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
        "rollouts": [{"tokens": list(range(36)), "reward": 1.0, "commit": commit}] * 8,
        "checkpoint_hash": "sha256:test",
    }


def test_callback_fires_when_state_not_open():
    """HTTP submit during state != OPEN must invoke the callback."""
    s = ValidatorServer()
    s.set_current_state(WindowState.TRAINING)
    captured: list[tuple[str, str]] = []
    s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))

    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission(hotkey="hkA"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == RejectReason.WINDOW_NOT_ACTIVE.value
    assert captured == [("hkA", "window_not_active")]


def test_no_callback_does_not_crash_when_state_not_open():
    """Server with no callback set still rejects cleanly when state != OPEN."""
    s = ValidatorServer()
    s.set_current_state(WindowState.TRAINING)
    # No callback registered.
    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission())
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_callback_fires_on_worker_drop():
    """When the submit worker finds the batcher has been swapped out, the
    callback is invoked with ``worker_dropped`` for that hotkey."""

    async def run():
        s = ValidatorServer()
        captured: list[tuple[str, str]] = []
        s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))

        # Build two distinct batcher stubs. The "old" one ends up not being the
        # active one when the worker pulls the item off the queue.
        old_batcher = MagicMock()
        old_batcher.window_start = 100
        new_batcher = MagicMock()
        new_batcher.window_start = 101

        s.active_batcher = new_batcher
        request = MagicMock()
        request.miner_hotkey = "hkB"
        request.prompt_idx = 7

        await s._submit_queue.put((request, old_batcher))
        # Run one iteration of the worker manually: stop it after the first
        # queue item by injecting a sentinel that raises CancelledError.
        async def runner():
            try:
                await asyncio.wait_for(s._submit_worker(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

        await runner()
        assert captured == [("hkB", "worker_dropped")]

    asyncio.run(run())
