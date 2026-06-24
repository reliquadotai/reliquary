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
    }


def _sealed_batcher(window_start: int = 500) -> MagicMock:
    batcher = MagicMock()
    batcher.window_start = window_start
    batcher.valid_count = 0
    batcher.window_open_drand_round = None
    batcher._seal_trigger_round = None
    batcher.is_sealed.return_value = True
    return batcher


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
    assert s._per_window_counts.get("hkA", 0) == 0


def test_no_callback_does_not_crash_when_state_not_open():
    """Server with no callback set still rejects cleanly when state != OPEN."""
    s = ValidatorServer()
    s.set_current_state(WindowState.TRAINING)
    # No callback registered.
    with TestClient(s.app) as client:
        resp = client.post("/submit", json=_submission())
    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


def test_rate_limit_caps_per_hotkey_per_window():
    """After MAX submissions from the same hotkey, /submit returns RATE_LIMITED."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW

    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    s.set_active_batcher(_sealed_batcher(window_start=500))
    captured: list[tuple[str, str]] = []
    s.set_late_drop_callback(lambda hk, reason: captured.append((hk, reason)))

    with TestClient(s.app) as client:
        # First N submissions target the live window and consume quota.
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW):
            r = client.post("/submit", json=_submission(hotkey="hkA"))
            assert r.json()["reason"] == RejectReason.BATCH_FILLED.value
        # N+1th gets RATE_LIMITED instead.
        r = client.post("/submit", json=_submission(hotkey="hkA"))
        assert r.json()["reason"] == RejectReason.RATE_LIMITED.value

    # The callback was invoked N times with "batch_filled" then once with
    # "rate_limited".
    assert captured[-1] == ("hkA", "rate_limited")
    n_rate = sum(1 for _, r in captured if r == "rate_limited")
    assert n_rate == 1


def test_rate_limit_independent_across_hotkeys():
    """Each hotkey has its own counter — one saturated hotkey doesn't lock out others."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW

    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    s.set_active_batcher(_sealed_batcher(window_start=500))
    with TestClient(s.app) as client:
        # hkA saturates
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW + 1):
            client.post("/submit", json=_submission(hotkey="hkA"))
        # hkB's first submission must still go through (not rate-limited yet).
        r = client.post("/submit", json=_submission(hotkey="hkB"))
        assert r.json()["reason"] == RejectReason.BATCH_FILLED.value


def test_rate_limit_resets_on_batcher_swap():
    """Window boundary (set_active_batcher with a different batcher) clears
    the per-hotkey counter so the same hotkey can submit again next window."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW

    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    s.set_active_batcher(_sealed_batcher(window_start=500))
    with TestClient(s.app) as client:
        # Saturate hkA on window N
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW + 1):
            client.post("/submit", json=_submission(hotkey="hkA"))
        last = client.post("/submit", json=_submission(hotkey="hkA")).json()
        assert last["reason"] == RejectReason.RATE_LIMITED.value

        # New batcher → window swap → counter reset
        new_batcher = _sealed_batcher(window_start=501)
        s.set_active_batcher(new_batcher)

        # hkA's first submission on the new window goes through again.
        r = client.post("/submit", json=_submission(hotkey="hkA", window_start=501))
        assert r.json()["reason"] == RejectReason.BATCH_FILLED.value


def test_window_mismatch_does_not_burn_rate_limit_budget():
    """Signed stale-window submissions must not consume the live-window quota."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW

    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    s.set_active_batcher(_sealed_batcher(window_start=500))

    with TestClient(s.app) as client:
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW + 1):
            r = client.post(
                "/submit",
                json=_submission(hotkey="hkA", window_start=499),
            )
            assert r.status_code == 409
        assert s._per_window_counts.get("hkA", 0) == 0

        r = client.post("/submit", json=_submission(hotkey="hkA", window_start=500))
        assert r.status_code == 200
        assert r.json()["reason"] == RejectReason.BATCH_FILLED.value


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
