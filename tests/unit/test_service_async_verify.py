"""Tests for the validator's background drand-verify task.

Pins three properties:
  1. ``_set_window_randomness`` does NOT block on the bittensor_drand
     cross-check. The hot OPEN path stays sub-400ms even when the
     cross-check takes seconds. (Verified in Task 5.)
  2. If the cross-check fails after OPEN, the batcher's
     ``beacon_invalid`` flag flips to True. (Verified in Task 5.)
  3. ``_train_and_publish`` skips seal/train/archive when the flag
     is set — preserves the security gate end-to-end. (Verified in Task 5.)

This file initially holds the Task-4-level tests pinning the new
``_derive_randomness`` return signature.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_derive_randomness_returns_tuple_in_drand_mode(monkeypatch):
    """The hot-path refactor needs the raw beacon dict so the
    background verify task can cross-check the right (round, sig).
    _derive_randomness now returns (randomness_str, beacon_dict).
    """
    from reliquary.validator import service as svc_mod

    class _StubSvc:
        use_drand = True
        async def _derive_randomness(self, subtensor, target_window):
            return await svc_mod.ValidationService._derive_randomness(
                self, subtensor, target_window,
            )

    # Stub the drand calls so the test doesn't hit the network.
    fake_beacon = {
        "round": 999,
        "randomness": "aa" * 32,
        "signature": "bb" * 48,
    }
    fake_chain_info = {"genesis_time": 1692803367, "period": 3}

    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_beacon",
        lambda round_id, use_drand: fake_beacon,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain",
        lambda: fake_chain_info,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_current_drand_round",
        lambda *a, **kw: 999,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_window_randomness",
        lambda *a, **kw: "computed_randomness_hex",
    )

    result = await _StubSvc()._derive_randomness(subtensor=None, target_window=42)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2
    randomness, beacon = result
    assert randomness == "computed_randomness_hex"
    assert beacon == fake_beacon


@pytest.mark.asyncio
async def test_derive_randomness_returns_none_beacon_in_mock_mode(monkeypatch):
    """When use_drand is False, the legacy block-hash path returns
    (randomness, None) so the verify scheduler skips cleanly."""
    from reliquary.validator import service as svc_mod

    class _StubSvc:
        use_drand = False
        async def _derive_randomness(self, subtensor, target_window):
            return await svc_mod.ValidationService._derive_randomness(
                self, subtensor, target_window,
            )

    async def _fake_block_hash(subtensor, window_n):
        return "0xdeadbeef"

    monkeypatch.setattr(
        "reliquary.infrastructure.chain.get_block_hash",
        _fake_block_hash,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.chain.compute_window_randomness",
        lambda block_hash: "mock_randomness",
    )

    randomness, beacon = await _StubSvc()._derive_randomness(
        subtensor="stub", target_window=7,
    )
    assert randomness == "mock_randomness"
    assert beacon is None


# ---------------------------------------------------------------------------
# Task 5 — background _verify_beacon_async + _train_and_publish gate
# ---------------------------------------------------------------------------


def _make_test_batcher():
    """Minimal stub mirroring just the batcher surface area the
    service uses inside _train_and_publish's gate.
    """
    class _B:
        randomness = None
        beacon_invalid = False
        def set_prompt_range(self):
            pass
        def seal_batch(self):
            raise AssertionError("seal_batch must not be called for invalid windows")
    return _B()


def _make_test_service(use_drand: bool):
    """Bypass __init__: we only need a handful of attrs to drive the
    methods under test. Mirrors the stub-Svc pattern in
    tests/unit/test_window_randomness_retry.py.
    """
    from reliquary.validator.service import ValidationService
    svc = ValidationService.__new__(ValidationService)
    svc.use_drand = use_drand
    svc._active_batcher = None
    svc._window_n = 0
    svc._verify_task = None
    svc._last_beacon = None
    svc.server = MagicMock()
    svc._checkpoint_store = MagicMock(current_manifest=lambda: object())
    svc._checkpoint_n = 0
    svc._publish_every = 100
    svc.train_model = None
    svc.verify_model = None
    return svc


@pytest.mark.asyncio
async def test_set_window_randomness_does_not_block_on_verify(monkeypatch):
    """The hot-path requirement: even with a slow (5s) verify,
    _set_window_randomness must return in <300ms so _activate_window
    can publish the OPEN state to miners on the drand boundary.
    """
    from reliquary.validator import service as svc_mod

    # 5-second blocking verify — if the hot path awaits this, the test
    # will exceed its time budget by an order of magnitude.
    def _slow_verify(*args, **kwargs):
        time.sleep(5.0)
        return True

    fake_beacon = {"round": 1, "randomness": "aa" * 32, "signature": "bb" * 48}

    async def _stub_derive(self, subtensor, target_window):
        return "computed_randomness", fake_beacon

    monkeypatch.setattr(svc_mod.ValidationService, "_derive_randomness", _stub_derive)
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.verify_beacon_signature",
        _slow_verify,
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain",
        lambda: {"hash": "test_hash", "genesis_time": 0, "period": 3},
    )

    svc = _make_test_service(use_drand=True)
    svc._active_batcher = _make_test_batcher()
    svc._window_n = 42

    t0 = time.monotonic()
    await svc._set_window_randomness(subtensor=None)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.3, (
        f"_set_window_randomness took {elapsed:.2f}s — the verify is "
        f"blocking the hot path instead of running in background."
    )
    assert svc._active_batcher.randomness == "computed_randomness"
    # Verify task should be scheduled — assert it exists then cancel
    # so the slow sleep doesn't hang the test.
    assert svc._verify_task is not None
    svc._verify_task.cancel()


@pytest.mark.asyncio
async def test_async_verify_failure_flips_beacon_invalid(monkeypatch):
    """If the background verify returns False, the batcher's
    beacon_invalid flag must be set so _train_and_publish drops
    the window.
    """
    from reliquary.validator import service as svc_mod

    fake_beacon = {"round": 7, "randomness": "aa" * 32, "signature": "bb" * 48}

    async def _stub_derive(self, subtensor, target_window):
        return "computed_randomness", fake_beacon

    monkeypatch.setattr(svc_mod.ValidationService, "_derive_randomness", _stub_derive)
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.verify_beacon_signature",
        lambda *a, **kw: False,  # forged beacon
    )
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain",
        lambda: {"hash": "test_hash", "genesis_time": 0, "period": 3},
    )

    svc = _make_test_service(use_drand=True)
    svc._active_batcher = _make_test_batcher()
    svc._window_n = 7

    await svc._set_window_randomness(subtensor=None)
    # Wait for the background task to complete.
    await svc._verify_task

    assert svc._active_batcher.beacon_invalid is True


@pytest.mark.asyncio
async def test_train_and_publish_skips_when_beacon_invalid(monkeypatch):
    """End-to-end gate: a window with beacon_invalid=True must NOT
    seal, train, publish, or archive. Miners' submissions for that
    window are dropped on the floor — same end-state as a sync-rejected
    beacon, just delivered late by the background verify.
    """
    svc = _make_test_service(use_drand=True)
    svc._active_batcher = _make_test_batcher()
    svc._active_batcher.beacon_invalid = True
    svc._window_n = 7

    # If any of these get called, the gate failed.
    seal_called = MagicMock()
    svc._active_batcher.seal_batch = seal_called

    archive_called = MagicMock()

    async def _no_archive(*a, **kw):
        archive_called()

    monkeypatch.setattr(svc, "_archive_window", _no_archive)

    await svc._train_and_publish()

    assert seal_called.call_count == 0, "seal_batch should not run for an invalidated window"
    assert archive_called.call_count == 0, "_archive_window should not run for an invalidated window"
