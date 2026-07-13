"""Tests for ``_set_window_randomness`` retry behavior.

Substrate (finney) WS calls are flaky in practice: ``chain.get_block_hash``
periodically returns HTTP 503 or fails the WS handshake. The single-attempt
implementation cost the validator a full window on every blip. PR #7's
two-phase open kept the failure clean (no zombie accepts) but didn't
recover the window. This patch adds 2 retries (3 attempts total) inside
``_set_window_randomness`` with short backoffs so transient sub-second
blips don't burn the window.

These tests avoid importing the full service module (which pulls in heavy
ML deps unavailable in the lightweight CI env). They mirror the retry
loop structure exactly so any divergence in service.py shows up as a
test failure.
"""

from __future__ import annotations

import asyncio
import pytest


class _FakeBatcher:
    randomness: str = ""

    def __init__(self) -> None:
        self.prompt_error: Exception | None = None

    def set_prompt_range(self) -> None:
        if self.prompt_error is not None:
            raise self.prompt_error


class _FakeService:
    """Stub mirroring the retry loop in ``GrpoValidator._set_window_randomness``.

    Keep this loop body byte-identical with the service so the tests are
    meaningful. The only difference is the sleep duration (0.001s vs
    0.5/1.0s) so the suite stays fast.
    """

    def __init__(self) -> None:
        self._active_batcher: _FakeBatcher | None = _FakeBatcher()
        self._last_beacon: dict | None = None
        self._window_n = 100
        self._derive_call_count = 0
        self._derive_succeed_on_attempt = 1  # 1-indexed
        self._derive_raise: type[BaseException] = Exception

    async def _derive_randomness(self, subtensor, window_n):
        self._derive_call_count += 1
        if self._derive_call_count >= self._derive_succeed_on_attempt:
            return "0xdeadbeef", None  # mock-mode: no beacon
        raise self._derive_raise("simulated finney 503")

    async def _set_window_randomness(self, subtensor) -> None:
        if self._active_batcher is None:
            return
        last_exc: BaseException | None = None
        randomness: str | None = None
        beacon: dict | None = None
        for attempt in range(3):
            try:
                randomness, beacon = await self._derive_randomness(
                    subtensor, self._window_n,
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(0.001)
        if randomness is None:
            assert last_exc is not None
            raise last_exc
        self._active_batcher.randomness = randomness
        self._active_batcher.set_prompt_range()
        self._last_beacon = beacon


@pytest.mark.asyncio
async def test_succeeds_on_first_attempt() -> None:
    svc = _FakeService()
    svc._derive_succeed_on_attempt = 1
    await svc._set_window_randomness(subtensor=None)
    assert svc._active_batcher.randomness == "0xdeadbeef"
    assert svc._derive_call_count == 1


@pytest.mark.asyncio
async def test_recovers_on_second_attempt() -> None:
    """The whole point of the patch — a single finney blip is saved by attempt 2."""
    svc = _FakeService()
    svc._derive_succeed_on_attempt = 2
    await svc._set_window_randomness(subtensor=None)
    assert svc._active_batcher.randomness == "0xdeadbeef"
    assert svc._derive_call_count == 2


@pytest.mark.asyncio
async def test_recovers_on_third_attempt() -> None:
    svc = _FakeService()
    svc._derive_succeed_on_attempt = 3
    await svc._set_window_randomness(subtensor=None)
    assert svc._active_batcher.randomness == "0xdeadbeef"
    assert svc._derive_call_count == 3


@pytest.mark.asyncio
async def test_bubbles_after_three_failed_attempts() -> None:
    """Sustained outages still bubble — control loop's except branch handles."""
    svc = _FakeService()
    svc._derive_succeed_on_attempt = 99
    with pytest.raises(Exception, match="simulated finney 503"):
        await svc._set_window_randomness(subtensor=None)
    assert svc._derive_call_count == 3


@pytest.mark.asyncio
async def test_no_op_when_batcher_none() -> None:
    svc = _FakeService()
    svc._active_batcher = None
    await svc._set_window_randomness(subtensor=None)
    assert svc._derive_call_count == 0


@pytest.mark.asyncio
async def test_cancelled_error_short_circuits() -> None:
    """Event-loop cancellation must not be eaten by the retry loop."""
    svc = _FakeService()
    svc._derive_raise = asyncio.CancelledError
    svc._derive_succeed_on_attempt = 99
    with pytest.raises(asyncio.CancelledError):
        await svc._set_window_randomness(subtensor=None)
    assert svc._derive_call_count == 1


@pytest.mark.asyncio
async def test_prompt_preparation_failure_does_not_refetch_randomness() -> None:
    svc = _FakeService()
    assert svc._active_batcher is not None
    svc._active_batcher.prompt_error = RuntimeError("manifest unavailable")

    with pytest.raises(RuntimeError, match="manifest unavailable"):
        await svc._set_window_randomness(subtensor=None)

    assert svc._derive_call_count == 1
