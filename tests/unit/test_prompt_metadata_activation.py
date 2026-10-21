"""Prepare exact prompt metadata before exposing an active window to miners."""

import asyncio
import threading
from unittest.mock import AsyncMock

import httpx
import pytest

from reliquary.shared.prompt_range import window_prompt_range
from tests.unit.test_state_machine import _make_service
from tests.unit.test_validator_server import FakeEnv, _batcher


class _LazyEnv(FakeEnv):
    def __init__(self, name, rows, *, blocked=False, error=None):
        self.name = name
        self.rows = rows
        self.error = error
        self.ready = False
        self.loads = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        if not blocked:
            self.release.set()

    def __len__(self):
        if not self.ready:
            self.loads += 1
            self.started.set()
            try:
                assert self.release.wait(2), "metadata was not prepared before serving"
                if self.error is not None:
                    raise self.error
                self.ready = True
            finally:
                self.finished.set()
        return self.rows


def _service(envs, tmp_path, monkeypatch):
    from reliquary.validator import service as service_module

    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", False)
    service = _make_service()
    service.use_drand = False
    service._derive_randomness = AsyncMock(return_value=("seed", None))
    service._open_window()
    batchers = {}
    for env in envs:
        batcher = _batcher(window_start=service._candidate_window_n)
        batcher.env = env  # Keep the source cold until real window preparation.
        batchers[env.name] = batcher
    service._active_batchers = batchers
    service._set_window_preparation_stage("admission_pools")
    return service


@pytest.mark.asyncio
async def test_metadata_is_ready_off_loop_before_first_active_miner_state(tmp_path, monkeypatch):
    from reliquary.validator.server import PROMPT_RANGE_SIZE

    first = _LazyEnv("fake", 1000, blocked=True)
    second = _LazyEnv("other", 3000)
    service = _service([first, second], tmp_path, monkeypatch)
    inactive = _LazyEnv("inactive", 1, error=RuntimeError("unused source"))
    service.envs[inactive.name] = inactive
    preparation = asyncio.create_task(service.server.prepare_admission_pools(service._active_batchers))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=service.server.app), base_url="http://test"
    ) as client:
        try:
            assert await asyncio.to_thread(first.started.wait, 1)
            service._derive_randomness.assert_not_awaited()
            assert not service.server._active_batchers
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/miner-state")).status_code == 503
        finally:
            first.release.set()
            await preparation

        assert first.ready and second.ready
        assert inactive.loads == 0
        for env in (first, second):
            env.release.clear()  # A new cold read would block the first request.
        await service._set_window_randomness(None)
        service._activate_window()
        response = await asyncio.wait_for(client.get("/miner-state"), timeout=0.5)
        assert response.status_code == 200
        for env in (first, second):
            state = response.json()["environments"][env.name]
            assert state["prompt_range"] == list(window_prompt_range("seed", env.name, env.rows, PROMPT_RANGE_SIZE))
            assert env.loads == 1
        etag = response.headers["etag"]
        assert (await client.get("/miner-state", headers={"If-None-Match": etag})).status_code == 304
        service.server._miner_state_response_cache = None
        repeated = await client.get("/miner-state")
        assert repeated.content == response.content
        assert repeated.headers["etag"] == etag
        assert first.loads == second.loads == 1


@pytest.mark.asyncio
async def test_metadata_failure_leaves_candidate_unexposed_without_retrying_randomness(tmp_path, monkeypatch):
    env = _LazyEnv("fake", 1000, error=RuntimeError("source unavailable"))
    service = _service([env], tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="source unavailable"):
        await service.server.prepare_admission_pools(service._active_batchers)
        await service._set_window_randomness(None)
    service._derive_randomness.assert_not_awaited()
    assert not service.server._active_batchers


@pytest.mark.asyncio
async def test_cancelled_metadata_preparation_never_exposes_a_window(tmp_path, monkeypatch):
    env = _LazyEnv("fake", 1000, blocked=True)
    service = _service([env], tmp_path, monkeypatch)
    preparation = asyncio.create_task(service.server.prepare_admission_pools(service._active_batchers))
    try:
        assert await asyncio.to_thread(env.started.wait, 1)
        preparation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparation
    finally:
        env.release.set()
        await asyncio.to_thread(env.finished.wait, 1)
        if not preparation.done():
            preparation.cancel()
            await asyncio.gather(preparation, return_exceptions=True)
    assert not service.server._active_batchers
