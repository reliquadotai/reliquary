"""Regression seams carried from PR #215 into the newer V1 control plane."""
import asyncio
from collections import deque
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from reliquary.miner import submitter
from reliquary.protocol.submission import BatchSubmissionResponse, RejectReason
from reliquary.validator.server import ValidatorServer
from tests.unit.test_state_response_cache import _server


def test_verdict_cursor_paginates_after_ring_rollover():
    server = ValidatorServer()
    server._verdicts["miner"] = deque(maxlen=2)
    for index in range(3):
        server.record_verdict("miner", f"{index:064x}", False, RejectReason.BAD_SCHEMA)
    client = TestClient(server.app)
    first = client.get("/miner-verdicts/miner?after=0&limit=1").json()
    assert first["truncated"] and first["oldest_available_cursor"] == 2
    assert first["next_cursor"] == 2 and len(first["verdicts"]) == 1
    second = client.get("/miner-verdicts/miner?after=2&limit=1").json()
    assert not second["truncated"] and second["next_cursor"] == 3
    empty = client.get("/miner-verdicts/miner?after=3").json()
    assert empty["verdicts"] == [] and empty["next_cursor"] == 3


def test_cursor_restart_when_new_stream_has_already_caught_up():
    first = ValidatorServer()
    first.record_verdict("miner", "a" * 64, False, RejectReason.BAD_SCHEMA)
    page = TestClient(first.app).get("/miner-verdicts/miner").json()
    second = ValidatorServer()
    second.record_verdict("miner", "b" * 64, False, RejectReason.BAD_SCHEMA)
    second.record_verdict("miner", "c" * 64, False, RejectReason.BAD_SCHEMA)
    client = TestClient(second.app)
    reset = client.get("/miner-verdicts/miner", params={"after": page["next_cursor"], "stream_id": page["stream_id"]}).json()
    assert reset["truncated"] and reset["next_cursor"] == 2
    assert [v["merkle_root"] for v in reset["verdicts"]] == ["b" * 64, "c" * 64]
    assert reset["stream_id"] != page["stream_id"]
    legacy = client.get("/verdicts/miner").json()
    assert all("sequence" not in v and "_sequence" not in v for v in legacy["verdicts"])


def test_local_probes_never_build_diagnostic_health(monkeypatch):
    server = ValidatorServer()
    monkeypatch.setattr(server, "_health_payload", lambda: pytest.fail("diagnostic builder called"))
    server._process_health_snapshot = {"status": "ok"}
    client = TestClient(server.app)
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").status_code == 200
    server._content_cooldown_health_callback = lambda: {"complete": False}
    assert client.get("/readyz").status_code == 503
    assert client.get("/livez").status_code == 200
    server._content_cooldown_health_callback = lambda: {"complete": True}
    server._prompt_source_health_callback = lambda: {"math": {"status": "degraded"}}
    assert client.get("/readyz").json()["reasons"] == ["prompt_source_degraded"]


def test_miner_state_reuses_cooldown_membership():
    from tests.unit.test_miner_state_protocol import _server as miner_server

    class NoScan(list):
        def __iter__(self):
            pytest.fail("cooldown history scanned despite frozen membership")

    server = miner_server()
    for batcher in server._active_batchers.values():
        batcher.cooldown_prompts_snapshot = NoScan(batcher.cooldown_prompts_snapshot)
    assert TestClient(server.app).get("/miner-state").status_code == 200


@pytest.mark.parametrize("query", ["env=", "env=fake&env=", "env=&env=fake", "window=500&window=invalid", "window=invalid&window=500", "window="])
def test_state_fastpath_query_matches_framework(query, monkeypatch):
    server = _server()
    cached = TestClient(server.app)
    cached.get("/state")
    observed = cached.get("/state?" + query)
    monkeypatch.setenv("RELIQUARY_STATE_FASTPATH", "0")
    uncached = TestClient(_server().app).get("/state?" + query)
    assert observed.status_code == uncached.status_code
    assert observed.json() == uncached.json()


@pytest.mark.asyncio
async def test_verdict_monitor_waits_for_submission_and_cancels_pending_get(monkeypatch):
    started, cancelled, submitted = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def get(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(submitter, "get_verdicts_page_v1", get)
    async with submitter.monitor_submission_verdicts("https://test", "miner", None, submitted):
        await asyncio.sleep(0)
        assert not started.is_set()
        submitted.set()
        await asyncio.wait_for(started.wait(), 1)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_retry_after_stays_private():
    async def post(*args, **kwargs):
        return httpx.Response(503, headers={"Retry-After": "1.5"})

    response = await submitter._post_with_retry(
        "https://test", lambda attempt: b"{}", BatchSubmissionResponse,
        client=SimpleNamespace(post=post), timeout=1,
    )
    assert response._retry_after_seconds == 1.5
    assert response.model_dump() == {"accepted": False, "reason": RejectReason.WINDOW_NOT_ACTIVE}
