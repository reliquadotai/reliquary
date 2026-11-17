"""Concurrent route grants keep the auction's allowance and total budget."""
from __future__ import annotations

import asyncio
import secrets
import threading
import time

import httpx
import pytest

from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401
from tests.unit.test_token_auction_routes import Clock, _ask, _auction, _lease, _route, _Records


@pytest.mark.parametrize(("capacity", "already_held"), [(1, 0), (10, 3)])
def test_concurrent_lease_requests_keep_the_allowance_and_budget(fake_r2, monkeypatch, capacity, already_held):
    auction = _auction(Clock())
    auction._budget = 4
    client, router = _route(fake_r2, auction, _Records(), prompts=8)
    assert _ask(client, "5A", 0.2, capacity=capacity)["accepted"]
    if already_held:
        assert len(_lease(client, "5A", count=already_held)["leases"]) == already_held

    snapshot_taken = threading.Event()
    another_finished = threading.Event()
    first = True
    read_lock = threading.Lock()
    held, unreserve = auction._held, auction.unreserve

    def paused_holdings():
        nonlocal first
        snapshot = held()
        with read_lock:
            pause, first = first, False
        if pause:
            snapshot_taken.set()
            # Without serialization the next request can grant and unreserve while this snapshot is stale.
            # With it the next worker waits, so let the first one proceed after the bounded pause.
            another_finished.wait(2)
        return snapshot

    def finished(terms):
        unreserve(terms)
        another_finished.set()

    monkeypatch.setattr(auction, "_held", paused_holdings)
    monkeypatch.setattr(auction, "unreserve", finished)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://v") as http:
            def request():
                return http.post("/corpus/lease", json={
                    "job_id": "swe-v1", "miner_hotkey": "5A", "issued_at": int(time.time()),
                    "nonce": secrets.token_hex(16), "signature": "ok",
                })

            first_request = asyncio.create_task(request())
            assert await asyncio.to_thread(snapshot_taken.wait, 5)
            second_request = asyncio.create_task(request())
            return await asyncio.gather(first_request, second_request)

    responses = [response.json() for response in asyncio.run(run())]
    assert sum(response["leased"] for response in responses) == 1, responses
    assert next(response for response in responses if not response["leased"])["reason"] == "allowance_full"
    assert router.draw_leases["swe-v1"].counts() == {"5A": already_held + 1}
    assert not auction._reserved


def test_a_failed_grant_releases_its_reservation_and_the_next_request_runs(fake_r2, monkeypatch):
    auction = _auction(Clock())
    client, router = _route(fake_r2, auction, _Records(), prompts=8)
    assert _ask(client, "5A", 0.2, capacity=2)["accepted"]
    assert _lease(client, "5A")["leased"]

    def failed(*args, **kwargs):
        raise RuntimeError("lease grant failed")

    with monkeypatch.context() as patch:
        patch.setattr(router.draw_leases["swe-v1"], "grant", failed)
        with pytest.raises(RuntimeError, match="lease grant failed"):
            _lease(client, "5A")
    assert not auction._reserved
    assert _lease(client, "5A")["leased"]
