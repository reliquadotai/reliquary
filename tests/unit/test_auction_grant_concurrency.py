"""Concurrent route grants keep the auction's allowance and total budget."""
from __future__ import annotations

import asyncio
import secrets
import threading
import time

import httpx
import pytest

from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.validator.corpus_service import build_corpus_jobs_router
from tests.unit.test_corpus_service import CHECKPOINT, EOS, _faithful_prompt, _r2_client, _text_for, fake_r2  # noqa: F401
from tests.unit.test_token_auction_routes import Clock, _ask, _auction, _lease, _route, _Records


def _lease_body():
    return {"job_id": "swe-v1", "miner_hotkey": "5A", "issued_at": int(time.time()),
            "nonce": secrets.token_hex(16), "signature": "ok"}


def _pause_first_snapshot(auction, monkeypatch):
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
    return snapshot_taken


@pytest.mark.parametrize(("capacity", "already_held"), [(1, 0), (10, 3)])
def test_concurrent_lease_requests_keep_the_allowance_and_budget(fake_r2, monkeypatch, capacity, already_held):
    auction = _auction(Clock())
    auction._budget = 4
    client, router = _route(fake_r2, auction, _Records(), prompts=8)
    assert _ask(client, "5A", 0.2, capacity=capacity)["accepted"]
    if already_held:
        assert len(_lease(client, "5A", count=already_held)["leases"]) == already_held
    snapshot_taken = _pause_first_snapshot(auction, monkeypatch)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://v") as http:
            def request():
                return http.post("/corpus/lease", json=_lease_body())

            first_request = asyncio.create_task(request())
            assert await asyncio.to_thread(snapshot_taken.wait, 5)
            second_request = asyncio.create_task(request())
            return await asyncio.gather(first_request, second_request)

    responses = [response.json() for response in asyncio.run(run())]
    assert sum(response["leased"] for response in responses) == 1, responses
    assert next(response for response in responses if not response["leased"])["reason"] == "allowance_full"
    assert router.draw_leases["swe-v1"].counts() == {"5A": already_held + 1}
    assert not auction._reserved


def test_a_swap_and_a_scoped_lease_share_the_same_allowance(fake_r2, monkeypatch):
    auction = _auction(Clock())
    client, router = _route(fake_r2, auction, _Records(), prompts=8)
    client.app.include_router(build_corpus_jobs_router({"swe-v1": router}))
    path = "/corpus/jobs/swe-v1"
    ask = {**_lease_body(), "micro_alpha_per_mtoken": 200_000, "capacity": 1, "swap": True}
    assert client.post(f"{path}/ask", json=ask).json()["accepted"]
    lease = client.post(f"{path}/lease", json=_lease_body()).json()
    assert lease["leased"]

    tokens = [7] * 16 + [EOS]
    submission = CorpusSubmissionRequest(
        job_id="swe-v1", miner_hotkey="5A", cursor=lease["slot"], prompt_index=lease["prompt_index"],
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(lease["prompt_index"]),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok",
    )
    snapshot_taken = _pause_first_snapshot(auction, monkeypatch)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://v") as http:
            submit = asyncio.create_task(http.post(f"{path}/submit", json=submission.model_dump()))
            assert await asyncio.to_thread(snapshot_taken.wait, 5), (
                submit.result().json() if submit.done() else "submission still running"
            )
            return await asyncio.gather(submit, http.post(f"{path}/lease", json=_lease_body()))

    submitted, leased = [response.json() for response in asyncio.run(run())]
    assert submitted["accepted"]
    assert len(submitted["detail"]["next_leases"]) == 1
    assert not leased["leased"] and leased["reason"] == "allowance_full"
    assert router.draw_leases["swe-v1"].counts() == {"5A": 1}
    assert not auction._reserved


@pytest.mark.parametrize("failed_step", ["lease_seconds", "grant"])
def test_a_failed_grant_releases_its_reservation_and_the_next_request_runs(fake_r2, monkeypatch, failed_step):
    auction = _auction(Clock())
    client, router = _route(fake_r2, auction, _Records(), prompts=8)
    assert _ask(client, "5A", 0.2, capacity=2)["accepted"]
    assert _lease(client, "5A")["leased"]

    def failed(*args, **kwargs):
        raise RuntimeError("lease grant failed")

    with monkeypatch.context() as patch:
        target = auction if failed_step == "lease_seconds" else router.draw_leases["swe-v1"]
        patch.setattr(target, failed_step, failed)
        with pytest.raises(RuntimeError, match="lease grant failed"):
            _lease(client, "5A")
    assert not auction._reserved
    assert _lease(client, "5A")["leased"]
