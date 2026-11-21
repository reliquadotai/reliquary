"""What a hold of a released order does to the leases in flight, through the real
routes. These tests pin TODAY's behaviour: a lease refused or left to run out during
a hold can be charged. Whether it should be is an open decision on the charging rule;
if that rule changes, the ``charged`` assertions here change with it."""

from __future__ import annotations

import asyncio
import functools
import secrets
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus import draw_leases
from reliquary.corpus.token_auction import AuctionParams, TokenAuction
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)

EMISSION = 100.0
SECONDS = 1000.0
JOB = "swe-v1"


class Clock:
    def __init__(self, now=6000.0):
        self.now = now

    def __call__(self):
        return self.now


class World:
    def __init__(self, fake_r2, monkeypatch, prompts=1):  # noqa: F811
        from reliquary.infrastructure import corpus_job_store as job_store
        from reliquary.validator.corpus_service import (
            CorpusJobRoutes, build_corpus_jobs_router, build_corpus_router,
        )
        from tests.unit.test_corpus_service import (
            _CountingStore, _Environment, _manifest, _Renderer, _SeededJob, _Tokenizer,
        )
        from tests.unit.test_token_auction_routes import _Records

        self.clock = Clock()
        self.up = True
        monkeypatch.setattr(draw_leases, "DrawLeases", functools.partial(
            draw_leases.DrawLeases, clock=self.clock, seconds=SECONDS))
        self.auction = TokenAuction(AuctionParams(ceiling=0.5 / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                                                  initial_leases=8), clock=self.clock, mean_tokens=100)
        raw = {**_manifest(), "prompt_count": prompts, "slots_per_prompt": 1, "seed": 5,
               "sampling": {**_manifest()["sampling"], "draw": "gumbel_max"}}
        asyncio.run(job_store.write_job(raw, None, **fake_r2))
        seeded = _SeededJob(_CountingStore(fake_r2), _Renderer(), _Environment(), raw)
        router = build_corpus_router(
            job_id=JOB, store=seeded.store, tokenizer=_Tokenizer(), renderer=seeded.renderer,
            verify_signature=lambda request: True, verify_lease_signature=lambda request: True,
            prompt_job_for=seeded.prompt_job_for, records=_Records(), durable_records=False, auction=self.auction)
        self.routes = CorpusJobRoutes({JOB: router})
        # The order control's hold of a judged order: held whenever its judge is not up.
        self.routes.hold(JOB, lambda: self.up, why=lambda: "awaiting_audit", latch=False)
        app = FastAPI()
        app.include_router(build_corpus_jobs_router(self.routes, legacy=False))
        self.client = TestClient(app)

    def _signed(self, hotkey):
        return {"job_id": JOB, "miner_hotkey": hotkey, "issued_at": int(time.time()),
                "nonce": secrets.token_hex(16), "signature": "ok"}

    def ask(self, hotkey, capacity=8):
        body = {**self._signed(hotkey), "micro_alpha_per_mtoken": 200_000, "capacity": capacity}
        assert self.client.post(f"/corpus/jobs/{JOB}/ask", json=body).json()["accepted"]

    def lease(self, hotkey):
        answer = self.client.post(f"/corpus/jobs/{JOB}/lease", json=self._signed(hotkey)).json()
        assert answer["leased"], answer
        return answer

    def submit(self, hotkey, lease):
        from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
        from tests.unit.test_corpus_service import CHECKPOINT, EOS, _faithful_prompt, _text_for

        tokens = [7] * 16 + [EOS]
        request = CorpusSubmissionRequest(
            job_id=JOB, miner_hotkey=hotkey, cursor=lease["slot"], prompt_index=lease["prompt_index"],
            checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(lease["prompt_index"]),
            completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok")
        return self.client.post(f"/corpus/jobs/{JOB}/submit", json=request.model_dump())

    def release(self, hotkey, lease):
        body = {"job_id": JOB, "miner_hotkey": hotkey, "lease_id": lease["lease_id"], "signature": "ok"}
        return self.client.post(f"/corpus/jobs/{JOB}/lease/release", json=body)

    def orders(self):
        # Any miner polling the job list: it runs the lease book's expiry, held or not.
        return self.client.get("/corpus/jobs").json()

    def charged(self, hotkey) -> tuple[bool, int]:
        return self.auction.debt_alpha(hotkey) > 0, len(self.auction._missed.get(hotkey, ()))


@pytest.fixture
def world(fake_r2, monkeypatch):  # noqa: F811
    return World(fake_r2, monkeypatch)


def test_a_hold_refuses_the_submission_of_a_live_lease_without_charging_it(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.up = False
    refused = world.submit("5A", lease)
    assert (refused.status_code, refused.json()["detail"]) == (409, "job_paused")
    assert world.charged("5A") == (False, 0)
    world.up = True
    assert world.submit("5A", lease).json()["accepted"] is True
    world.orders()
    assert world.charged("5A") == (False, 0)


def test_a_lease_that_runs_out_in_a_short_hold_is_still_delivered_uncharged(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.up = False
    world.clock.now += SECONDS + 1
    world.orders()
    assert world.submit("5A", lease).status_code == 409
    world.up = True
    assert world.submit("5A", lease).json()["accepted"] is True
    world.clock.now += 2 * SECONDS
    world.orders()
    assert world.charged("5A") == (False, 0)


def test_today_a_hold_outlasting_the_late_delivery_window_charges_a_refused_lease(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.up = False
    assert world.submit("5A", lease).status_code == 409
    # Past its time and the lapsed window, the judge still down and the miner still refused.
    world.clock.now += 2 * SECONDS + 1
    world.orders()
    charged, missed = world.charged("5A")
    assert charged and missed == 1
    world.up = True
    # Debt: its capacity is 1 until the debt is repaid.
    assert world.auction._capacity("5A", world.auction._offers["5A"], world.clock.now) == 1


def test_today_another_miner_taking_a_lapsed_slot_after_the_hold_charges_the_refused_holder(world):
    world.ask("5A")
    world.ask("5B")
    lease = world.lease("5A")
    world.up = False
    assert world.submit("5A", lease).status_code == 409
    world.clock.now += SECONDS + 1          # lapsed, still inside its late-delivery window
    world.orders()
    assert world.charged("5A") == (False, 0)
    world.up = True
    other = world.lease("5B")               # the first to come back gets the slot
    assert (other["prompt_index"], other["slot"]) == (lease["prompt_index"], lease["slot"])
    assert world.submit("5B", other).json()["accepted"] is True
    charged, missed = world.charged("5A")
    assert charged and missed == 1


def test_today_a_release_refused_by_the_hold_turns_a_partial_charge_into_a_full_one(fake_r2, monkeypatch):  # noqa: F811
    world = World(fake_r2, monkeypatch, prompts=2)
    world.ask("5A", capacity=1)
    world.ask("5C", capacity=1)
    early, refused = world.lease("5A"), world.lease("5C")
    world.clock.now += 0.1 * SECONDS
    assert world.release("5A", early).json()["reason"] == "released"
    partial = world.auction.debt_alpha("5A")
    assert partial > 0 and world.charged("5A")[1] == 0      # early release: a tenth, capacity kept
    world.up = False
    assert world.release("5C", refused).status_code == 409
    world.clock.now += 2 * SECONDS + 1
    world.orders()
    assert world.auction.debt_alpha("5C") == pytest.approx(10 * partial)
    assert world.charged("5C")[1] == 1
