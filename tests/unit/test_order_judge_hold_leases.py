"""A hold of an order by its own control (judge down, unseeded, stale, or a restart's seed) stops its leases' clock:
through the real routes, nothing a hold kept a miner from delivering is charged; what it still does not deliver
afterwards is charged as ever."""

from __future__ import annotations

import asyncio
import functools
import json
import random
import secrets
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus import draw_leases
from reliquary.corpus.draw_leases import DrawLeases
from reliquary.corpus.token_auction import AuctionParams, TokenAuction
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)
from tests.unit.test_draw_leases import _job

EMISSION = 100.0
SECONDS = 1000.0
JOB = "swe-v1"
HELD = "awaiting_audit"


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
            DrawLeases, clock=self.clock, seconds=SECONDS))
        self.auction = TokenAuction(AuctionParams(ceiling=0.5 / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                                                  initial_leases=8), clock=self.clock, mean_tokens=100)
        raw = {**_manifest(), "prompt_count": prompts, "slots_per_prompt": 1, "seed": 5,
               "sampling": {**_manifest()["sampling"], "draw": "gumbel_max"}}
        asyncio.run(job_store.write_job(raw, None, **fake_r2))
        seeded = _SeededJob(_CountingStore(fake_r2), _Renderer(), _Environment(), raw)
        self.routes = CorpusJobRoutes()
        self.router = build_corpus_router(
            job_id=JOB, store=seeded.store, tokenizer=_Tokenizer(), renderer=seeded.renderer,
            verify_signature=lambda request: True, verify_lease_signature=lambda request: True,
            prompt_job_for=seeded.prompt_job_for, records=_Records(), durable_records=False, auction=self.auction,
            job=seeded.job, lease_held=functools.partial(self.routes.lease_clock_stopped, JOB))
        self.routes.add(JOB, self.router)
        # As the order control holds a judged order: whenever its judge is not up.
        self.routes.hold(JOB, lambda: self.up, why=lambda: HELD, latch=False)
        self.routes.stop_lease_clock_on(JOB, HELD)
        app = FastAPI()
        app.include_router(build_corpus_jobs_router(self.routes, legacy=False))
        self.client = TestClient(app)

    def _signed(self, hotkey):
        return {"job_id": JOB, "miner_hotkey": hotkey, "issued_at": int(time.time()),
                "nonce": secrets.token_hex(16), "signature": "ok"}

    def ask(self, hotkey, capacity=8):
        body = {**self._signed(hotkey), "micro_alpha_per_mtoken": 200_000, "capacity": capacity}
        assert self.client.post(f"/corpus/jobs/{JOB}/ask", json=body).json()["accepted"]

    def lease_answer(self, hotkey):
        return self.client.post(f"/corpus/jobs/{JOB}/lease", json=self._signed(hotkey)).json()

    def lease(self, hotkey):
        answer = self.lease_answer(hotkey)
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

    def book(self) -> DrawLeases:
        return self.router.draw_leases[JOB]

    def charged(self, hotkey) -> tuple[bool, int, int]:
        return (self.auction.debt_alpha(hotkey) > 0, len(self.auction._missed.get(hotkey, ())),
                len(self.book()._unused.get(hotkey, ())))


NOT_CHARGED = (False, 0, 0)


@pytest.fixture
def world(fake_r2, monkeypatch):  # noqa: F811
    return World(fake_r2, monkeypatch)


def test_a_submission_refused_by_a_hold_lands_after_it_uncharged(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.up = False
    refused = world.submit("5A", lease)
    assert (refused.status_code, refused.json()["detail"]) == (409, "job_paused")
    world.up = True
    assert world.submit("5A", lease).json()["accepted"] is True
    world.orders()
    assert world.charged("5A") == NOT_CHARGED


def test_a_hold_longer_than_the_lease_and_its_late_window_charges_nothing(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.clock.now += 0.25 * SECONDS
    world.up = False
    world.orders()
    world.clock.now += 5 * SECONDS
    world.orders()
    assert world.charged("5A") == NOT_CHARGED
    world.up = True
    world.orders()
    assert world.charged("5A") == NOT_CHARGED
    # Its time left is what it had when the hold began: still its own, delivered at its price.
    (kept,) = world.book().held_by("5A")
    assert kept.expires_at - world.clock.now == pytest.approx(0.75 * SECONDS)
    assert kept.expires_at - kept.granted_at == pytest.approx(SECONDS)
    assert world.submit("5A", lease).json()["accepted"] is True
    assert world.charged("5A") == NOT_CHARGED


def test_a_miner_that_never_delivers_is_charged_once_its_moved_time_is_up(world):
    world.ask("5A")
    world.lease("5A")
    world.up = False
    world.orders()
    world.clock.now += 5 * SECONDS
    world.orders()
    world.up = True
    world.orders()
    world.clock.now += 2 * SECONDS - 1
    world.orders()
    assert world.charged("5A") == NOT_CHARGED
    world.clock.now += 2
    world.orders()
    charged, missed, unused = world.charged("5A")
    assert charged and missed == 1 and unused == 1


def test_a_slot_lapsed_when_a_hold_began_and_taken_by_another_after_it_is_not_charged(world):
    world.ask("5A")
    world.ask("5B")
    lease = world.lease("5A")
    world.clock.now += SECONDS + 1          # lapsed: its slot is open, its late delivery still taken
    world.orders()
    assert not world.book().held_by("5A") and world.charged("5A") == NOT_CHARGED
    world.up = False
    world.orders()
    assert world.submit("5A", lease).status_code == 409
    world.clock.now += 3 * SECONDS
    world.orders()
    world.up = True
    world.ask("5B")                         # its offer aged out during the hold: a miner posts it again
    other = world.lease("5B")
    assert (other["prompt_index"], other["slot"]) == (lease["prompt_index"], lease["slot"])
    assert world.submit("5B", other).json()["accepted"] is True
    assert world.charged("5A") == NOT_CHARGED


def test_a_lease_running_through_a_hold_keeps_its_slot_from_others_after_it(world):
    world.ask("5A")
    world.ask("5B")
    world.lease("5A")
    world.up = False
    world.orders()
    world.clock.now += 5 * SECONDS
    world.up = True
    assert world.lease_answer("5B")["leased"] is False


def test_a_lapse_with_no_hold_is_still_charged_when_another_takes_the_slot(world):
    world.ask("5A")
    world.ask("5B")
    lease = world.lease("5A")
    world.clock.now += SECONDS + 1
    world.orders()
    other = world.lease("5B")
    assert (other["prompt_index"], other["slot"]) == (lease["prompt_index"], lease["slot"])
    assert world.submit("5B", other).json()["accepted"] is True
    charged, missed, unused = world.charged("5A")
    assert charged and missed == 1 and unused == 1


def test_a_lease_can_be_released_during_a_hold_at_what_it_held_when_it_began(fake_r2, monkeypatch):  # noqa: F811
    world = World(fake_r2, monkeypatch, prompts=2)
    world.ask("5A", capacity=1)
    world.ask("5C", capacity=1)
    before, during = world.lease("5A"), world.lease("5C")
    world.clock.now += 0.1 * SECONDS
    assert world.release("5A", before).json()["reason"] == "released"
    partial = world.auction.debt_alpha("5A")
    assert partial > 0 and world.charged("5A")[1] == 0
    world.up = False
    world.orders()
    world.clock.now += 5 * SECONDS          # none of this counts as holding it
    answer = world.release("5C", during)
    assert answer.status_code == 200 and answer.json()["reason"] == "released"
    assert world.auction.debt_alpha("5C") == pytest.approx(partial)
    assert world.charged("5C")[1:] == (0, 0)


def test_an_operator_pause_still_refuses_a_release(world):
    world.ask("5A")
    lease = world.lease("5A")
    world.routes.set_admission(JOB, "paused")
    assert world.release("5A", lease).status_code == 409


def test_only_the_controls_own_hold_stops_the_clock(world):
    world.ask("5A")
    world.lease("5A")
    world.routes.hold(JOB, lambda: world.up, why=lambda: "no_gateway", latch=False)
    world.up = False
    world.clock.now += 2 * SECONDS + 1
    world.orders()
    charged, missed, unused = world.charged("5A")
    assert charged and missed == 1 and unused == 1


# -- a restart ------------------------------------------------------------------


def _free(book):
    first = book._job.prompt_start
    return lambda start: ((first + k, 0) for k in range(book._job.prompt_count))


def _book(path, clock, held, on_unused=None):
    return DrawLeases(_job(prompts=4), clock=clock, rng=random.Random(0), seconds=100.0, limit_per_hotkey=False,
                      on_unused=on_unused, state_path=path, held=held)


def test_a_restart_during_a_hold_keeps_every_lease_and_the_whole_hold_counts(tmp_path):
    clock, charged = Clock(), []
    path, hold = tmp_path / "leases.json", {"on": False}
    first = _book(path, clock, lambda: hold["on"])
    lease = first.grant("5A", _free(first), price=1e-6)
    clock.now += 20
    hold["on"] = True
    first.expire()
    assert json.loads(path.read_text())["stopped_at"] == clock.now
    clock.now += 500                               # the process is down past the lease and its late window

    back = _book(path, clock, lambda: hold["on"], on_unused=lambda l, h: charged.append(l.lease_id))
    assert [l.lease_id for l in back.held_by("5A")] == [lease.lease_id] and charged == []
    clock.now += 300
    hold["on"] = False
    back.expire()
    (kept,) = back.held_by("5A")
    assert kept.expires_at - clock.now == pytest.approx(80.0) and charged == []
    assert "stopped_at" not in json.loads(path.read_text())
    # And the move itself survives one more restart.
    again = _book(path, clock, lambda: False, on_unused=lambda l, h: charged.append(l.lease_id))
    assert again.held_by("5A")[0].expires_at == pytest.approx(kept.expires_at)


def test_the_seed_hold_after_a_normal_restart_stops_the_clock(tmp_path):
    clock, charged = Clock(), []
    path, seeding = tmp_path / "leases.json", {"on": False}
    first = _book(path, clock, lambda: seeding["on"])
    lease = first.grant("5A", _free(first), price=1e-6)
    clock.now += 10
    # A normal restart: the clock was running; the new process starts held until it has seeded.
    seeding["on"] = True
    back = _book(path, clock, lambda: seeding["on"], on_unused=lambda l, h: charged.append(l.lease_id))
    clock.now += 1000
    back.expire()
    assert charged == [] and back.held_by("5A")[0].lease_id == lease.lease_id
    seeding["on"] = False
    (kept,) = back.held_by("5A")
    assert kept.expires_at - clock.now == pytest.approx(90.0)


def test_leases_with_no_hold_ever_keep_todays_clock(tmp_path):
    clock, charged = Clock(), []
    book = DrawLeases(_job(prompts=4), clock=clock, rng=random.Random(0), seconds=100.0, limit_per_hotkey=False,
                      on_unused=lambda l, h: charged.append(h))
    book.grant("5A", _free(book), price=1e-6)
    clock.now += 201
    book.expire()
    assert charged == [1.0]
