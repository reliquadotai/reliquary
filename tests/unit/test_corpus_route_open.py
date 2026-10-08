"""`GET .../open`: which of a job's prompts still have a slot, as one bit per
source row. A read like the cursor: no signature, no write, any prompt order."""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.job import parse_job
from reliquary.corpus.slots import OPEN_ENCODING, parse_open_map
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.validator import corpus_service
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    _Tokenizer,
    _manifest,
    _r2_client,
    fake_r2,
    seeded_job,
)

JOB = "open-v1"


def _declare(fake_r2, job_id=JOB, **overrides):
    raw = _manifest()
    raw.update(job_id=job_id, prompt_order="free", prompt_count=20, slots_per_prompt=2)
    raw.update(overrides)
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    return parse_job(raw)


def _seed(fake_r2, slots, job_id=JOB):
    snapshot = {
        "schema": "reliquary/corpus-ledgers/v2",
        "slots": {str(k): v for k, v in slots.items()},
        "cursors": {}, "seen_pending": [], "seen_segments": [],
    }
    _, etag = asyncio.run(job_store.read_ledgers(job_id, **fake_r2))
    asyncio.run(job_store.write_ledgers(job_id, snapshot, etag, **fake_r2))


def _router(seeded_job, job_id=JOB):
    return corpus_service.build_corpus_router(
        job_id=job_id, store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer, prompt_job_for=seeded_job.prompt_job_for,
        verify_signature=lambda request: True,
    )


def _client(router):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _open_rows(body):
    return parse_open_map(body).indices()


def test_a_free_job_names_the_prompts_that_still_have_a_slot(fake_r2, seeded_job):
    _declare(fake_r2)
    _seed(fake_r2, {0: 2, 3: 1, 7: 2, 19: 2})
    response = _client(_router(seeded_job)).get("/corpus/open")
    assert response.status_code == 200
    body = response.json()
    assert {k: body[k] for k in ("job_id", "prompt_start", "prompt_count", "open_count",
                                 "encoding")} == {
        "job_id": JOB, "prompt_start": 0, "prompt_count": 20, "open_count": 17,
        "encoding": OPEN_ENCODING}
    assert OPEN_ENCODING == "bitmap-msb0-base64"
    assert isinstance(body["as_of"], float)
    assert len(base64.b64decode(body["open"])) == 3
    assert _open_rows(body) == [i for i in range(20) if i not in (0, 7, 19)]


def test_the_open_map_agrees_with_the_slot_ledger(fake_r2, seeded_job):
    job = _declare(fake_r2)
    slots = {1: 2, 2: 1, 5: 2, 6: 2, 11: 1, 18: 2}
    _seed(fake_r2, slots)
    body = _client(_router(seeded_job)).get("/corpus/open").json()
    snapshot, _ = asyncio.run(job_store.read_ledgers(JOB, **fake_r2))
    ledger = corpus_service.rebuild_ledgers(job, snapshot).slots
    assert _open_rows(body) == [i for i in range(20) if ledger.remaining(i) > 0]
    assert body["open_count"] == len(_open_rows(body))


def test_a_job_that_starts_at_a_row_answers_in_source_indices(fake_r2, seeded_job):
    _declare(fake_r2, prompt_start=500, prompt_count=10)
    _seed(fake_r2, {500: 2, 509: 2, 504: 1})
    body = _client(_router(seeded_job)).get("/corpus/open").json()
    assert (body["prompt_start"], body["prompt_count"], body["open_count"]) == (500, 10, 8)
    assert _open_rows(body) == [501, 502, 503, 504, 505, 506, 507, 508]


def test_a_full_job_has_nothing_open(fake_r2, seeded_job):
    _declare(fake_r2, prompt_count=5)
    _seed(fake_r2, {i: 2 for i in range(5)})
    body = _client(_router(seeded_job)).get("/corpus/open").json()
    assert body["open_count"] == 0 and _open_rows(body) == []


def test_a_job_nobody_has_touched_is_open_everywhere(fake_r2, seeded_job):
    _declare(fake_r2)
    body = _client(_router(seeded_job)).get("/corpus/open").json()
    assert body["open_count"] == 20 and _open_rows(body) == list(range(20))


def test_a_miner_walk_job_answers_too(fake_r2, seeded_job):
    _declare(fake_r2, prompt_order="miner_walk", slots_per_prompt=1)
    _seed(fake_r2, {4: 1})
    body = _client(_router(seeded_job)).get("/corpus/open").json()
    assert _open_rows(body) == [i for i in range(20) if i != 4]


def test_an_unknown_job_is_a_404(seeded_job):
    response = _client(_router(seeded_job, "never-declared")).get("/corpus/open")
    assert (response.status_code, response.json()["detail"]) == (404, "corpus_job_unknown")


def test_the_read_writes_nothing_and_is_cacheable_for_seconds(fake_r2, seeded_job):
    _declare(fake_r2)
    _seed(fake_r2, {0: 2})
    before = seeded_job.ledger_writes()
    response = _client(_router(seeded_job)).get("/corpus/open")
    assert response.headers["cache-control"] == "public, max-age=5"
    assert seeded_job.ledger_writes() == before


def test_the_answer_is_served_from_memory_within_its_few_seconds(fake_r2, seeded_job, monkeypatch):
    _declare(fake_r2)
    _seed(fake_r2, {0: 2})
    now = [1000.0]
    monkeypatch.setattr(corpus_service, "_open_clock", lambda: now[0])
    client = _client(_router(seeded_job))
    first = client.get("/corpus/open").json()
    reads = []
    read_ledgers, read_job = seeded_job.store.read_ledgers, seeded_job.store.read_job
    seeded_job.store.read_ledgers = lambda job_id: reads.append("ledgers") or read_ledgers(job_id)
    seeded_job.store.read_job = lambda job_id: reads.append("job") or read_job(job_id)
    _seed(fake_r2, {0: 2, 1: 2})
    # Within the window: the same answer, and the bucket is not read at all.
    assert client.get("/corpus/open").json() == first
    assert reads == []
    now[0] += corpus_service.OPEN_CACHE_SECONDS + 0.1
    later = client.get("/corpus/open").json()
    assert later["open_count"] == 18 and later["as_of"] >= first["as_of"]


def test_a_source_too_large_for_a_bitmap_is_refused_by_name(fake_r2, seeded_job, monkeypatch):
    _declare(fake_r2)
    monkeypatch.setattr(corpus_service, "OPEN_MAX_PROMPTS", 19)
    response = _client(_router(seeded_job)).get("/corpus/open")
    assert (response.status_code, response.json()["detail"]) == (409, "corpus_job_open_too_large")


def test_the_job_scoped_path_and_the_legacy_default(fake_r2, seeded_job):
    _declare(fake_r2, "open-a-v1")
    _declare(fake_r2, "open-b-v1", prompt_count=8)
    _seed(fake_r2, {0: 2}, "open-a-v1")
    _seed(fake_r2, {3: 2, 4: 2}, "open-b-v1")
    app = FastAPI()
    app.include_router(corpus_service.build_corpus_jobs_router(
        {"open-a-v1": _router(seeded_job, "open-a-v1"),
         "open-b-v1": _router(seeded_job, "open-b-v1")}))
    client = TestClient(app)
    scoped = client.get("/corpus/jobs/open-b-v1/open")
    assert scoped.headers["cache-control"] == "public, max-age=5"
    assert _open_rows(scoped.json()) == [0, 1, 2, 5, 6, 7]
    assert client.get("/corpus/open").json()["job_id"] == "open-a-v1"
    missing = client.get("/corpus/jobs/nope/open")
    assert (missing.status_code, missing.json()["detail"]) == (404, "corpus_job_not_served")


def test_a_retired_job_answers_410(fake_r2, seeded_job):
    _declare(fake_r2)
    routes = corpus_service.CorpusJobRoutes({JOB: _router(seeded_job)})
    routes.retire(JOB)
    app = FastAPI()
    app.include_router(corpus_service.build_corpus_jobs_router(routes, legacy=True))
    client = TestClient(app)
    for path in (f"/corpus/jobs/{JOB}/open", "/corpus/open"):
        response = client.get(path)
        assert (response.status_code, response.json()["detail"]) == (410, "job_retired")


def test_a_paused_job_still_answers(fake_r2, seeded_job):
    _declare(fake_r2)
    routes = corpus_service.CorpusJobRoutes({JOB: _router(seeded_job)})
    routes.set_admission(JOB, "paused")
    app = FastAPI()
    app.include_router(corpus_service.build_corpus_jobs_router(routes, legacy=True))
    assert JOB in routes.paused
    assert TestClient(app).get(f"/corpus/jobs/{JOB}/open").json()["open_count"] == 20


@pytest.mark.parametrize("bad", [
    {"encoding": "ranges"},
    {"open": "!!!"},
    {"open": base64.b64encode(b"\xff").decode()},      # too short for the count
    {"prompt_count": "20"},
    {"prompt_count": 0},
    {"prompt_start": -1},
    {"open_count": 21},
])
def test_a_body_this_reader_cannot_trust_is_refused(bad):
    body = {"job_id": JOB, "prompt_start": 0, "prompt_count": 20, "open_count": 20,
            "as_of": 1.0, "encoding": "bitmap-msb0-base64",
            "open": base64.b64encode(b"\xff\xff\xf0").decode(), **bad}
    with pytest.raises(ValueError):
        parse_open_map(body)
