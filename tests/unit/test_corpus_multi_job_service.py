"""One HTTP face for several corpus jobs: submit dispatches on `job_id`, reads
are job-scoped, and the legacy reads serve the first job listed (the operator's
default, so live miners keep working when a job is added)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from tests.unit.test_corpus_service import (  # noqa: F401
    CHECKPOINT, EOS, _Tokenizer, _faithful_prompt, _manifest, _r2_client, _text_for,
    fake_r2, seeded_job,
)


def _router(seeded_job, job_id):
    from reliquary.validator.corpus_service import build_corpus_router

    return build_corpus_router(
        job_id=job_id, store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer, verify_signature=lambda request: True,
        prompt_job_for=seeded_job.prompt_job_for,
    )


def _app(routers):
    from reliquary.validator.corpus_service import build_corpus_jobs_router

    app = FastAPI()
    if len(routers) == 1:
        app.include_router(next(iter(routers.values())))
    app.include_router(build_corpus_jobs_router(routers))
    return TestClient(app)


@pytest.fixture
def two_jobs(seeded_job, fake_r2):
    raw = {**_manifest(), "job_id": "swe-v2"}
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    return _app({"swe-v1": _router(seeded_job, "swe-v1"),
                 "swe-v2": _router(seeded_job, "swe-v2")})


@pytest.fixture
def one_job(seeded_job):
    return _app({"swe-v1": _router(seeded_job, "swe-v1")})


def _body(job_id, hotkey="5Hot"):
    tokens = [7] * 16 + [EOS]
    return CorpusSubmissionRequest(
        job_id=job_id, miner_hotkey=hotkey, cursor=0, prompt_index=0,
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(0),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok",
    ).model_dump()


def _ledger(job_id):
    snapshot, _ = asyncio.run(job_store.read_ledgers(job_id))
    return snapshot


def test_the_jobs_list_names_every_served_job(two_jobs):
    assert two_jobs.get("/corpus/jobs").json() == {"jobs": ["swe-v1", "swe-v2"]}


def test_a_job_scoped_read_answers_for_that_job(two_jobs):
    assert two_jobs.get("/corpus/jobs/swe-v2/job").json()["job_id"] == "swe-v2"
    assert two_jobs.get("/corpus/jobs/swe-v1/job").json()["job_id"] == "swe-v1"
    assert two_jobs.get("/corpus/jobs/swe-v2/cursor/5Hot").json() == {"hotkey": "5Hot", "cursor": 0}


def test_a_job_scoped_read_of_an_unserved_job_is_404(two_jobs):
    for path in ("/corpus/jobs/nope/job", "/corpus/jobs/nope/cursor/5Hot"):
        response = two_jobs.get(path)
        assert response.status_code == 404
        assert response.json() == {"detail": "corpus_job_not_served"}


def test_the_legacy_reads_serve_the_first_listed_job(seeded_job, fake_r2):
    asyncio.run(job_store.write_job({**_manifest(), "job_id": "swe-v2"}, None, **fake_r2))
    # Listed second-first: the default is the operator's order, not the sorted one.
    client = _app({"swe-v2": _router(seeded_job, "swe-v2"),
                   "swe-v1": _router(seeded_job, "swe-v1")})

    assert client.get("/corpus/job").json()["job_id"] == "swe-v2"
    assert client.get("/corpus/cursor/5Hot").json() == {"hotkey": "5Hot", "cursor": 0}
    assert client.get("/corpus/jobs").json() == {"jobs": ["swe-v1", "swe-v2"]}
    # A legacy miner's submission names its job, and still reaches only that one.
    assert client.post("/corpus/submit", json=_body("swe-v2")).json()["accepted"] is True
    assert _ledger("swe-v2") and not _ledger("swe-v1")


def test_a_submission_reaches_only_its_own_jobs_ledgers(two_jobs):
    body = two_jobs.post("/corpus/submit", json=_body("swe-v2")).json()

    assert body["accepted"] is True
    assert _ledger("swe-v2")
    assert not _ledger("swe-v1")
    # The same work is new to the other job: its ledgers are its own.
    assert two_jobs.post("/corpus/submit", json=_body("swe-v1")).json()["accepted"] is True


def test_a_submission_for_an_unserved_job_lists_every_served_one(two_jobs):
    body = two_jobs.post("/corpus/submit", json=_body("other-job")).json()

    assert body["accepted"] is False
    assert body["reason"] == "job_not_served"
    assert body["detail"] == {"job_id": "other-job", "serves": ["swe-v1", "swe-v2"]}


def test_one_job_keeps_the_legacy_routes_and_gains_the_scoped_ones(one_job):
    assert one_job.get("/corpus/job").json()["job_id"] == "swe-v1"
    assert one_job.get("/corpus/cursor/5Hot").json() == {"hotkey": "5Hot", "cursor": 0}
    assert one_job.get("/corpus/jobs").json() == {"jobs": ["swe-v1"]}
    assert one_job.get("/corpus/jobs/swe-v1/job").json()["job_id"] == "swe-v1"

    refused = one_job.post("/corpus/submit", json=_body("other-job")).json()
    # Always a list, one job or several.
    assert refused["detail"] == {"job_id": "other-job", "serves": ["swe-v1"]}
    assert one_job.post("/corpus/submit", json=_body("swe-v1")).json()["accepted"] is True


def test_two_jobs_on_one_source_split_it_by_range(seeded_job, fake_r2):
    """The SFT/RL split: one source, two jobs owning disjoint row ranges. Each
    admits only its own rows, keys its ledger by them, and a miner reading
    either job's manifest walks inside that job's range."""
    from reliquary.corpus.job import parse_job
    from reliquary.corpus.walk import job_walk_index

    for job_id, start in (("sft-v1", 0), ("rl-v1", 500)):
        raw = {**_manifest(), "job_id": job_id, "prompt_order": "miner_walk",
               "prompt_count": 500, **({"prompt_start": start} if start else {})}
        asyncio.run(job_store.write_job(raw, None, **fake_r2))
    client = _app({"sft-v1": _router(seeded_job, "sft-v1"),
                   "rl-v1": _router(seeded_job, "rl-v1")})

    def post(job_id, index, cursor):
        tokens = [7] * 16 + [EOS]
        body = CorpusSubmissionRequest(
            job_id=job_id, miner_hotkey="5Hot", cursor=cursor, prompt_index=index,
            checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(index),
            completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok",
        ).model_dump()
        return client.post("/corpus/submit", json=body).json()

    jobs = {job_id: parse_job(client.get(f"/corpus/jobs/{job_id}/job").json())
            for job_id in ("sft-v1", "rl-v1")}
    assert "prompt_start" not in client.get("/corpus/jobs/sft-v1/job").json()
    assert (jobs["sft-v1"].prompt_start, jobs["rl-v1"].prompt_start) == (0, 500)

    taken = {}
    for job_id, job in jobs.items():
        walked = [job_walk_index(job, "5Hot", cursor) for cursor in range(3)]
        assert all(job.owns(index) for index in walked)
        for cursor, index in enumerate(walked):
            assert post(job_id, index, cursor)["accepted"] is True
        taken[job_id] = {int(i) for i in _ledger(job_id)["slots"]}
        assert taken[job_id] == set(walked)
    assert max(taken["sft-v1"]) < 500 <= min(taken["rl-v1"])

    # A row of the other job's range is not this job's, whatever the cursor.
    crossed = post("sft-v1", 500 + 1, 3)
    assert crossed["reason"] == "prompt_mismatch"
    assert post("rl-v1", 499, 3)["reason"] == "prompt_mismatch"
