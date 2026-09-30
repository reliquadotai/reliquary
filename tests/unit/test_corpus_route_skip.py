"""`GET .../next/{hotkey}` and the signed `POST .../skip`, against the real
admission logic and the real job store over a fake bucket.

The skip is money-path: it moves a cursor. So every refusal here is also
checked for moving nothing, and the one success for moving exactly what a
`prompt_full` refusal moves."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.walk import job_walk_index
from reliquary.corpus.job import parse_job
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSkipRequest, CorpusSubmissionRequest
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    CHECKPOINT,
    EOS,
    _Tokenizer,
    _faithful_prompt,
    _manifest,
    _r2_client,
    _text_for,
    fake_r2,
    seeded_job,
)

JOB = "walk-v1"
HOTKEY = "5Hot"


def _walk_manifest(job_id=JOB, **overrides):
    raw = _manifest()
    raw.update(job_id=job_id, prompt_order="miner_walk", slots_per_prompt=1)
    raw.update(overrides)
    return raw


def _declare(fake_r2, job_id=JOB, **overrides):
    raw = _walk_manifest(job_id, **overrides)
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    return parse_job(raw)


def _seed(fake_r2, job_id, *, slots=None, cursors=None, pending=()):
    snapshot = {
        "schema": "reliquary/corpus-ledgers/v2",
        "slots": {str(k): v for k, v in (slots or {}).items()},
        "cursors": dict(cursors or {}),
        "seen_pending": sorted(pending),
        "seen_segments": [],
    }
    asyncio.run(job_store.write_ledgers(job_id, snapshot, None, **fake_r2))
    return snapshot


def _ledger(fake_r2, job_id=JOB):
    snapshot, _ = asyncio.run(job_store.read_ledgers(job_id, **fake_r2))
    return snapshot


class _Records:
    def __init__(self):
        self.written = []

    async def write_submission(self, job_id, submission_id, record):
        self.written.append(submission_id)
        return True


def _router(seeded_job, job_id=JOB, **kwargs):
    from reliquary.validator.corpus_service import build_corpus_router

    kwargs.setdefault("verify_signature", lambda request: request.signature != "bad")
    kwargs.setdefault("verify_skip_signature", lambda request: request.signature != "bad")
    return build_corpus_router(
        job_id=job_id, store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer, prompt_job_for=seeded_job.prompt_job_for, **kwargs,
    )


def _client(router):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _skip_body(job, *, cursor=0, prompt_index=None, hotkey=HOTKEY, signature="ok", job_id=None,
               to_cursor=None):
    index = job_walk_index(job, hotkey, cursor) if prompt_index is None else prompt_index
    return CorpusSkipRequest(
        job_id=job_id or job.job_id, miner_hotkey=hotkey, cursor=cursor, prompt_index=index,
        to_cursor=cursor + 1 if to_cursor is None else to_cursor, signature=signature,
    ).model_dump()


def _skip_to(job, cursor, full, hotkey=HOTKEY):
    """What `next` should name: the first later position with a free slot."""
    from reliquary.corpus.admission import MAX_SKIP_STEPS

    for step in range(cursor + 1, cursor + MAX_SKIP_STEPS):
        if job_walk_index(job, hotkey, step) not in full:
            return step
    return cursor + MAX_SKIP_STEPS


def _submission(job, *, cursor=0, hotkey=HOTKEY, filler=7, signature="ok"):
    index = job_walk_index(job, hotkey, cursor)
    tokens = [filler] * 16 + [EOS]
    return CorpusSubmissionRequest(
        job_id=job.job_id, miner_hotkey=hotkey, cursor=cursor, prompt_index=index,
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(index),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature=signature,
    )


@pytest.fixture
def walk(fake_r2, seeded_job):
    return _declare(fake_r2)


# --------------------------------------------------------------------------
# next
# --------------------------------------------------------------------------


def test_next_names_the_walk_position_of_a_fresh_hotkey(walk, seeded_job):
    client = _client(_router(seeded_job))
    body = client.get(f"/corpus/next/{HOTKEY}").json()
    assert body == {"cursor": 0, "prompt_index": job_walk_index(walk, HOTKEY, 0),
                    "slots_remaining": 1, "skip_to": 1}


def test_next_follows_the_stored_cursor_and_reports_a_full_prompt(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 3)
    _seed(fake_r2, JOB, slots={index: 1}, cursors={HOTKEY: 3})
    before = seeded_job.ledger_writes()
    body = _client(_router(seeded_job)).get(f"/corpus/next/{HOTKEY}").json()
    assert body == {"cursor": 3, "prompt_index": index, "slots_remaining": 0,
                    "skip_to": _skip_to(walk, 3, {index})}
    # A read: nothing written.
    assert seeded_job.ledger_writes() == before


def test_next_of_a_started_job_names_a_source_row(fake_r2, seeded_job):
    job = _declare(fake_r2, "walk-range-v1", prompt_start=500, prompt_count=400)
    body = _client(_router(seeded_job, "walk-range-v1")).get(f"/corpus/next/{HOTKEY}").json()
    assert body["prompt_index"] == job_walk_index(job, HOTKEY, 0)
    assert 500 <= body["prompt_index"] < 900


def test_next_of_an_unknown_job_is_404(seeded_job):
    response = _client(_router(seeded_job, "never-declared")).get(f"/corpus/next/{HOTKEY}")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# skip: the one success
# --------------------------------------------------------------------------


def test_a_full_prompt_is_skipped_and_only_the_cursor_moves(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 0)
    seeded = _seed(fake_r2, JOB, slots={index: 1}, cursors={"5Other": 4}, pending={"ab" * 32})
    records = _Records()
    accepted = []
    client = _client(_router(seeded_job, records=records, on_accepted=accepted.append))
    before = seeded_job.ledger_writes()

    body = client.post("/corpus/skip", json=_skip_body(walk)).json()

    assert body == {"reason": "accepted", "skipped": True, "cursor": 1,
                    "slots_remaining": 0, "detail": {}}
    assert _ledger(fake_r2) == {**seeded, "cursors": {"5Other": 4, HOTKEY: 1}}
    assert seeded_job.ledger_writes() == before + 1
    # Nothing paid, nothing recorded, nothing audited.
    assert records.written == [] and accepted == []
    assert client.get(f"/corpus/next/{HOTKEY}").json()["cursor"] == 1


def test_a_skip_leaves_the_ledger_a_prompt_full_submission_leaves(fake_r2, seeded_job):
    """Two identical jobs, one full prompt each: one miner submits into it and
    is refused `prompt_full`, the other skips it. The stored ledgers agree."""
    jobs = {name: _declare(fake_r2, name) for name in ("walk-a-v1", "walk-b-v1")}
    for name, job in jobs.items():
        _seed(fake_r2, name, slots={job_walk_index(job, HOTKEY, 0): 1})

    full = _client(_router(seeded_job, "walk-a-v1")).post(
        "/corpus/submit", json=_submission(jobs["walk-a-v1"]).model_dump()).json()
    assert full["reason"] == "prompt_full"
    skipped = _client(_router(seeded_job, "walk-b-v1")).post(
        "/corpus/skip", json=_skip_body(jobs["walk-b-v1"])).json()
    assert skipped["skipped"] is True

    a, b = _ledger(fake_r2, "walk-a-v1"), _ledger(fake_r2, "walk-b-v1")
    assert a["cursors"] == b["cursors"] == {HOTKEY: 1}
    assert {k: v for k, v in a.items() if k != "slots"} == {k: v for k, v in b.items() if k != "slots"}
    assert list(a["slots"].values()) == list(b["slots"].values()) == [1]


def _full_run(walk, fake_r2, length):
    full = [job_walk_index(walk, HOTKEY, c) for c in range(length)]
    if job_walk_index(walk, HOTKEY, length) in full:
        pytest.skip("walk revisits a row inside the run")
    return _seed(fake_r2, JOB, slots=dict.fromkeys(full, 1))


def test_one_skip_crosses_a_run_of_full_prompts_to_the_first_open_one(walk, fake_r2, seeded_job):
    seeded = _full_run(walk, fake_r2, 3)
    client = _client(_router(seeded_job))
    position = client.get(f"/corpus/next/{HOTKEY}").json()
    assert position["slots_remaining"] == 0 and position["skip_to"] == 3
    before = seeded_job.ledger_writes()

    answer = client.post("/corpus/skip", json=_skip_body(walk, to_cursor=3)).json()

    assert answer["skipped"] is True and answer["cursor"] == 3
    # One ledger write for the whole run.
    assert seeded_job.ledger_writes() == before + 1
    assert _ledger(fake_r2) == {**seeded, "cursors": {HOTKEY: 3}}
    assert client.get(f"/corpus/next/{HOTKEY}").json() == {
        "cursor": 3, "prompt_index": job_walk_index(walk, HOTKEY, 3), "slots_remaining": 1,
        "skip_to": _skip_to(walk, 3, set(map(int, seeded["slots"])))}
    refused = client.post("/corpus/skip", json=_skip_body(walk, cursor=3, to_cursor=4)).json()
    assert refused["reason"] == "prompt_not_full"


@pytest.mark.parametrize("open_at", [1, 2])
def test_a_skip_over_an_open_prompt_is_refused_whole(walk, fake_r2, seeded_job, open_at):
    """A miner can never jump over a prompt it could have answered."""
    full = [job_walk_index(walk, HOTKEY, c) for c in range(4) if c != open_at]
    open_index = job_walk_index(walk, HOTKEY, open_at)
    if open_index in full:
        pytest.skip("walk revisits a row inside the run")
    _seed(fake_r2, JOB, slots=dict.fromkeys(full, 1))
    answer = _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                                      _skip_body(walk, to_cursor=4), "prompt_not_full")
    assert answer["detail"] == {"cursor": open_at, "prompt_index": open_index,
                                "slots_remaining": 1}


def test_a_skip_longer_than_the_bound_is_refused(walk, fake_r2, seeded_job):
    from reliquary.corpus.admission import MAX_SKIP_STEPS

    _full_run(walk, fake_r2, 1)
    answer = _refused_without_a_write(
        seeded_job, fake_r2, _client(_router(seeded_job)),
        _skip_body(walk, to_cursor=MAX_SKIP_STEPS + 1), "malformed_submission")
    assert answer["detail"]["max_skip_steps"] == MAX_SKIP_STEPS
    # A to_cursor of 0 never parses.
    body = dict(_skip_body(walk), to_cursor=0)
    assert _client(_router(seeded_job)).post("/corpus/skip", json=body).status_code == 422


# --------------------------------------------------------------------------
# skip: refusals move nothing
# --------------------------------------------------------------------------


def _refused_without_a_write(seeded_job, fake_r2, client, body, reason, job_id=JOB):
    before_ledger = _ledger(fake_r2, job_id)
    before_writes = seeded_job.ledger_writes()
    answer = client.post("/corpus/skip", json=body).json()
    assert answer["skipped"] is False
    assert answer["reason"] == reason
    assert answer["cursor"] is None
    assert seeded_job.ledger_writes() == before_writes
    assert _ledger(fake_r2, job_id) == before_ledger
    return answer


def test_a_prompt_with_a_slot_left_is_not_skipped(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, cursors={HOTKEY: 0})
    answer = _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                                      _skip_body(walk), "prompt_not_full")
    assert answer["slots_remaining"] == 1
    assert answer["detail"] == {"cursor": 0, "prompt_index": job_walk_index(walk, HOTKEY, 0),
                                "slots_remaining": 1}


def test_a_skip_at_a_stale_cursor_is_bad_cursor(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1}, cursors={HOTKEY: 2})
    answer = _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                                      _skip_body(walk, cursor=0), "bad_cursor")
    assert answer["detail"] == {"expected": 2, "got": 0}


def test_a_skip_naming_another_full_prompt_is_prompt_mismatch(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 0)
    other = (index + 1) % walk.prompt_count
    _seed(fake_r2, JOB, slots={index: 1, other: 1})
    _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                             _skip_body(walk, prompt_index=other), "prompt_mismatch")


def test_a_bad_skip_signature_is_refused_before_the_store_is_read(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    reads = seeded_job.store.job_reads
    _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                             _skip_body(walk, signature="bad"), "bad_signature")
    assert seeded_job.store.job_reads == reads


def test_a_validator_without_a_skip_verifier_says_it_cannot_verify(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    client = _client(_router(seeded_job, verify_skip_signature=None))
    _refused_without_a_write(seeded_job, fake_r2, client, _skip_body(walk),
                             "signature_unverifiable")


def test_the_submission_verifier_is_never_asked_about_a_skip(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    asked = []
    client = _client(_router(seeded_job, verify_signature=lambda r: asked.append(r) or True,
                             verify_skip_signature=lambda r: False))
    _refused_without_a_write(seeded_job, fake_r2, client, _skip_body(walk), "bad_signature")
    assert asked == []


def test_an_unregistered_hotkey_cannot_skip(walk, fake_r2, seeded_job):
    from reliquary.validator.corpus_registration import NOT_REGISTERED

    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})

    async def registration(hotkey):
        return NOT_REGISTERED

    _refused_without_a_write(seeded_job, fake_r2,
                             _client(_router(seeded_job, registration=registration)),
                             _skip_body(walk), "hotkey_not_registered")


def test_an_unknown_registration_is_a_retryable_503(walk, fake_r2, seeded_job):
    async def registration(hotkey):
        return "unavailable"

    response = _client(_router(seeded_job, registration=registration)).post(
        "/corpus/skip", json=_skip_body(walk))
    assert response.status_code == 503


def test_a_banned_hotkey_cannot_skip(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})

    async def banned(hotkey):
        return True

    _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job, is_banned=banned)),
                             _skip_body(walk), "miner_banned")


def test_a_skip_for_another_job_is_not_served(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    answer = _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job)),
                                      _skip_body(walk, job_id="other-v1"), "job_not_served")
    assert answer["detail"] == {"job_id": "other-v1", "serves": [JOB]}


def test_a_complete_job_skips_nothing(fake_r2, seeded_job):
    job = _declare(fake_r2, "tiny-v1", prompt_count=1)
    _seed(fake_r2, "tiny-v1", slots={0: 1})
    _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job, "tiny-v1")),
                             _skip_body(job), "job_complete", job_id="tiny-v1")


def test_a_free_job_refuses_a_skip(fake_r2, seeded_job):
    job = _declare(fake_r2, "free-v1", prompt_order="free")
    _seed(fake_r2, "free-v1", slots={job_walk_index(job, HOTKEY, 0): 1})
    answer = _refused_without_a_write(seeded_job, fake_r2, _client(_router(seeded_job, "free-v1")),
                                      _skip_body(job), "malformed_submission", job_id="free-v1")
    assert answer["detail"] == {"prompt_order": "free"}


# --------------------------------------------------------------------------
# skip under contention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("skip_first", [True, False])
@pytest.mark.parametrize("full", [True, False])
def test_a_skip_racing_a_submit_moves_the_cursor_once(walk, fake_r2, seeded_job, skip_first, full):
    """One process: the ledger lock serialises them in either order, and the
    cursor lands one step on whoever wins."""
    index = job_walk_index(walk, HOTKEY, 0)
    _seed(fake_r2, JOB, slots={index: 1} if full else {})
    router = _router(seeded_job)
    skip = CorpusSkipRequest(**_skip_body(walk))
    submit = _submission(walk)

    async def race():
        calls = [router.skip_corpus(skip), router.submit_corpus(submit)]
        if not skip_first:
            calls.reverse()
        return await asyncio.gather(*calls)

    answers = asyncio.run(race())
    if not skip_first:
        answers.reverse()
    skipped, submitted = answers
    ledger = _ledger(fake_r2)
    assert ledger["cursors"] == {HOTKEY: 1}
    assert ledger["slots"] == {str(index): 1}
    if full:
        # Exactly one of them stepped over the full prompt; the other is stale.
        assert sorted([skipped.reason.value, submitted.reason.value]) in (
            ["accepted", "bad_cursor"], ["bad_cursor", "prompt_full"])
        assert submitted.accepted is False
    else:
        # The open slot goes to the work, never to the skip.
        assert submitted.accepted is True
        assert skipped.skipped is False
        assert skipped.reason.value in ("prompt_not_full", "bad_cursor")
        assert len(ledger["seen_pending"]) == 1


@pytest.mark.parametrize("skip_first", [True, False])
def test_a_range_skip_racing_a_submit_into_its_landing_prompt(walk, fake_r2, seeded_job, skip_first):
    """The skip lands ON the open prompt at cursor 2; a submit at cursor 0
    (full) races it. Whoever wins, the cursor is where one of them put it and
    the open prompt stays unanswered by the skip."""
    seeded = _full_run(walk, fake_r2, 2)
    router = _router(seeded_job)
    skip = CorpusSkipRequest(**_skip_body(walk, to_cursor=2))
    submit = _submission(walk)

    async def race():
        calls = [router.skip_corpus(skip), router.submit_corpus(submit)]
        if not skip_first:
            calls.reverse()
        answers = await asyncio.gather(*calls)
        return answers if skip_first else answers[::-1]

    skipped, submitted = asyncio.run(race())
    ledger = _ledger(fake_r2)
    assert ledger["slots"] == seeded["slots"]
    if skipped.skipped:
        assert submitted.reason.value == "bad_cursor"
        assert ledger["cursors"] == {HOTKEY: 2}
    else:
        assert submitted.reason.value == "prompt_full"
        assert skipped.reason.value == "bad_cursor"
        assert ledger["cursors"] == {HOTKEY: 1}


def test_a_skip_that_loses_the_swap_is_decided_again_on_the_winner(walk, fake_r2, seeded_job):
    """Another validator process moved this hotkey between our read and our
    write: the retry reads its ledger and refuses, rather than moving twice."""
    index = job_walk_index(walk, HOTKEY, 0)
    seeded = _seed(fake_r2, JOB, slots={index: 1})
    winner = {**seeded, "cursors": {HOTKEY: 1}}
    seeded_job.fail_next_ledger_write_with_conflict(competing_snapshot=winner)

    answer = _client(_router(seeded_job)).post("/corpus/skip", json=_skip_body(walk)).json()

    assert answer["reason"] == "bad_cursor"
    assert _ledger(fake_r2)["cursors"] == {HOTKEY: 1}


def test_a_skip_whose_swap_never_settles_is_a_503(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    seeded_job.fail_next_ledger_write_with_conflict(times=100)
    response = _client(_router(seeded_job, max_write_attempts=2)).post(
        "/corpus/skip", json=_skip_body(walk))
    assert response.status_code == 503
    assert response.json()["detail"] == "corpus_ledger_contention"


# --------------------------------------------------------------------------
# the real signatures: neither one stands in for the other
# --------------------------------------------------------------------------


def test_a_submission_signature_is_refused_as_a_skip_and_vice_versa(fake_r2, seeded_job):
    bt = pytest.importorskip("bittensor")
    from reliquary.protocol.signatures import (
        sign_corpus_skip,
        sign_corpus_submission,
        verify_corpus_signature,
        verify_corpus_skip_signature,
    )

    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    hotkey = keypair.ss58_address
    job = _declare(fake_r2)
    index = job_walk_index(job, hotkey, 0)
    _seed(fake_r2, JOB, slots={index: 1})
    client = _client(_router(seeded_job, verify_signature=verify_corpus_signature,
                             verify_skip_signature=verify_corpus_skip_signature))

    submission = _submission(job, hotkey=hotkey, signature="00").model_dump()
    submission["signature"] = sign_corpus_submission(wallet, dict(submission, signature=""))
    skip = _skip_body(job, hotkey=hotkey, signature="00")

    replayed = client.post("/corpus/skip", json=dict(skip, signature=submission["signature"]))
    assert replayed.json()["reason"] == "bad_signature"

    skip["signature"] = sign_corpus_skip(wallet, skip)
    as_submission = client.post("/corpus/submit", json=dict(submission, signature=skip["signature"]))
    assert as_submission.json()["reason"] == "bad_signature"
    assert _ledger(fake_r2)["cursors"] == {}

    assert client.post("/corpus/skip", json=skip).json()["skipped"] is True
    # A resend of the same signed skip is stale, not a second step.
    assert client.post("/corpus/skip", json=skip).json()["reason"] == "bad_cursor"
    assert _ledger(fake_r2)["cursors"] == {hotkey: 1}


# --------------------------------------------------------------------------
# old miners
# --------------------------------------------------------------------------


def test_the_submit_wire_is_unchanged_for_miners_that_never_skip(walk, fake_r2, seeded_job):
    assert set(CorpusSubmissionRequest.model_fields) == {
        "job_id", "miner_hotkey", "cursor", "prompt_index", "checkpoint_sha256",
        "rendered_prompt", "completions", "signature"}
    index = job_walk_index(walk, HOTKEY, 0)
    _seed(fake_r2, JOB, slots={index: 1})
    client = _client(_router(seeded_job))

    full = client.post("/corpus/submit", json=_submission(walk).model_dump()).json()
    assert full == {"reason": "prompt_full", "accepted": False, "slots_remaining": 0,
                    "detail": {"prompt_index": index}}
    stale = client.post("/corpus/submit", json=_submission(walk).model_dump()).json()
    assert stale == {"reason": "bad_cursor", "accepted": False, "slots_remaining": None,
                     "detail": {"expected": 1, "got": 0}}
    assert client.get(f"/corpus/cursor/{HOTKEY}").json() == {"hotkey": HOTKEY, "cursor": 1}
    nxt = job_walk_index(walk, HOTKEY, 1)
    if nxt != index:
        ok = client.post("/corpus/submit", json=_submission(walk, cursor=1).model_dump()).json()
        assert ok == {"reason": "accepted", "accepted": True, "slots_remaining": 0, "detail": {}}


# --------------------------------------------------------------------------
# several jobs
# --------------------------------------------------------------------------


@pytest.fixture
def two_walks(fake_r2, seeded_job):
    from reliquary.validator.corpus_service import build_corpus_jobs_router

    jobs = {name: _declare(fake_r2, name) for name in ("walk-a-v1", "walk-b-v1")}
    for name, job in jobs.items():
        _seed(fake_r2, name, slots={job_walk_index(job, HOTKEY, 0): 1})
    app = FastAPI()
    app.include_router(build_corpus_jobs_router(
        {name: _router(seeded_job, name) for name in jobs}))
    return jobs, TestClient(app)


def test_a_scoped_skip_moves_only_its_own_job(two_walks, fake_r2):
    jobs, client = two_walks
    b_before = _ledger(fake_r2, "walk-b-v1")
    answer = client.post("/corpus/jobs/walk-a-v1/skip", json=_skip_body(jobs["walk-a-v1"])).json()
    assert answer["skipped"] is True
    assert _ledger(fake_r2, "walk-a-v1")["cursors"] == {HOTKEY: 1}
    assert _ledger(fake_r2, "walk-b-v1") == b_before
    assert client.get(f"/corpus/jobs/walk-a-v1/next/{HOTKEY}").json()["cursor"] == 1
    b = jobs["walk-b-v1"]
    assert client.get(f"/corpus/jobs/walk-b-v1/next/{HOTKEY}").json() == {
        "cursor": 0, "prompt_index": job_walk_index(b, HOTKEY, 0), "slots_remaining": 0,
        "skip_to": _skip_to(b, 0, {job_walk_index(b, HOTKEY, 0)})}


def test_a_scoped_skip_carrying_another_jobs_body_is_not_served(two_walks, fake_r2):
    jobs, client = two_walks
    before = (_ledger(fake_r2, "walk-a-v1"), _ledger(fake_r2, "walk-b-v1"))
    answer = client.post("/corpus/jobs/walk-a-v1/skip", json=_skip_body(jobs["walk-b-v1"])).json()
    assert answer["reason"] == "job_not_served"
    assert (_ledger(fake_r2, "walk-a-v1"), _ledger(fake_r2, "walk-b-v1")) == before


def test_the_legacy_skip_dispatches_on_the_body_and_next_serves_the_first_job(two_walks, fake_r2):
    jobs, client = two_walks
    answer = client.post("/corpus/skip", json=_skip_body(jobs["walk-b-v1"])).json()
    assert answer["skipped"] is True
    assert _ledger(fake_r2, "walk-b-v1")["cursors"] == {HOTKEY: 1}
    assert _ledger(fake_r2, "walk-a-v1")["cursors"] == {}
    assert client.get(f"/corpus/next/{HOTKEY}").json()["cursor"] == 0
    unserved = client.post("/corpus/skip", json=_skip_body(jobs["walk-b-v1"], job_id="nope")).json()
    assert unserved["reason"] == "job_not_served"


def test_scoped_next_and_skip_of_an_unserved_job_are_404(two_walks):
    jobs, client = two_walks
    assert client.get(f"/corpus/jobs/nope/next/{HOTKEY}").status_code == 404
    response = client.post("/corpus/jobs/nope/skip", json=_skip_body(jobs["walk-a-v1"]))
    assert response.status_code == 404
    assert response.json()["detail"] == "corpus_job_not_served"


# --------------------------------------------------------------------------
# the validator app
# --------------------------------------------------------------------------


def _app(seeded_job, jobs, **kwargs):
    from reliquary.validator.corpus_validator import build_corpus_jobs_app

    served = [SimpleNamespace(entry=SimpleNamespace(task_id=f"task-{j}", job_id=j, contract=None),
                              job=seeded_job.job, renderer=seeded_job.renderer,
                              auditor=SimpleNamespace(enqueue=lambda sid: None), is_banned=None)
              for j in jobs]
    return TestClient(build_corpus_jobs_app(
        jobs=served, store=seeded_job.store, records=None, tokenizer=_Tokenizer(),
        verify_signature=lambda r: True, proof_chunk_tokens=None,
        prompt_job_for=seeded_job.prompt_job_for, **kwargs))


@pytest.mark.parametrize("jobs", [("walk-v1",), ("walk-v1", "walk-b-v1")])
def test_the_app_mounts_next_and_skip_on_both_paths(walk, fake_r2, seeded_job, jobs):
    for extra in jobs[1:]:
        _declare(fake_r2, extra)
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    client = _app(seeded_job, jobs, verify_skip_signature=lambda r: True)
    assert client.get(f"/corpus/next/{HOTKEY}").json()["slots_remaining"] == 0
    assert client.get(f"/corpus/jobs/{JOB}/next/{HOTKEY}").json()["slots_remaining"] == 0
    assert client.post(f"/corpus/jobs/{JOB}/skip", json=_skip_body(walk)).json()["skipped"]
    assert client.post("/corpus/skip", json=_skip_body(walk, cursor=1)).json()["reason"] in (
        "accepted", "prompt_not_full")


def test_an_app_given_no_skip_verifier_refuses_every_skip(walk, fake_r2, seeded_job):
    _seed(fake_r2, JOB, slots={job_walk_index(walk, HOTKEY, 0): 1})
    client = _app(seeded_job, (JOB,))
    assert client.post("/corpus/skip", json=_skip_body(walk)).json()["reason"] == (
        "signature_unverifiable")


def test_the_production_validator_wires_the_skip_verifier():
    import inspect

    from reliquary.validator import corpus_validator

    source = inspect.getsource(corpus_validator.run_corpus_validator)
    assert "verify_skip_signature=verify_corpus_skip_signature" in source


# --------------------------------------------------------------------------
# the read cache
# --------------------------------------------------------------------------


def _counting_rebuilds(monkeypatch):
    from reliquary.validator import corpus_service

    calls = []
    real = corpus_service.rebuild_ledgers

    def counting(job, snapshot):
        calls.append(job.job_id)
        return real(job, snapshot)

    monkeypatch.setattr(corpus_service, "rebuild_ledgers", counting)
    return calls


def test_reads_under_one_etag_rebuild_the_ledger_once(walk, fake_r2, seeded_job, monkeypatch):
    _full_run(walk, fake_r2, 2)
    calls = _counting_rebuilds(monkeypatch)
    client = _client(_router(seeded_job))
    first = client.get(f"/corpus/next/{HOTKEY}").json()
    assert client.get(f"/corpus/next/{HOTKEY}").json() == first
    assert client.get(f"/corpus/cursor/{HOTKEY}").json() == {"hotkey": HOTKEY, "cursor": 0}
    assert client.get("/corpus/cursor/5Other").json()["cursor"] == 0
    assert len(calls) == 1


def test_a_new_etag_is_rebuilt_and_read_fresh(walk, fake_r2, seeded_job, monkeypatch):
    _full_run(walk, fake_r2, 2)
    calls = _counting_rebuilds(monkeypatch)
    client = _client(_router(seeded_job))
    assert client.get(f"/corpus/cursor/{HOTKEY}").json()["cursor"] == 0
    assert client.post("/corpus/skip", json=_skip_body(walk, to_cursor=2)).json()["skipped"]
    before = len(calls)
    assert client.get(f"/corpus/cursor/{HOTKEY}").json()["cursor"] == 2
    assert client.get(f"/corpus/next/{HOTKEY}").json()["cursor"] == 2
    assert len(calls) == before + 1


def test_the_cache_is_never_moved_by_a_skip(walk, fake_r2, seeded_job):
    """The accept path decides on its own fresh copy: a cached state a skip
    had advanced would serve a cursor the bucket never saw."""
    _full_run(walk, fake_r2, 2)
    seeded_job.fail_next_ledger_write_with_conflict(times=100)
    client = _client(_router(seeded_job, max_write_attempts=2))
    assert client.get(f"/corpus/cursor/{HOTKEY}").json()["cursor"] == 0
    assert client.post("/corpus/skip", json=_skip_body(walk, to_cursor=2)).status_code == 503
    assert client.get(f"/corpus/cursor/{HOTKEY}").json()["cursor"] == 0


# --------------------------------------------------------------------------
# refusals never queue for the ledger
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stale", [
    dict(cursor=1, to_cursor=2),          # bad_cursor
    dict(prompt_index=None, to_cursor=3),  # an open prompt in the range
    dict(to_cursor=10_000),               # past the bound
])
def test_a_refused_skip_never_takes_the_ledger_lock(walk, fake_r2, seeded_job, stale):
    from fastapi import HTTPException

    _full_run(walk, fake_r2, 2)
    router = _router(seeded_job, ledger_lock_timeout=0.2)
    refused = CorpusSkipRequest(**_skip_body(walk, **stale))
    valid = CorpusSkipRequest(**_skip_body(walk, to_cursor=2))

    async def with_the_lock_held():
        await router.ledger_lock.acquire()
        try:
            answer = await router.skip_corpus(refused)
            with pytest.raises(HTTPException) as waited:
                await router.skip_corpus(valid)
            return answer, waited.value
        finally:
            router.ledger_lock.release()

    answer, waited = asyncio.run(with_the_lock_held())
    assert answer.skipped is False
    assert answer.reason.value in ("bad_cursor", "prompt_not_full", "malformed_submission")
    # The valid one did queue, and timed out behind the held lock.
    assert (waited.status_code, waited.detail) == (503, "corpus_ledger_contention")
    assert _ledger(fake_r2)["cursors"] == {}


# --------------------------------------------------------------------------
# the seen set a skip carries through
# --------------------------------------------------------------------------


def test_a_skip_over_an_unmigrated_ledger_seals_its_seen_set_like_submit(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 0)
    seen = sorted(f"{i:064x}" for i in range(5))
    v1 = {"schema": "reliquary/corpus-ledgers/v1", "slots": {str(index): 1}, "cursors": {},
          "seen": seen}
    asyncio.run(job_store.write_ledgers(JOB, v1, None, **fake_r2))

    client = _client(_router(seeded_job, seal_threshold=4, segment_max=3))
    assert client.post("/corpus/skip", json=_skip_body(walk)).json()["skipped"] is True

    after = _ledger(fake_r2)
    assert after["schema"] == "reliquary/corpus-ledgers/v2"
    assert after["seen_pending"] == []
    assert after["cursors"] == {HOTKEY: 1} and after["slots"] == {str(index): 1}
    assert [ref["count"] for ref in after["seen_segments"]] == [3, 2]
    sealed = []
    for ref in after["seen_segments"]:
        sealed += asyncio.run(job_store.read_seen_segment(JOB, ref["id"], **fake_r2))
    assert sorted(sealed) == seen


def test_a_skip_below_the_threshold_leaves_pending_as_it_is(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 0)
    seeded = _seed(fake_r2, JOB, slots={index: 1}, pending={"ab" * 32, "cd" * 32})
    client = _client(_router(seeded_job, seal_threshold=4))
    assert client.post("/corpus/skip", json=_skip_body(walk)).json()["skipped"] is True
    assert _ledger(fake_r2) == {**seeded, "cursors": {HOTKEY: 1}}


def test_a_skip_over_a_corrupt_seen_set_is_refused_by_name(walk, fake_r2, seeded_job):
    """The same integrity step submit runs: a digest both pending and sealed."""
    digest = "ef" * 32
    segment = asyncio.run(job_store.write_seen_segment(JOB, [digest], **fake_r2))
    index = job_walk_index(walk, HOTKEY, 0)
    snapshot = {"schema": "reliquary/corpus-ledgers/v2", "slots": {str(index): 1},
                "cursors": {}, "seen_pending": [digest],
                "seen_segments": [{"id": segment, "count": 1}]}
    asyncio.run(job_store.write_ledgers(JOB, snapshot, None, **fake_r2))
    before = seeded_job.ledger_writes()

    response = _client(_router(seeded_job)).post("/corpus/skip", json=_skip_body(walk))

    assert (response.status_code, response.json()["detail"]) == (500, "corpus_ledger_corrupt")
    assert seeded_job.ledger_writes() == before
    assert _ledger(fake_r2) == snapshot


def test_a_skip_over_a_ledger_naming_a_missing_segment_is_refused(walk, fake_r2, seeded_job):
    index = job_walk_index(walk, HOTKEY, 0)
    snapshot = {"schema": "reliquary/corpus-ledgers/v2", "slots": {str(index): 1},
                "cursors": {}, "seen_pending": [],
                "seen_segments": [{"id": "0" * 64, "count": 1}]}
    asyncio.run(job_store.write_ledgers(JOB, snapshot, None, **fake_r2))
    response = _client(_router(seeded_job)).post("/corpus/skip", json=_skip_body(walk))
    assert response.status_code == 500
    assert _ledger(fake_r2) == snapshot


def test_next_on_a_free_job_is_a_409_naming_why(fake_r2, seeded_job):
    _declare(fake_r2, "free-next-v1", prompt_order="free")
    response = _client(_router(seeded_job, "free-next-v1")).get(f"/corpus/next/{HOTKEY}")
    assert response.status_code == 409
    assert response.json()["detail"] == "corpus_job_not_miner_walk"
