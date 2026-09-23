"""What the route hands the rest of the validator once it has admitted."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.protocol.signatures import corpus_submission_id
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    CHECKPOINT, EOS, _Tokenizer, _faithful_prompt, _r2_client, _text_for, fake_r2, seeded_job,
)


class _Records:
    def __init__(self, fail=0):
        self.written = {}
        self.fail = fail

    async def write_submission(self, job_id, submission_id, record):
        if self.fail:
            self.fail -= 1
            raise OSError("bucket down")
        self.written[submission_id] = record
        return True


class _AlreadyRecordedRecords:
    """A store that has already seen this exact submission: create-only, so
    the write reports False rather than raising."""

    def __init__(self):
        self.write_calls = 0

    async def write_submission(self, job_id, submission_id, record):
        self.write_calls += 1
        return False


def _client(seeded_job, records, accepted, chunk=None):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer, verify_signature=lambda r: True,
        prompt_job_for=seeded_job.prompt_job_for, records=records,
        on_accepted=accepted.append, proof_chunk_tokens=chunk,
    ))
    return TestClient(app)


def _request(tokens, proofs=None, prompt_index=0):
    return CorpusSubmissionRequest(
        job_id="swe-v1", miner_hotkey="5Hot", cursor=0, prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(prompt_index),
        completions=[{"tokens": tokens, "text": _text_for(tokens), "proofs": proofs or []}],
        signature="ok",
    )


def test_an_accepted_submission_is_recorded_and_announced(seeded_job):
    records, accepted = _Records(), []
    request = _request([7] * 16 + [EOS])
    body = _client(seeded_job, records, accepted).post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is True
    sid = corpus_submission_id(request)
    assert accepted == [sid]
    record = records.written[sid]
    assert record["hotkey"] == "5Hot" and record["token_count"] == 17
    assert record["completions"][0]["tokens"] == [7] * 16 + [EOS]


def test_a_refused_submission_is_neither_recorded_nor_announced(seeded_job):
    records, accepted = _Records(), []
    request = _request([7] * 3 + [EOS])  # under the job's floor of 16
    body = _client(seeded_job, records, accepted).post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is False
    assert records.written == {} and accepted == []


def test_a_transient_record_failure_is_retried(seeded_job):
    records, accepted = _Records(fail=2), []
    request = _request([7] * 16 + [EOS])
    _client(seeded_job, records, accepted).post("/corpus/submit", json=request.model_dump())
    assert accepted == [corpus_submission_id(request)]


def test_a_record_that_never_lands_is_not_announced(seeded_job):
    records, accepted = _Records(fail=10), []
    request = _request([7] * 16 + [EOS])
    body = _client(seeded_job, records, accepted).post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is True  # the slot is consumed either way
    assert accepted == []


def test_missing_proofs_are_refused_when_the_task_proves(seeded_job):
    records, accepted = _Records(), []
    request = _request([7] * 40 + [EOS])  # 41 tokens need 2 chunks of 32
    body = _client(seeded_job, records, accepted, chunk=32).post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is False and body["reason"] == "bad_proof_shape"


def test_the_miner_reads_the_job_and_its_cursor(seeded_job):
    client = _client(seeded_job, _Records(), [])
    assert client.get("/corpus/job").json()["job_id"] == "swe-v1"
    assert client.get("/corpus/cursor/5Hot").json() == {"hotkey": "5Hot", "cursor": 0}


def test_an_already_recorded_submission_is_not_announced(seeded_job):
    """``write_submission`` is create-only: a resend of the same signed
    submission returns False rather than raising, and that is not a failure
    to retry -- it means the record is already queued under this id."""
    records, accepted = _AlreadyRecordedRecords(), []
    request = _request([7] * 16 + [EOS])
    body = _client(seeded_job, records, accepted).post("/corpus/submit", json=request.model_dump()).json()
    assert body["accepted"] is True
    assert records.write_calls == 1  # no retry: False is not an exception
    assert accepted == []


def test_an_on_accepted_that_raises_still_returns_the_accepted_response(seeded_job):
    """A subscriber's own bug must not turn a slot and a record that both
    landed into a bare 500 -- the retry that response would invite is then
    refused as a duplicate, since the ledger already moved."""
    from reliquary.validator.corpus_service import build_corpus_router

    records = _Records()
    calls: list[str] = []

    def _boom(submission_id: str) -> None:
        calls.append(submission_id)
        raise RuntimeError("webhook down")

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer, verify_signature=lambda r: True,
        prompt_job_for=seeded_job.prompt_job_for, records=records,
        on_accepted=_boom,
    ))
    client = TestClient(app)
    request = _request([7] * 16 + [EOS])
    response = client.post("/corpus/submit", json=request.model_dump())
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    sid = corpus_submission_id(request)
    assert calls == [sid]
    assert sid in records.written


def test_a_corrupt_ledger_snapshot_is_named_on_the_cursor_route(seeded_job):
    """The same translation `submit_corpus` applies to a corrupt snapshot:
    a bare lookup error would name every miner on the job as the cause."""
    seeded_job.seed_ledgers({"slots": {"99999999": 1}, "cursors": {}, "seen": []})
    response = _client(seeded_job, _Records(), []).get("/corpus/cursor/5Hot")
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_ledger_corrupt"


def test_a_corrupt_manifest_is_named_on_the_job_route(seeded_job, _r2_client):
    """The same translation `submit_corpus` applies to a manifest that no
    longer parses, so `GET /corpus/job` fails the same way a submission does."""
    _r2_client.objects["reliquary/corpus/jobs/swe-v1.json"] = (
        b'{"schema": "nope"}',
        '"tampered"',
    )
    response = _client(seeded_job, _Records(), []).get("/corpus/job")
    assert response.status_code == 500
    assert response.json()["detail"] == "corpus_job_manifest_corrupt"
