"""Submissions and verdicts are write-once; the settlement state is CAS."""

import asyncio
import copy

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

ID = "c" * 64


@pytest.fixture
def r2(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    monkeypatch.setattr(records, "get_s3_client", lambda **kw: client)
    return client


def run(coro):
    return asyncio.run(coro)


def test_a_submission_is_written_once(r2):
    assert run(records.write_submission("math-v1", ID, {"hotkey": "5A"})) is True
    assert run(records.write_submission("math-v1", ID, {"hotkey": "5B"})) is False
    assert run(records.read_submission("math-v1", ID)) == {"hotkey": "5A"}


def test_a_verdict_is_written_once(r2):
    assert run(records.write_verdict("math-v1", ID, {"passed": True})) is True
    assert run(records.write_verdict("math-v1", ID, {"passed": False})) is False
    assert run(records.read_verdict("math-v1", ID)) == {"passed": True}


def test_listings_return_ids_only_for_their_job(r2):
    other = "d" * 64
    run(records.write_submission("math-v1", ID, {}))
    run(records.write_submission("math-v2", other, {}))
    run(records.write_verdict("math-v1", ID, {}))
    assert run(records.list_submission_ids("math-v1")) == [ID]
    assert run(records.list_verdict_ids("math-v1")) == [ID]
    assert run(records.list_verdict_ids("math-v2")) == []


def test_the_ledger_and_job_keys_are_not_listed_as_submissions(r2):
    # The ledgers object shares the job prefix and must not read as a submission.
    run(job_store.write_ledgers("math-v1", {"schema": "x"}, None))
    assert run(records.list_submission_ids("math-v1")) == []


def test_settlement_is_compare_and_swap(r2):
    state, etag = run(records.read_settlement("math-v1"))
    assert (state, etag) == ({}, None)
    etag = run(records.write_settlement("math-v1", {"last_window": 1}, None))
    with pytest.raises(job_store.CorpusStoreConflict):
        run(records.write_settlement("math-v1", {"last_window": 2}, None))
    run(records.write_settlement("math-v1", {"last_window": 2}, etag))
    assert run(records.read_settlement("math-v1"))[0] == {"last_window": 2}


@pytest.mark.parametrize("bad", ["../x", "C" * 64, "c" * 63, ""])
def test_an_id_that_is_not_a_digest_never_reaches_a_key(r2, bad):
    with pytest.raises(ValueError):
        run(records.write_submission("math-v1", bad, {}))


def test_grades_and_regrades_are_written_once_under_their_own_keys(r2):
    assert run(records.write_grade("math-v1", ID, {"status": "ok"})) is True
    assert run(records.write_grade("math-v1", ID, {"status": "error"})) is False
    assert run(records.read_grade("math-v1", ID)) == {"status": "ok"}
    assert run(records.list_grade_ids("math-v1")) == [ID]
    assert run(records.read_regrade("math-v1", ID)) is None
    assert run(records.write_regrade("math-v1", ID, {"status": "ok", "graded_by": ["g2"]})) is True
    assert run(records.write_regrade("math-v1", ID, {"status": "error"})) is False
    assert run(records.read_regrade("math-v1", ID)) == {"status": "ok", "graded_by": ["g2"]}
    assert run(records.list_grade_ids("math-v1")) == [ID]
    assert run(records.list_submission_ids("math-v1")) == []
    store = records.BucketRecordStore()
    assert run(store.read_grade("math-v1", ID)) == {"status": "ok"}
    assert run(store.list_grade_ids("math-v1")) == [ID]


def test_a_later_regrade_generation_supersedes_and_stays_bounded(r2):
    assert run(records.write_regrade("math-v1", ID, {"generation": 1})) is True
    assert run(records.write_regrade("math-v1", ID, {"generation": 2}, 2)) is True
    assert run(records.write_regrade("math-v1", ID, {"generation": 9}, 2)) is False
    assert run(records.read_regrade("math-v1", ID)) == {"generation": 2}
    with pytest.raises(ValueError):
        run(records.write_regrade("math-v1", ID, {}, records.MAX_REGRADE_GENERATIONS + 1))
    assert run(records.list_grade_ids("math-v1")) == []


@pytest.fixture
def staged_r2(r2, monkeypatch):
    original = r2.get_object

    async def get_object(Bucket, Key, Range=None):
        response = await original(Bucket=Bucket, Key=Key)
        if Range is not None:
            start, end = Range.removeprefix("bytes=").split("-")
            response["Body"]._data = response["Body"]._data[int(start):int(end) + 1]
        return response

    monkeypatch.setattr(r2, "get_object", get_object)
    return r2


def _record(*, episode=False):
    completion = {"tokens": [1, 2], "text": "answer", "proofs": []}
    if episode:
        completion = {"tokens": [1, 2, 3], "prompt_tokens": [9],
                      "turns": [{"start": 0, "end": 2, "proofs": []}],
                      "final_diff": "", "stop": "agent_completed"}
    return {"schema": records.RECORD_SCHEMA_V2 if episode else records.RECORD_SCHEMA_V1,
            "job_id": "math-v1", "submission_id": ID, "hotkey": "5A", "cursor": 0,
            "prompt_index": 0, "rendered_prompt": "question", "received_at": 123.5,
            "token_count": 2, "completions": [completion]}


@pytest.mark.parametrize("episode", [False, True])
def test_staging_is_not_auditable_and_recovery_keeps_the_exact_receipt(staged_r2, episode):
    record = _record(episode=episode)
    ref = run(records.stage_submission("math-v1", ID, record))
    assert ref["received_at"] == record["received_at"]
    assert run(records.list_submission_ids("math-v1")) == []
    assert run(records.read_submission("math-v1", ID)) is None
    assert run(records.list_verdict_ids("math-v1")) == []
    assert run(records.read_staged_submission("math-v1", ref)) == record
    # A new store after the ledger commit has only its durable reference.
    store = records.BucketRecordStore()
    assert run(store.promote_submission("math-v1", ref)) == record
    before = dict(staged_r2.objects)
    assert run(store.promote_submission("math-v1", ref)) == record
    assert staged_r2.objects == before
    assert run(records.list_submission_ids("math-v1")) == [ID]
    assert run(records.read_submission("math-v1", ID)) == record


def test_lost_create_acknowledgments_are_resolved_by_exact_readback(staged_r2, monkeypatch):
    put = staged_r2.put_object

    async def lost_ack(**kwargs):
        await put(**kwargs)
        raise OSError("acknowledgment lost")

    monkeypatch.setattr(staged_r2, "put_object", lost_ack)
    ref = run(records.stage_submission("math-v1", ID, _record()))
    assert run(records.promote_submission("math-v1", ref)) == _record()


def test_publication_failure_keeps_the_staged_body_for_recovery(staged_r2, monkeypatch):
    ref = run(records.stage_submission("math-v1", ID, _record()))
    put = staged_r2.put_object

    async def unavailable(**kwargs):
        if "/submissions/" in kwargs["Key"]:
            raise OSError("unavailable")
        return await put(**kwargs)

    monkeypatch.setattr(staged_r2, "put_object", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        run(records.promote_submission("math-v1", ref))
    assert run(records.list_submission_ids("math-v1")) == []
    assert run(records.read_staged_submission("math-v1", ref)) == _record()
    monkeypatch.setattr(staged_r2, "put_object", put)
    assert run(records.promote_submission("math-v1", ref)) == _record()


def test_a_create_conflict_without_an_object_is_not_success(staged_r2, monkeypatch):
    async def contended(*args, **kwargs):
        raise job_store.CorpusStoreConflict("in flight and absent")

    monkeypatch.setattr(records, "_put", contended)
    with pytest.raises(job_store.CorpusStoreConflict):
        run(records.stage_submission("math-v1", ID, _record()))
    assert not staged_r2.objects


@pytest.mark.parametrize("change", ["received_at", "proofs"])
def test_different_existing_canonical_bytes_are_never_overwritten(staged_r2, change):
    record = _record()
    ref = run(records.stage_submission("math-v1", ID, record))
    different = copy.deepcopy(record)
    if change == "received_at":
        different["received_at"] += 1
    else:
        different["completions"][0]["proofs"] = ["YQ=="]
    run(records.write_submission("math-v1", ID, different))
    before = dict(staged_r2.objects)
    with pytest.raises(ValueError, match="committed body"):
        run(records.promote_submission("math-v1", ref))
    assert staged_r2.objects == before
    assert run(records.read_staged_submission("math-v1", ref)) == record


@pytest.mark.parametrize("fault", ["absent", "hash", "timestamp", "job", "sid"])
def test_a_missing_or_mismatched_staged_body_is_never_published(staged_r2, fault):
    ref = run(records.stage_submission("math-v1", ID, _record()))
    key = records._key("math-v1", "staged-submissions", ref["sha256"])
    if fault == "absent":
        staged_r2.objects.pop(key)
    elif fault == "hash":
        staged_r2.objects[key] = (b"{}", '"changed"')
    elif fault == "timestamp":
        ref["received_at"] += 1
    elif fault == "job":
        staged_r2.objects[records._key("math-v2", "staged-submissions", ref["sha256"])] = staged_r2.objects[key]
    else:
        ref["submission_id"] = "d" * 64
    job = "math-v2" if fault == "job" else "math-v1"
    with pytest.raises(ValueError):
        run(records.promote_submission(job, ref))
    assert run(records.list_submission_ids(job)) == []


def test_oversized_staged_objects_are_refused_before_decoding(staged_r2, monkeypatch):
    ref = run(records.stage_submission("math-v1", ID, _record()))
    monkeypatch.setattr(records, "MAX_STAGED_RECORD_BYTES", 8)
    with pytest.raises(ValueError, match="byte bound"):
        run(records.read_staged_submission("math-v1", ref))
    with pytest.raises(ValueError, match="byte bound"):
        run(records.stage_submission("math-v1", ID, _record()))


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("job_id", "math-v2"), ("submission_id", "d" * 64),
    ("received_at", float("nan")), ("received_at", True), ("token_count", 9),
    ("completions", [{}]),
    ("completions", [{"tokens": [1.0, 2], "text": "answer", "proofs": []}]),
])
def test_invalid_receipts_never_reach_staging(staged_r2, field, value):
    record = _record()
    record[field] = value
    with pytest.raises(ValueError):
        run(records.stage_submission("math-v1", ID, record))
    assert not staged_r2.objects


@pytest.mark.parametrize("field", ["completions", "rendered_prompt"])
def test_recovery_enforces_the_native_aggregate_wire_limits(staged_r2, field):
    import hashlib

    from reliquary.protocol.corpus_submission import (
        MAX_COMPLETIONS_PER_SUBMISSION,
        MAX_RENDERED_PROMPT_CHARS,
    )

    record = _record()
    if field == "completions":
        record[field] *= MAX_COMPLETIONS_PER_SUBMISSION + 1
        record["token_count"] = 2 * len(record[field])
    else:
        record[field] = "x" * (MAX_RENDERED_PROMPT_CHARS + 1)
    body = job_store._encode(record)
    sha = hashlib.sha256(body).hexdigest()
    staged_r2.objects[records._key("math-v1", "staged-submissions", sha)] = (body, '"seed"')
    ref = {"submission_id": ID, "sha256": sha, "received_at": record["received_at"]}
    with pytest.raises(ValueError):
        run(records.promote_submission("math-v1", ref))
    assert run(records.list_submission_ids("math-v1")) == []
