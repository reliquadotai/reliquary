"""Submissions and verdicts are write-once; the settlement state is CAS."""

import asyncio

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
