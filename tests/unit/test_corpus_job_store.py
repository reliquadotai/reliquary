"""The job manifest and the ledgers are shared state: two validators that
disagree about which slots are filled pay twice for one slot."""

from __future__ import annotations

import pytest

from reliquary.corpus.job import JobError
from reliquary.infrastructure import corpus_job_store as store


def _manifest(job_id="swe-v1"):
    return {
        "schema": "reliquary/corpus-job/v1",
        "job_id": job_id,
        "checkpoint_repo": "org/Frozen",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": "a" * 64,
        "prompt_source": "openmathinstruct",
        "prompt_count": 1000,
        "renderer_id": "reliquary-external-prompt-v1",
        "eos_token_id": 151645,
        "sampling": {
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "min_new_tokens": 16, "max_new_tokens": 4096, "n": 4,
        },
        "slots_per_prompt": 8,
        "filter": None,
        "prompt_order": "free",
        "deadline_round": 5_000_000,
    }


class _FakeMultiObjectR2:
    """An object store keyed by object key, each holding (body, etag), with
    real per-key conditional-put semantics — unlike ``task_registry_store``'s
    fixture, this store addresses more than one key (a job plus its ledgers,
    across job ids), so the fake must key on the object key, not be a
    singleton."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self._version = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _client_error("NoSuchKey")
        body, etag = self.objects[Key]

        class _Body:
            def __init__(self, data):
                self._data = data

            async def read(self):
                return self._data

        return {"Body": _Body(body), "ETag": etag}

    async def put_object(self, Bucket, Key, Body, **condition):
        current = self.objects.get(Key)
        current_etag = current[1] if current else None
        if "IfMatch" in condition and condition["IfMatch"] != current_etag:
            raise _client_error("PreconditionFailed")
        if "IfNoneMatch" in condition and current is not None:
            raise _client_error("PreconditionFailed")
        self._version += 1
        etag = f'"v{self._version}"'
        self.objects[Key] = (Body, etag)
        return {"ETag": etag}


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "PutObject")


@pytest.fixture
def _r2_client(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: client)
    return client


@pytest.fixture
def fake_r2(_r2_client):
    """The kwargs the store takes. The fake client is installed by monkeypatch,
    so the store's own bucket/credential defaults are exercised untouched."""
    return {}


@pytest.fixture
def seed_object(_r2_client):
    """Write straight into the fake bucket, bypassing the store entirely —
    the store has no raw-write API and must not grow one for a test."""

    def _seed(key: str, body: bytes) -> None:
        _r2_client.objects[key] = (body, '"seed"')

    return _seed


@pytest.mark.asyncio
async def test_an_absent_job_is_not_an_error(fake_r2):
    job, etag = await store.read_job("ghost", **fake_r2)
    assert job is None
    assert etag is None


@pytest.mark.asyncio
async def test_a_written_job_reads_back_as_a_parsed_spec(fake_r2):
    etag = await store.write_job(_manifest(), None, **fake_r2)
    assert etag
    job, read_etag = await store.read_job("swe-v1", **fake_r2)
    assert job is not None
    assert job.job_id == "swe-v1"
    assert job.sampling.n == 4
    assert read_etag == etag


@pytest.mark.asyncio
async def test_a_stored_manifest_that_is_not_a_job_is_refused_on_read(fake_r2, seed_object):
    # A hand-edited object in the bucket must fail here, naming the field,
    # rather than at the first submission it would misjudge. Seeded through
    # the fixture, not through the store: the store has no raw-write API and
    # must not grow one for a test.
    seed_object("reliquary/corpus/jobs/broken.json", b'{"schema": "nope"}')
    with pytest.raises(JobError):
        await store.read_job("broken", **fake_r2)


@pytest.mark.asyncio
async def test_a_stale_etag_conflicts_rather_than_overwriting(fake_r2):
    first = await store.write_job(_manifest(), None, **fake_r2)
    await store.write_job({**_manifest(), "prompt_count": 2000}, first, **fake_r2)
    with pytest.raises(store.CorpusStoreConflict):
        await store.write_job({**_manifest(), "prompt_count": 3000}, first, **fake_r2)


@pytest.mark.asyncio
async def test_ledgers_round_trip_through_their_snapshots(fake_r2):
    snapshot = {"slots": {"3": 2, "7": 8}, "cursors": {"5Hot": 41}}
    etag = await store.write_ledgers("swe-v1", snapshot, None, **fake_r2)
    read, read_etag = await store.read_ledgers("swe-v1", **fake_r2)
    assert read == snapshot
    assert read_etag == etag


@pytest.mark.asyncio
async def test_a_job_id_that_is_not_a_job_id_is_refused_before_any_write(fake_r2):
    # The id becomes a bucket key; a traversal or a wildcard must never reach it.
    with pytest.raises(ValueError):
        await store.write_job({**_manifest(job_id="../../etc/passwd")}, None, **fake_r2)
