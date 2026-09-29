"""Moving a live v1 ledger to v2 and back: slots, cursors and the seen set
must come out exactly as they went in, and a writer racing the migration
must never be overwritten."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from reliquary.cli.main import app as cli
from reliquary.corpus.job import parse_job
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_job_store import BucketJobStore
from reliquary.validator.corpus_service import (
    LEDGER_SCHEMA_V1,
    LEDGER_SCHEMA_V2,
    LedgerSnapshotError,
    downgrade_ledgers_v1,
    ensure_ledgers_v2,
    verify_ledgers,
)
from tests.unit.test_corpus_ledger_v2 import seen_union
from tests.unit.test_corpus_route_seen import _app, _body, _digest, _post
from tests.unit.test_corpus_service import (  # noqa: F401 - fixtures
    _manifest,
    _r2_client,
    fake_r2,
    seeded_job,
)

LEDGER_KEY = "reliquary/corpus/jobs/swe-v1/ledgers.json"
BACKUP_KEY = "reliquary/corpus/jobs/swe-v1/ledgers.v1-backup.json"
SEEN_PREFIX = "reliquary/corpus/jobs/swe-v1/seen/"


def _digests(count, salt=""):
    return sorted(hashlib.sha256(f"{salt}{i}".encode()).hexdigest() for i in range(count))


def _v1(count=7000):
    # One slot per seen digest (n=1), eight to a prompt, as a real v1 ledger holds.
    slots = {str(i): min(8, count - 8 * i) for i in range((count + 7) // 8)}
    return {
        "schema": LEDGER_SCHEMA_V1,
        "slots": slots,
        "cursors": {"5HotA": 3, "5HotB": 7},
        "seen": _digests(count),
    }


def _seed(r2, snapshot, key=LEDGER_KEY):
    r2.objects[key] = (job_store._encode(snapshot), '"seeded"')


def _ledger(r2):
    return json.loads(r2.objects[LEDGER_KEY][0])


def _segments(r2):
    return sorted(k for k in r2.objects if k.startswith(SEEN_PREFIX))


@pytest.fixture
def job(seeded_job):
    return seeded_job.job


def test_startup_migrates_v1_preserving_slots_cursors_union(job, _r2_client):
    v1 = _v1()
    _seed(_r2_client, v1)

    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"

    ledger = _ledger(_r2_client)
    assert ledger["schema"] == LEDGER_SCHEMA_V2
    assert ledger["slots"] == v1["slots"] and ledger["cursors"] == v1["cursors"]
    assert ledger["seen_pending"] == []
    assert [ref["count"] for ref in ledger["seen_segments"]] == [4096, 2904]
    assert seen_union(_r2_client.objects, "swe-v1") == set(v1["seen"])


def test_migration_is_a_no_op_on_v2_and_on_an_absent_ledger(job, _r2_client):
    store = BucketJobStore()
    assert asyncio.run(ensure_ledgers_v2(store, job)) == "absent"
    assert LEDGER_KEY not in _r2_client.objects

    _seed(_r2_client, _v1(50))
    assert asyncio.run(ensure_ledgers_v2(store, job)) == "migrated"
    objects = dict(_r2_client.objects)
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "v2"
    assert _r2_client.objects == objects


class _RacedStore(BucketJobStore):
    """Lands a v1 writer's change the first time a segment is written, the
    way an older validator still serving the job would."""

    def __init__(self, r2, change):
        super().__init__()
        self._r2 = r2
        self._change = change
        self.segment_writes = 0

    async def write_seen_segment(self, job_id, digests):
        self.segment_writes += 1
        if self._change is not None:
            ledger = json.loads(self._r2.objects[LEDGER_KEY][0])
            self._change(ledger)
            self._change = None
            _seed(self._r2, ledger)
            self._r2.objects[LEDGER_KEY] = (self._r2.objects[LEDGER_KEY][0], '"raced"')
        return await super().write_seen_segment(job_id, digests)


def test_migration_idempotent_and_cas_guarded(job, _r2_client):
    v1 = _v1(5000)
    _seed(_r2_client, v1)

    def older_writer_admits(ledger):
        ledger["slots"]["999"] = 1

    store = _RacedStore(_r2_client, older_writer_admits)
    assert asyncio.run(ensure_ledgers_v2(store, job)) == "migrated"

    ledger = _ledger(_r2_client)
    # The racing write survived: the migration re-read it rather than
    # overwriting it with what it had read first.
    assert ledger["slots"]["999"] == 1
    assert seen_union(_r2_client.objects, "swe-v1") == set(v1["seen"])
    # Sealed twice, but the same content has the same name.
    assert store.segment_writes == 4
    assert len(_segments(_r2_client)) == 2


def test_migration_after_a_racing_seen_change_keeps_the_new_digest(job, _r2_client):
    v1 = _v1(100)
    _seed(_r2_client, v1)
    extra = hashlib.sha256(b"raced").hexdigest()

    store = _RacedStore(_r2_client, lambda ledger: ledger["seen"].append(extra))
    assert asyncio.run(ensure_ledgers_v2(store, job)) == "migrated"
    assert seen_union(_r2_client.objects, "swe-v1") == set(v1["seen"]) | {extra}


def test_migration_writes_backup_once(job, _r2_client):
    v1 = _v1(30)
    _seed(_r2_client, v1)
    store = BucketJobStore()

    assert asyncio.run(ensure_ledgers_v2(store, job)) == "migrated"
    assert json.loads(_r2_client.objects[BACKUP_KEY][0]) == v1
    backup = _r2_client.objects[BACKUP_KEY]

    # Downgrade, change, migrate again: the first backup is the one kept.
    assert asyncio.run(downgrade_ledgers_v1(BucketJobStore(), job)) == "downgraded"
    ledger = _ledger(_r2_client)
    ledger["cursors"]["5HotC"] = 1
    _seed(_r2_client, ledger)
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"
    assert _r2_client.objects[BACKUP_KEY] == backup


def test_route_migrates_v1_inline_if_startup_skipped(seeded_job, _r2_client):
    raw = _manifest()
    raw["job_id"] = "seen-v1"
    asyncio.run(job_store.write_job(raw, None))
    seen = _digests(3000)
    _r2_client.objects["reliquary/corpus/jobs/seen-v1/ledgers.json"] = (
        job_store._encode({"slots": {"5": 8}, "cursors": {}, "seen": seen}), '"v1"'
    )
    client = TestClient(_app(seeded_job, BucketJobStore(), threshold=1024))

    assert _post(client, _body(1))[1]["accepted"] is True

    ledger = json.loads(_r2_client.objects["reliquary/corpus/jobs/seen-v1/ledgers.json"][0])
    assert ledger["schema"] == LEDGER_SCHEMA_V2
    assert ledger["seen_pending"] == []
    assert sum(ref["count"] for ref in ledger["seen_segments"]) == 3001
    assert seen_union(_r2_client.objects, "seen-v1") == set(seen) | {_digest(1)}
    assert ledger["slots"] == {"5": 8, "0": 1}
    assert _post(client, _body(1, hotkey="5Copy"))[1]["reason"] == "hash_duplicate"


def test_downgrade_restores_exact_v1(seeded_job, job, _r2_client):
    v1 = _v1(2500)
    _seed(_r2_client, v1)
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"
    assert asyncio.run(downgrade_ledgers_v1(BucketJobStore(), job)) == "downgraded"
    assert _ledger(_r2_client) == v1

    # And with v2-era accepts on top: pending and sealed both come back.
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"
    client = TestClient(_app(seeded_job, BucketJobStore(), job_id="swe-v1", threshold=2))
    for fill in (1, 2, 3):
        assert _post(client, _body(fill, prompt_index=500 + fill, job_id="swe-v1"))[1]["accepted"]
    union = seen_union(_r2_client.objects, "swe-v1")
    assert asyncio.run(downgrade_ledgers_v1(BucketJobStore(), job)) == "downgraded"
    ledger = _ledger(_r2_client)
    assert set(ledger) == {"schema", "slots", "cursors", "seen"}
    assert ledger["seen"] == sorted(union)
    assert asyncio.run(downgrade_ledgers_v1(BucketJobStore(), job)) == "v1"


def test_downgrade_refuses_a_ledger_with_a_missing_segment(job, _r2_client):
    _seed(_r2_client, _v1(40))
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"
    del _r2_client.objects[_segments(_r2_client)[0]]
    before = _ledger(_r2_client)

    with pytest.raises(LedgerSnapshotError):
        asyncio.run(downgrade_ledgers_v1(BucketJobStore(), job))
    assert _ledger(_r2_client) == before


def test_verify_reports_sizes_and_the_seen_count_the_slots_imply(job, _r2_client):
    v1 = _v1(2000)
    _seed(_r2_client, v1)
    before = asyncio.run(verify_ledgers(BucketJobStore(), job))
    assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), job)) == "migrated"
    after = asyncio.run(verify_ledgers(BucketJobStore(), job))

    assert (before["schema"], after["schema"]) == (LEDGER_SCHEMA_V1, LEDGER_SCHEMA_V2)
    for report in (before, after):
        assert report["seen"] == report["expected_seen"] == 2000
        assert report["filled"] == 2000 and report["hotkeys"] == 2
        assert report["problems"] == []
    assert after["segments"] == 1 and after["pending"] == 0
    assert after["backup"] is True
    assert after["ledger_bytes"] < before["ledger_bytes"] / 10

    # I1 by count: one slot per accepted submission, n digests each.
    ledger = _ledger(_r2_client)
    ledger["slots"]["999"] = 1
    _seed(_r2_client, ledger)
    assert asyncio.run(verify_ledgers(BucketJobStore(), job))["problems"]


def _cli(*args):
    return CliRunner().invoke(cli, ["corpus", "ledgers", *args])


def test_ledgers_cli_migrate_verify_downgrade(job, _r2_client):
    v1 = _v1(300)
    _seed(_r2_client, v1)

    result = _cli("migrate", "--job", "swe-v1")
    assert result.exit_code == 0, result.output
    assert "migrated" in result.output
    result = _cli("verify", "--job", "swe-v1")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["seen"] == 300
    result = _cli("downgrade", "--job", "swe-v1")
    assert result.exit_code == 0, result.output
    assert _ledger(_r2_client) == v1

    assert _cli("verify", "--job", "nope-v1").exit_code != 0


def test_ledgers_cli_verify_fails_on_a_missing_segment(job, _r2_client):
    _seed(_r2_client, _v1(40))
    assert _cli("migrate", "--job", "swe-v1").exit_code == 0
    del _r2_client.objects[_segments(_r2_client)[0]]

    result = _cli("verify", "--job", "swe-v1")
    assert result.exit_code != 0
    assert "absent" in result.output


def _startup_then_first_submit(seeded_job, store):
    """The startup path as both entry points run it, then one submission on
    the router it warmed; returns the store's calls from the submission only."""
    from reliquary.validator.corpus_service import migrate_ledgers_at_startup
    import httpx

    job = parse_job({**_manifest(), "job_id": "seen-v1"})

    async def scenario():
        index = await migrate_ledgers_at_startup(store, job)
        store.log.clear()
        app = _app(seeded_job, store, seen_index=index)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://v"
        ) as http:
            return (await http.post("/corpus/submit", json=_body(1))).json()

    return asyncio.run(scenario()), store.log


@pytest.mark.parametrize("already_v2", [False, True])
def test_the_first_submit_after_startup_loads_no_segments(seeded_job, _r2_client, already_v2):
    from tests.unit.test_corpus_route_seen import _Store

    raw = {**_manifest(), "job_id": "seen-v1"}
    asyncio.run(job_store.write_job(raw, None))
    _r2_client.objects["reliquary/corpus/jobs/seen-v1/ledgers.json"] = (
        job_store._encode({"slots": {"5": 8}, "cursors": {}, "seen": _digests(9000)}), '"v1"'
    )
    if already_v2:
        # A restart: the ledger was migrated by an earlier process.
        assert asyncio.run(ensure_ledgers_v2(BucketJobStore(), parse_job(raw))) == "migrated"

    verdict, calls = _startup_then_first_submit(seeded_job, _Store())

    assert verdict["accepted"] is True
    assert "segment_get" not in calls
