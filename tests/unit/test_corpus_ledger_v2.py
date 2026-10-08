"""Ledger schema v2: the seen set lives in sealed segments plus a short pending
list, and each schema is read under its own field set so that no binary ever
drops a field it does not know on its next write."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from reliquary.corpus.job import parse_job
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_job_store import BucketJobStore
from reliquary.validator.corpus_service import (
    LEDGER_SCHEMA,
    LEDGER_SCHEMA_V1,
    LEDGER_SCHEMA_V2,
    LEDGER_SCHEMA_V3,
    LedgerSnapshotError,
    SeenIndex,
    SeenView,
    SegmentRef,
    ledger_snapshot,
    rebuild_ledgers,
)
from tests.unit.test_corpus_service import _manifest
from tests.unit.test_corpus_store_cache import _CountingR2


def seen_union(objects, job_id):
    """Every digest a stored ledger counts as seen, read straight from a fake
    bucket's objects: v1's list, or v2's pending plus its segments."""
    import json

    prefix = f"reliquary/corpus/jobs/{job_id}"
    ledgers = json.loads(objects[f"{prefix}/ledgers.json"][0])
    if "seen" in ledgers:
        return set(ledgers["seen"])
    union = set(ledgers["seen_pending"])
    for ref in ledgers["seen_segments"]:
        digests = json.loads(objects[f"{prefix}/seen/{ref['id']}.json"][0])["digests"]
        assert len(digests) == ref["count"] and union.isdisjoint(digests)
        union.update(digests)
    return union


def _d(i):
    return hashlib.sha256(str(i).encode()).hexdigest()


@pytest.fixture
def bucket(monkeypatch):
    client = _CountingR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    return client


@pytest.fixture
def job():
    return parse_job(_manifest())


def test_this_binary_writes_v2():
    assert LEDGER_SCHEMA == LEDGER_SCHEMA_V2


def test_v2_roundtrip(job):
    refs = (SegmentRef("a" * 64, 3), SegmentRef("b" * 64, 2))
    v2 = {
        "schema": LEDGER_SCHEMA_V2,
        "slots": {"4": 2},
        "cursors": {"5Hot": 3},
        "seen_pending": [_d(2), _d(1)],
        "seen_segments": [{"id": r.id, "count": r.count} for r in refs],
    }
    state = rebuild_ledgers(job, v2)
    assert state.schema == LEDGER_SCHEMA_V2
    assert state.pending == {_d(1), _d(2)}
    assert state.segments == refs
    again = ledger_snapshot(state.slots, state.cursors, state.pending, state.segments)
    assert again == {**v2, "seen_pending": sorted([_d(1), _d(2)])}
    # Parsing copies: the snapshot the store may share with its memory is untouched.
    state.pending.add(_d(9))
    assert v2["seen_pending"] == [_d(2), _d(1)]


def test_v1_reads_as_all_pending(job):
    state = rebuild_ledgers(
        job, {"schema": LEDGER_SCHEMA_V1, "slots": {"1": 1}, "cursors": {}, "seen": ["x", "y"]}
    )
    assert state.schema == LEDGER_SCHEMA_V1
    assert state.pending == {"x", "y"}
    assert state.segments == ()
    # The empty read before a job's first write, and pre-marker objects, are v1.
    assert rebuild_ledgers(job, {}).schema == LEDGER_SCHEMA_V1
    assert rebuild_ledgers(job, {"seen": ["x"]}).pending == {"x"}


def test_fields_enforced_per_schema(job):
    with pytest.raises(LedgerSnapshotError, match="seen"):
        rebuild_ledgers(job, {
            "schema": LEDGER_SCHEMA_V2, "slots": {}, "cursors": {},
            "seen_pending": [], "seen_segments": [], "seen": [],
        })
    with pytest.raises(LedgerSnapshotError, match="seen_pending"):
        rebuild_ledgers(job, {"schema": LEDGER_SCHEMA_V1, "seen": [], "seen_pending": []})
    with pytest.raises(LedgerSnapshotError, match="seen_pending"):
        rebuild_ledgers(job, {"seen_pending": []})
    # A v2 object that lost either seen field would read as a smaller set.
    for missing in ("seen_pending", "seen_segments"):
        v2 = {"schema": LEDGER_SCHEMA_V2, "seen_pending": [], "seen_segments": []}
        del v2[missing]
        with pytest.raises(LedgerSnapshotError, match=missing):
            rebuild_ledgers(job, v2)


@pytest.mark.parametrize("bad", [
    {"seen_pending": ["x", "x"]},
    {"seen_pending": [17]},
    {"seen_pending": "x"},
    {"seen_segments": [{"id": "a" * 64}]},
    {"seen_segments": [{"id": "a" * 64, "count": 0}]},
    {"seen_segments": [{"id": "a" * 64, "count": True}]},
    {"seen_segments": [{"id": "../x", "count": 1}]},
    {"seen_segments": [{"id": "a" * 64, "count": 1, "extra": 1}]},
    {"seen_segments": [{"id": "a" * 64, "count": 1}, {"id": "a" * 64, "count": 1}]},
])
def test_malformed_v2_seen_fields_are_refused(job, bad):
    v2 = {"schema": LEDGER_SCHEMA_V2, "seen_pending": [], "seen_segments": [], **bad}
    with pytest.raises(LedgerSnapshotError):
        rebuild_ledgers(job, v2)


def test_unknown_schema_refused(job):
    with pytest.raises(LedgerSnapshotError, match="v4"):
        rebuild_ledgers(job, {"schema": "reliquary/corpus-ledgers/v4"})


@pytest.mark.parametrize("refs", [None, {}, [None],
    [{"submission_id": "a" * 64, "sha256": "b" * 64, "received_at": float("nan")}],
    [{"submission_id": "a" * 64, "sha256": "b" * 64, "received_at": True}],
    [{"submission_id": "a" * 64, "sha256": "B" * 64, "received_at": 1.0}],
    [{"submission_id": "a" * 64, "sha256": "b" * 64, "received_at": 1.0}] * 2,
])
def test_v3_ref_corruption_never_reads_as_an_empty_pending_set(job, refs):
    with pytest.raises(LedgerSnapshotError):
        rebuild_ledgers(job, {"schema": LEDGER_SCHEMA_V3, "seen_pending": [],
                             "seen_segments": [], "pending_records": refs})


def test_v3_requires_upgraded_v2_readers_even_after_cleanup(job):
    v2 = ledger_snapshot(rebuild_ledgers(job, {}).slots, rebuild_ledgers(job, {}).cursors, ())
    v3 = {**v2, "schema": LEDGER_SCHEMA_V3, "pending_records": []}
    assert rebuild_ledgers(job, v2).records == rebuild_ledgers(job, v3).records == ()
    # The previous binary rejects the new marker and its extra field; there
    # is no writer activation until every ledger reader has been upgraded.
    frozen_v2_fields = {"schema", "slots", "cursors", "seen_pending", "seen_segments", "failed"}
    def frozen_v2_reader(snapshot):
        if snapshot.get("schema", LEDGER_SCHEMA_V1) not in (LEDGER_SCHEMA_V1, LEDGER_SCHEMA_V2):
            raise LedgerSnapshotError("unknown ledger schema")
        if set(snapshot) - frozen_v2_fields:
            raise LedgerSnapshotError("unknown ledger fields")

    frozen_v2_reader(v2)
    with pytest.raises(LedgerSnapshotError, match="schema"):
        frozen_v2_reader(v3)
    with pytest.raises(LedgerSnapshotError, match="fields"):
        frozen_v2_reader({**v3, "schema": LEDGER_SCHEMA_V2})


# The field check `rebuild_ledgers` ran before v2, frozen here: every binary
# already deployed runs exactly this against whatever this one writes.
_FROZEN_V1_FIELDS = frozenset({"schema", "slots", "cursors", "seen"})


def _frozen_v1_reader(snapshot):
    unknown = sorted(set(snapshot) - _FROZEN_V1_FIELDS)
    if unknown:
        raise LedgerSnapshotError(f"ledger fields this binary cannot read: {unknown}")
    return set(snapshot.get("seen") or [])


def test_v2_object_trips_frozen_v1_reader(job):
    state = rebuild_ledgers(job, {"seen": [_d(1)]})
    written = ledger_snapshot(state.slots, state.cursors, state.pending, ())
    assert "seen" not in written
    with pytest.raises(LedgerSnapshotError, match="seen_pending"):
        _frozen_v1_reader(written)


def _seal(store, digests):
    digests = sorted(digests)
    segment_id = asyncio.run(store.write_seen_segment("swe-v1", digests))
    return SegmentRef(segment_id, len(digests))


def test_index_holds_exactly_the_referenced_segments(bucket):
    store = BucketJobStore()
    first = _seal(store, [_d(1), _d(2)])
    orphan = _seal(store, [_d(3)])
    second = _seal(store, [_d(4)])
    index = SeenIndex(store, "swe-v1")

    asyncio.run(index.ensure((first,)))
    assert _d(1) in index and _d(4) not in index and len(index) == 2
    asyncio.run(index.ensure((first, second)))
    assert _d(4) in index and _d(3) not in index and len(index) == 3
    # A reference list that diverges (a downgrade, then a re-migration) is
    # rebuilt for exactly that list.
    asyncio.run(index.ensure((orphan,)))
    assert _d(3) in index and _d(1) not in index and len(index) == 1
    view = SeenView(index, {_d(9)})
    assert _d(9) in view and _d(3) in view and _d(1) not in view
    assert len(view) == 2 and set(view) == {_d(3), _d(9)}


def test_overlap_between_segments_or_pending_is_corrupt(bucket):
    store = BucketJobStore()
    first = _seal(store, [_d(1), _d(2)])
    overlapping = _seal(store, [_d(2), _d(3)])
    index = SeenIndex(store, "swe-v1")

    with pytest.raises(LedgerSnapshotError):
        asyncio.run(index.ensure((first, overlapping)))
    # A failed ensure leaves the index as it was, not half-built.
    assert len(index) == 0
    asyncio.run(index.ensure((first,)))
    with pytest.raises(LedgerSnapshotError):
        asyncio.run(index.ensure((first, overlapping)))
    assert len(index) == 2
    with pytest.raises(LedgerSnapshotError):
        index.check_pending({_d(2)})
    index.check_pending({_d(3)})


def test_segment_count_mismatch_is_corrupt(bucket):
    store = BucketJobStore()
    ref = _seal(store, [_d(1), _d(2)])
    index = SeenIndex(store, "swe-v1")
    with pytest.raises(LedgerSnapshotError):
        asyncio.run(index.ensure((SegmentRef(ref.id, 3),)))


def test_a_missing_segment_is_corrupt_and_a_remembered_one_costs_no_get(bucket):
    store = BucketJobStore()
    index = SeenIndex(store, "swe-v1")
    with pytest.raises(LedgerSnapshotError):
        asyncio.run(index.ensure((SegmentRef("0" * 64, 1),)))

    ref = _seal(store, [_d(1)])
    index.remember(ref.id, (_d(1),))
    gets = len(bucket.gets)
    asyncio.run(index.ensure((ref,)))
    assert _d(1) in index
    assert len(bucket.gets) == gets
