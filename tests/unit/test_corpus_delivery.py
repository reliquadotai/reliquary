"""R4 export v2: passing rows as Parquet shards, streamed and read in parallel."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from reliquary.corpus.delivery import (
    LocalDirectorySink,
    R2DeliverySink,
    delivery_rows,
    export_delivery,
)


def _sid(i: int) -> str:
    return f"{i:064x}"


class _Records:
    """Submissions 0..n-1, every third failing, one passing record missing."""

    def __init__(self, n=30, text_len=50, missing=(4,)):
        self.subs, self.verdicts = {}, {}
        for i in range(n):
            sid = _sid(i)
            self.verdicts[sid] = {"passed": i % 3 != 2, "hotkey": f"HK{i}", "token_count": 2}
            if i not in missing:
                self.subs[sid] = {
                    "hotkey": f"HK{i}", "prompt_index": i, "rendered_prompt": f"q{i}",
                    "completions": [{"text": f"{i}-{c}-" + "z" * text_len, "tokens": [1, 2, c]}
                                    for c in range(2)],
                }
        self.in_flight = self.peak = 0
        self.reads = 0

    async def _track(self, value):
        self.in_flight += 1
        self.reads += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return value

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return await self._track(self.verdicts.get(sid))

    async def read_submission(self, job_id, sid):
        return await self._track(self.subs.get(sid))


JOB = SimpleNamespace(job_id="math-v1", filter=None,
                      to_contract=lambda: {"job_id": "math-v1", "prompt_count": 30})


def _rows(records, **kw):
    async def go():
        counts = {}
        rows = [row async for row in delivery_rows(job=JOB, records=records, counts=counts, **kw)]
        return rows, counts

    return asyncio.run(go())


def test_rows_are_the_completions_of_passing_submissions_with_no_hotkey():
    rows, counts = _rows(_Records(n=6, missing=()))
    assert [(r["submission_id"], r["completion_index"]) for r in rows] == [
        (_sid(i), c) for i in (0, 1, 3, 4) for c in range(2)]
    assert all("hotkey" not in r and "HK" not in json.dumps(r) for r in rows)
    assert rows[0]["prompt"] == "q0" and rows[0]["completion_tokens"] == 3
    assert rows[0]["accepted"] is None and rows[0]["score"] is None
    assert counts == {"verdicts": 6, "passing_submissions": 4, "missing_records": 0, "rows": 8,
                      "rows_accepted": None}


def test_a_passing_verdict_without_its_record_is_skipped_and_counted():
    rows, counts = _rows(_Records(n=6, missing=(4,)))
    assert _sid(4) not in {r["submission_id"] for r in rows}
    assert counts["missing_records"] == 1


def test_the_grader_annotates_rows_and_never_drops_one():
    rows, counts = _rows(_Records(n=6, missing=()),
                         grade=lambda prompt_index, text: (prompt_index % 2 == 0, 0.5))
    assert len(rows) == 8
    assert {(r["prompt_index"], r["accepted"]) for r in rows} == {
        (0, True), (1, False), (3, False), (4, True)}
    assert counts["rows_accepted"] == 4


def test_reads_are_parallel_but_bounded():
    records = _Records(n=200, missing=())
    _rows(records, concurrency=8, window=64)
    assert 1 < records.peak <= 8


def _export(tmp_path, records, delivery_id="d1", **kw):
    sink = LocalDirectorySink(tmp_path / "bucket")
    result = asyncio.run(export_delivery(
        job=JOB, records=records, sink=sink, delivery_id=delivery_id,
        work_dir=tmp_path / "work", clock=lambda: 1234.0, **kw))
    return sink, result


def test_an_export_writes_shards_a_manifest_with_hashes_and_a_report(tmp_path):
    records = _Records(n=90, text_len=4000, missing=(4,))
    sink, result = _export(tmp_path, records, shard_max_bytes=60_000, row_group_rows=5)
    root = tmp_path / "bucket" / "deliveries" / "d1"
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert len(manifest["shards"]) > 1
    total = 0
    for shard in manifest["shards"]:
        path = root / shard["name"]
        data = path.read_bytes()
        assert len(data) == shard["bytes"] <= 60_000
        assert hashlib.sha256(data).hexdigest() == shard["sha256"]
        table = pq.read_table(path)
        assert table.num_rows == shard["rows"]
        assert "hotkey" not in table.column_names
        total += shard["rows"]
    assert total == manifest["rows"] == report["counts"]["rows"]
    assert report["counts"]["missing_records"] == 1
    assert report["job"] == {"job_id": "math-v1", "prompt_count": 30}
    assert report["filter"] == {"applied": False, "note": "the job declares no filter"}
    assert sorted(result["keys"]) == sorted(
        [f"deliveries/d1/{s['name']}" for s in manifest["shards"]]
        + ["deliveries/d1/manifest.json", "deliveries/d1/report.json"])
    # Streamed: no local shard outlives its upload.
    assert not list((tmp_path / "work").glob("**/*.parquet"))


def test_the_same_delivery_again_returns_the_stored_keys_without_reading(tmp_path):
    records = _Records(n=9, missing=())
    _, first = _export(tmp_path, records)
    reads = records.reads
    _, second = _export(tmp_path, records)
    assert second == first and records.reads == reads


def test_an_empty_job_delivers_a_manifest_and_no_shard(tmp_path):
    records = _Records(n=0)
    _, result = _export(tmp_path, records)
    assert result["rows"] == 0 and result["shards"] == []


@pytest.mark.parametrize("bad", ["../x", "", "a/b", "x" * 200])
def test_a_delivery_id_that_is_not_a_name_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        _export(tmp_path, _Records(n=1), delivery_id=bad)


def test_the_r2_sink_uploads_through_the_scoped_client(tmp_path):
    calls = []

    class _Client:
        def upload_file(self, filename, bucket, key, Config=None):
            calls.append(("file", bucket, key, Path(filename).read_bytes(), Config is not None))

        def put_object(self, Bucket, Key, Body, ContentType=None):
            calls.append(("json", Bucket, Key, Body, ContentType))

        def get_object(self, Bucket, Key):
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    sink = R2DeliverySink(bucket="platform", client=_Client())
    path = tmp_path / "f.parquet"
    path.write_bytes(b"PAR1")
    asyncio.run(sink.put_file("deliveries/d1/part-00000.parquet", path))
    asyncio.run(sink.put_json("deliveries/d1/manifest.json", {"a": 1}))
    assert asyncio.run(sink.get_json("deliveries/d1/manifest.json")) is None
    assert calls[0] == ("file", "platform", "deliveries/d1/part-00000.parquet", b"PAR1", True)
    assert calls[1][:3] == ("json", "platform", "deliveries/d1/manifest.json")
    assert json.loads(calls[1][3]) == {"a": 1}
