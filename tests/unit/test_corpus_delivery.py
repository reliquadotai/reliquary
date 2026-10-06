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
    HTTPDeliverySink,
    LocalDirectorySink,
    R2DeliverySink,
    delivery_rows,
    export_delivery,
    instruction_source_for_job,
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


def _export(tmp_path, records, delivery_id="d1", *, job=JOB, **kw):
    sink = LocalDirectorySink(tmp_path / "bucket")
    result = asyncio.run(export_delivery(
        job=job, records=records, sink=sink, delivery_id=delivery_id,
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
    from reliquary.protocol.release_contract import canonical_sha256

    assert manifest["job_manifest_sha256"] == canonical_sha256(report["job"])
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


def test_a_delivery_id_cannot_be_reused_for_another_job(tmp_path):
    records = _Records(n=0)
    _export(tmp_path, records)
    another = SimpleNamespace(**{**vars(JOB), "job_id": "math-v2"})
    with pytest.raises(ValueError, match="delivery belongs to another job"):
        _export(tmp_path, records, job=another)


def test_an_empty_job_delivers_a_manifest_and_no_shard(tmp_path):
    records = _Records(n=0)
    _, result = _export(tmp_path, records)
    assert result["rows"] == 0 and result["shards"] == []


INSTRUCTION_JOB = SimpleNamespace(
    **vars(JOB), renderer_id="chat-template-v1", episode=None,
    prompt_start=0, prompt_count=100_000, prompt_end=100_000, prompt_source="fixture",
    owns=lambda index: 0 <= index < 100_000)


class _RawSource:
    def task_for(self, index):
        return SimpleNamespace(prompt=f"Raw question {index}?", metadata={})


def _instructions(tmp_path, manifest):
    rows = []
    for shard in manifest["instruction_shards"]:
        data = (tmp_path / "bucket" / shard["key"]).read_bytes()
        assert len(data) == shard["bytes"] <= 5 * 1024 * 1024
        assert hashlib.sha256(data).hexdigest() == shard["sha256"]
        parsed = [json.loads(line) for line in data.decode().splitlines()]
        assert len(parsed) == shard["rows"] <= 10_000
        assert all(set(row) == {"prompt", "response"} for row in parsed)
        assert shard["key"] in manifest["keys"]
        rows.extend(parsed)
    return rows


@pytest.mark.parametrize("renderer", ["chat-template-v1", "chat-template-thinking-v1"])
def test_instruction_companion_uses_raw_prompts_not_rendered_records(tmp_path, renderer):
    records = _Records(n=3, missing=())
    job = SimpleNamespace(**{**vars(INSTRUCTION_JOB), "renderer_id": renderer})
    _, manifest = _export(tmp_path, records, job=job, instruction_source=_RawSource())
    rows = _instructions(tmp_path, manifest)
    assert [row["prompt"] for row in rows] == ["Raw question 0?"] * 2 + ["Raw question 1?"] * 2
    assert rows[0]["response"] == records.subs[_sid(0)]["completions"][0]["text"]
    parquet = pq.read_table(tmp_path / "bucket" / manifest["shards"][0]["key"])
    assert parquet.column("prompt").to_pylist() == ["q0", "q0", "q1", "q1"]
    assert manifest["instruction"]["rows"] == 4
    assert manifest["instruction"]["policy"] == "audited_passing_completions"
    assert manifest["instruction"]["filter_applied"] is False
    reads = records.reads
    _, repeated = _export(tmp_path, records, job=job, instruction_source=None)
    assert repeated == manifest and records.reads == reads


def test_instruction_companion_filters_only_when_a_grader_really_ran(tmp_path):
    job = SimpleNamespace(**{**vars(INSTRUCTION_JOB),
                             "filter": SimpleNamespace(grader_id="fixture", threshold=0.5)})
    _, manifest = _export(tmp_path, _Records(n=6, missing=()), job=job,
                         instruction_source=_RawSource(),
                         grade=lambda index, text: (index % 2 == 0, 0.5))
    assert len(_instructions(tmp_path, manifest)) == 4
    assert manifest["rows"] == 8  # rejected rows remain in the original delivery
    assert manifest["instruction"]["omitted"] == {"filter_rejected": 4}
    assert manifest["instruction"]["policy"] == "accepted_completions"
    assert manifest["instruction"]["filter_applied"] is True
    _, unfiltered = _export(tmp_path, _Records(n=6, missing=()), "unfiltered", job=job,
                           instruction_source=_RawSource(), filter_note="filter unavailable")
    assert len(_instructions(tmp_path, unfiltered)) == 8
    assert unfiltered["instruction"]["filter_applied"] is False
    report = json.loads((tmp_path / "bucket" / unfiltered["report"]).read_text())
    assert report["filter"] == {"applied": False, "note": "filter unavailable"}


def test_instruction_shards_respect_actual_byte_and_example_limits(tmp_path):
    # Independent limits: long UTF-8 examples reach 5 MiB first; short rows reach 10k first.
    for delivery_id, records in (("bytes", _Records(n=100, text_len=32_000, missing=())),
                                 ("examples", _Records(n=7600, text_len=1, missing=()))):
        if delivery_id == "bytes":
            for record in records.subs.values():
                for completion in record["completions"]:
                    completion["text"] = "é" * 32_000
        _, manifest = _export(tmp_path, records, delivery_id, job=INSTRUCTION_JOB,
                              instruction_source=_RawSource())
        assert len(manifest["instruction_shards"]) > 1
        assert len(_instructions(tmp_path, manifest)) == manifest["rows"]
    assert not list((tmp_path / "work").glob("**/*.jsonl"))


@pytest.mark.parametrize("blank", [" \t\n", "\ufeff", " \t\ufeff\n"])
def test_instruction_omissions_are_explicit_and_do_not_drop_parquet_rows(tmp_path, blank):
    from reliquary.validator.corpus_service import CorpusPromptSourceError

    class _Source:
        def task_for(self, index):
            if index == 4:
                raise CorpusPromptSourceError("fixture source no longer available")
            return SimpleNamespace(prompt=("😀" * 16_001 if index == 3 else "raw"),
                                   metadata={"system": "retain this instruction"} if index == 0
                                   else {})

    records = _Records(n=6, missing=())
    records.subs[_sid(1)]["completions"][0]["text"] = blank
    records.subs[_sid(1)]["completions"][1]["text"] = "😀" * 16_000
    _, manifest = _export(tmp_path, records, job=INSTRUCTION_JOB, instruction_source=_Source())
    assert manifest["rows"] == 8
    assert len(_instructions(tmp_path, manifest)) == 1
    assert manifest["instruction"]["omitted"] == {
        "system_message": 2, "invalid_or_oversized_text": 3, "prompt_source_unavailable": 2}


def test_legacy_and_unavailable_sources_do_not_claim_instruction_compatibility(tmp_path):
    job = SimpleNamespace(**{**vars(INSTRUCTION_JOB), "renderer_id": "single-turn-v1"})
    _, manifest = _export(tmp_path, _Records(n=1, missing=()), job=job,
                         instruction_source=_RawSource())
    assert manifest["instruction_shards"] == [] and manifest["rows"] == 2
    assert manifest["instruction"]["omitted"] == {"unsupported_renderer": 2}
    _, unavailable = _export(tmp_path, _Records(n=1, missing=()), "unavailable",
                            job=INSTRUCTION_JOB, instruction_note="prompt_source_unavailable")
    assert unavailable["instruction_shards"] == []
    assert unavailable["instruction"]["source"] == {
        "supported": False, "reason": "prompt_source_unavailable"}


def test_a_canonical_eval_source_system_turn_is_not_silently_lost(tmp_path):
    from reliquary.validator.corpus_service import SingleTurnPromptJob

    class _Environment:
        def get_problem(self, position):
            return {"prompt": "raw user question", "system": "essential system instruction"}

    job = SimpleNamespace(**{**vars(INSTRUCTION_JOB), "prompt_source": "eval-set:fixture"})
    _, manifest = _export(tmp_path, _Records(n=1, missing=()), job=job,
                         instruction_source=SingleTurnPromptJob(job, _Environment()))
    assert manifest["rows"] == 2 and manifest["instruction_shards"] == []
    assert manifest["instruction"]["omitted"] == {"system_message": 2}


def test_instruction_source_resolution_uses_the_canonical_single_turn_job(monkeypatch):
    from reliquary.validator import corpus_service

    class _Environment:
        def get_problem(self, position):
            return {"prompt": f"canonical raw {position}"}

    canonical = corpus_service.SingleTurnPromptJob(INSTRUCTION_JOB, _Environment())
    monkeypatch.setattr(corpus_service, "prompt_job_for_spec", lambda job: canonical)
    source, note = instruction_source_for_job(INSTRUCTION_JOB)
    assert source is canonical and note is None
    assert source.task_for(3).prompt == "canonical raw 3"
    legacy = SimpleNamespace(**{**vars(INSTRUCTION_JOB), "renderer_id": "single-turn-v1"})
    assert instruction_source_for_job(legacy) == (None, "unsupported_renderer")
    episode = SimpleNamespace(**{**vars(INSTRUCTION_JOB), "episode": object()})
    assert instruction_source_for_job(episode) == (None, "episode_schema")
    monkeypatch.setattr(corpus_service, "prompt_job_for_spec", lambda job: object())
    assert instruction_source_for_job(INSTRUCTION_JOB) == (None, "unsupported_prompt_source")

    def missing(job):
        raise corpus_service.CorpusPromptSourceError("no local fixture source")

    monkeypatch.setattr(corpus_service, "prompt_job_for_spec", missing)
    assert instruction_source_for_job(INSTRUCTION_JOB) == (None, "prompt_source_unavailable")


def test_an_instruction_upload_failure_never_commits_a_manifest(tmp_path):
    class _FailingSink(LocalDirectorySink):
        async def put_file(self, key, path):
            if key.endswith(".jsonl"):
                raise OSError("fixture upload failure")
            await super().put_file(key, path)

    with pytest.raises(OSError, match="upload failure"):
        asyncio.run(export_delivery(job=INSTRUCTION_JOB, records=_Records(n=1, missing=()),
                                   sink=_FailingSink(tmp_path / "bucket"), delivery_id="d1",
                                   instruction_source=_RawSource(), work_dir=tmp_path / "work"))
    assert not (tmp_path / "bucket" / "deliveries" / "d1" / "manifest.json").exists()
    assert not list((tmp_path / "work").glob("**/*.jsonl"))
    _, retried = _export(tmp_path, _Records(n=1, missing=()), job=INSTRUCTION_JOB,
                         instruction_source=_RawSource())
    assert len(_instructions(tmp_path, retried)) == 2


@pytest.mark.parametrize("bad", ["../x", "", "a/b", "x" * 200])
def test_a_delivery_id_that_is_not_a_name_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        _export(tmp_path, _Records(n=1), delivery_id=bad)


def test_the_r2_sink_uploads_through_the_scoped_client(tmp_path):
    calls = []

    class _Client:
        def upload_file(self, filename, bucket, key, Config=None, ExtraArgs=None):
            calls.append(("file", bucket, key, Path(filename).read_bytes(), Config is not None,
                          ExtraArgs))

        def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None):
            calls.append(("json", Bucket, Key, Body, ContentType, Metadata))

        def get_object(self, Bucket, Key):
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    sink = R2DeliverySink(bucket="platform", client=_Client())
    path = tmp_path / "f.parquet"
    path.write_bytes(b"PAR1")
    asyncio.run(sink.put_file("deliveries/d1/part-00000.parquet", path))
    asyncio.run(sink.put_json("deliveries/d1/manifest.json", {"a": 1}))
    assert asyncio.run(sink.get_json("deliveries/d1/manifest.json")) is None
    assert calls[0] == ("file", "platform", "deliveries/d1/part-00000.parquet", b"PAR1", True,
                        {"Metadata": {"sha256": hashlib.sha256(b"PAR1").hexdigest()}})
    assert calls[1][:3] == ("json", "platform", "deliveries/d1/manifest.json")
    assert json.loads(calls[1][3]) == {"a": 1}
    assert calls[1][5] == {"sha256": hashlib.sha256(calls[1][3]).hexdigest()}


RUN_ID = "12345678-1234-4234-8234-123456789012"
HTTP_KEY = f"deliveries/subnet-{RUN_ID}/part-00000.parquet"


@pytest.mark.parametrize("url", ["http://example.test", "https://u:p@example.test",
                                  "https://example.test/path", "https://example.test?x=1",
                                  "https://example.test/#x"])
def test_http_delivery_requires_one_https_origin(url):
    with pytest.raises(ValueError, match="HTTPS origin"):
        HTTPDeliverySink(url=url, secret="s" * 32)


def test_http_delivery_streams_files_with_checked_size_and_digest(tmp_path, monkeypatch):
    sink = HTTPDeliverySink(url="https://example.test", secret="s" * 32)
    calls, headers, chunks = [], {}, []
    assert sink.evaluation_supported is False
    assert sink.accepts_delivery_id("subnet-" + RUN_ID)
    assert not sink.accepts_delivery_id("order-other")

    class _Connection:
        def putrequest(self, method, route):
            calls.append((method, route))

        def putheader(self, key, value):
            headers[key] = value

        def endheaders(self):
            pass

        def send(self, body):
            chunks.append(body)

        def getresponse(self):
            return SimpleNamespace(status=201, read=lambda n: b"")

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(sink, "_connection", _Connection)
    data = b"x" * (2 * 1024 * 1024 + 1)
    path = tmp_path / "part.parquet"
    path.write_bytes(data)
    asyncio.run(sink.put_file(HTTP_KEY, path))
    assert calls == [("PUT", f"/api/internal/subnet/tasks/{RUN_ID}/deliveries/part-00000.parquet"),
                     "closed"]
    assert headers["Authorization"] == "Bearer " + "s" * 32
    assert headers["Content-Length"] == str(len(data))
    assert headers["X-Content-SHA256"] == hashlib.sha256(data).hexdigest()
    assert len(chunks) == 3 and max(map(len, chunks)) <= 1024 * 1024
    assert b"".join(chunks) == data
    with pytest.raises(ValueError, match="subnet-run"):
        asyncio.run(sink.put_file("deliveries/another/part.parquet", path))
    monkeypatch.setattr(sink, "max_file_bytes", 10)
    with pytest.raises(ValueError, match="size limit"):
        asyncio.run(sink.put_file(HTTP_KEY, path))


def test_http_delivery_reads_manifest_for_restart_and_refuses_redirects(monkeypatch):
    sink = HTTPDeliverySink(url="https://example.test", secret="s" * 32)
    status = [200]
    document = {"job_id": "order-ops-" + RUN_ID, "keys": [HTTP_KEY]}
    requests = []

    class _Connection:
        def request(self, method, route, headers):
            requests.append((method, route, headers))

        def getresponse(self):
            return SimpleNamespace(status=status[0],
                                   read=lambda n: json.dumps(document).encode())

        def close(self):
            pass

    monkeypatch.setattr(sink, "_connection", _Connection)
    key = f"deliveries/subnet-{RUN_ID}/manifest.json"
    assert asyncio.run(sink.get_json(key)) == document
    assert requests[0][0] == "GET"
    assert requests[0][2] == {"Authorization": "Bearer " + "s" * 32}
    status[0] = 404
    assert asyncio.run(sink.get_json(key)) is None
    status[0] = 302
    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(sink.get_json(key))
    with pytest.raises(ValueError, match="manifest and report"):
        asyncio.run(sink.get_json(HTTP_KEY))


def test_export_obeys_the_sink_shard_limit(tmp_path):
    class _BoundedSink(LocalDirectorySink):
        max_file_bytes = 64 * 1024

    sink = _BoundedSink(tmp_path / "bucket")
    manifest = asyncio.run(export_delivery(
        job=JOB, records=_Records(n=90, text_len=4000), sink=sink, delivery_id="d1",
        work_dir=tmp_path / "work"))
    assert len(manifest["shards"]) > 1
    assert all(shard["bytes"] <= sink.max_file_bytes for shard in manifest["shards"])


def test_manifest_upload_failure_replays_an_immutable_report(tmp_path):
    class _ImmutableSink(LocalDirectorySink):
        fail = True

        async def put_file(self, key, path):
            existing = self._path(key)
            if existing.exists():
                assert existing.read_bytes() == path.read_bytes()
                return
            await super().put_file(key, path)

        async def put_json(self, key, document):
            if key.endswith("manifest.json") and self.fail:
                self.fail = False
                raise RuntimeError("fixture missing final acknowledgement")
            existing = await self.get_json(key)
            if existing is not None:
                assert existing == document
                return
            await super().put_json(key, document)

    sink = _ImmutableSink(tmp_path / "bucket")
    with pytest.raises(RuntimeError, match="acknowledgement"):
        asyncio.run(export_delivery(job=JOB, records=_Records(n=1, missing=()), sink=sink,
                                   delivery_id="d1", clock=lambda: 123.0))
    manifest = asyncio.run(export_delivery(job=JOB, records=_Records(n=1, missing=()), sink=sink,
                                          delivery_id="d1", clock=lambda: 456.0))
    assert manifest["created_at"] == 123.0
