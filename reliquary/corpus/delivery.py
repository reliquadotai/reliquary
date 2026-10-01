"""Export v2: a job's passing rows delivered as Parquet shards.

Streamed end to end: verdicts and records are read a window at a time, a
bounded number in flight, and each shard is uploaded and deleted locally as
soon as it closes, so neither memory nor disk ever holds a whole job. The
manifest is written last; its presence is what makes a delivery complete.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DELIVERY_SCHEMA = "reliquary/corpus-delivery/v2"
DELIVERY_PREFIX = "deliveries"
SHARD_MAX_BYTES = 500 * 1024 * 1024
ROW_GROUP_ROWS = 2048
# A row group is also flushed at this many raw bytes: long completions are ~128 KB each.
ROW_GROUP_MAX_BYTES = 64 * 1024 * 1024
# Reads in flight at once, as the auditor reads records.
READ_CONCURRENCY = 16
# Verdict ids handled per window: what bounds the records held in memory
# (a record of 8 completions near 32k tokens is ~1 MB).
READ_WINDOW = 64
# Room left in a shard for the Parquet footer and page headers.
SHARD_OVERHEAD_BYTES = 1024 * 1024

_DELIVERY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

ROW_FIELDS = ("job_id", "submission_id", "prompt_index", "completion_index", "prompt",
              "completion", "completion_tokens", "accepted", "score")


def validated_delivery_id(delivery_id: Any) -> str:
    if not isinstance(delivery_id, str) or not _DELIVERY_ID_RE.fullmatch(delivery_id):
        raise ValueError(f"delivery id {delivery_id!r} is not a name")
    return delivery_id


def _row_schema():
    import pyarrow as pa

    return pa.schema([
        ("job_id", pa.string()), ("submission_id", pa.string()),
        ("prompt_index", pa.int64()), ("completion_index", pa.int32()),
        ("prompt", pa.string()), ("completion", pa.string()),
        ("completion_tokens", pa.int32()), ("accepted", pa.bool_()), ("score", pa.float64()),
    ])


async def _bounded(calls, gate: asyncio.Semaphore) -> list:
    async def one(call):
        async with gate:
            return await call

    return await asyncio.gather(*(one(c) for c in calls))


async def delivery_rows(*, job, records, counts: dict, grade=None,
                        concurrency: int = READ_CONCURRENCY,
                        window: int = READ_WINDOW) -> AsyncIterator[dict]:
    """One row per completion of every passing submission, in verdict-id order.

    No hotkey: the delivery is the work, not who did it. ``grade`` annotates
    (``accepted``, ``score``) and never drops a row. ``counts`` is filled in as
    the rows go.
    """
    ids = list(await records.list_verdict_ids(job.job_id))
    counts.update(verdicts=len(ids), passing_submissions=0, missing_records=0, rows=0,
                  rows_accepted=0 if grade is not None else None)
    gate = asyncio.Semaphore(concurrency)
    for start in range(0, len(ids), window):
        chunk = ids[start:start + window]
        verdicts = await _bounded((records.read_verdict(job.job_id, s) for s in chunk), gate)
        passing = [sid for sid, v in zip(chunk, verdicts) if v and v.get("passed")]
        found = await _bounded((records.read_submission(job.job_id, s) for s in passing), gate)
        rows = []
        for sid, record in zip(passing, found):
            if record is None:
                # A verdict can be visible before its record (R2 is not transactional).
                logger.warning("corpus delivery: submission %s passed but has no record", sid[:12])
                counts["missing_records"] += 1
                continue
            counts["passing_submissions"] += 1
            for index, completion in enumerate(record["completions"]):
                rows.append({
                    "job_id": job.job_id, "submission_id": sid,
                    "prompt_index": int(record["prompt_index"]), "completion_index": index,
                    "prompt": record["rendered_prompt"], "completion": completion["text"],
                    "completion_tokens": len(completion.get("tokens") or ()),
                    "accepted": None, "score": None,
                })
        if grade is not None and rows:
            def annotate(batch=rows):
                for row in batch:
                    accepted, score = grade(row["prompt_index"], row["completion"])
                    row["accepted"], row["score"] = bool(accepted), float(score)

            # Graders are synchronous and may be slow: off the event loop.
            await asyncio.to_thread(annotate)
        for row in rows:
            counts["rows"] += 1
            if row["accepted"]:
                counts["rows_accepted"] += 1
            yield row


def _raw_size(row: Mapping) -> int:
    """An upper bound on a row's encoded bytes before compression."""
    size = 0
    for value in row.values():
        size += 8 + (len(value.encode()) if isinstance(value, str) else 0)
    return size


class _ShardWriter:
    """Parquet shards no larger than ``max_bytes``, handed to ``on_close`` as
    each one closes."""

    def __init__(self, directory: Path, *, max_bytes: int, row_group_rows: int,
                 on_close: Callable[[Path, int], Any]) -> None:
        self._directory = directory
        self._max = max_bytes
        self._group_rows = row_group_rows
        self._on_close = on_close
        self._schema = _row_schema()
        self._buffer: list[dict] = []
        self._buffer_bytes = 0
        self._writer = None
        self._path: Path | None = None
        self._shard_bytes = 0
        self._shard_rows = 0
        self.index = 0

    async def add(self, row: dict) -> None:
        self._buffer.append(row)
        self._buffer_bytes += _raw_size(row)
        if len(self._buffer) >= self._group_rows or self._buffer_bytes >= ROW_GROUP_MAX_BYTES:
            await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        budget = self._max - SHARD_OVERHEAD_BYTES if self._max > 2 * SHARD_OVERHEAD_BYTES \
            else self._max // 2
        if self._writer is not None and self._shard_bytes + self._buffer_bytes > budget:
            await self._close_shard()
        if self._writer is None:
            self._path = self._directory / f"part-{self.index:05d}.parquet"
            self._writer = pq.ParquetWriter(str(self._path), self._schema, compression="zstd")
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        await asyncio.to_thread(self._writer.write_table, table)
        self._shard_bytes += self._buffer_bytes
        self._shard_rows += len(self._buffer)
        self._buffer, self._buffer_bytes = [], 0

    async def _close_shard(self) -> None:
        await asyncio.to_thread(self._writer.close)
        path, rows = self._path, self._shard_rows
        self._writer, self._path, self._shard_bytes, self._shard_rows = None, None, 0, 0
        self.index += 1
        size = path.stat().st_size
        if size > self._max:
            raise RuntimeError(f"shard {path.name} is {size} bytes, over {self._max}")
        await self._on_close(path, rows)

    async def close(self) -> None:
        await self._flush()
        if self._writer is not None:
            await self._close_shard()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def export_delivery(*, job, records, sink, delivery_id: str, grade=None,
                          filter_note: str | None = None, work_dir: str | Path | None = None,
                          shard_max_bytes: int = SHARD_MAX_BYTES,
                          row_group_rows: int = ROW_GROUP_ROWS,
                          concurrency: int = READ_CONCURRENCY, window: int = READ_WINDOW,
                          clock: Callable[[], float] = time.time) -> dict:
    """Write ``deliveries/{delivery_id}/``: the shards, ``report.json``, then
    ``manifest.json``. A delivery whose manifest exists is returned as stored."""
    delivery_id = validated_delivery_id(delivery_id)
    prefix = f"{DELIVERY_PREFIX}/{delivery_id}"
    manifest_key = f"{prefix}/manifest.json"
    stored = await sink.get_json(manifest_key)
    if stored is not None:
        return stored
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"delivery-{delivery_id}-", dir=root))
    shards: list[dict] = []

    async def uploaded(path: Path, rows: int) -> None:
        digest = await asyncio.to_thread(_sha256, path)
        size = path.stat().st_size
        key = f"{prefix}/{path.name}"
        await sink.put_file(key, path)
        shards.append({"name": path.name, "key": key, "rows": rows, "bytes": size,
                       "sha256": digest})
        path.unlink()

    counts: dict = {}
    try:
        writer = _ShardWriter(directory, max_bytes=shard_max_bytes,
                              row_group_rows=row_group_rows, on_close=uploaded)
        async for row in delivery_rows(job=job, records=records, counts=counts, grade=grade,
                                       concurrency=concurrency, window=window):
            await writer.add(row)
        await writer.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    report = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "created_at": clock(), "job": job.to_contract(), "counts": counts,
        "filter": ({"applied": True, "grader_id": job.filter.grader_id,
                    "threshold": job.filter.threshold} if grade is not None
                   else {"applied": False, "note": filter_note or "the job declares no filter"}),
        "shards": len(shards),
    }
    report_key = f"{prefix}/report.json"
    await sink.put_json(report_key, report)
    manifest = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "created_at": report["created_at"], "rows": counts.get("rows", 0), "shards": shards,
        "columns": list(ROW_FIELDS), "report": report_key,
        "keys": [s["key"] for s in shards] + [report_key, manifest_key],
    }
    await sink.put_json(manifest_key, manifest)
    logger.info("corpus delivery %s of %s: %d rows in %d shards", delivery_id, job.job_id,
                manifest["rows"], len(shards))
    return manifest


class LocalDirectorySink:
    """A directory standing in for the platform bucket (tests, dry runs)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def put_file(self, key: str, path: Path) -> None:
        await asyncio.to_thread(shutil.copyfile, path, self._path(key))

    async def put_json(self, key: str, document: Mapping) -> None:
        self._path(key).write_text(json.dumps(document, sort_keys=True))

    async def get_json(self, key: str) -> dict | None:
        path = self._root / key
        return json.loads(path.read_text()) if path.exists() else None

    async def put_bytes(self, key: str, body: bytes) -> None:
        self._path(key).write_bytes(body)

    async def get_bytes(self, key: str) -> bytes | None:
        path = self._root / key
        return path.read_bytes() if path.exists() else None

    async def get_file(self, key: str, path: Path) -> bool:
        source = self._root / key
        if not source.exists():
            return False
        await asyncio.to_thread(shutil.copyfile, source, path)
        return True


class R2DeliverySink:
    """The platform bucket, through credentials scoped to it and held by the
    admin host only (``RELIQUARY_PLATFORM_R2_*``), never the subnet's."""

    def __init__(self, *, bucket: str, client=None) -> None:
        self._bucket = bucket
        self._client = client

    @classmethod
    def from_environment(cls) -> "R2DeliverySink":
        import boto3
        from botocore.config import Config

        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise RuntimeError(f"{name} is not set; deliveries need the platform bucket")
            return value

        account = required("RELIQUARY_PLATFORM_R2_ACCOUNT_ID")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("RELIQUARY_PLATFORM_R2_ENDPOINT_URL")
            or f"https://{account}.r2.cloudflarestorage.com",
            region_name=os.getenv("R2_REGION", "us-east-1"),
            aws_access_key_id=required("RELIQUARY_PLATFORM_R2_ACCESS_KEY_ID"),
            aws_secret_access_key=required("RELIQUARY_PLATFORM_R2_SECRET_ACCESS_KEY"),
            config=Config(connect_timeout=15, read_timeout=60,
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
        return cls(bucket=required("RELIQUARY_PLATFORM_BUCKET"), client=client)

    async def put_file(self, key: str, path: Path) -> None:
        from boto3.s3.transfer import TransferConfig

        config = TransferConfig(multipart_threshold=32 * 1024 * 1024,
                                multipart_chunksize=32 * 1024 * 1024, max_concurrency=8)
        await asyncio.to_thread(self._client.upload_file, str(path), self._bucket, key,
                                Config=config)

    async def put_json(self, key: str, document: Mapping) -> None:
        body = json.dumps(document, sort_keys=True).encode()
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key,
                                Body=body, ContentType="application/json")

    async def get_json(self, key: str) -> dict | None:
        from botocore.exceptions import ClientError

        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket,
                                               Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return json.loads(await asyncio.to_thread(response["Body"].read))

    async def put_bytes(self, key: str, body: bytes) -> None:
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key, Body=body)

    async def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket,
                                               Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return await asyncio.to_thread(response["Body"].read)

    async def get_file(self, key: str, path: Path) -> bool:
        """Stream an object to ``path``; False when it does not exist."""
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self._client.download_file, self._bucket, key, str(path))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise
        return True


__all__ = [
    "DELIVERY_SCHEMA",
    "LocalDirectorySink",
    "R2DeliverySink",
    "ROW_FIELDS",
    "SHARD_MAX_BYTES",
    "delivery_rows",
    "export_delivery",
    "validated_delivery_id",
]
