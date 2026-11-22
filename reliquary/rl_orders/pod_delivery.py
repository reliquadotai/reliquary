"""The final model of a training order that ran on rented pods (Miles or any other trainer backend).

The run's published checkpoint (anchor and lossless deltas, in the transport bucket) is mutable and pruned. Once the
order's bill is final, its controller has stopped and its pods' tokens are gone, the exact version the bill closed on
is copied to the private bucket under ``reliquary/rl-delivery/<task>/``, without touching the local disk: a model can
outgrow what the control host has free. The manifest is written last and only once, so its presence means every
listed object was hashed against the publication.

Reads hash each chunk they return against the manifest. This is the delivery of
``reliquary.rl_orders.checkpoint_delivery`` without its local cache and its single-process stop proof.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import Field, model_validator

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.rl_orders.checkpoint_delivery import (
    _RANGE, _SHA, CHUNK_BYTES, FREEZE_SECONDS, INDEX_BYTES, MANIFEST_BYTES, MAX_CHUNKS, MAX_FILES, MAX_READ_BYTES,
    ArtifactRead, CheckpointConflict, CheckpointRangeError, CheckpointUnavailable, R2CheckpointSource, TrainingCheckpointStore, _Artifact, _artifact_id, _Checkpoint, _Strict,
)
from reliquary.shared.ckpt_index import INDEX_FORMAT
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.task_id import TASK_ID_RE
from reliquary.signer.protocol import RlCheckpointIndex

logger = logging.getLogger(__name__)

SCHEMA = "reliquary/pod-training-checkpoint/v1"
PREFIX = "reliquary/rl-delivery/"
# An anchor of a 27B model alone is about 50 GiB: what the manifest's chunk list can describe, not the local cache's 64.
MAX_TOTAL_BYTES = MAX_CHUNKS * CHUNK_BYTES
MIN_COPY_BYTES_PER_S = 4 * 1024 ** 2                   # what a freeze's deadline allows at worst
WAIT_RETRY_S = 60.0
FAILURE_RETRY_S = 60.0
FAILURE_RETRY_MAX_S = 3600.0
_ABSENT = {"NoSuchKey", "404", "NotFound"}
_CONFLICT = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class _PodManifest(_Strict):
    schema_: str = Field(alias="schema", pattern=r"^reliquary/pod-training-checkpoint/v1$")
    task_id: str = Field(min_length=1, max_length=160)
    training_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,95}$")
    order_sha256: str = Field(pattern=_SHA)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    checkpoint: _Checkpoint
    index_sha256: str = Field(pattern=_SHA)
    files: list[_Artifact] = Field(min_length=2, max_length=MAX_FILES)
    manifest_sha256: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def bounded(self):
        if not TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid checkpoint task")
        if len({f.artifact_id for f in self.files}) != len(self.files) or len({f.name for f in self.files}) != len(self.files):
            raise ValueError("duplicate checkpoint artifact")
        if sum(f.size for f in self.files) > MAX_TOTAL_BYTES or sum(len(f.chunks) for f in self.files) > MAX_CHUNKS:
            raise ValueError("checkpoint exceeds delivery limits")
        if sum(f.kind == "index" for f in self.files) != 1 or not any(f.kind == "weights" for f in self.files):
            raise ValueError("checkpoint has no complete reconstruction")
        document = self.model_dump(by_alias=True)
        digest = document.pop("manifest_sha256")
        if digest != canonical_sha256(document):
            raise ValueError("checkpoint manifest checksum differs")
        return self


def _code(exc: Exception) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


def _content_type(name: str) -> str:
    return ("application/json" if name.endswith(".json") else
            "text/plain" if name.endswith((".txt", ".jinja")) else "application/octet-stream")


def _blocks(source: Iterable[bytes]) -> Iterable[bytes]:
    """The source cut into exact ``CHUNK_BYTES`` blocks: the manifest's chunks and the upload's parts."""
    buffer = bytearray()
    for data in source:
        buffer.extend(data)
        while len(buffer) >= CHUNK_BYTES:
            yield bytes(buffer[:CHUNK_BYTES])
            del buffer[:CHUNK_BYTES]
    if buffer:
        yield bytes(buffer)


def private_client():
    """A sync client on the private bucket, from the same ``R2_*`` variables as ``storage.get_s3_client``."""
    import boto3
    from botocore.config import Config

    account = os.getenv("R2_ACCOUNT_ID", "")
    return boto3.client(
        "s3", endpoint_url=os.getenv("R2_ENDPOINT_URL") or f"https://{account}.r2.cloudflarestorage.com",
        region_name=os.getenv("R2_REGION", "auto"), aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        config=Config(signature_version="s3v4", connect_timeout=15, read_timeout=60,
                      retries={"max_attempts": 3, "mode": "standard"},
                      request_checksum_calculation="when_required", response_checksum_validation="when_required"))


class PodCheckpointStore:
    """``client``/``bucket``: the private bucket. ``source``: the publication (``read_index``, ``iter_object``)."""

    def __init__(self, client, bucket: str, source=None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.client, self.bucket, self.source, self.clock = client, bucket, source, clock

    @classmethod
    def from_env(cls, *, with_source: bool = False) -> "PodCheckpointStore":
        source = None
        if with_source:
            from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client

            source = R2CheckpointSource(public_checkpoint_client(), public_checkpoint_bucket())
        return cls(private_client(), os.getenv("R2_BUCKET_ID", "reliquary"), source)

    @staticmethod
    def _folder(task_id: str) -> str:
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise ValueError("invalid checkpoint task")
        return f"{PREFIX}{task_id}"

    def _index(self, run_id: str, version: int) -> dict:
        if self.source is None:
            raise CheckpointUnavailable("checkpoint source unavailable")
        raw = self.source.read_index(run_id)
        if not isinstance(raw, bytes) or not 0 < len(raw) <= INDEX_BYTES:
            raise CheckpointUnavailable("checkpoint index unavailable or too large")
        index = strict_json_loads(raw)
        RlCheckpointIndex.model_validate(index)
        latest = index["latest"]["version"]
        if index["run"] != run_id or latest < version:
            raise CheckpointUnavailable("the publication is behind the version the bill closed on")
        if latest == version:
            return index
        # A trainer stopped between publishing a version and its adoption leaves the index one ahead of the bill.
        target = next((a["sha256"] for a in index["anchors"] if a["version"] == version), None) or next(
            (d["target_sha256"] for d in index["deltas"] if d["to"] == version), None)
        if target is None:
            raise CheckpointUnavailable("the publication no longer lists the version the bill closed on")
        return {**index, "latest": {"version": version, "sha256": target},
                "anchors": [a for a in index["anchors"] if a["version"] <= version],
                "deltas": [d for d in index["deltas"] if d["to"] <= version]}

    def _stored_chunks(self, key: str, size: int, sha256: str, deadline: float) -> list[dict] | None:
        """The chunks of an object an earlier attempt already stored whole, or None: a retry resumes after it."""
        try:
            if int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]) != size:
                return None
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except Exception as exc:
            if _code(exc) in _ABSENT:
                return None
            raise
        digest, chunks, count = hashlib.sha256(), [], 0
        try:
            for block in _blocks(iter(lambda: body.read(CHUNK_BYTES), b"")):
                if self.clock() >= deadline:
                    raise CheckpointUnavailable("checkpoint freeze deadline reached")
                count += len(block)
                digest.update(block)
                chunks.append({"offset": count - len(block), "size": len(block),
                               "sha256": hashlib.sha256(block).hexdigest()})
        finally:
            body.close()
        return chunks if count == size and digest.hexdigest() == sha256 else None

    def _copy(self, task_id: str, *, kind: str, name: str, size: int, sha256: str, source: Iterable[bytes],
              deadline: float) -> dict:
        key = f"{self._folder(task_id)}/objects/{sha256}"
        kept = self._stored_chunks(key, size, sha256, deadline)
        if kept is not None:
            artifact = {"name": name, "kind": kind, "size": size, "sha256": sha256,
                        "content_type": _content_type(name), "chunks": kept}
            artifact["artifact_id"] = _artifact_id(artifact)
            return artifact
        digest, chunks, parts, count, upload_id = hashlib.sha256(), [], [], 0, None
        small = bytearray()
        try:
            for block in _blocks(source):
                if self.clock() >= deadline:
                    raise CheckpointUnavailable("checkpoint freeze deadline reached")
                count += len(block)
                if count > size:
                    raise CheckpointUnavailable("checkpoint source exceeds declared size")
                digest.update(block)
                chunks.append({"offset": count - len(block), "size": len(block),
                               "sha256": hashlib.sha256(block).hexdigest()})
                if size <= CHUNK_BYTES:
                    small.extend(block)
                    continue
                if upload_id is None:
                    upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=key)["UploadId"]
                number = len(parts) + 1
                etag = self.client.upload_part(Bucket=self.bucket, Key=key, UploadId=upload_id, PartNumber=number,
                                               Body=block)["ETag"]
                parts.append({"PartNumber": number, "ETag": etag})
            if count != size or digest.hexdigest() != sha256:
                raise CheckpointUnavailable("checkpoint bytes differ from publication")
            if upload_id is None:
                self.client.put_object(Bucket=self.bucket, Key=key, Body=bytes(small))
            else:
                self.client.complete_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id,
                                                      MultipartUpload={"Parts": parts})
                upload_id = None
        finally:
            if upload_id is not None:
                try:
                    self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
                except Exception:
                    logger.warning("pod delivery: could not abort the upload of %s", key)
        if int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]) != size:
            raise CheckpointUnavailable("stored checkpoint object size differs")
        artifact = {"name": name, "kind": kind, "size": size, "sha256": sha256,
                    "content_type": _content_type(name), "chunks": chunks}
        artifact["artifact_id"] = _artifact_id(artifact)
        return artifact

    def freeze(self, entry: Any, version: int, *, timeout_seconds: int | None = None) -> dict:
        """Copy the run's checkpoint at ``version`` and publish its manifest; a repeat answers the same manifest."""
        order = entry.params["training_order"]
        task_id, run_id = entry.task_id, entry.task_id          # a training task's run is its task id
        existing = self.load_manifest(task_id)
        if existing is not None:
            if existing["checkpoint"]["version"] != version or existing["order_sha256"] != order.get("order_sha256"):
                raise CheckpointConflict("this order's model was already frozen differently")
            return existing
        index = self._index(run_id, version)
        index_sha256 = canonical_sha256(index)
        plan, selected = TrainingCheckpointStore._selected(index, max_total_bytes=MAX_TOTAL_BYTES)
        if timeout_seconds is None:
            total = sum(file["size"] for _, _, file in selected)
            timeout_seconds = max(FREEZE_SECONDS * 2, total // MIN_COPY_BYTES_PER_S)
        deadline = self.clock() + timeout_seconds
        files, logical = [], {}
        for kind, name, file in selected:
            artifact = self._copy(task_id, kind=kind, name=name, size=file["size"], sha256=file["sha256"],
                                  source=self.source.iter_object(file["key"], file["size"]), deadline=deadline)
            files.append(artifact)
            logical[file["key"]] = "objects/" + artifact["artifact_id"]

        def mapped(file: Mapping) -> dict:
            return {**file, "key": logical[file["key"]]}

        anchor = {**plan.anchor, "files": [mapped(f) for f in plan.anchor["files"]],
                  "model_files": [mapped(f) for f in plan.anchor.get("model_files", [])]}
        reconstruction = canonical_json_bytes({"format": INDEX_FORMAT, "run": index["run"], "latest": index["latest"],
                                               "anchors": [anchor], "deltas": [mapped(f) for f in plan.deltas]})
        files.append(self._copy(task_id, kind="index", name="checkpoint-index.json", size=len(reconstruction),
                                sha256=hashlib.sha256(reconstruction).hexdigest(), source=[reconstruction],
                                deadline=deadline))
        if canonical_sha256(self._index(run_id, version)) != index_sha256:
            raise CheckpointUnavailable("checkpoint publication changed during freeze")
        manifest = {"schema": SCHEMA, "task_id": task_id, "training_id": order["training_id"],
                    "order_sha256": order["order_sha256"], "run_id": run_id, "checkpoint": index["latest"],
                    "index_sha256": index_sha256, "files": files}
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _PodManifest.model_validate(manifest)
        raw = canonical_json_bytes(manifest)
        if len(raw) > MANIFEST_BYTES:
            raise CheckpointUnavailable("checkpoint manifest exceeds limits")
        try:
            self.client.put_object(Bucket=self.bucket, Key=f"{self._folder(task_id)}/manifest.json", Body=raw,
                                   ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:
            if _code(exc) not in _CONFLICT:
                raise
            kept = self.load_manifest(task_id)
            if kept is None or kept["manifest_sha256"] != manifest["manifest_sha256"]:
                raise CheckpointConflict("another manifest was published for this order") from exc
        return manifest

    def load_manifest(self, task_id: str) -> dict | None:
        key = f"{self._folder(task_id)}/manifest.json"
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            try:
                raw = body.read(MANIFEST_BYTES + 1)
            finally:
                body.close()
        except Exception as exc:
            if _code(exc) in _ABSENT:
                return None
            raise CheckpointUnavailable("checkpoint manifest unreadable") from exc
        if len(raw) > MANIFEST_BYTES:
            raise CheckpointUnavailable("checkpoint manifest exceeds limit")
        manifest = _PodManifest.model_validate(strict_json_loads(raw)).model_dump(by_alias=True)
        if manifest["task_id"] != task_id:
            raise CheckpointUnavailable("checkpoint manifest belongs to another task")
        return manifest

    def open_range(self, task_id: str, artifact_id: str, range_header: str | None = None) -> ArtifactRead:
        manifest = self.load_manifest(task_id)
        artifact = next((f for f in (manifest or {}).get("files", []) if f["artifact_id"] == artifact_id), None)
        if artifact is None:
            raise FileNotFoundError("checkpoint artifact unavailable")
        size = artifact["size"]
        match = _RANGE.fullmatch(range_header) if isinstance(range_header, str) and len(range_header) <= 128 else None
        if range_header is not None and match is None:
            raise CheckpointRangeError("send one explicit bytes=start-end range")
        start, end = map(int, match.groups()) if match else (0, size - 1)
        if size == 0 and range_header is None:
            return ArtifactRead(b"", artifact, 0, -1, 0, False, hashlib.sha256(b"").hexdigest())
        if start < 0 or end >= size or end < start or end - start + 1 > MAX_READ_BYTES:
            raise CheckpointRangeError("checkpoint range outside object or over 16 MiB")
        covering = [c for c in artifact["chunks"] if c["offset"] + c["size"] > start and c["offset"] <= end]
        first, stop = covering[0]["offset"], covering[-1]["offset"] + covering[-1]["size"]
        key = f"{self._folder(task_id)}/objects/{artifact['sha256']}"
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key, Range=f"bytes={first}-{stop - 1}")["Body"]
            try:
                held = body.read(stop - first + 1)
            finally:
                body.close()
        except Exception as exc:
            raise CheckpointUnavailable("checkpoint object unreadable") from exc
        if len(held) != stop - first:
            raise CheckpointUnavailable("checkpoint object range incomplete")
        for chunk in covering:
            at = chunk["offset"] - first
            if hashlib.sha256(held[at:at + chunk["size"]]).hexdigest() != chunk["sha256"]:
                raise CheckpointUnavailable("stored checkpoint checksum differs")
        data = held[start - first:end - first + 1]
        return ArtifactRead(data, artifact, start, end, size, range_header is not None, hashlib.sha256(data).hexdigest())


def trainer_tokens_live(task_id: str, documents: Iterable[Mapping], *, now: float) -> bool:
    """Whether a pod of ``task_id`` still holds a usable token."""
    return any(d.get("scope") == "trainer" and d.get("task_id") == task_id and d.get("status") == "active"
               and float(d.get("expires_at", 0)) > now for d in documents)


def deliver_order(entry: Any, store: PodCheckpointStore, *, is_final: Callable[[Any], bool],
                  read_summary: Callable[[str], Mapping | None], read_executors: Callable[[], Iterable[Mapping]],
                  clock: Callable[[], float] = time.time) -> str:
    """One attempt: ``delivered`` (the manifest exists), ``no_steps`` (the order trained nothing: nothing to deliver)
    or ``waiting`` (its bill is not final yet, or a pod still holds a token)."""
    if store.load_manifest(entry.task_id) is not None:
        return "delivered"
    if not is_final(entry):
        return "waiting"
    summary = read_summary(entry.task_id)
    if summary is None or summary.get("task_id") != entry.task_id:
        return "waiting"
    version = summary.get("latest_checkpoint")
    if version is None or int(version) <= 0:
        return "no_steps"
    if trainer_tokens_live(entry.task_id, read_executors(), now=clock()):
        return "waiting"
    store.freeze(entry, int(version))
    return "delivered"


def read_executors() -> list[dict]:
    from reliquary.infrastructure import corpus_executor_store

    return asyncio.run(corpus_executor_store.list_executors())


class PodDeliveries:
    """Runs ``deliver(entry)`` for one order at a time, off the supervisor's tick: a model takes minutes to copy.
    An order that failed is retried with a doubling delay; until it succeeds its bill stays held on the platform."""

    def __init__(self, deliver: Callable[[Any], str] | None = None, *, clock: Callable[[], float] = time.time) -> None:
        self._deliver = deliver if deliver is not None else self._from_env
        self._clock = clock
        self._store: PodCheckpointStore | None = None
        self._thread: threading.Thread | None = None
        self._done: set[str] = set()
        self._retry_at: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    def _from_env(self, entry: Any) -> str:
        from reliquary.validator.rl_supervisor import admin_final
        from reliquary.validator.rl_spend import read_spend

        if self._store is None:
            self._store = PodCheckpointStore.from_env(with_source=True)
        return deliver_order(entry, self._store, is_final=admin_final, read_summary=read_spend,
                             read_executors=read_executors)

    def offer(self, entry: Any) -> bool:
        """Start ``entry``'s delivery if it is due and nothing else is copying."""
        task_id = entry.task_id
        if task_id in self._done or self._clock() < self._retry_at.get(task_id, 0.0):
            return False
        if self._thread is not None and self._thread.is_alive():
            return False
        self._thread = threading.Thread(target=self._run, args=(entry,), name="pod-delivery", daemon=True)
        self._thread.start()
        return True

    def _run(self, entry: Any) -> None:
        task_id = entry.task_id
        try:
            outcome = self._deliver(entry)
        except Exception:
            failures = self._failures.get(task_id, 0) + 1
            self._failures[task_id] = failures
            delay = min(FAILURE_RETRY_S * 2 ** (failures - 1), FAILURE_RETRY_MAX_S)
            self._retry_at[task_id] = self._clock() + delay
            logger.exception("pod delivery of %s failed; retrying in %.0fs", task_id, delay)
            return
        if outcome == "waiting":
            self._retry_at[task_id] = self._clock() + WAIT_RETRY_S
            return
        self._done.add(task_id)
        self._retry_at.pop(task_id, None)
        self._failures.pop(task_id, None)
        logger.info("pod delivery of %s: %s", task_id, outcome)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
