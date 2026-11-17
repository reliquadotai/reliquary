"""Private, immutable customer checkpoint snapshots copied from the existing RL publication.

The published anchor/delta index is a source, never the customer's mutable delivery. A stopped
runtime and its final usage receipt pin one exact index. Every reachable object is copied and
hashed before a storage-free reconstruction index and manifest become visible. Reads verify
bounded cached chunks before returning bytes; this is a service manifest, not a miner signature.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.shared.ckpt_index import (
    INDEX_FORMAT, anchor_key, delta_key, index_key, model_file_key, model_file_name_ok, plan_download,
)
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.task_id import TASK_ID_RE
from reliquary.signer.protocol import RlCheckpointIndex

SCHEMA = "reliquary/training-checkpoint/v1"
INDEX_BYTES = MANIFEST_BYTES = 2 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
MAX_READ_BYTES = 16 * 1024 * 1024
MAX_OBJECT_BYTES = 16 * 1024 ** 3
MAX_TOTAL_BYTES = 64 * 1024 ** 3
MAX_FILES = 128
MAX_CHUNKS = 16384
FREEZE_SECONDS = 1800
_SHA = r"^[0-9a-f]{64}$"
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RANGE = re.compile(r"^bytes=(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$")
MODEL_FILES = frozenset({
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "tokenizer.model",
    "spiece.model", "vocab.json", "vocab.txt", "merges.txt", "preprocessor_config.json",
    "processor_config.json", "image_processor_config.json", "video_preprocessor_config.json",
    "audio_preprocessor_config.json",
})


class CheckpointUnavailable(RuntimeError):
    pass


class CheckpointConflict(ValueError):
    pass


class CheckpointRangeError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class _Chunk(_Strict):
    offset: int = Field(ge=0)
    size: int = Field(ge=1, le=CHUNK_BYTES)
    sha256: str = Field(pattern=_SHA)


class _Artifact(_Strict):
    artifact_id: str = Field(pattern=_SHA)
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    kind: str = Field(pattern=r"^(weights|delta|model|index)$")
    size: int = Field(ge=0, le=MAX_OBJECT_BYTES)
    sha256: str = Field(pattern=_SHA)
    content_type: str = Field(pattern=r"^(application/(json|octet-stream)|text/plain)$")
    chunks: list[_Chunk] = Field(max_length=MAX_CHUNKS)

    @model_validator(mode="after")
    def complete_chunks(self):
        if self.kind in {"weights", "index"} and self.size == 0:
            raise ValueError("checkpoint weights and reconstruction index must be nonempty")
        offset = 0
        for chunk in self.chunks:
            if chunk.offset != offset or chunk.size != min(CHUNK_BYTES, self.size - offset):
                raise ValueError("checkpoint chunks are not a complete contiguous partition")
            offset += chunk.size
        if offset != self.size or self.artifact_id != _artifact_id(self.model_dump()):
            raise ValueError("checkpoint artifact identity or size differs")
        return self


class _Checkpoint(_Strict):
    version: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA)


class _Manifest(_Strict):
    schema_: str = Field(alias="schema", pattern=r"^reliquary/training-checkpoint/v1$")
    task_id: str = Field(min_length=1, max_length=160)
    training_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,95}$")
    order_sha256: str = Field(pattern=_SHA)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    runtime_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    checkpoint: _Checkpoint
    usage_sha256: str = Field(pattern=_SHA)
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


def _artifact_id(artifact: Mapping[str, Any]) -> str:
    return canonical_sha256({key: artifact[key] for key in ("name", "kind", "size", "sha256")})


def _private_dir(path: Path, *, create: bool = True) -> None:
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        else:
            _fsync_dir(path.parent)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise CheckpointUnavailable("checkpoint cache directory must be private and owned")


@contextmanager
def _open_file(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise CheckpointUnavailable("checkpoint cache file must be private and regular")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle, metadata
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ArtifactRead:
    data: bytes
    artifact: dict
    start: int
    end: int
    total: int
    partial: bool
    sha256: str


class R2CheckpointSource:
    """Read-only adapter over the existing checkpoint bucket; no listing or presigned URLs."""

    def __init__(self, client, bucket: str):
        self.client, self.bucket = client, bucket

    def read_index(self, run_id: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=index_key(run_id))
        body = response["Body"]
        try:
            data = body.read(INDEX_BYTES + 1)
            if len(data) > INDEX_BYTES:
                raise CheckpointUnavailable("checkpoint index exceeds limit")
            return data
        finally:
            body.close()

    def iter_object(self, key: str, expected_size: int) -> Iterable[bytes]:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        try:
            if response.get("ContentLength") != expected_size:
                raise CheckpointUnavailable("checkpoint source size differs")
            while chunk := body.read(CHUNK_BYTES):
                yield chunk
        finally:
            body.close()


class TrainingCheckpointStore:
    def __init__(self, root: str | Path, source=None, *, clock=time.monotonic):
        self.root, self.source, self.clock = Path(root), source, clock
        if not self.root.is_absolute() or ".." in self.root.parts:
            raise ValueError("checkpoint root must be absolute")
        _private_dir(self.root)

    def _folder(self, task_id: str, *, create: bool = True) -> Path:
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise ValueError("invalid checkpoint task")
        parent = self.root / task_id
        _private_dir(parent, create=create)
        folder = parent / "checkpoint"
        _private_dir(folder, create=create)
        _private_dir(folder / "objects", create=create)
        return folder

    def _source_index(self, stopped) -> dict:
        if self.source is None:
            raise CheckpointUnavailable("checkpoint source unavailable")
        if not _COMPONENT.fullmatch(stopped.run_id):
            raise ValueError("invalid checkpoint run")
        raw = self.source.read_index(stopped.run_id)
        if not isinstance(raw, bytes) or not 0 < len(raw) <= INDEX_BYTES:
            raise CheckpointUnavailable("checkpoint index unavailable or too large")
        index = strict_json_loads(raw)
        RlCheckpointIndex.model_validate(index)
        if index["run"] != stopped.run_id or index["latest"]["version"] != stopped.latest_checkpoint:
            raise CheckpointUnavailable("checkpoint is not the stopped runtime's adopted version")
        return index

    def checkpoint_reference(self, stopped) -> dict:
        index = self._source_index(stopped)
        # Resolve a complete path before usage can bind this checkpoint.
        self._selected(index)
        return {"version": index["latest"]["version"], "index_sha256": canonical_sha256(index)}

    @staticmethod
    def _selected(index: dict) -> tuple[Any, list[tuple[str, str, dict]]]:
        plan = plan_download(index, None)
        anchor = plan.anchor
        if anchor is None:
            raise CheckpointUnavailable("checkpoint has no anchor")
        selected = []
        for file in anchor["files"]:
            name = file["key"].rsplit("/", 1)[-1]
            if (not _COMPONENT.fullmatch(name) or not name.endswith(".safetensors")
                    or file["key"] != anchor_key(index["run"], anchor["version"], name)):
                raise ValueError("checkpoint weight key is outside its exact anchor")
            if file["size"] == 0:
                raise CheckpointUnavailable("checkpoint weight object is empty")
            selected.append(("weights", name, file))
        # The source publisher also carries unrelated plain files from model_dir.
        model_files = [file for file in (anchor.get("model_files") or []) if file["name"] in MODEL_FILES]
        names = {file.get("name") for file in model_files}
        tokenizers = {"tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "spiece.model"}
        if "config.json" not in names or not names.intersection(tokenizers):
            raise CheckpointUnavailable("checkpoint has no complete model configuration and tokenizer")
        for file in model_files:
            name = file["name"]
            if (not model_file_name_ok(name) or not _COMPONENT.fullmatch(name)
                    or file["key"] != model_file_key(index["run"], anchor["version"], name)):
                raise ValueError("checkpoint model key differs")
            selected.append(("model", name, file))
        plan.anchor = {**anchor, "model_files": model_files}
        for file in plan.deltas:
            if file["key"] != delta_key(index["run"], file["from"], file["to"]):
                raise ValueError("checkpoint delta key differs")
            selected.append(("delta", file["key"].rsplit("/", 1)[-1], file))
        if len(selected) + 1 > MAX_FILES or sum(file["size"] for _, _, file in selected) > MAX_TOTAL_BYTES:
            raise ValueError("checkpoint exceeds delivery limits")
        if len({file["key"] for _, _, file in selected}) != len(selected):
            raise ValueError("checkpoint lists duplicate source objects")
        for _, _, file in selected:
            if file["size"] > MAX_OBJECT_BYTES:
                raise ValueError("checkpoint object exceeds delivery limit")
        return plan, selected

    def _materialize(self, folder: Path, *, kind: str, name: str, size: int, sha256: str,
                     source: Iterable[bytes], deadline: float) -> dict:
        digest, chunks, buffer, count = hashlib.sha256(), [], bytearray(), 0
        descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=folder / "objects")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for data in source:
                    if self.clock() >= deadline:
                        raise CheckpointUnavailable("checkpoint freeze deadline reached")
                    if not isinstance(data, bytes) or len(data) > CHUNK_BYTES or count + len(data) > size:
                        raise CheckpointUnavailable("checkpoint source exceeds declared size")
                    count += len(data)
                    digest.update(data)
                    handle.write(data)
                    buffer.extend(data)
                    while len(buffer) >= CHUNK_BYTES:
                        block = bytes(buffer[:CHUNK_BYTES])
                        del buffer[:CHUNK_BYTES]
                        chunks.append({"offset": len(chunks) * CHUNK_BYTES, "size": len(block),
                                       "sha256": hashlib.sha256(block).hexdigest()})
                if buffer:
                    chunks.append({"offset": len(chunks) * CHUNK_BYTES, "size": len(buffer),
                                   "sha256": hashlib.sha256(buffer).hexdigest()})
                if count != size or digest.hexdigest() != sha256 or self.clock() >= deadline:
                    raise CheckpointUnavailable("checkpoint bytes differ from publication")
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o400)
            destination = folder / "objects" / sha256
            try:
                os.link(temporary, destination, follow_symlinks=False)
                _fsync_dir(destination.parent)
            except FileExistsError:
                with _open_file(destination) as (handle, metadata):
                    if metadata.st_size != size or _hash_stream(handle, deadline, self.clock) != sha256:
                        raise CheckpointUnavailable("immutable checkpoint object differs")
            content_type = ("application/json" if name.endswith(".json") else
                            "text/plain" if name.endswith((".txt", ".jinja")) else "application/octet-stream")
            artifact = {"name": name, "kind": kind, "size": size, "sha256": sha256,
                        "content_type": content_type, "chunks": chunks}
            artifact["artifact_id"] = _artifact_id(artifact)
            return artifact
        finally:
            Path(temporary).unlink(missing_ok=True)

    def freeze(self, stopped, usage_receipt: Mapping[str, Any], *, timeout_seconds: int = FREEZE_SECONDS) -> dict:
        from reliquary.validator.rl_training_usage import stopped_document, validate_usage_envelope

        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
            raise ValueError("invalid checkpoint freeze deadline")
        usage_receipt = validate_usage_envelope(usage_receipt)
        receipt, digest = usage_receipt["receipt"], usage_receipt["sha256"]
        if receipt["termination_sha256"] != canonical_sha256(stopped_document(stopped)):
            raise ValueError("checkpoint usage termination differs")
        expected = {"task_id": stopped.task_id, "order_sha256": stopped.order_sha256,
                    "run_id": stopped.run_id, "runtime_id": stopped.runtime_id}
        if any(receipt.get(key) != value for key, value in expected.items()) or receipt.get("checkpoint") is None:
            raise ValueError("checkpoint usage belongs to another runtime")
        reference = receipt["checkpoint"]
        folder = self._folder(stopped.task_id)
        descriptor = os.open(folder / ".freeze.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointUnavailable("checkpoint freeze is already running") from exc
            existing = self.load_manifest(stopped.task_id)
            deadline = self.clock() + timeout_seconds
            if existing is not None:
                if (existing["usage_sha256"] != digest
                        or existing["checkpoint"]["version"] != reference.get("version")
                        or existing["index_sha256"] != reference.get("index_sha256")
                        or any(existing[key] != value for key, value in expected.items())):
                    raise CheckpointConflict("checkpoint was already frozen for different usage")
                for artifact in existing["files"]:
                    with _open_file(folder / "objects" / artifact["sha256"]) as (handle, metadata):
                        if metadata.st_size != artifact["size"] or _hash_stream(handle, deadline, self.clock) != artifact["sha256"]:
                            raise CheckpointUnavailable("frozen checkpoint cache differs")
                return existing
            index = self._source_index(stopped)
            if reference != {"version": index["latest"]["version"], "index_sha256": canonical_sha256(index)}:
                raise ValueError("checkpoint index differs from final usage")
            plan, selected = self._selected(index)
            files, logical = [], {}
            for kind, name, file in selected:
                artifact = self._materialize(folder, kind=kind, name=name, size=file["size"], sha256=file["sha256"],
                                             source=self.source.iter_object(file["key"], file["size"]), deadline=deadline)
                files.append(artifact)
                logical[file["key"]] = "objects/" + artifact["artifact_id"]
            def mapped(file):
                return {**file, "key": logical[file["key"]]}
            anchor = {**plan.anchor, "files": [mapped(file) for file in plan.anchor["files"]],
                      "model_files": [mapped(file) for file in plan.anchor.get("model_files", [])]}
            reconstruction = {"format": INDEX_FORMAT, "run": index["run"], "latest": index["latest"],
                              "anchors": [anchor], "deltas": [mapped(file) for file in plan.deltas]}
            data = canonical_json_bytes(reconstruction)
            files.append(self._materialize(folder, kind="index", name="checkpoint-index.json", size=len(data),
                                           sha256=hashlib.sha256(data).hexdigest(), source=[data], deadline=deadline))
            if canonical_sha256(self._source_index(stopped)) != reference["index_sha256"]:
                raise CheckpointUnavailable("checkpoint publication changed during freeze")
            manifest = {"schema": SCHEMA, **expected, "training_id": receipt["training_id"],
                        "checkpoint": index["latest"], "usage_sha256": digest,
                        "index_sha256": reference["index_sha256"], "files": files}
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            _Manifest.model_validate(manifest)
            raw = canonical_json_bytes(manifest)
            if len(raw) > MANIFEST_BYTES or self.clock() >= deadline:
                raise CheckpointUnavailable("checkpoint manifest exceeds limits")
            _immutable_file(folder / "manifest.json", raw)
            return manifest
        finally:
            os.close(descriptor)

    def load_manifest(self, task_id: str) -> dict | None:
        try:
            path = self._folder(task_id, create=False) / "manifest.json"
            with _open_file(path) as (handle, metadata):
                if metadata.st_size > MANIFEST_BYTES:
                    raise CheckpointUnavailable("checkpoint manifest exceeds limit")
                raw = handle.read(MANIFEST_BYTES + 1)
        except FileNotFoundError:
            return None
        manifest = _Manifest.model_validate(strict_json_loads(raw)).model_dump(by_alias=True)
        if manifest["task_id"] != task_id:
            raise CheckpointUnavailable("checkpoint manifest belongs to another task")
        return manifest

    def open_range(self, task_id: str, artifact_id: str, range_header: str | None = None) -> ArtifactRead:
        if not isinstance(artifact_id, str) or not re.fullmatch(_SHA, artifact_id):
            raise FileNotFoundError("checkpoint artifact unavailable")
        manifest = self.load_manifest(task_id)
        artifact = next((file for file in (manifest or {}).get("files", []) if file["artifact_id"] == artifact_id), None)
        if artifact is None:
            raise FileNotFoundError("checkpoint artifact unavailable")
        size = artifact["size"]
        match = _RANGE.fullmatch(range_header) if isinstance(range_header, str) and len(range_header) <= 128 else None
        if range_header is not None:
            if match is None:
                raise CheckpointRangeError("send one explicit bytes=start-end range")
            start, end = map(int, match.groups())
        else:
            start, end = 0, size - 1
        if start < 0 or end >= size or end < start or end - start + 1 > MAX_READ_BYTES:
            if not (range_header is None and size == 0):
                raise CheckpointRangeError("checkpoint range outside object or over 16 MiB")
        result = bytearray()
        with _open_file(self._folder(task_id, create=False) / "objects" / artifact["sha256"]) as (handle, metadata):
            if metadata.st_size != size:
                raise CheckpointUnavailable("checkpoint cache size differs")
            for chunk in artifact["chunks"]:
                offset, stop = chunk["offset"], chunk["offset"] + chunk["size"]
                if stop <= start or offset > end:
                    continue
                handle.seek(offset)
                block = handle.read(chunk["size"])
                if len(block) != chunk["size"] or hashlib.sha256(block).hexdigest() != chunk["sha256"]:
                    raise CheckpointUnavailable("checkpoint cache checksum differs")
                result.extend(block[max(0, start - offset):min(len(block), end - offset + 1)])
        data = bytes(result)
        if len(data) != max(0, end - start + 1):
            raise CheckpointUnavailable("checkpoint cache range incomplete")
        return ArtifactRead(data, artifact, start, end, size, range_header is not None, hashlib.sha256(data).hexdigest())


def _hash_stream(handle, deadline: float, clock) -> str:
    digest = hashlib.sha256()
    while data := handle.read(CHUNK_BYTES):
        if clock() >= deadline:
            raise CheckpointUnavailable("checkpoint verification deadline reached")
        digest.update(data)
    return digest.hexdigest()


def _immutable_file(path: Path, raw: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        try:
            os.link(temporary, path, follow_symlinks=False)
            _fsync_dir(path.parent)
        except FileExistsError:
            with _open_file(path) as (handle, metadata):
                if metadata.st_size != len(raw) or handle.read(len(raw) + 1) != raw:
                    raise CheckpointConflict("immutable checkpoint manifest differs")
    finally:
        Path(temporary).unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
