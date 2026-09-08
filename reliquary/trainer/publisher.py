"""Durable HF checkpoint publication, R2 mirror and conditional candidate commit.

A single local transaction retains its snapshot until the R2 manifest commits.
Recovery recognizes only this transaction's immutable HF commit; foreign or
unavailable HEADs freeze publication. HF history is never rewritten.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable
import uuid

from reliquary.shared.checkpoint_identity import (
    canonical_checkpoint_identity,
    require_checkpoint_number,
    require_checkpoint_repository,
    require_immutable_checkpoint_revision,
)
from reliquary.shared.strict_json import strict_json_loads
from reliquary.trainer.storage_guard import HfStorageGuard
from reliquary.validator.control import write_json

logger = logging.getLogger(__name__)
CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"
R2_CHECKPOINT_PREFIX = "reliquary/checkpoints"
PUBLICATION_RECEIPT = "reliquary_publication.json"
PENDING_PUBLICATION = "publication.json"


class PublicationConflict(RuntimeError):
    """Local or remote state cannot be attributed to this publication."""


def checkpoint_key(revision: str, filename: str) -> str:
    revision = require_immutable_checkpoint_revision(revision)
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("checkpoint filename must be a single path component")
    return f"{R2_CHECKPOINT_PREFIX}/{revision}/{filename}"


def _multipart_transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=32 * 1024 * 1024,
        max_concurrency=16,
    )


def _default_hf_head(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, timeout=30).sha


async def _default_hf_upload(**kwargs) -> str:
    from huggingface_hub import HfApi

    return (await asyncio.to_thread(HfApi().upload_folder, **kwargs)).oid


def _default_hf_verify(
    *, repo_id, revision, parent_revision, commit_message, files
) -> None:
    """Validate commit provenance and content without downloading model weights."""
    from huggingface_hub import HfApi

    api = HfApi()
    commits = api.list_repo_commits(repo_id, revision=revision)
    if (
        len(commits) < 2
        or commits[0].commit_id != revision
        or commits[0].title != commit_message
        or commits[1].commit_id != parent_revision
    ):
        raise PublicationConflict("HF HEAD is not the exact pending publication commit")
    paths = {
        item.path: item
        for item in api.get_paths_info(repo_id, list(files), revision=revision)
    }
    if set(paths) != set(files):
        raise PublicationConflict("HF publication file set is incomplete")
    for name, expected in files.items():
        item = paths[name]
        lfs = getattr(item, "lfs", None)
        digest = lfs.sha256 if lfs is not None else getattr(item, "blob_id", None)
        expected_digest = expected["sha256"] if lfs is not None else expected["blob_id"]
        if item.size != expected["size"] or digest != expected_digest:
            raise PublicationConflict(f"HF publication content mismatch: {name}")


def _file_identity(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise PublicationConflict("snapshot requires regular files without symlinks")
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            blob.update(chunk)
        os.fsync(handle.fileno())
    return {"size": size, "sha256": sha.hexdigest(), "blob_id": blob.hexdigest()}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TrainerPublisher:
    def __init__(
        self,
        *,
        repo_id: str,
        staging_dir: str,
        tokenizer: Any,
        r2_client: Any,
        bucket: str,
        checkpoint_number_floor: int | None = None,
        save_fn: Callable | None = None,
        hf_upload_fn: Callable | None = None,
        storage_guard: HfStorageGuard | None = None,
        hf_head_fn: Callable | None = None,
        hf_verify_fn: Callable | None = None,
    ) -> None:
        from reliquary.validator.checkpoint import _default_save_hf_format

        self.repo_id = require_checkpoint_repository(repo_id)
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.staging_dir.parent)
        self.tokenizer, self._r2, self._bucket = tokenizer, r2_client, bucket
        self._save = save_fn or _default_save_hf_format
        self._hf_upload = hf_upload_fn or _default_hf_upload
        self._hf_head = hf_head_fn or _default_hf_head
        self._hf_verify = hf_verify_fn or _default_hf_verify
        self._storage_guard = storage_guard or HfStorageGuard()
        self._checkpoint_number_floor = (
            require_checkpoint_number(
                checkpoint_number_floor,
                field="trainer publisher checkpoint number floor",
            )
            if checkpoint_number_floor is not None
            else None
        )
        self._pending = self.staging_dir / PENDING_PUBLICATION

    @contextmanager
    def _lock(self):
        # The persistent inode must never be unlinked while another process waits.
        with (self.staging_dir / ".publication.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PublicationConflict(
                    "another publisher owns the staging directory"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _snapshot(self, transaction: dict) -> Path:
        return self.staging_dir / f"ckpt_{transaction['manifest']['checkpoint_n']}"

    def has_pending(self) -> bool:
        return self._pending.exists()

    def _read_candidate(self) -> tuple[dict | None, str | None]:
        from botocore.exceptions import ClientError

        try:
            response = self._r2.get_object(
                Bucket=self._bucket, Key=CANDIDATE_MANIFEST_KEY
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "NoSuchKey",
                "404",
                "NotFound",
            }:
                return None, None
            raise
        body = response["Body"]
        try:
            raw = body.read(1024 * 1024 + 1)
        finally:
            body.close()
        if len(raw) > 1024 * 1024:
            raise PublicationConflict("candidate manifest exceeds limit")
        manifest = strict_json_loads(raw)
        if not isinstance(manifest, dict):
            raise PublicationConflict("candidate manifest must be an object")
        canonical_checkpoint_identity(
            manifest.get("checkpoint_n"),
            manifest.get("repo_id"),
            manifest.get("revision"),
            field="existing candidate",
        )
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise PublicationConflict(
                "candidate manifest requires an ETag for conditional replacement"
            )
        return manifest, etag

    def _read_pending(self) -> dict | None:
        if not self._pending.exists():
            if any(self.staging_dir.glob("ckpt_*")):
                raise PublicationConflict(
                    "unattributed staging snapshot; explicit recovery required"
                )
            return None
        transaction = strict_json_loads(self._pending.read_bytes())
        if (
            not isinstance(transaction, dict)
            or type(transaction.get("schema_version")) is not int
            or transaction.get("schema_version") != 1
            or transaction.get("state")
            not in {"preparing", "prepared", "uploading", "uploaded", "committed"}
        ):
            raise PublicationConflict("invalid pending publication schema/state")
        from reliquary.shared.training_payload import active_training_identity
        from reliquary.trainer.journal import active_journal_key_space

        manifest = transaction["manifest"]
        if (
            transaction["bucket"] != self._bucket
            or manifest["repo_id"] != self.repo_id
            or any(manifest.get(k) != v for k, v in active_training_identity().items())
            or manifest.get("journal_key_space") != active_journal_key_space()
        ):
            raise PublicationConflict(
                "pending publication belongs to another repository/profile/run"
            )
        require_checkpoint_number(manifest["checkpoint_n"])
        require_checkpoint_number(manifest["trained_window_cursor"])
        require_immutable_checkpoint_revision(transaction["parent_revision"])
        if not re.fullmatch(r"[0-9a-f]{32}", transaction["publication_id"]):
            raise PublicationConflict("invalid pending publication identity")
        revision = transaction["revision"]
        if revision is not None:
            require_immutable_checkpoint_revision(revision)
        if transaction["state"] in {"uploaded", "committed"} and revision is None:
            raise PublicationConflict(
                "uploaded transaction requires immutable revision"
            )
        return transaction

    async def publish(
        self,
        model: Any,
        *,
        checkpoint_n: int,
        lr_schedule_step: int | None,
        trained_window_cursor: int,
        reason: str,
        parent_revision: str,
    ) -> str:
        from reliquary.shared.training_payload import active_training_identity
        from reliquary.trainer.journal import active_journal_key_space
        from reliquary.validator.checkpoint_profile import write_checkpoint_profile

        checkpoint_n = require_checkpoint_number(
            checkpoint_n, field="trainer published checkpoint number"
        )
        trained_window_cursor = require_checkpoint_number(
            trained_window_cursor, field="trainer published window cursor"
        )
        if lr_schedule_step is not None:
            lr_schedule_step = require_checkpoint_number(
                lr_schedule_step, field="trainer published LR schedule step"
            )
        parent_revision = require_immutable_checkpoint_revision(
            parent_revision, field="trainer publication parent"
        )
        if (
            not isinstance(reason, str)
            or not reason
            or reason.strip() != reason
            or len(reason) > 128
            or "\n" in reason
            or "\r" in reason
        ):
            raise ValueError("trainer publication reason must be canonical text")
        if (
            self._checkpoint_number_floor is not None
            and checkpoint_n <= self._checkpoint_number_floor
        ):
            raise ValueError(
                "trainer published checkpoint number must advance monotonically"
            )
        with self._lock():
            if self._read_pending() is not None:
                raise PublicationConflict(
                    "recover pending publication before training or publishing again"
                )
            if await asyncio.to_thread(self._hf_head, self.repo_id) != parent_revision:
                raise PublicationConflict(
                    "HF HEAD unavailable or differs from explicit publication parent"
                )
            previous, etag = await asyncio.to_thread(self._read_candidate)
            if previous is not None and previous["checkpoint_n"] >= checkpoint_n:
                raise PublicationConflict(
                    "candidate checkpoint number does not advance"
                )
            manifest = {
                **active_training_identity(),
                "checkpoint_n": checkpoint_n,
                "repo_id": self.repo_id,
                "trained_window_cursor": trained_window_cursor,
                "reason": reason,
                "journal_key_space": active_journal_key_space(),
            }
            transaction = {
                "schema_version": 1,
                "state": "preparing",
                "bucket": self._bucket,
                "publication_id": uuid.uuid4().hex,
                "manifest": manifest,
                "parent_revision": parent_revision,
                "revision": None,
                "previous_manifest": previous,
                "previous_etag": etag,
                "files": {},
                "prune_revisions": await asyncio.to_thread(
                    self._existing_mirror_revisions
                ),
            }
            write_json(self._pending, transaction)
            snapshot = self._snapshot(transaction)
            snapshot.mkdir()
            # Save failures cannot have remote effects. Recovery of preparing state
            # discards only this incomplete snapshot and replays from the old checkpoint.
            await asyncio.to_thread(self._save, model, self.tokenizer, snapshot)
            extra = {
                "trained_window_cursor": trained_window_cursor,
                "journal_key_space": active_journal_key_space(),
            }
            if lr_schedule_step is not None:
                extra["lr_schedule_step"] = lr_schedule_step
            write_checkpoint_profile(snapshot, extra=extra)
            files = {
                path.name: await asyncio.to_thread(_file_identity, path)
                for path in sorted(snapshot.iterdir())
            }
            if not files:
                raise PublicationConflict("cannot publish an empty snapshot")
            receipt = {
                "publication_id": transaction["publication_id"],
                "parent_revision": parent_revision,
                "manifest": manifest,
                "files": files,
            }
            write_json(snapshot / PUBLICATION_RECEIPT, receipt)
            files[PUBLICATION_RECEIPT] = await asyncio.to_thread(
                _file_identity, snapshot / PUBLICATION_RECEIPT
            )
            _fsync_directory(snapshot)
            transaction.update(state="prepared", files=files)
            write_json(self._pending, transaction)
            return (await self._resume(transaction))["revision"]

    async def recover_pending(self) -> dict | None:
        """Call before resolve_resume_point/model loading; never requires a model.

        Returns the exact committed manifest, or None when no external publication
        exists (including a crash during local snapshot preparation).
        """
        with self._lock():
            transaction = self._read_pending()
            if transaction is None:
                return None
            if transaction["state"] == "preparing":
                self._cleanup(transaction)
                return None
            return await self._resume(transaction)

    async def _resume(self, transaction: dict) -> dict:
        snapshot = self._snapshot(transaction)
        revision = transaction["revision"]
        if transaction["state"] != "committed":
            if (
                not transaction["files"]
                or PUBLICATION_RECEIPT not in transaction["files"]
            ):
                raise PublicationConflict(
                    "pending publication lacks its snapshot receipt"
                )
            for name, expected in transaction["files"].items():
                checkpoint_key(transaction["parent_revision"], name)
                if await asyncio.to_thread(_file_identity, snapshot / name) != expected:
                    raise PublicationConflict(
                        f"local publication content changed: {name}"
                    )
            receipt = strict_json_loads((snapshot / PUBLICATION_RECEIPT).read_bytes())
            if receipt != {
                "publication_id": transaction["publication_id"],
                "parent_revision": transaction["parent_revision"],
                "manifest": transaction["manifest"],
                "files": {
                    name: item
                    for name, item in transaction["files"].items()
                    if name != PUBLICATION_RECEIPT
                },
            }:
                raise PublicationConflict(
                    "pending metadata differs from the immutable snapshot receipt"
                )
        head = require_immutable_checkpoint_revision(
            await asyncio.to_thread(self._hf_head, self.repo_id)
        )
        commit_message = (
            f"checkpoint {transaction['manifest']['checkpoint_n']} "
            f"({transaction['manifest']['reason']}) [{transaction['publication_id']}]"
        )
        if revision is None:
            if head != transaction["parent_revision"]:
                # An upload can succeed before the response or local journal fsync.
                # Identify that exact commit by parent + unique title + every file hash.
                if transaction["state"] != "uploading":
                    raise PublicationConflict(
                        "HF HEAD advanced before this transaction uploaded"
                    )
                await asyncio.to_thread(
                    self._hf_verify,
                    repo_id=self.repo_id,
                    revision=head,
                    parent_revision=transaction["parent_revision"],
                    commit_message=commit_message,
                    files=transaction["files"],
                )
                revision = head
            else:
                await asyncio.to_thread(
                    self._storage_guard.assert_upload_allowed,
                    repo_id=self.repo_id,
                    upload_bytes=sum(
                        item["size"] for item in transaction["files"].values()
                    ),
                )
                transaction["state"] = "uploading"
                write_json(self._pending, transaction)
                revision = require_immutable_checkpoint_revision(
                    await self._hf_upload(
                        folder_path=str(snapshot),
                        repo_id=self.repo_id,
                        commit_message=commit_message,
                        parent_commit=transaction["parent_revision"],
                    ),
                    field="trainer checkpoint publisher revision",
                )
            if revision == transaction["parent_revision"]:
                raise PublicationConflict(
                    "publication did not create a new immutable commit"
                )
            transaction.update(state="uploaded", revision=revision)
            write_json(self._pending, transaction)
        elif head != revision:
            raise PublicationConflict(
                "HF HEAD differs from the pending immutable publication"
            )
        if await asyncio.to_thread(self._hf_head, self.repo_id) != revision:
            raise PublicationConflict("HF HEAD changed before candidate publication")
        manifest = {**transaction["manifest"], "revision": revision}
        current, etag = await asyncio.to_thread(self._read_candidate)
        if current != manifest:
            if (
                transaction["state"] == "committed"
                or etag != transaction["previous_etag"]
                or current != transaction["previous_manifest"]
            ):
                raise PublicationConflict(
                    "candidate manifest advanced or changed during publication"
                )
            config = _multipart_transfer_config()
            for name in sorted(transaction["files"]):
                await asyncio.to_thread(
                    self._r2.upload_file,
                    str(snapshot / name),
                    self._bucket,
                    checkpoint_key(revision, name),
                    Config=config,
                )
            if await asyncio.to_thread(self._hf_head, self.repo_id) != revision:
                raise PublicationConflict("HF HEAD changed during mirror upload")
            condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
            await asyncio.to_thread(
                self._r2.put_object,
                Bucket=self._bucket,
                Key=CANDIDATE_MANIFEST_KEY,
                Body=json.dumps(manifest, sort_keys=True).encode(),
                **condition,
            )
        transaction["state"] = "committed"
        write_json(self._pending, transaction)
        self._checkpoint_number_floor = manifest["checkpoint_n"]
        previous = transaction["previous_manifest"]
        await asyncio.to_thread(
            self._prune_mirror,
            revisions=transaction["prune_revisions"],
            keep={
                revision,
                previous["revision"] if previous else transaction["parent_revision"],
            },
        )
        self._cleanup(transaction)
        logger.info(
            "Published/recovered checkpoint %d to %s@%s",
            manifest["checkpoint_n"],
            self.repo_id,
            revision,
        )
        return manifest

    def _existing_mirror_revisions(self) -> list[str]:
        # Bound deletion candidates BEFORE this publication. A later publisher's
        # new mirror can never be deleted by an old or recovering process.
        try:
            response = self._r2.list_objects_v2(
                Bucket=self._bucket, Prefix=f"{R2_CHECKPOINT_PREFIX}/", Delimiter="/"
            )
            return [
                entry["Prefix"].removeprefix(f"{R2_CHECKPOINT_PREFIX}/").rstrip("/")
                for entry in response.get("CommonPrefixes", [])
                if re.fullmatch(
                    f"{R2_CHECKPOINT_PREFIX}/[0-9a-f]{{40}}/", entry["Prefix"]
                )
            ]
        except Exception:
            logger.exception("mirror retention listing failed (non-fatal)")
            return []

    def _prune_mirror(self, *, revisions: list[str], keep: set[str]) -> None:
        try:
            for revision in revisions:
                require_immutable_checkpoint_revision(revision)
                if revision in keep:
                    continue
                prefix = f"{R2_CHECKPOINT_PREFIX}/{revision}/"
                response = self._r2.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
                for item in response.get("Contents", []):
                    if item["Key"].startswith(prefix):
                        self._r2.delete_object(Bucket=self._bucket, Key=item["Key"])
        except Exception:
            logger.exception("mirror prune failed (non-fatal)")

    def _cleanup(self, transaction: dict) -> None:
        snapshot = self._snapshot(transaction)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        self._pending.unlink()
        _fsync_directory(self.staging_dir)
