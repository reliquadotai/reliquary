"""Persistent retry queue for detached-trainer payload uploads.

Same design as ``archive_queue.ArchiveQueue`` (atomic .tmp+rename enqueue,
background drain with per-file exponential backoff, restart rescan), but
files are opaque bytes and land under the ``reliquary/training/`` R2
prefix — deliberately disjoint from ``reliquary/dataset/`` which the
dashboard consumes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from reliquary.constants import (
    FILL_CLOSED_EMISSIONS_PER_WINDOW,
    FILL_CLOSED_ENABLED,
)
from reliquary.shared.strict_json import strict_json_loads

logger = logging.getLogger(__name__)

R2_TRAINING_PREFIX = "reliquary/training"

# Same backoff table as ArchiveQueue — tuned on observed R2 outages.
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (5, 30, 120, 600, 1800)

_PAYLOAD_SUFFIX = ".npz"
_TOMBSTONE_SUFFIX = ".tombstone.json"
_EPOCH_MARKER_SUFFIX = ".epoch.json"
_STEP_CURSOR_FILENAME = "step-cursor.json"
_STEP_CURSOR_SCHEMA_VERSION = 1

# R40 #1c: fetch_step_cursor's cache TTL. Matches the order of magnitude
# of service.FILL_CLOSED_ROTATION_POLL_SECONDS (2.0) by value, not by
# import -- importing from reliquary.validator.service here would be
# circular (service.py imports this module).
_STEP_CURSOR_CACHE_TTL_SECONDS = 2.0


def payload_key(window_start: int) -> str:
    return f"{R2_TRAINING_PREFIX}/window-{int(window_start)}{_PAYLOAD_SUFFIX}"


def tombstone_key(window_start: int) -> str:
    return f"{R2_TRAINING_PREFIX}/window-{int(window_start)}{_TOMBSTONE_SUFFIX}"


def encoded_window_journal_key(window_start: int, batch_index: int = 0) -> int:
    """The journal key one training-payload slot is filed under.

    v4/v5 (``FILL_CLOSED_ENABLED`` off): one payload per window;
    unchanged, byte-for-byte -- the key IS the window number, and
    ``batch_index`` is ignored.

    v6 (R11): a still-open window can emit up to
    ``FILL_CLOSED_EMISSIONS_PER_WINDOW`` training payloads (one per
    B_BATCH-per-environment cycle) rather than exactly one at seal.
    Reusing ``enqueue_payload``'s one-slot-per-window key as-is would let
    each later emission in the same window silently overwrite the
    previous one, so this encodes ``window_start *
    FILL_CLOSED_EMISSIONS_PER_WINDOW + batch_index`` instead.
    ``WindowJournal.next_entry``'s cursor already just advances by
    ``stride`` (1) and fetches ``payload_key(cursor + stride)`` one
    integer at a time -- it needs no change at all: this encoding makes
    exactly ``FILL_CLOSED_EMISSIONS_PER_WINDOW`` consecutive integers
    cover one window before rolling into the next window's range, so the
    cursor already walks it correctly and ``publish_every`` (measured in
    emitted training-payload entries) holds unchanged.
    """
    if not FILL_CLOSED_ENABLED:
        return int(window_start)
    if not 0 <= batch_index < FILL_CLOSED_EMISSIONS_PER_WINDOW:
        raise ValueError(
            "batch_index must be in "
            f"[0, {FILL_CLOSED_EMISSIONS_PER_WINDOW}), got {batch_index}"
        )
    return int(window_start) * FILL_CLOSED_EMISSIONS_PER_WINDOW + int(
        batch_index
    )


def step_cursor_key() -> str:
    """R2 key for the trainer's single per-step consumption cursor (v6.1).

    One object for the whole trainer, overwritten every step -- there is
    no window/batch_index in this key, unlike ``payload_key``/
    ``tombstone_key``.
    """
    return f"{R2_TRAINING_PREFIX}/{_STEP_CURSOR_FILENAME}"


def epoch_marker_key(epoch_id: str) -> str:
    if (
        not isinstance(epoch_id, str)
        or len(epoch_id) != 64
        or any(character not in "0123456789abcdef" for character in epoch_id)
    ):
        raise ValueError("epoch_id must be lowercase SHA-256")
    return f"{R2_TRAINING_PREFIX}/epoch-{epoch_id}{_EPOCH_MARKER_SUFFIX}"


def _default_queue_dir() -> str:
    explicit = os.environ.get("RELIQUARY_TRAINING_PAYLOAD_QUEUE_DIR")
    if explicit:
        return explicit
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return os.path.join(state_dir, "pending_training_payloads")


def _default_upload(key: str, body: bytes) -> None:
    """Sync boto3 PUT via storage's proven helper. Runs inside to_thread."""
    from reliquary.infrastructure.storage import _sync_boto3_put

    account_id = os.getenv("R2_ACCOUNT_ID", "")
    endpoint = (
        os.getenv("R2_ENDPOINT_URL")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    _sync_boto3_put(
        os.getenv("R2_BUCKET_ID", "reliquary"), key, body,
        account_id,
        os.getenv("R2_ACCESS_KEY_ID", ""),
        os.getenv("R2_SECRET_ACCESS_KEY", ""),
        endpoint,
        os.getenv("R2_REGION", "us-east-1"),
    )


def _r2_client(config):
    """Build a boto3 S3 client from the standard R2 env vars.

    Shared by every synchronous R2 call this module makes DIRECTLY
    (``_default_delete``, ``_default_fetch_step_cursor``) so each call
    site supplies only the ``botocore.config.Config`` (timeouts, retries)
    it needs instead of re-deriving the account/endpoint/credential
    resolution a second (or third) time. The upload PUT already has its
    own proven helper for the equivalent (``storage._sync_boto3_put``);
    this is that helper's counterpart for the smaller synchronous calls
    that live in this module.
    """
    import boto3

    account_id = os.getenv("R2_ACCOUNT_ID", "")
    endpoint = (
        os.getenv("R2_ENDPOINT_URL")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("R2_REGION", "us-east-1"),
        config=config,
    )


def _default_delete(key: str) -> None:
    """Best-effort sync boto3 DeleteObject (tombstone superseded by a
    replayed payload). Runs inside to_thread."""
    from botocore.config import Config

    client = _r2_client(Config(retries={"max_attempts": 2, "mode": "standard"}))
    client.delete_object(
        Bucket=os.getenv("R2_BUCKET_ID", "reliquary"), Key=key,
    )


# R38: deliberately tighter than the upload/delete config above. Those run
# off the hot path inside the background drain loop and can afford 15s
# connect / 30s read with up to 3 retries. This one backs the trainer
# step-cursor GET -- a hung store must fail fast, not stall a poller, so:
# short timeouts, no retry.
_STEP_CURSOR_FETCH_CONNECT_TIMEOUT_SECONDS = 2
_STEP_CURSOR_FETCH_READ_TIMEOUT_SECONDS = 2

# R40 #1a: a fresh boto3 client costs ~50-200ms to construct on its own,
# on top of the network round trip -- paid on every single poll tick
# before this. One client, built once on first use, reused forever.
# boto3 clients are safe for concurrent use (the SDK's own documented
# pattern); the lock only guards the one-time construction race.
_STEP_CURSOR_CLIENT: Any = None
_STEP_CURSOR_CLIENT_LOCK = threading.Lock()


def _cached_step_cursor_client() -> Any:
    global _STEP_CURSOR_CLIENT
    if _STEP_CURSOR_CLIENT is None:
        with _STEP_CURSOR_CLIENT_LOCK:
            if _STEP_CURSOR_CLIENT is None:
                from botocore.config import Config

                _STEP_CURSOR_CLIENT = _r2_client(
                    Config(
                        connect_timeout=_STEP_CURSOR_FETCH_CONNECT_TIMEOUT_SECONDS,
                        read_timeout=_STEP_CURSOR_FETCH_READ_TIMEOUT_SECONDS,
                        retries={"max_attempts": 1, "mode": "standard"},
                    )
                )
    return _STEP_CURSOR_CLIENT


def _default_fetch_step_cursor() -> bytes | None:
    """Best-effort sync boto3 GetObject for the trainer's step-cursor
    object (the R2 counterpart of ``write_step_cursor``'s local write,
    read by a DIFFERENT process than the one that wrote it -- e.g. the
    validator reading what the detached trainer's drain uploaded).

    Returns ``None`` on ANY failure (missing key, network error, timeout)
    -- never raises. Uses the memoised client (``_cached_step_cursor_client``)
    with a config deliberately tighter than the upload/delete path's (see
    the module-level timeout constants above).
    """
    client = _cached_step_cursor_client()
    try:
        response = client.get_object(
            Bucket=os.getenv("R2_BUCKET_ID", "reliquary"),
            Key=step_cursor_key(),
        )
        return response["Body"].read()
    except Exception:
        return None


def _parse_step_cursor(raw: bytes) -> int | None:
    """Shared by ``read_step_cursor`` and ``fetch_step_cursor``: a
    corrupt/torn/wrong-schema body reads as ``None``, never raises."""
    try:
        data = strict_json_loads(raw)
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "journal_key",
            "written_at",
        }:
            return None
        schema_version = data["schema_version"]
        journal_key = data["journal_key"]
        written_at = data["written_at"]
        if (
            type(schema_version) is not int
            or schema_version != _STEP_CURSOR_SCHEMA_VERSION
        ):
            return None
        if type(journal_key) is not int or journal_key < 0:
            return None
        if (
            isinstance(written_at, bool)
            or not isinstance(written_at, (int, float))
            or not math.isfinite(written_at)
            or written_at < 0
        ):
            return None
        return journal_key
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        return None


class TrainingPayloadQueue:
    """Durable producer/consumer queue for training payloads + tombstones."""

    def __init__(self, queue_dir: str | None = None) -> None:
        self.queue_dir = Path(queue_dir or _default_queue_dir())
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._attempts: dict[str, int] = {}
        self._next_attempt_at: dict[str, float] = {}
        self._uploads_succeeded_total = 0
        self._upload_failures_total = 0
        self._last_upload_success_ts: float | None = None
        self._last_upload_failure_ts: float | None = None
        # R40 #1: fetch_step_cursor's fire-and-collect cache. 0.0 reads as
        # "infinitely stale" against time.monotonic() (always positive),
        # so the first call always kicks a fetch.
        self._step_cursor_cache_value: int | None = None
        self._step_cursor_cache_at: float = 0.0
        self._step_cursor_cache_lock = threading.Lock()
        self._step_cursor_fetch_in_flight = False
        self._step_cursor_fetch_thread: threading.Thread | None = None
        # R40 #4: last body this instance's drain actually uploaded, so an
        # unchanged cursor (idle trainer between real steps) is not
        # re-PUT every drain cycle forever.
        self._step_cursor_last_uploaded_body: bytes | None = None
        # Experimental fill-closed entries are create-only journal commits,
        # not replaceable cache values.  Serialise their artifact + receipt
        # pair so two proof-worker callbacks cannot race the same key through
        # the shared ``.tmp`` filename.
        self._journal_commit_lock = threading.Lock()
        self._journal_commit_dir = self.queue_dir / "journal_commits"
        if FILL_CLOSED_ENABLED:
            self._journal_commit_dir.mkdir(parents=True, exist_ok=True)
            self._recover_committed_journal_entries()

    # ---------------- producer ----------------

    def _enqueue(self, filename: str, data: bytes) -> Path:
        final_path = self.queue_dir / filename
        tmp_path = self.queue_dir / (filename + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, final_path)
        return final_path

    def _enqueue_durable(self, filename: str, data: bytes) -> Path:
        """Atomic local commit with file and parent-directory durability."""
        final_path = self.queue_dir / filename
        tmp_path = self.queue_dir / (filename + ".tmp")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        directory_fd = os.open(final_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return final_path

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _journal_staging_path(self, slot: int, kind: str) -> Path:
        return self._journal_commit_dir / f"window-{slot}.{kind}.body"

    def _publish_staged_journal_entry(
        self, staging_path: Path, final_path: Path,
    ) -> None:
        """Make a receipt-backed artifact visible to the upload scanner."""
        os.replace(staging_path, final_path)
        # The rename crosses directories.  Persist both the visible name and
        # removal of the hidden staging name before reporting success.
        self._fsync_directory(self.queue_dir)
        self._fsync_directory(self._journal_commit_dir)

    @staticmethod
    def _validate_journal_receipt(
        receipt: Any, *, source: str,
    ) -> dict[str, Any]:
        fields = {"schema_version", "journal_key", "kind", "sha256", "size"}
        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise RuntimeError(f"journal commit receipt {source} has invalid fields")
        schema_version = receipt.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise RuntimeError(f"journal commit receipt {source} has invalid schema")
        slot = receipt.get("journal_key")
        size = receipt.get("size")
        kind = receipt.get("kind")
        digest = receipt.get("sha256")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise RuntimeError(f"journal commit receipt {source} has invalid key")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeError(f"journal commit receipt {source} has invalid size")
        if kind not in {"payload", "tombstone"}:
            raise RuntimeError(f"journal commit receipt {source} has invalid kind")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"journal commit receipt {source} has invalid digest")
        return receipt

    @staticmethod
    def _artifact_matches_receipt(path: Path, receipt: dict[str, Any]) -> bool:
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"journal artifact {path} is unreadable") from exc
        return (
            len(body) == receipt["size"]
            and hashlib.sha256(body).hexdigest() == receipt["sha256"]
        )

    def _recover_committed_journal_entries(self) -> None:
        """Finish receipt-backed commits interrupted before visible rename.

        The uploader only scans the queue root.  Staging bodies under the
        receipt directory therefore cannot disappear before their receipt is
        durable.  On restart, a valid receipt plus hidden body is enough to
        finish the rename without reconstructing any in-memory batch state.
        """
        with self._journal_commit_lock:
            for receipt_path in sorted(self._journal_commit_dir.glob("window-*.json")):
                try:
                    receipt = strict_json_loads(receipt_path.read_bytes())
                except (OSError, ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"journal commit receipt {receipt_path.name} is unreadable"
                    ) from exc
                receipt = self._validate_journal_receipt(
                    receipt, source=receipt_path.name,
                )
                slot = receipt["journal_key"]
                if receipt_path.name != f"window-{slot}.json":
                    raise RuntimeError(
                        f"journal commit receipt {receipt_path.name} names a different key"
                    )
                kind = receipt["kind"]
                suffix = (
                    _TOMBSTONE_SUFFIX if kind == "tombstone" else _PAYLOAD_SUFFIX
                )
                other_suffix = (
                    _PAYLOAD_SUFFIX if kind == "tombstone" else _TOMBSTONE_SUFFIX
                )
                other_kind = "payload" if kind == "tombstone" else "tombstone"
                final_path = self.queue_dir / f"window-{slot}{suffix}"
                other_path = self.queue_dir / f"window-{slot}{other_suffix}"
                staging_path = self._journal_staging_path(slot, kind)
                other_staging_path = self._journal_staging_path(slot, other_kind)
                if other_path.exists() or other_staging_path.exists():
                    raise RuntimeError(
                        f"journal key {slot} also has a conflicting {other_kind} artifact"
                    )
                if final_path.exists():
                    if not self._artifact_matches_receipt(final_path, receipt):
                        raise RuntimeError(
                            f"journal key {slot} artifact differs from its receipt"
                        )
                    if staging_path.exists():
                        if not self._artifact_matches_receipt(staging_path, receipt):
                            raise RuntimeError(
                                f"journal key {slot} staging body differs from its receipt"
                            )
                        staging_path.unlink()
                        self._fsync_directory(self._journal_commit_dir)
                elif staging_path.exists():
                    if not self._artifact_matches_receipt(staging_path, receipt):
                        raise RuntimeError(
                            f"journal key {slot} staging body differs from its receipt"
                        )
                    self._publish_staged_journal_entry(staging_path, final_path)
            orphaned = sorted(self._journal_commit_dir.glob("window-*.body"))
            if orphaned:
                # The body reached disk but its commit receipt did not.  Only
                # the original producer can retry it byte-identically and
                # restore reward/index state.  A fresh process has no safe
                # basis to manufacture that in-memory transaction, so startup
                # remains closed for explicit abort/quarantine recovery.
                raise RuntimeError(
                    "unreceipted fill-closed journal staging body requires "
                    f"recovery: {orphaned[0].name}"
                )

    def enqueue_payload(self, window_start: int, data: bytes) -> Path:
        # A window replayed under the same number after a fatal restart
        # supersedes its tombstone (journal readers prefer the payload;
        # this removes a not-yet-uploaded local marker too).
        stale = self.queue_dir / (
            f"window-{int(window_start)}{_TOMBSTONE_SUFFIX}"
        )
        try:
            stale.unlink()
        except OSError:
            pass
        path = self._enqueue(
            f"window-{int(window_start)}{_PAYLOAD_SUFFIX}", data,
        )
        logger.info(
            "TrainingPayloadQueue: enqueued window %d (%d bytes)",
            window_start, len(data),
        )
        return path

    def enqueue_tombstone(self, window_start: int, data: bytes) -> Path:
        path = self._enqueue(
            f"window-{int(window_start)}{_TOMBSTONE_SUFFIX}", data,
        )
        logger.info(
            "TrainingPayloadQueue: enqueued tombstone for window %d",
            window_start,
        )
        return path

    def _enqueue_committed_journal_entry(
        self,
        window_start: int,
        data: bytes,
        *,
        is_tombstone: bool,
    ) -> Path:
        """Create one immutable fill-closed journal slot.

        The ordinary payload methods intentionally retain their historical
        replacement semantics for legacy windows.  Progressive fill-closed
        emission needs a stronger contract: a journal key is committed once,
        byte-identically retryable, and never allowed to change kind or body.

        The opaque artifact is first written under a hidden staging name, its
        small digest receipt is committed second, and only then is the artifact
        renamed into the upload queue.  The receipt is retained after upload,
        so retries remain idempotent.  Restart finishes a receipt-backed hidden
        rename; an unreceipted body can only be claimed byte-identically.
        """
        if type(window_start) is not int or window_start < 0:
            raise ValueError("journal key must be a non-negative integer")
        slot = window_start
        body = bytes(data)
        kind = "tombstone" if is_tombstone else "payload"
        digest = hashlib.sha256(body).hexdigest()
        suffix = _TOMBSTONE_SUFFIX if is_tombstone else _PAYLOAD_SUFFIX
        other_suffix = _PAYLOAD_SUFFIX if is_tombstone else _TOMBSTONE_SUFFIX
        final_path = self.queue_dir / f"window-{slot}{suffix}"
        other_path = self.queue_dir / f"window-{slot}{other_suffix}"
        receipt_path = self._journal_commit_dir / f"window-{slot}.json"
        staging_path = self._journal_staging_path(slot, kind)
        other_kind = "payload" if is_tombstone else "tombstone"
        other_staging_path = self._journal_staging_path(slot, other_kind)
        receipt = {
            "schema_version": 1,
            "journal_key": slot,
            "kind": kind,
            "sha256": digest,
            "size": len(body),
        }

        with self._journal_commit_lock:
            try:
                existing_receipt = strict_json_loads(receipt_path.read_bytes())
            except FileNotFoundError:
                existing_receipt = None
            except (OSError, ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"journal commit receipt for key {slot} is unreadable"
                ) from exc
            if existing_receipt is not None:
                existing_receipt = self._validate_journal_receipt(
                    existing_receipt, source=receipt_path.name,
                )
                expected = {
                    "schema_version": 1,
                    "journal_key": slot,
                    "kind": kind,
                    "sha256": digest,
                    "size": len(body),
                }
                if existing_receipt != expected:
                    raise RuntimeError(
                        f"journal key {slot} already has a different commit"
                    )
                if other_path.exists() or other_staging_path.exists():
                    raise RuntimeError(
                        f"journal key {slot} also has a conflicting {other_kind} artifact"
                    )
                if final_path.exists() and not self._artifact_matches_receipt(
                    final_path, expected,
                ):
                    raise RuntimeError(
                        f"journal key {slot} artifact differs from its receipt"
                    )
                if staging_path.exists():
                    if not self._artifact_matches_receipt(staging_path, expected):
                        raise RuntimeError(
                            f"journal key {slot} staging body differs from its receipt"
                        )
                    if final_path.exists():
                        staging_path.unlink()
                        self._fsync_directory(self._journal_commit_dir)
                    else:
                        self._publish_staged_journal_entry(
                            staging_path, final_path,
                        )
                return final_path

            if other_path.exists() or other_staging_path.exists():
                raise RuntimeError(
                    f"journal key {slot} already has a {other_kind} artifact"
                )
            if final_path.exists():
                if final_path.read_bytes() != body:
                    raise RuntimeError(
                        f"journal key {slot} already has different bytes"
                    )
            elif staging_path.exists():
                if staging_path.read_bytes() != body:
                    raise RuntimeError(
                        f"journal key {slot} already has different staged bytes"
                    )
            else:
                self._enqueue_durable(
                    str(Path("journal_commits") / staging_path.name), body,
                )
            self._enqueue_durable(
                str(Path("journal_commits") / receipt_path.name),
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
            # A legacy/pre-staging artifact may have been uploaded while the
            # receipt was written.  Otherwise publish the hidden body now.
            if not final_path.exists() and staging_path.exists():
                self._publish_staged_journal_entry(staging_path, final_path)
            logger.info(
                "TrainingPayloadQueue: committed %s for journal key %d (%s)",
                kind,
                slot,
                digest[:12],
            )
            return final_path

    def enqueue_committed_payload(self, window_start: int, data: bytes) -> Path:
        """Durably create or byte-identically replay one payload slot."""
        return self._enqueue_committed_journal_entry(
            window_start, data, is_tombstone=False,
        )

    def enqueue_committed_tombstone(
        self, window_start: int, data: bytes,
    ) -> Path:
        """Durably create or byte-identically replay one tombstone slot."""
        return self._enqueue_committed_journal_entry(
            window_start, data, is_tombstone=True,
        )

    def enqueue_epoch_marker(self, epoch_id: str, data: bytes) -> Path:
        from reliquary.shared.training_payload import (
            decode_checkpoint_epoch_marker,
        )

        marker = decode_checkpoint_epoch_marker(data)
        if marker["checkpoint_epoch"].epoch_id != epoch_id:
            raise ValueError("checkpoint epoch marker identifier differs")
        epoch_marker_key(epoch_id)
        path = self._enqueue(f"epoch-{epoch_id}{_EPOCH_MARKER_SUFFIX}", data)
        logger.info(
            "TrainingPayloadQueue: enqueued terminal marker for epoch %s",
            epoch_id[:12],
        )
        return path

    # ---------------- v6.1: trainer step cursor ----------------

    def write_step_cursor(self, journal_key: int) -> None:
        """Overwrite the single per-step consumption cursor object.

        Called by the trainer after it consumes one journal entry (trained
        payload or tombstone) so the validator can pace picks on real
        consumption (Amendment v6.1) instead of a declared interval. This
        is a plain, synchronous, atomic local write -- the same
        temp-file+rename discipline as ``_enqueue`` -- never a growing
        log: each call replaces the previous object in place. Delivery to
        R2 rides the existing drain/upload transport (see ``_pending``),
        unlike a payload/tombstone the local file is NOT deleted after a
        successful upload, since it must stay in place to be overwritten
        by the next step.
        """
        if type(journal_key) is not int or journal_key < 0:
            raise ValueError(
                "trainer journal cursor must be a non-negative integer"
            )
        body = json.dumps(
            {
                "schema_version": _STEP_CURSOR_SCHEMA_VERSION,
                "journal_key": journal_key,
                "written_at": time.time(),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._enqueue(_STEP_CURSOR_FILENAME, body)

    def read_step_cursor(self) -> int | None:
        """Read back the local step-cursor object, or ``None`` if it is
        absent, unreadable, or unparseable.

        A torn read (process crashed mid-write, or the reader raced a
        writer -- the ``os.replace`` rename is atomic but a caller could
        still catch the file between an unrelated partial write from a
        future format) must look exactly like "no cursor yet": this is
        advisory pacing telemetry read from a validator poll loop, and
        raising here would take that loop down over a stale timestamp.
        """
        path = self.queue_dir / _STEP_CURSOR_FILENAME
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        return _parse_step_cursor(raw)

    def fetch_step_cursor(
        self, fetch_fn: Callable[[], bytes | None] | None = None,
    ) -> int | None:
        """The last COMPLETED remote read of the trainer's step-cursor
        object -- NEVER blocks on network I/O.

        ``read_step_cursor`` reads the LOCAL file THIS process's own
        drain wrote -- of no use to a DIFFERENT process reading what
        ANOTHER process's drain already uploaded (R38: the validator
        reading what the detached trainer published; the two run on
        different hosts and never share a local queue_dir).

        R40 #1: this call site sits on the validator's miner-serving
        event loop (``_wait_for_window_seal``'s 0.5s poll tick, and the
        rotation-wait's own poll), so a synchronous GET here -- even a
        short-timeout one -- would still stall that loop in multi-second
        chunks under an R2 outage (picks 3..16 of every window). Instead
        this is fire-and-collect, cached: it returns the LAST value a
        background fetch completed with, immediately, and kicks a NEW
        background fetch only when that value is stale (>=
        ``_STEP_CURSOR_CACHE_TTL_SECONDS``, matching the order of
        magnitude of ``service.FILL_CLOSED_ROTATION_POLL_SECONDS``) AND
        no fetch is already in flight -- so at most one GET runs at a
        time, and at most one per TTL window regardless of how often the
        caller polls. The throttle lives HERE, not with the caller
        (superseding the previous "makes no attempt to cache" contract).

        A cursor up to one TTL window stale is harmless: the pick gate's
        comparison is ``>=``, and a stale-but-still-valid cursor merely
        holds a pick back by up to that same window, never opens one
        early. On a fetch FAILURE (network error, timeout, missing key,
        bad schema) the cache is left at whatever it already held --
        never regressed to ``None`` -- so a transient R2 hiccup degrades
        to "slightly stale" rather than "picks stop", which the fetch
        would otherwise cause every time it was consulted during an
        outage. The very first call, before anything has ever completed,
        returns ``None`` (which the gate already treats as "not yet").
        """
        fn = fetch_fn or _default_fetch_step_cursor
        now = time.monotonic()
        with self._step_cursor_cache_lock:
            stale = (
                now - self._step_cursor_cache_at
                >= _STEP_CURSOR_CACHE_TTL_SECONDS
            )
            kick = stale and not self._step_cursor_fetch_in_flight
            if kick:
                self._step_cursor_fetch_in_flight = True
            value = self._step_cursor_cache_value
        if kick:
            thread = threading.Thread(
                target=self._refresh_step_cursor_cache,
                args=(fn,),
                daemon=True,
                name="step-cursor-fetch",
            )
            self._step_cursor_fetch_thread = thread
            thread.start()
        return value

    def _refresh_step_cursor_cache(
        self, fn: Callable[[], bytes | None],
    ) -> None:
        """Background-thread body kicked by ``fetch_step_cursor``."""
        # try/finally, not just the fetch's try/except: if ANYTHING in this
        # body ever raises past it, the in-flight flag must still drop, or
        # the cache freezes forever with no crash to point at it.
        try:
            try:
                raw = fn()
            except Exception:
                raw = None
            parsed = None if raw is None else _parse_step_cursor(raw)
            with self._step_cursor_cache_lock:
                if parsed is not None:
                    self._step_cursor_cache_value = parsed
                # Reset the TTL clock on EVERY completion, success or not --
                # this is what turns a repeated failure into a natural ~TTL
                # retry cadence instead of hammering every poll tick.
                self._step_cursor_cache_at = time.monotonic()
        finally:
            with self._step_cursor_cache_lock:
                self._step_cursor_fetch_in_flight = False

    # ---------------- consumer ----------------

    def _pending(self) -> list[Path]:
        return sorted(
            (
                path
                for pattern in ("window-*", "epoch-*", _STEP_CURSOR_FILENAME)
                for path in self.queue_dir.glob(pattern)
                if not path.name.endswith(".tmp")
            ),
            key=lambda path: (path.name.startswith("epoch-"), path.name),
        )

    @staticmethod
    def _key_for(path: Path) -> str | None:
        name = path.name
        if name == _STEP_CURSOR_FILENAME:
            return step_cursor_key()
        if name.startswith("window-") and (
            name.endswith(_TOMBSTONE_SUFFIX) or name.endswith(_PAYLOAD_SUFFIX)
        ):
            return f"{R2_TRAINING_PREFIX}/{name}"
        if name.startswith("epoch-") and name.endswith(_EPOCH_MARKER_SUFFIX):
            return f"{R2_TRAINING_PREFIX}/{name}"
        return None

    def _backoff_delay(self, attempts: int) -> float:
        if attempts <= 0:
            return 0.0
        idx = min(attempts, len(RETRY_BACKOFF_SECONDS)) - 1
        return float(RETRY_BACKOFF_SECONDS[idx])

    def snapshot(self) -> dict:
        pending = self._pending()
        return {
            "depth": len(pending),
            "uploads_succeeded_total": self._uploads_succeeded_total,
            "upload_failures_total": self._upload_failures_total,
            "last_upload_success_ts": self._last_upload_success_ts,
            "last_upload_failure_ts": self._last_upload_failure_ts,
        }

    async def _try_upload(
        self,
        path: Path,
        upload_fn: Callable[[str, bytes], None],
        delete_fn: Callable[[str], None] | None = None,
    ) -> bool:
        key = self._key_for(path)
        if key is None:
            logger.error(
                "TrainingPayloadQueue: dropping malformed file: %s", path,
            )
            try:
                path.unlink()
            except OSError:
                pass
            return False
        try:
            body = path.read_bytes()
        except OSError as e:
            logger.error("TrainingPayloadQueue: failed to read %s: %s", path, e)
            return False
        if (
            path.name == _STEP_CURSOR_FILENAME
            and body == self._step_cursor_last_uploaded_body
        ):
            # R40 #4: unlike a payload/tombstone, the cursor is never
            # deleted after upload (it must stay local, overwritten in
            # place) -- so it re-enters ``_pending()`` on EVERY drain
            # cycle regardless of whether the trainer took a new step
            # since the last one. Without this check an idle trainer PUTs
            # an identical object every ~2s forever, on every profile
            # including v4/v5 where nothing even reads it: ~43k Class A
            # ops/day/trainer, and this repo has hit a real R2 Class A
            # billing incident from exactly this shape of waste before.
            # A genuinely new step still uploads promptly (different
            # body -- journal_key or written_at differs).
            return True
        if path.name.startswith("epoch-"):
            from reliquary.shared.training_payload import (
                decode_checkpoint_epoch_marker,
            )

            marker = decode_checkpoint_epoch_marker(body)
            binding = marker["checkpoint_epoch"]
            pending_names = {item.name for item in self.queue_dir.glob("window-*")}
            for window_start in range(
                binding.first_window,
                binding.first_window + binding.window_count,
            ):
                if (
                    f"window-{window_start}{_PAYLOAD_SUFFIX}" in pending_names
                    or f"window-{window_start}{_TOMBSTONE_SUFFIX}" in pending_names
                ):
                    # The marker is the commit point consumed by detached
                    # trainers and cannot overtake any lane artifact.
                    return False
        try:
            await asyncio.to_thread(upload_fn, key, body)
        except Exception as e:
            self._upload_failures_total += 1
            self._last_upload_failure_ts = time.time()
            attempts = self._attempts.get(str(path), 0) + 1
            self._attempts[str(path)] = attempts
            delay = self._backoff_delay(attempts)
            self._next_attempt_at[str(path)] = (
                asyncio.get_running_loop().time() + delay
            )
            logger.warning(
                "TrainingPayloadQueue: upload failed for %s (attempt %d): "
                "%s. Backing off %.0fs.", key, attempts, e, delay,
            )
            return False
        self._attempts.pop(str(path), None)
        self._next_attempt_at.pop(str(path), None)
        self._uploads_succeeded_total += 1
        self._last_upload_success_ts = time.time()
        if delete_fn is not None and key.endswith(_PAYLOAD_SUFFIX):
            # Best-effort: remove a previously uploaded tombstone for the
            # same window so a late journal read prefers the payload.
            sibling = key[: -len(_PAYLOAD_SUFFIX)] + _TOMBSTONE_SUFFIX
            try:
                await asyncio.to_thread(delete_fn, sibling)
            except Exception:
                logger.debug(
                    "tombstone cleanup failed for %s (non-fatal)", sibling,
                )
        if path.name == _STEP_CURSOR_FILENAME:
            # Not a consumed-once queue entry: a single overwritten object
            # that must stay local so the next step's write can replace it
            # in place. Remember what was just uploaded (R40 #4) so the
            # NEXT drain cycle can skip re-uploading if nothing changed.
            self._step_cursor_last_uploaded_body = body
            return True
        try:
            path.unlink()
        except OSError as e:
            logger.warning(
                "TrainingPayloadQueue: uploaded %s but delete failed: %s",
                key, e,
            )
        return True

    async def drain_once(
        self,
        upload_fn: Callable[[str, bytes], None] | None = None,
        delete_fn: Callable[[str], None] | None = None,
    ) -> int:
        """Attempt every pending file once (honoring backoff). Returns the
        number of successful uploads."""
        fn = upload_fn or _default_upload
        if delete_fn is None and upload_fn is None:
            delete_fn = _default_delete
        now = asyncio.get_running_loop().time()
        uploaded = 0
        for path in self._pending():
            next_at = self._next_attempt_at.get(str(path), 0.0)
            if next_at > now:
                continue
            if await self._try_upload(path, fn, delete_fn):
                uploaded += 1
        return uploaded

    async def run_forever(
        self,
        upload_fn: Callable[[str, bytes], None] | None = None,
        delete_fn: Callable[[str], None] | None = None,
    ) -> None:
        logger.info(
            "TrainingPayloadQueue worker starting. queue_dir=%s",
            self.queue_dir,
        )
        while True:
            try:
                await self.drain_once(upload_fn=upload_fn, delete_fn=delete_fn)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TrainingPayloadQueue drain crashed")
            await asyncio.sleep(2.0)


_QUEUE: TrainingPayloadQueue | None = None


def get_training_payload_queue() -> TrainingPayloadQueue:
    """Return the process-wide TrainingPayloadQueue (created lazily)."""
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = TrainingPayloadQueue()
    return _QUEUE
