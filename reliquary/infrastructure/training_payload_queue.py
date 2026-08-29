"""Persistent retry queue for detached-trainer payload uploads.

Same design as ``archive_queue.ArchiveQueue`` (atomic .tmp+rename enqueue,
background drain with per-file exponential backoff, restart rescan), but
files are opaque bytes and land under the ``reliquary/training/`` R2
prefix — deliberately disjoint from ``reliquary/dataset/`` which the
dashboard consumes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Callable

from reliquary.constants import (
    FILL_CLOSED_EMISSIONS_PER_WINDOW,
    FILL_CLOSED_ENABLED,
)

logger = logging.getLogger(__name__)

R2_TRAINING_PREFIX = "reliquary/training"

# Same backoff table as ArchiveQueue — tuned on observed R2 outages.
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (5, 30, 120, 600, 1800)

_PAYLOAD_SUFFIX = ".npz"
_TOMBSTONE_SUFFIX = ".tombstone.json"
_EPOCH_MARKER_SUFFIX = ".epoch.json"


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


def _default_delete(key: str) -> None:
    """Best-effort sync boto3 DeleteObject (tombstone superseded by a
    replayed payload). Runs inside to_thread."""
    import boto3
    from botocore.config import Config

    account_id = os.getenv("R2_ACCOUNT_ID", "")
    endpoint = (
        os.getenv("R2_ENDPOINT_URL")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("R2_REGION", "us-east-1"),
        config=Config(retries={"max_attempts": 2, "mode": "standard"}),
    )
    client.delete_object(
        Bucket=os.getenv("R2_BUCKET_ID", "reliquary"), Key=key,
    )


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

    # ---------------- producer ----------------

    def _enqueue(self, filename: str, data: bytes) -> Path:
        final_path = self.queue_dir / filename
        tmp_path = self.queue_dir / (filename + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, final_path)
        return final_path

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

    # ---------------- consumer ----------------

    def _pending(self) -> list[Path]:
        return sorted(
            (
                path
                for pattern in ("window-*", "epoch-*")
                for path in self.queue_dir.glob(pattern)
                if not path.name.endswith(".tmp")
            ),
            key=lambda path: (path.name.startswith("epoch-"), path.name),
        )

    @staticmethod
    def _key_for(path: Path) -> str | None:
        name = path.name
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
