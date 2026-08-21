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

logger = logging.getLogger(__name__)

R2_TRAINING_PREFIX = "reliquary/training"

# Same backoff table as ArchiveQueue — tuned on observed R2 outages.
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (5, 30, 120, 600, 1800)

_PAYLOAD_SUFFIX = ".npz"
_TOMBSTONE_SUFFIX = ".tombstone.json"


def payload_key(window_start: int) -> str:
    return f"{R2_TRAINING_PREFIX}/window-{int(window_start)}{_PAYLOAD_SUFFIX}"


def tombstone_key(window_start: int) -> str:
    return (
        f"{R2_TRAINING_PREFIX}/window-{int(window_start)}{_TOMBSTONE_SUFFIX}"
    )


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

    # ---------------- consumer ----------------

    def _pending(self) -> list[Path]:
        return sorted(
            p for p in self.queue_dir.glob("window-*")
            if not p.name.endswith(".tmp")
        )

    @staticmethod
    def _key_for(path: Path) -> str | None:
        name = path.name
        if not name.startswith("window-"):
            return None
        if name.endswith(_TOMBSTONE_SUFFIX) or name.endswith(_PAYLOAD_SUFFIX):
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
        self, path: Path, upload_fn: Callable[[str, bytes], None],
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
        try:
            path.unlink()
        except OSError as e:
            logger.warning(
                "TrainingPayloadQueue: uploaded %s but delete failed: %s",
                key, e,
            )
        return True

    async def drain_once(
        self, upload_fn: Callable[[str, bytes], None] | None = None,
    ) -> int:
        """Attempt every pending file once (honoring backoff). Returns the
        number of successful uploads."""
        fn = upload_fn or _default_upload
        now = asyncio.get_running_loop().time()
        uploaded = 0
        for path in self._pending():
            next_at = self._next_attempt_at.get(str(path), 0.0)
            if next_at > now:
                continue
            if await self._try_upload(path, fn):
                uploaded += 1
        return uploaded

    async def run_forever(
        self, upload_fn: Callable[[str, bytes], None] | None = None,
    ) -> None:
        logger.info(
            "TrainingPayloadQueue worker starting. queue_dir=%s",
            self.queue_dir,
        )
        while True:
            try:
                await self.drain_once(upload_fn=upload_fn)
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
