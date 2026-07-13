"""Persistent retry queue for window archive uploads.

Decouples the validator's main loop from R2 reliability. Window archives
are written to a local directory and uploaded asynchronously by a
long-lived background worker. If R2 is unavailable, the worker retries
with exponential backoff per file. On process restart, the queue
directory is re-scanned and pending uploads resume — so a window's
audit data is never lost even if R2 is down for hours.

Design:
- ``enqueue(window_start, data)`` is called from the validator's main
  loop after ``seal_batch + train_step``. It writes the gzipped payload
  to ``{queue_dir}/window-{N}.json.gz`` atomically (.tmp + rename) and
  returns immediately. **Main loop never blocks on R2.**
- ``run_forever()`` is a background asyncio task that scans the queue
  directory continuously, attempts uploads via the same sync-boto3 path
  used in ``storage.upload_window_dataset`` (proven reliable), and
  deletes files on success.
- On startup, the queue directory is scanned and any pending files from
  before the last shutdown are uploaded first.

Failure modes handled:
- R2 down: retry with exponential backoff, keep file until success
- Process restart: pending files persist, picked up on next start
- Disk near full: caller's responsibility (the validator volume is
  dedicated, payloads are ~200KB each, so a 1GB volume buffers ~5000
  windows = ~20 days at our cadence)
- Single bad payload: failure on one file doesn't block the rest;
  worker moves on and circles back after the per-file backoff
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


# Backoff schedule per file (seconds). After the last entry, retry stays
# at that rate indefinitely. Tuned for R2 outage patterns observed in
# production: most transient blips clear within a minute; longer-duration
# issues (Cloudflare regional incidents) can last 30-60 min.
RETRY_BACKOFF_SECONDS: tuple[int, ...] = (5, 30, 120, 600, 1800)


def _default_queue_dir() -> str:
    """Resolve the queue directory.

    Priority:
    1. RELIQUARY_ARCHIVE_QUEUE_DIR env var (explicit override)
    2. {RELIQUARY_STATE_DIR}/pending_archives (validator's state volume)
    3. /tmp/reliquary_pending_archives (last-resort fallback for tests)
    """
    explicit = os.environ.get("RELIQUARY_ARCHIVE_QUEUE_DIR")
    if explicit:
        return explicit
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return os.path.join(state_dir, "pending_archives")


class ArchiveQueue:
    """Persistent retry queue for ``upload_window_dataset`` payloads."""

    def __init__(self, queue_dir: str | None = None) -> None:
        self.queue_dir = Path(queue_dir or _default_queue_dir())
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        # Per-file attempt counts for backoff calculation.
        self._attempts: dict[str, int] = {}
        # When the worker is paused waiting on a file's backoff, we
        # record the earliest time at which that file may be retried.
        self._next_attempt_at: dict[str, float] = {}
        self._uploads_succeeded_total = 0
        self._upload_failures_total = 0
        self._last_upload_success_ts: float | None = None
        self._last_upload_failure_ts: float | None = None
        self._last_uploaded_window: int | None = None
        self._last_failed_window: int | None = None

    # ------------------------------------------------------------------
    # Producer API — called from the validator's main loop
    # ------------------------------------------------------------------

    def enqueue(self, window_start: int, data: dict) -> Path:
        """Write the archive payload to disk and return immediately.

        Writes atomically via ``.tmp + os.replace`` so a process crash
        mid-write cannot leave a half-written file in the queue.
        """
        payload = json.dumps(data, separators=(",", ":")).encode()
        compressed = gzip.compress(payload)

        final_path = self.queue_dir / f"window-{window_start}.json.gz"
        tmp_path = self.queue_dir / f"window-{window_start}.json.gz.tmp"
        with open(tmp_path, "wb") as f:
            f.write(compressed)
        os.replace(tmp_path, final_path)

        logger.info(
            "ArchiveQueue: enqueued window %d (%d bytes, path=%s)",
            window_start, len(compressed), final_path,
        )
        return final_path

    # ------------------------------------------------------------------
    # Consumer API — the background worker
    # ------------------------------------------------------------------

    def _pending(self) -> list[Path]:
        """Sorted oldest-first list of pending files in the queue."""
        return sorted(
            p for p in self.queue_dir.glob("window-*.json.gz")
            if not p.name.endswith(".tmp")
        )

    @staticmethod
    def _window_n_from_path(path: Path) -> int | None:
        """Parse ``window-<N>.json.gz`` -> ``N`` or None on malformed name."""
        try:
            stem = path.name  # window-N.json.gz
            return int(stem.split("-", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            return None

    def _backoff_delay(self, attempts: int) -> float:
        """Exponential backoff: lookup table with floor at the last entry."""
        if attempts <= 0:
            return 0.0
        if attempts <= len(RETRY_BACKOFF_SECONDS):
            return float(RETRY_BACKOFF_SECONDS[attempts - 1])
        return float(RETRY_BACKOFF_SECONDS[-1])

    def snapshot(self, *, now: float | None = None) -> dict:
        """Return a secret-free, JSON-safe queue health snapshot."""
        pending = self._pending()
        oldest_window = (
            self._window_n_from_path(pending[0]) if pending else None
        )
        oldest_age_seconds = None
        if pending:
            try:
                current = time.time() if now is None else float(now)
                oldest_age_seconds = max(
                    0.0, current - pending[0].stat().st_mtime
                )
            except OSError:
                oldest_age_seconds = None
        return {
            "depth": len(pending),
            "oldest_window": oldest_window,
            "oldest_age_seconds": oldest_age_seconds,
            "uploads_succeeded_total": self._uploads_succeeded_total,
            "upload_failures_total": self._upload_failures_total,
            "last_upload_success_ts": self._last_upload_success_ts,
            "last_upload_failure_ts": self._last_upload_failure_ts,
            "last_uploaded_window": self._last_uploaded_window,
            "last_failed_window": self._last_failed_window,
        }

    async def _try_upload(self, path: Path) -> bool:
        """Attempt to upload one pending file. Returns True on success.

        Imports the sync-boto3 helper lazily to avoid an import cycle
        between this module and ``infrastructure.storage`` at process
        startup.
        """
        window_n = self._window_n_from_path(path)
        if window_n is None:
            logger.error("ArchiveQueue: dropping malformed pending file: %s", path)
            try:
                path.unlink()
            except OSError:
                pass
            return False

        try:
            body = path.read_bytes()
        except OSError as e:
            logger.error("ArchiveQueue: failed to read %s: %s", path, e)
            return False

        # Local import to avoid storage.py <-> archive_queue.py import cycle.
        from reliquary.infrastructure.storage import _sync_boto3_put

        account_id = os.getenv("R2_ACCOUNT_ID", "")
        access_key_id = os.getenv("R2_ACCESS_KEY_ID", "")
        secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY", "")
        endpoint = (
            os.getenv("R2_ENDPOINT_URL")
            or f"https://{account_id}.r2.cloudflarestorage.com"
        )
        region = os.getenv("R2_REGION", "us-east-1")
        bucket = os.getenv("R2_BUCKET_ID", "reliquary")
        key = f"reliquary/dataset/window-{window_n}.json.gz"

        try:
            await asyncio.to_thread(
                _sync_boto3_put,
                bucket, key, body,
                account_id, access_key_id, secret_access_key,
                endpoint, region,
            )
        except Exception as e:
            self._upload_failures_total += 1
            self._last_upload_failure_ts = time.time()
            self._last_failed_window = window_n
            attempts = self._attempts.get(str(path), 0) + 1
            self._attempts[str(path)] = attempts
            delay = self._backoff_delay(attempts)
            self._next_attempt_at[str(path)] = (
                asyncio.get_running_loop().time() + delay
            )
            logger.warning(
                "ArchiveQueue: upload failed for window %d (attempt %d): %s. "
                "Backing off %.0fs before retry.",
                window_n, attempts, e, delay,
            )
            return False

        attempts_used = self._attempts.pop(str(path), 0) + 1
        self._next_attempt_at.pop(str(path), None)
        self._uploads_succeeded_total += 1
        self._last_upload_success_ts = time.time()
        self._last_uploaded_window = window_n
        try:
            path.unlink()
        except OSError as e:
            # Upload succeeded but we couldn't delete the file. Log + keep
            # going; the next pass will retry, R2 will return success
            # again (PUT is idempotent on the same key), and we'll try
            # to delete again.
            logger.warning(
                "ArchiveQueue: upload OK for window %d but failed to "
                "delete %s: %s", window_n, path, e,
            )
        logger.info(
            "Uploaded GRPO dataset for window %d (%d bytes, key=%s, "
            "attempts=%d)",
            window_n, len(body), key, attempts_used,
        )
        return True

    async def run_forever(self) -> None:
        """Long-lived worker. Cancels cleanly on asyncio shutdown."""
        logger.info(
            "ArchiveQueue worker starting. queue_dir=%s, backoff=%s",
            self.queue_dir, RETRY_BACKOFF_SECONDS,
        )
        # Telemetry: count of queue-depth observations above 1, useful
        # to spot R2 outage durations in logs.
        last_depth_log = 0.0
        while True:
            try:
                pending = self._pending()
                if not pending:
                    await asyncio.sleep(5)
                    continue

                # Periodic queue-depth log (max once per minute).
                now = asyncio.get_running_loop().time()
                if len(pending) > 1 and (now - last_depth_log) > 60:
                    logger.info(
                        "ArchiveQueue: depth=%d, oldest=%s",
                        len(pending), pending[0].name,
                    )
                    last_depth_log = now

                # Pick the first file whose backoff has elapsed.
                target: Path | None = None
                for p in pending:
                    next_at = self._next_attempt_at.get(str(p), 0.0)
                    if next_at <= now:
                        target = p
                        break

                if target is None:
                    # Everything is in backoff. Sleep until the soonest
                    # next-attempt time, capped at 10s so we stay
                    # responsive to new enqueues.
                    earliest = min(
                        self._next_attempt_at.get(str(p), now) for p in pending
                    )
                    await asyncio.sleep(max(1.0, min(10.0, earliest - now)))
                    continue

                await self._try_upload(target)
                # Small pause to avoid tight loop on fast-path success.
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                logger.info("ArchiveQueue worker cancelled — draining stopped")
                raise
            except Exception:
                # Unexpected error — log and keep the worker alive.
                logger.exception("ArchiveQueue worker iteration failed")
                await asyncio.sleep(10)


# ----------------------------------------------------------------------
# Module-level singleton accessor
# ----------------------------------------------------------------------

_QUEUE: ArchiveQueue | None = None


def get_archive_queue() -> ArchiveQueue:
    """Return the process-wide ArchiveQueue (created lazily)."""
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = ArchiveQueue()
    return _QUEUE


__all__ = ["ArchiveQueue", "get_archive_queue", "RETRY_BACKOFF_SECONDS"]
