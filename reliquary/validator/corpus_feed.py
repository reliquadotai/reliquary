"""The arrival feed: how a judge process learns what the front accepted.

In one process the route calls ``auditor.enqueue`` the moment a record is
written, and the auditor's sibling rule relies on it: an unaudited pass of X
happens ``hold + 420 s`` after X arrived, by which time every sibling received
inside X's hold has finished its record write (<= 405 s) and been enqueued, so
a drawn sibling's failure always reaches X before X is paid.

Across processes the front queues each accepted id for its judge
(``JudgeLink.accepted``, never awaited by the route) and posts the queue over
a unix socket, plus a heartbeat every couple of seconds. The judge
(``ArrivalFeed``) enqueues what it receives and answers ``covered()`` -- the
auditor's ``arrivals_covered``: the instant up to which every accepted id has
been enqueued here, or None:

- after a new front (a new ``epoch``, the first one this judge sees included)
  or a queue overflow (``dropped``), None until a full listing of the store has
  enqueued every pending record;
- while its id queue or an accepted body awaiting publication is pending,
  None until the front can vouch for a complete cut again;
- then the newest ``as_of`` received (the front's clock when a post emptied its
  queue: everything accepted before it was in that post or an earlier one).

The auditor passes a record unaudited only once ``covered()`` reaches that
record's receipt + hold + 405 s (every sibling that could catch it was
accepted by then). A cut replaces the prior one, including after a clock rollback.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How old the newest as_of may be for ``complete()`` (health, logs only).
FEED_FRESH_SECONDS = 10.0
# An empty post at least this often, so an idle front still vouches.
FEED_HEARTBEAT_SECONDS = 2.0
# How long the sender waits after the first id to gather more into one post.
FEED_GATHER_SECONDS = 0.05
FEED_POST_IDS = 5000
FEED_MAX_IDS = int(os.environ.get("RELIQUARY_CORPUS_FEED_MAX_IDS", "200000"))
# A listing that failed is retried this soon.
LISTING_RETRY_SECONDS = 5.0
STATUS_TIMEOUT_SECONDS = 2.0


class UdsClient:
    """JSON over HTTP on a unix socket; one connection pool per event loop."""

    def __init__(self, socket_path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = str(socket_path)
        self._timeout = timeout
        self._client = None
        self._loop = None

    def _http(self):
        import httpx

        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=self.path),
                base_url="http://corpus-judge", timeout=self._timeout)
            self._loop = loop
        return self._client

    async def post(self, path: str, body: Mapping, *, timeout: float | None = None) -> dict:
        response = await self._http().post(path, json=body,
                                           timeout=timeout or self._timeout)
        response.raise_for_status()
        return response.json()

    async def get(self, path: str, *, timeout: float | None = None) -> dict | None:
        response = await self._http().get(path, timeout=timeout or self._timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


class JudgeLink:
    """The front's side of one judge process: its jobs' accepted ids, in order.

    ``accepted`` only appends (the route never waits on a judge); ``run``
    posts. Past ``max_ids`` queued the oldest are dropped and the next post
    says so, which makes the judge list the store before trusting the feed.
    """

    def __init__(self, socket_path: str | Path, job_ids, *, epoch: str | None = None,
                 clock: Callable[[], float] = time.time, max_ids: int = FEED_MAX_IDS,
                 heartbeat_seconds: float = FEED_HEARTBEAT_SECONDS,
                 gather_seconds: float = FEED_GATHER_SECONDS, client=None) -> None:
        self.job_ids = set(job_ids)
        self.epoch = epoch or secrets.token_hex(8)
        self._clock = clock
        self._max_ids = max_ids
        self._heartbeat = heartbeat_seconds
        self._gather = gather_seconds
        self._client = client or UdsClient(socket_path)
        self._seq = itertools.count(1)
        self._queue: collections.deque[tuple[int, str, str]] = collections.deque()
        self._drops = 0
        self._drops_sent = 0
        self._wake: asyncio.Event | None = None
        self._failing_since: float | None = None
        self.posted = 0
        self.pending_record_arrivals: dict[str, Mapping[str, float]] = {}

    def __len__(self) -> int:
        return len(self._queue)

    def accepted(self, job_id: str, submission_id: str) -> None:
        self._queue.append((next(self._seq), job_id, submission_id))
        while len(self._queue) > self._max_ids:
            self._queue.popleft()
            self._drops += 1
        if self._wake is not None:
            self._wake.set()

    async def flush_once(self) -> bool:
        """One post of at most FEED_POST_IDS ids (or a heartbeat). True when it landed."""
        cut = self._clock()
        chunk = list(itertools.islice(self._queue, FEED_POST_IDS))
        drops = self._drops
        ids: dict[str, list[str]] = {}
        for _, job_id, sid in chunk:
            ids.setdefault(job_id, []).append(sid)
        body = {"epoch": self.epoch, "dropped": drops > self._drops_sent, "ids": ids,
                "as_of": (cut if len(chunk) == len(self._queue)
                          and not any(self.pending_record_arrivals.values()) else None)}
        try:
            await self._client.post("/feed", body)
        except Exception as exc:  # noqa: BLE001 - the judge may be restarting
            if self._failing_since is None:
                self._failing_since = cut
                logger.warning("corpus feed to %s failed (%r); %d id(s) kept for it",
                               self._client.path, exc, len(self._queue))
            return False
        if self._failing_since is not None:
            logger.info("corpus feed to %s back after %.0f s", self._client.path,
                        cut - self._failing_since)
            self._failing_since = None
        last = chunk[-1][0] if chunk else 0
        while self._queue and self._queue[0][0] <= last:
            self._queue.popleft()
        self._drops_sent = drops
        self.posted += len(chunk)
        return True

    async def run(self) -> None:
        self._wake = asyncio.Event()
        while True:
            if not self._queue:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._heartbeat)
                except asyncio.TimeoutError:
                    pass
                else:
                    await asyncio.sleep(self._gather)
            landed = await self.flush_once()
            if not landed:
                await asyncio.sleep(0.5)

    # -- the status routes' reads -------------------------------------------

    async def stats(self, job_id: str) -> dict | None:
        return await self._client.get(f"/jobs/{job_id}/stats", timeout=STATUS_TIMEOUT_SECONDS)

    async def miner(self, job_id: str, hotkey: str) -> dict | None:
        return await self._client.get(f"/jobs/{job_id}/miners/{hotkey}",
                                      timeout=STATUS_TIMEOUT_SECONDS)


class ArrivalFeed:
    """The judge's side: enqueue what the front posts, and say whether an
    accepted record could still be missing (``complete``)."""

    def __init__(self, auditors: Mapping[str, Any] | None = None, *,
                 clock: Callable[[], float] = time.time,
                 fresh_seconds: float = FEED_FRESH_SECONDS,
                 retry_seconds: float = LISTING_RETRY_SECONDS) -> None:
        self.auditors = dict(auditors or {})
        self._clock = clock
        self._fresh = fresh_seconds
        self._retry = retry_seconds
        self.epoch: str | None = None
        self.covered_until: float | None = None
        self._generation = 0
        self._listed = 0
        self._tasks: set[asyncio.Task] = set()
        self.unknown_ids = 0

    def covered(self) -> float | None:
        """Every id the front accepted before this instant is enqueued here;
        None while a listing or the front's complete cut is pending."""
        if self.epoch is None or self._listed != self._generation:
            return None
        return self.covered_until

    def complete(self) -> bool:
        covered = self.covered()
        return covered is not None and self._clock() - covered <= self._fresh

    def state(self) -> dict:
        return {"epoch": self.epoch, "covered_until": self.covered_until,
                "listed": self._listed == self._generation, "complete": self.complete(),
                "covered": self.covered()}

    def receive(self, doc: Mapping) -> None:
        for job_id, ids in (doc.get("ids") or {}).items():
            auditor = self.auditors.get(job_id)
            if auditor is None:
                self.unknown_ids += len(ids)
                logger.warning("corpus feed: %d id(s) for job %s, not judged here",
                               len(ids), job_id)
                continue
            for sid in ids:
                auditor.enqueue(sid)
        if doc.get("epoch") != self.epoch or doc.get("dropped"):
            if doc.get("epoch") != self.epoch:
                logger.info("corpus feed: front epoch %s (was %s); listing the store before "
                            "any unaudited pass", doc.get("epoch"), self.epoch)
            else:
                logger.warning("corpus feed: the front dropped ids; listing the store")
            self.epoch = doc.get("epoch")
            self._generation += 1
            task = asyncio.ensure_future(self._list(self._generation))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        as_of = doc.get("as_of")
        self.covered_until = None
        if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
            try:
                cut = float(as_of)
            except OverflowError:
                pass
            else:
                if math.isfinite(cut) and cut >= 0:
                    self.covered_until = cut

    async def _list(self, generation: int) -> None:
        while generation == self._generation:
            try:
                for auditor in list(self.auditors.values()):
                    await auditor.rescan_store()
            except Exception:
                logger.exception("corpus feed: listing failed; retrying")
                await asyncio.sleep(self._retry)
                continue
            self._listed = max(self._listed, generation)
            logger.info("corpus feed: store listed for generation %d", generation)
            return

    def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()

    async def aclose(self) -> None:
        tasks = list(self._tasks)
        self.close()
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "ArrivalFeed",
    "FEED_FRESH_SECONDS",
    "FEED_HEARTBEAT_SECONDS",
    "JudgeLink",
    "UdsClient",
]
