"""FastAPI server: receives v2 GRPO market submissions, exposes window state.

/submit drops requests on an asyncio queue (worker thread drains it off the
event loop so GRAIL verification doesn't block HTTP responses). Under
TestClient (no worker running), /submit runs synchronously so tests see
the real verdict.

/verdicts/{hotkey} surfaces the real per-submission verdicts (accept /
specific reject reason) that ``/submit`` cannot return in real time. Under
the production worker path /submit replies with a provisional ``SUBMITTED``
sentinel and the actual verdict lands in this endpoint a few seconds later,
once the worker has run the full verification pipeline. Miners learn the
truth without having to wait minutes for the R2 archive upload.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from reliquary.constants import (
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    VALIDATOR_HTTP_PORT,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    Verdict,
    VerdictsResponse,
)
from reliquary.validator.batcher import GrpoWindowBatcher

logger = logging.getLogger(__name__)


# How many recent verdicts to remember per hotkey. Bounded so the
# ring buffer can't grow without limit if a misbehaving miner spams.
# At ~250 B per verdict × 200 entries × ~50 hotkeys ≈ 2.5 MB — cheap.
VERDICT_CAP_PER_HOTKEY = 200


class _Health(BaseModel):
    status: str
    active_window: int | None


class ValidatorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = VALIDATOR_HTTP_PORT) -> None:
        self.host = host
        self.port = port
        self.active_batcher: GrpoWindowBatcher | None = None
        self.app: FastAPI = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._submit_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None
        from reliquary.protocol.submission import WindowState
        self._current_state: WindowState = WindowState.READY
        self._current_checkpoint = None  # ManifestEntry | None
        self._late_drop_callback: Callable[[str, str], None] | None = None
        # Per-hotkey submission counter. Reset every time the active
        # batcher swaps (= window boundary). Read in /submit before any
        # heavier check so a saturated miner trip the rate limit on the
        # cheapest possible path.
        self._per_window_counts: dict[str, int] = {}
        # Per-hotkey ring buffer of recent verdicts. Keys are miner ss58
        # addresses; values are deques of ``Verdict``-shaped dicts (stored
        # as plain dicts to keep the hot path serialization-free).
        # asyncio is single-threaded so no lock is needed — every mutation
        # site runs on the event loop.
        self._verdicts: dict[str, collections.deque[dict]] = {}

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        # New batcher → new window → reset per-hotkey counters.
        if batcher is not self.active_batcher:
            self._per_window_counts = {}
        self.active_batcher = batcher

    def set_current_state(self, state) -> None:
        self._current_state = state

    def set_current_checkpoint(self, entry) -> None:
        self._current_checkpoint = entry

    def set_late_drop_callback(
        self, fn: Callable[[str, str], None] | None,
    ) -> None:
        """Register a callback fired as ``(hotkey, reason)`` on every late
        drop — reasons are ``"window_not_active"`` (HTTP-level) or
        ``"worker_dropped"`` (queue worker). Service registers in __init__.
        """
        self._late_drop_callback = fn

    def record_verdict(
        self,
        hotkey: str,
        merkle_root: str,
        accepted: bool,
        reason: RejectReason | str,
        *,
        window_n: int | None = None,
    ) -> None:
        """Record a per-submission verdict for ``/verdicts/{hotkey}``.

        Called from every code path that decides accept/reject:

          * HTTP rate-limit / window-not-active / batch-filled early cutoffs
            in the ``/submit`` handler (before the request even reaches the
            queue worker)
          * ``_submit_worker`` after each ``batcher.accept_submission``
            returns its real verdict (the path that's currently invisible
            to miners because /submit returned ``SUBMITTED`` provisionally)
          * ``_submit_worker`` late drops for items dequeued after the
            batcher swap or seal (``worker_dropped`` / ``batch_filled``)

        The verdict is stored in a per-hotkey ring buffer
        (``VERDICT_CAP_PER_HOTKEY`` entries). Older verdicts roll off
        silently. Read-side: ``GET /verdicts/{hotkey}`` filters by hotkey
        and (optionally) by a ``since`` unix timestamp.
        """
        if hotkey not in self._verdicts:
            self._verdicts[hotkey] = collections.deque(maxlen=VERDICT_CAP_PER_HOTKEY)
        # Normalise enum → value so the ring is a uniform dict shape.
        reason_str = reason.value if isinstance(reason, RejectReason) else reason
        self._verdicts[hotkey].append({
            "merkle_root": merkle_root,
            "window_n": window_n,
            "accepted": accepted,
            "reason": reason_str,
            "ts": time.time(),
        })

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Reliquary Validator", version="2.0")

        @app.get("/health", response_model=_Health)
        async def health() -> _Health:
            return _Health(
                status="ok",
                active_window=(
                    self.active_batcher.window_start if self.active_batcher else None
                ),
            )

        @app.post("/submit", response_model=BatchSubmissionResponse)
        async def submit(request: BatchSubmissionRequest) -> BatchSubmissionResponse:
            from reliquary.protocol.submission import WindowState
            # Rate limit FIRST — cheapest reject before any state/queue work.
            hk = request.miner_hotkey
            n = self._per_window_counts.get(hk, 0)
            if n >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                if self._late_drop_callback is not None:
                    self._late_drop_callback(hk, "rate_limited")
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.RATE_LIMITED,
                    window_n=request.window_start,
                )
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.RATE_LIMITED,
                )
            self._per_window_counts[hk] = n + 1

            # v2.1: reject if state != OPEN
            if self._current_state != WindowState.OPEN:
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "window_not_active",
                    )
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.WINDOW_NOT_ACTIVE,
                    window_n=request.window_start,
                )
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
                )

            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            if request.window_start != batcher.window_start:
                raise HTTPException(status_code=409, detail="window_mismatch")

            # Early-cutoff: once the batcher has sealed (B_BATCH distinct
            # non-cooldown valid submissions received), ``select_batch``
            # will pick those by ``arrived_at``. Further submissions land
            # strictly later in arrival order, so they cannot displace
            # any of the already-selected entries. Queuing them costs
            # ~5–25 s of GRAIL forward pass per item with zero protocol
            # benefit, inflates the OPEN→TRAIN latency, and lets the
            # worker keep grinding into the TRAINING phase. Reject here
            # the moment the batch is closed.
            if batcher.is_sealed():
                if self._late_drop_callback is not None:
                    self._late_drop_callback(hk, "batch_filled")
                self.record_verdict(
                    hk, request.merkle_root, False, RejectReason.BATCH_FILLED,
                    window_n=request.window_start,
                )
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.BATCH_FILLED,
                )

            # Under TestClient (no worker running) we run synchronously so
            # tests see the real ``ACCEPTED`` verdict; under uvicorn we enqueue
            # for the worker and return ``SUBMITTED`` — a distinct sentinel
            # that tells the miner the request is queued, not yet validated.
            # The real verdict (accept/reject post-GRAIL) surfaces in the
            # validator's logs and in the R2 archive. The queue is unbounded
            # by design: pressure from a saturated window is relieved by
            # silently draining the leftover queue at the next batcher swap
            # (see ``_submit_worker`` below), not by HTTP-level rejections.
            if self._worker_task is None:
                resp = batcher.accept_submission(request)
                # Sync path (tests) — the real verdict is already known
                # before we return, so record it directly.
                self.record_verdict(
                    hk, request.merkle_root, resp.accepted, resp.reason,
                    window_n=request.window_start,
                )
                return resp

            await self._submit_queue.put((request, batcher))
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.SUBMITTED,
            )

        @app.get("/state", response_model=GrpoBatchState)
        async def state() -> GrpoBatchState:
            """Current window + checkpoint state. Lock-free: reads only the
            batcher's snapshot fields (set at construction) and the atomic
            ``valid_count`` counter. The submit worker holds ``batcher._lock``
            for up to ~25s per GRAIL verify, so this handler MUST NOT touch
            it — otherwise miners polling /state starve the event loop and
            timeout cascades hit every endpoint (see 2026-05-12 outage).
            """
            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            cp = self._current_checkpoint
            return GrpoBatchState(
                state=self._current_state,
                window_n=batcher.window_start,
                anchor_block=batcher.window_start,
                cooldown_prompts=batcher.cooldown_prompts_snapshot,
                valid_submissions=batcher.valid_count,
                checkpoint_n=cp.checkpoint_n if cp else 0,
                checkpoint_repo_id=cp.repo_id if cp else None,
                checkpoint_revision=cp.revision if cp else None,
                randomness=batcher.randomness,
            )

        @app.get("/checkpoint")
        async def checkpoint():
            cp = self._current_checkpoint
            if cp is None:
                raise HTTPException(status_code=404, detail="no_checkpoint")
            return {
                "checkpoint_n": cp.checkpoint_n,
                "repo_id": cp.repo_id,
                "revision": cp.revision,
                "signature": cp.signature,
            }

        @app.get("/verdicts/{hotkey}", response_model=VerdictsResponse)
        async def verdicts(hotkey: str, since: float = 0.0) -> VerdictsResponse:
            """Recent per-submission verdicts for ``hotkey``, ordered by
            ``ts`` ascending. The default ``since=0`` returns every verdict
            currently in the ring; pass the timestamp of the last verdict
            you saw to get only newer ones (incremental polling).

            Bounded read — at most ``VERDICT_CAP_PER_HOTKEY`` entries per
            hotkey live in the ring, so even a degenerate ``since=0`` poll
            never returns more than ~200 entries. Lock-free in the same way
            ``/state`` is (event-loop-only writes, atomic dict.get).

            Why this exists: ``/submit`` under the production worker path
            returns a provisional ``SUBMITTED`` sentinel, not the real
            verdict — that's known only after the worker drains the queue
            and runs the full verification pipeline (~5-25 s of GRAIL per
            item). Without this endpoint, the truth was only visible via
            the R2 archive (minutes-late, batched per window). Now miners
            can learn within seconds whether a specific submission cleared
            GRAIL, was rejected as a duplicate hash, hit the rate limit,
            or failed any other check — diagnosable by ``merkle_root``.

            Privacy: same trust model as the R2 archive (public). Anyone
            can query any hotkey's verdicts; we don't auth this. If you
            need confidential feedback, run a private validator.
            """
            ring = self._verdicts.get(hotkey)
            if not ring:
                return VerdictsResponse(verdicts=[])
            out = [
                Verdict(**entry) for entry in ring
                if entry["ts"] > since
            ]
            return VerdictsResponse(verdicts=out)

        return app

    async def _submit_worker(self) -> None:
        # Lazy import — keeps the module loadable in CPU-only test envs.
        from reliquary.validator.service import _try_empty_cuda_cache

        while True:
            try:
                request, batcher = await self._submit_queue.get()
            except asyncio.CancelledError:
                return
            # Silently drop items whose batcher is no longer the active one.
            # This is what relieves pressure from a saturated window: the
            # queue is unbounded by design, so a busy window can pile up
            # dozens of pending items behind the in-flight GRAIL. As soon
            # as the service's main loop opens the next window and swaps
            # ``active_batcher``, every leftover item is for a sealed
            # batcher whose ``_valid`` will never be re-archived — running
            # GRAIL on them would burn ~5-25s per item for nothing and
            # would keep the next window starving for cycles. We log at
            # info so operators can confirm the drain is happening; the
            # miner has already received a provisional ``SUBMITTED`` from
            # the /submit response and learns the real outcome (or its
            # absence) from the R2 archive.
            if batcher is not self.active_batcher:
                logger.info(
                    "dropping late submission prompt=%d hotkey=%s "
                    "(batcher window=%d no longer active)",
                    request.prompt_idx, request.miner_hotkey[:12],
                    batcher.window_start,
                )
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "worker_dropped",
                    )
                # Surface to the miner via /verdicts so they don't keep
                # interpreting the SUBMITTED sentinel as an accept.
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root, False,
                    RejectReason.WORKER_DROPPED,
                    window_n=request.window_start,
                )
                continue
            # Drain past-seal items without running GRAIL. The HTTP early-
            # cutoff catches submissions that arrive AFTER seal; this catches
            # the ones already in the queue from BEFORE seal that haven't
            # been dequeued yet. Together they cap per-window GRAIL work at
            # ~B_BATCH × verify-time instead of letting it grow with raw
            # arrival rate. Same accounting bucket as the HTTP path so a
            # miner inspecting late_drops sees one consistent metric.
            if batcher.is_sealed():
                logger.info(
                    "dropping post-seal queue item prompt=%d hotkey=%s "
                    "(batcher window=%d already filled)",
                    request.prompt_idx, request.miner_hotkey[:12],
                    batcher.window_start,
                )
                if self._late_drop_callback is not None:
                    self._late_drop_callback(
                        request.miner_hotkey, "batch_filled",
                    )
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root, False,
                    RejectReason.BATCH_FILLED,
                    window_n=request.window_start,
                )
                continue
            try:
                response = await asyncio.to_thread(
                    batcher.accept_submission, request
                )
                if response.accepted:
                    logger.info(
                        "accepted prompt=%d hotkey=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                    )
                else:
                    rewards = [r.reward for r in request.rollouts]
                    logger.warning(
                        "rejected prompt=%d hotkey=%s reason=%s rewards=%s",
                        request.prompt_idx, request.miner_hotkey[:12],
                        response.reason.value, rewards,
                    )
                # The verdict the /submit response *didn't* carry, now
                # observable to the miner via /verdicts.
                self.record_verdict(
                    request.miner_hotkey, request.merkle_root,
                    response.accepted, response.reason,
                    window_n=request.window_start,
                )
            except Exception as e:
                logger.exception(
                    "submission worker failed on prompt %d", request.prompt_idx
                )
                # OOM-recovery: when CUDA allocator can't get a handle
                # (CUBLAS_STATUS_ALLOC_FAILED, out-of-memory etc.) we MUST
                # release the cached pool before the next submission lands,
                # otherwise every subsequent forward pass fails too. The
                # generic .empty_cache() call covers all the cuBLAS / cuDNN
                # / activation-pool fragmentation scenarios we've observed.
                msg = str(e).lower()
                if any(s in msg for s in ("out of memory", "cublas", "cuda")):
                    await asyncio.to_thread(_try_empty_cuda_cache)
            finally:
                # Always reclaim activation memory after a forward pass so
                # back-to-back GRAIL verifies don't accumulate fragmentation.
                # The helper is a no-op on CPU-only hosts. Cost: ~ms; benefit:
                # prevents the multi-hour drift that took down the validator
                # on 2026-05-11.
                await asyncio.to_thread(_try_empty_cuda_cache)

    async def start(self) -> None:
        if self._task is not None:
            return
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port,
            log_level="warning", access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        self._worker_task = asyncio.create_task(self._submit_worker())
        await asyncio.sleep(0)
        logger.info("Validator HTTP server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
            self._server = None
