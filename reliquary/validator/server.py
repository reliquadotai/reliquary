"""FastAPI server: receives v2 GRPO market submissions, exposes window state.

/submit drops requests on an asyncio queue (worker thread drains it off the
event loop so GRAIL verification doesn't block HTTP responses). Under
TestClient (no worker running), /submit runs synchronously so tests see
the real verdict.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)
from reliquary.validator.batcher import GrpoWindowBatcher

logger = logging.getLogger(__name__)


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

    def set_active_batcher(self, batcher: GrpoWindowBatcher | None) -> None:
        self.active_batcher = batcher

    def set_current_state(self, state) -> None:
        self._current_state = state

    def set_current_checkpoint(self, entry) -> None:
        self._current_checkpoint = entry

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
            # v2.1: reject if state != OPEN
            if self._current_state != WindowState.OPEN:
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
                )

            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            if request.window_start != batcher.window_start:
                raise HTTPException(status_code=409, detail="window_mismatch")

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
                return batcher.accept_submission(request)

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
