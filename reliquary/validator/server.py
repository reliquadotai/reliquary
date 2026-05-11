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

from reliquary.constants import MAX_SUBMIT_QUEUE_DEPTH, VALIDATOR_HTTP_PORT
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

            # Under TestClient (no worker running) we run synchronously so tests
            # see the real accept verdict; under uvicorn we enqueue for the
            # worker and return a provisional ACCEPTED. The worker's real
            # verdict surfaces in logs.
            if self._worker_task is None:
                return batcher.accept_submission(request)

            # Bound the queue: GRAIL verification is ~5-25s per item but
            # miners can submit much faster than that. Without a bound,
            # the queue accumulates during a window and at seal time we
            # have dozens of items that will run through GRAIL only to
            # land in a sealed batcher's _valid (never archived). Miners
            # whose items are in that drainage receive a provisional
            # ``accepted=True`` here but their submission is silently
            # dropped post-seal — they're unaware. Rejecting at the
            # enqueue boundary tells them honestly to back off.
            if self._submit_queue.qsize() >= MAX_SUBMIT_QUEUE_DEPTH:
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_BUSY,
                )

            await self._submit_queue.put((request, batcher))
            return BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED,
            )

        @app.get("/state", response_model=GrpoBatchState)
        async def state() -> GrpoBatchState:
            """Current window + checkpoint state. v2.1 has one active batcher per validator."""
            batcher = self.active_batcher
            if batcher is None:
                raise HTTPException(status_code=503, detail="no_active_window")
            cp = self._current_checkpoint
            return GrpoBatchState(
                state=self._current_state,
                window_n=batcher.window_start,
                anchor_block=batcher.window_start,
                cooldown_prompts=sorted(
                    batcher._cooldown.current_cooldown_set(batcher.window_start)
                ),
                valid_submissions=len(batcher.valid_submissions()),
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
            # Drop items whose batcher is no longer the active one. This
            # is a defense-in-depth complement to the queue-depth bound:
            # the bound limits how many such items can sneak in, this
            # drop catches the residual ones that were enqueued between
            # /submit and the next active-batcher swap. Without it, the
            # worker would spend ~30s of GRAIL on a sealed batcher whose
            # _valid is never archived, blocking the new window's
            # progress that much longer.
            if batcher is not self.active_batcher:
                logger.debug(
                    "dropping late submission prompt=%d (batcher window=%d "
                    "no longer active)",
                    request.prompt_idx, batcher.window_start,
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
