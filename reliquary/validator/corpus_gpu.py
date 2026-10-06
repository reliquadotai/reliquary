"""The GPU process of the split corpus validator, and the judges' client to it.

The checkpoint is loaded once, here. Every judge (and every auditor left in
the front) sends the rows its pass must audit -- ``(tokens, prompt_len,
proofs)``, what ``score_sequences`` takes -- and gets back the per-chunk
comparisons, never a verdict: the decision stays with the auditor, as with a
remote executor (``corpus_audit_protocol``), whose score encoding this reuses.

Requests wait in one FIFO; consecutive ones are merged into one
``score_sequences`` call so rows of several judges share padded sub-batches.
A merged call that raises is re-run request by request, so one judge's bad
record never fails another's. Transport is HTTP over a unix socket.
"""

from __future__ import annotations

import array
import asyncio
import collections
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

GPU_SOCKET = "gpu.sock"
GPU_INFO = "gpu-info.json"


def _batch_tokens() -> int:
    from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

    return AUDIT_BATCH_TOKENS


def merge_tokens() -> int:
    """Tokens of queued requests merged into one ``score_sequences`` call."""
    return int(os.environ.get("RELIQUARY_CORPUS_GPU_MERGE_TOKENS", 4 * _batch_tokens()))


def queue_tokens() -> int:
    """Tokens queued past which a new request is refused 503 (retried)."""
    return int(os.environ.get("RELIQUARY_CORPUS_GPU_QUEUE_TOKENS", 16 * _batch_tokens()))


# Forward failures in a row (out of memory aside) after which the card is
# taken for broken: the GPU process exits and the supervisor reloads the model.
FATAL_AFTER = 3
_STICKY = ("illegal memory access", "device-side assert", "unspecified launch failure",
           "cuda error", "cudnn_status", "cublas_status", "nccl error", "ecc error")


def is_out_of_memory(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def is_sticky_fault(exc: BaseException) -> bool:
    """A CUDA error the context does not recover from: every later forward
    fails too, until the process reloads."""
    if is_out_of_memory(exc):
        return False
    text = str(exc).lower()
    return type(exc).__name__ == "AcceleratorError" or any(m in text for m in _STICKY)


class GpuBusy(Exception):
    """The queue is full: retry later."""


class GpuScoreError(Exception):
    """The GPU process raised something other than an audit error."""


def scores_to_wire(scores: Sequence) -> list:
    """``score_sequences`` output as the executor's ``ItemScore`` fields."""
    return [[status, [[int(r.exp_mismatches), float(r.mant_err_mean), float(r.mant_err_median)]
                      for r in results]]
            for status, results in scores]


def scores_from_wire(wire: Sequence) -> list:
    """Back to ``(status, (ChunkResult, ...))``, as the dispatcher rebuilds them."""
    from reliquary.protocol.toploc import ChunkResult

    return [(status, tuple(ChunkResult(int(e), float(m), float(d)) for e, m, d in chunks))
            for status, chunks in wire]


def encode_request(rows: Sequence, *, chunk_tokens: int, topk: int) -> bytes:
    """A JSON header line, then every row's token ids as int32, then every
    proof (base64 never holds a newline) joined by newlines. A pass's rows
    are tens of megabytes as JSON, whose encode held the sender's GIL for
    seconds; packed, they are a copy."""
    items, proofs = [], []
    tokens = array.array("i")
    for row_tokens, prompt_len, row_proofs, *rest in rows:
        tokens.extend(row_tokens)
        proofs.extend(row_proofs)
        items.append([len(row_tokens), int(prompt_len), len(row_proofs)]
                     + ([[[int(a), int(b)] for a, b in rest[0]]] if rest and rest[0] is not None else []))
    header = json.dumps({"chunk_tokens": int(chunk_tokens), "topk": int(topk), "items": items,
                         "token_bytes": len(tokens) * tokens.itemsize})
    if sys.byteorder != "little":
        tokens.byteswap()
    return b"".join([header.encode(), b"\n", tokens.tobytes(), "\n".join(proofs).encode()])


def decode_request(body: bytes) -> tuple[list, int, int]:
    end = body.index(b"\n")
    header = json.loads(body[:end])
    start = end + 1
    tokens = array.array("i")
    tokens.frombytes(body[start:start + header["token_bytes"]])
    if sys.byteorder != "little":
        tokens.byteswap()
    flat = tokens.tolist()
    text = body[start + header["token_bytes"]:]
    total = sum(item[2] for item in header["items"])
    # An empty proof is a proof (a failing one), never "no proof".
    proofs = text.decode().split("\n") if total else []
    if len(proofs) != total or len(flat) != sum(item[0] for item in header["items"]):
        raise ValueError("a request's proofs or tokens do not match its header")
    rows, at, proof_at = [], 0, 0
    for n_tokens, prompt_len, n_proofs, *spans in header["items"]:
        row = (flat[at:at + n_tokens], prompt_len, proofs[proof_at:proof_at + n_proofs])
        rows.append(row + ([tuple(s) for s in spans[0]],) if spans else row)
        at += n_tokens
        proof_at += n_proofs
    return rows, int(header["chunk_tokens"]), int(header["topk"])


@dataclass
class _Request:
    rows: list
    chunk_tokens: int
    topk: int
    future: asyncio.Future
    tokens: int = field(init=False)
    queued_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = sum(len(row[0]) for row in self.rows)


class GpuBatcher:
    """The FIFO in front of the model. ``score(rows, chunk_tokens, topk)``
    returns ``(scores, forward_seconds, verify_seconds)`` and runs on
    ``executor`` (one thread: one forward at a time)."""

    def __init__(self, score: Callable[..., tuple], *, executor=None,
                 merge_tokens_limit: int | None = None,
                 queue_tokens_limit: int | None = None, on_error=None,
                 on_fatal=None, fatal_after: int = FATAL_AFTER) -> None:
        self._score = score
        self._executor = executor
        self._merge = merge_tokens_limit if merge_tokens_limit is not None else merge_tokens()
        self._limit = queue_tokens_limit if queue_tokens_limit is not None else queue_tokens()
        self._queue: collections.deque[_Request] = collections.deque()
        self._queued_tokens = 0
        self._wake: asyncio.Event | None = None
        # Called after a failed call, before the retries (frees CUDA memory).
        self._on_error = on_error
        # Called with the reason once the card looks broken (the process exits).
        self._on_fatal = on_fatal
        self._fatal_after = fatal_after
        self._failures = 0
        self.stats = collections.Counter()

    @property
    def queued_tokens(self) -> int:
        return self._queued_tokens

    async def submit(self, rows: list, chunk_tokens: int, topk: int) -> tuple:
        request = _Request(rows, chunk_tokens, topk, asyncio.get_running_loop().create_future())
        if self._queue and self._queued_tokens + request.tokens > self._limit:
            self.stats["refused"] += 1
            raise GpuBusy(f"{self._queued_tokens} tokens queued")
        self._queue.append(request)
        self._queued_tokens += request.tokens
        if self._wake is not None:
            self._wake.set()
        return await request.future

    def _take(self) -> list[_Request]:
        first = self._queue.popleft()
        batch, tokens = [first], first.tokens
        while (self._queue and (self._queue[0].chunk_tokens, self._queue[0].topk)
               == (first.chunk_tokens, first.topk)
               and tokens + self._queue[0].tokens <= self._merge):
            request = self._queue.popleft()
            batch.append(request)
            tokens += request.tokens
        self._queued_tokens -= tokens
        return batch

    async def _call(self, rows, chunk_tokens, topk):
        from reliquary.validator.corpus_judge_threads import run_in

        return await run_in(self._executor, self._score, rows, chunk_tokens, topk)

    def _failed(self, exc: BaseException) -> None:
        if self._on_error is not None:
            self._on_error()
        if not is_out_of_memory(exc):
            self._failures += 1
        if self._on_fatal is not None and (is_sticky_fault(exc)
                                           or self._failures >= self._fatal_after):
            logger.critical("corpus gpu: %r after %d failed forward(s) in a row; exiting so the "
                            "model is reloaded", exc, self._failures)
            self._on_fatal(str(exc))
            self._on_fatal = None

    @staticmethod
    def _settle(request: _Request, result=None, exc: BaseException | None = None) -> None:
        if request.future.done():
            return  # its caller went away
        if exc is not None:
            request.future.set_exception(exc)
        else:
            request.future.set_result(result)

    async def _run_batch(self, batch: list[_Request]) -> None:
        live = [r for r in batch if not r.future.done()]
        if not live:
            return
        rows = [row for r in live for row in r.rows]
        started = time.monotonic()
        try:
            scores, forward, verify = await self._call(rows, live[0].chunk_tokens, live[0].topk)
        except Exception as exc:  # noqa: BLE001 - handed to the requests
            self.stats["failed_batches"] += 1
            self._failed(exc)
            if len(live) == 1:
                self._settle(live[0], exc=exc)
                return
            logger.warning("corpus gpu batch of %d requests failed (%r); each alone",
                           len(live), exc)
            for request in live:
                try:
                    result = await self._call(request.rows, request.chunk_tokens, request.topk)
                except Exception as alone:  # noqa: BLE001
                    self._failed(alone)
                    self._settle(request, exc=alone)
                else:
                    self._failures = 0
                    self._settle(request, result)
            return
        self._failures = 0
        total = sum(r.tokens for r in live) or 1
        at = 0
        for request in live:
            share = request.tokens / total
            self._settle(request, (scores[at:at + len(request.rows)],
                                   forward * share, verify * share))
            at += len(request.rows)
        self.stats["batches"] += 1
        self.stats["requests"] += len(live)
        logger.info(
            "corpus gpu batch: requests=%d rows=%d tokens=%d forward=%.2fs verify=%.2fs "
            "oldest_wait=%.1fs queued_tokens=%d tokens_per_s=%.0f",
            len(live), len(rows), total, forward, verify,
            started - min(r.queued_at for r in live), self._queued_tokens,
            total / (forward + verify) if forward + verify > 0 else 0.0)

    async def run(self) -> None:
        self._wake = asyncio.Event()
        while True:
            while not self._queue:
                self._wake.clear()
                await self._wake.wait()
            await self._run_batch(self._take())


def build_gpu_app(batcher: GpuBatcher, info: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/info")
    async def gpu_info() -> dict:
        return info

    @app.post("/score")
    async def score(request: Request):
        body = await request.body()
        rows, chunk_tokens, topk = await asyncio.to_thread(decode_request, body)
        try:
            scores, forward, verify = await batcher.submit(rows, chunk_tokens, topk)
        except GpuBusy:
            return Response(status_code=503)
        except Exception as exc:  # noqa: BLE001 - the judge decides what it means
            # By class, not name: torch.AcceleratorError and friends subclass
            # RuntimeError and are audit errors like it.
            return JSONResponse({"error": str(exc)[:2000], "kind": type(exc).__name__,
                                 "value_error": isinstance(exc, ValueError),
                                 "runtime_error": isinstance(exc, RuntimeError)})
        return JSONResponse({"scores": scores_to_wire(scores), "forward_seconds": forward,
                             "verify_seconds": verify})

    return app


def write_info(run_dir: str | Path, info: dict) -> None:
    path = Path(run_dir) / GPU_INFO
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, path)


async def read_info(run_dir: str | Path, *, poll_seconds: float = 1.0,
                    timeout: float | None = None) -> dict:
    """The info the GPU process published, waiting until it exists (the first
    model load); kept across GPU restarts, so later readers never wait."""
    path = Path(run_dir) / GPU_INFO
    start = time.monotonic()
    logged = False
    while True:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            pass
        if timeout is not None and time.monotonic() - start > timeout:
            raise TimeoutError(f"{path} not written within {timeout} s")
        if not logged:
            logger.info("waiting for the GPU process to publish %s", path)
            logged = True
        await asyncio.sleep(poll_seconds)


def _load_model(directory: str):
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_only_model

    return load_text_only_model(str(directory), torch_dtype=torch.bfloat16,
                                attn_implementation=ATTN_IMPLEMENTATION).to("cuda").eval()


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        logger.debug("empty_cache failed", exc_info=True)


async def serve_unix(app, path: str | Path, *, services=()) -> None:
    """``app`` on the unix socket ``path`` (a stale one is removed first).

    Signals are left to the process (``corpus_split.child_main``): an
    internal server has nothing to drain, and its owner must not keep running
    after the server alone shut down."""
    import contextlib

    import uvicorn
    from reliquary.validator.corpus_validator import _run_corpus_services

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    path = Path(path)
    path.unlink(missing_ok=True)
    server = _Server(uvicorn.Config(app, uds=str(path), log_level="warning", ws="none",
                                    timeout_keep_alive=600))
    await _run_corpus_services(server, services)


async def run_gpu_process(*, directory: str, run_dir: str, model_id: str = "",
                          model_revision: str = "", batch_tokens: int | None = None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from reliquary.validator import corpus_audit

    batch = batch_tokens if batch_tokens is not None else _batch_tokens()
    logger.info("corpus gpu: loading %s", directory)
    model = await asyncio.to_thread(_load_model, directory)
    info = {"vocab_size": int(model.get_input_embeddings().num_embeddings),
            "model_id": model_id, "model_revision": model_revision, "pid": os.getpid()}

    def score(rows, chunk_tokens, topk):
        # Looked up per call, so a test can stand in for the forward.
        return corpus_audit.score_sequences(model, rows, chunk_tokens=chunk_tokens, topk=topk,
                                            batch_tokens=batch)

    loop = asyncio.get_running_loop()

    def fatal(reason: str) -> None:
        # The answers in flight go out first; then the supervisor reloads us.
        loop.call_later(1.0, os._exit, 1)

    executor = ThreadPoolExecutor(1, thread_name_prefix="corpus-gpu")
    batcher = GpuBatcher(score, executor=executor,
                         on_error=_free_cuda, on_fatal=fatal)
    app = build_gpu_app(batcher, info)

    async def ready() -> None:
        # Published once the socket exists: a reader of the info can connect.
        for _ in range(200):
            if (Path(run_dir) / GPU_SOCKET).exists():
                break
            await asyncio.sleep(0.05)
        write_info(run_dir, info)
        logger.info("corpus gpu: ready (vocab %d)", info["vocab_size"])

    try:
        await serve_unix(app, Path(run_dir) / GPU_SOCKET, services=[batcher.run(), ready()])
    finally:
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)


class GpuScorer:
    """The auditor's ``scorer``: rows to the GPU process, scores back.

    A GPU process that is down, restarting or full is waited for (retried
    forever): audits wait, they never fail on our side for it. An error the
    forward itself raised comes back as the in-process forward would raise it.
    """

    def __init__(self, socket_path: str | Path, *, chunk_tokens: int, topk: int,
                 executor=None, retry_seconds: float = 1.0, max_retry_seconds: float = 10.0,
                 sleep=asyncio.sleep) -> None:
        self._path = str(socket_path)
        self._chunk_tokens, self._topk = chunk_tokens, topk
        self._executor = executor
        self._retry, self._max_retry = retry_seconds, max_retry_seconds
        self._sleep = sleep
        self._client = None
        self._loop = None
        self.retries = 0

    def _http(self):
        import httpx

        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=self._path),
                base_url="http://corpus-gpu", timeout=httpx.Timeout(None, connect=5.0))
            self._loop = loop
        return self._client

    async def _reset(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass

    async def __call__(self, rows: Sequence) -> tuple[list, float, float]:
        import httpx

        from reliquary.validator.corpus_judge_threads import run_in

        if not rows:
            return [], 0.0, 0.0
        body = await run_in(self._executor, lambda: encode_request(
            rows, chunk_tokens=self._chunk_tokens, topk=self._topk))
        wait = self._retry
        while True:
            try:
                response = await self._http().post(
                    "/score", content=body,
                    headers={"content-type": "application/octet-stream"})
            except httpx.TransportError as exc:
                await self._reset()
                why = repr(exc)
            else:
                if response.status_code == 200:
                    doc = await run_in(self._executor, response.json)
                    if "error" in doc:
                        message = f"{doc.get('kind')}: {doc['error']}"
                        logger.error("corpus gpu process failed a forward: %s", message[:500])
                        if doc.get("value_error"):
                            raise ValueError(message)
                        if doc.get("runtime_error"):
                            raise RuntimeError(message)
                        raise GpuScoreError(message)
                    return scores_from_wire(doc["scores"]), doc["forward_seconds"], \
                        doc["verify_seconds"]
                if response.status_code != 503:
                    # Not unavailability: a fault of ours, raised like a crash.
                    raise GpuScoreError(f"HTTP {response.status_code}: {response.text[:200]}")
                why = "busy"
            self.retries += 1
            logger.warning("corpus gpu process unavailable (%s); audits wait, retrying in %.1f s",
                           why, wait)
            await self._sleep(wait)
            wait = min(wait * 2, self._max_retry)


__all__ = [
    "GPU_INFO",
    "GPU_SOCKET",
    "GpuBatcher",
    "GpuBusy",
    "GpuScoreError",
    "GpuScorer",
    "build_gpu_app",
    "decode_request",
    "encode_request",
    "read_info",
    "run_gpu_process",
    "scores_from_wire",
    "scores_to_wire",
    "serve_unix",
    "write_info",
]
