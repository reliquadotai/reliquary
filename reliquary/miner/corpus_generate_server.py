"""The agentic miner's loopback generate endpoint (spec §5 N1).

verifiers' train client renders every turn to token ids and posts them here;
one assistant turn is one request (prompt = the whole history). One engine
thread drives vLLM's in-process engine step by step, so concurrent episodes
share batches; prefix caching is on (gate M1), and each finished turn's rows
come from the hidden-state capture through ``turn_rows``. The job's sampling
is authoritative: of the client's ``sampling_params`` only ``max_tokens`` is
read, clamped to the turn and total budgets. Every turn is logged under its
session (the client's ``X-Session-ID``, the verifiers trace id) until the
miner takes the session to build its trajectory.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import queue
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request

from reliquary.corpus.trajectory import GeneratedTurn

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Session-ID"
# renderers refuses vLLM's -9999.0 sentinel and any non-finite logprob.
LOGPROB_FLOOR = -9998.0


@dataclass(frozen=True)
class Finished:
    request_id: str
    completion_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()
    finish_reason: str = "stop"
    proofs: tuple[str, ...] = ()
    error: str | None = None


class TurnCore(Protocol):
    def add(self, request_id: str, prompt_ids: list[int], max_tokens: int) -> None: ...

    def step(self) -> list[Finished]: ...

    def has_unfinished(self) -> bool: ...

    # Optional: ``abort(request_id)`` stops a request nobody waits for any more.


MAX_STEP_FAILURES = 5
_DROPPED_KEPT = 4096


@dataclass
class SessionLog:
    turns: list[GeneratedTurn] = field(default_factory=list)
    linear: bool = True
    touched: float = 0.0


def _settle(future: asyncio.Future, done: Finished) -> None:
    if not future.done():
        future.set_result(done)


class GenerateEngine:
    def __init__(self, core: TurnCore, *, max_total_tokens: int, max_tokens_per_turn: int,
                 session_ttl_seconds: float = 7200.0, clock=time.monotonic) -> None:
        self._core = core
        self.max_total_tokens = int(max_total_tokens)
        self._per_turn = int(max_tokens_per_turn)
        self._ttl = session_ttl_seconds
        self._clock = clock
        self._inbox: queue.Queue = queue.Queue()
        self._pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._sessions: dict[str, SessionLog] = {}
        self._lock = threading.Lock()
        self._inflight: dict[str, str] = {}  # request id -> session id
        self._dropped: OrderedDict[str, None] = OrderedDict()
        self._step_failures = 0
        self.healthy = True
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the engine thread ---------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="generate-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._inbox.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._drain(block=not self._core.has_unfinished())
            if not self._core.has_unfinished():
                continue
            try:
                finished = self._core.step()
            except Exception as exc:  # engine.step() itself failed: every waiter learns it
                logger.exception("generate engine step failed")
                self._step_failures += 1
                if self._step_failures >= MAX_STEP_FAILURES:
                    self.healthy = False
                with self._lock:
                    waiting = list(self._pending)
                for request_id in waiting:
                    self._resolve(Finished(request_id, error=f"engine step failed: {exc}"))
                    self._abort(request_id)
                self._stopping.wait(0.05)
                continue
            self._step_failures = 0
            for done in finished:
                self._resolve(done)

    def _abort(self, request_id: str) -> None:
        abort = getattr(self._core, "abort", None)
        if abort is None:
            return
        try:
            abort(request_id)
        except Exception:
            logger.exception("aborting request %s failed", request_id)

    def _drain(self, *, block: bool) -> None:
        try:
            item = self._inbox.get(timeout=0.05) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while item is not None:
            if item[0] == "abort":
                self._abort(item[1])
            else:
                _, request_id, prompt_ids, max_tokens = item
                with self._lock:
                    waited = request_id in self._pending
                if waited:  # a waiter that already gave up is not worth a prefill
                    try:
                        self._core.add(request_id, prompt_ids, max_tokens)
                    except Exception as exc:
                        self._resolve(Finished(request_id, error=f"request refused by the engine: {exc}"))
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return

    def _resolve(self, done: Finished) -> None:
        with self._lock:
            entry = self._pending.pop(done.request_id, None)
            self._inflight.pop(done.request_id, None)
        if entry is not None:
            loop, future = entry
            loop.call_soon_threadsafe(_settle, future, done)

    # -- the event loop side ---------------------------------------------------

    async def generate(self, session_id: str, prompt_ids: list[int], max_tokens: int | None) -> dict:
        cap = min(self._per_turn, self.max_total_tokens - len(prompt_ids),
                  max_tokens if max_tokens else self._per_turn)
        if not self.healthy:
            raise RuntimeError("the generate engine is unhealthy (repeated engine step failures)")
        if cap < 1:
            raise ValueError(f"a prompt of {len(prompt_ids)} tokens leaves no room under "
                             f"{self.max_total_tokens}")
        self._note_prompt(session_id, prompt_ids)
        request_id = secrets.token_hex(16)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            self._pending[request_id] = (loop, future)
            self._inflight[request_id] = session_id
        self._inbox.put(("add", request_id, list(prompt_ids), cap))
        try:
            done: Finished = await future
        except BaseException:  # cancelled (client gone): free the engine slot
            with self._lock:
                self._pending.pop(request_id, None)
                self._inflight.pop(request_id, None)
            self._inbox.put(("abort", request_id))
            raise
        if done.error is not None:
            raise RuntimeError(done.error)
        with self._lock:
            if session_id in self._dropped:
                raise RuntimeError("the session was dropped while the turn ran")
            log = self._sessions.setdefault(session_id, SessionLog())
            log.turns.append(GeneratedTurn(tuple(prompt_ids), done.completion_ids, done.proofs))
            log.touched = self._clock()
        return {"request_id": request_id, "choices": [{
            "index": 0, "token_ids": list(done.completion_ids),
            "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": lp}
                                     for t, lp in zip(done.completion_ids, done.logprobs)]},
            "finish_reason": done.finish_reason}]}

    def _note_prompt(self, session_id: str, prompt_ids: list[int]) -> None:
        now = self._clock()
        with self._lock:
            if session_id in self._dropped:
                raise ValueError("this session was dropped")
            for stale in [s for s, log in self._sessions.items() if now - log.touched > self._ttl]:
                del self._sessions[stale]
            log = self._sessions.setdefault(session_id, SessionLog(touched=now))
            log.touched = now
            if log.turns:
                previous = log.turns[-1]
                history = list(previous.prompt_ids) + list(previous.completion_ids)
                if prompt_ids[:len(history)] != history or len(prompt_ids) <= len(history):
                    log.linear = False

    def take_session(self, session_id: str) -> SessionLog | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._dropped[session_id] = None
            while len(self._dropped) > _DROPPED_KEPT:
                self._dropped.popitem(last=False)
            running = [r for r, s in self._inflight.items() if s == session_id]
        for request_id in running:
            self._resolve(Finished(request_id, error="session dropped"))
            self._inbox.put(("abort", request_id))


def _logprob(entry: Any, token: int) -> float:
    value = getattr((entry or {}).get(token), "logprob", None)
    if value is None or not math.isfinite(value):
        return LOGPROB_FLOOR
    return max(float(value), LOGPROB_FLOOR)


class VllmTurnCore:
    """vLLM's in-process V1 engine with prefix caching and hidden-state
    capture: each finished turn comes back with its span proofs."""

    def __init__(self, checkpoint_dir: str, *, sampling, proof, stop_token_ids,
                 max_total_tokens: int, max_num_seqs: int = 16,
                 gpu_memory_utilization: float | None = None) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        from vllm import LLM, SamplingParams

        from reliquary.miner.corpus_miner import _has_vision_encoder
        from reliquary.miner.vllm_hidden_capture import capture_hidden_states

        self._capture_cm = capture_hidden_states()
        self._capture = self._capture_cm.__enter__()
        memory = ({} if gpu_memory_utilization is None
                  else {"gpu_memory_utilization": gpu_memory_utilization})
        extra = ({"limit_mm_per_prompt": {"image": 0, "video": 0}}
                 if _has_vision_encoder(checkpoint_dir) else {})
        self._llm = LLM(model=checkpoint_dir, dtype="bfloat16", enable_prefix_caching=True,
                        seed=secrets.randbelow(2**31 - 1) + 1, max_model_len=max_total_tokens,
                        max_num_seqs=max_num_seqs, **memory, **extra)
        self._engine = self._llm.llm_engine
        self._params = SamplingParams
        # The job's sampling; only the stop ids end a turn (ignore_eos turns off
        # the checkpoint's own generation_config terminators).
        self._sampling = dict(n=1, temperature=sampling.temperature, top_p=sampling.top_p,
                              top_k=sampling.top_k if sampling.top_k > 0 else -1,
                              stop_token_ids=list(stop_token_ids), ignore_eos=True,
                              logprobs=1, detokenize=False)
        self._stops = frozenset(int(t) for t in stop_token_ids)
        self._proof = proof
        self._prompt_len: dict[str, int] = {}

    def add(self, request_id: str, prompt_ids: list[int], max_tokens: int) -> None:
        from vllm.inputs import TokensPrompt

        self._engine.add_request(request_id, TokensPrompt(prompt_token_ids=prompt_ids),
                                 self._params(max_tokens=max_tokens, **self._sampling))
        self._prompt_len[request_id] = len(prompt_ids)

    def abort(self, request_id: str) -> None:
        self._prompt_len.pop(request_id, None)
        try:
            self._engine.abort_request([request_id])
        finally:
            self._sweep_capture()

    def pending_capture_count(self) -> int:
        """Captured requests still held: 0 once every batch has drained."""
        return len(self._capture.ids())

    def _sweep_capture(self) -> None:
        # Rows of requests nobody will pop: the async surplus row recorded
        # after a pop, or a request aborted / failed by an engine-step error.
        for captured in self._capture.ids():
            if self._own_id(captured) is None:
                self._capture.discard(captured)

    def has_unfinished(self) -> bool:
        return bool(self._engine.has_unfinished_requests())

    def step(self) -> list[Finished]:
        outputs = self._engine.step()
        results = []
        for output in outputs:
            if not getattr(output, "finished", False):
                continue
            engine_id = str(getattr(output, "request_id", ""))
            request_id = self._own_id(engine_id) or engine_id
            try:
                results.append(self._finish(output))
            except Exception as exc:  # one bad turn must not fail its batch-mates
                logger.exception("finishing request %s failed", engine_id)
                self._prompt_len.pop(request_id, None)
                results.append(Finished(request_id, error=f"turn cannot be finished: {exc}"))
        self._sweep_capture()
        return results

    def _own_id(self, engine_id: str) -> str | None:
        return next((k for k in self._prompt_len
                     if engine_id == k or engine_id.startswith(k + "-")), None)

    def _finish(self, output) -> Finished:
        from reliquary.miner.vllm_hidden_capture import turn_rows
        from reliquary.protocol.toploc_proof import build_span_proofs

        request_id = self._own_id(output.request_id) or output.request_id
        prompt_len = self._prompt_len.pop(request_id, None)
        completion = output.outputs[0]
        tokens = tuple(int(t) for t in completion.token_ids)
        try:
            rows = self._capture.pop(request_id)
            if prompt_len is None or not tokens:
                raise ValueError("an unknown request or an empty completion")
            rows = turn_rows(rows, len(tokens),
                             prompt_rows=prompt_len - int(output.num_cached_tokens or 0))
            proofs = build_span_proofs(rows, chunk_tokens=self._proof.chunk_tokens,
                                       topk=self._proof.topk)
        except (ValueError, KeyError) as exc:
            # A request preempted and recomputed, or rows attributed elsewhere:
            # this turn cannot be proven, so the episode cannot be submitted.
            return Finished(request_id, error=f"turn cannot be proven: {exc}")
        entries = list(completion.logprobs or [])
        logprobs = tuple(_logprob(entries[i] if i < len(entries) else None, t)
                         for i, t in enumerate(tokens))
        reason = "stop" if tokens[-1] in self._stops else "length"
        return Finished(request_id, tokens, logprobs, reason,
                        tuple(base64.b64encode(p).decode() for p in proofs))


def build_generate_app(engine: GenerateEngine, *, model_name: str) -> FastAPI:
    app = FastAPI()

    @app.post("/inference/v1/generate")
    async def generate(request: Request) -> dict:
        session = request.headers.get(SESSION_HEADER)
        if not session:
            raise HTTPException(status_code=400, detail="missing X-Session-ID: a turn must name its episode")
        body = await request.json()
        ids = body.get("token_ids")
        if not isinstance(ids, list) or not ids or not all(
                isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids):
            raise HTTPException(status_code=400, detail="token_ids must be a non-empty list of ids")
        if body.get("model") not in (None, model_name):
            raise HTTPException(status_code=404, detail=f"this endpoint serves {model_name}")
        max_tokens = (body.get("sampling_params") or {}).get("max_tokens")
        if isinstance(max_tokens, bool):
            raise HTTPException(status_code=400, detail="max_tokens must be an integer")
        try:
            return await engine.generate(session, ids,
                                         max_tokens if isinstance(max_tokens, int) else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model",
                                            "max_model_len": engine.max_total_tokens}]}

    return app


__all__ = ["Finished", "GenerateEngine", "SESSION_HEADER", "SessionLog", "TurnCore",
           "VllmTurnCore", "build_generate_app"]
