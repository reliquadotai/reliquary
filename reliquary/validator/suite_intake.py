"""The intake of a suite episode trajectory (design 2026-10-13): the generic
turn checks, then the replay RL admission runs, in a process pool under a
wall deadline.

A SWE trajectory is parsed here and replayed later by grade executors; a suite
one is replayed at intake, because the replay IS the check that the
observations are the environment's (the parse a SWE job does against its
renderer). It runs in spawned CPU-only workers, as admission's replays do:
never on the event loop, and never in the route's threads, which only wait.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import multiprocessing
import os
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any

from reliquary.corpus.checks import (
    check_short_turns,
    check_turn_budget,
    check_turn_proof_shape,
    check_turn_spans,
    check_turn_termination,
    completion_digest,
)
from reliquary.corpus.suite_episode import (
    SUITE_STOPS,
    SuiteReplayOutcome,
    TokenizerCodec,
    episode_seed,
    replay_outcome,
    suite_span_refusal,
    turn_ids,
)
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
from reliquary.validator.agentic_intake import IntakeFacts, IntakeRefusal, ReplayUnavailable
from reliquary.validator.corpus_text import REASON_PROMPT_MISMATCH, REASON_TOKEN_OUT_OF_VOCAB

logger = logging.getLogger(__name__)

# A telecom replay takes milliseconds; this bounds a pathological one. The
# clock starts when a worker starts the replay, never while it is queued.
SUITE_REPLAY_DEADLINE_SECONDS = 30.0
# How long a caller waits for a replay, queue included, before giving up on
# the pool for now (``ReplayUnavailable``: retryable, never a refusal).
SUITE_REPLAY_WAIT_SECONDS = 180.0
SUITE_REPLAY_WORKERS = int(os.environ.get("RELIQUARY_CORPUS_SUITE_REPLAY_WORKERS", "2"))
DEADLINE_REFUSAL = ("bad_observation", {"why": "the replay ran past its deadline"})


class _ReplayTimeout(TimeoutError):
    pass


_WORKER_TOKENIZER: Any | None = None


def initialize_suite_replay_worker(tokenizer_json: str) -> None:
    """CPU-only and below the control plane, as ``initialize_admission_worker``
    (not imported: that module loads torch, which a replay never needs)."""
    global _WORKER_TOKENIZER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        os.nice(5)
    except OSError:
        pass
    from tokenizers import Tokenizer

    _WORKER_TOKENIZER = Tokenizer.from_str(tokenizer_json)


@contextmanager
def _deadline(seconds: float):
    """Admission's ``_deadline``: SIGALRM in the worker's main thread, armed
    when the worker starts the replay."""
    def _raise(_signum, _frame):
        raise _ReplayTimeout("suite replay deadline exceeded")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, max(float(seconds), 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _replay(env_name: str, encode, decode, **kwargs) -> SuiteReplayOutcome:
    from reliquary.environment.registry import get_environment_spec

    return replay_outcome(get_environment_spec(env_name).create(), encode=encode,
                          decode=decode, **kwargs)


def replay_in_worker(env_name: str, deadline_seconds: float, **kwargs) -> SuiteReplayOutcome:
    """Top-level (picklable): one replay in a spawned worker."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("suite replay worker has no tokenizer")
    codec = TokenizerCodec(_WORKER_TOKENIZER)
    try:
        with _deadline(deadline_seconds):
            return _replay(env_name, codec.encode, codec.decode, **kwargs)
    except _ReplayTimeout:
        return SuiteReplayOutcome(DEADLINE_REFUSAL, deadline=True)


class InlineSuiteReplayer:
    """In this process, with the caller's codec: tests, and the export."""

    workers = 2

    def __init__(self, *, encode, decode) -> None:
        self._encode, self._decode = encode, decode

    def replay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        return _replay(env_name, self._encode, self._decode, **kwargs)

    async def areplay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        return await asyncio.to_thread(self.replay, env_name, **kwargs)

    def close(self) -> None:
        pass


class PoolSuiteReplayer:
    """Replays in spawned workers that hold the checkpoint's tokenizer, one
    pool for every suite job of the process (built on first use).

    A worker that dies (OOM, a crash in the environment) breaks a
    ``ProcessPoolExecutor`` for good: every later submit raises
    ``BrokenProcessPool``. So a broken pool is dropped and rebuilt, and the
    replay retried once on the new one; a second break, or no answer within
    ``wait_seconds``, is ``ReplayUnavailable`` (the route answers 503, the
    grader retries at its next rescan), never a verdict on the trajectory."""

    def __init__(self, tokenizer_json: str, *, workers: int = SUITE_REPLAY_WORKERS,
                 deadline_seconds: float = SUITE_REPLAY_DEADLINE_SECONDS,
                 wait_seconds: float = SUITE_REPLAY_WAIT_SECONDS) -> None:
        self._tokenizer_json = tokenizer_json
        self._workers = max(1, int(workers))
        self._deadline = float(deadline_seconds)
        self._wait = float(wait_seconds)
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self.rebuilds = 0

    @property
    def workers(self) -> int:
        return self._workers

    @classmethod
    def for_tokenizer(cls, tokenizer, **kwargs) -> "PoolSuiteReplayer":
        serializer = getattr(getattr(tokenizer, "backend_tokenizer", None), "to_str", None)
        if not callable(serializer):
            raise RuntimeError("a suite episode job needs a fast tokenizer to replay with")
        return cls(str(serializer()), **kwargs)

    def _executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._workers, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_suite_replay_worker, initargs=(self._tokenizer_json,))
            return self._pool

    def _drop(self, pool: ProcessPoolExecutor, exc: BaseException) -> None:
        with self._lock:
            if self._pool is not pool:
                return                      # another caller rebuilt it already
            self._pool = None
            self.rebuilds += 1
        logger.error("SUITE REPLAY POOL BROKEN (%s: %s): a worker died; rebuilding it "
                     "(rebuild %d)", type(exc).__name__, exc, self.rebuilds)
        pool.shutdown(wait=False, cancel_futures=True)

    def replay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        """Blocks the calling thread (never call it on the event loop)."""
        for attempt in range(2):
            pool = self._executor()
            try:
                future = pool.submit(replay_in_worker, env_name, self._deadline, **kwargs)
                return future.result(timeout=self._wait)
            except BrokenProcessPool as exc:
                self._drop(pool, exc)
            except FutureTimeout:
                future.cancel()
                raise ReplayUnavailable(f"no replay answer within {self._wait:.0f} s") from None
        raise ReplayUnavailable("the replay pool broke twice in a row")

    async def areplay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        for attempt in range(2):
            pool = self._executor()
            try:
                future = asyncio.wrap_future(
                    pool.submit(replay_in_worker, env_name, self._deadline, **kwargs))
                return await asyncio.wait_for(future, self._wait)
            except BrokenProcessPool as exc:
                self._drop(pool, exc)
            except asyncio.TimeoutError:
                raise ReplayUnavailable(f"no replay answer within {self._wait:.0f} s") from None
        raise ReplayUnavailable("the replay pool broke twice in a row")

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = None


def replay_kwargs(job, *, prompt_index: int, prompt_ids, tokens, spans, stop: str) -> dict:
    """What one replay of ``job``'s trajectory needs, besides the codec."""
    episode = job.episode
    return {"renderer_id": episode.renderer, "max_turns": episode.max_turns,
            "task_index": int(prompt_index), "seed": episode_seed(job.job_id, prompt_index),
            "prompt_ids": [int(t) for t in prompt_ids], "tokens": [int(t) for t in tokens],
            "spans": [(int(s), int(e)) for s, e in spans], "stop": stop}


class SuiteEpisodeIntake:
    """The suite twin of ``EpisodeIntake``: the same interface for the route
    (``check``, ``source``), the replay in place of the parse."""

    def __init__(self, *, job, source, tokenizer, vocab_size: int | None, chunk_tokens: int,
                 replayer, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        self._job = job
        self._source = source
        self._codec = TokenizerCodec(tokenizer)
        self._vocab_size = vocab_size
        self._chunk_tokens = chunk_tokens
        self._min_chunk = min_chunk_tokens
        self._replayer = replayer
        # The replay workers' coding (the codec locks: intake runs on threads).
        self._renderer = source.renderer(self._codec.encode)
        self.terminator_id, self.stop_ids = turn_ids(self._codec.encode)
        self.initial_ids = functools.lru_cache(maxsize=4096)(self._initial_ids)

    @property
    def source(self):
        return self._source

    @property
    def replayer(self):
        return self._replayer

    @property
    def renderer(self):
        return self._renderer

    def _initial_ids(self, prompt_index: int) -> tuple[int, ...]:
        return tuple(self._renderer.encode_initial(self._source.task(prompt_index)))

    def check(self, request) -> IntakeFacts | IntakeRefusal:
        trajectory = request.trajectory
        if trajectory is None or request.completions:
            return IntakeRefusal("malformed_submission", {
                "why": "an episode job takes one trajectory, not completions",
                "trajectory": trajectory is not None, "completions": len(request.completions)})
        if not self._job.owns(request.prompt_index):
            return IntakeRefusal("prompt_mismatch", {"got": request.prompt_index})
        if trajectory.final_diff != "":
            return IntakeRefusal("malformed_submission",
                                 {"why": "a suite episode has no diff: final_diff must be ''"})
        if trajectory.stop not in SUITE_STOPS:
            return IntakeRefusal("bad_stop", {"stop": trajectory.stop, "allowed": list(SUITE_STOPS)})
        expected = self._source.initial_text(request.prompt_index)
        if expected != request.rendered_prompt:
            return IntakeRefusal(REASON_PROMPT_MISMATCH, {
                "prompt_index": request.prompt_index, "expected_chars": len(expected),
                "rendered_chars": len(request.rendered_prompt)})
        prompt_ids = self.initial_ids(request.prompt_index)
        tokens = trajectory.tokens
        if self._vocab_size is not None and max(tokens) >= self._vocab_size:
            return IntakeRefusal(REASON_TOKEN_OUT_OF_VOCAB, {"vocab_size": self._vocab_size})
        spans = [(turn.start, turn.end) for turn in trajectory.turns]
        episode = self._job.episode
        refusal = check_turn_spans(spans, len(tokens), episode.max_turns)
        if not refusal.ok:
            return IntakeRefusal(refusal.reason or "", dict(refusal.detail))
        # Each span one turn: no stop before its end, no turn markup at all.
        inside = suite_span_refusal(tokens, spans, encode=self._codec.encode,
                                    decode=self._codec.decode)
        if inside is not None:
            return IntakeRefusal(inside[0], dict(inside[1]))
        # Only after the spans are known sound; cheap ones before the replay.
        checks = (
            lambda: check_short_turns(spans, self._min_chunk),
            lambda: check_turn_budget(spans, prompt_len=len(prompt_ids), length=len(tokens),
                                      max_tokens_per_turn=episode.max_tokens_per_turn,
                                      max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_termination(tokens, spans, prompt_len=len(prompt_ids),
                                           terminator_id=self.terminator_id,
                                           stop_ids=self.stop_ids,
                                           max_tokens_per_turn=episode.max_tokens_per_turn,
                                           max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_proof_shape(spans, [len(turn.proofs) for turn in trajectory.turns],
                                           self._chunk_tokens),
        )
        for check in checks:
            result = check()
            if not result.ok:
                return IntakeRefusal(result.reason or "", dict(result.detail))
        outcome = self._replayer.replay(episode.env.name, **replay_kwargs(
            self._job, prompt_index=request.prompt_index, prompt_ids=prompt_ids, tokens=tokens,
            spans=spans, stop=trajectory.stop))
        if outcome.refusal is not None:
            reason, detail = outcome.refusal
            return IntakeRefusal(reason, dict(detail))
        return IntakeFacts(prompt_ids=prompt_ids, token_count=sum(e - s for s, e in spans),
                           digest=completion_digest(request.prompt_index, tokens),
                           last_token_id=int(tokens[-1]), task_id=outcome.task_id,
                           reward=outcome.reward, success=outcome.success)


def build_suite_intake(job, *, tokenizer, vocab_size: int | None, chunk_tokens: int,
                       replayer) -> SuiteEpisodeIntake:
    """The intake a validator serves a suite job with; refuses (a ValueError,
    permanent until a restart) an environment this binary cannot serve."""
    from reliquary.corpus.suite_episode import suite_source
    from reliquary.eval.sets import refuse_held_out_overlap

    try:
        # `jobs create` checks it too, but a manifest can be written without
        # it: a served job never takes held-out rows or the RL share.
        refuse_held_out_overlap(job.prompt_source, job.prompt_start, job.prompt_count)
        source = suite_source(job.episode)
    except ValueError as exc:
        raise ValueError(f"job {job.job_id!r}: {exc}") from exc
    if len(source) < job.prompt_end:
        raise ValueError(f"job {job.job_id!r} owns rows up to {job.prompt_end}, environment "
                         f"{job.episode.env.name!r} has {len(source)}")
    return SuiteEpisodeIntake(job=job, source=source, tokenizer=tokenizer, vocab_size=vocab_size,
                              chunk_tokens=chunk_tokens, replayer=replayer)


__all__ = ["InlineSuiteReplayer", "PoolSuiteReplayer", "SUITE_REPLAY_DEADLINE_SECONDS",
           "SuiteEpisodeIntake", "build_suite_intake", "replay_in_worker", "replay_kwargs"]
