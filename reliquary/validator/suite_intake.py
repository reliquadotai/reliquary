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
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

# A telecom replay takes ~30 ms; this bounds a pathological one. The clock
# starts when a worker starts the replay, never while it is queued.
SUITE_REPLAY_DEADLINE_SECONDS = 30.0
# How long a caller waits for a replay, queue included, before giving up on
# the pool for now (``ReplayUnavailable``: retryable, never a refusal).
SUITE_REPLAY_WAIT_SECONDS = 180.0
SUITE_REPLAY_WORKERS = int(os.environ.get("RELIQUARY_CORPUS_SUITE_REPLAY_WORKERS", "2"))
# Address space per replay worker: an environment that balloons dies alone
# (MemoryError, or the worker, which the pool then replaces).
SUITE_REPLAY_WORKER_MEMORY_BYTES = int(os.environ.get(
    "RELIQUARY_CORPUS_SUITE_REPLAY_WORKER_MEMORY_BYTES", str(2 << 30)))
# Intake checks waiting on the pool at once, on threads of their own: never
# the loop's default executor, which the corpus ledger turns run on.
SUITE_INTAKE_THREADS = int(os.environ.get("RELIQUARY_CORPUS_SUITE_INTAKE_THREADS", "8"))
DEADLINE_REFUSAL = ("bad_observation", {"why": "the replay ran past its deadline"})


class _ReplayTimeout(TimeoutError):
    pass


_WORKER_TOKENIZER: Any | None = None
# Set by the timer itself. An environment may catch any exception its tools
# raise (telecom turns one into an "Error: ..." observation), which would
# swallow the timer's: the flag is what says the deadline passed.
_DEADLINE_HIT = False


PARENT_WATCH_SECONDS = 1.0


def _exit_with_parent(parent: int, every: float = PARENT_WATCH_SECONDS) -> threading.Thread:
    """A worker leaves when the validator that spawned it is gone, however it
    went. ``close`` kills the workers on a clean shutdown; this covers the
    rest (a SIGKILL, a crash before cleanup): an idle worker blocks on its
    call queue forever and would otherwise outlive the validator."""
    def watch() -> None:
        while True:
            time.sleep(every)
            if os.getppid() != parent:
                os._exit(0)

    thread = threading.Thread(target=watch, name="suite-replay-parent-watch", daemon=True)
    thread.start()
    return thread


def initialize_suite_replay_worker(tokenizer_json: str,
                                   memory_bytes: int = SUITE_REPLAY_WORKER_MEMORY_BYTES) -> None:
    """CPU-only, below the control plane and memory-bounded, as
    ``initialize_admission_worker`` (not imported: that module loads torch,
    which a replay never needs)."""
    global _WORKER_TOKENIZER
    _exit_with_parent(os.getppid())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        os.nice(5)
    except OSError:
        pass
    if memory_bytes > 0:
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (OSError, ValueError):
            logger.warning("suite replay worker: RLIMIT_AS not set", exc_info=True)
    from tokenizers import Tokenizer

    _WORKER_TOKENIZER = Tokenizer.from_str(tokenizer_json)


@contextmanager
def _deadline(seconds: float):
    """Admission's ``_deadline``: SIGALRM in the worker's main thread, armed
    when the worker starts the replay; it also sets ``_DEADLINE_HIT``."""
    global _DEADLINE_HIT
    _DEADLINE_HIT = False

    def _raise(_signum, _frame):
        global _DEADLINE_HIT
        _DEADLINE_HIT = True
        raise _ReplayTimeout("suite replay deadline exceeded")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, max(float(seconds), 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _replay(env_name: str, encode, decode, *, split: str = "train",
            **kwargs) -> SuiteReplayOutcome:
    """One replay on ``env_name`` built at ``split`` (``replay_kwargs`` names
    the job's; a caller that names none replays on ``train``, as before)."""
    from reliquary.environment.registry import get_environment_spec

    return replay_outcome(get_environment_spec(env_name).create(split=split), encode=encode,
                          decode=decode, **kwargs)


def replay_in_worker(env_name: str, deadline_seconds: float, **kwargs) -> SuiteReplayOutcome:
    """Top-level (picklable): one replay in a spawned worker. Past the
    deadline the outcome is the deadline's, whatever the replay returned
    after an environment swallowed the timer."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("suite replay worker has no tokenizer")
    codec = TokenizerCodec(_WORKER_TOKENIZER)
    try:
        with _deadline(deadline_seconds):
            outcome = _replay(env_name, codec.encode, codec.decode, **kwargs)
    except _ReplayTimeout:
        outcome = None
    if _DEADLINE_HIT or outcome is None:
        return SuiteReplayOutcome(DEADLINE_REFUSAL, deadline=True)
    return outcome


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

    @staticmethod
    def _kill(pool: ProcessPoolExecutor) -> None:
        """Kill ``pool``'s workers and shut it down without waiting: a busy
        or wedged worker would otherwise run on, past the pool and past the
        validator."""
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                process.kill()
            except (OSError, AttributeError):
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    def _drop(self, pool: ProcessPoolExecutor, why: str) -> None:
        """Retire ``pool`` (once, whoever asks first) and kill its workers: a
        wedged one never frees its slot otherwise. The next call builds anew."""
        with self._lock:
            if self._pool is not pool:
                return                      # another caller rebuilt it already
            self._pool = None
            self.rebuilds += 1
        logger.error("SUITE REPLAY POOL DROPPED (%s); its workers are killed and the pool "
                     "rebuilt (rebuild %d)", why, self.rebuilds)
        self._kill(pool)

    def _settled(self, pool, outcome: SuiteReplayOutcome) -> SuiteReplayOutcome:
        if outcome.deadline:
            # The worker hit its deadline: whatever state the timer left it
            # in, it is replaced.
            self._drop(pool, "a replay ran past its deadline")
        return outcome

    def _timed_out(self, pool, future) -> ReplayUnavailable:
        if future.running():
            self._drop(pool, f"a started replay gave no answer within {self._wait:.0f} s")
        else:
            future.cancel()
        return ReplayUnavailable(f"no replay answer within {self._wait:.0f} s")

    def replay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        """Blocks the calling thread (never call it on the event loop)."""
        for attempt in range(2):
            pool = self._executor()
            try:
                future = pool.submit(replay_in_worker, env_name, self._deadline, **kwargs)
                return self._settled(pool, future.result(timeout=self._wait))
            except BrokenProcessPool as exc:
                self._drop(pool, f"{type(exc).__name__}: {exc}: a worker died")
            except FutureTimeout:
                raise self._timed_out(pool, future) from None
        raise ReplayUnavailable("the replay pool broke twice in a row")

    async def areplay(self, env_name: str, **kwargs) -> SuiteReplayOutcome:
        for attempt in range(2):
            pool = self._executor()
            try:
                future = pool.submit(replay_in_worker, env_name, self._deadline, **kwargs)
                outcome = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)),
                                                 self._wait)
                return self._settled(pool, outcome)
            except BrokenProcessPool as exc:
                self._drop(pool, f"{type(exc).__name__}: {exc}: a worker died")
            except asyncio.TimeoutError:
                raise self._timed_out(pool, future) from None
        raise ReplayUnavailable("the replay pool broke twice in a row")

    def close(self) -> None:
        """At validator shutdown: the workers are killed, not left to finish
        (a restart must not leak them)."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            self._kill(pool)


def replay_kwargs(job, source, *, prompt_index: int, prompt_ids, tokens, spans,
                  stop: str, with_split: bool = True) -> dict:
    """What one replay of ``job``'s trajectory needs, besides the codec: the
    task row ``prompt_index`` plays through ``source``'s row map, and (unless
    ``with_split`` is off: a replay on the source's own environment, built at
    it already) the split the replay builds the environment at. The seed stays
    the row's (``episode_seed(job_id, prompt_index)``): only the task lookup is
    mapped."""
    episode = job.episode
    kwargs = {"renderer_id": episode.renderer, "max_turns": episode.max_turns,
              "task_index": source.rows.index(int(prompt_index)),
              "seed": episode_seed(job.job_id, prompt_index),
              "prompt_ids": [int(t) for t in prompt_ids], "tokens": [int(t) for t in tokens],
              "spans": [(int(s), int(e)) for s, e in spans], "stop": stop}
    if with_split:
        kwargs["split"] = source.rows.split
    return kwargs


_INTAKE_THREADS: ThreadPoolExecutor | None = None
_INTAKE_THREADS_LOCK = threading.Lock()


def _intake_threads() -> ThreadPoolExecutor:
    """One bounded pool for every suite intake of the process."""
    global _INTAKE_THREADS
    with _INTAKE_THREADS_LOCK:
        if _INTAKE_THREADS is None:
            _INTAKE_THREADS = ThreadPoolExecutor(SUITE_INTAKE_THREADS,
                                                 thread_name_prefix="corpus-suite-intake")
        return _INTAKE_THREADS


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
        # The route runs `check` here (it waits on the replay pool): bounded,
        # and never the loop's default executor, which the ledger turns use.
        self.executor = _intake_threads()

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
        if getattr(trajectory, "transcript", None) is not None:
            # Only a signed-sandbox job carries a transcript: a suite record that stored
            # one would read as signed wherever the record alone decides.
            return IntakeRefusal("malformed_submission", {
                "transcript": "a suite job takes no sandbox transcript"})
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
            self._job, self._source, prompt_index=request.prompt_index, prompt_ids=prompt_ids,
            tokens=tokens, spans=spans, stop=trajectory.stop))
        if outcome.deadline:
            # ~30 ms of work past 30 s on a worker's own clock: the host is
            # starved, nothing is known against the trajectory (a 503).
            raise ReplayUnavailable("the replay ran past its deadline")
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
    from reliquary.corpus.suite_episode import job_suite_source
    from reliquary.eval.rows import job_row_map
    from reliquary.eval.sets import refuse_held_out_overlap

    try:
        # `jobs create` checks it too, but a manifest can be written without
        # it: a served job never takes held-out rows or the RL share. An eval
        # job's rows are its set's references, not a train range. Checked
        # before the environment is built (the map is the one the source uses).
        if job_row_map(job).is_identity:
            refuse_held_out_overlap(job.prompt_source, job.prompt_start, job.prompt_count)
        source = job_suite_source(job)
    except ValueError as exc:
        raise ValueError(f"job {job.job_id!r}: {exc}") from exc
    if len(source) < job.prompt_end:
        raise ValueError(f"job {job.job_id!r} owns rows up to {job.prompt_end}, environment "
                         f"{job.episode.env.name!r} has {len(source)}")
    return SuiteEpisodeIntake(job=job, source=source, tokenizer=tokenizer, vocab_size=vocab_size,
                              chunk_tokens=chunk_tokens, replayer=replayer)


__all__ = ["InlineSuiteReplayer", "PoolSuiteReplayer", "SUITE_REPLAY_DEADLINE_SECONDS",
           "SuiteEpisodeIntake", "build_suite_intake", "replay_in_worker", "replay_kwargs"]
