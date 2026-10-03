"""Judge every accepted submission: audit it with the job's model, wait out its
hold, or pass it unaudited, and record a verdict.

With `audit_q = 1` (the default) every record is audited on arrival, as in V0.
A submission is paid only once a passing verdict exists for it. A
validator-side error writes no verdict, so the submission stays pending instead
of being charged to a miner for our fault.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
from collections import Counter
from dataclasses import replace
import heapq
import logging
import math
import os
import re
import time
from collections.abc import Callable

import torch

from reliquary.corpus.audit_policy import (
    PASS_IDS,
    AuditParams,
    MinerState,
    after_confirmed_failure,
    after_pass,
    decision,
    drawn,
    effective_state,
)
from reliquary.corpus.encoding import prompt_token_ids
from reliquary.protocol.profiles import ProofProfile
from reliquary.validator.corpus_audit import outcome_from_scores, score_sequences
from reliquary.validator.corpus_text import REASON_TOKEN_OUT_OF_VOCAB

logger = logging.getLogger(__name__)

VERDICT_SCHEMA = "reliquary/corpus-verdict/v1"
# A pass withdrawn after the executor that scored it was quarantined.
VOIDED_SCHEMA = "reliquary/corpus-voided/v1"

RESCAN_SECONDS = 60.0
# The minute rescan requeues pending records the auditor already knows; the
# store is listed only this often (100k keys took 100-300 s on 2026-09-28).
FULL_RESCAN_SECONDS = 1800.0
MAX_CONSECUTIVE_VALIDATOR_ERRORS = 5
# Padded size (rows x longest sequence) a sub-batch's forward pass may reach.
# What fits depends on the card and checkpoint (131,072 overran an H100 beside
# Qwen3.8-27B), so the operator may lower it.
AUDIT_BATCH_TOKENS = int(os.environ.get("RELIQUARY_CORPUS_AUDIT_BATCH_TOKENS", "131072"))
# Store reads in flight at once when many records must be read before judging.
READ_CONCURRENCY = 16
# Verdict writes in flight at once. One at a time, each create-only PUT took
# ~1.5 s on 2026-10-02 and capped a job near 2,400 verdicts an hour.
WRITE_CONCURRENCY = 32
# drand rounds fetched at once when a pass needs many (one per sampled record).
DRAND_CONCURRENCY = 16
# Records read at once when a restart seeds the hold window's arrivals.
SEED_SLICE_IDS = 2048
# Metadata (tail) reads in flight at once while seeding: a few KB each.
SEED_META_CONCURRENCY = 64
# `run()` judges at most this many due ids per pass, the earliest due first.
RUN_BATCH_IDS = 512
# Records one pass audits at most (its own and its siblings'), and their
# tokens: one GPU call, then the pass's verdicts are written. What is over is
# judged in the next passes. Unbounded, the first pass after a 110k restart
# backlog audited every drawn sibling in one call and wrote nothing for an
# hour (2026-10-02 20:05).
PASS_AUDIT_ROWS = int(os.environ.get("RELIQUARY_CORPUS_PASS_AUDIT_ROWS", "512"))
PASS_AUDIT_TOKENS = int(os.environ.get("RELIQUARY_CORPUS_PASS_AUDIT_TOKENS", "2000000"))
# Siblings one pass decides for its payable records (each may need a drand
# round): the oldest payable records first, the rest wait for the next passes.
# Unbounded, the first pass after a 130k restart decided thousands of siblings
# over ~1.5k rounds and never finished (2026-10-02 23:02).
PASS_SIBLINGS = int(os.environ.get("RELIQUARY_CORPUS_PASS_SIBLINGS", "2048"))
# How soon a record whose draw round is not out yet is judged again: one
# quicknet period. Every arrival of those seconds then shares one pass.
UNDECIDABLE_RETRY_SECONDS = 3.0
# Past the instant a hotkey's last 1/q arrivals leave the hold window, so a
# waiting record is judged again once the slow-hotkey rule may audit it.
WINDOW_EPSILON_SECONDS = 1e-3
# Propagation slack after the draw round's publication before it is fetched:
# asking too early reads as "no beacon", which audits (safe, but wastes the sampling).
BEACON_GRACE_SECONDS = 2.0
# How long a round that just failed to fetch is left unfetched before the next
# attempt: every sampled submission whose draw lands on a bad round would
# otherwise refetch it once per judging pass.
NEGATIVE_BEACON_CACHE_SECONDS = 30.0
# The same for a round older than OLD_ROUND_ROUNDS: relays that fail on an old
# round (2026-10-02: "All relays/paths failed" for rounds of a 15 h backlog)
# keep failing, and every pass that races it again waits out the whole race.
OLD_ROUND_NEGATIVE_CACHE_SECONDS = 1800.0
OLD_ROUND_ROUNDS = 200
# The route stamps received_at before its record write, which it tries
# RECORD_WRITE_ATTEMPTS (3) times, each up to 3 botocore attempts of 15 s
# connect + 30 s read: 405 s. An unaudited pass waits this long past the hold,
# so every sibling received inside that hold is visible before it is paid.
ACCEPT_SLACK_SECONDS = 420.0
# How much of the accept slack is margin over the route's record-write bound
# (420 - 405): a sibling received inside a record's hold was handed over by
# its receipt + hold + (slack - this margin).
FEED_MARGIN_SECONDS = 15.0
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WORST_ZERO = {"worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0}
_BANNED_VOID = {"passed": False, "audited": False, "reason": "banned"}


class CorpusAuditorHalted(Exception):
    """Too many audits in a row failed on our side: this validator cannot audit."""


class CorpusAuditor:
    def __init__(self, *, job_id: str, records, model, tokenizer, proof: ProofProfile,
                 rescan_every_seconds: float = RESCAN_SECONDS,
                 max_validator_errors: int = MAX_CONSECUTIVE_VALIDATOR_ERRORS,
                 params: AuditParams = AuditParams(), miner_states=None,
                 beacon: Callable[[int], str | None] | None = None,
                 round_at: Callable[[float], int] | None = None,
                 clock: Callable[[], float] = time.time,
                 accept_slack_seconds: float = ACCEPT_SLACK_SECONDS,
                 gpu_lock: asyncio.Lock | None = None,
                 on_verdict: Callable[[str, dict], None] | None = None,
                 remote=None,
                 on_voided: Callable[[str, dict], None] | None = None,
                 threads=None,
                 scorer: Callable | None = None,
                 vocab_size: int | None = None,
                 arrivals_covered: Callable[[], float | None] | None = None) -> None:
        self._job_id = job_id
        # In a judge process: ``await scorer(rows)`` scores (tokens,
        # prompt_len, proofs) rows on the GPU process, as ``score_sequences``
        # would here, and ``vocab_size`` stands for the model's embeddings.
        self._scorer = scorer
        self._vocab_size = vocab_size
        # In a judge process: the instant up to which every arrival the front
        # accepted has been enqueued here (None: not known yet). A record is
        # passed unaudited only once it covers all the siblings that could
        # catch it (`_covers`); otherwise it waits.
        self._arrivals_covered = arrivals_covered
        # A `JudgeThreads`: drand races, forwards and record preparation run
        # on these, never on the loop's default executor the route needs.
        self._threads = threads
        # A `RemoteAuditDispatcher`: used while an executor is connected.
        self._remote = remote
        # Per executor, the passes written from its scores: re-audited here if
        # it is ever quarantined.
        self._remote_scored: dict[str, list[str]] = {}
        if remote is not None and callable(getattr(remote, "subscribe", None)):
            remote.subscribe(self.reaudit_executor)
        # Told of every verdict that stands, for the job's in-memory status.
        self._on_verdict = on_verdict
        # Told of every pass voided after its executor's quarantine.
        self._on_voided = on_voided
        # Shared by every job's auditor on one loaded model: one forward pass
        # at a time, and asyncio.Lock wakes waiters FIFO so no job starves.
        self._gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self._records = records
        self._model = model
        self._tokenizer = tokenizer
        self._proof = proof
        # Scheduled or in flight: a rescan must not audit the same id twice at once.
        self._queued: set[str] = set()
        # Each scheduled id's next judging time, and a heap of (time, received,
        # id) over them: a pass takes only what is due, earliest (then oldest)
        # first, so a record waiting out its hold is not re-judged every minute.
        # A heap entry whose time is not `_due[id]` is stale and skipped.
        self._due: dict[str, float] = {}
        self._heap: list[tuple[float, float, str]] = []
        self._wake: asyncio.Event | None = None
        # Set by a pass for the ids it leaves pending: when to judge them again.
        self._next_due: dict[str, float] = {}
        # Write errors of the current judge_many, raised after its backward audit.
        self._deferred: list[BaseException] = []
        # Pending ids whose draw is known and undrawn: as a sibling they can
        # neither be audited nor be undecidable, so a pass need not decide them.
        self._undrawn: set[str] = set()
        # The fewest arrivals in a hold that keep a hotkey off the slow-hotkey rule.
        self._enough = self._fewest_not_slow(params.q)
        self._rescan_every = rescan_every_seconds
        self._max_validator_errors = max_validator_errors
        self._validator_errors = 0
        self._params = params
        self._miner_states = miner_states
        self._beacon = beacon
        self._round_at = round_at
        self._clock = clock
        self._accept_slack = accept_slack_seconds
        # Records are immutable: (hotkey, received_at, token_count) read once,
        # so a rescan every minute does not re-read every record from the store.
        self._meta: dict[str, tuple[str, float, int]] = {}
        # Per hotkey, the sorted arrival times still inside the hold window;
        # `recent_submissions` is counted from here, never from a job listing.
        self._arrivals: dict[str, list[float]] = {}
        # Pending records are read once per process to seed `_arrivals`; judged
        # ones are not re-read, so `recent` can only undercount (more audits).
        self._seeded = False
        self._randomness: dict[int, str] = {}
        # Rounds raced in the current judge_many: one race per round per pass,
        # never one per record whose draw lands on a failing round.
        self._raced: set[int] = set()
        # Round -> when its fetch last failed; a negative cache, so a bad
        # round is retried at most once every NEGATIVE_BEACON_CACHE_SECONDS.
        self._failed_rounds: dict[int, float] = {}
        # Ids whose verdict stands (written, found, or listed): never judged again.
        self._judged: set[str] = set()
        # Per hotkey, the ids read but not judged yet: the siblings an unaudited
        # pass must wait for (queue lag can exceed the hold).
        self._unjudged: dict[str, set[str]] = {}
        # Pending ids whose read failed, until one succeeds or a verdict stands:
        # their hotkey is unknown, so every unaudited pass waits for them.
        self._unreadable: set[str] = set()
        # Seconds per phase and decisions of the current judge pass, logged
        # once per pass so an idle GPU shows what it was waiting for.
        self._phase: Counter = Counter()
        self._choices: Counter = Counter()

    async def _in(self, pool: str, func, *args):
        """``func`` on the judges' ``pool`` threads, or the default executor."""
        from reliquary.validator.corpus_judge_threads import run_in

        return await run_in(getattr(self._threads, pool, None), func, *args)

    @contextlib.contextmanager
    def _timed(self, phase: str):
        start = time.monotonic()
        try:
            yield
        finally:
            self._phase[phase] += time.monotonic() - start

    @staticmethod
    def _fewest_not_slow(q: float) -> int:
        # decision(): a hotkey is slow while recent_submissions < 1 / q.
        n = max(0, math.ceil(1.0 / q))
        while n > 0 and n - 1 >= 1.0 / q:
            n -= 1
        while n < 1.0 / q:
            n += 1
        return n

    def enqueue(self, submission_id: str) -> None:
        """Judge ``submission_id`` now, unless it is already scheduled."""
        if submission_id in self._queued or submission_id in self._judged:
            return
        self._schedule(submission_id, self._clock())

    def _schedule(self, submission_id: str, at: float) -> None:
        self._queued.add(submission_id)
        self._due[submission_id] = at
        received = self._meta[submission_id][1] if submission_id in self._meta else at
        heapq.heappush(self._heap, (at, received, submission_id))
        if len(self._heap) > 4 * len(self._due) + 1024:
            self._heap = [(t, r, s) for t, r, s in self._heap if self._due.get(s) == t]
            heapq.heapify(self._heap)
        if self._wake is not None:
            self._wake.set()

    def _take_due(self, now: float, limit: int) -> list[str]:
        """Up to ``limit`` scheduled ids due by ``now``; they stay in `_queued`."""
        batch: list[str] = []
        while self._heap and self._heap[0][0] <= now and len(batch) < limit:
            at, _, submission_id = heapq.heappop(self._heap)
            if self._due.get(submission_id) != at:
                continue
            del self._due[submission_id]
            if submission_id in self._judged:
                self._queued.discard(submission_id)
                continue
            batch.append(submission_id)
        return batch

    def _reschedule(self, batch: list[str]) -> None:
        """After a pass: every id it left pending is due again when the pass
        said, or at the next rescan (a read or validator error)."""
        retry = self._clock() + self._rescan_every
        for submission_id in batch:
            self._queued.discard(submission_id)
            if submission_id not in self._judged:
                self._schedule(submission_id, self._next_due.get(submission_id, retry))
        self._next_due.clear()

    def _covered(self) -> float | None:
        if self._arrivals_covered is None:
            return math.inf
        try:
            return self._arrivals_covered()
        except Exception:
            logger.exception("corpus arrival feed state unreadable; holding unaudited passes")
            return None

    def _covers(self, covered: float | None, received_at: float) -> bool:
        """Every sibling received inside this record's hold has been handed
        over: each was accepted by receipt + hold + (slack - margin)."""
        if covered is None:
            return False
        return covered >= (received_at + self._params.hold_seconds
                           + self._accept_slack - FEED_MARGIN_SECONDS)

    def _wait_until(self, submission_id: str, now: float) -> float:
        """When a record decided "wait" at ``now`` could be decided otherwise
        with nothing else happening: its hold (and slack) ends, or its hotkey's
        arrivals inside the hold fall below 1/q. A failure of its hotkey is
        acted on in the pass that confirms it (the backward audit)."""
        hotkey, received_at, _ = self._meta[submission_id]
        at = received_at + self._params.hold_seconds + self._accept_slack
        return max(min(at, self._window_ends(hotkey, now)), now)

    def _retry_at(self, hotkey: str, now: float) -> float:
        """A record left undecided this pass (its or a sibling's draw round
        not out yet): one drand period on, or sooner if its hotkey turns slow."""
        return max(min(now + UNDECIDABLE_RETRY_SECONDS, self._window_ends(hotkey, now)), now)

    def _window_ends(self, hotkey: str, now: float) -> float:
        """When ``hotkey``'s arrivals inside the hold, as counted at ``now``,
        fall below 1/q if no other arrives."""
        arrivals = self._arrivals.get(hotkey, [])
        seen = bisect.bisect_right(arrivals, now)
        if not self._enough or seen < self._enough:
            return math.inf
        return arrivals[seen - self._enough] + self._params.hold_seconds + WINDOW_EPSILON_SECONDS

    async def pending_ids(self) -> list[str]:
        with self._timed("list"):
            submitted = await self._records.list_submission_ids(self._job_id)
            judged = set(await self._records.list_verdict_ids(self._job_id))
        for sid in judged - self._judged:
            self._mark_judged(sid)
        return [sid for sid in submitted if sid not in judged]

    def pending_count(self, hotkey: str) -> int:
        """Records of ``hotkey`` read and awaiting a verdict."""
        return len(self._unjudged.get(hotkey, ()))

    def _mark_judged(self, submission_id: str) -> None:
        self._judged.add(submission_id)
        self._unreadable.discard(submission_id)
        self._undrawn.discard(submission_id)
        if submission_id in self._meta:
            self._unjudged.get(self._meta[submission_id][0], set()).discard(submission_id)

    def queue_lag(self, pending: list[str]) -> float | None:
        """Seconds since the oldest pending record we have read was received."""
        times = [self._meta[sid][1] for sid in pending if sid in self._meta]
        return self._clock() - min(times) if times else None

    def _prepare(self, records: list[dict]) -> tuple[list[dict | None], list[tuple]]:
        """The records failed before any forward pass, and every completion the
        rest need scored as ``(record, completion, tokens, prompt_len, proofs)``."""
        worst_zero = _WORST_ZERO
        results: list[dict | None] = [None] * len(records)
        vocabulary = (self._vocab_size if self._vocab_size is not None
                      else self._model.get_input_embeddings().num_embeddings)
        items = []
        for i, record in enumerate(records):
            if not record["completions"]:
                # Fail closed like sequence_verdict does for an empty chunk sequence:
                # no completions must never read as a vacuous pass paid like honest work.
                results[i] = {"passed": False, "reason": "no_completions", **worst_zero}
                continue
            # The miner's fault, not ours: checked before the prefill so it becomes
            # a failed verdict instead of a validator-side error that halts the
            # auditor.
            out_of_vocab = any(
                completion["tokens"]
                and (min(completion["tokens"]) < 0 or max(completion["tokens"]) >= vocabulary)
                for completion in record["completions"]
            )
            if out_of_vocab:
                results[i] = {"passed": False, "reason": REASON_TOKEN_OUT_OF_VOCAB, **worst_zero}
                continue
            prompt = prompt_token_ids(self._tokenizer, record["rendered_prompt"])
            for c_idx, completion in enumerate(record["completions"]):
                items.append((i, c_idx, prompt + list(completion["tokens"]), len(prompt),
                              completion["proofs"]))
        return results, items

    def _aggregate(self, records: list[dict], results: list[dict | None],
                   outcomes: dict) -> list[dict]:
        """One verdict body per record: the first failing completion's reason and
        the worst chunk measures over all of them."""
        for i, record in enumerate(records):
            if results[i] is not None:
                continue
            passed, reason = True, None
            worst = dict(_WORST_ZERO)
            for c_idx in range(len(record["completions"])):
                outcome = outcomes[i, c_idx]
                for result in outcome.results:
                    worst["worst_exp"] = max(worst["worst_exp"], result.exp_mismatches)
                    worst["worst_mant_mean"] = max(worst["worst_mant_mean"], float(result.mant_err_mean))
                    worst["worst_mant_median"] = max(worst["worst_mant_median"], float(result.mant_err_median))
                if not outcome.passed and passed:
                    passed, reason = False, outcome.reason
            results[i] = {"passed": passed, "reason": reason, **worst}
        return results

    def _judge_many(self, records: list[dict]) -> list[dict]:
        """Judge several records at once: every completion of every record that
        needs the GPU is packed, sorted by length, into shared forward passes."""
        results, items = self._prepare(records)
        scores, forward_seconds, verify_seconds = score_sequences(
            self._model, [(tokens, n, proofs) for _, _, tokens, n, proofs in items],
            chunk_tokens=self._proof.chunk_tokens, topk=self._proof.topk,
            batch_tokens=AUDIT_BATCH_TOKENS)
        outcomes = {(i, c_idx): outcome_from_scores(status, chunks, self._proof)
                    for (i, c_idx, *_), (status, chunks) in zip(items, scores)}
        self._aggregate(records, results, outcomes)
        self._log_batch(records, [(len(t), i, c) for i, c, t, _, _ in items],
                        forward_seconds, verify_seconds)
        return results

    def _log_batch(self, records: list[dict], queue: list, forward: float,
                   verify: float, called: float | None = None) -> None:
        """One line per judged batch: what it held, how long its oldest record
        had waited, and the speed of the GPU forward and the proof check."""
        completion_tokens = sum(len(records[i]["completions"][c]["tokens"]) for _, i, c in queue)
        arrivals = [float(r["received_at"]) for r in records if r.get("received_at") is not None]
        wait = f"{self._clock() - min(arrivals):.1f}s" if arrivals else "-"
        busy = forward + verify
        logger.info(
            "corpus audit batch: job=%s records=%d completions=%d completion_tokens=%d "
            "oldest_wait=%s call=%.1fs forward=%.3fs verify=%.3fs tokens_per_s=%.0f",
            self._job_id, len(records), len(queue), completion_tokens, wait,
            busy if called is None else called, forward, verify,
            completion_tokens / busy if busy > 0 else 0.0,
        )

    async def _read(self, submission_id: str) -> dict | None:
        try:
            record = await self._records.read_submission(self._job_id, submission_id)
        except Exception:
            # Isolated per id: one bad read must not stall the rest of the batch.
            logger.exception("corpus read of %s failed; leaving it pending", submission_id[:12])
            self._unreadable.add(submission_id)
            return None
        if record is None:
            logger.error("corpus submission %s has no record", submission_id[:12])
            self._unreadable.add(submission_id)
            return None
        self._unreadable.discard(submission_id)
        self._remember(submission_id, record)
        return record

    def _remember(self, submission_id: str, meta) -> None:
        """A record's (hotkey, arrival, size), from the record or its metadata."""
        if submission_id in self._meta:
            return
        received = meta.get("received_at")
        # An older record carries no arrival time: its hold counts from when
        # we first saw it, never as already over.
        received_at = float(received) if received is not None else self._clock()
        self._meta[submission_id] = (meta["hotkey"], received_at, int(meta["token_count"]))
        bisect.insort(self._arrivals.setdefault(meta["hotkey"], []), received_at)
        if submission_id not in self._judged:
            self._unjudged.setdefault(meta["hotkey"], set()).add(submission_id)

    async def _read_meta_all(self, submission_ids, reader) -> None:
        """Each id's metadata (``reader``: the store's tail read),
        SEED_META_CONCURRENCY at a time; a failed one is unreadable, as a
        failed record read is."""
        gate = asyncio.Semaphore(SEED_META_CONCURRENCY)

        async def one(submission_id):
            async with gate:
                try:
                    meta = await reader(self._job_id, submission_id)
                except Exception:
                    logger.exception("corpus metadata read of %s failed; leaving it pending",
                                     submission_id[:12])
                    self._unreadable.add(submission_id)
                    return
            if meta is None:
                logger.error("corpus submission %s has no record", submission_id[:12])
                self._unreadable.add(submission_id)
                return
            self._unreadable.discard(submission_id)
            self._remember(submission_id, meta)

        with self._timed("read"):
            await asyncio.gather(*(one(sid) for sid in dict.fromkeys(submission_ids)))

    async def _read_all(self, submission_ids) -> dict[str, dict]:
        """_read for many ids, READ_CONCURRENCY at a time; the readable ones."""
        gate = asyncio.Semaphore(READ_CONCURRENCY)

        async def one(submission_id):
            async with gate:
                return submission_id, await self._read(submission_id)

        with self._timed("read"):
            pairs = await asyncio.gather(*(one(sid) for sid in dict.fromkeys(submission_ids)))
        return {sid: record for sid, record in pairs if record is not None}

    def _recent(self, hotkey: str, now: float) -> int:
        arrivals = self._arrivals.get(hotkey, [])
        # Older than the hold window: never counted again, as `now` only grows.
        del arrivals[:bisect.bisect_left(arrivals, now - self._params.hold_seconds)]
        return bisect.bisect_right(arrivals, now)

    async def _forward(self, records: list[dict], *, local: bool = False) -> list[dict]:
        if not local and self._remote is not None and self._remote.connected():
            # An executor computes the chunk scores; the decision stays here.
            results, items = await self._in("codec", self._prepare, records)
            scores = await self._remote.score(
                [{"tokens": tokens, "prompt_len": n, "proofs": proofs}
                 for _, _, tokens, n, proofs in items])
            outcomes, scored_by = {}, {}
            for (i, c_idx, *_), (status, chunks, executor) in zip(items, scores):
                outcomes[i, c_idx] = outcome_from_scores(status, chunks, self._proof)
                if executor is not None:
                    scored_by.setdefault(i, set()).add(executor)
            judged = self._aggregate(records, results, outcomes)
            for i, executors in scored_by.items():
                judged[i] = {**judged[i], "scored_by": sorted(executors)}
            return judged
        if self._scorer is not None:
            return await self._scored(records)
        waited = time.monotonic()
        async with self._gpu_lock:
            # Shared FIFO with every job's auditor: the wait is not this job's work.
            self._phase["gpu_wait"] += time.monotonic() - waited
            with self._timed("forward"):
                return await self._in("gpu", self._judge_many, records)

    async def _scored(self, records: list[dict]) -> list[dict]:
        """``_judge_many`` with the forward on the GPU process: the same
        preparation and decision here, only the chunk scores cross."""
        waited = time.monotonic()
        async with self._gpu_lock:
            # As in-process: one job of this process prepares and scores at a
            # time, so their record preparations never pile up at once.
            self._phase["gpu_wait"] += time.monotonic() - waited
            results, items = await self._in("codec", self._prepare, records)
            forward = verify = 0.0
            scores: list = []
            called = time.monotonic()
            if items:
                with self._timed("forward"):
                    scores, forward, verify = await self._scorer(
                        [(tokens, n, proofs) for _, _, tokens, n, proofs in items])
            called = time.monotonic() - called
        outcomes = {(i, c_idx): outcome_from_scores(status, chunks, self._proof)
                    for (i, c_idx, *_), (status, chunks) in zip(items, scores)}
        self._aggregate(records, results, outcomes)
        self._log_batch(records, [(len(t), i, c) for i, c, t, _, _ in items], forward, verify,
                        called)
        return results

    async def _audit_outcomes(self, records: list[dict], *,
                              local: bool = False) -> list[dict | str]:
        """One outcome per record; a string is a validator-side error message."""
        batch_failed, batch_error = False, ""
        try:
            judged: list = await self._forward(records, local=local)
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # Record only the message here, then leave the block: `exc` and its
            # traceback pin every frame that was live when the batch failed
            # (including this batch's hidden states), and the retries below
            # must not run while any of that memory is still referenced.
            batch_failed, batch_error = True, str(exc)
            del exc

        if batch_failed:
            # Ours, not the miner's: one record's fault must not stall the rest
            # of the batch, so retry each alone before giving up on any of them.
            logger.error(
                "corpus audit batch of %d failed on the validator: %s; retrying one by one",
                len(records), batch_error,
            )
            if torch.cuda.is_available():
                # The failed batch's activations are unreachable now; hand that
                # memory back before the smaller retries ask for their own.
                torch.cuda.empty_cache()
            judged = []
            for record in records:
                try:
                    judged.append((await self._forward([record], local=local))[0])
                except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as solo_exc:
                    # A message, not the exception object: so this record's
                    # traceback (and whatever activations it pins) cannot
                    # outlive this line, into the next record's retry.
                    judged.append(str(solo_exc))
                    del solo_exc
        return judged

    def _verdict(self, submission_id: str, hotkey: str, token_count: int, outcome: dict,
                 draw: dict | None) -> dict:
        verdict = {
            "schema": VERDICT_SCHEMA,
            "submission_id": submission_id,
            "hotkey": hotkey,
            "token_count": int(token_count),
            "audited_at": self._clock(),
            **outcome,
        }
        if draw is not None:
            verdict["draw"] = draw
        return verdict

    async def _write(self, submission_id: str, verdict: dict) -> tuple[dict, bool]:
        """Create-only: the verdict that stands, and whether this call wrote it."""
        written = await self._records.write_verdict(self._job_id, submission_id, verdict)
        if written:
            self._mark_judged(submission_id)
            self._report(submission_id, verdict)
            return verdict, True
        standing = await self._records.read_verdict(self._job_id, submission_id)
        self._mark_judged(submission_id)
        self._report(submission_id, standing)
        return standing, False

    # Verdict writes of this auditor in flight at once, over every concurrent
    # `_write_all`; 1 writes them one at a time.
    write_concurrency = WRITE_CONCURRENCY

    async def _write_all(self, items: list[tuple[str, dict]]) -> list:
        """_write for each (id, verdict), `write_concurrency` at a time: per item
        (standing, written), or the exception its write raised. Every write
        is tried, so one store error does not leave the others unwritten."""
        if not items:
            return []
        # One gate per auditor (and event loop), shared by overlapping calls.
        key = (asyncio.get_running_loop(), self.write_concurrency)
        if getattr(self, "_write_gate", (None, None))[0] != key:
            self._write_gate = (key, asyncio.Semaphore(max(1, self.write_concurrency)))
        gate = self._write_gate[1]

        async def one(submission_id, verdict):
            async with gate:
                return await self._write(submission_id, verdict)

        with self._timed("write"):
            return await asyncio.gather(*(one(sid, v) for sid, v in items),
                                        return_exceptions=True)

    @staticmethod
    def _raise_first(results: list) -> None:
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _report(self, submission_id: str, verdict: dict | None) -> None:
        if self._on_verdict is None or verdict is None:
            return
        try:
            self._on_verdict(submission_id, verdict)
        except Exception:
            logger.exception("corpus verdict report for %s failed", submission_id[:12])

    def _report_voided(self, submission_id: str, document: dict) -> None:
        if self._on_voided is None:
            return
        try:
            self._on_voided(submission_id, document)
        except Exception:
            logger.exception("corpus void report for %s failed", submission_id[:12])

    async def _state(self, hotkey: str, now: float) -> MinerState:
        return (await self._states([hotkey], now))[hotkey]

    async def _states(self, hotkeys, now: float) -> dict[str, MinerState]:
        """Each hotkey's state: miners.json is one document, so it is read once
        for all of them (one read per hotkey was 8 s of a pass on 2026-10-02)."""
        hotkeys = sorted(set(hotkeys))
        if self._miner_states is None:
            return {hotkey: MinerState() for hotkey in hotkeys}
        if not hotkeys:
            return {}
        with self._timed("state"):
            many = getattr(self._miner_states, "get_many", None)
            if many is not None:
                got = await many(hotkeys)
            else:
                got = dict(zip(hotkeys, await asyncio.gather(
                    *(self._miner_states.get(hotkey) for hotkey in hotkeys))))
        return {hotkey: await self._end_ban(hotkey, got[hotkey], now) for hotkey in hotkeys}

    async def _end_ban(self, hotkey: str, state: MinerState, now: float) -> MinerState:
        if state.banned_until is not None and now >= state.banned_until:
            # Persist the end of a ban as a fresh probation (§7.3), so passes
            # counted before or during the ban never shorten it.
            def end_ban(m: MinerState) -> MinerState:
                if m.banned_until is not None and now >= m.banned_until:
                    return replace(m, banned_until=None, audited_passed=0)
                return m

            state = await self._miner_states.update(hotkey, end_ban)
        return state

    async def audit_many(self, submission_ids: list[str],
                         draws: dict[str, dict] | None = None) -> list[dict | None]:
        ids: list[str] = []
        records: list[dict] = []
        for submission_id in submission_ids:
            record = await self._read(submission_id)
            if record is not None:
                ids.append(submission_id)
                records.append(record)
        if not records:
            return []
        results, _ = await self._audit_records(ids, records, draws or {})
        return results

    async def _audit_records(self, ids: list[str], records: list[dict],
                             draws: dict[str, dict], *,
                             deferred: list | None = None) -> tuple[list[dict | None], set[str]]:
        """Audit, re-audit each failure alone, write the verdicts and move each
        hotkey's state. Returns the verdicts and the hotkeys with a confirmed failure."""
        outcomes = await self._audit_outcomes(records)
        for k, outcome in enumerate(outcomes):
            if isinstance(outcome, dict) and not outcome["passed"]:
                # §7.2: only a failure that a second, separate audit repeats counts.
                # Always on this GPU: an executor alone can never fail a miner.
                outcomes[k] = (await self._audit_outcomes([records[k]], local=True))[0]

        for submission_id, outcome in zip(ids, outcomes):
            if isinstance(outcome, dict):
                self._validator_errors = 0
            else:
                # Ours, not the miner's: leave it pending for the next rescan.
                self._validator_errors += 1
                logger.error(
                    "corpus audit of %s failed on the validator: %s", submission_id[:12], outcome
                )

        results: list[dict | None] = [None] * len(ids)
        judged = [k for k, outcome in enumerate(outcomes) if isinstance(outcome, dict)]
        now = self._clock()
        failures: dict[str, list[str]] = {}
        for k in judged:
            if not outcomes[k]["passed"]:
                failures.setdefault(records[k]["hotkey"], []).append(ids[k])
        failed_hotkeys = set(failures)
        states: dict[str, MinerState] = {}
        if failures:
            # Escalate before the verdicts exist: a crash in between then
            # re-audits the records, and the retry counts nothing twice
            # (idempotent by submission id), instead of leaving a caught
            # cheater unsuspected while its held records are paid.
            states.update(await self._escalate(failures, now))

        # Failures first: a ban they cause must void this batch's passes of that hotkey.
        order = sorted(judged, key=lambda k: outcomes[k]["passed"])
        # Read after the escalation, as each pass's own read was.
        states.update(await self._states(
            {records[k]["hotkey"] for k in order if outcomes[k]["passed"]} - set(states), now))
        planned = []
        for k in order:
            submission_id, record = ids[k], records[k]
            hotkey = record["hotkey"]
            outcome = {**outcomes[k], "audited": True}
            if outcome["passed"]:
                if effective_state(states[hotkey], now, self._params) == "banned":
                    outcome = dict(_BANNED_VOID)
            planned.append((k, outcome, self._verdict(
                submission_id, hotkey, record["token_count"], outcome,
                draws.get(submission_id) if outcome["audited"] else None)))
        landed = await self._write_all([(ids[k], verdict) for k, _, verdict in planned])
        passes: dict[str, list[float]] = {}
        for (k, outcome, _), result in zip(planned, landed):
            if isinstance(result, BaseException):
                continue
            results[k], written = result
            # Only the call that wrote a passing verdict counts it: a repeat audit
            # (stale queue entry, restart) must never count twice.
            if written and outcome["passed"] and outcome["audited"]:
                hotkey = records[k]["hotkey"]
                passes.setdefault(hotkey, []).append((ids[k], outcome["worst_mant_mean"]))
                for executor_id in outcome.get("scored_by", ()):
                    self._remote_scored.setdefault(executor_id, []).append(ids[k])

        def count(batch: list[tuple[str, float]]) -> Callable[[MinerState], MinerState]:
            def change(m: MinerState) -> MinerState:
                for sid, mant_mean in batch:
                    if effective_state(m, now, self._params) == "banned":
                        return m
                    m = after_pass(m, self._params, mant_mean, sid)
                return m
            return change

        # At most PASS_IDS per hotkey per write: a retry of a write that landed
        # finds every one of its ids still in pass_ids and counts none twice.
        while passes and self._miner_states is not None:
            chunk = {hotkey: batch[:PASS_IDS] for hotkey, batch in passes.items()}
            passes = {hotkey: batch[PASS_IDS:] for hotkey, batch in passes.items()
                      if batch[PASS_IDS:]}
            await self._miner_states.update_many(
                {hotkey: count(batch) for hotkey, batch in chunk.items()})
        # Counted first: a write that failed leaves its record pending, the rest stand.
        if deferred is not None:
            deferred.extend(r for r in landed if isinstance(r, BaseException))
        else:
            self._raise_first(landed)
        return results, failed_hotkeys

    async def _escalate(self, failures: dict[str, list[str]],
                        now: float) -> dict[str, MinerState]:
        """Each hotkey's confirmed failures, in one write (miners.json is one key)."""
        if self._miner_states is None:
            return {}

        def escalate(sids: list[str]) -> Callable[[MinerState], MinerState]:
            def change(m: MinerState) -> MinerState:
                for sid in sids:
                    m = after_confirmed_failure(m, self._params, now, sid)
                return m
            return change

        return await self._miner_states.update_many(
            {hotkey: escalate(sids) for hotkey, sids in failures.items()})

    async def reaudit_executor(self, executor_id: str) -> list[str]:
        """Re-audit on this GPU every pass written from a quarantined executor's
        scores that is unsettled or settled inside the hold window. A failure
        (confirmed by a second audit, as §7.2 asks) is charged to its miner as
        any failed audit is, and its submission is voided so it is not paid."""
        sids = self._remote_scored.pop(executor_id, [])
        if not sids:
            return []
        now = self._clock()
        state, _ = await self._records.read_settlement(self._job_id) \
            if callable(getattr(self._records, "read_settlement", None)) else ({}, None)
        settled = set((state or {}).get("settled") or ())
        chosen = []
        for sid in dict.fromkeys(sids):
            if sid in settled:
                verdict = await self._records.read_verdict(self._job_id, sid)
                audited_at = float((verdict or {}).get("audited_at") or 0.0)
                if now - audited_at > self._params.hold_seconds:
                    continue
            chosen.append(sid)
        records = await self._read_all(chosen)
        ids = [sid for sid in chosen if sid in records]
        failed: dict[str, dict] = {}
        for sid, outcome in zip(ids, await self._audit_outcomes(
                [records[sid] for sid in ids], local=True)):
            if isinstance(outcome, dict) and not outcome["passed"]:
                again = (await self._audit_outcomes([records[sid]], local=True))[0]
                if isinstance(again, dict) and not again["passed"]:
                    failed[sid] = again
        if failed:
            by_hotkey: dict[str, list[str]] = {}
            for sid in failed:
                by_hotkey.setdefault(records[sid]["hotkey"], []).append(sid)
            await self._escalate(by_hotkey, now)
            writer = getattr(self._records, "write_voided", None)
            for sid, outcome in failed.items():
                if writer is not None:
                    document = {
                        "schema": VOIDED_SCHEMA, "submission_id": sid,
                        "hotkey": records[sid]["hotkey"], "executor_id": executor_id,
                        "reason": "executor_quarantined", "voided_at": now, **outcome}
                    await writer(self._job_id, sid, document)
                    self._report_voided(sid, document)
        logger.warning("corpus job %s: re-audited %d pass(es) scored by quarantined executor "
                       "%s; %d failed", self._job_id, len(ids), executor_id, len(failed))
        return sorted(failed)

    async def _randomness_for(self, round_number: int) -> str | None:
        """The drand randomness of a round, lowercased, or None (fetch error,
        malformed): the caller then audits. A round that just failed is not
        refetched for NEGATIVE_BEACON_CACHE_SECONDS -- every sampled
        submission whose draw lands on that round would otherwise repeat the
        same failing network call, one per judging pass."""
        randomness = self._randomness.get(round_number)
        if randomness is not None:
            return randomness
        failed_at = self._failed_rounds.get(round_number)
        if failed_at is not None and self._clock() - failed_at < self._negative_window(round_number):
            return None
        if round_number in self._raced:
            return None  # failed already in this pass
        self._raced.add(round_number)
        try:
            with self._timed("drand"):
                value = await self._in("beacon", self._beacon, round_number)
        except Exception:
            logger.warning("drand round %d unavailable; auditing", round_number, exc_info=True)
            value = None
        if isinstance(value, str) and _HEX64.fullmatch(value.lower()):
            randomness = self._randomness[round_number] = value.lower()
            self._failed_rounds.pop(round_number, None)
        else:
            if value is not None:
                logger.error("drand round %d gave malformed randomness %r; auditing",
                             round_number, value)
            self._failed_rounds[round_number] = self._clock()
        return randomness

    def _negative_window(self, round_number: int) -> float:
        try:
            old = int(self._round_at(self._clock())) - round_number > OLD_ROUND_ROUNDS
        except Exception:
            old = False
        return OLD_ROUND_NEGATIVE_CACHE_SECONDS if old else NEGATIVE_BEACON_CACHE_SECONDS

    async def _prefetch_rounds(self, submission_ids, now: float, states: dict) -> int:
        """Fetch together the drand rounds _decide will ask for one by one; it
        then reads them from the cache. A round that fails is not raced again
        this pass (``_raced``). Returns how many rounds were raced."""
        if self._params.q >= 1.0 or self._beacon is None or self._round_at is None:
            return 0
        rounds = set()
        for sid in submission_ids:
            hotkey, received_at, _ = self._meta[sid]
            if effective_state(states[hotkey], now, self._params) != "sampled":
                continue
            try:
                round_number = int(self._round_at(received_at)) + 1
                if int(self._round_at(now - BEACON_GRACE_SECONDS)) <= round_number:
                    continue  # not out yet: _decide calls it undecidable
            except Exception:
                return 0
            failed_at = self._failed_rounds.get(round_number)
            if (round_number not in self._randomness and round_number not in self._raced
                    and (failed_at is None
                         or now - failed_at >= self._negative_window(round_number))):
                rounds.add(round_number)
        gate = asyncio.Semaphore(DRAND_CONCURRENCY)

        async def one(round_number):
            async with gate:
                await self._randomness_for(round_number)

        await asyncio.gather(*(one(r) for r in sorted(rounds)))
        return len(rounds)

    async def _decide(self, submission_id: str, now: float,
                      state: MinerState) -> tuple[str, dict | None]:
        """decision() for one known record, with its draw; "undecidable" while
        its draw round is not out yet (the rescan comes back for it)."""
        hotkey, received_at, _ = self._meta[submission_id]
        randomness, draw, recent = None, None, 0
        if self._params.q < 1.0 and effective_state(state, now, self._params) == "sampled":
            if not self._seeded:
                await self._seed(await self.pending_ids())
            recent = self._recent(hotkey, now)
            if (recent >= 1.0 / self._params.q and self._beacon is not None
                    and self._round_at is not None):
                # round_at(t) is the first round published strictly after t;
                # one more round keeps the miner signing before its
                # randomness exists even with our clock a period behind (§6).
                # It may raise -- the drand chain's genesis/period can still
                # be unresolved (a lazy `round_at` retries on its own
                # schedule) -- caught here rather than propagated, so an
                # unresolved chain audits this submission (randomness stays
                # None below, decision() then reads that as "audit") instead
                # of crashing the whole batch out of the drain loop.
                try:
                    round_number = int(self._round_at(received_at)) + 1
                    round_not_out_yet = int(self._round_at(now - BEACON_GRACE_SECONDS)) <= round_number
                except Exception:
                    logger.warning(
                        "round_at unavailable for %s; auditing", submission_id[:12],
                        exc_info=True,
                    )
                else:
                    if round_not_out_yet:
                        return "undecidable", None
                    randomness = await self._randomness_for(round_number)
                    if randomness is not None:
                        draw = {"round": round_number, "q": self._params.q,
                                "drawn": drawn(randomness, submission_id, self._params.q)}
                        if not draw["drawn"] and submission_id not in self._judged:
                            self._undrawn.add(submission_id)
        choice = decision(state, params=self._params, now=now, received_at=received_at,
                          recent_submissions=recent, randomness_hex=randomness,
                          submission_id=submission_id, slack_seconds=self._accept_slack)
        return choice, draw

    async def _judge_once(self, submission_ids: list[str]) -> set[str]:
        now = self._clock()
        # The backward audit and the rescan bring a caught hotkey's records back.
        submission_ids = [sid for sid in submission_ids if sid not in self._judged]
        for submission_id in submission_ids:
            # This pass's word on when to judge it again replaces an earlier one.
            self._next_due.pop(submission_id, None)
        read: dict[str, dict] = {}
        read.update(await self._read_all(
            [sid for sid in submission_ids if sid not in self._meta]))
        known = [sid for sid in dict.fromkeys(submission_ids) if sid in self._meta]
        states = await self._states({self._meta[sid][0] for sid in known}, now)

        audit_ids, draws, unaudited, voided = [], {}, [], []
        # Per hotkey, arrival times of records whose draw round is not out yet.
        undecided: dict[str, list[float]] = {}
        with self._timed("decide"):
            rounds_needed = await self._prefetch_rounds(known, now, states)
        for submission_id in known:
            hotkey, received_at, _ = self._meta[submission_id]
            with self._timed("decide"):
                choice, draw = await self._decide(submission_id, now, states[hotkey])
            self._choices[choice] += 1
            if choice == "audit":
                audit_ids.append(submission_id)
                if draw is not None:
                    draws[submission_id] = draw
            elif choice == "pass_unaudited":
                unaudited.append((submission_id, draw))
            elif choice == "void_banned":
                voided.append(submission_id)
            elif choice == "undecidable":
                undecided.setdefault(hotkey, []).append(received_at)
                self._next_due[submission_id] = self._retry_at(hotkey, now)
            elif choice == "wait":
                self._next_due[submission_id] = self._wait_until(submission_id, now)

        if not unaudited:
            logger.info("corpus judge pass started: job=%s ids=%d payable=0 payable_waiting=0 "
                        "siblings=0 rounds_needed=%d", self._job_id, len(known), rounds_needed)
        covered: float | None = math.inf
        if unaudited:
            # Sampled BEFORE the siblings are collected: only what was enqueued
            # by then is considered, so only that may vouch for X (a listing
            # finishing mid-pass must not).
            covered = self._covered()
            # Queue lag can exceed the hold: a drawn sibling received within X's
            # hold may still sit in the queue. Decide every such sibling now and
            # audit the drawn ones in this pass, so a failure among them reaches
            # X through the same-pass guard below instead of after X is paid.
            await self._read_all(sorted(
                sid for sid in self._queued | self._unreadable
                if sid not in self._meta and sid not in self._judged))
            in_pass = set(known)
            # A sibling known undrawn decides "wait" or "pass_unaudited" here:
            # its hotkey's state and recent count are X's, its draw is fixed.
            candidates = {}
            for hotkey in {self._meta[sid][0] for sid, _ in unaudited}:
                candidates[hotkey] = sorted(
                    (self._meta[sid][1], sid) for sid in self._unjudged.get(hotkey, ())
                    if sid not in in_pass and sid not in self._undrawn)
            # The oldest payable records first, while their siblings fit the
            # pass; the others wait for the next passes (nothing is paid early).
            hold_end: dict[str, float] = {}
            taken: dict[str, int] = {}
            kept, waiting = [], []
            for submission_id, draw in sorted(unaudited, key=lambda u: self._meta[u[0]][1]):
                hotkey, received_at, _ = self._meta[submission_id]
                until = max(hold_end.get(hotkey, 0.0), received_at + self._params.hold_seconds)
                count = bisect.bisect_right(candidates[hotkey], (until, "\uffff"))
                grown = sum(taken.values()) - taken.get(hotkey, 0) + count
                if grown > PASS_SIBLINGS:
                    if not hold_end:
                        # Even the oldest does not fit: decide the first of its
                        # siblings now (their draws become known, the undrawn
                        # leave its set) and pay it in a later pass.
                        hold_end[hotkey], taken[hotkey] = until, PASS_SIBLINGS
                    waiting.append(submission_id)
                    continue
                hold_end[hotkey], taken[hotkey] = until, count
                kept.append((submission_id, draw))
            for submission_id in waiting:
                self._next_due[submission_id] = now
            unaudited = [u for u in unaudited if u in kept]
            siblings = {hotkey: [sid for _, sid in candidates[hotkey][:taken[hotkey]]]
                        for hotkey in hold_end}
            with self._timed("decide"):
                rounds_needed += await self._prefetch_rounds(
                    [sid for sids in siblings.values() for sid in sids], now, states)
            logger.info("corpus judge pass started: job=%s ids=%d payable=%d payable_waiting=%d "
                        "siblings=%d rounds_needed=%d", self._job_id, len(known), len(kept),
                        len(waiting), sum(taken.values()), rounds_needed)
            for hotkey, sids in siblings.items():
                for sid in sids:
                    with self._timed("decide"):
                        choice, draw = await self._decide(sid, now, states[hotkey])
                    self._choices["sibling_" + choice] += 1
                    if choice == "audit":
                        audit_ids.append(sid)
                        if draw is not None:
                            draws[sid] = draw
                    elif choice == "undecidable":
                        undecided.setdefault(hotkey, []).append(self._meta[sid][1])

        # The pass's budget: the oldest audits first, the rest in later passes.
        # A hotkey with a deferred audit pays nothing unaudited this pass (the
        # deferred one may be the drawn sibling that would catch it).
        audit_ids, later = self._within_budget(audit_ids, set(submission_ids))
        deferred_hotkeys = {self._meta[sid][0] for sid in later}
        for sid in later:
            self._next_due[sid] = now
            self._schedule(sid, now)

        failed: set[str] = set()
        errored: set[str] = set()
        pairs: list[tuple[str, dict]] = []
        if audit_ids:
            # Known records (seeded, siblings) are not in `read`: fetched together.
            read.update(await self._read_all([sid for sid in audit_ids if sid not in read]))
            records = [read.get(sid) for sid in audit_ids]
            pairs = [(sid, r) for sid, r in zip(audit_ids, records) if r is not None]
            errored = {self._meta[sid][0] for sid, r in zip(audit_ids, records) if r is None}
        # Nothing the audit does reads the store: what is unreadable is known now.
        unreadable = bool(self._unreadable)
        if unaudited and unreadable:
            logger.error(
                "%d pending corpus record(s) unreadable (e.g. %s); every unaudited pass "
                "waits until they read or get a verdict",
                len(self._unreadable), min(self._unreadable)[:12])
        if unaudited:
            # And again now: the lesser of the two (a new front since resets it).
            after = self._covered()
            covered = None if covered is None or after is None else min(covered, after)
        uncovered = 0
        audited_hotkeys = {self._meta[sid][0] for sid in audit_ids}
        early, late = [], []
        for submission_id, draw in unaudited:
            hotkey, received_at, token_count = self._meta[submission_id]
            # Wait while a sibling that could still catch this record is
            # undecided or hit a validator error this pass, or while any
            # pending record is unreadable (its hotkey could be this one).
            if unreadable or hotkey in errored:
                continue  # judged again at the next rescan
            if not self._covers(covered, received_at):
                # A sibling the front accepted may not have been handed over
                # yet: as with an unreadable record, it could catch this one.
                uncovered += 1
                if covered is not None:
                    self._next_due[submission_id] = self._retry_at(hotkey, now)
                continue
            if hotkey in deferred_hotkeys or any(
                    t <= received_at + self._params.hold_seconds
                    for t in undecided.get(hotkey, ())):
                self._next_due[submission_id] = self._retry_at(hotkey, now)
                continue
            verdict = (submission_id, self._verdict(
                submission_id, hotkey, token_count,
                {"passed": True, "audited": False, "reason": None, **_WORST_ZERO}, draw))
            # No record of this hotkey is audited in this pass, so no failure
            # or error of it can come from the audit: written alongside it.
            (late if hotkey in audited_hotkeys else early).append(verdict)
        if uncovered:
            logger.warning("corpus job %s: arrival feed covers up to %s; %d unaudited pass(es) "
                           "wait", self._job_id, "nothing" if covered is None
                           else f"{now - covered:.0f} s ago", uncovered)
        for submission_id in voided:
            hotkey, _, token_count = self._meta[submission_id]
            early.append((submission_id, self._verdict(
                submission_id, hotkey, token_count, dict(_BANNED_VOID), None)))
        alongside = asyncio.ensure_future(self._write_all(early))
        try:
            if pairs:
                with self._timed("audit"):
                    results, failed = await self._audit_records(
                        [sid for sid, _ in pairs], [r for _, r in pairs], draws,
                        deferred=self._deferred)
                errored |= {r["hotkey"] for (_, r), result in zip(pairs, results) if result is None}
        finally:
            landed = await alongside
        # After the audits and escalations are durable, as before.
        landed += await self._write_all([
            (sid, verdict) for sid, verdict in late
            # A failed hotkey is suspect now: the backward audit decides its records.
            if verdict["hotkey"] not in failed and verdict["hotkey"] not in errored])
        # Raised by judge_many once the backward audit of `failed` has run.
        self._deferred.extend(r for r in landed if isinstance(r, BaseException))
        return failed

    def _within_budget(self, audit_ids: list[str], in_pass: set[str]) -> tuple[list[str], list[str]]:
        """The audits this pass makes (oldest first, at least one) and those
        over PASS_AUDIT_ROWS / PASS_AUDIT_TOKENS, left for the next passes."""
        if not audit_ids:
            return [], []
        order = sorted(dict.fromkeys(audit_ids), key=lambda sid: (self._meta[sid][1], sid))
        kept, tokens = [], 0
        for sid in order:
            size = int(self._meta[sid][2])
            if kept and (len(kept) >= PASS_AUDIT_ROWS or tokens + size > PASS_AUDIT_TOKENS):
                break
            kept.append(sid)
            tokens += size
        later = order[len(kept):]
        if later:
            self._choices["audit_deferred"] += len(later)
            logger.info("corpus job %s: pass audits %d record(s) (%d tokens); %d deferred to the "
                        "next passes", self._job_id, len(kept), tokens, len(later))
        keep = set(kept)
        return [sid for sid in audit_ids if sid in keep], later

    async def judge_many(self, submission_ids: list[str]) -> None:
        """Decide each record: audit now, wait out its hold, pass it unaudited, or
        void it for a ban; then audit backwards after every confirmed failure."""
        start = time.monotonic()
        self._next_due.clear()
        self._deferred = []
        self._raced = set()
        try:
            failed = await self._judge_once(list(submission_ids))
            # At q = 1 every held record is already being audited on arrival.
            while failed and self._params.q < 1.0:
                # §7.2: every record of a hotkey just found cheating that has no
                # verdict yet is audited (it is suspect now) before it can be paid.
                pending = await self.pending_ids()
                await self._read_all([sid for sid in pending if sid not in self._meta])
                held = [sid for sid in pending if sid in self._meta and self._meta[sid][0] in failed]
                failed = await self._judge_once(held)
            # A write that failed leaves its record pending; raised only now, so
            # it never kept a caught hotkey's held records from their audit.
            self._raise_first(self._deferred)
        finally:
            # decide includes drand; audit includes gpu_wait, forward and the writes of
            # audited verdicts; drand sums its concurrent fetches, the others are wall time.
            logger.info(
                "corpus judge pass: job=%s ids=%d total=%.1fs list=%.1fs read=%.1fs "
                "state=%.1fs decide=%.1fs drand=%.1fs audit=%.1fs gpu_wait=%.1fs "
                "forward=%.1fs write=%.1fs choices=%s",
                self._job_id, len(submission_ids), time.monotonic() - start,
                *(self._phase[p] for p in ("list", "read", "state", "decide", "drand", "audit",
                                           "gpu_wait", "forward", "write")),
                dict(self._choices))
            self._phase.clear()
            self._choices.clear()

    async def audit(self, submission_id: str) -> dict | None:
        results = await self.audit_many([submission_id])
        return results[0] if results else None

    def _known_pending(self) -> list[str]:
        """Pending records this process knows of: read and not judged, or unreadable."""
        return [sid for sid in (*self._meta, *self._unreadable) if sid not in self._judged]

    async def _rescan_once(self, *, full: bool) -> None:
        # Only what is not scheduled yet: a record waiting out its hold keeps its time.
        pending = await self.pending_ids() if full else self._known_pending()
        for submission_id in pending:
            self.enqueue(submission_id)
        lag = self.queue_lag(pending)
        # An undrawn record waits one hold plus the accept slack by design;
        # far beyond that, the auditor is not keeping up with the traffic.
        level = (logging.WARNING if lag is not None
                 and lag > self._params.hold_seconds + self._accept_slack
                 + 2 * self._rescan_every
                 else logging.INFO)
        logger.log(level, "corpus audit queue lag: %d pending, oldest received %s s ago",
                   len(pending), "-" if lag is None else f"{lag:.0f}")

    async def rescan_store(self) -> None:
        """List the store now and schedule every pending record not yet
        scheduled (the arrival feed's net after a front restart)."""
        await self._rescan_once(full=True)

    async def _rescan_forever(self) -> None:
        # The retry for an id whose audit failed or that was waiting out its
        # hold: from memory every period, from a store listing every
        # FULL_RESCAN_SECONDS as the net for anything the route did not hand over.
        last_full = time.monotonic()
        while True:
            await asyncio.sleep(self._rescan_every)
            full = time.monotonic() - last_full >= FULL_RESCAN_SECONDS
            try:
                await self._rescan_once(full=full)
                if full:
                    last_full = time.monotonic()
            except Exception:
                logger.exception("corpus pending rescan failed; retrying next period")

    async def _seed(self, pending: list[str]) -> None:
        """Read every pending record once, for the hold window's arrivals. In
        slices: only the arrival times are kept, so 50k records read after a
        restart are never all in memory at once."""
        unread = [sid for sid in pending if sid not in self._meta]
        reader = getattr(self._records, "read_submission_meta", None)
        started = time.monotonic()
        for i in range(0, len(unread), SEED_SLICE_IDS):
            if reader is not None:
                # The tail of each record (hotkey, arrival, size), never the
                # whole: a restart on 122k records read 12-24 GB before (2026-10-02).
                await self._read_meta_all(unread[i:i + SEED_SLICE_IDS], reader)
            else:
                await self._read_all(unread[i:i + SEED_SLICE_IDS])
            if (i // SEED_SLICE_IDS) % 10 == 9:
                logger.info("corpus job %s: seeded %d/%d pending record(s) in %.0f s",
                            self._job_id, min(i + SEED_SLICE_IDS, len(unread)), len(unread),
                            time.monotonic() - started)
        if unread:
            logger.info("corpus job %s: seeded %d pending record(s) in %.0f s (%s)", self._job_id,
                        len(unread), time.monotonic() - started,
                        "metadata" if reader is not None else "whole records")
        self._seeded = True

    async def _start(self) -> None:
        """List the job once, read every pending record (the hold window's
        arrivals, which the first sampled decision needs) and schedule them all
        now: the first passes then take them oldest first."""
        pending = await self.pending_ids()
        await self._seed(pending)
        for submission_id in pending:
            self.enqueue(submission_id)

    async def _next_batch(self) -> list[str]:
        """The due ids, at most RUN_BATCH_IDS, once at least one is due."""
        idle = time.monotonic()
        while True:
            batch = self._take_due(self._clock(), RUN_BATCH_IDS)
            if batch:
                if time.monotonic() - idle > 5.0:
                    logger.info("corpus auditor idle %.1fs waiting for work",
                                time.monotonic() - idle)
                return batch
            wait = self._heap[0][0] - self._clock() if self._heap else self._rescan_every
            if self._wake is None:
                self._wake = asyncio.Event()
            self._wake.clear()
            # An enqueue wakes it early; a long wait is capped so a clock that
            # jumped is noticed within one rescan period.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(),
                                       timeout=min(max(wait, 0.01), self._rescan_every))

    async def run(self) -> None:
        await self._start()
        rescan = asyncio.create_task(self._rescan_forever())
        try:
            while True:
                batch = await self._next_batch()
                try:
                    await self.judge_many(batch)
                except Exception:
                    # A store hiccup (e.g. a transient ConnectionError) must not kill
                    # the drain loop: the submissions stay pending and are
                    # judged again at the next rescan period.
                    logger.exception("corpus audit of a batch crashed the drain loop")
                finally:
                    self._reschedule(batch)
                if self._validator_errors >= self._max_validator_errors:
                    # Spec §6: a validator-side fault stops the worker loudly. A
                    # process that looks healthy while paying nobody is worse.
                    logger.critical(
                        "corpus audit failed on the validator %d times in a row; stopping",
                        self._validator_errors,
                    )
                    raise CorpusAuditorHalted(
                        f"{self._validator_errors} consecutive validator-side audit errors"
                    )
        finally:
            rescan.cancel()
