"""Grading an episode job's accepted trajectories (spec §5 N5).

Every accepted trajectory is graded (unless its TOPLOC verdict already
failed); every passing one is replayed, and a failing one when drawn
(``replay_fraction_failed``, the drand round after its arrival,
domain-separated from the audit draw). The actions replayed come from the
proven tokens (``trajectory_parse``, under the job's turn limit).

Only one outcome sanctions: an ``ok`` replay decision that does not certify
the episode, agreed by executors of two distinct providers (rulings P16/P17).
It is the TOPLOC failure's path: escalation of the miner's state, then a void.
Every other outcome -- a failing grade, a replay one executor alone failed,
``timeout``, ``error``, ``ungradeable``, ``disputed`` -- sanctions nobody and
certifies nothing; it is written in the grade document as it is. Grades never
change payment otherwise; the settler waits for a submission's grade so a
void lands before payment.

What an executor decided alone is graded again, into ``regrades/``, when that
executor is quarantined (``regrade_executor``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from reliquary.corpus.audit_policy import after_confirmed_failure, replay_drawn
from reliquary.corpus.replay_compare import allowed_mismatches
from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
from reliquary.infrastructure.corpus_record_store import MAX_REGRADE_GENERATIONS, RECORD_SCHEMA_V2
from reliquary.validator.corpus_grade_remote import replay_certified

logger = logging.getLogger(__name__)

GRADE_SCHEMA = "reliquary/corpus-grade/v1"
# The auditor's (corpus_auditor imports torch; this module must not).
VOIDED_SCHEMA = "reliquary/corpus-voided/v1"
GRADE_CONCURRENCY = 64
RESCAN_SECONDS = 60.0
# Grade documents read at once when a quarantine needs those written before boot.
INDEX_READ_CONCURRENCY = 16
# Distinct providers a failed replay needs before it sanctions anyone.
SANCTION_PROVIDERS = 2


def _lone(decision_by) -> str | None:
    """The executor that decided alone, or None."""
    deciders = set(decision_by or ())
    return next(iter(deciders)) if len(deciders) == 1 else None


class CorpusGrader:
    def __init__(self, *, job, records, dispatcher, renderer, source, params, miner_states=None,
                 beacon: Callable[[int], str | None] | None = None,
                 round_at: Callable[[float], int] | None = None,
                 on_voided: Callable[[str, dict], None] | None = None,
                 clock: Callable[[], float] = time.time, concurrency: int = GRADE_CONCURRENCY,
                 rescan_seconds: float = RESCAN_SECONDS) -> None:
        self._job = job
        self._records = records
        self._dispatcher = dispatcher
        self._renderer = renderer
        self._source = source
        self._params = params
        self._miner_states = miner_states
        self._beacon = beacon
        self._round_at = round_at
        self.on_voided = on_voided
        self._clock = clock
        self._concurrency = concurrency
        self._rescan = rescan_seconds
        self._final: set[str] = set()
        self._seeded = False
        self._inflight: set[str] = set()
        self._gate: asyncio.Semaphore | None = None
        # Per executor, the submissions with a decision it made alone: graded
        # again if it is quarantined.
        self._sole: dict[str, set[str]] = {}
        # Grades listed at boot, not yet read for their lone deciders: read
        # only when a quarantine needs them.
        self._unindexed: set[str] = set()
        # Graded, waiting for the drand round of the replay draw (ruling P7):
        # the rescan retries the draw, never the grade.
        self._awaiting_draw: dict[str, dict] = {}
        # Regrades that failed (a store or dispatcher error): retried by the rescan.
        self._regrade_retry: set[str] = set()
        # Graded alone by a quarantined executor, regrade not yet written:
        # never ready, so nothing it certified alone is paid meanwhile.
        self._regrading: set[str] = set()
        # Quarantined executors whose grades from before boot are not yet
        # indexed: every unindexed grade is held until they are.
        self._held_executors: set[str] = set()
        self._regrades_inflight: set[str] = set()
        self._executor_regrades_inflight: set[str] = set()
        # Every task this grader starts, held until done: the loop keeps only
        # weak references, and a collected task is a grade or regrade lost.
        self._tasks: set[asyncio.Task] = set()
        # Every executor this grader has seen quarantined; never shrinks.
        self._quarantined: set[str] = set()
        # The regrade generation each submission's next regrade writes.
        self._generation: dict[str, tuple[int, list[str]]] = {}
        # Submissions listed but without a grade, as of the last rescan.
        self._ungraded: int | None = None

    # -- what the settler and the route see -----------------------------------

    async def _seed(self) -> None:
        if not self._seeded:
            listed = set(await self._records.list_grade_ids(self._job.job_id))
            self._unindexed |= listed - self._final
            self._final |= listed
            self._seeded = True

    async def ready(self, ids) -> set[str]:
        """Which of ``ids`` may be paid: graded, and not waiting for a regrade
        after a quarantine (the settler pays only those)."""
        await self._seed()
        held = self._unindexed if self._held_executors else ()
        return {sid for sid in ids
                if sid in self._final and sid not in self._regrading and sid not in held}

    def status(self) -> dict:
        """For the job status route."""
        stats = getattr(self._dispatcher, "stats", None) or {}
        return {"graded": len(self._final), "ungraded": self._ungraded,
                "grading": len(self._inflight), "awaiting_draw": len(self._awaiting_draw),
                "regrading": len(self._regrading),
                # Non-empty: every grade from before boot is held (job-wide).
                "held_executors": sorted(self._held_executors),
                "unindexed": len(self._unindexed),
                "dispatcher_waiting": stats.get("waiting")}

    def _gated(self):
        if self._gate is None:
            self._gate = asyncio.Semaphore(self._concurrency)
        return self._gate

    def enqueue(self, submission_id: str) -> None:
        if submission_id not in self._final and submission_id not in self._inflight:
            self._start(submission_id)

    def _start(self, submission_id: str) -> None:
        self._gated()
        self._inflight.add(submission_id)
        self._spawn(self._guarded(submission_id))

    async def _guarded(self, submission_id: str) -> None:
        try:
            async with self._gate:
                await self.grade_one(submission_id)
        except Exception:
            logger.exception("grading %s failed; retried on the next rescan", submission_id[:12])
        finally:
            self._inflight.discard(submission_id)

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait until every task this grader started (and any they started)
        is done; their errors are theirs to log."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def rescan_once(self) -> None:
        """One pass of ``run``: grade what is new, relaunch what must be redone.
        Everything it launches runs in the background (``drain`` awaits it)."""
        await self._seed()
        listed = await self._records.list_submission_ids(self._job.job_id)
        for sid in listed:
            self.enqueue(sid)
        self._ungraded = sum(1 for sid in listed if sid not in self._final)
        for executor_id in sorted(self._held_executors - self._executor_regrades_inflight):
            # Its regrade could not index the grades from before boot.
            self._spawn(self._regrade_executor_logged(executor_id))
        for sid in sorted(self._regrade_retry - self._regrades_inflight):
            # In the background: a slow regrade never holds the rescan.
            self._spawn(self._regrade(sid))

    async def run(self) -> None:
        while True:
            try:
                await self.rescan_once()
            except Exception:
                logger.exception("grader rescan of %s failed", self._job.job_id)
            await asyncio.sleep(self._rescan)

    # -- one submission ---------------------------------------------------------

    async def grade_one(self, submission_id: str, *, regrade: bool = False) -> dict | None:
        """Grade (and replay) one submission and write its document; None
        while its replay draw waits for an unpublished drand round."""
        waiting = None if regrade else self._awaiting_draw.get(submission_id)
        if waiting is not None:
            return await self._decide_replay(submission_id, waiting, regrade=regrade)
        job_id = self._job.job_id
        record = await self._records.read_submission(job_id, submission_id)
        if record is None or record.get("schema") != RECORD_SCHEMA_V2:
            return None
        base = {"schema": GRADE_SCHEMA, "submission_id": submission_id, "hotkey": record["hotkey"],
                "prompt_index": record["prompt_index"]}
        verdict = await self._records.read_verdict(job_id, submission_id)
        if verdict is not None and not verdict.get("passed"):
            return await self._write(submission_id, {
                **base, "status": "audit_failed", "graded_success": False, "replay": None,
                "replay_certified": False, "graded_at": self._clock()}, regrade=regrade)
        trajectory = record["completions"][0]
        try:
            parsed = await asyncio.to_thread(
                parse_trajectory, self._renderer, prompt_ids=trajectory["prompt_tokens"],
                tokens=trajectory["tokens"],
                spans=[(turn["start"], turn["end"]) for turn in trajectory["turns"]],
                stop=trajectory["stop"], max_turns=self._job.episode.max_turns)
        except TrajectoryRefused as refused:
            # Intake parsed it already: only a renderer change can land here.
            logger.error("accepted trajectory %s no longer parses: %s", submission_id[:12], refused)
            return await self._write(submission_id, {
                **base, "status": "unparseable", "reason": refused.reason, "graded_success": False,
                "replay": None, "replay_certified": False, "graded_at": self._clock()}, regrade=regrade)
        instance_id = self._source.instance_id(record["prompt_index"])
        item = {"submission_id": submission_id, "task_index": record["prompt_index"],
                "instance_id": instance_id, "mode": "grade", "final_diff": trajectory["final_diff"],
                "actions": []}
        graded = await self._dispatcher.decide(item)
        state = {"base": base, "instance_id": instance_id, "item": item, "graded": graded,
                 "received_at": record.get("received_at"),
                 "actions": [{"tool": a.tool, "arguments": a.arguments, "observation": a.observation}
                             for a in parsed.actions]}
        return await self._decide_replay(submission_id, state, regrade=regrade)

    async def _decide_replay(self, submission_id: str, state: dict, *, regrade: bool) -> dict | None:
        graded = state["graded"]
        # Only an ok decision is a grade; a single executor's failing grade is
        # no evidence against anyone, only "not a certified success".
        graded_success = graded.status == "ok" and bool((graded.result or {}).get("tests_passed"))
        replay = None
        if graded.status == "ok":
            draw = None
            if not graded_success:
                draw = await self._draw(submission_id, state["received_at"])
                if draw is None and regrade:
                    # A regrade has no rescan to wait on: check rather than skip.
                    draw = {"fraction": self._job.episode.replay_fraction_failed, "drawn": True,
                            "why": "round unpublished at regrade"}
                if draw is None:
                    self._awaiting_draw[submission_id] = state
                    return None              # the round is not out yet: the next rescan
            self._awaiting_draw.pop(submission_id, None)
            if graded_success or draw["drawn"]:
                voided = await self._replay_void(submission_id)
                if voided is not None:
                    # Voided by a replay before a crash kept its grade from
                    # being written: the grade is that replay, never a new one.
                    replay = self._replay_from_void(voided, draw)
                else:
                    replayed = await self._dispatcher.decide(
                        {**state["item"], "mode": "replay", "actions": state["actions"]})
                    replay = self._replay_document(submission_id, replayed, draw)
                    if replay["failed"]:
                        await self._confirmed_failure(submission_id, state["base"]["hotkey"],
                                                      replay)
            else:
                replay = {"drawn": False, "draw": draw}
        document = {**state["base"], "status": graded.status, "instance_id": state["instance_id"],
                    "graded_success": graded_success, "grade": graded.result,
                    "graded_by": list(graded.graded_by), "replay": replay,
                    "replay_certified": bool(replay and replay.get("certified")),
                    "graded_at": self._clock()}
        return await self._write(submission_id, document, regrade=regrade)

    async def _replay_void(self, submission_id: str) -> dict | None:
        reader = getattr(self._records, "read_voided", None)
        if reader is None:
            return None
        document = await reader(self._job.job_id, submission_id)
        if document is None or document.get("reason") != "replay_failed":
            return None
        return document

    @staticmethod
    def _replay_from_void(voided: dict, draw) -> dict:
        replay = voided.get("replay") or {}
        return {"drawn": True, "draw": draw, "status": "ok", "certified": False, "failed": True,
                "unconfirmed": False, "from_void": True,
                "replay_diff_equal": replay.get("replay_diff_equal"),
                "observations_compared": replay.get("observations_compared"),
                "observations_mismatched": list(replay.get("observations_mismatched") or []),
                "allowed": replay.get("allowed"), "graded_by": list(voided.get("graded_by") or []),
                "providers": list(voided.get("providers") or [])}

    @staticmethod
    def _replay_document(submission_id: str, decision, draw) -> dict:
        result = decision.result or {}
        ok = decision.status == "ok"
        certified = ok and replay_certified(result)
        # A sanction needs agreement across providers, whatever the dispatcher
        # promised: checked here again, never taken on trust.
        agreed = (len(set(decision.graded_by)) >= SANCTION_PROVIDERS
                  and len(set(getattr(decision, "providers", ()) or ())) >= SANCTION_PROVIDERS)
        unconfirmed = ok and not certified and not agreed
        if unconfirmed:
            logger.error("replay of %s failed without two providers agreeing (%s, providers %s); "
                         "no sanction, not certified", submission_id[:12], decision.graded_by,
                         getattr(decision, "providers", ()))
        compared = int(result.get("observations_compared") or 0)
        return {"drawn": True, "draw": draw, "status": decision.status, "certified": certified,
                "failed": ok and not certified and agreed, "unconfirmed": unconfirmed,
                "replay_diff_equal": result.get("replay_diff_equal"),
                "observations_compared": compared,
                "observations_mismatched": list(result.get("observations_mismatched") or []),
                "allowed": allowed_mismatches(compared), "graded_by": list(decision.graded_by),
                "providers": list(getattr(decision, "providers", ()) or ())}

    async def _draw(self, submission_id: str, received_at) -> dict | None:
        fraction = self._job.episode.replay_fraction_failed
        if fraction >= 1.0 or fraction <= 0.0:
            return {"fraction": fraction, "drawn": fraction >= 1.0}
        if self._beacon is None or self._round_at is None or received_at is None:
            # No way to draw: check rather than skip (a replay alone sanctions nobody).
            return {"fraction": fraction, "drawn": True, "why": "no beacon"}
        try:
            round_number = int(self._round_at(float(received_at))) + 1
        except Exception:
            return {"fraction": fraction, "drawn": True, "why": "round unknown"}
        randomness = await asyncio.to_thread(self._beacon, round_number)
        if randomness is None:
            return None
        return {"round": round_number, "fraction": fraction,
                "drawn": replay_drawn(randomness, submission_id, fraction)}

    async def _confirmed_failure(self, submission_id: str, hotkey: str, replay: dict) -> None:
        """The TOPLOC failure's path: escalate first (idempotent by id), then
        the create-only void, so a crash in between never counts twice."""
        now = self._clock()
        if self._miner_states is not None:
            await self._miner_states.update_many(
                {hotkey: lambda m: after_confirmed_failure(m, self._params, now, submission_id)})
        document = {"schema": VOIDED_SCHEMA, "submission_id": submission_id, "hotkey": hotkey,
                    "reason": "replay_failed", "voided_at": now, "graded_by": replay["graded_by"],
                    "providers": replay["providers"],
                    "replay": {k: replay[k] for k in ("replay_diff_equal", "observations_compared",
                                                      "observations_mismatched", "allowed")}}
        written = await self._records.write_voided(self._job.job_id, submission_id, document)
        logger.warning("corpus job %s: %s voided, replay failed (%s)", self._job.job_id,
                       submission_id[:12], document["replay"])
        if written and self.on_voided is not None:
            try:
                self.on_voided(submission_id, document)
            except Exception:
                logger.exception("void report for %s failed", submission_id[:12])

    async def _write(self, submission_id: str, document: dict, *, regrade: bool) -> dict:
        if regrade:
            generation, regraded_for = self._generation.get(submission_id, (1, []))
            document = {**document, "generation": generation, "regraded_for": regraded_for}
            written = await self._records.write_regrade(self._job.job_id, submission_id,
                                                        document, generation)
            if written is False:
                logger.warning("corpus job %s: %s already has regrade generation %d; this one "
                               "(by %s) was not written", self._job.job_id, submission_id[:12],
                               generation, document.get("graded_by"))
        else:
            await self._records.write_grade(self._job.job_id, submission_id, document)
        self._final.add(submission_id)
        # A regrade too: one decided alone by an executor quarantined later is redone.
        self._index(submission_id, document)
        self._regrade_if_quarantined(submission_id, document)
        return document

    def _is_quarantined(self, executor_id: str) -> bool:
        return (executor_id in self._quarantined
                or executor_id in set(getattr(self._dispatcher, "quarantined", ()) or ()))

    def _caught(self, document: dict) -> list[str]:
        """The quarantined executors that decided part of ``document`` alone."""
        lone = {_lone(document.get("graded_by")),
                _lone((document.get("replay") or {}).get("graded_by"))} - {None}
        return sorted(e for e in lone if self._is_quarantined(e))

    def _regrade_if_quarantined(self, submission_id: str, document: dict) -> None:
        """A decision made alone by an executor quarantined while this grade
        was in flight: the quarantine's regrade found nothing to redo yet, so
        the grade is held and redone here."""
        caught = self._caught(document)
        if not caught:
            return
        for executor_id in caught:
            self._sole.get(executor_id, set()).discard(submission_id)
        self._regrading.add(submission_id)
        logger.warning("corpus job %s: %s was decided alone by quarantined executor(s) %s; "
                       "regrading it", self._job.job_id, submission_id[:12], caught)
        self._spawn(self._regrade(submission_id))

    def _index(self, submission_id: str, document: dict) -> None:
        """Each decision one executor made alone, under that executor."""
        for deciders in (document.get("graded_by"),
                         (document.get("replay") or {}).get("graded_by")):
            lone = _lone(deciders)
            if lone is not None:
                self._sole.setdefault(lone, set()).add(submission_id)

    async def _index_unindexed(self) -> None:
        """Read the grades written before this process started, once. They
        stay in ``_unindexed`` (held while an executor is) until all are read."""
        pending = sorted(self._unindexed)
        gate = asyncio.Semaphore(INDEX_READ_CONCURRENCY)

        reader = getattr(self._records, "read_regrade", None)

        async def one(sid):
            async with gate:
                grade = await self._records.read_grade(self._job.job_id, sid)
                regrade = await reader(self._job.job_id, sid) if reader is not None else None
                return sid, grade, regrade

        for sid, grade, regrade in await asyncio.gather(*(one(sid) for sid in pending)):
            # Both: ``_regrade`` decides from the latest, whichever executor it is for.
            for document in (grade, regrade):
                if document is not None:
                    self._index(sid, document)
        self._unindexed -= set(pending)

    async def _latest(self, submission_id: str) -> dict | None:
        reader = getattr(self._records, "read_regrade", None)
        latest = await reader(self._job.job_id, submission_id) if reader is not None else None
        if latest is None:
            latest = await self._records.read_grade(self._job.job_id, submission_id)
        return latest

    async def _regrade(self, submission_id: str) -> None:
        """Regrade until the latest document has no decision made alone by a
        quarantined executor, decided from what is stored (so a restart
        resumes it, and never regrades twice what a regrade already redid).

        Bound: a regrade is written only for an executor that is not already
        in the latest document's ``regraded_for`` (one redo per submission
        per executor), and the dispatcher never leases to a quarantined
        executor, so each generation needs a newly quarantined lone decider;
        past ``MAX_REGRADE_GENERATIONS`` the submission stays held (never paid)
        and an operator is alerted."""
        if submission_id in self._regrades_inflight:
            return
        self._regrades_inflight.add(submission_id)
        try:
            async with self._gated():
                released = await self._regrade_until_clean(submission_id)
        except Exception:
            logger.exception("re-grading %s failed; retried on the next rescan",
                             submission_id[:12])
            self._regrade_retry.add(submission_id)
        else:
            self._regrade_retry.discard(submission_id)
            if released:
                # Payable again (unless voided: the settler reads voids).
                self._regrading.discard(submission_id)
        finally:
            self._regrades_inflight.discard(submission_id)

    async def _regrade_until_clean(self, submission_id: str) -> bool:
        for _ in range(MAX_REGRADE_GENERATIONS + 1):
            latest = await self._latest(submission_id)
            if latest is None:
                return True                      # nothing graded: nothing to hold
            caught = self._caught(latest)
            if not caught:
                await self._settle_regrade(submission_id, latest)
                return True
            generation = int(latest.get("generation") or 0)
            regraded_for = list(latest.get("regraded_for") or [])
            if generation >= MAX_REGRADE_GENERATIONS or set(caught) <= set(regraded_for):
                logger.error("corpus job %s: %s still decided alone by quarantined %s after "
                             "%d regrade(s); held from payment, needs an operator",
                             self._job.job_id, submission_id[:12], caught, generation)
                return False
            self._generation[submission_id] = (generation + 1,
                                               sorted(set(regraded_for) | set(caught)))
            try:
                if await self.grade_one(submission_id, regrade=True) is None:
                    return True                  # the record is gone or not an episode
            finally:
                self._generation.pop(submission_id, None)
        return False

    async def _settle_regrade(self, submission_id: str, latest: dict) -> None:
        """A regrade written before a restart that its void did not follow."""
        replay = latest.get("replay") or {}
        if latest.get("generation") is None or not replay.get("failed"):
            return
        reader = getattr(self._records, "read_voided", None)
        if reader is not None and await reader(self._job.job_id, submission_id) is not None:
            return
        await self._confirmed_failure(submission_id, latest["hotkey"], {
            "graded_by": list(replay.get("graded_by") or []),
            "providers": list(replay.get("providers") or []),
            **{k: replay.get(k) for k in ("replay_diff_equal", "observations_compared",
                                          "observations_mismatched", "allowed")}})

    def hold_executor(self, executor_id: str) -> None:
        """Synchronous, at the quarantine itself: what the executor decided
        alone stops being payable before anything awaits."""
        self._held_executors.add(executor_id)
        self._quarantined.add(executor_id)
        self._regrading |= self._sole.get(executor_id, set())
        # A grade waiting for its draw is dropped: graded again from the start.
        for sid, state in list(self._awaiting_draw.items()):
            if executor_id in set(state["graded"].graded_by):
                del self._awaiting_draw[sid]

    async def _regrade_executor_logged(self, executor_id: str) -> None:
        if executor_id in self._executor_regrades_inflight:
            return
        self._executor_regrades_inflight.add(executor_id)
        try:
            await self.regrade_executor(executor_id)
        except Exception:
            logger.exception("regrade of executor %s failed; retried on the next rescan",
                             executor_id)
        finally:
            self._executor_regrades_inflight.discard(executor_id)

    async def regrade_executor(self, executor_id: str) -> list[str]:
        """Grade again, without it, what a quarantined executor decided alone."""
        self.hold_executor(executor_id)
        await self._seed()
        if self._unindexed:
            # On failure the executor stays held; the rescan retries.
            await self._index_unindexed()
        sids = sorted(self._sole.pop(executor_id, set()))
        self._regrading |= set(sids)
        self._held_executors.discard(executor_id)
        await asyncio.gather(*(self._regrade(sid) for sid in sids))
        if sids:
            logger.warning("corpus job %s: re-graded %d submission(s) of quarantined executor %s",
                           self._job.job_id, len(sids), executor_id)
        return sids


__all__ = ["CorpusGrader", "GRADE_SCHEMA", "MAX_REGRADE_GENERATIONS"]
