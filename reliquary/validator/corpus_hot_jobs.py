"""The set of corpus jobs one validator process serves, changed without a restart.

Every ``JOB_REFRESH_SECONDS`` the registry is read again. A new active
corpus-generation entry on this process's model is wired (ledger migration,
renderer, auditor, settler, routes); one on another model is ignored; one this
binary cannot serve is refused. A retired entry stops admitting at once, keeps
auditing and settling until drained, and is then unwired. Every decision about
an entry is logged once, never once per refresh.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

JOB_REFRESH_SECONDS = 60.0

# Transient wiring failures of one task before they are logged as errors.
WIRE_FAILURES_LOUD = 5

OTHER_MODEL = "other_model"
REFUSED = "refused"


def _entry_profile(entry):
    from reliquary.validator.corpus_validator import _entry_profile as own_profile

    return own_profile(entry)


def eval_entry_screen(entry) -> tuple[str, str] | None:
    """The corpus control's answer to an ``order-eval-`` entry, before any
    manifest read: not its job (the eval control serves it)."""
    from reliquary.eval.prompt_source import is_eval_job_id

    if is_eval_job_id(getattr(entry, "job_id", "")):
        return OTHER_MODEL, "an evaluation job, served by the eval control"
    return None


def hot_job_refusal(entry, job, *, process_profile, process_contract: Mapping[str, Any],
                    fingerprint: str, profile_of=_entry_profile) -> tuple[str, str] | None:
    """Why a registry entry cannot join this running process, or None.

    ``(OTHER_MODEL, why)`` is an entry for another checkpoint: not ours, not an
    error. ``(REFUSED, why)`` is one on our checkpoint that this process still
    cannot serve. The environment the job draws from renders its rows through
    the PROCESS contract (``render_active_prompt``), so a job is served only
    when its own contract declares that environment exactly as the process does.
    """
    from reliquary.constants import PROTOCOL_GATED_PROMPT_SOURCES
    from reliquary.protocol.profiles import toploc_proof
    from reliquary.validator.corpus_validator import startup_refusal

    from reliquary.eval.prompt_source import is_eval_job_id

    if is_eval_job_id(getattr(entry, "job_id", "")):
        # Served by the eval control alone, whatever its model: two processes
        # auditing and settling one job would pay its records twice.
        return OTHER_MODEL, "an evaluation job, served by the eval control"
    if getattr(entry, "contract", None) is None:
        return REFUSED, "it carries no contract to check against the one this process runs"
    try:
        own = profile_of(entry)
    except ValueError as exc:
        return REFUSED, f"its contract cannot be read: {exc}"
    if (own.model_id, own.model_revision) != (process_profile.model_id,
                                              process_profile.model_revision):
        return OTHER_MODEL, (
            f"its model {own.model_id!r}@{own.model_revision!r} is not this process's "
            f"{process_profile.model_id!r}@{process_profile.model_revision!r}"
        )
    if toploc_proof(own) != toploc_proof(process_profile):
        return REFUSED, "its toploc proof is not the one this process audits with"
    refusal = startup_refusal(entry, job, own, fingerprint)
    if refusal:
        return REFUSED, refusal
    source = job.prompt_source
    declared = (entry.contract.get("environments") or {}).get(source)
    served = (process_contract.get("environments") or {}).get(source)
    if declared is None or declared != served:
        return REFUSED, (
            f"its contract declares prompt source {source!r} differently from the contract "
            "this process runs, which renders that source's rows; restart with the merged "
            "contract of `reliquary tasks contract`"
        )
    gate = PROTOCOL_GATED_PROMPT_SOURCES.get(source)
    if gate is not None and gate(entry.contract.get("protocol_version")) != gate(
        process_contract.get("protocol_version")
    ):
        return REFUSED, (
            f"it reads {source!r} under protocol version {entry.contract.get('protocol_version')}, "
            f"whose rows differ from this process's {process_contract.get('protocol_version')}"
        )
    return None


async def job_drained(*, auditor, records, job_id: str) -> bool:
    """Every accepted submission has a verdict and every verdict is settled:
    what `jobs status` prints as ``drained: yes``."""
    if await auditor.pending_ids():
        return False
    verdicts = set(await records.list_verdict_ids(job_id))
    state, _ = await records.read_settlement(job_id)
    state = state or {}
    return state.get("pending") is None and verdicts <= set(state.get("settled") or ())


class CorpusJobSet:
    """The wired jobs, their background tasks, and the refresh that changes them.

    ``wire(entry, cap, job)`` builds a job's wiring (it raises to refuse);
    ``router_for(wiring)`` its router; ``jobs_of(wiring)`` the coroutines that
    run for it (its auditor and its settler). A background task that fails is
    raised out of ``run``, as it was out of the process's gather before.
    """

    def __init__(self, *, routes, router_for, wire: Callable[..., Awaitable[Any]],
                 jobs_of: Callable[[Any], Iterable[Awaitable[Any]]],
                 read_entries: Callable[[], Awaitable[Mapping[str, Any]]] | None = None,
                 read_job: Callable[[str], Awaitable[Any]] | None = None,
                 admit: Callable[[Any, Any], tuple[str, str] | None] | None = None,
                 drained: Callable[[Any], Awaitable[bool]] | None = None,
                 refresh_every_seconds: float = JOB_REFRESH_SECONDS,
                 clock: Callable[[], float] = time.time,
                 screen: Callable[[Any], tuple[str, str] | None] | None = None,
                 on_unwired: Callable[[Any], None] | None = None) -> None:
        # ``screen(entry)`` decides from the entry alone, before any manifest
        # read; ``on_unwired(wiring)`` releases what a drained job held.
        self._screen = screen
        self._on_unwired = on_unwired
        self._routes = routes
        self._router_for = router_for
        self._wire = wire
        self._jobs_of = jobs_of
        self._read_entries = read_entries
        self._read_job = read_job
        self._admit = admit
        self._drained = drained
        self._refresh_every = refresh_every_seconds
        self._clock = clock
        self.served: dict[str, Any] = {}
        # Jobs retired and drained, with their final status: the route still answers.
        self.finished: dict[str, dict] = {}
        self._status_cache: dict[str, tuple[float, dict]] = {}
        self._status_locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, list[asyncio.Task]] = {}
        # Task ids already decided against (ignored or refused): logged once.
        self._passed_over: set[str] = set()
        self._failure: asyncio.Future | None = None
        # One refresh at a time: two interleaved would wire one entry twice.
        self._refresh_lock = asyncio.Lock()
        # Jobs retired as of the end of the last refresh: only those may be unwired.
        self._retired_before: set[str] = set()
        # Transient wiring failures per task id, retried every refresh.
        self._wire_failures: collections.Counter = collections.Counter()

    @property
    def refreshing(self) -> bool:
        return self._read_entries is not None

    def task_ids(self) -> set[str]:
        return {str(w.entry.task_id) for w in self.served.values()}

    def hot_task_ids(self) -> set[str]:
        """The tasks wired after boot (their archives are not in RELIQUARY_TASK_ID)."""
        return {str(w.entry.task_id) for w in self.served.values()
                if not getattr(w, "at_boot", False)}

    def is_retired(self, job_id: str) -> bool:
        return job_id in self._routes.retired

    def adopt(self, wiring) -> None:
        """A job wired at boot: its routes already exist, start its tasks."""
        wiring.at_boot = True
        self.served[str(wiring.entry.job_id)] = wiring
        self._start(wiring)

    def _start(self, wiring) -> None:
        job_id = str(wiring.entry.job_id)
        tasks = [asyncio.ensure_future(c) for c in self._jobs_of(wiring)]
        for task in tasks:
            task.add_done_callback(self._watch)
        self._tasks[job_id] = tasks

    def _watch(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is None:
            return
        if self._failure is not None and not self._failure.done():
            self._failure.set_exception(task.exception())

    async def refresh(self) -> None:
        """One pass over the registry. A read that fails changes nothing."""
        async with self._refresh_lock:
            await self._refresh()

    async def _refresh(self) -> None:
        try:
            entries = await self._read_entries()
        except Exception:
            logger.exception("corpus job refresh: the task registry could not be read; "
                             "serving the same jobs until the next refresh")
            return
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

        corpus = {task_id: entry for task_id, entry in sorted(entries.items())
                  if getattr(entry, "mechanism", None) == MECHANISM_CORPUS_GENERATION}
        by_task = {str(w.entry.task_id): job_id for job_id, w in self.served.items()}
        for task_id, job_id in by_task.items():
            entry = corpus.get(task_id)
            wiring = self.served[job_id]
            if entry is None or entry.status != "active" or str(entry.job_id) != job_id:
                if not self.is_retired(job_id):
                    logger.info("corpus task %s (job %s) is %s: admission stopped, draining",
                                task_id, job_id, "retired" if entry is not None else "gone")
                    self._routes.retire(job_id)
                continue
            cap = float(entry.params["cap"])
            if cap != float(wiring.cap):
                logger.info("corpus task %s cap %s -> %s", task_id, wiring.cap, cap)
                wiring.cap = cap
                wiring.settler.set_cap(cap)
        for task_id, entry in corpus.items():
            if entry.status != "active" or task_id in by_task or task_id in self._passed_over:
                continue
            await self._consider(entry)
        # Never in the refresh that retired it: a submit admitted just before
        # may not have written its record yet.
        for job_id in [j for j in self.served if j in self._retired_before]:
            await self._maybe_unwire(job_id)
        self._retired_before = {j for j in self.served if self.is_retired(j)}

    async def _consider(self, entry) -> None:
        task_id, job_id = str(entry.task_id), str(entry.job_id)
        if job_id in self.served or job_id in self.finished:
            self._pass_over(task_id, REFUSED, f"job {job_id!r} was already served by this process")
            return
        screened = self._screen(entry) if self._screen is not None else None
        if screened is not None:
            self._pass_over(task_id, *screened)
            return
        try:
            job = await self._read_job(job_id)
        except Exception:
            # Transient: not remembered, tried again next refresh.
            logger.exception("corpus task %s: manifest of job %s could not be read", task_id, job_id)
            return
        if job is None:
            self._pass_over(task_id, REFUSED, f"job {job_id!r} has no manifest")
            return
        verdict = self._admit(entry, job)
        if verdict is not None:
            self._pass_over(task_id, *verdict)
            return
        try:
            wiring = await self._wire(entry, float(entry.params["cap"]), job)
            router = self._router_for(wiring)
        except ValueError as exc:
            # Deterministic (the renderer, the prompt source): remembered.
            logger.exception("corpus task %s refused: job %s could not be wired", task_id, job_id)
            self._pass_over(task_id, REFUSED, f"wiring failed: {exc}", log=False)
            return
        except Exception:
            # A store or transport fault: tried again next refresh, never taking
            # the other jobs down.
            self._wire_failures[task_id] += 1
            failures = self._wire_failures[task_id]
            logger.log(logging.ERROR if failures >= WIRE_FAILURES_LOUD else logging.WARNING,
                       "corpus task %s: wiring job %s failed (%d time(s)); retrying next refresh",
                       task_id, job_id, failures, exc_info=True)
            return
        self._wire_failures.pop(task_id, None)
        self._routes.add(job_id, router, contract=entry.contract)
        self.served[job_id] = wiring
        self._start(wiring)
        logger.info("corpus task %s wired: job %s now served without a restart", task_id, job_id)

    def _pass_over(self, task_id: str, kind: str, why: str, *, log: bool = True) -> None:
        self._passed_over.add(task_id)
        if log:
            if kind == OTHER_MODEL:
                logger.info("corpus task %s ignored: %s", task_id, why)
            else:
                logger.warning("corpus task %s refused: %s", task_id, why)

    async def _maybe_unwire(self, job_id: str) -> None:
        wiring = self.served[job_id]
        if self._routes.in_flight[job_id]:
            return
        try:
            # The gate is closed (retired) and nothing is in flight: no record can
            # appear after this check.
            if not await self._drained(wiring) or self._routes.in_flight[job_id]:
                return
        except Exception:
            logger.exception("corpus job %s: drain check failed; retrying next refresh", job_id)
            return
        try:
            final = await self._compute_status(job_id, drained=True)
        except Exception:
            # Unwiring must not wait on a status read: the last one, marked drained.
            logger.warning("corpus job %s: final status unreadable; keeping the last one", job_id)
            last = self._status_cache.get(job_id)
            final = {**(last[1] if last else {"job_id": job_id}), "state": "drained"}
        for task in self._tasks.pop(job_id, ()):
            task.cancel()
        self._routes.remove(job_id)
        del self.served[job_id]
        self.finished[job_id] = final
        self._status_cache.pop(job_id, None)
        self._status_locks.pop(job_id, None)
        if self._on_unwired is not None:
            try:
                self._on_unwired(wiring)
            except Exception:
                logger.exception("corpus job %s: releasing its wiring failed", job_id)
        logger.info("corpus job %s drained and unwired", job_id)

    async def _compute_status(self, job_id: str, *, drained: bool = False) -> dict:
        from reliquary.validator.corpus_job_status import job_status

        wiring = self.served[job_id]
        # The wired manifest: one ledger GET per recompute, no manifest read.
        job, state = await self._routes.routers[job_id].ledger_state(wiring.job)
        return job_status(job_id=job_id, job=job or wiring.job, slots=state.slots,
                          stats=wiring.stats, settled=getattr(wiring.settler, "settled_count", 0),
                          totals=getattr(wiring.settler, "totals", None),
                          retired=self.is_retired(job_id), drained=drained)

    async def status(self, job_id: str) -> dict | None:
        """The public status of a served or drained job, recomputed at most once
        per ``STATUS_CACHE_SECONDS``; a failed recompute serves the last one."""
        from reliquary.validator.corpus_job_status import STATUS_CACHE_SECONDS

        if job_id in self.finished:
            return self.finished[job_id]
        if job_id not in self.served:
            return None
        now = self._clock()
        cached = self._status_cache.get(job_id)
        if cached is not None and now - cached[0] < STATUS_CACHE_SECONDS:
            return cached[1]
        lock = self._status_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            # One recompute per period, however many requests wait on it.
            cached = self._status_cache.get(job_id)
            if cached is not None and self._clock() - cached[0] < STATUS_CACHE_SECONDS:
                return cached[1]
            return await self._recompute(job_id, cached)

    async def _recompute(self, job_id: str, cached) -> dict:
        now = self._clock()
        try:
            fresh = await self._compute_status(job_id)
        except Exception:
            if cached is None:
                raise
            logger.warning("corpus status of %s: ledger unreadable, serving the last one", job_id)
            return cached[1]
        if job_id in self.served:
            self._status_cache[job_id] = (now, fresh)
        return fresh

    async def run(self) -> None:
        """Refresh forever (when a registry reader is given) and raise the first
        background task failure."""
        self._failure = asyncio.get_running_loop().create_future()
        for tasks in self._tasks.values():
            for task in tasks:
                if task.done():
                    self._watch(task)
        try:
            while True:
                if self.refreshing:
                    await self.refresh()
                done, _ = await asyncio.wait({self._failure}, timeout=self._refresh_every)
                if done:
                    self._failure.result()
        finally:
            for tasks in self._tasks.values():
                for task in tasks:
                    task.cancel()


__all__ = [
    "CorpusJobSet",
    "JOB_REFRESH_SECONDS",
    "OTHER_MODEL",
    "REFUSED",
    "eval_entry_screen",
    "hot_job_refusal",
    "job_drained",
]
