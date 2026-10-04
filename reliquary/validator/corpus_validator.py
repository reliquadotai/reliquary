"""The validator of a corpus task: one process, one card, no RL machinery.

It serves the submission route, audits every accepted submission on its own
GPU, settles verified tokens into this task's archives, and sets weights only
when told to (the RL validator's setter already pays every task).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import re
import time
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Response

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC

logger = logging.getLogger(__name__)

# How long `/corpus/tasks` serves one registry read.
TASKS_CACHE_SECONDS = 60.0
# Store connections shared by every job's auditor and settler (32 writes, 16 reads a job).
JUDGE_POOL_CONNECTIONS = 64

_HEX64 = re.compile(r"[0-9a-f]{64}")


def startup_refusal(entry, job, profile, local_fingerprint: str) -> str | None:
    toploc = [p for p in getattr(profile, "proofs", ()) if p.scheme == PROOF_SCHEME_TOPLOC]
    if not toploc:
        return "the task contract names no toploc proof; a corpus task is paid only on audited work"
    if toploc[0].mode != "enforce":
        return "the task contract's toploc proof is not enforce"
    # The contract pins both the repo and the revision its proof thresholds
    # were measured on; a job declared against either the wrong repo or the
    # wrong revision is not the checkpoint the contract describes.
    if profile.model_id != job.checkpoint_repo or profile.model_revision != job.checkpoint_revision:
        return (
            f"the task contract's model {profile.model_id!r}@{profile.model_revision!r} is not "
            f"the job's checkpoint {job.checkpoint_repo!r}@{job.checkpoint_revision!r}"
        )
    if local_fingerprint != job.checkpoint_sha256:
        return "the loaded checkpoint's fingerprint does not match the job's checkpoint_sha256"
    return None


def _contract_toploc(entry):
    """The toploc proof an entry's own carried contract declares, or None."""
    contract = getattr(entry, "contract", None)
    if contract is None:
        return None
    from reliquary.protocol.profiles import profile_from_contract, toploc_proof

    return toploc_proof(profile_from_contract(contract))


def _entry_profile(entry):
    """The profile an entry's own carried contract describes."""
    from reliquary.protocol.profiles import profile_from_contract

    return profile_from_contract(entry.contract)


def multi_job_refusal(pairs, *, proof_of=_contract_toploc,
                      process_contract=None) -> str | None:
    """Why several ``(entry, job)`` pairs cannot share one loaded model, or None.

    One process audits every job with one checkpoint and one toploc proof, so
    the jobs must name the same checkpoint and the entries the same proof; two
    tasks naming one job would pay the same records twice. The environment a
    job draws from renders its rows through the process contract, so there it
    must be exactly the one the job's own task declares.
    """
    first_entry, first_job = pairs[0]
    if process_contract is not None:
        from reliquary.eval.prompt_source import declared_environment

        served = process_contract.get("environments") or {}
        for entry, job in pairs:
            contract = getattr(entry, "contract", None) or {}
            name = declared_environment(contract, job.prompt_source) or job.prompt_source
            own = (contract.get("environments") or {}).get(name)
            if own is None or served.get(name) != own:
                return (
                    f"task {entry.task_id!r}'s contract declares prompt source "
                    f"{job.prompt_source!r} differently from the contract this process runs; "
                    "start it with the merged contract of `reliquary tasks contract`"
                )
    seen: dict[str, str] = {}
    for entry, job in pairs:
        if job.job_id in seen:
            return f"tasks {seen[job.job_id]!r} and {entry.task_id!r} both declare job {job.job_id!r}"
        seen[job.job_id] = entry.task_id
    for entry, job in pairs[1:]:
        for field in ("checkpoint_repo", "checkpoint_revision", "checkpoint_sha256"):
            if getattr(job, field) != getattr(first_job, field):
                return (
                    f"job {job.job_id!r} declares {field} {getattr(job, field)!r} but job "
                    f"{first_job.job_id!r} declares {getattr(first_job, field)!r}; one "
                    "validator serves several jobs only on one checkpoint"
                )
        if proof_of(entry) != proof_of(first_entry):
            return (
                f"task {entry.task_id!r} carries a different toploc proof than task "
                f"{first_entry.task_id!r}; one validator audits every job with one proof"
            )
    return None


# Verified rounds, for the life of the process (rounds never change), and on
# disk when RELIQUARY_CORPUS_DRAND_CACHE names a file (one JSON line a round):
# a restarted judge does not race the rounds of its backlog again.
_BEACONS: dict[int, str] = {}
_DISK_LOADED: set[str] = set()
_BEACONS_LOCK = __import__("threading").Lock()
DRAND_CACHE_ENV = "RELIQUARY_CORPUS_DRAND_CACHE"


def _cached_beacon(round_number: int) -> str | None:
    import json
    import os

    path = os.environ.get(DRAND_CACHE_ENV)
    with _BEACONS_LOCK:
        if path and path not in _DISK_LOADED:
            _DISK_LOADED.add(path)
            try:
                with open(path) as handle:
                    for line in handle:
                        try:
                            doc = json.loads(line)
                            if _HEX64.fullmatch(doc["randomness"]):
                                _BEACONS[int(doc["round"])] = doc["randomness"]
                        except (ValueError, KeyError, TypeError):
                            continue
            except FileNotFoundError:
                pass
        return _BEACONS.get(round_number)


def _remember_beacon(round_number: int, randomness: str) -> None:
    import json
    import os

    path = os.environ.get(DRAND_CACHE_ENV)
    with _BEACONS_LOCK:
        _BEACONS[round_number] = randomness
        if path:
            try:
                with open(path, "a") as handle:
                    handle.write(json.dumps({"round": round_number, "randomness": randomness}) + "\n")
            except OSError:
                logger.warning("drand cache %s not writable", path, exc_info=True)


def drand_beacon(round_number: int) -> str | None:
    """A verified round: from the cache, else the first relay whose answer
    verifies by BLS here, else two agreeing relays, else the cross-checked
    path below (``_drand_beacon_checked``)."""
    from reliquary.infrastructure import drand

    cached = _cached_beacon(round_number)
    if cached is not None:
        return cached
    beacon = None
    for fetch in (drand.get_verified_beacon, drand.get_agreed_beacon):
        try:
            beacon = fetch(round_number)
        except Exception:
            logger.debug("drand %s for round %d failed", fetch.__name__, round_number,
                         exc_info=True)
        if beacon:
            break
    randomness = beacon["randomness"] if beacon else _drand_beacon_checked(round_number)
    if randomness is not None:
        _remember_beacon(round_number, randomness)
    return randomness


def _drand_beacon_checked(round_number: int) -> str | None:
    """The randomness of drand round ``round_number``, lowercased, or
    ``None``: a fetch error, a relay answering for the wrong round, malformed
    randomness, or a signature ``verify_beacon_signature`` cannot confirm --
    which includes ``bittensor_drand`` not being installed in this image, where
    it already fails closed (returns ``False``, never raises). ``None`` means
    "audit this submission" to the caller (spec §6): never a guess at a round
    this validator could not actually check.
    """
    from reliquary.infrastructure import drand

    try:
        data = drand.get_drand_beacon(round_id=round_number, use_fallback=False)
    except Exception:
        logger.warning("drand round %d unavailable; auditing", round_number, exc_info=True)
        return None
    if data.get("round") != round_number:
        logger.error(
            "drand asked for round %d, relay answered round %r; auditing",
            round_number, data.get("round"),
        )
        return None
    randomness = data.get("randomness")
    if not isinstance(randomness, str):
        logger.error("drand round %d gave non-string randomness %r; auditing",
                     round_number, randomness)
        return None
    randomness = randomness.lower()
    if not _HEX64.fullmatch(randomness):
        logger.error("drand round %d gave malformed randomness %r; auditing",
                     round_number, randomness)
        return None
    if not drand.verify_beacon_signature(
        data.get("chain_hash"), round_number, randomness, data.get("signature")
    ):
        logger.error("drand round %d failed signature verification; auditing", round_number)
        return None
    return randomness


def make_round_at(genesis_time: float, period: float):
    """The first drand round published strictly after ``t`` (spec §6): round
    ``r`` is published at ``genesis_time + (r - 1) * period``, so the smallest
    ``r`` whose publication time exceeds ``t`` is
    ``floor((t - genesis_time) / period) + 2``.
    """

    def round_at(t: float) -> int:
        return math.floor((t - genesis_time) / period) + 2

    return round_at


class LazyRoundAt:
    """``round_at``, resolved on first use rather than once at process
    startup: an ``/info`` fetch that fails while this validator boots must
    not turn sampling off for the rest of its life (fix round 1, finding 2).

    A successful resolution is cached forever -- a chain's genesis time and
    period never change once published. A failed one is retried at most once
    every ``retry_seconds``, never on every call (the drand relays are not
    free). While unresolved, calling this raises instead of returning an int;
    ``CorpusAuditor`` catches that and audits the submission, exactly as it
    does a missing beacon (never guesses a round from nothing).
    """

    def __init__(self, *, retry_seconds: float = 60.0, clock=time.time) -> None:
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._resolved = None
        self._last_attempt: float | None = None

    def __call__(self, t: float) -> int:
        if self._resolved is None:
            self._resolve()
        return self._resolved(t)

    def _resolve(self) -> None:
        now = self._clock()
        if self._last_attempt is not None and now - self._last_attempt < self._retry_seconds:
            raise RuntimeError(
                "drand chain genesis/period not resolved yet; retry throttled"
            )
        self._last_attempt = now
        from reliquary.infrastructure import drand

        chain = drand.get_current_chain()
        genesis_time, period = chain.get("genesis_time"), chain.get("period")
        if genesis_time is None or period is None:
            logger.warning(
                "drand chain genesis/period not yet known (genesis_time=%r period=%r); "
                "auditing every sampled submission until they resolve",
                genesis_time, period,
            )
            raise RuntimeError("drand chain genesis/period not yet known")
        self._resolved = make_round_at(genesis_time, period)


def build_corpus_audit_wiring(*, entry, job, records):
    """This task's audit parameters, per-hotkey state, ban check, and drand
    draw -- everything ``run_corpus_validator`` hands the auditor and the
    route, assembled apart from the model and HTTP setup so it is cheap to
    build in a test.

    ``entry.params`` may carry no ``audit_*`` keys at all:
    ``AuditParams.from_params`` then defaults to ``q = 1.0``, V0's full audit.
    ``beacon`` and ``round_at`` are always real callables, never ``None``:
    resolving the drand chain's genesis time and period is ``round_at``'s own
    job now (``LazyRoundAt``), deferred to first use and retried on its own
    schedule, so a chain that is not yet known when this process starts still
    turns sampling on later without a restart.
    """
    from reliquary.corpus.audit_policy import AuditParams, effective_state
    from reliquary.validator.corpus_miner_states import MinerStates

    params = AuditParams.from_params(entry.params)
    miner_states = MinerStates(records, job.job_id)

    async def is_banned(hotkey: str) -> bool:
        state = await miner_states.get(hotkey)
        return effective_state(state, time.time(), params) == "banned"

    return params, miner_states, is_banned, drand_beacon, LazyRoundAt()


def wire_job_judge(w, *, records, judge_records, judge_threads, archives, proof, model,
                   tokenizer, gpu_lock=None, remote=None, scorer=None, vocab_size=None,
                   arrivals_covered=None, auditor_kwargs=None) -> None:
    """One job's auditor, settler and status books, on ``w`` (which carries
    ``entry``, ``job``, ``cap`` and ``stats``): what judges and pays the job,
    in whichever process runs it.

    ``records`` is the route-side store (the ban check, the status books);
    ``judge_records`` the judges' own (connections and codec threads).
    ``scorer``/``vocab_size`` stand for ``model`` in a process that has none.
    """
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from reliquary.validator.corpus_miner_states import MinerStates
    from reliquary.validator.corpus_miner_status import (
        MinerBook, feed, proof_thresholds, read_recent_windows,
    )
    from reliquary.validator.corpus_settlement import (
        SETTLE_FULL_LIST_SECONDS, CorpusSettler, settler_fed,
    )

    params, miner_states, w.is_banned, beacon, round_at = build_corpus_audit_wiring(
        entry=w.entry, job=w.job, records=records
    )
    # The miner status route's view: in memory, fed by the same reports.
    w.audit_params, w.miner_states = params, miner_states
    w.miners = MinerBook(job_id=w.job.job_id, task_id=w.entry.task_id, records=records,
                         read_windows=read_recent_windows,
                         thresholds=proof_thresholds(proof))
    on_verdict, on_settled = feed(w.stats, w.miners)
    # `entry.cap` does not exist on `TaskEntry` (the cap lives in
    # `params["cap"]`); the CLI passes the value `TaskConfig` already resolved.
    # Fed by the auditor: the store is listed only as the net.
    from reliquary.validator.corpus_periods import is_period_task

    grader = getattr(w, "grader", None)
    if is_period_task(w.entry):
        # Paid on its own clock (design 2026-10-03): closes a period when the
        # auditor holds nothing undecided received in it -- and, for an episode
        # job, its grader nothing ungraded or held (ruling P21).
        from reliquary.infrastructure.corpus_period_store import R2PeriodArchives
        from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler, oldest_of

        sources = [lambda: w.auditor.oldest_pending_received_at()]
        if grader is not None:
            sources.append(lambda: grader.oldest_unready_received_at())
        w.settler = CorpusPeriodSettler(
            task_id=w.entry.task_id, job_id=w.job.job_id, cap=w.cap, records=judge_records,
            archives=R2PeriodArchives(guard=archives), oldest_pending=oldest_of(*sources),
            on_settled=on_settled, full_list_every_seconds=SETTLE_FULL_LIST_SECONDS,
            executor=judge_threads.codec,
            ready=grader.ready if grader is not None else None)
    else:
        w.settler = CorpusSettler(task_id=w.entry.task_id, job_id=w.job.job_id, cap=w.cap,
                                  records=judge_records, archives=archives,
                                  on_settled=on_settled,
                                  full_list_every_seconds=SETTLE_FULL_LIST_SECONDS,
                                  executor=judge_threads.codec,
                                  ready=grader.ready if grader is not None else None)
    w.auditor = CorpusAuditor(job_id=w.job.job_id, records=judge_records, model=model,
                              tokenizer=tokenizer, proof=proof, params=params,
                              miner_states=MinerStates(judge_records, w.job.job_id),
                              beacon=beacon, round_at=round_at,
                              gpu_lock=gpu_lock,
                              on_verdict=settler_fed(w.settler, on_verdict),
                              remote=remote, on_voided=w.miners.voided,
                              threads=judge_threads, scorer=scorer, vocab_size=vocab_size,
                              arrivals_covered=arrivals_covered,
                              **(auditor_kwargs or {}))
    w.settler.on_window = w.miners.window
    if getattr(w, "grader", None) is not None:
        w.grader.on_voided = w.miners.voided


async def settle_forever(task_id: str, settler, every_seconds: float) -> None:
    while True:
        try:
            window = await settler.settle_once()
            if window is not None:
                logger.info("corpus task %s settled window %d", task_id, window)
        except Exception:
            logger.exception("corpus settlement failed; retrying next period")
        await asyncio.sleep(every_seconds)


def judge_jobs(w, *, intake_only: bool, settle_every_seconds: float, settle=None) -> list:
    """The coroutines that judge one job in this process: the auditor and the
    settler unless intake-only, and the grader of an episode job."""
    if getattr(w, "judge_link", None) is not None:
        return []
    settle = settle or settle_forever
    jobs = ([] if intake_only else
            [w.auditor.run(), settle(w.entry.task_id, w.settler, settle_every_seconds)])
    if getattr(w, "grader", None) is not None:
        jobs.append(w.grader.run())
    return jobs


def grade_refusal(job, dispatcher) -> None:
    """``ValueError`` (permanent until a restart) for an episode job that
    ``dispatcher`` cannot grade: none (the process started without an episode
    job), or another env pin."""
    pin = (job.episode.env.package, job.episode.env.version)
    if dispatcher is None:
        raise ValueError(f"episode job {job.job_id!r} joined a validator started without "
                         "one; restart it to grade the job")
    if tuple(dispatcher.env_pin) != pin:
        raise ValueError(f"this validator grades {tuple(dispatcher.env_pin)}, episode job "
                         f"{job.job_id!r} pins {pin}; restart it to grade the job")


async def warm_drand_chain(executor) -> None:
    """Resolve and cache the drand chain's genesis and period on ``executor``
    (blocking HTTP), never the loop. A failure is left to first use, as before."""
    from reliquary.validator.corpus_judge_threads import run_in

    def resolve() -> None:
        from reliquary.infrastructure import drand

        drand.get_current_chain()

    try:
        await run_in(executor, resolve)
    except Exception:  # noqa: BLE001
        logger.warning("drand chain not resolved at start; resolved at first use",
                       exc_info=True)


def wire_job_grader(w, *, records, judge_records, dispatcher, parse_executor=None,
                    beacon_executor=None) -> None:
    """An episode job's grader, on ``w`` (which carries ``entry``, ``job`` and
    ``episode_intake``), leasing to ``dispatcher``; once per job. A job this
    process cannot grade (no dispatcher, another env pin) is refused with a
    ``ValueError``, which the job set takes as permanent until a restart.

    The grader parses with ``w.grade_renderer`` when set (its own lock, never
    the intake's) on ``parse_executor``, and reads drand on ``beacon_executor``:
    in a process serving submit routes, never their default executor."""
    if w.job.episode is None or getattr(w, "grader", None) is not None:
        return
    grade_refusal(w.job, dispatcher)
    from reliquary.validator.corpus_grading import CorpusGrader

    params, miner_states, _, beacon, round_at = build_corpus_audit_wiring(
        entry=w.entry, job=w.job, records=records)
    w.grader = CorpusGrader(job=w.job, records=judge_records, dispatcher=dispatcher,
                            renderer=(getattr(w, "grade_renderer", None)
                                      or w.episode_intake.renderer),
                            source=w.episode_intake.source,
                            params=params, miner_states=miner_states, beacon=beacon,
                            round_at=round_at, parse_executor=parse_executor,
                            beacon_executor=beacon_executor)
    for executor_id in sorted(getattr(dispatcher, "quarantined", ()) or ()):
        # Quarantined before this grader existed (before a restart, or before
        # a hot add): held now, before any settlement; its first rescan regrades.
        w.grader.hold_executor(executor_id)


async def regrade_everywhere(graders, executor_id: str) -> list:
    """The grade quarantine listener: every grader regrades what the executor
    decided alone; one grader's failure is logged and never stops the others."""
    results = await asyncio.gather(*(g.regrade_executor(executor_id) for g in graders),
                                   return_exceptions=True)
    done = []
    for grader, result in zip(graders, results):
        if isinstance(result, BaseException):
            job = getattr(getattr(grader, "_job", None), "job_id", "?")
            logger.error("regrade of quarantined executor %s failed for job %s: %r",
                         executor_id, job, result)
        else:
            done.append(result)
    return done


def _config_vocab_size(directory) -> int:
    import json

    config = json.loads((directory / "config.json").read_text())
    size = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    if not isinstance(size, int):
        raise RuntimeError(f"{directory}/config.json names no vocab_size")
    return size


class JudgedElsewhere:
    """The front's stand-ins for a job judged in another process: what the
    job set calls on a wiring's auditor and settler."""

    settled_count = None
    totals = None

    def __init__(self, records, job_id: str) -> None:
        self._records, self._job_id = records, job_id

    def set_cap(self, cap: float) -> None:
        # The judge process reads the registry itself (`refresh_caps`).
        return None

    async def pending_ids(self) -> list[str]:
        """For the drain check: listed, as the auditor's own net does."""
        submitted = await self._records.list_submission_ids(self._job_id)
        judged = set(await self._records.list_verdict_ids(self._job_id))
        return [sid for sid in submitted if sid not in judged]


def wire_job_front_only(w, *, records, link) -> None:
    """A job whose judge runs in another process: the front keeps its ban
    check and miner states, and hands each accepted id to ``link``."""
    params, miner_states, w.is_banned, _, _ = build_corpus_audit_wiring(
        entry=w.entry, job=w.job, records=records
    )
    w.audit_params, w.miner_states = params, miner_states
    w.judge_link = link
    w.miners = None
    w.auditor = w.settler = JudgedElsewhere(records, str(w.job.job_id))
    job_id = str(w.job.job_id)

    def on_accepted(submission_id: str) -> None:
        w.stats.accepted()
        link.accepted(job_id, submission_id)

    w.on_accepted = on_accepted


SPLIT_EPISODE_REFUSAL = ("episode job {job_id!r} is in a split judge group, but judge processes "
                         "host no grader; leave it out of RELIQUARY_CORPUS_SPLIT_JUDGES (the front "
                         "audits, grades and settles it)")


def split_episode_refusal(split, job):
    """``(REFUSED, why)`` for an episode job a split validator cannot serve
    (one a judge process would judge: judge processes host no grader), else
    None. The front serves every other episode job as the single process does.
    A permanent refusal, not a transient one to retry."""
    if split is None or getattr(job, "episode", None) is None:
        return None
    if str(job.job_id) not in getattr(split, "links", {}):
        return None
    from reliquary.validator.corpus_hot_jobs import REFUSED

    return REFUSED, SPLIT_EPISODE_REFUSAL.format(job_id=str(job.job_id))


def build_corpus_app(*, entry, job, store, records, tokenizer, renderer, verify_signature,
                     auditor, proof_chunk_tokens, prompt_job_for=None,
                     vocab_size=None, is_banned=None, registration=None,
                     contract=None, seen_index=None, verify_skip_signature=None) -> FastAPI:
    return build_corpus_jobs_app(
        jobs=[SimpleNamespace(entry=entry, job=job, renderer=renderer, auditor=auditor,
                              is_banned=is_banned, seen_index=seen_index)],
        store=store, records=records, tokenizer=tokenizer, verify_signature=verify_signature,
        proof_chunk_tokens=proof_chunk_tokens, prompt_job_for=prompt_job_for,
        vocab_size=vocab_size, registration=registration, contract=contract,
        verify_skip_signature=verify_skip_signature,
    )


def build_corpus_jobs_app(*, jobs, store, records, tokenizer, verify_signature,
                          proof_chunk_tokens, prompt_job_for=None, vocab_size=None,
                          registration=None, contract=None,
                          verify_skip_signature=None) -> FastAPI:
    """One app over one ``build_corpus_router`` per job (each with its own
    renderer, auditor queue and ban check); the registration gate is shared.

    The legacy routes answer for the first job in ``jobs``; the job-scoped ones
    for every job. ``app.state.corpus_routes`` is the live routing table and
    ``app.state.corpus_router_for`` builds a router for a job wired later.
    """
    from reliquary.validator.corpus_service import (
        CorpusJobRoutes, build_corpus_jobs_router, build_corpus_router, prompt_job_for_spec,
    )

    def router_for(served):
        return build_corpus_router(
            job_id=str(served.entry.job_id), store=store, tokenizer=tokenizer,
            renderer=served.renderer, verify_signature=verify_signature,
            verify_skip_signature=verify_skip_signature,
            prompt_job_for=(getattr(served, "prompt_job_for", None) or prompt_job_for
                            or prompt_job_for_spec),
            records=records,
            on_accepted=getattr(served, "on_accepted", None) or served.auditor.enqueue,
            proof_chunk_tokens=proof_chunk_tokens,
            vocab_size=vocab_size, is_banned=getattr(served, "is_banned", None),
            registration=registration,
            seen_index=getattr(served, "seen_index", None),
            episode_intake=getattr(served, "episode_intake", None), job=getattr(served, "job", None),
        )

    # Miners have no registry access: each job's own task contract. With one
    # job at boot, the contract this process runs.
    routes = CorpusJobRoutes()
    for served in jobs:
        routes.add(str(served.entry.job_id), router_for(served),
                   contract=contract if len(jobs) == 1 else getattr(served.entry, "contract", None),
                   prompt_source=getattr(served.job, "prompt_source", None))
    app = FastAPI()
    app.include_router(build_corpus_jobs_router(routes, legacy=True))
    app.state.corpus_routes = routes
    app.state.corpus_router_for = router_for
    default_job = routes.default
    legacy_contract = routes.contracts.get(default_job)
    if len(routes.routers) > 1:
        logger.info("corpus legacy paths serve job %s (first listed); job-scoped paths serve %s",
                    default_job, sorted(routes.routers))

    @app.get("/corpus/contract")
    async def corpus_contract() -> dict:
        if legacy_contract is None:
            raise HTTPException(status_code=404, detail="corpus_contract_unknown")
        return legacy_contract

    @app.get("/corpus/jobs/{job_id}/status")
    async def corpus_job_status(job_id: str) -> dict:
        # Public: counts only. The job set is attached once the process runs.
        job_set = getattr(app.state, "corpus_jobs", None)
        try:
            status = await job_set.status(job_id) if job_set is not None else None
        except Exception as exc:
            logger.warning("corpus status of %s unavailable: %r", job_id, exc)
            raise HTTPException(status_code=503, detail="corpus_status_unavailable") from exc
        if status is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return status

    tasks_cache: dict = {}

    async def read_tasks() -> dict:
        reader = getattr(app.state, "task_registry_reader", None)
        if reader is None:
            from reliquary.infrastructure.task_registry_store import read_registry as reader
        now = time.time()
        entries, _ = await reader()
        tasks = [
            {"task_id": e.task_id, "mechanism": e.mechanism, "cap": float(e.params["cap"]),
             "status": e.status, "job_id": getattr(e, "job_id", None),
             "retired_at": getattr(e, "retired_at", None)}
            for e in sorted(entries.values(), key=lambda e: e.task_id)
        ]
        body = {"as_of": now, "tasks": tasks,
                "active_cap_total": round(math.fsum(t["cap"] for t in tasks
                                                    if t["status"] == "active"), 9)}
        tasks_cache.update(at=now, body=body)
        return body

    async def refresh_tasks() -> None:
        try:
            await read_tasks()
        except Exception as exc:
            logger.warning("corpus tasks: registry unavailable: %r", exc)

    def refresh_tasks_behind() -> None:
        # One refresh at a time; the task is kept so it is not collected mid-read.
        running = tasks_cache.get("refresh")
        if running is None or running.done():
            tasks_cache["refresh"] = asyncio.get_running_loop().create_task(refresh_tasks())

    # The serving process warms it at start (`run_corpus_validator`).
    app.state.warm_corpus_tasks = refresh_tasks

    @app.get("/corpus/tasks")
    async def corpus_tasks() -> dict:
        # Public: every declared task's emission share. Served from the last registry
        # read at once; a stale one is refreshed behind, so no request waits on R2.
        if "body" in tasks_cache:
            if tasks_cache["at"] + TASKS_CACHE_SECONDS <= time.time():
                refresh_tasks_behind()
            return tasks_cache["body"]
        try:
            return await read_tasks()
        except Exception as exc:
            logger.warning("corpus tasks: registry unavailable: %r", exc)
            raise HTTPException(status_code=503, detail="task_registry_unavailable") from exc

    async def miner_status(job_id: str, hotkey: str) -> dict:
        # Public: this hotkey's own state and counts, from memory, cached.
        from reliquary.validator.corpus_miner_status import valid_hotkey

        if not valid_hotkey(hotkey):
            raise HTTPException(status_code=400, detail="invalid_hotkey")
        job_set = getattr(app.state, "corpus_jobs", None)
        try:
            status = await job_set.miner_status(job_id, hotkey) if job_set is not None else None
        except Exception as exc:
            logger.warning("corpus miner status on %s unavailable: %r", job_id, exc)
            raise HTTPException(status_code=503, detail="corpus_miner_status_unavailable") from exc
        if status is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return status

    @app.get("/corpus/jobs/{job_id}/miners/{hotkey}")
    async def corpus_miner_status(job_id: str, hotkey: str) -> dict:
        return await miner_status(job_id, hotkey)

    @app.get("/corpus/miners/{hotkey}")
    async def corpus_miner_status_legacy(hotkey: str) -> dict:
        # The legacy paths answer for the default job.
        if routes.default is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return await miner_status(routes.default, hotkey)

    @app.get("/corpus/jobs/{job_id}/eval-prompts")
    async def corpus_eval_prompts(job_id: str) -> Response:
        """An eval job's prompt lines, byte for byte as its manifest hashes them:
        miners cannot build a frozen set themselves."""
        from reliquary.eval.prompt_source import (
            is_eval_source, job_prompt_lines, parse_eval_source,
        )

        source = routes.prompt_sources.get(job_id)
        if job_id not in routes.routers or source is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        if not is_eval_source(source):
            raise HTTPException(status_code=404, detail="not_an_eval_job")
        body = await asyncio.to_thread(job_prompt_lines, parse_eval_source(source))
        return Response(content=body, media_type="application/x-ndjson")

    @app.get("/corpus/jobs/{job_id}/contract")
    async def corpus_job_contract(job_id: str) -> dict:
        if job_id not in routes.routers:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        if routes.contracts.get(job_id) is None:
            raise HTTPException(status_code=404, detail="corpus_contract_unknown")
        return routes.contracts[job_id]

    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def log_malformed(request, exc):
        # The miner gets the full 422; the log names the fields, never their content.
        hotkey = exc.body.get("miner_hotkey") if isinstance(exc.body, dict) else None
        fields = sorted({(".".join(str(p) for p in e.get("loc", ())), e.get("type")) for e in exc.errors()})
        logger.warning("corpus submission malformed from %s: %s", str(hotkey)[:48], fields[:10])
        return await request_validation_exception_handler(request, exc)

    return app


async def run_corpus_validator(*, wallet, netuid, signer_client, http_host, http_port,
                               set_weights: bool, entry=None, cap: float | None = None,
                               jobs=None, settle_every_seconds: float = 60.0,
                               registration_gate: bool = True, read_registry=None,
                               refresh_every_seconds: float | None = None,
                               remote_audit: bool = False,
                               recheck_fraction: float | None = None,
                               split=None, auditor_kwargs=None,
                               intake_only: bool = False) -> None:
    """Serve one corpus task (``entry``, ``cap``) or several (``jobs``, a list
    of ``(entry, cap)``) from one process and one loaded model.

    With ``read_registry`` (an async callable returning the registry's
    entries) the job set is hot: re-read every ``refresh_every_seconds``, new
    jobs on this model are wired and retired ones drained without a restart.
    Without it the jobs given here are the jobs served, as before.

    With ``remote_audit`` the ``/corpus/internal/audit/...`` routes are mounted
    and connected executors score the audits; with none connected, this
    process's GPU audits as before.

    With ``split`` (``corpus_split.FrontSplit``) this is the front of the split
    validator: no model is loaded (the supervisor checked the checkpoint and
    the GPU process scores), the jobs in ``split.links`` are judged in their
    own processes and every other job here, scoring on the GPU process. An
    episode job is served here exactly as by the single process (intake, audit
    v2 with its spans on the GPU process's wire, grader, grade routes, payment
    gate); one in ``split.links`` is refused (judge processes host no grader).

    With ``intake_only`` no model is loaded and nothing is audited or settled:
    the route takes submissions and episode jobs are graded (the end-to-end
    run, while the miner holds the card). Payment waits for the audits of a
    later start without it.
    """
    if intake_only and (remote_audit or split is not None):
        raise RuntimeError("intake-only serves no audit; unset RELIQUARY_CORPUS_REMOTE_AUDIT "
                           "and RELIQUARY_CORPUS_SPLIT")
    if split is not None and remote_audit:
        raise RuntimeError("remote audit executors are not served by the split validator; "
                           "unset RELIQUARY_CORPUS_REMOTE_AUDIT or RELIQUARY_CORPUS_SPLIT")
    if split is not None and set_weights:
        raise RuntimeError("the split validator does not set weights; run it with "
                           "--no-set-weights (the RL validator's setter pays every task)")
    import threading
    from pathlib import Path

    import torch
    import uvicorn
    from huggingface_hub import snapshot_download

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import (
        verify_corpus_signature,
        verify_corpus_skip_signature,
    )
    from reliquary.shared.modeling import load_text_only_model, load_tokenizer
    from reliquary.validator.corpus_judge_threads import (
        JudgeThreads, judge_record_store, run_in,
    )
    from reliquary.validator.corpus_hot_jobs import (
        JOB_REFRESH_SECONDS, CorpusJobSet, hot_job_refusal, job_drained, order_entry_screen,
    )
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job
    from reliquary.validator.corpus_settlement import R2Archives

    served = list(jobs) if jobs is not None else [(entry, cap)]
    from reliquary.eval.prompt_source import is_order_job_id

    for task_entry, _ in served:
        if is_order_job_id(task_entry.job_id):
            # Its own process serves it; two would pay its records twice.
            raise RuntimeError(f"task {task_entry.task_id!r} is an order job (eval or "
                               "generation): the order control (eval control) serves it, "
                               "never the corpus control")
    store = BucketJobStore()

    # A tokenizer isn't loaded yet, but the renderer only calls `encode` once
    # a submission arrives -- by then `tokenizer_box` is populated. Resolving
    # the prompt source here, before any download or model load, makes a bad
    # `prompt_source`/`renderer_id` declaration a refusal that costs seconds,
    # not a checkpoint download and a GPU load.
    tokenizer_box: dict = {}

    def encode(text: str) -> list[int]:
        encoded = tokenizer_box["tokenizer"].encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    from reliquary.validator.corpus_service import migrate_ledgers_at_startup

    several = len(served) > 1
    manifests = []
    for task_entry, task_cap in served:
        job, _ = await store.read_job(str(task_entry.job_id))
        if job is None:
            raise RuntimeError(
                f"task {task_entry.task_id!r} declares job {task_entry.job_id!r} but it has no manifest"
            )
        manifests.append((task_entry, task_cap, job))
    for _, _, job in manifests:
        # Before any download or ledger migration: a judge process cannot grade.
        refusal = split_episode_refusal(split, job)
        if refusal is not None:
            raise RuntimeError(refusal[1])

    if several:
        # Before any ledger is migrated: a start that refuses touches nothing.
        # The CLI refuses a contract-less entry among several ids before this.
        carried = all(getattr(e, "contract", None) is not None for e, _, _ in manifests)
        refusal = multi_job_refusal(
            [(e, job) for e, _, job in manifests],
            process_contract=ACTIVE_PROTOCOL_PROFILE.to_generation_contract() if carried else None,
        )
        if refusal:
            raise RuntimeError(refusal)

    def build_renderer(job, own_profile):
        return renderer_for_job(
            job, encode, tokenizer=lambda: tokenizer_box["tokenizer"], profile=own_profile
        )

    def prepared(task_entry, task_cap, job, own_profile, renderer, seen_index):
        from reliquary.validator.corpus_job_status import JobStats

        w = SimpleNamespace(
            entry=task_entry, cap=task_cap, job=job, renderer=renderer, seen_index=seen_index,
            prompt_job_for=(functools.partial(prompt_job_for_spec, profile=own_profile)
                            if own_profile is not None else None),
            stats=JobStats(),
        )

        def on_accepted(submission_id: str) -> None:
            w.stats.accepted()
            if not intake_only:
                w.auditor.enqueue(submission_id)
            if getattr(w, "grader", None) is not None:
                w.grader.enqueue(submission_id)

        w.on_accepted = on_accepted
        return w

    wiring = []
    # Episode jobs that failed to wire at startup, by task id: logged, left
    # unserved, and the others start (one job's environment never takes every
    # job's intake down). A hot job set retries them at its refresh.
    unserved: dict[str, str] = {}

    def not_served(task_entry, job, step: str, exc: BaseException) -> None:
        why = f"{step} failed: {type(exc).__name__}: {exc}"
        unserved[str(task_entry.task_id)] = why
        logger.error("corpus task %s: EPISODE JOB %s IS NOT SERVED: %s. The other jobs are "
                     "served; fix it and restart%s", task_entry.task_id, job.job_id, why,
                     " (the hot job set retries it at each refresh)" if read_registry else "",
                     exc_info=(type(exc), exc, exc.__traceback__))

    for task_entry, task_cap, job in manifests:
        if job.episode is not None:
            try:
                own_profile = (_entry_profile(task_entry) if several
                               and getattr(task_entry, "contract", None) is not None else None)
                renderer = build_renderer(job, own_profile)
                seen_index = await migrate_ledgers_at_startup(store, job)
            except Exception as exc:  # noqa: BLE001 - isolated: the others still start
                not_served(task_entry, job, "its renderer or ledger migration", exc)
                continue
            wiring.append(prepared(task_entry, task_cap, job, own_profile, renderer, seen_index))
            continue
        # Before anything serves: the route would otherwise seal a v1 seen set
        # inside its first submission's ledger turn. One ledger, one index, per job.
        seen_index = await migrate_ledgers_at_startup(store, job)
        # With several jobs the process runs their merged contract; each job's
        # renderer is still checked against its OWN task's contract.
        own_profile = (_entry_profile(task_entry)
                       if several and getattr(task_entry, "contract", None) is not None else None)
        try:
            renderer = build_renderer(job, own_profile)
        except ValueError as exc:
            # `CorpusPromptSourceError` (an unbuildable/mismatched prompt source)
            # is a `ValueError` subclass; an episode job's `renderer_id` naming no
            # known renderer raises the same plain `ValueError` from `renderer_for`
            # -- `jobs create` never checks that name either. One clause covers
            # both: both are the job declaring a rendering this binary cannot do.
            raise RuntimeError(
                f"job {job.job_id!r} declares renderer {job.renderer_id!r} for "
                f"prompt source {job.prompt_source!r}, which cannot be built: {exc}"
            ) from exc
        wiring.append(prepared(task_entry, task_cap, job, own_profile, renderer, seen_index))

    # Only the rehearsal turns the gate off: its local keys are not on the chain.
    registered = None
    if registration_gate:
        from reliquary.validator.corpus_registration import (
            RegisteredHotkeys, load_registered_hotkeys,
        )

        registered = RegisteredHotkeys(load=lambda: load_registered_hotkeys(netuid))
        if not await registered.refresh():
            logger.warning("subnet registrations unknown at start; miners get 503 until they load")

    # Every job names this one checkpoint (`multi_job_refusal`): load it once.
    first = wiring[0].job
    if split is None:
        directory = Path(snapshot_download(first.checkpoint_repo,
                                           revision=first.checkpoint_revision))
        fingerprint = checkpoint_fingerprint(directory)
        for w in wiring:
            refusal = startup_refusal(w.entry, w.job, ACTIVE_PROTOCOL_PROFILE, fingerprint)
            if refusal:
                raise RuntimeError(refusal if len(wiring) == 1
                                   else f"task {w.entry.task_id!r}: {refusal}")
    else:
        # Checked by the supervisor before any child started.
        directory, fingerprint = Path(split.directory), split.fingerprint

    tokenizer = load_tokenizer(str(directory))
    tokenizer_box["tokenizer"] = tokenizer
    checkpoint_dir = str(directory)
    scorer = None
    if split is None and not intake_only:
        model = load_text_only_model(
            str(directory), torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda").eval()
        proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
        vocab_size = model.get_input_embeddings().num_embeddings
    elif split is None:
        # Intake and grading only: the card is someone else's (the miner's, in
        # the end-to-end run); audits start when the process restarts without it.
        model, proof = None, toploc_proof(ACTIVE_PROTOCOL_PROFILE)
        vocab_size = _config_vocab_size(Path(checkpoint_dir))
    else:
        from reliquary.validator.corpus_gpu import read_info

        model, proof = None, split.proof
        vocab_size = (await read_info(split.run_dir))["vocab_size"]
    records = BucketRecordStore()
    # The auditors' and settlers' own connections (miners.json included): their
    # reads and writes in flight (up to 32 a job) never queue the route's behind
    # botocore's 10. The route's ban check keeps the route's client.
    # And their own threads: the route's default executor never waits for
    # a drand race, a forward or a record decode of theirs.
    judge_threads = JudgeThreads()
    judge_records = judge_record_store(judge_threads,
                                       max_pool_connections=JUDGE_POOL_CONNECTIONS)
    # One model, one forward pass at a time across every job; one job needs
    # none, unless more may join it.
    hot = read_registry is not None
    gpu_lock = (asyncio.Lock() if split is not None or len(wiring) > 1 or hot or remote_audit
                else None)
    if split is not None:
        from reliquary.validator.corpus_gpu import GPU_SOCKET, GpuScorer

        # The GPU process orders every forward; the lock keeps this process's
        # auditors from preparing their records all at once.
        scorer = GpuScorer(Path(split.run_dir) / GPU_SOCKET, chunk_tokens=proof.chunk_tokens,
                           topk=proof.topk, executor=judge_threads.codec)
    remote = directory = None
    if remote_audit:
        from reliquary.infrastructure import corpus_executor_store as executor_store
        from reliquary.validator.corpus_audit import rows_of_items, score_sequences
        from reliquary.validator.corpus_audit_remote import (
            RECHECK_FRACTION, ExecutorDirectory, RemoteAuditDispatcher,
        )
        from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

        async def local_scores(items):
            # The trusted verifier: this GPU, in turn with every job's auditor.
            async with gpu_lock:
                scores, _, _ = await run_in(judge_threads.gpu, lambda: score_sequences(
                    model, rows_of_items(items),
                    chunk_tokens=proof.chunk_tokens, topk=proof.topk,
                    batch_tokens=AUDIT_BATCH_TOKENS))
            return scores

        directory = ExecutorDirectory(model_id=first.checkpoint_repo,
                                      model_revision=first.checkpoint_revision)
        remote = RemoteAuditDispatcher(
            directory=directory, proof=proof, local_scores=local_scores,
            recheck_fraction=RECHECK_FRACTION if recheck_fraction is None else recheck_fraction,
            quarantine=lambda executor_id, reason: executor_store.set_executor_status(
                executor_id, "quarantined", reason=reason),
            record_heartbeat=lambda executor_id, at, detail: executor_store.record_heartbeat(
                executor_id, at=at, detail=detail),
        )
    # One grade dispatcher for every episode job: one env pin per validator.
    grade_dispatcher = grade_directory = None
    graders: dict[str, object] = {}
    pins = {(w.job.episode.env.package, w.job.episode.env.version)
            for w in wiring if w.job.episode is not None}
    if len(pins) > 1:
        raise RuntimeError(f"one validator grades one env pin, these jobs name {sorted(pins)}")
    if pins:
        from reliquary.infrastructure import corpus_executor_store as grade_store
        from reliquary.validator.corpus_audit_remote import ExecutorDirectory
        from reliquary.validator.corpus_grade_remote import RemoteGradeDispatcher

        from reliquary.validator.corpus_grade_remote import check_replay_lease

        (package, version), = pins
        # Ruling P26: never serve a job whose replays outlive the replay lease.
        for w in [w for w in wiring if w.job.episode is not None]:
            try:
                await asyncio.to_thread(check_replay_lease, w.job)
            except Exception as exc:  # noqa: BLE001 - isolated: the others still start
                not_served(w.entry, w.job, "its replay lease check", exc)
                wiring.remove(w)
        grade_directory = ExecutorDirectory(model_id=package, model_revision=version, scope="grade")
        grade_dispatcher = RemoteGradeDispatcher(
            directory=grade_directory, env_package=package, env_version=version,
            quarantine=lambda executor_id, reason: grade_store.set_executor_status(
                executor_id, "quarantined", reason=reason, scope="grade"),
            record_heartbeat=lambda executor_id, at, detail: grade_store.record_heartbeat(
                executor_id, at=at, detail=detail))
        # Quarantines survive a restart: the registry's are refused and held
        # before any grader (and so any settlement) is wired.
        try:
            await grade_directory.refresh()
        except Exception as exc:  # noqa: BLE001 - no grader without the quarantines
            for w in [w for w in wiring if w.job.episode is not None]:
                not_served(w.entry, w.job, "reading the grade executor registry", exc)
                wiring.remove(w)
            grade_dispatcher = grade_directory = None
        else:
            grade_dispatcher.load_quarantined()
    # Every grader's trajectory parses, on threads of their own (bounded):
    # never the default executor the submit routes' ledger turns run on.
    grade_parse_threads = None
    if grade_dispatcher is not None:
        from concurrent.futures import ThreadPoolExecutor

        from reliquary.validator.corpus_grading import GRADE_PARSE_THREADS

        grade_parse_threads = ThreadPoolExecutor(GRADE_PARSE_THREADS,
                                                 thread_name_prefix="corpus-grade-parse")
    job_set: CorpusJobSet | None = None
    archives = R2Archives(served=lambda: job_set.hot_task_ids() if job_set is not None else ())

    def episode_intake_for(w):
        # Loads the task set and the renderers: blocking, so a hot-added job
        # builds it off the event loop (`wire_hot`). The grader gets its own
        # renderer: its parses never queue on the intake's renderer lock.
        from reliquary.validator import agentic_intake

        intake = agentic_intake.build_episode_intake(
            w.job, checkpoint_dir=checkpoint_dir, tokenizer=tokenizer,
            vocab_size=vocab_size, chunk_tokens=proof.chunk_tokens)
        w.grade_renderer = agentic_intake.build_grade_renderer(w.job, checkpoint_dir=checkpoint_dir)
        return intake

    def audit_and_settle(w) -> None:
        refusal = split_episode_refusal(split, w.job)
        if refusal is not None:
            raise ValueError(refusal[1])                       # never paid ungraded
        if w.job.episode is not None and getattr(w, "episode_intake", None) is None:
            w.episode_intake = episode_intake_for(w)
        if w.job.episode is not None:
            wire_job_grader(w, records=records, judge_records=judge_records,
                            dispatcher=grade_dispatcher, parse_executor=grade_parse_threads,
                            beacon_executor=judge_threads.beacon)
            graders[str(w.job.job_id)] = w.grader
        link = split.links.get(str(w.job.job_id)) if split is not None else None
        if link is not None:
            wire_job_front_only(w, records=records, link=link)
            return
        wire_job_judge(w, records=records, judge_records=judge_records,
                       judge_threads=judge_threads, archives=archives, proof=proof,
                       model=model, tokenizer=tokenizer, gpu_lock=gpu_lock, remote=remote,
                       scorer=scorer, vocab_size=vocab_size if split is not None else None,
                       auditor_kwargs=auditor_kwargs)

    for w in list(wiring):
        if w.job.episode is None:
            audit_and_settle(w)
            continue
        try:
            audit_and_settle(w)
        except Exception as exc:  # noqa: BLE001 - isolated: the others still start
            not_served(w.entry, w.job, "its intake, grader or auditor wiring", exc)
            graders.pop(str(w.job.job_id), None)
            wiring.remove(w)
    if not wiring:
        raise RuntimeError("no corpus job left to serve: " + "; ".join(
            f"{task}: {why}" for task, why in sorted(unserved.items())))
    if any(w.job.episode is not None for w in wiring):
        # The drand chain once, off the loop: an episode job's auditor and
        # grader here would otherwise resolve it on the loop at first use.
        await warm_drand_chain(judge_threads.beacon)

    app = build_corpus_jobs_app(jobs=wiring, store=store, records=records, tokenizer=tokenizer,
                                verify_signature=verify_corpus_signature,
                                verify_skip_signature=verify_corpus_skip_signature,
                                proof_chunk_tokens=proof.chunk_tokens,
                                vocab_size=vocab_size,
                                registration=registered.reason if registered is not None else None,
                                contract=getattr(wiring[0].entry, "contract", None) if len(wiring) == 1 else None)

    async def wire_hot(task_entry, task_cap, job):
        refusal = split_episode_refusal(split, job)
        if refusal is not None:
            # Backstop: `admit` refuses it first, for good; a ValueError is permanent.
            raise ValueError(refusal[1])
        # The renderer first: a job refused for it leaves its ledger untouched.
        own_profile = _entry_profile(task_entry)
        renderer = build_renderer(job, own_profile)
        if job.episode is not None:
            # Before its intake (a task set download): a job this process
            # cannot grade is refused for nothing.
            grade_refusal(job, grade_dispatcher)
        seen_index = await migrate_ledgers_at_startup(store, job)
        w = prepared(task_entry, task_cap, job, own_profile, renderer, seen_index)
        if job.episode is not None:
            w.episode_intake = await asyncio.to_thread(episode_intake_for, w)
            await warm_drand_chain(judge_threads.beacon)
        audit_and_settle(w)
        return w

    async def read_job(job_id):
        job, _ = await store.read_job(job_id)
        return job

    process_contract = (ACTIVE_PROTOCOL_PROFILE.to_generation_contract() if hot else {})

    async def drained(w) -> bool:
        return await job_drained(auditor=w.auditor, records=records, job_id=w.job.job_id)

    job_set = CorpusJobSet(
        routes=app.state.corpus_routes, router_for=app.state.corpus_router_for,
        wire=wire_hot,
        jobs_of=lambda w: judge_jobs(w, intake_only=intake_only,
                                     settle_every_seconds=settle_every_seconds),
        read_entries=read_registry, read_job=read_job,
        screen=order_entry_screen,
        admit=lambda task_entry, job: split_episode_refusal(split, job) or hot_job_refusal(
            task_entry, job, process_profile=ACTIVE_PROTOCOL_PROFILE,
            process_contract=process_contract, fingerprint=fingerprint),
        drained=drained,
        refresh_every_seconds=(refresh_every_seconds if refresh_every_seconds is not None
                               else JOB_REFRESH_SECONDS),
    )
    app.state.corpus_jobs = job_set
    app.state.corpus_unserved = unserved
    for w in wiring:
        job_set.adopt(w)

    if set_weights:
        from reliquary.validator.weight_only import WeightOnlyValidator

        threading.Thread(
            target=lambda: asyncio.run(WeightOnlyValidator(wallet=wallet, netuid=netuid,
                                                           signer_client=signer_client).run()),
            name="weight-setter", daemon=True,
        ).start()

    background = [registered.refresh_forever()] if registered is not None else []
    if split is not None:
        # One sender per judge process, whatever number of jobs it judges.
        background += [link.run() for link in {id(k): k for k in split.links.values()}.values()]
    if remote is not None:
        from reliquary.validator.corpus_audit_remote import build_audit_executor_router

        app.include_router(build_audit_executor_router(remote, directory))
        app.state.corpus_audit_remote = remote
        background.append(remote.run())

    if grade_dispatcher is not None:
        from reliquary.validator.corpus_grade_remote import build_grade_executor_router

        app.include_router(build_grade_executor_router(grade_dispatcher, grade_directory))
        app.state.corpus_grade_remote = grade_dispatcher
        # What a quarantined executor decided alone is graded again, in every
        # episode job this process grades (hot-added ones included).
        grade_dispatcher.hold_on_quarantine(lambda executor_id: [
            grader.hold_executor(executor_id) for grader in list(graders.values())])
        grade_dispatcher.subscribe(
            lambda executor_id: regrade_everywhere(list(graders.values()), executor_id))
        background.append(grade_dispatcher.run())

    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    await asyncio.gather(server.serve(), job_set.run(), app.state.warm_corpus_tasks(), *background)


__all__ = [
    "JudgedElsewhere",
    "LazyRoundAt",
    "build_corpus_app",
    "build_corpus_audit_wiring",
    "build_corpus_jobs_app",
    "drand_beacon",
    "judge_jobs",
    "make_round_at",
    "multi_job_refusal",
    "run_corpus_validator",
    "settle_forever",
    "startup_refusal",
    "wire_job_front_only",
    "wire_job_grader",
    "wire_job_judge",
]
