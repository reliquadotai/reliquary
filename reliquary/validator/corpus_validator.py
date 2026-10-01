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

from fastapi import FastAPI, HTTPException

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC

logger = logging.getLogger(__name__)

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
        served = process_contract.get("environments") or {}
        for entry, job in pairs:
            own = ((getattr(entry, "contract", None) or {}).get("environments") or {}).get(
                job.prompt_source
            )
            if own is None or served.get(job.prompt_source) != own:
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


def drand_beacon(round_number: int) -> str | None:
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
        )

    # Miners have no registry access: each job's own task contract. With one
    # job at boot, the contract this process runs.
    routes = CorpusJobRoutes()
    for served in jobs:
        routes.add(str(served.entry.job_id), router_for(served),
                   contract=contract if len(jobs) == 1 else getattr(served.entry, "contract", None))
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
                               recheck_fraction: float | None = None) -> None:
    """Serve one corpus task (``entry``, ``cap``) or several (``jobs``, a list
    of ``(entry, cap)``) from one process and one loaded model.

    With ``read_registry`` (an async callable returning the registry's
    entries) the job set is hot: re-read every ``refresh_every_seconds``, new
    jobs on this model are wired and retired ones drained without a restart.
    Without it the jobs given here are the jobs served, as before.

    With ``remote_audit`` the ``/corpus/internal/audit/...`` routes are mounted
    and connected executors score the audits; with none connected, this
    process's GPU audits as before.
    """
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
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from reliquary.validator.corpus_hot_jobs import (
        JOB_REFRESH_SECONDS, CorpusJobSet, eval_entry_screen, hot_job_refusal, job_drained,
    )
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job
    from reliquary.validator.corpus_settlement import CorpusSettler, R2Archives

    served = list(jobs) if jobs is not None else [(entry, cap)]
    from reliquary.eval.prompt_source import is_eval_job_id

    for task_entry, _ in served:
        if is_eval_job_id(task_entry.job_id):
            # Its own process serves it; two would pay its records twice.
            raise RuntimeError(f"task {task_entry.task_id!r} is an evaluation job: the eval "
                               "control serves it, never the corpus control")
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
            w.auditor.enqueue(submission_id)

        w.on_accepted = on_accepted
        return w

    wiring = []
    for task_entry, task_cap, job in manifests:
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
    directory = Path(snapshot_download(first.checkpoint_repo, revision=first.checkpoint_revision))
    fingerprint = checkpoint_fingerprint(directory)
    for w in wiring:
        refusal = startup_refusal(w.entry, w.job, ACTIVE_PROTOCOL_PROFILE, fingerprint)
        if refusal:
            raise RuntimeError(refusal if len(wiring) == 1 else f"task {w.entry.task_id!r}: {refusal}")

    tokenizer = load_tokenizer(str(directory))
    tokenizer_box["tokenizer"] = tokenizer
    model = load_text_only_model(
        str(directory), torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda").eval()
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    records = BucketRecordStore()
    # One model, one forward pass at a time across every job; one job needs
    # none, unless more may join it.
    hot = read_registry is not None
    gpu_lock = asyncio.Lock() if len(wiring) > 1 or hot or remote_audit else None
    remote = directory = None
    if remote_audit:
        from reliquary.infrastructure import corpus_executor_store as executor_store
        from reliquary.validator.corpus_audit import score_sequences
        from reliquary.validator.corpus_audit_remote import (
            RECHECK_FRACTION, ExecutorDirectory, RemoteAuditDispatcher,
        )
        from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

        async def local_scores(items):
            # The trusted verifier: this GPU, in turn with every job's auditor.
            async with gpu_lock:
                scores, _, _ = await asyncio.to_thread(
                    score_sequences, model,
                    [(i["tokens"], i["prompt_len"], i["proofs"]) for i in items],
                    chunk_tokens=proof.chunk_tokens, topk=proof.topk,
                    batch_tokens=AUDIT_BATCH_TOKENS)
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
    job_set: CorpusJobSet | None = None
    archives = R2Archives(served=lambda: job_set.hot_task_ids() if job_set is not None else ())

    def audit_and_settle(w) -> None:
        params, miner_states, w.is_banned, beacon, round_at = build_corpus_audit_wiring(
            entry=w.entry, job=w.job, records=records
        )
        w.auditor = CorpusAuditor(job_id=w.job.job_id, records=records, model=model,
                                  tokenizer=tokenizer, proof=proof, params=params,
                                  miner_states=miner_states, beacon=beacon, round_at=round_at,
                                  gpu_lock=gpu_lock, on_verdict=w.stats.observe,
                                  remote=remote)
        # `entry.cap` does not exist on `TaskEntry` (the cap lives in
        # `params["cap"]`); the CLI passes the value `TaskConfig` already resolved.
        w.settler = CorpusSettler(task_id=w.entry.task_id, job_id=w.job.job_id, cap=w.cap,
                                  records=records, archives=archives,
                                  on_settled=w.stats.settled)

    for w in wiring:
        audit_and_settle(w)

    app = build_corpus_jobs_app(jobs=wiring, store=store, records=records, tokenizer=tokenizer,
                                verify_signature=verify_corpus_signature,
                                verify_skip_signature=verify_corpus_skip_signature,
                                proof_chunk_tokens=proof.chunk_tokens,
                                vocab_size=model.get_input_embeddings().num_embeddings,
                                registration=registered.reason if registered is not None else None,
                                contract=getattr(wiring[0].entry, "contract", None) if len(wiring) == 1 else None)

    async def settle_forever(task_id: str, settler) -> None:
        while True:
            try:
                window = await settler.settle_once()
                if window is not None:
                    logger.info("corpus task %s settled window %d", task_id, window)
            except Exception:
                logger.exception("corpus settlement failed; retrying next period")
            await asyncio.sleep(settle_every_seconds)

    async def wire_hot(task_entry, task_cap, job):
        # The renderer first: a job refused for it leaves its ledger untouched.
        own_profile = _entry_profile(task_entry)
        renderer = build_renderer(job, own_profile)
        seen_index = await migrate_ledgers_at_startup(store, job)
        w = prepared(task_entry, task_cap, job, own_profile, renderer, seen_index)
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
        jobs_of=lambda w: [w.auditor.run(), settle_forever(w.entry.task_id, w.settler)],
        read_entries=read_registry, read_job=read_job,
        screen=eval_entry_screen,
        admit=lambda task_entry, job: hot_job_refusal(
            task_entry, job, process_profile=ACTIVE_PROTOCOL_PROFILE,
            process_contract=process_contract, fingerprint=fingerprint),
        drained=drained,
        refresh_every_seconds=(refresh_every_seconds if refresh_every_seconds is not None
                               else JOB_REFRESH_SECONDS),
    )
    app.state.corpus_jobs = job_set
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
    if remote is not None:
        from reliquary.validator.corpus_audit_remote import build_audit_executor_router

        app.include_router(build_audit_executor_router(remote, directory))
        app.state.corpus_audit_remote = remote
        background.append(remote.run())

    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    await asyncio.gather(server.serve(), job_set.run(), *background)


__all__ = [
    "LazyRoundAt",
    "build_corpus_app",
    "build_corpus_audit_wiring",
    "build_corpus_jobs_app",
    "drand_beacon",
    "make_round_at",
    "multi_job_refusal",
    "run_corpus_validator",
    "startup_refusal",
]
