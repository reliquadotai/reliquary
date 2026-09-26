"""The validator of a corpus task: one process, one card, no RL machinery.

It serves the submission route, audits every accepted submission on its own
GPU, settles verified tokens into this task's archives, and sets weights only
when told to (the RL validator's setter already pays every task).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time

from fastapi import FastAPI

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
                     vocab_size=None, is_banned=None, registration=None) -> FastAPI:
    from reliquary.validator.corpus_service import build_corpus_router, prompt_job_for_spec

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id=str(entry.job_id), store=store, tokenizer=tokenizer, renderer=renderer,
        verify_signature=verify_signature, prompt_job_for=prompt_job_for or prompt_job_for_spec,
        records=records, on_accepted=auditor.enqueue, proof_chunk_tokens=proof_chunk_tokens,
        vocab_size=vocab_size, is_banned=is_banned, registration=registration,
    ))
    return app


async def run_corpus_validator(*, entry, wallet, netuid, signer_client, http_host, http_port,
                               cap: float, set_weights: bool,
                               settle_every_seconds: float = 60.0,
                               registration_gate: bool = True) -> None:
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
    from reliquary.protocol.signatures import verify_corpus_signature
    from reliquary.shared.modeling import load_text_only_model, load_tokenizer
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from reliquary.validator.corpus_service import renderer_for_job
    from reliquary.validator.corpus_settlement import CorpusSettler, R2Archives

    store = BucketJobStore()
    job, _ = await store.read_job(str(entry.job_id))
    if job is None:
        raise RuntimeError(f"task {entry.task_id!r} declares job {entry.job_id!r} but it has no manifest")

    # A tokenizer isn't loaded yet, but the renderer only calls `encode` once
    # a submission arrives -- by then `tokenizer_box` is populated. Resolving
    # the prompt source here, before any download or model load, makes a bad
    # `prompt_source`/`renderer_id` declaration a refusal that costs seconds,
    # not a checkpoint download and a GPU load.
    tokenizer_box: dict = {}

    def encode(text: str) -> list[int]:
        encoded = tokenizer_box["tokenizer"].encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    try:
        renderer = renderer_for_job(
            job, encode, tokenizer=lambda: tokenizer_box["tokenizer"]
        )
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

    # Only the rehearsal turns the gate off: its local keys are not on the chain.
    registered = None
    if registration_gate:
        from reliquary.validator.corpus_registration import (
            RegisteredHotkeys, load_registered_hotkeys,
        )

        registered = RegisteredHotkeys(load=lambda: load_registered_hotkeys(netuid))
        if not await registered.refresh():
            logger.warning("subnet registrations unknown at start; miners get 503 until they load")

    directory = Path(snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision))
    refusal = startup_refusal(entry, job, ACTIVE_PROTOCOL_PROFILE, checkpoint_fingerprint(directory))
    if refusal:
        raise RuntimeError(refusal)

    tokenizer = load_tokenizer(str(directory))
    tokenizer_box["tokenizer"] = tokenizer
    model = load_text_only_model(
        str(directory), torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda").eval()
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    records = BucketRecordStore()
    params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
        entry=entry, job=job, records=records
    )
    auditor = CorpusAuditor(job_id=job.job_id, records=records, model=model,
                            tokenizer=tokenizer, proof=proof, params=params,
                            miner_states=miner_states, beacon=beacon, round_at=round_at)

    app = build_corpus_app(entry=entry, job=job, store=store, records=records, tokenizer=tokenizer,
                           renderer=renderer,
                           verify_signature=verify_corpus_signature, auditor=auditor,
                           proof_chunk_tokens=proof.chunk_tokens,
                           vocab_size=model.get_input_embeddings().num_embeddings,
                           is_banned=is_banned,
                           registration=registered.reason if registered is not None else None)
    # `entry.cap` does not exist on `TaskEntry` (the cap lives in
    # `params["cap"]`); the CLI passes the value `TaskConfig` already resolved.
    settler = CorpusSettler(task_id=entry.task_id, job_id=job.job_id, cap=cap,
                            records=records, archives=R2Archives())

    async def settle_forever() -> None:
        while True:
            try:
                window = await settler.settle_once()
                if window is not None:
                    logger.info("corpus task %s settled window %d", entry.task_id, window)
            except Exception:
                logger.exception("corpus settlement failed; retrying next period")
            await asyncio.sleep(settle_every_seconds)

    if set_weights:
        from reliquary.validator.weight_only import WeightOnlyValidator

        threading.Thread(
            target=lambda: asyncio.run(WeightOnlyValidator(wallet=wallet, netuid=netuid,
                                                           signer_client=signer_client).run()),
            name="weight-setter", daemon=True,
        ).start()

    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    background = [registered.refresh_forever()] if registered is not None else []
    await asyncio.gather(server.serve(), auditor.run(), settle_forever(), *background)


__all__ = [
    "LazyRoundAt",
    "build_corpus_app",
    "build_corpus_audit_wiring",
    "drand_beacon",
    "make_round_at",
    "run_corpus_validator",
    "startup_refusal",
]
