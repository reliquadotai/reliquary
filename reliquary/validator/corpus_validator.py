"""The validator of a corpus task: one process, one card, no RL machinery.

It serves the submission route, audits every accepted submission on its own
GPU, settles verified tokens into this task's archives, and sets weights only
when told to (the RL validator's setter already pays every task).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC

logger = logging.getLogger(__name__)


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


def build_corpus_app(*, entry, job, store, records, tokenizer, renderer, verify_signature,
                     auditor, proof_chunk_tokens, prompt_job_for=None) -> FastAPI:
    from reliquary.validator.corpus_service import build_corpus_router, prompt_job_for_spec

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id=str(entry.job_id), store=store, tokenizer=tokenizer, renderer=renderer,
        verify_signature=verify_signature, prompt_job_for=prompt_job_for or prompt_job_for_spec,
        records=records, on_accepted=auditor.enqueue, proof_chunk_tokens=proof_chunk_tokens,
    ))
    return app


async def run_corpus_validator(*, entry, wallet, netuid, signer_client, http_host, http_port,
                               cap: float, set_weights: bool,
                               settle_every_seconds: float = 60.0) -> None:
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
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
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
        renderer = renderer_for_job(job, encode)
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

    directory = Path(snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision))
    refusal = startup_refusal(entry, job, ACTIVE_PROTOCOL_PROFILE, checkpoint_fingerprint(directory))
    if refusal:
        raise RuntimeError(refusal)

    tokenizer = load_tokenizer(str(directory))
    tokenizer_box["tokenizer"] = tokenizer
    model = load_text_generation_model(
        str(directory), torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda").eval()
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    records = BucketRecordStore()
    auditor = CorpusAuditor(job_id=job.job_id, records=records, model=model,
                            tokenizer=tokenizer, proof=proof)

    app = build_corpus_app(entry=entry, job=job, store=store, records=records, tokenizer=tokenizer,
                           renderer=renderer,
                           verify_signature=verify_corpus_signature, auditor=auditor,
                           proof_chunk_tokens=proof.chunk_tokens)
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
    await asyncio.gather(server.serve(), auditor.run(), settle_forever())


__all__ = ["build_corpus_app", "run_corpus_validator", "startup_refusal"]
