"""A pinned operator generation control sharing an existing GPU scorer."""
from __future__ import annotations

import asyncio
from pathlib import Path


async def run_generation_control(*, task_ids: list[str], checkpoint_dir: str,
                                 gpu_run_dir: str, netuid: int = 81,
                                 http_host: str = "127.0.0.1", http_port: int = 8792) -> None:
    """Own only ``<admin prefix>gen-ops-`` jobs; never load a GPU model.

    The initial tasks and live GPU pin are checked before any ledger migration.
    Other compatible operator generation jobs join through the existing registry
    refresh. The external GPU process remains owned by its original supervisor.
    """
    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.infrastructure.corpus_job_store import read_job
    from reliquary.infrastructure.task_registry_store import read_registry
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
    from reliquary.validator.corpus_gpu import GPU_SOCKET
    from reliquary.validator.corpus_hot_jobs import generation_entry_screen, hot_job_refusal
    from reliquary.validator.corpus_split import FrontSplit
    from reliquary.validator.corpus_validator import run_corpus_validator

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("initial task ids must be distinct")
    directory = Path(checkpoint_dir).resolve(strict=True)
    run_dir = Path(gpu_run_dir).resolve(strict=True)
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    if proof is None or proof.mode != "enforce":
        raise ValueError("the pinned generation control requires enforced TOPLOC")
    entries, _ = await read_registry()
    selected = []
    for task_id in task_ids:
        entry = entries.get(task_id)
        if (entry is None or entry.status != "active"
                or entry.mechanism != MECHANISM_CORPUS_GENERATION
                or generation_entry_screen(entry) is not None):
            raise ValueError("an initial task is not an active operator generation task")
        selected.append(entry)
    fingerprint = await asyncio.to_thread(checkpoint_fingerprint, directory)
    import httpx

    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(run_dir / GPU_SOCKET)),
                                 base_url="http://generation-scorer", timeout=10) as client:
        response = await client.get("/info")
        response.raise_for_status()
        info = response.json()
    if info is None or (info.get("model_id"), info.get("model_revision")) != (
            ACTIVE_PROTOCOL_PROFILE.model_id, ACTIVE_PROTOCOL_PROFILE.model_revision):
        raise ValueError("the live shared GPU scorer does not match the pinned generation model")
    contract = ACTIVE_PROTOCOL_PROFILE.to_generation_contract()
    for entry in selected:
        job, _ = await read_job(str(entry.job_id))
        if job is None or job.job_id != entry.job_id or job.submit != "scoped":
            raise ValueError("an initial task has no matching scoped job manifest")
        refusal = hot_job_refusal(entry, job, process_profile=ACTIVE_PROTOCOL_PROFILE,
                                  process_contract=contract, fingerprint=fingerprint,
                                  generation_only=True)
        if refusal is not None:
            raise ValueError("an initial generation task cannot use the shared pinned scorer: " + refusal[1])

    async def registry_entries():
        found, _ = await read_registry()
        return found

    await run_corpus_validator(
        jobs=[(entry, float(entry.params["cap"])) for entry in selected], wallet=None,
        netuid=netuid, signer_client=None, http_host=http_host, http_port=http_port,
        set_weights=False, read_registry=registry_entries, remote_audit=True,
        generation_only=True,
        split=FrontSplit(directory=str(directory), fingerprint=fingerprint, proof=proof,
                         run_dir=str(run_dir), links={}),
    )
