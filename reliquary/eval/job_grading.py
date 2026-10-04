"""Grading an evaluation our own corpus validator served.

The validator audited every submission with the TOPLOC proof of the job's
contract on the model it loaded; this grades the passing completions against
the set on CPU, with the same grader the order evaluations use
(`grade_evaluation`), and writes ``report.json``, ``manifest.json`` and
``graded.parquet`` to a local directory. No admin service and no qualification:
the trust is the validator's, which we run.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class JobNotGradable(RuntimeError):
    """The job is not an eval job, is not drained, or misses samples."""


async def _thresholds(job_id: str, entries: Callable | None) -> tuple[str | None, dict | None]:
    if entries is None:
        from reliquary.infrastructure.task_registry_store import read_registry

        async def entries():
            found, _ = await read_registry()
            return found.values()
    for entry in await entries():
        if getattr(entry, "job_id", None) != job_id:
            continue
        proofs = (getattr(entry, "contract", None) or {}).get("proofs") or ()
        toploc = [p for p in proofs if p.get("scheme") == "toploc-v1"]
        if toploc:
            return entry.task_id, {k: toploc[0].get(k) for k in (
                "exp_mismatch_threshold", "mant_mean_threshold", "mant_median_threshold")}
        return entry.task_id, None
    return None, None


async def grade_served_job(job_id: str, *, out: str | Path, allow_incomplete: bool = False,
                           records: Any = None, subnet: Any = None,
                           read_job: Callable | None = None,
                           read_ledgers: Callable | None = None,
                           entries: Callable | None = None,
                           open_taskset: Callable | None = None,
                           require_sandbox=None, work_dir: str | Path | None = None,
                           clock: Callable[[], float] = time.time) -> dict:
    """Grade a drained eval job our validator served; the manifest, its files in ``out``."""
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval import grading
    from reliquary.eval.prompt_source import (
        is_eval_source, is_order_job_id, load_eval_rows, parse_eval_source,
    )
    from reliquary.validator.corpus_job_status import stored_job_counts
    from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS, rebuild_ledgers

    if read_job is None or read_ledgers is None:
        from reliquary.infrastructure import corpus_job_store

        read_job = read_job or corpus_job_store.read_job
        read_ledgers = read_ledgers or corpus_job_store.read_ledgers
    if records is None:
        from reliquary.infrastructure.corpus_record_store import BucketRecordStore

        records = BucketRecordStore()
    if subnet is None:
        from reliquary.eval.storage import SubnetEvalStore

        subnet = SubnetEvalStore()
    if is_order_job_id(job_id):
        raise JobNotGradable(f"{job_id} is an order job: the admin service grades it")
    job, _ = await read_job(job_id)
    if job is None:
        raise JobNotGradable(f"job {job_id} has no manifest")
    if not is_eval_source(job.prompt_source):
        raise JobNotGradable(f"job {job_id} reads {job.prompt_source!r}, not an eval set")
    source = parse_eval_source(job.prompt_source)
    samples = job.slots_per_prompt * job.sampling.n
    if not (await stored_job_counts(records, job_id))["drained"]:
        raise JobNotGradable(f"job {job_id} is not drained: every submission must be audited "
                             "and settled first (reliquary jobs status)")
    collected = await grading.collect_job_records(job, records)
    snapshot, _ = await read_ledgers(job_id)
    slots = rebuild_ledgers(job, snapshot).slots
    complete = grading.job_complete(job, collected, samples, slots)
    exhausted = grading.exhausted_prompts(job, collected, samples, slots)
    if not complete and not allow_incomplete:
        raise JobNotGradable(f"job {job_id} misses samples on some problems: pass "
                             "--allow-incomplete to count them as failures")
    task_id, thresholds = await _thresholds(job_id, entries)
    rows = load_eval_rows(source)
    provenance = {
        "model": job.checkpoint_repo, "revision": job.checkpoint_revision,
        "model_sha": job.checkpoint_revision, "checkpoint_sha256": job.checkpoint_sha256,
        "sampling": {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                     "top_k": job.sampling.top_k},
        "thinking": CHAT_TEMPLATE_RENDERERS.get(job.renderer_id, False),
        "max_new_tokens": job.sampling.max_new_tokens, "seed": job.seed,
        "eos_token_id": job.eos_token_id,
        "verification": {"scheme": "toploc-v1", "source": "task contract",
                         "validator": "our corpus validator, on the job's model",
                         "task_id": task_id, "thresholds": thresholds,
                         "sampling_verified": False},
        "job_complete": complete, "prompts_exhausted": len(exhausted),
        "exhausted_problem_ids": [rows[i]["problem_id"] for i in exhausted],
        **({"allow_incomplete": True} if allow_incomplete else {}),
    }
    extra = {} if open_taskset is None else {"open_taskset": open_taskset}
    with tempfile.TemporaryDirectory() as scratch:
        sink = LocalDirectorySink(Path(scratch))
        manifest = await grading.grade_evaluation(
            eval_id=job_id, set_ids=[source.set_id], completion_keys=[],
            problems_per_set={source.set_id: source.count},
            samples_per_set={source.set_id: samples}, provenance=provenance,
            job_rows=grading.JobRows(job=job, collected=collected), platform=sink,
            subnet=subnet, require_sandbox=require_sandbox, work_dir=work_dir, clock=clock,
            **extra)
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        graded = Path(scratch) / grading.evaluation_prefix(job_id)
        for name in ("report.json", "manifest.json", "graded.parquet"):
            shutil.copyfile(graded / name, directory / name)
    return manifest


__all__ = ["JobNotGradable", "grade_served_job"]
