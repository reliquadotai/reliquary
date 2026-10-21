"""Grading an evaluation our own corpus validator served.

The validator audited every submission with the TOPLOC proof of the job's
contract on the model it loaded; this grades the passing completions against
the set on CPU, with the same grader the order evaluations use
(`grade_evaluation`), and writes ``report.json``, ``manifest.json`` and
``graded.parquet`` to a local directory. No admin service and no qualification:
the trust is the validator's, which we run.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class JobNotGradable(RuntimeError):
    """The job is not an eval job, is not drained, or misses samples."""


# HF backups of the async trainer: <namespace>/<prefix>-step-<n:08d> (trainer/hf_backup.py).
_BACKUP_STEP = re.compile(r"-step-(\d+)$")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def rl_provenance(model: str | None, *, run_id: str | None = None, step: int | None = None,
                  checkpoint_version: int | None = None) -> dict:
    """The RL identity a report carries: ``rl_run_id`` (the trainer's wandb run id), ``rl_step`` and
    ``rl_checkpoint_version``. Explicit values win; the step (and the version, one step per checkpoint) is read
    from an HF backup's repo name. Empty when nothing is known: the report is unchanged."""
    if step is None and model:
        found = _BACKUP_STEP.search(model)
        if found:
            step = int(found.group(1))
            checkpoint_version = step if checkpoint_version is None else checkpoint_version
    if run_id is not None and not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"rl run id {run_id!r}: letters, digits, '.', '_' or '-', at most 128")
    for name, value in (("rl step", step), ("rl checkpoint version", checkpoint_version)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    named = {"rl_run_id": run_id, "rl_step": step, "rl_checkpoint_version": checkpoint_version}
    return {k: v for k, v in named.items() if v is not None}


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
                           rl: dict | None = None,
                           clock: Callable[[], float] = time.time) -> dict:
    """Grade a drained eval job our validator served; the manifest, its files in ``out``.
    ``rl``: ``rl_provenance`` keywords (run_id, step, checkpoint_version)."""
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval import grading
    from reliquary.eval.prompt_source import (
        REFERENCE_INTERACTIONS, is_eval_source, is_order_job_id, load_eval_rows,
        parse_eval_source, set_interaction,
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
    # The set's rows (the ones the job ran) say how it is graded; its card in
    # the bucket is read, and checked against them, by the grading itself.
    rows = await asyncio.to_thread(load_eval_rows, source)
    try:
        interaction = set_interaction(rows)
    except ValueError as exc:
        raise JobNotGradable(f"job {job_id}: eval set {source.set_id}: {exc}") from exc
    if (interaction in REFERENCE_INTERACTIONS) != (job.episode is not None):
        raise JobNotGradable(f"job {job_id}: eval set {source.set_id} holds {interaction} "
                             f"rows, but the job is {'an' if job.episode else 'no'} episode job")
    if interaction in REFERENCE_INTERACTIONS:
        # Graded already by the job's grader: its grade documents, never the records.
        job_rows = await grading.EpisodeJobRows.collect(job, records,
                                                        allow_incomplete=allow_incomplete)
        collected = {"samples_by_prompt": job_rows.samples_by_prompt}
    else:
        collected = await grading.collect_job_records(job, records)
        job_rows = grading.JobRows(job=job, collected=collected)
    snapshot, _ = await read_ledgers(job_id)
    slots = rebuild_ledgers(job, snapshot).slots
    complete = grading.job_complete(job, collected, samples, slots)
    exhausted = grading.exhausted_prompts(job, collected, samples, slots)
    if not complete and not allow_incomplete:
        raise JobNotGradable(f"job {job_id} misses samples on some problems: pass "
                             "--allow-incomplete to count them as failures")
    try:
        rl_identity = rl_provenance(job.checkpoint_repo, **(rl or {}))
    except ValueError as exc:
        raise JobNotGradable(f"job {job_id}: {exc}") from exc
    task_id, thresholds = await _thresholds(job_id, entries)
    # An episode's turns still carry TOPLOC proofs (the thresholds); its score
    # is its grader's, from the grade documents.
    scheme, origin = ((grading.EPISODE_SCHEMES[interaction], "grade documents")
                      if interaction in REFERENCE_INTERACTIONS
                      else ("toploc-v1", "task contract"))
    verification = {"scheme": scheme, "source": origin,
                    "validator": "our corpus validator, on the job's model",
                    "task_id": task_id, "thresholds": thresholds, "sampling_verified": False}
    provenance = {
        "model": job.checkpoint_repo, "revision": job.checkpoint_revision,
        "model_sha": job.checkpoint_revision, "checkpoint_sha256": job.checkpoint_sha256,
        "sampling": {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                     "top_k": job.sampling.top_k},
        "thinking": CHAT_TEMPLATE_RENDERERS.get(job.renderer_id, False),
        "max_new_tokens": job.sampling.max_new_tokens, "seed": job.seed,
        "eos_token_id": job.eos_token_id,
        "verification": verification,
        "job_complete": complete, "prompts_exhausted": len(exhausted),
        "exhausted_problem_ids": [rows[i]["problem_id"] for i in exhausted],
        **({"allow_incomplete": True} if allow_incomplete else {}),
        **rl_identity,
    }
    extra = {} if open_taskset is None else {"open_taskset": open_taskset}
    with tempfile.TemporaryDirectory() as scratch:
        sink = LocalDirectorySink(Path(scratch))
        manifest = await grading.grade_evaluation(
            eval_id=job_id, set_ids=[source.set_id], completion_keys=[],
            problems_per_set={source.set_id: source.count},
            samples_per_set={source.set_id: samples}, provenance=provenance,
            job_rows=job_rows, platform=sink,
            subnet=subnet, require_sandbox=require_sandbox, work_dir=work_dir, clock=clock,
            **extra)
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        graded = Path(scratch) / grading.evaluation_prefix(job_id)
        for name in ("report.json", "manifest.json", "graded.parquet"):
            shutil.copyfile(graded / name, directory / name)
    return manifest


GRADING_FILES = ("report.json", "manifest.json", "graded.parquet")


async def upload_grading(job_id: str, directory: str | Path, *, subnet: Any = None) -> list[str]:
    """Put a served job's grading files under ``evaluations/<job>/`` in the subnet bucket, create-only (the same
    bytes again are a no-op, other bytes are refused: a grading is not rewritten in place)."""
    from reliquary.eval.grading import evaluation_prefix

    if subnet is None:
        from reliquary.eval.storage import SubnetEvalStore

        subnet = SubnetEvalStore()
    keys = []
    for name in GRADING_FILES:
        key = f"{evaluation_prefix(job_id)}/{name}"
        await subnet.put_bytes(key, (Path(directory) / name).read_bytes())
        keys.append(key)
    return keys


__all__ = ["GRADING_FILES", "JobNotGradable", "grade_served_job", "rl_provenance", "upload_grading"]
