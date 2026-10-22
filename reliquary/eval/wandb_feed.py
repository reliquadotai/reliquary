"""An RL run's held-out feed (``run_feed``) in W&B, in a dedicated evaluation run beside the trainer's.

Two writes to that run (``<group>-evals`` by default, never the trainer's own: it has one writer):

* ``publish_summary``: the whole ``reliquary.run-benchmarks.v1`` envelope as the summary key
  ``reliquary_run_benchmarks``, which the platform's run monitor reads (``WANDB_RL_BENCHMARK_RUN_ID``,
  reliquary-platform ``worker/run-benchmarks.ts`` ``readWandbRunBenchmarks``). Idempotent: an envelope equal to the
  stored one but for ``observed_at`` is not written again. W&B stores a nested summary object flattened, as
  ``reliquary_run_benchmarks.schema`` ... ``reliquary_run_benchmarks.benchmarks`` (lists stay whole): a reader rebuilds
  the envelope from those keys (``stored_envelope``).
* ``log_feed``: each new point as a history row, ``eval/<short>/pass@1`` and its ``delta`` to the baseline, plotted
  against ``rl_step``. Logged points are remembered in a small JSON file, so each one is logged once.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAINER_SUFFIX = "-trainer"
SUMMARY_KEY = "reliquary_run_benchmarks"
DEFAULT_PROJECT = "reliquary-validator"
JOB_TYPE = "evaluation"


def eval_run_ids(trainer_run_id: str) -> tuple[str, str]:
    """``(group, run id)`` of the eval run beside ``trainer_run_id`` (``{group}-trainer``, ``telemetry.init``)."""
    group = trainer_run_id[:-len(TRAINER_SUFFIX)] if trainer_run_id.endswith(TRAINER_SUFFIX) else trainer_run_id
    return group, f"{group}-evals"


def target(trainer_run_id: str, *, run_id: str | None = None, project: str | None = None,
           entity: str | None = None) -> dict:
    """Where the evaluation run is: flags, then ``WANDB_PROJECT`` / ``WANDB_ENTITY``, then defaults."""
    from reliquary.eval.run_feed import RUN_ID

    group, default_id = eval_run_ids(trainer_run_id)
    run_id = run_id or default_id
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"wandb run id {run_id!r}: letters, digits, '.', '_' or '-'")
    if run_id == trainer_run_id:
        raise ValueError("the evaluation run must not be the trainer's run: the trainer is its only writer")
    return {"run_id": run_id, "group": group,
            "project": project or os.getenv("WANDB_PROJECT") or DEFAULT_PROJECT,
            "entity": entity or os.getenv("WANDB_ENTITY") or None}


def _wandb_init(*, run_id: str, group: str, project: str | None = None, entity: str | None = None,
                config: dict | None = None):
    import wandb

    run = wandb.init(project=project or os.getenv("WANDB_PROJECT") or DEFAULT_PROJECT,
                     entity=entity or os.getenv("WANDB_ENTITY") or None, id=run_id, group=group,
                     job_type=JOB_TYPE, resume="allow", config=config or {})
    run.define_metric("rl_step")
    run.define_metric("eval/*", step_metric="rl_step")
    return run


def stored_envelope(summary: Mapping[str, Any]) -> Any:
    """The envelope in a run's raw summary (``summaryMetrics``): the object itself, or rebuilt from its flattened
    ``reliquary_run_benchmarks.<path>`` keys as the W&B server stores it. None when absent."""
    if isinstance(summary.get(SUMMARY_KEY), Mapping):
        return summary[SUMMARY_KEY]
    prefix, found = SUMMARY_KEY + ".", {}
    for key, value in summary.items():
        if not key.startswith(prefix):
            continue
        *parents, leaf = key[len(prefix):].split(".")
        node = found
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return found or None


def _stored_summary(*, run_id: str, project: str, entity: str | None) -> Any:
    """The envelope stored in the run's summary, None when the run or the key does not exist."""
    import wandb

    api = wandb.Api()
    try:
        run = api.run(f"{entity or api.default_entity}/{project}/{run_id}")
    except Exception:           # wandb.errors.CommError: no such run yet
        return None
    return stored_envelope(json.loads(json.dumps(dict(run.summary_metrics or {}))))


def _same(a: Any, b: Any) -> bool:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    return {**a, "observed_at": None} == {**b, "observed_at": None}


def publish_summary(feed: Mapping[str, Any], *, run_id: str | None = None, project: str | None = None,
                    entity: str | None = None, init: Callable[..., Any] = _wandb_init,
                    stored: Callable[..., Any] = _stored_summary, now_ms: float | None = None) -> dict:
    """Write ``feed`` into the evaluation run's summary under ``reliquary_run_benchmarks``. The envelope keeps the
    TRAINER's run id (the dashboard matches it against its trainer source); the platform's checks run first."""
    from reliquary.eval.run_feed import encode_feed, parse_feed

    where = target(feed["run_id"], run_id=run_id, project=project, entity=entity)
    body = encode_feed(feed)                                    # the platform's 512 KiB bound
    parse_feed(json.loads(body), feed["run_id"], time.time() * 1000 if now_ms is None else now_ms)
    out = {"run": where["run_id"], "project": where["project"], "entity": where["entity"],
           "key": SUMMARY_KEY, "bytes": len(body)}
    if _same(stored(run_id=where["run_id"], project=where["project"], entity=where["entity"]), feed):
        return {**out, "written": False}
    run = init(run_id=where["run_id"], group=where["group"], project=where["project"], entity=where["entity"],
               config={"trainer_run_id": feed["run_id"]})
    run.summary[SUMMARY_KEY] = json.loads(body)
    run.finish()
    return {**out, "written": True}


def new_rows(feed: Mapping[str, Any], short_of: Mapping[str, str], logged: set[str]) -> tuple[list[dict], list[str]]:
    """One wandb row per step not yet logged: ``eval/<short>/pass@1`` and its ``delta`` to the baseline. Returns
    the rows (by step) and the ``<short>@<step>`` marks they cover. The baseline (step null: the start checkpoint)
    is plotted at ``rl_step`` 0. Ungraded points wait for a later feed."""
    rows: dict[int, dict] = {}
    marks = []
    for bench in feed["benchmarks"]:
        key = short_of.get(bench["id"]) or bench["id"]
        base = bench.get("baseline")
        base_score = None if base is None else base.get("score")
        for point in ([base] if base is not None else []) + list(bench["checkpoints"]):
            step = 0 if point.get("step") is None else int(point["step"])
            mark = f"{key}@{step}"
            if point.get("score") is None or mark in logged:
                continue
            row = rows.setdefault(step, {"rl_step": step})
            row[f"eval/{key}/pass@1"] = float(point["score"])
            if base_score is not None:
                row[f"eval/{key}/delta"] = float(point["score"]) - float(base_score)
            marks.append(mark)
    return [rows[s] for s in sorted(rows)], marks


def log_feed(feed: Mapping[str, Any], *, short_of: Mapping[str, str], state_path: Path,
             init: Callable[..., Any] = _wandb_init, run_id: str | None = None, project: str | None = None,
             entity: str | None = None) -> dict:
    """Log the feed's new points; fail-soft (the feed itself is unaffected), unlogged points retried next time."""
    state_path = Path(state_path)
    logged = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    rows, marks = new_rows(feed, short_of, logged)
    where = target(feed["run_id"], run_id=run_id, project=project, entity=entity)
    if not rows:
        return {"run": where["run_id"], "logged": 0}
    try:
        run = init(run_id=where["run_id"], group=where["group"], project=where["project"], entity=where["entity"],
                   config={"trainer_run_id": feed["run_id"]})
        for row in rows:
            run.log(row)
        run.finish()
    except Exception as exc:        # wandb down or misconfigured: the scores stay in the feed
        logger.warning("wandb eval log failed (%s): %d points kept for the next feed", exc, len(marks))
        return {"run": where["run_id"], "logged": 0, "error": type(exc).__name__}
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(logged | set(marks))))
    tmp.replace(state_path)
    return {"run": where["run_id"], "logged": len(marks)}


def mirror(feed: Mapping[str, Any], *, short_of: Mapping[str, str], state_path: Path, **where: Any) -> dict:
    """Both writes, fail-soft (``eval schedule``): a W&B outage never stops the schedule; the next feed retries."""
    out: dict[str, Any] = {}
    try:
        out["summary"] = publish_summary(feed, **where)
    except Exception as exc:        # noqa: BLE001
        logger.warning("wandb summary write failed (%s), retried with the next feed", exc)
        out["summary"] = {"written": False, "error": type(exc).__name__}
    out["history"] = log_feed(feed, short_of=short_of, state_path=state_path, **where)
    return out


__all__ = ["SUMMARY_KEY", "eval_run_ids", "log_feed", "mirror", "new_rows", "publish_summary", "stored_envelope",
           "target"]
