"""Mirror an RL run's held-out feed (``run_feed``) into wandb, beside the trainer's run.

The scores go to a sibling run ``<group>-evals`` of the trainer's group, plotted against ``rl_step`` (the trainer's
step): a second writer on the trainer's own run would need wandb's shared mode on the trainer too. Points already
logged are remembered in a small JSON file, so each one is logged once however often the feed is rebuilt.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRAINER_SUFFIX = "-trainer"


def eval_run_ids(trainer_run_id: str) -> tuple[str, str]:
    """``(group, run id)`` of the eval run beside ``trainer_run_id`` (``{group}-trainer``, ``telemetry.init``)."""
    group = trainer_run_id[:-len(TRAINER_SUFFIX)] if trainer_run_id.endswith(TRAINER_SUFFIX) else trainer_run_id
    return group, f"{group}-evals"


def new_rows(feed: Mapping[str, Any], short_of: Mapping[str, str], logged: set[str]) -> tuple[list[dict], list[str]]:
    """One wandb row per step not yet logged: ``eval/<short>/pass@1`` and its ``delta`` to the baseline. Returns
    the rows (by step) and the ``<short>@<step>`` marks they cover. Ungraded points wait for a later feed."""
    rows: dict[int, dict] = {}
    marks = []
    for bench in feed["benchmarks"]:
        key = short_of.get(bench["id"]) or bench["id"]
        base = bench.get("baseline")
        base_score = None if base is None else base.get("score")
        for point in ([base] if base is not None else []) + list(bench["checkpoints"]):
            mark = f"{key}@{point['step']}"
            if point.get("score") is None or mark in logged:
                continue
            row = rows.setdefault(int(point["step"]), {"rl_step": int(point["step"])})
            row[f"eval/{key}/pass@1"] = float(point["score"])
            if base_score is not None:
                row[f"eval/{key}/delta"] = float(point["score"]) - float(base_score)
            marks.append(mark)
    return [rows[s] for s in sorted(rows)], marks


def _wandb_init(*, run_id: str, group: str, config: dict):
    import wandb

    run = wandb.init(project=os.getenv("WANDB_PROJECT", "reliquary-validator"), entity=os.getenv("WANDB_ENTITY"),
                     id=run_id, group=group, job_type="evals", resume="allow", config=config)
    run.define_metric("rl_step")
    run.define_metric("eval/*", step_metric="rl_step")
    return run


def log_feed(feed: Mapping[str, Any], *, short_of: Mapping[str, str], state_path: Path,
             init: Callable[..., Any] = _wandb_init) -> dict:
    """Log the feed's new points; fail-soft (the feed itself is unaffected), unlogged points retried next time."""
    state_path = Path(state_path)
    logged = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    rows, marks = new_rows(feed, short_of, logged)
    group, run_id = eval_run_ids(feed["run_id"])
    if not rows:
        return {"run": run_id, "logged": 0}
    try:
        run = init(run_id=run_id, group=group, config={"trainer_run_id": feed["run_id"]})
        for row in rows:
            run.log(row)
        run.finish()
    except Exception as exc:        # wandb down or misconfigured: the scores stay in the feed
        logger.warning("wandb eval log failed (%s): %d points kept for the next feed", exc, len(marks))
        return {"run": run_id, "logged": 0, "error": type(exc).__name__}
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(logged | set(marks))))
    tmp.replace(state_path)
    return {"run": run_id, "logged": len(marks)}


__all__ = ["eval_run_ids", "log_feed", "new_rows"]
