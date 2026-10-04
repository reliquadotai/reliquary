"""Close a finished corpus task: stop its pay and free its share of the pool.

`tasks close` sets the task's cap to 0 (a 0 cap pays nothing and counts for
nothing in the sum of caps) and retires it, once nothing is owed any more:

- its job is drained: every submission has a verdict and every verdict is
  settled;
- a period-settled task's replayed pay is below ``CLOSE_THRESHOLD`` of its cap
  (it has decayed: what it earned has been paid);
- a task settled the old way never decays by itself (its pay is frozen until
  its archives leave the window horizon), so it closes only when the operator
  says to cut that tail (``cut_tail``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable


class TaskNotClosable(RuntimeError):
    pass


def current_drand_round(now: float | None = None) -> int:
    from reliquary.infrastructure import drand

    chain = drand.get_current_chain()
    genesis, period = chain.get("genesis_time"), chain.get("period")
    if genesis is None or not period:
        raise RuntimeError("drand chain info is not known yet")
    return int(math.floor(((time.time() if now is None else now) - genesis) / period)) + 1


async def close_task(task_id: str, *, cut_tail: bool = False,
                     read_registry: Callable[[], Awaitable] | None = None,
                     records=None, period_weights: Callable[..., Awaitable] | None = None,
                     set_cap: Callable[..., Awaitable] | None = None,
                     retire: Callable[..., Awaitable] | None = None,
                     drand_round: Callable[[], int] = current_drand_round) -> str:
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, MECHANISM_NATIVE_AFFINE_POINTS
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.corpus_job_status import stored_job_counts

    if read_registry is None:
        from reliquary.infrastructure.task_registry_store import read_registry
    if set_cap is None or retire is None:
        from reliquary.infrastructure import task_registry_store as store

        set_cap = set_cap or store.set_task_cap
        retire = retire or store.retire_task_entry
    if period_weights is None:
        from reliquary.validator.weight_only import WeightOnlyValidator

        period_weights = WeightOnlyValidator._period_weights

    entries, _ = await read_registry()
    entry = entries.get(task_id)
    if entry is None:
        raise TaskNotClosable(f"no task {task_id!r} in the registry")
    if entry.mechanism not in {MECHANISM_CORPUS_GENERATION, MECHANISM_NATIVE_AFFINE_POINTS}:
        raise TaskNotClosable(f"{task_id} is {entry.mechanism!r}: only a corpus task or native Affine task closes")
    cap = float(entry.params.get("cap", 0.0))
    if entry.mechanism == MECHANISM_NATIVE_AFFINE_POINTS:
        from reliquary.integrations.affine_competition import read_archive

        if await read_archive(entry) is None:
            raise TaskNotClosable("native Affine task has no finalized settlement archive")
    else:
        if records is None:
            from reliquary.infrastructure.corpus_record_store import BucketRecordStore

            records = BucketRecordStore()
        if not (await stored_job_counts(records, entry.job_id))["drained"]:
            raise TaskNotClosable(f"job {entry.job_id} is not drained: every submission must be "
                                  "audited and settled first (reliquary jobs status)")
    if cp.is_period_task(entry):
        paying = sum((await period_weights({task_id: entry})).get(task_id, {}).values())
        if cap > 0 and paying >= cp.CLOSE_THRESHOLD * cap:
            raise TaskNotClosable(
                f"{task_id} still pays {paying:.6f} of the pool ({paying / cap:.1%} of its cap): "
                "what it earned is not paid yet; close it once it has decayed")
    elif not cut_tail and cap > 0:
        raise TaskNotClosable(
            f"{task_id} is settled by RL window: its pay does not decay by itself and only "
            "ends when its archives leave the window horizon. Pass --cut-tail to stop paying "
            "it now")
    if cap > 0:
        if entry.status != "active":
            raise TaskNotClosable(f"{task_id} is {entry.status} with cap {cap}: a retired cap "
                                  "cannot change")
        await set_cap(task_id, 0.0)
    if entry.status == "active":
        await retire(task_id, drand_round())
    return f"closed {task_id}: cap 0, retired; its share of the pool is free"


__all__ = ["TaskNotClosable", "close_task", "current_drand_round"]
