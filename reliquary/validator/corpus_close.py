"""Close a finished corpus task: stop new pay and free its share of the pool.

`tasks close` sets the task's cap to 0 (a 0 cap pays nothing new and counts for
nothing in the sum of caps) and retires it, once its job is drained: every
submission has a verdict and every verdict is settled.

It no longer waits for the task's earned pay to decay: a cap only prices the
periods still to be worked, and each period archive keeps paying, up to the cap
it was settled under, until its tail runs out (``corpus_periods.pay_ceiling``).
A task once settled by RL window has nothing left to pay: the weight setter no
longer reads its window archives.

A finished job's cap already goes to 0 by itself (``corpus_autoclose``); close
is what retires it, once the operator has taken it out of every corpus
validator's RELIQUARY_TASK_ID (a retired task named there stops that validator
from starting).
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


async def close_task(task_id: str, *,
                     read_registry: Callable[[], Awaitable] | None = None,
                     records=None, period_weights: Callable[..., Awaitable] | None = None,
                     set_cap: Callable[..., Awaitable] | None = None,
                     retire: Callable[..., Awaitable] | None = None,
                     drand_round: Callable[[], int] = current_drand_round) -> str:
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.corpus_job_status import stored_job_counts

    if read_registry is None:
        from reliquary.infrastructure.task_registry_store import read_registry
    if set_cap is None or retire is None:
        from reliquary.infrastructure import task_registry_store as store

        set_cap = set_cap or store.set_task_cap
        retire = retire or store.retire_task_entry
    if records is None:
        from reliquary.infrastructure.corpus_record_store import BucketRecordStore

        records = BucketRecordStore()
    if period_weights is None:
        from reliquary.validator.weight_only import WeightOnlyValidator

        period_weights = WeightOnlyValidator._period_weights

    entries, _ = await read_registry()
    entry = entries.get(task_id)
    if entry is None:
        raise TaskNotClosable(f"no task {task_id!r} in the registry")
    if entry.mechanism != MECHANISM_CORPUS_GENERATION:
        raise TaskNotClosable(f"{task_id} is {entry.mechanism!r}: only a corpus task closes")
    cap = float(entry.params.get("cap", 0.0))
    if not (await stored_job_counts(records, entry.job_id))["drained"]:
        raise TaskNotClosable(f"job {entry.job_id} is not drained: every submission must be "
                              "audited and settled first (reliquary jobs status)")
    tail = 0.0
    if cp.is_period_task(entry):
        # Told, not waited for: the tail is paid whatever the cap becomes.
        tail = sum((await period_weights({task_id: entry})).get(task_id, {}).values())
    if cap > 0:
        if entry.status != "active":
            raise TaskNotClosable(f"{task_id} is {entry.status} with cap {cap}: a retired cap "
                                  "cannot change")
        await set_cap(task_id, 0.0)
    if entry.status == "active":
        await retire(task_id, drand_round())
    paying = (f"; its earned tail ({tail:.6f} of the pool this period) keeps being paid "
              "until it runs out" if tail > 0 else "")
    return (f"closed {task_id}: cap 0, retired; its share of the pool is free{paying}. "
            "Take it out of every corpus validator's RELIQUARY_TASK_ID before restarting it")


__all__ = ["TaskNotClosable", "close_task", "current_drand_round"]
