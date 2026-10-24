#!/usr/bin/env python3
"""Move a period task's queued archives to the earliest entries with room.

A settler older than ``cp.CATCHUP_ENTRIES`` gave every archive its own entry
period, strictly increasing: a backlog settled at once queued one per period,
ahead of every later period's pay, for as long as the job ran. This rewrites
each archive whose entry is still in the future at the first entry from the
next period on holding fewer than ``CATCHUP_ENTRIES`` archives, then deletes
the old one. The rewards are copied unchanged: only the date they enter moves.

Run it only once EVERY weight setter replays with the catch-up bound (the
release carrying ``CATCHUP_ENTRIES``): an older one clamps a task at one cap,
and would burn what is paid back.

Safe to stop at any point and run again. The new archive is written first and
names the one it replaces (``replaces_entry_period``), which the weight setter
skips while both exist; the old one's entry is in the future meanwhile, so no
weight set counts both.

Dry run by default; ``--execute`` writes.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from reliquary.infrastructure.corpus_period_store import (
    R2PeriodArchives,
    delete_period_archive,
)
from reliquary.validator import corpus_periods as cp


def plan(listed: list[tuple[int, int]], due: int, *,
         per_period: int = cp.CATCHUP_ENTRIES) -> list[tuple[int, int, int]]:
    """``(work, old entry, new entry)`` for every archive that can enter
    earlier, oldest work first."""
    queued = sorted((work, entry) for work, entry in listed if entry > due)
    taken = [entry for work, entry in listed if entry <= due]
    moves = []
    for work, entry in queued:
        new = min(cp.entry_for(due, taken, per_period=per_period), entry)
        taken.append(new)
        if new < entry:
            moves.append((work, entry, new))
    return moves


async def _stale(archives, task_id, listed) -> list[tuple[int, int]]:
    """Old archives left behind by an interrupted run: replaced, not deleted."""
    by_work: dict[int, list[int]] = {}
    for work, entry in listed:
        by_work.setdefault(work, []).append(entry)
    stale = []
    for work, entries in by_work.items():
        for entry in entries:
            doc = await archives.read(task_id, work, entry)
            moved = (doc or {}).get("replaces_entry_period")
            if moved is not None and int(moved) in entries:
                stale.append((work, int(moved)))
    return stale


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    archives = R2PeriodArchives()
    current = cp.period_of(time.time())
    due = current + 1
    listed = await archives.list(args.task_id)
    stale = await _stale(archives, args.task_id, listed)
    listed = [key for key in listed if key not in stale]
    moves = plan(listed, due)

    print(f"task {args.task_id}: period {current}, {len(listed)} archives, "
          f"{len(stale)} stale, {len(moves)} to move (entries from {due})")
    for work, old, new in moves:
        print(f"  work {work}: entry {old} -> {new}")
    moved = {(work, old): new for work, old, new in moves}
    docs = []
    for work, entry in listed:
        doc = await archives.read(args.task_id, work, entry) or {}
        docs.append({"entry_period": moved.get((work, entry), entry),
                     "rewards_by_hotkey": doc.get("rewards_by_hotkey") or {}})
    # What the task pays once moved (the weight setter bounds it at CATCHUP_ENTRIES caps).
    for period in range(current, due + 8):
        print(f"  pay in period {period}: {sum(cp.replay(docs, period).values()):.4f}")
    if not args.execute:
        print("dry run: nothing written (--execute to move)")
        return 0

    for work, entry in stale:
        await delete_period_archive(args.task_id, work, entry)
        print(f"  deleted stale {work}-{entry}")
    for work, old, new in moves:
        doc = await archives.read(args.task_id, work, old)
        if doc is None:
            raise RuntimeError(f"archive {work}-{old} vanished; run again")
        await archives.write(args.task_id, work, new,
                             {**doc, "entry_period": new, "replaces_entry_period": old})
        await delete_period_archive(args.task_id, work, old)
        print(f"  moved {work}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
