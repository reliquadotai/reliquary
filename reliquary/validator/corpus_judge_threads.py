"""The judges' own threads, apart from the event loop's default executor.

The route's every thread hop (admission, ledger and record encode/decode,
the DNS lookups of new store connections) runs on the loop's default executor
(min(32, cpu + 4) threads). On 2026-10-02 four judges catching up a backlog
filled it with drand relay races (each holds its thread until a relay
answers), GPU forwards and record decodes, and the route's hops queued behind
them: 6 s for a trivial admit, 10-35 s per store call, 503s. Judge work runs
here instead, in pools sized so the judges cannot outgrow them.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor

# Record and verdict encode/decode of every job's judge and settler.
CODEC_THREADS = 8
# drand relay races in flight at once, across jobs: each also opens a
# connection to every relay, so this bounds those sockets too.
BEACON_THREADS = 16
# Large records read at once across every judge (decoded on CODEC_THREADS).
JUDGE_READS = 32


class JudgeThreads:
    """``codec``, ``beacon`` and ``gpu`` (one thread: forwards run one at a
    time under the GPU lock, and never wait for a thread while holding it)."""

    def __init__(self, *, codec: int = CODEC_THREADS, beacon: int = BEACON_THREADS) -> None:
        self.codec = ThreadPoolExecutor(codec, thread_name_prefix="corpus-judge-codec")
        self.beacon = ThreadPoolExecutor(beacon, thread_name_prefix="corpus-judge-drand")
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="corpus-judge-gpu")

    def shutdown(self) -> None:
        for pool in (self.codec, self.beacon, self.gpu):
            pool.shutdown(wait=False, cancel_futures=True)


async def run_in(executor, func, *args):
    """``asyncio.to_thread`` on ``executor`` (the default one when None)."""
    if executor is None:
        return await asyncio.to_thread(func, *args)
    loop = asyncio.get_running_loop()
    call = functools.partial(contextvars.copy_context().run, func, *args)
    return await loop.run_in_executor(executor, call)


def judge_record_store(threads: JudgeThreads, *, max_pool_connections: int = 64):
    """The record store the judges and settlers use: its own connections, its
    codec on ``threads.codec``, at most JUDGE_READS record reads at once."""
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    return BucketRecordStore(max_pool_connections=max_pool_connections,
                             executor=threads.codec, max_reads=JUDGE_READS)
