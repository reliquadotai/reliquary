"""Judging can no longer slow the routes: real processes, real time.

The prod shape of 2026-10-02 (scaled by env): four jobs, the math one 10-20k records behind
(16-32k token completions, 0.5-1 MB each), its settlement 300k ids deep, the
three others 1,500 behind; R2-like latency; a GPU at 4.8k tokens/s whose
proof check spends its real CPU. Miners submit 16-32k token math records at
2 a second for two minutes while the judges catch up.

- Split (math in its own judge process, the other three in a second one):
  the front answers as if nothing were judging.
- The single process, same scenario: the same probe degrades -- the test
  sees the problem the split removes.

Heavy (5-6 processes, ~3-4 GB RAM, ~6 min): run with
``RELIQUARY_CORPUS_SPLIT_LOAD_TEST=1``."""

from __future__ import annotations

import json
import os
import random
import time

import pytest

from reliquary.infrastructure import corpus_record_store as record_store
from tests.unit import corpus_split_harness as h

pytestmark = pytest.mark.skipif(
    os.environ.get("RELIQUARY_CORPUS_SPLIT_LOAD_TEST") != "1",
    reason="multi-process load test: set RELIQUARY_CORPUS_SPLIT_LOAD_TEST=1")

JOBS = ["math-v1", "code-v1", "if-v1", "logic-v1"]
# The run measured on 2026-10-02: 20000 / 1500 / 300k / 180 s (see the asserts).
MATH_BACKLOG = int(os.environ.get("RELIQUARY_CORPUS_SPLIT_LOAD_BACKLOG", "10000"))
OTHER_BACKLOG = int(os.environ.get("RELIQUARY_CORPUS_SPLIT_LOAD_OTHERS", "800"))
SETTLED = int(os.environ.get("RELIQUARY_CORPUS_SPLIT_LOAD_SETTLED", "300000"))
LOAD_SECONDS = float(os.environ.get("RELIQUARY_CORPUS_SPLIT_LOAD_SECONDS", "90"))


def _settled_state(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return json.dumps({
        "schema": "reliquary/corpus-settlement/v1", "last_window": 100,
        "settled": sorted("%064x" % rng.getrandbits(256) for _ in range(n)),
        "other_max_seen": None, "other_max_seen_at": None, "advanced_at": None, "pending": None,
        "totals": {"verdicts": n, "passed": n, "verified_tokens": n * 20_000, "complete": True},
    }).encode()


def _scenario(root):
    h.seed_bucket(root, JOBS)
    now = time.time()
    h.seed_backlog(root, "math-v1", MATH_BACKLOG, now=now, oldest=4 * 3600, newest=0.0)
    h.put_raw(root, record_store._settlement_key("math-v1"), _settled_state(SETTLED, 1))
    for k, job in enumerate(JOBS[1:]):
        h.seed_backlog(root, job, OTHER_BACKLOG, now=now, oldest=4 * 3600, newest=0.0,
                       seed=10 + k)
        h.put_raw(root, record_store._settlement_key(job), _settled_state(SETTLED // 10, 2 + k))
    return [(h.entry(f"corpus-{job.split('-')[0]}", job, audit_hold_seconds=3600.0), 0.1)
            for job in JOBS]


def _probe(base):
    import asyncio

    asyncio.run(h.wait_http(base, timeout=180))
    time.sleep(3.0)
    return h.load_in_process(base, "math-v1", seconds=LOAD_SECONDS, rate=2.0, seed=3)


def _measure(tmp_path, mode):
    root = tmp_path / mode
    root.mkdir()
    log = tmp_path / f"{mode}.log"
    with h.environment(h.harness_env(root, CORPUS_HARNESS_LOG=log)):
        served = _scenario(root)
        port = h.free_port()
        base = f"http://127.0.0.1:{port}"
        if mode == "split":
            spec = h.split_spec(root, served, [["math-v1"], JOBS[1:]], port=port)
            with h.SupervisorThread(spec):
                results = _probe(base)
        else:
            process = h.start_single(served, port=port)
            try:
                results = _probe(base)
            finally:
                process.kill()
                process.join(10)
        reads = log.read_text().count("record reads in this process")
    stats = h.quantiles(results)
    stats["judge_record_reads_k"] = reads
    print(f"\n{mode}: {stats}")
    return stats


def test_the_front_stays_fast_while_judges_catch_up_and_the_single_process_does_not(tmp_path):
    split = _measure(tmp_path, "split")
    single = _measure(tmp_path, "single")
    # Both judged as hard: the probe ran while the backlog was being read.
    assert split["judge_record_reads_k"] >= 5 and single["judge_record_reads_k"] >= 5
    assert split["errors"] == 0 and split["accepted"] == split["n"] > 100
    # Measured on 2026-10-02 (8 vCPU, shared): split p50 0.45 s p99 0.70 s, at
    # the no-judge baseline (0.43 / 0.62); single p50 0.66 s p99 1.56 s.
    assert split["p99"] < 1.0, split
    assert single["p99"] > max(1.0, 1.5 * split["p99"]), (single, split)
    assert single["p90"] > 1.5 * split["p90"], (single, split)
