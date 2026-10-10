"""A judge pass fetches the drand rounds its bounded work needs, once each.

Prod 2026-10-02 23:02-23:35: the first math pass after a restart (130k
pending) decided every sibling of every payable record in its 512 ids
(thousands of records, ~1.5k rounds) and re-raced, one at a time, each round
whose relays had failed once the 30 s negative cache expired; it never
finished. Here: the siblings one pass decides are bounded, a round that failed
is not raced again in the same pass, a pass announces itself, and the miner
book's backfill paces its reads."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.validator import corpus_auditor
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_judge_sim as sim
from tests.unit import test_corpus_judge_equivalence as equivalence
from tests.unit import _main_corpus_auditor as main_module

HOLD = 3600.0


class _SlowBadBeacon(sim.Beacon):
    """Every round answers in 0.2 s, except one in 97: its relays fail after 90 s."""

    def __init__(self):
        super().__init__(latency=0.2)
        self.per_round = collections.Counter()

    def virtual_seconds(self, round_number):
        return 90.0 if round_number % 97 == 0 else 0.2

    def __call__(self, round_number):
        self.per_round[round_number] += 1
        if round_number % 97 == 0:
            raise ConnectionError("All relays/paths failed")
        return super().__call__(round_number)


def _restart(monkeypatch, *, siblings, n=20000, hours=1.0):
    """``n`` records over the last 3 h from 6 sampled hotkeys, a restarted judge."""
    monkeypatch.setattr(corpus_auditor, "PASS_SIBLINGS", siblings)
    loop = sim.VirtualTimeLoop()
    beacon = _SlowBadBeacon()
    passes = []

    async def scenario():
        store = equivalence._Store()
        clock = sim._Clock()
        hotkeys = [f"5Hk{c}" for c in "ABCDEF"]
        for hotkey in hotkeys:
            store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
        params = AuditParams(q=0.15, probation_submissions=5, hold_seconds=HOLD,
                             ban_after_failures=1000)
        auditor = corpus_auditor.CorpusAuditor(
            job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
            params=params, miner_states=MinerStates(store, "math-v1", clock=clock),
            beacon=beacon, round_at=sim.round_at, clock=clock,
            accept_slack_seconds=30.0, rescan_every_seconds=20.0,
            # The pass's own races are counted here, not the background prefetch's.
            prefetch_rounds=False)

        async def forward(records, *, local=False):
            await asyncio.sleep(0.05 * len(records))
            return [dict(equivalence._OK) for _ in records]

        auditor._forward = forward
        real = auditor.judge_many

        async def judge_many(ids):
            start = sim.virtual_clock()
            before = sum(beacon.per_round.values())
            await real(ids)
            passes.append((sim.virtual_clock() - start, sum(beacon.per_round.values()) - before))

        auditor.judge_many = judge_many
        t0 = sim.virtual_clock()
        for i in range(n):
            sid = "%064x" % (i + 1)
            store.submissions[sid] = {"hotkey": hotkeys[i % 6],
                                      "received_at": t0 - 3 * 3600 + 3 * 3600 * i / n,
                                      "token_count": 100, "completions": [{"tokens": [1]}]}
        runner = asyncio.ensure_future(auditor.run())
        await asyncio.sleep(hours * 3600)
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        return store

    try:
        store = loop.run_until_complete(scenario())
    finally:
        loop.close()
    return store, passes, beacon


def test_a_pass_races_each_round_once_and_its_siblings_are_bounded(monkeypatch):
    store, passes, beacon = _restart(monkeypatch, siblings=256)
    # A failing round is raced once per pass, never once per record of it.
    assert passes and max(fetched for _, fetched in passes) <= 256 + 512, passes[:5]
    assert passes[0][0] < 15 * 60, passes[:3]
    # Verdicts flow, unaudited ones included, and old rounds that fail are audited.
    unaudited = [v for v in store.verdicts.values() if not v["audited"]]
    assert len(store.verdicts) > 3000 and unaudited


def test_with_unbounded_siblings_a_pass_races_many_more_rounds(monkeypatch):
    """What the bound removes: a pass's siblings (and rounds) grow with the backlog."""
    _, bounded, _ = _restart(monkeypatch, siblings=64, hours=0.5)
    _, unbounded, _ = _restart(monkeypatch, siblings=10**9, hours=0.5)
    assert max(f for _, f in unbounded) > 2 * max(f for _, f in bounded), (bounded[:3],
                                                                          unbounded[:3])


def test_each_pass_announces_its_work(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_auditor"):
        _restart(monkeypatch, siblings=256, n=600, hours=0.2)
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("corpus judge pass started")]
    assert lines and all("job=math-v1" in l and "rounds_needed=" in l and "siblings=" in l
                         for l in lines), lines[:3]


@pytest.mark.parametrize("seed", [7, 3])
def test_a_sibling_bounded_judge_writes_main_s_verdicts(monkeypatch, seed):
    monkeypatch.setattr(corpus_auditor, "PASS_SIBLINGS", 4)
    faults = equivalence.ALL
    main, _ = equivalence._run(main_module, "poll", faults, seed)
    bounded, _ = equivalence._run(corpus_auditor, "run", faults, seed)
    assert equivalence._outcome(bounded) == equivalence._outcome(main)


def test_the_miner_book_backfill_paces_its_reads(monkeypatch):
    from reliquary.validator import corpus_miner_status
    from reliquary.validator.corpus_miner_status import MinerBook

    class _Records:
        def __init__(self):
            self.times = []

        async def read_settlement(self, job_id):
            return {}, None

        async def list_verdict_ids(self, job_id):
            return ["%064x" % i for i in range(400)]

        async def read_verdict(self, job_id, sid):
            self.times.append(asyncio.get_running_loop().time())
            return {"hotkey": "5A", "passed": True, "token_count": 1, "audited_at": 0.0}

    records = _Records()
    book = MinerBook(job_id="math-v1", task_id="t", records=records)
    loop = sim.VirtualTimeLoop()
    try:
        loop.run_until_complete(book.backfill())
    finally:
        loop.close()
    rate = len(records.times) / (records.times[-1] - records.times[0])
    assert book.complete and rate <= corpus_miner_status.BACKFILL_READS_PER_SECOND * 1.05
