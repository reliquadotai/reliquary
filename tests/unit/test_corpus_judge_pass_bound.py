"""A judge pass audits at most a bounded number of records (rows and tokens),
so its verdicts are written minutes after it starts, never after one
GPU call covering a whole restart backlog's siblings. What a pass defers is
judged in the next ones; an unaudited record whose drawn sibling was
deferred waits for it. Prod 2026-10-02 20:05: the first math pass after a
restart (110k pending) wrote nothing for 50+ minutes."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.validator import corpus_auditor
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_judge_sim as sim
from tests.unit import test_corpus_judge_equivalence as equivalence
from tests.unit import _main_corpus_auditor as main_module

HOLD = 300.0


def _backlog_run(monkeypatch, *, rows, tokens, n=3000):
    """``n`` records up to an hour old from 6 hotkeys, steady traffic on top,
    every forward recorded (records per call, and when)."""
    monkeypatch.setattr(corpus_auditor, "PASS_AUDIT_ROWS", rows)
    monkeypatch.setattr(corpus_auditor, "PASS_AUDIT_TOKENS", tokens)
    loop = sim.VirtualTimeLoop()
    calls, first_verdict = [], {}

    async def scenario():
        store = equivalence._Store()
        clock = sim._Clock()
        hotkeys = [f"5Hk{c}" for c in "ABCDEF"]
        for hotkey in hotkeys:
            store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
        params = AuditParams(q=0.25, probation_submissions=5, hold_seconds=HOLD,
                             ban_after_failures=1000)
        auditor = corpus_auditor.CorpusAuditor(
            job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
            params=params, miner_states=MinerStates(store, "math-v1", clock=clock),
            beacon=sim.Beacon(latency=0.0), round_at=sim.round_at, clock=clock,
            accept_slack_seconds=30.0, rescan_every_seconds=20.0)

        async def forward(records, *, local=False):
            calls.append((sim.virtual_clock(), len(records)))
            await asyncio.sleep(0.5 * len(records))      # the GPU, 0.5 s a record
            return [dict(equivalence._OK) for _ in records]

        auditor._forward = forward
        t0 = sim.virtual_clock()
        for i in range(n):
            sid = "%064x" % (i + 1)
            store.submissions[sid] = {"hotkey": hotkeys[i % 6],
                                      "received_at": t0 - 3600 + 3600 * i / n,
                                      "token_count": 100, "completions": [{"tokens": [1]}]}

        async def arrive():
            for i in range(10**6):
                sid = "%064x" % (10**7 + i)
                store.submissions[sid] = {"hotkey": hotkeys[i % 6],
                                          "received_at": sim.virtual_clock(),
                                          "token_count": 100, "completions": [{"tokens": [1]}]}
                auditor.enqueue(sid)
                await asyncio.sleep(2.0)

        feeder = asyncio.ensure_future(arrive())
        runner = asyncio.ensure_future(auditor.run())
        start = sim.virtual_clock()
        while sim.virtual_clock() < start + 3 * 3600:
            await asyncio.sleep(10.0)
            if store.verdicts and "at" not in first_verdict:
                first_verdict["at"] = sim.virtual_clock() - start
        feeder.cancel()
        runner.cancel()
        for task in (feeder, runner):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        backlog = {"%064x" % (i + 1) for i in range(n)}
        return store, backlog

    try:
        store, backlog = loop.run_until_complete(scenario())
    finally:
        loop.close()
    return calls, first_verdict.get("at"), store, backlog


def test_every_forward_is_bounded_and_verdicts_flow_from_the_first_minutes(monkeypatch):
    calls, first, store, backlog = _backlog_run(monkeypatch, rows=64, tokens=10**9)
    assert max(size for _, size in calls) <= 64, max(size for _, size in calls)
    # The first verdicts within a few minutes (one bounded call, ~32 s of GPU).
    assert first is not None and first < 300, first
    # Everything is judged in the end, unaudited ones included.
    assert backlog <= set(store.verdicts)
    assert any(not store.verdicts[sid]["audited"] for sid in backlog)


def test_the_token_budget_bounds_a_call_too(monkeypatch):
    calls, _, store, backlog = _backlog_run(monkeypatch, rows=10**6, tokens=40 * 100)
    assert max(size for _, size in calls) <= 40
    assert backlog <= set(store.verdicts)


def test_unbounded_one_call_audits_a_pass_and_all_its_drawn_siblings(monkeypatch):
    """The failure itself: a call grows with the backlog, not with the pass."""
    calls, _, _, _ = _backlog_run(monkeypatch, rows=10**6, tokens=10**12)
    assert max(size for _, size in calls) > 2 * 64, max(size for _, size in calls)


@pytest.mark.parametrize("seed", [7, 3])
def test_a_bounded_judge_writes_main_s_verdicts(monkeypatch, seed):
    """The faulted scenario of the equivalence test (ban, unreadable record,
    failed write, quarantined executor, restart) with passes of 3 audits."""
    monkeypatch.setattr(corpus_auditor, "PASS_AUDIT_ROWS", 3)
    faults = equivalence.ALL
    main, _ = equivalence._run(main_module, "poll", faults, seed)
    bounded, _ = equivalence._run(corpus_auditor, "run", faults, seed)
    assert equivalence._outcome(bounded) == equivalence._outcome(main)
