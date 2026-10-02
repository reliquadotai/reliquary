"""The judge's scheduler: a record is judged again only when its decision can
change, oldest first, and the verdicts are the ones a judge re-deciding every
pending record continuously would write. Virtual time, in-memory stores."""

from __future__ import annotations

import asyncio
import contextlib
import random

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState, drawn
from reliquary.validator import corpus_auditor
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_judge_sim as sim

PARAMS = AuditParams(q=0.25, probation_submissions=5, hold_seconds=300.0,
                     suspect_seconds=86400.0, ban_after_failures=1000)
# Arrivals land at .1 s past a whole second; with this slack a hold ends on a
# TICK, and a hotkey's window can only empty at .101 s, alone in its TICK. The
# judge that decides every TICK then sees every event in the scheduler's order.
SLACK = 29.9
TICK = 0.25


def _scenario(seed=7):
    """(arrival offset, hotkey) pairs: three steady miners, one that stops,
    one new (on probation), and a sampled cheater's burst of failing records."""
    rng = random.Random(seed)
    arrivals = []
    for hotkey, start, stop in [("5Steady0", 0, 1500), ("5Steady1", 3, 1500),
                                ("5Steady2", 5, 1500), ("5Stops", 2, 400),
                                ("5New", 20, 1500)]:
        t = start
        while t < stop:
            arrivals.append((t + 0.1, hotkey))
            t += rng.randint(4, 15)
    arrivals += [(600 + i + 0.1, "5Cheat") for i in range(12)]
    return sorted(arrivals)


def _ids(n, seed):
    rng = random.Random(seed)
    return ["%064x" % rng.getrandbits(256) for _ in range(n)]


async def _judge(arrivals, ids, *, mode, end):
    store = sim.Store(latency=(0.0, 0.0))
    for hotkey in ("5Steady0", "5Steady1", "5Steady2", "5Stops", "5Cheat"):
        store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
    clock = sim._Clock()
    auditor = corpus_auditor.CorpusAuditor(
        job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
        params=PARAMS, miner_states=MinerStates(store, "math-v1", clock=clock),
        beacon=sim.Beacon(latency=0.0), round_at=sim.round_at, clock=clock,
        accept_slack_seconds=SLACK, rescan_every_seconds=60.0)
    gpu = sim.Gpu(tokens_per_second=1e12, cheaters={"5Cheat"})
    auditor._judge_many = gpu
    t0 = sim.virtual_clock()
    landed = []

    async def arrive():
        for (offset, hotkey), sid in zip(arrivals, ids):
            await asyncio.sleep(t0 + offset - sim.virtual_clock())
            store.submissions[sid] = {"hotkey": hotkey, "received_at": sim.virtual_clock(),
                                      "token_count": 100, "completions": [{"tokens": [1]}]}
            landed.append(sid)
            if mode == "run":
                auditor.enqueue(sid)

    feeder = asyncio.ensure_future(arrive())
    if mode == "run":
        runner = asyncio.ensure_future(auditor.run())
        await asyncio.sleep(end)
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
    else:
        # The reference: every pending record re-decided every TICK, on the grid.
        tick = 0
        while sim.virtual_clock() < t0 + end:
            await auditor.judge_many([sid for sid in landed if sid not in store.verdicts])
            tick += 1
            await asyncio.sleep(max(0.0, t0 + tick * TICK - sim.virtual_clock()))
    feeder.cancel()
    return store, gpu


def _run(mode, arrivals, ids, end=2400.0):
    loop = sim.VirtualTimeLoop()
    try:
        return loop.run_until_complete(_judge(arrivals, ids, mode=mode, end=end))
    finally:
        loop.close()


def _key(v):
    return (v["hotkey"], v["passed"], v["audited"], v["reason"], v.get("draw"))


@pytest.mark.parametrize("seed", [7, 1, 5])
def test_the_scheduler_writes_the_verdicts_of_a_judge_that_never_stops_deciding(seed):
    arrivals = _scenario(seed)
    ids = _ids(len(arrivals), seed=seed + 10)
    reference, _ = _run("poll", arrivals, ids)
    scheduled, gpu = _run("run", arrivals, ids)
    assert set(reference.verdicts) == set(ids)
    assert {sid: _key(v) for sid, v in scheduled.verdicts.items()} == \
        {sid: _key(v) for sid, v in reference.verdicts.items()}
    verdicts = list(scheduled.verdicts.values())
    hotkey_of = dict(zip(ids, (h for _, h in arrivals)))
    # Every path is taken: unaudited passes, draws, the slow-hotkey audit of
    # the miner that stopped, probation, and a cheater caught and audited back.
    assert any(v["passed"] and not v["audited"] for v in verdicts)
    assert any(v.get("draw", {}).get("drawn") for v in verdicts)
    assert any(v["audited"] and v.get("draw") is None and hotkey_of[v["submission_id"]] == "5Stops"
               for v in verdicts)
    cheats = [v for v in verdicts if v["hotkey"] == "5Cheat"]
    assert len(cheats) == 12 and all(not v["passed"] and v["audited"] for v in cheats)
    assert sum(not v.get("draw", {}).get("drawn", False) for v in cheats) > 0
    assert gpu.audited < 2 * len(ids)


@pytest.mark.parametrize("seed", [1, 2])
def test_a_judge_without_sibling_pruning_or_parallel_writes_writes_the_same_verdicts(
        monkeypatch, seed):
    """The pass itself: siblings known undrawn are not re-decided and the
    writes go out together; with neither, the same passes write the same."""
    arrivals = _scenario(seed)
    ids = _ids(len(arrivals), seed=seed)
    fast, _ = _run("run", arrivals, ids)

    class _NeverKept(set):
        def add(self, item):
            pass

    real_init = corpus_auditor.CorpusAuditor.__init__

    def plain(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self._undrawn = _NeverKept()
        self.write_concurrency = 1

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "__init__", plain)
    slow, _ = _run("run", arrivals, ids)
    assert {sid: _key(v) for sid, v in fast.verdicts.items()} == \
        {sid: _key(v) for sid, v in slow.verdicts.items()}


def test_a_waiting_record_is_not_judged_again_before_its_hold_ends():
    arrivals = [(i * 5 + 0.1, "5Steady0") for i in range(60)]
    ids = [sid for sid in _ids(400, seed=3) if not drawn(sim.Beacon()(1), sid, 0.25)][:60]
    store, _ = _run("run", arrivals, ids, end=200.0)
    # Mid-hold: nothing paid, and each record looked at a handful of times,
    # not once a pass.
    assert store.verdicts == {} or all(v["audited"] for v in store.verdicts.values())
    assert store.calls["read_submission"] <= len(arrivals) + 5


def test_due_ids_come_out_earliest_first_then_oldest_first():
    auditor = corpus_auditor.CorpusAuditor(
        job_id="math-v1", records=None, model=None, tokenizer=None, proof=None,
        clock=lambda: 100.0)
    auditor._meta = {"a" * 64: ("5H", 50.0, 1), "b" * 64: ("5H", 10.0, 1),
                     "c" * 64: ("5H", 5.0, 1)}
    auditor._schedule("a" * 64, 90.0)
    auditor._schedule("b" * 64, 90.0)
    auditor._schedule("c" * 64, 120.0)
    auditor._schedule("a" * 64, 95.0)  # rescheduled: the earlier entry is stale
    assert auditor._take_due(100.0, 10) == ["b" * 64, "a" * 64]
    assert auditor._take_due(130.0, 10) == ["c" * 64]
    assert auditor._queued == {"a" * 64, "b" * 64, "c" * 64}


def test_a_rescan_does_not_pull_a_waiting_record_forward():
    auditor = corpus_auditor.CorpusAuditor(
        job_id="math-v1", records=None, model=None, tokenizer=None, proof=None,
        clock=lambda: 100.0)
    auditor._meta = {"a" * 64: ("5H", 50.0, 1)}
    auditor._schedule("a" * 64, 5000.0)
    auditor.enqueue("a" * 64)
    assert auditor._take_due(100.0, 10) == []
    assert auditor._due == {"a" * 64: 5000.0}


@pytest.mark.parametrize("q, enough", [(1.0, 1), (0.5, 2), (0.25, 4), (0.15, 7),
                                       (0.1, 10), (0.3, 4)])
def test_the_slow_hotkey_threshold_matches_the_policy(q, enough):
    assert corpus_auditor.CorpusAuditor._fewest_not_slow(q) == enough
    assert enough >= 1.0 / q and enough - 1 < 1.0 / q
