"""The judge before the scheduler (main at dfa575d3, frozen in
`_main_corpus_auditor`) and the judge now write the same verdicts and voids:
both re-deciding every pending record every TICK, and the new one driven by
its own scheduler. Faults on the way: a ban and its end, an unreadable
record, a failed verdict write, a quarantined executor, a restart."""

from __future__ import annotations

import asyncio
import contextlib
import random

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.validator import corpus_auditor as new_module
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import _main_corpus_auditor as main_module
from tests.unit import corpus_judge_sim as sim

TICK = 0.25
# Holds end on a TICK; arrivals land .1 s past a second (see the schedule tests).
SLACK = 29.9
END = 2600.0
RESTART_AT = 1500.0
QUARANTINE_AT = 1000.0
UNREADABLE = (700.0, 800.0)
_OK = {"passed": True, "reason": None, "worst_exp": 0, "worst_mant_mean": 0.01,
       "worst_mant_median": 0.01}
_BAD = {**_OK, "passed": False, "reason": "exp_mismatch"}


def _params(faults):
    return AuditParams(q=0.25, probation_submissions=5, hold_seconds=300.0,
                       suspect_seconds=86400.0, ban_seconds=600.0,
                       ban_after_failures=2 if "ban" in faults else 1000)


def _scenario(seed):
    rng = random.Random(seed)
    arrivals = []
    for hotkey, start, stop in [("5Steady0", 0, 2000), ("5Steady1", 3, 2000),
                                ("5Steady2", 5, 2000), ("5Stops", 2, 400),
                                ("5New", 20, 2000), ("5Lazy", 1, 850)]:
        t = start
        while t < stop:
            arrivals.append((t + 0.1, hotkey))
            t += rng.randint(4, 15)
    # A sampled cheater: a burst caught by a draw, records sent while banned
    # (or suspect), and records sent long after the ban ended.
    arrivals += [(600 + i + 0.1, "5Cheat") for i in range(12)]
    arrivals += [(900 + 5 * i + 0.1, "5Cheat") for i in range(6)]
    arrivals += [(1800 + 5 * i + 0.1, "5Cheat") for i in range(6)]
    arrivals.sort()
    rng2 = random.Random(seed + 10)
    ids = ["%064x" % rng2.getrandbits(256) for _ in arrivals]
    return arrivals, ids


class _Store(sim.Store):
    """Zero latency; one record unreadable for a while, one write failing twice."""

    def __init__(self, *, unreadable=None, failing=None):
        super().__init__(latency=(0.0, 0.0))
        self.unreadable, self.failing = unreadable, failing
        self.fails_left = 2
        self.unreadable_hits = 0
        self.voided: dict[str, dict] = {}

    async def read_submission(self, job_id, sid):
        start, end = UNREADABLE
        if sid == self.unreadable and start <= sim.virtual_clock() - self.t0 < end:
            self.unreadable_hits += 1
            raise OSError("read reset")
        return await super().read_submission(job_id, sid)

    async def write_verdict(self, job_id, sid, verdict):
        if sid == self.failing and self.fails_left:
            self.fails_left -= 1
            raise ConnectionError("PUT reset")
        return await super().write_verdict(job_id, sid, verdict)

    async def read_settlement(self, job_id):
        return {}, None

    async def write_voided(self, job_id, sid, document):
        if sid in self.voided:
            return False
        self.voided[sid] = document
        return True


def _forward(cheaters):
    """The GPU, and an executor that passes every "5Lazy" record it scores."""

    async def forward(records, *, local=False):
        out = []
        for r in records:
            if r["hotkey"] == "5Lazy" and not local:
                out.append({**_OK, "scored_by": ["ex1"]})
            elif r["hotkey"] in cheaters or r["hotkey"] == "5Lazy":
                out.append(dict(_BAD))
            else:
                out.append(dict(_OK))
        return out

    return forward


def _auditor(module, store, faults):
    clock = sim._Clock()
    auditor = module.CorpusAuditor(
        job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
        params=_params(faults), miner_states=MinerStates(store, "math-v1", clock=clock),
        beacon=sim.Beacon(latency=0.0), round_at=sim.round_at, clock=clock,
        accept_slack_seconds=SLACK, rescan_every_seconds=60.0)
    auditor._forward = _forward({"5Cheat"})
    return auditor


async def _judge(module, mode, faults, seed):
    arrivals, ids = _scenario(seed)
    hotkey_of = dict(zip(ids, (h for _, h in arrivals)))
    steady = [sid for (t, h), sid in zip(arrivals, ids) if h == "5Steady1" and t > 400]
    # Arrives while its record cannot be read: every unaudited pass waits.
    late = next(sid for (t, h), sid in zip(arrivals, ids)
                if h == "5Steady1" and UNREADABLE[0] < t < UNREADABLE[1] - 20)
    store = _Store(unreadable=late if "unreadable" in faults else None,
                   failing=steady[-1] if "write" in faults else None)
    store.t0 = t0 = sim.virtual_clock()
    for hotkey in ("5Steady0", "5Steady1", "5Steady2", "5Stops", "5Cheat", "5Lazy"):
        store.miners[hotkey] = MinerState(audited_passed=50).to_dict()
    auditor = _auditor(module, store, faults)
    landed = []
    events = {"restart": "restart" in faults, "quarantine": "quarantine" in faults}

    async def arrive():
        for (offset, hotkey), sid in zip(arrivals, ids):
            await asyncio.sleep(t0 + offset - sim.virtual_clock())
            store.submissions[sid] = {"hotkey": hotkey, "received_at": sim.virtual_clock(),
                                      "token_count": 100, "completions": [{"tokens": [1]}]}
            landed.append(sid)
            if mode == "run":
                auditor.enqueue(sid)

    feeder = asyncio.ensure_future(arrive())
    runner = asyncio.ensure_future(auditor.run()) if mode == "run" else None
    tick = 0
    while sim.virtual_clock() < t0 + END:
        now = sim.virtual_clock() - t0
        if events["quarantine"] and now >= QUARANTINE_AT:
            events["quarantine"] = False
            await auditor.reaudit_executor("ex1")
        if events["restart"] and now >= RESTART_AT:
            events["restart"] = False
            if runner is not None:
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner
            auditor = _auditor(module, store, faults)
            if mode == "run":
                runner = asyncio.ensure_future(auditor.run())
        if mode == "poll":
            with contextlib.suppress(OSError, ConnectionError):
                await auditor.judge_many([sid for sid in landed if sid not in store.verdicts])
        tick += 1
        await asyncio.sleep(max(0.0, t0 + tick * TICK - sim.virtual_clock()))
    feeder.cancel()
    if runner is not None:
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
    return store, hotkey_of


def _run(module, mode, faults, seed):
    loop = sim.VirtualTimeLoop()
    try:
        return loop.run_until_complete(_judge(module, mode, faults, seed))
    finally:
        loop.close()


def _key(v):
    return (v["hotkey"], v["passed"], v["audited"], v["reason"], v.get("draw"))


def _outcome(store):
    return ({sid: _key(v) for sid, v in store.verdicts.items()},
            {sid: (d["hotkey"], d["reason"]) for sid, d in store.voided.items()})


ALL = ("ban", "unreadable", "write", "quarantine", "restart")


@pytest.mark.parametrize("seed", [7, 3])
def test_main_and_the_scheduled_judge_write_the_same_verdicts(seed):
    faults = ALL
    main, hotkey_of = _run(main_module, "poll", faults, seed)
    polled, _ = _run(new_module, "poll", faults, seed)
    scheduled, _ = _run(new_module, "run", faults, seed)
    assert set(main.verdicts) == set(hotkey_of)
    assert _outcome(polled) == _outcome(main)
    assert _outcome(scheduled) == _outcome(main)
    verdicts = main.verdicts.values()
    assert main.verdicts[next(iter(main.voided))]["passed"] is True  # voided after paying
    # Each fault happened: a ban voided records, then ended into probation
    # audits; the executor's passes were voided; the failed write and the
    # unreadable record were judged in the end.
    cheat = [v for v in verdicts if v["hotkey"] == "5Cheat"]
    assert any(v["reason"] == "banned" for v in cheat)
    first = min(r["received_at"] for r in main.submissions.values())
    assert any(v["audited"] and v["reason"] == "exp_mismatch"
               for sid, v in main.verdicts.items()
               if hotkey_of[sid] == "5Cheat" and main.submissions[sid]["received_at"] > first + 1700)
    assert main.voided and all(hotkey_of[sid] == "5Lazy" for sid in main.voided)
    assert main.fails_left == 0 and scheduled.fails_left == 0
    assert main.unreadable_hits and scheduled.unreadable_hits
