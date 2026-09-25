"""The judge: each accepted record is audited now, waits out its hold, or is
passed unaudited; a failure is re-audited before it counts, and a confirmed one
audits every held record of that hotkey. Tiny CPU models, in-memory stores."""

import asyncio
from dataclasses import replace

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState, drawn
from reliquary.validator.corpus_auditor import CorpusAuditor
from tests.unit.test_corpus_audit import _tiny
from tests.unit.test_corpus_auditor import _record, _Records, _Tokenizer
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF

RAND = "ab" * 32
Q = 0.5
HOLD = 1000.0
T0 = 10_000.0
HK = "5Hot"


class _Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class _States:
    def __init__(self, initial=None):
        self.states = dict(initial or {})
        self.updates = 0

    async def get(self, hotkey):
        return self.states.get(hotkey, MinerState())

    async def update(self, hotkey, change, attempts=5):
        self.updates += 1
        self.states[hotkey] = change(self.states.get(hotkey, MinerState()))
        return self.states[hotkey]


class _Beacon:
    def __init__(self, value=RAND):
        self.value = value
        self.calls = []

    def __call__(self, round_number):
        self.calls.append(round_number)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _ids(want_drawn, n, start=0):
    found, i = [], start
    while len(found) < n:
        sid = f"{i:064x}"
        if drawn(RAND, sid, Q) == want_drawn:
            found.append(sid)
        i += 1
    return found


_MODELS = {}
_RECORDS = {}


def _model(seed):
    if seed not in _MODELS:
        _MODELS[seed] = _tiny(seed)
    return _MODELS[seed]


def _rec(seed, received_at=T0, hotkey=HK):
    if seed not in _RECORDS:
        _RECORDS[seed] = _record(_model(seed))
    return {**_RECORDS[seed], "hotkey": hotkey, "received_at": received_at}


def _judge(records, states, clock, *, params=None, beacon=None):
    return CorpusAuditor(
        job_id="math-v1", records=records, model=_model(0), tokenizer=_Tokenizer(), proof=PROOF,
        params=params or AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10),
        miner_states=states, beacon=beacon, round_at=lambda t: int(t) // 3, clock=clock,
    )


SAMPLED = MinerState(audited_passed=100)


def test_q_one_is_v0():
    ids = _ids(False, 3)
    records = _Records({sid: _rec(0) for sid in ids})
    states = _States({HK: SAMPLED})
    beacon = _Beacon()
    auditor = _judge(records, states, _Clock(T0 + 1), params=AuditParams(), beacon=beacon)
    asyncio.run(auditor.judge_many(ids))
    assert set(records.verdicts) == set(ids)
    assert all(v["passed"] is True and v["audited"] is True and "draw" not in v
               for v in records.verdicts.values())
    assert beacon.calls == []
    assert states.states[HK].audited_passed == 103


def test_a_sampled_miners_undrawn_record_waits_then_passes_unaudited():
    ids = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in ids})
    states = _States({HK: SAMPLED})
    clock = _Clock(T0 + 10)
    auditor = _judge(records, states, clock, beacon=_Beacon())
    asyncio.run(auditor.judge_many(ids))
    assert records.verdicts == {}
    clock.now = T0 + HOLD
    asyncio.run(auditor.judge_many(ids))
    for sid in ids:
        v = records.verdicts[sid]
        assert (v["passed"], v["audited"], v["reason"]) == (True, False, None)
        assert v["worst_exp"] == 0 and v["worst_mant_mean"] == 0.0
        assert v["draw"] == {"round": int(T0 + 3) // 3, "q": Q}
    assert states.states[HK] == SAMPLED


def test_a_drawn_record_is_audited_immediately():
    hit = _ids(True, 1)[0]
    other = _ids(False, 1)[0]
    records = _Records({hit: _rec(0), other: _rec(0)})
    states = _States({HK: SAMPLED})
    auditor = _judge(records, states, _Clock(T0 + 10), beacon=_Beacon())
    asyncio.run(auditor.judge_many([hit, other]))
    assert set(records.verdicts) == {hit}
    v = records.verdicts[hit]
    assert (v["passed"], v["audited"]) == (True, True)
    assert v["draw"] == {"round": int(T0 + 3) // 3, "q": Q}
    assert states.states[HK].audited_passed == 101


def test_a_failure_is_reaudited_before_it_counts(monkeypatch):
    sid = _ids(False, 1)[0]
    records = _Records({sid: _rec(0)})
    states = _States({HK: SAMPLED})
    auditor = _judge(records, states, _Clock(T0 + 1), params=AuditParams())
    real = auditor._judge_many
    calls = []

    def flaky(batch):
        calls.append(len(batch))
        if len(calls) == 1:
            return [{"passed": False, "reason": "exp_mismatch", "worst_exp": 99,
                     "worst_mant_mean": 9.0, "worst_mant_median": 9.0}]
        return real(batch)

    monkeypatch.setattr(auditor, "_judge_many", flaky)
    asyncio.run(auditor.judge_many([sid]))
    assert calls == [1, 1]
    assert records.verdicts[sid]["passed"] is True
    state = states.states[HK]
    assert state.confirmed_failures == [] and state.suspect_until is None
    assert state.audited_passed == 101


def test_a_confirmed_failure_audits_every_held_record_of_that_hotkey():
    hit = _ids(True, 1)[0]
    held = _ids(False, 3)
    records = _Records({sid: _rec(1) for sid in [hit, *held]})
    states = _States({HK: SAMPLED})
    clock = _Clock(T0 + 10)
    auditor = _judge(records, states, clock, beacon=_Beacon())
    # Only the drawn record arrives through the queue; the held ones already waited.
    asyncio.run(auditor.judge_many(held))
    assert records.verdicts == {}
    asyncio.run(auditor.judge_many([hit]))
    assert set(records.verdicts) == {hit, *held}
    assert all(v["passed"] is False and v["audited"] is True for v in records.verdicts.values())
    state = states.states[HK]
    assert len(state.confirmed_failures) == 4 and state.suspect_until == T0 + 10 + 86400
    # Nothing left for the hold to pass unaudited.
    clock.now = T0 + HOLD + 1
    asyncio.run(auditor.judge_many([hit, *held]))
    assert all(v["audited"] is True for v in records.verdicts.values())


def test_a_suspect_miners_held_records_are_audited_not_waved_through():
    now = T0 + HOLD + 5
    held = _ids(False, 2)
    # Two recent records keep the hotkey above the slow-hotkey rate (1/q per hold).
    recent = _ids(False, 2, start=1000)
    records = _Records({**{sid: _rec(0) for sid in held},
                        **{sid: _rec(0, received_at=now - 10) for sid in recent}})
    states = _States({HK: replace(SAMPLED, suspect_until=now + 100)})
    auditor = _judge(records, states, _Clock(now), beacon=_Beacon())
    asyncio.run(auditor.judge_many([*held, *recent]))
    assert set(records.verdicts) == {*held, *recent}
    assert all(v["passed"] is True and v["audited"] is True for v in records.verdicts.values())


def test_a_ban_voids_pending_records_and_ends_into_probation():
    held = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in held})
    states = _States({HK: MinerState(audited_passed=150, banned_until=T0 + 500)})
    clock = _Clock(T0 + 10)
    auditor = _judge(records, states, clock, beacon=_Beacon())
    asyncio.run(auditor.judge_many(held))
    for sid in held:
        assert records.verdicts[sid] == {**records.verdicts[sid], "passed": False,
                                         "audited": False, "reason": "banned"}
    # The ban ends: a new record is judged in probation, from a fresh count.
    late = _ids(False, 1, start=1000)[0]
    records.submissions[late] = _rec(0, received_at=T0 + 600)
    clock.now = T0 + 601
    asyncio.run(auditor.judge_many([late]))
    assert records.verdicts[late]["audited"] is True and records.verdicts[late]["passed"] is True
    state = states.states[HK]
    assert state.banned_until is None and state.audited_passed == 1


@pytest.mark.parametrize("beacon", [None, _Beacon(None), _Beacon(ConnectionError("drand down"))])
def test_no_beacon_audits(beacon):
    ids = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in ids})
    states = _States({HK: SAMPLED})
    # Mid-hold: with a beacon saying "not drawn" these records would wait.
    auditor = _judge(records, states, _Clock(T0 + 10), beacon=beacon)
    asyncio.run(auditor.judge_many(ids))
    assert set(records.verdicts) == set(ids)
    assert all(v["audited"] is True and "draw" not in v for v in records.verdicts.values())


def test_an_unaudited_pass_is_never_written_before_the_hold_ends():
    ids = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in ids})
    states = _States({HK: SAMPLED})
    clock = _Clock(T0 + HOLD - 1e-6)
    auditor = _judge(records, states, clock, beacon=_Beacon())
    asyncio.run(auditor.judge_many(ids))
    assert records.verdicts == {}
    # A record with no received_at counts from when the judge first saw it.
    old = _ids(False, 1, start=2000)[0]
    records.submissions[old] = {k: v for k, v in _rec(0).items() if k != "received_at"}
    asyncio.run(auditor.judge_many([old, *ids]))
    assert old not in records.verdicts
    clock.now = T0 + HOLD
    asyncio.run(auditor.judge_many([old, *ids]))
    assert set(records.verdicts) == set(ids)


def test_a_restart_mid_hold_neither_loses_nor_doubles_a_verdict():
    hit = _ids(True, 1)[0]
    held = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in [hit, *held]})
    states = _States({HK: SAMPLED})
    clock = _Clock(T0 + 10)
    first = _judge(records, states, clock, beacon=_Beacon())
    asyncio.run(first.judge_many([hit, *held]))
    audited = dict(records.verdicts)
    assert set(audited) == {hit}

    # The hold ends; [now - hold, now] still holds all three, so no slow-hotkey audit.
    clock.now = T0 + HOLD
    second = _judge(records, states, clock, beacon=_Beacon())

    async def _again():
        await second.judge_many(await second.pending_ids())
        await second.judge_many([hit, *held])  # a stale queue entry changes nothing

    asyncio.run(_again())
    assert set(records.verdicts) == {hit, *held}
    assert records.verdicts[hit] == audited[hit]
    assert all(records.verdicts[sid]["audited"] is False for sid in held)
    assert states.states[HK].audited_passed == 101
