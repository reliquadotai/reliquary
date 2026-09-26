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


def _round_at(t):
    # Contract: the first drand round published strictly after t (3 s chain, round r at 3r).
    return int(t // 3) + 1


def _published(round_number):
    return 3.0 * round_number


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


def _judge(records, states, clock, *, params=None, beacon=None, round_at=_round_at):
    return CorpusAuditor(
        job_id="math-v1", records=records, model=_model(0), tokenizer=_Tokenizer(), proof=PROOF,
        params=params or AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10),
        miner_states=states, beacon=beacon, round_at=round_at, clock=clock,
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
        assert v["draw"] == {"round": _round_at(T0) + 1, "q": Q, "drawn": False}
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
    assert v["draw"] == {"round": _round_at(T0) + 1, "q": Q, "drawn": True}
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


def test_a_raising_round_at_audits():
    """The drand chain's genesis/period may not be resolved yet: `round_at`
    then raises instead of returning an int, and the auditor treats that
    exactly like a missing beacon -- audit, never guess a round from
    nothing (fix round 1, finding 2)."""
    ids = _ids(False, 2)
    records = _Records({sid: _rec(0) for sid in ids})
    states = _States({HK: SAMPLED})
    beacon = _Beacon()

    def _unresolved(t):
        raise RuntimeError("drand chain genesis/period not resolved yet")

    auditor = _judge(records, states, _Clock(T0 + 10), beacon=beacon, round_at=_unresolved)
    asyncio.run(auditor.judge_many(ids))
    assert set(records.verdicts) == set(ids)
    assert all(v["audited"] is True and "draw" not in v for v in records.verdicts.values())
    assert beacon.calls == []


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


# --- Fix round 1: escalation before the verdict, listing cost, draw round, same-pass guard ---


class _CrashingStates(_States):
    """The first update fails: before applying it, or after (a crash between the
    committed state and the verdict write)."""

    def __init__(self, initial, applied):
        super().__init__(initial)
        self.applied = applied
        self.crashed = False

    async def update(self, hotkey, change, attempts=5):
        if not self.crashed:
            self.crashed = True
            if self.applied:
                await super().update(hotkey, change, attempts)
            raise ConnectionError("store down")
        return await super().update(hotkey, change, attempts)


@pytest.mark.parametrize("applied", [False, True])
def test_a_crash_around_the_escalation_still_ends_suspect_counted_once(applied):
    sid = _ids(False, 1)[0]
    records = _Records({sid: _rec(1)})
    states = _CrashingStates({HK: SAMPLED}, applied)
    clock = _Clock(T0 + 10)
    params = AuditParams(ban_after_failures=10)
    with pytest.raises(ConnectionError):
        asyncio.run(_judge(records, states, clock, params=params).judge_many([sid]))

    restarted = _judge(records, states, clock, params=params)

    async def _again():
        await restarted.judge_many(await restarted.pending_ids())

    asyncio.run(_again())
    v = records.verdicts[sid]
    assert (v["passed"], v["audited"]) == (False, True)
    state = states.states[HK]
    assert state.suspect_until == T0 + 10 + params.suspect_seconds
    assert len(state.confirmed_failures) == 1 and state.failure_ids == [sid]


class _CountingRecords(_Records):
    def __init__(self, submissions):
        super().__init__(submissions)
        self.listings = 0
        self.reads = {}

    async def list_submission_ids(self, job_id):
        self.listings += 1
        return await super().list_submission_ids(job_id)

    async def read_submission(self, job_id, sid):
        self.reads[sid] = self.reads.get(sid, 0) + 1
        return await super().read_submission(job_id, sid)


def test_judging_many_batches_lists_the_job_at_most_once():
    ids = _ids(False, 6)
    records = _CountingRecords({sid: _rec(0) for sid in ids})
    auditor = _judge(records, _States({HK: SAMPLED}), _Clock(T0 + 10), beacon=_Beacon())
    for batch in (ids[:2], ids[2:4], ids[4:]):
        asyncio.run(auditor.judge_many(batch))
    assert records.verdicts == {}
    assert records.listings <= 1
    assert records.reads == {sid: 1 for sid in ids}


def test_a_restart_reads_only_pending_records():
    judged = _ids(False, 3)
    pending = _ids(False, 2, start=500)
    records = _CountingRecords({sid: _rec(0) for sid in [*judged, *pending]})
    for sid in judged:
        records.verdicts[sid] = {"passed": True}
    auditor = _judge(records, _States({HK: SAMPLED}), _Clock(T0 + 10), beacon=_Beacon())
    asyncio.run(auditor.judge_many(pending))
    assert set(records.reads) == set(pending)


@pytest.mark.parametrize("offset", [0.0, 0.5, 2.9])
def test_the_draw_round_is_one_round_past_the_first_published_after_receipt(offset):
    # One extra round: a validator clock up to one period behind real time
    # still draws on a round published after the miner really sent.
    received = _published(3333) + offset
    ids = _ids(False, 2)
    records = _Records({sid: _rec(0, received_at=received) for sid in ids})
    beacon = _Beacon()
    clock = _Clock(_published(3334) + 2.5)
    auditor = _judge(records, _States({HK: SAMPLED}), clock, beacon=beacon)
    asyncio.run(auditor.judge_many(ids))
    # Round 3335 is not out yet: nothing is fetched, nothing decided.
    assert beacon.calls == [] and records.verdicts == {}
    clock.now = _published(3335) + 2.5
    asyncio.run(auditor.judge_many(ids))
    assert beacon.calls == [3335]
    clock.now = received + HOLD
    asyncio.run(auditor.judge_many(ids))
    for sid in ids:
        draw = records.verdicts[sid]["draw"]
        assert draw["round"] == 3335 and _published(draw["round"]) > received + 3.0


def test_a_failed_beacon_round_is_not_refetched_within_the_negative_cache_window():
    """A round the beacon just failed on is not refetched for
    NEGATIVE_BEACON_CACHE_SECONDS: every sampled submission whose draw lands
    on that round would otherwise repeat the same failing network call
    (fix round 1, finding 4)."""
    first = _ids(False, 2)
    records = _Records({sid: _rec(0, received_at=_published(41)) for sid in first})
    beacon = _Beacon(ConnectionError("drand down"))
    clock = _Clock(_published(43) + 2.5)
    auditor = _judge(records, _States({HK: SAMPLED}), clock, beacon=beacon)

    asyncio.run(auditor.judge_many(first))
    assert beacon.calls == [43]
    assert all(v["audited"] is True for v in records.verdicts.values())

    # A second submission drawing the same round, well inside the negative
    # cache window: no second fetch, still fully audited (fail safe).
    second = _ids(False, 2, start=2000)
    records.submissions.update({sid: _rec(0, received_at=_published(41)) for sid in second})
    clock.now += 5.0
    asyncio.run(auditor.judge_many(second))
    assert beacon.calls == [43]
    assert all(v["audited"] is True for v in records.verdicts.values())

    # Past the negative-cache window: retried.
    third = _ids(False, 2, start=4000)
    records.submissions.update({sid: _rec(0, received_at=_published(41)) for sid in third})
    clock.now += 30.0
    asyncio.run(auditor.judge_many(third))
    assert beacon.calls == [43, 43]


def test_a_held_record_is_audited_when_a_failure_lands_in_the_same_pass():
    held = _ids(False, 1)[0]
    hit = _ids(True, 1)[0]
    now = T0 + HOLD
    records = _Records({held: _rec(1), hit: _rec(1, received_at=now - 10)})
    states = _States({HK: SAMPLED})
    auditor = _judge(records, states, _Clock(now), beacon=_Beacon())
    asyncio.run(auditor.judge_many([held, hit]))
    assert records.verdicts[hit]["passed"] is False
    v = records.verdicts[held]
    assert (v["passed"], v["audited"]) == (False, True)
