"""A confirmed failure forfeits its hotkey's unpaid work received up to it: at a
sampled share q a cheat is caught with probability q, and the hold keeps at
least 1/q of its hotkey's submissions unpaid, so a caught cheat costs at least
what one cheat earns."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from reliquary.corpus.audit_policy import AuditParams, MinerState, after_confirmed_failure
from tests.unit.test_corpus_auditor import _Records
from tests.unit.test_corpus_judge import (
    HK, HOLD, T0, SAMPLED, _Beacon, _Clock, _ids, _judge, _rec, _States,
)

OTHER = "5Other"


class _Store(_Records):
    def __init__(self, submissions, settled=()):
        super().__init__(submissions)
        self.voided = {}
        self.settled = list(settled)

    async def write_voided(self, job_id, sid, document):
        self.voided.setdefault(sid, document)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return {"settled": list(self.settled)}, "etag"


def _paid(sid, hotkey=HK, received_at=T0):
    return {"submission_id": sid, "hotkey": hotkey, "passed": True, "audited": False,
            "received_at": received_at}


def test_a_caught_cheat_voids_its_hotkeys_unpaid_passes_received_before_it():
    hit = _ids(True, 1)[0]
    unpaid, paid, later, theirs = (_ids(False, 2, start=s) for s in (100, 200, 300, 400))
    store = _Store({hit: _rec(1, received_at=T0 + 50)}, settled=paid)
    for sid in [*unpaid, *paid]:
        store.verdicts[sid] = _paid(sid, received_at=T0)
    for sid in later:
        store.verdicts[sid] = _paid(sid, received_at=T0 + 60)
    for sid in theirs:
        store.verdicts[sid] = _paid(sid, hotkey=OTHER, received_at=T0)
    auditor = _judge(store, _States({HK: SAMPLED}), _Clock(T0 + 70), beacon=_Beacon())
    asyncio.run(auditor.judge_many([hit]))
    assert store.verdicts[hit]["passed"] is False
    assert set(store.voided) == set(unpaid)
    assert all(store.voided[sid]["reason"] == "forfeited" for sid in unpaid)


def test_a_held_record_passing_after_its_hotkey_was_caught_is_forfeited_not_paid():
    hit = _ids(True, 1)[0]
    held = _ids(False, 2)
    store = _Store({hit: _rec(1, received_at=T0), **{sid: _rec(0, received_at=T0 - 5) for sid in held}})
    states = _States({HK: SAMPLED})
    auditor = _judge(store, states, _Clock(T0 + 10), beacon=_Beacon())
    asyncio.run(auditor.judge_many(held))
    assert store.verdicts == {}
    asyncio.run(auditor.judge_many([hit]))
    for sid in held:
        assert (store.verdicts[sid]["passed"], store.verdicts[sid]["reason"]) == (False, "forfeited")
    # A forfeited record is no audited pass: probation counts nothing from it.
    assert states.states[HK].audited_passed == 0


def test_work_received_after_the_failure_is_judged_on_its_own():
    hit = _ids(True, 1)[0]
    after = _ids(False, 2, start=700)
    store = _Store({hit: _rec(1, received_at=T0), **{sid: _rec(0, received_at=T0 + 20) for sid in after}})
    states = _States({HK: SAMPLED})
    auditor = _judge(store, states, _Clock(T0 + 30), beacon=_Beacon())
    asyncio.run(auditor.judge_many([hit]))
    asyncio.run(auditor.judge_many(after))
    # Suspect now: audited, and paid when they pass.
    assert all(store.verdicts[sid]["passed"] is True and store.verdicts[sid]["audited"] is True
               for sid in after)
    assert store.voided == {}


def test_a_refusal_without_penalty_forfeits_nothing(monkeypatch):
    hit = _ids(True, 1)[0]
    unpaid = _ids(False, 2, start=100)
    store = _Store({hit: _rec(0, received_at=T0 + 50)})
    for sid in unpaid:
        store.verdicts[sid] = _paid(sid)
    auditor = _judge(store, _States({HK: SAMPLED}), _Clock(T0 + 70), beacon=_Beacon(),
                     params=AuditParams(q=0.5, hold_seconds=HOLD))
    drift = {"passed": False, "reason": "draw_mismatch", "penalty": False}
    monkeypatch.setattr(auditor, "_judge_many", lambda batch: [dict(drift) for _ in batch])
    asyncio.run(auditor.judge_many([hit]))
    assert store.voided == {}


def test_the_forfeit_line_is_set_even_when_the_failure_was_already_counted():
    params = AuditParams()
    sid = "{:064x}".format(1)
    counted = after_confirmed_failure(MinerState(), params, 100.0, sid, received_at=90.0)
    # A retry whose standing already holds the failure (written before the order's own document).
    again = after_confirmed_failure(replace(MinerState(), failure_ids=counted.failure_ids,
                                            confirmed_failures=counted.confirmed_failures),
                                    params, 101.0, sid, received_at=90.0)
    assert again.forfeit_before == 90.0 and len(again.confirmed_failures) == 1


def test_at_q_1_nothing_is_forfeited():
    # Every submission audited: no cheat is ever paid, so honest work is never voided for one.
    hit = _ids(True, 1)[0]
    unpaid = _ids(False, 2, start=100)
    store = _Store({hit: _rec(1, received_at=T0 + 50)})
    for sid in unpaid:
        store.verdicts[sid] = _paid(sid)
    auditor = _judge(store, _States({HK: SAMPLED}), _Clock(T0 + 70), beacon=_Beacon(),
                     params=AuditParams(q=1.0, hold_seconds=HOLD, ban_after_failures=10))
    asyncio.run(auditor.judge_many([hit]))
    assert store.verdicts[hit]["passed"] is False and store.voided == {}


def test_a_refusal_charged_for_frequent_refusals_forfeits_like_any_charged_failure(monkeypatch):
    # The frequent-refusal charge lands before failures are counted: it forfeits too.
    hit = _ids(True, 1)[0]
    unpaid = _ids(False, 2, start=100)
    store = _Store({hit: _rec(0, received_at=T0 + 50)})
    for sid in unpaid:
        store.verdicts[sid] = _paid(sid)
    auditor = _judge(store, _States({HK: SAMPLED}), _Clock(T0 + 70), beacon=_Beacon(),
                     params=AuditParams(q=0.5, hold_seconds=HOLD))
    auditor._draw_job = object()
    auditor._draw_counts = {HK: [40, 30], "5Honest": [2000, 0]}
    auditor._draw_totals = [2040, 30]
    drift = {"passed": False, "reason": "draw_mismatch", "penalty": False}
    monkeypatch.setattr(auditor, "_judge_many", lambda batch: [dict(drift) for _ in batch])
    asyncio.run(auditor.judge_many([hit]))
    assert store.verdicts[hit]["passed"] is False
    assert set(store.voided) == set(unpaid)
