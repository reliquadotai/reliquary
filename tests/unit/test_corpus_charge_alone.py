"""A forced-draw refusal is charged only if it is charged in a batch of its own: the same tokens packed with other
rows can move a Gumbel gap over the line."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reliquary.corpus.audit_policy import effective_state
from tests.unit.test_corpus_forfeit import OTHER, _Store
from tests.unit.test_corpus_judge import HK, SAMPLED, T0, _Beacon, _Clock, _ids, _judge, _rec, _States

CHARGED = {"passed": False, "reason": "draw_mismatch", "penalty": True, "forced_draw": {"far_misses": 2}}
UNCHARGED = {"passed": False, "reason": "draw_mismatch", "penalty": False, "forced_draw": {"far_misses": 1}}
PASSED = {"passed": True, "reason": None, "worst_mant_mean": 0.0}


def _run(alone, *, hotkeys=(HK, OTHER)):
    """Audit one record per hotkey: refused and charged while packed, then ``alone[hotkey]`` in a batch of its own."""
    ids = _ids(True, len(hotkeys))
    records = [_rec(k + 1, hotkey=hotkey) for k, hotkey in enumerate(hotkeys)]
    store = _Store(dict(zip(ids, records)))
    states = _States({hotkey: SAMPLED for hotkey in hotkeys})
    clock = _Clock(T0 + 70)
    auditor = _judge(store, states, clock, beacon=_Beacon())
    auditor._draw_job = SimpleNamespace(judge=None)
    calls = []

    async def outcomes(batch, *, local=False):
        calls.append((len(batch), local))
        if len(batch) == 1 and len(hotkeys) > 1:
            return [alone[batch[0]["hotkey"]]]
        return [dict(CHARGED) for _ in batch]

    auditor._audit_outcomes = outcomes
    _, failed = asyncio.run(auditor._audit_records(ids, records, {}))
    return SimpleNamespace(ids=ids, store=store, states=states, clock=clock, calls=calls, failed=failed)


def test_a_charge_that_does_not_hold_alone_is_refused_uncharged():
    run = _run({HK: UNCHARGED, OTHER: CHARGED})
    assert run.calls == [(2, False), (2, True), (1, True), (1, True)]
    assert run.failed == {OTHER}
    assert run.store.verdicts[run.ids[0]]["penalty"] is False
    assert run.store.verdicts[run.ids[1]]["penalty"] is True
    assert effective_state(run.states.states[HK], run.clock(), _params()) != "suspect"
    assert effective_state(run.states.states[OTHER], run.clock(), _params()) == "suspect"


def test_a_sample_that_passes_alone_is_not_refused():
    run = _run({HK: PASSED, OTHER: CHARGED})
    assert run.failed == {OTHER}
    assert run.store.verdicts[run.ids[0]]["passed"] is True


def test_a_validator_error_alone_charges_nothing_and_leaves_the_record_pending():
    run = _run({HK: "the card went away", OTHER: CHARGED})
    assert run.failed == {OTHER}
    assert run.ids[0] not in run.store.verdicts


def test_a_failure_already_alone_is_not_audited_a_third_time():
    run = _run({}, hotkeys=(HK,))
    assert run.calls == [(1, False), (1, True)]
    assert run.failed == {HK}


def _params():
    from reliquary.corpus.audit_policy import AuditParams
    from tests.unit.test_corpus_judge import HOLD, Q

    return AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10)
