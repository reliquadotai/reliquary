"""The stake rule per operator: an order's hotkeys registered under one coldkey are staked together, every guard
that keeps a held record unpaid spans them, and a confirmed failure of one forfeits the others' unpaid work and
turns them suspect (2026-11-19: 52 of a 9B order's 64 hotkeys were two coldkeys', audited at 38-49 % for their
hotkeys' slowness while each operator as a whole was among the fastest)."""

from __future__ import annotations

import asyncio

from reliquary.corpus.audit_policy import (
    AuditParams,
    MinerState,
    after_operator_failure,
    effective_state,
)
from reliquary.validator.corpus_registration import RegisteredHotkeys
from tests.unit.test_corpus_forfeit import _Store
from tests.unit.test_corpus_judge import HK, HOLD, Q, SAMPLED, T0, _Clock, _ids, _rec, _States
from tests.unit.test_corpus_stake_rule import _Staked, _judge_at, _model, _Tokenizer, PROOF, _Beacon, _round_at

HK2 = "5Two"
PARAMS = AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10)


def _auditor(store, clock, *, operators=None, states=None):
    lookup = None if operators is None else operators.get
    return _Staked(job_id="order-gen-x", records=store, model=_model(0), tokenizer=_Tokenizer(), proof=PROOF,
                   params=PARAMS, miner_states=states or _States({HK: SAMPLED, HK2: SAMPLED}),
                   beacon=_Beacon(), round_at=_round_at, clock=clock, accept_slack_seconds=0.0,
                   operator_of=lookup)


def _two_hotkeys(operators):
    """Y of HK audited (nothing staked before it), then X of HK2 undrawn 30 s later, judged at its hold's end."""
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=500)[0]
    store = _Store({y: _rec(0, received_at=T0), x: _rec(0, received_at=T0 + 30, hotkey=HK2)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock, operators=operators)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [x])
    return auditor, store, y, x


def test_an_operators_hotkeys_are_staked_together():
    auditor, store, y, x = _two_hotkeys({HK: "cold", HK2: "cold"})
    assert (store.verdicts[y]["audited"], store.verdicts[x]["audited"]) == (True, False)
    assert store.verdicts[x]["passed"] is True
    assert auditor.audit_reasons == {"slow_stake": 1}


def test_hotkeys_of_other_coldkeys_or_unknown_ones_are_staked_alone():
    for operators in ({HK: "cold", HK2: "other"}, {}, None):
        auditor, store, _, x = _two_hotkeys(operators)
        assert store.verdicts[x]["audited"] is True


def test_a_failure_forfeits_the_operators_other_hotkeys_and_turns_them_suspect():
    y = _ids(False, 1)[0]
    held = _ids(False, 1, start=300)[0]
    cheat = _ids(True, 1, start=600)[0]
    store = _Store({y: _rec(0, received_at=T0), held: _rec(0, received_at=T0 + 30, hotkey=HK2),
                    cheat: _rec(1, received_at=T0 + 50)})
    clock = _Clock(T0)
    states = _States({HK: SAMPLED, HK2: SAMPLED})
    auditor = _auditor(store, clock, operators={HK: "cold", HK2: "cold"}, states=states)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [held])
    assert held not in store.verdicts
    _judge_at(auditor, clock, T0 + 60, [cheat])
    assert store.verdicts[cheat]["passed"] is False
    sibling = states.states[HK2]
    # Suspect and forfeiting, but no failure of its own: no probation reset, no step toward a ban.
    assert effective_state(sibling, T0 + 60, PARAMS) == "suspect"
    assert sibling.forfeit_before == T0 + 50
    assert (sibling.confirmed_failures, sibling.audited_passed) == ([], SAMPLED.audited_passed)
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [held])
    assert (store.verdicts[held]["passed"], store.verdicts[held]["reason"]) == (False, "forfeited")


def test_a_held_record_waits_for_an_undecided_record_of_its_operators_other_hotkey():
    # X of HK2 is payable at its hold's end; a drawn cheat of HK inside X's hold was read but never judged. The
    # pass decides it as X's sibling, audits it, and X is forfeited instead of paid on a stake the cheat voids.
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=300)[0]
    cheat = _ids(True, 1, start=600)[0]
    store = _Store({y: _rec(0, received_at=T0), x: _rec(0, received_at=T0 + 30, hotkey=HK2),
                    cheat: _rec(1, received_at=T0 + 50)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock, operators={HK: "cold", HK2: "cold"})
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    auditor.enqueue(cheat)
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [x])
    assert store.verdicts[cheat]["passed"] is False
    assert store.verdicts[x]["passed"] is False


def test_an_operator_failure_keeps_a_ban_and_moves_forfeits_forward_only():
    banned = MinerState(banned_until=T0 + 100, forfeit_before=T0 + 40)
    after = after_operator_failure(banned, PARAMS, T0, T0 + 10)
    assert (after.banned_until, after.suspect_until, after.forfeit_before) == (T0 + 100, None, T0 + 40)
    suspect = after_operator_failure(MinerState(suspect_until=T0 + 10 ** 9), PARAMS, T0, None)
    assert (suspect.suspect_until, suspect.forfeit_before) == (T0 + 10 ** 9, None)


def test_the_audit_backlog_leaves_out_a_restarts_records():
    from reliquary.validator.corpus_auditor import DRAW_WAIT_SECONDS

    old = _ids(True, 3)
    store = _Store({sid: _rec(0, received_at=T0) for sid in old})
    clock = _Clock(T0 + DRAW_WAIT_SECONDS + 1)
    auditor = _auditor(store, clock)
    asyncio.run(auditor._start())
    # Found pending at the start: no lease open now sent them, fewer leases would not drain them sooner.
    assert auditor.audit_backlog() == 0
    new = _ids(True, 1, start=900)[0]
    store.submissions[new] = _rec(0, received_at=T0)
    auditor.enqueue(new)
    clock.now += DRAW_WAIT_SECONDS + 1  # its draw round is out: it waits for the audit now
    assert auditor.audit_backlog() == 1


def test_the_registration_snapshot_knows_each_hotkeys_coldkey():
    async def load():
        return {HK: "cold", HK2: None}

    registered = RegisteredHotkeys(load=load)
    assert asyncio.run(registered.refresh()) is True
    assert (registered.coldkey(HK), registered.coldkey(HK2), registered.coldkey("x")) == ("cold", None, None)
    assert asyncio.run(registered.reason(HK2)) is None
