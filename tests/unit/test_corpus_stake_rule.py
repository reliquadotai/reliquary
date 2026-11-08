"""The slow-hotkey rule on what a failure forfeits (`stake_at_receipt`, the order control's): a sampled record
is paid unaudited only if a confirmed failure of it would forfeit at least (1/q - 1) times its tokens of its
hotkey's work, counted once when its draw is known. A hotkey that stopped submitting no longer has its held
records audited at their hold's end, and what is counted is forfeited for real."""

from __future__ import annotations

import asyncio
import math

from reliquary.corpus.audit_policy import AuditParams
from reliquary.validator.corpus_auditor import CorpusAuditor
from reliquary.validator.corpus_periods import PERIOD_EPOCH, PERIOD_SECONDS
from tests.unit.test_corpus_auditor import _Tokenizer
from tests.unit.test_corpus_forfeit import _Store
from tests.unit.test_corpus_judge import (
    HK, HOLD, PROOF, Q, SAMPLED, T0, _Beacon, _Clock, _ids, _model, _rec, _round_at, _States,
)


class _Staked(CorpusAuditor):
    stake_at_receipt = True


def _auditor(store, clock, *, staked=True, states=None):
    cls = _Staked if staked else CorpusAuditor
    return cls(job_id="order-gen-x", records=store, model=_model(0), tokenizer=_Tokenizer(), proof=PROOF,
               params=AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10),
               miner_states=states or _States({HK: SAMPLED}), beacon=_Beacon(), round_at=_round_at,
               clock=clock, accept_slack_seconds=0.0)


def _judge_at(auditor, clock, now, ids):
    clock.now = now
    asyncio.run(auditor.judge_many(ids))


def _tail(staked):
    """Y audited (its hotkey's first sampled record holds no stake), then X undrawn 30 s later; the hotkey
    sends nothing more. X is judged again at its hold's end."""
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=500)[0]
    store = _Store({y: _rec(0, received_at=T0), x: _rec(0, received_at=T0 + 30)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock, staked=staked)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [x])
    return store.verdicts[y], store.verdicts[x]


def test_a_held_record_of_a_hotkey_that_stopped_is_paid_unaudited():
    y, x = _tail(staked=True)
    # Y had nothing before it: audited. X had Y's audited pass at stake: paid after its hold, unaudited.
    assert (y["audited"], y["passed"]) == (True, True)
    assert (x["audited"], x["passed"]) == (False, True)


def test_the_rule_on_arrivals_up_to_now_audits_that_record():
    # The corpus rule counts the hotkey's arrivals in the hold up to now: none after X, so X is audited.
    _, x = _tail(staked=False)
    assert x["audited"] is True


def test_a_stake_of_fewer_tokens_than_the_record_does_not_cover_it():
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=500)[0]
    big = {**_rec(0, received_at=T0 + 30), "token_count": 3 * _rec(0)["token_count"]}
    store = _Store({y: _rec(0, received_at=T0), x: big})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    # Y's tokens are a third of X's: at q = 0.5 a failure of X would forfeit less than X earns.
    assert auditor._stake_of(x, T0 + 40) < 1.0 / Q
    assert store.verdicts[x]["audited"] is True


def test_an_audited_pass_of_an_earlier_period_does_not_count_a_held_record_does():
    boundary = PERIOD_EPOCH + math.ceil((T0 + 2 * HOLD - PERIOD_EPOCH) / PERIOD_SECONDS) * PERIOD_SECONDS
    audited, held = _ids(False, 2)
    x = _ids(False, 1, start=500)[0]
    store = _Store({audited: _rec(0, received_at=boundary - 200), held: _rec(0, received_at=boundary - 100),
                    x: _rec(0, received_at=boundary + 50)})
    clock = _Clock(boundary)
    auditor = _auditor(store, clock)
    _judge_at(auditor, clock, boundary - 180, [audited])
    _judge_at(auditor, clock, boundary - 80, [held])
    assert held not in store.verdicts  # held: the audited pass of its own period covers it
    _judge_at(auditor, clock, boundary + 70, [x])
    # Only the held record counts for X: the audited pass settles with its own period, maybe before X.
    entries = auditor._forfeitable_at[HK]
    assert [(t - boundary, unaudited) for t, _, unaudited in entries] == [(-200, False), (-100, True),
                                                                          (50, True)]
    assert auditor._stake_of(x, boundary + 70) == 2.0
    assert x not in store.verdicts  # waits for its hold, unaudited


def test_the_stake_counted_is_forfeited_when_a_drawn_record_fails():
    y = _ids(False, 1)[0]
    held = _ids(False, 1, start=300)[0]
    cheat = _ids(True, 1, start=600)[0]
    store = _Store({y: _rec(0, received_at=T0), held: _rec(0, received_at=T0 + 30),
                    cheat: _rec(1, received_at=T0 + 50)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [held])
    assert held not in store.verdicts
    # The cheat is drawn: audited, fails twice, and forfeits what its stake counted.
    _judge_at(auditor, clock, T0 + 60, [cheat])
    assert store.verdicts[cheat]["passed"] is False
    assert y in store.voided and store.voided[y]["reason"] == "forfeited"
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [held])
    assert (store.verdicts[held]["passed"], store.verdicts[held]["reason"]) == (False, "forfeited")


def test_the_stake_is_counted_once():
    y = _ids(False, 1)[0]
    sid = _ids(False, 1, start=500)[0]
    store = _Store({y: _rec(0, received_at=T0), sid: _rec(0, received_at=T0 + 30)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [sid])
    first = auditor._stake[sid]
    # Much later, with everything before it gone from memory, the record keeps its first count.
    auditor._forfeitable_at[HK].clear()
    assert auditor._stake_of(sid, T0 + 30 + HOLD + 1) == first == 2.0
