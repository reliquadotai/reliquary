"""The slow-hotkey rule on what a failure forfeits (`stake_at_receipt`, the order control's): a sampled record
is paid unaudited only if a confirmed failure of it would forfeit at least (1/q - 1) times its tokens of its
hotkey's work, counted once when its draw is known. A hotkey that stopped submitting no longer has its held
records audited at their hold's end, and what is counted is forfeited for real."""

from __future__ import annotations

import asyncio
import math

import pytest

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


def test_only_the_records_own_settlement_period_counts():
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
    # The earlier period may settle, at another rate per token, before X fails: none of it counts.
    assert auditor._stake_of(x, boundary + 70) == 1.0
    assert store.verdicts[x]["audited"] is True


def test_a_prompt_counts_at_its_price():
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=500)[0]
    # Same output; X carries a long prompt the order pays for.
    store = _Store({y: {**_rec(0, received_at=T0), "prompt_token_count": 10},
                    x: {**_rec(0, received_at=T0 + 30), "prompt_token_count": 100_000}})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    auditor.stake_input_weight = 0.5
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    assert auditor._stake_of(x, T0 + 40) < 1.0 / Q
    assert store.verdicts[x]["audited"] is True


def test_a_short_draw_sample_and_an_empty_record_are_not_paid_on_a_stake():
    y = _ids(False, 1)[0]
    store = _Store({y: _rec(0, received_at=T0)})
    clock = _Clock(T0 + 20)
    auditor = _auditor(store, clock)
    asyncio.run(auditor.judge_many([y]))
    # A forced-draw refusal is charged per completion, from EVAL_CHARGE_MIN_POSITIONS positions: a record
    # with a shorter completion (whatever its total), or whose completions were never read, forfeits nothing.
    auditor._draw_job = object()
    long_ = {"tokens": [1] * 64, "text": ""}
    auditor._remember("a" * 64, {**_rec(0), "completions": [long_, long_]})
    auditor._remember("b" * 64, {**_rec(0), "completions": [{"tokens": [1] * 63, "text": ""}, long_]})
    auditor._remember("c" * 64, {"hotkey": HK, "received_at": T0, "token_count": 500})
    assert [auditor._stake_applies(sid * 64) for sid in "abc"] == [True, False, False]
    hotkey, received_at, _ = auditor._meta[y]
    auditor._meta[y] = (hotkey, received_at, 0)
    auditor._stake.clear()
    assert auditor._stake_of(y, T0 + 20) == 0.0


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


def test_audits_pending_leaves_out_held_records_their_stake_covers():
    y = _ids(False, 1)[0]
    x = _ids(False, 1, start=500)[0]
    late = _ids(False, 1, start=700)[0]
    store = _Store({y: _rec(0, received_at=T0), x: _rec(0, received_at=T0 + 30),
                    late: _rec(0, received_at=T0 + 50)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    assert auditor.audits_pending() is None  # before the seed: unknown
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [x])
    auditor.enqueue(late)
    # X waits out its hold, covered: its hold's end only pays it. `late` is not decided yet.
    assert auditor.audits_pending() == 1 and auditor.audits_idle_since is None
    _judge_at(auditor, clock, T0 + 60, [late])
    clock.now = T0 + 61
    assert auditor.audits_pending() == 0 and auditor.audits_idle_since == T0 + 61
    clock.now = T0 + 90
    assert auditor.audits_pending() == 0 and auditor.audits_idle_since == T0 + 61
    # A record whose decision may still turn into an audit (the corpus rule) counts again.
    auditor.stake_at_receipt = False
    assert auditor.audits_pending() == 2 and auditor.audits_idle_since is None


class _Flaky(_Store):
    def __init__(self, *args, fail=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail, self.reads = fail, 0

    async def read_submission(self, job_id, sid):
        self.reads += 1
        if self.fail:
            self.fail -= 1
            raise OSError("transient")
        return await super().read_submission(job_id, sid)


_LONG = {"tokens": [1] * 64, "text": ""}
_META = {"hotkey": HK, "received_at": T0, "token_count": 500}


def _restarted(store, states=None):
    """An auditor of a draw job that knows a record from its metadata only, as `_seed` leaves it."""
    clock = _Clock(T0 + 20)
    auditor = _auditor(store, clock, states=states)
    auditor._draw_job = object()
    return auditor


def test_a_record_seeded_from_its_metadata_is_read_in_full_once_for_the_stake_rule():
    # After a restart, `_seed` reads only each pending record's tail: no completions, so its shortest one is
    # unknown and the stake rule would fall back to the arrivals rule (2026-11-09: an order audited at 99 %).
    sid = _ids(False, 1)[0]
    store = _Flaky({sid: {**_rec(0, received_at=T0), "completions": [_LONG, _LONG]}}, fail=1)
    auditor = _restarted(store)
    auditor._remember(sid, _META)
    assert auditor._stake_applies(sid) is False
    states = {HK: SAMPLED}
    asyncio.run(auditor._learn_shortest([sid], states, T0 + 20))
    assert auditor._stake_applies(sid) is False          # the read failed: tried again later
    asyncio.run(auditor._learn_shortest([sid], states, T0 + 20))
    assert auditor._stake_applies(sid) is True
    # Read in full once, and not kept: a record whose full read finds no completion is not read every pass.
    auditor._shortest.pop(sid)
    asyncio.run(auditor._learn_shortest([sid], states, T0 + 20))
    assert store.reads == 2


def test_only_records_the_stake_rule_may_pay_are_read():
    sid = _ids(False, 1)[0]
    store = _Flaky({sid: {**_rec(0, received_at=T0), "completions": [_LONG]}})
    from reliquary.corpus.audit_policy import MinerState

    probation = MinerState(audited_passed=0)
    auditor = _restarted(store)
    auditor._remember(sid, _META)
    asyncio.run(auditor._learn_shortest([sid], {HK: probation}, T0 + 20))  # audited anyway
    corpus = _auditor(store, _Clock(T0 + 20))                               # not a draw job
    corpus._remember(sid, _META)
    asyncio.run(corpus._learn_shortest([sid], {HK: SAMPLED}, T0 + 20))
    unstaked = _auditor(store, _Clock(T0 + 20), staked=False)
    unstaked._draw_job = object()
    unstaked._remember(sid, _META)
    asyncio.run(unstaked._learn_shortest([sid], {HK: SAMPLED}, T0 + 20))
    assert store.reads == 0


def test_a_pass_learns_the_length_of_its_records_before_deciding():
    first = _ids(False, 1)[0]
    store = _Store({first: _rec(0, received_at=T0)})
    auditor = _restarted(store)
    auditor._remember(first, _META)
    seen = []

    class _Stop(Exception):
        pass

    async def learn(ids, states, now):
        # Before any decision: the pass's own records are read for their length first.
        seen.extend(ids)
        raise _Stop

    auditor._learn_shortest = learn
    auditor._seeded = True
    with pytest.raises(_Stop):
        asyncio.run(auditor._judge_once([first]))
    assert seen == [first]


def test_a_short_draw_sample_is_audited_instead_of_paid_unaudited(monkeypatch):
    # Its refusal is never charged (under EVAL_CHARGE_MIN_POSITIONS positions), so a two-token sample costs a
    # cheat nothing: it is audited rather than paid at its hold's end.
    short = _ids(False, 1, start=500)[0]
    monkeypatch.setattr(_Staked, "_short_draw", lambda self, sid: sid == short)
    y = _ids(False, 1)[0]
    store = _Store({y: _rec(0, received_at=T0), short: _rec(0, received_at=T0 + 30)})
    clock = _Clock(T0)
    auditor = _auditor(store, clock)
    _judge_at(auditor, clock, T0 + 20, [y])
    _judge_at(auditor, clock, T0 + 40, [short])
    _judge_at(auditor, clock, T0 + 30 + HOLD + 1, [short])
    assert store.verdicts[short]["audited"] is True


def test_only_a_forced_draw_record_with_a_short_completion_is_short():
    store = _Store({})
    auditor = _auditor(store, _Clock(T0), staked=False)
    long_ = {"tokens": [1] * 64, "text": ""}
    short = {"tokens": [1] * 2, "text": ""}
    auditor._remember("a" * 64, {**_rec(0), "completions": [short, long_]})
    assert not auditor._short_draw("a" * 64)           # not a forced-draw job
    auditor._draw_job = object()
    auditor._remember("b" * 64, {**_rec(0), "completions": [short, long_]})
    auditor._remember("c" * 64, {**_rec(0), "completions": [long_, long_]})
    auditor._remember("d" * 64, {"hotkey": HK, "received_at": T0, "token_count": 500})
    # Read even without the stake rule; a record never read in full is left to the usual rules.
    assert [auditor._short_draw(sid * 64) for sid in "bcd"] == [True, False, False]


def test_the_audit_backlog_leaves_out_records_still_waiting_for_their_draw():
    from reliquary.validator.corpus_auditor import DRAW_WAIT_SECONDS

    clock = _Clock(T0)
    sampled = _auditor(_Store({}), clock)
    full = CorpusAuditor(job_id="order-eval-x", records=_Store({}), model=_model(0), tokenizer=_Tokenizer(),
                         proof=PROOF, params=AuditParams(q=1.0, hold_seconds=HOLD), miner_states=_States({}),
                         beacon=_Beacon(), round_at=_round_at, clock=clock, accept_slack_seconds=0.0)
    for auditor in (sampled, full):
        for sid in _ids(False, 3):
            auditor.enqueue(sid)
    # Just received: a sampled job's records wait for their draw round, not for the audit.
    assert sampled.audit_backlog() == 0
    assert full.audit_backlog() == 3
    clock.now = T0 + DRAW_WAIT_SECONDS + 1
    assert sampled.audit_backlog() == 3
