"""The obligation ledger: what settled period archives owe each hotkey,
in whole rao, and when it falls due (J+2 at 12:00 UTC by default)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from reliquary.ledger import obligations as ob
from reliquary.validator import corpus_periods as cp

E = 147.60082398718242          # the miner alpha per period an order froze with its ceiling
PRICE = 1.446224650647019e-08   # its pool share per output token
CEILING = ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=PRICE,
                     max_price_alpha_per_mtoken=2.13463950106075, set_at=1_795_685_015.5,
                     pricing=ob.PRICING_FIXED)
TASK = "order-gen-3ecb8fb8"
WORK = 23_817   # ended 2026-11-26 14:05:27 UTC


def _utc(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def _doc(rewards, *, budget=None, cap=0.06, tokens=None, price=PRICE, **extra):
    doc = {"schema": "reliquary/corpus-period/v1", "task_id": TASK, "job_id": TASK,
           "mechanism": "corpus-generation", "work_period": WORK, "rewards_by_hotkey": dict(rewards),
           "cap": cap, "tokens": tokens if tokens is not None else 1_000_000, "verdicts": len(rewards)}
    if price is not None:
        doc.update(max_price_per_token=price, budget=budget if budget is not None else cap)
    doc.update(extra)
    return doc


def _ref(doc, *, work=WORK, entry=WORK + 1, settled_at=None, task=TASK):
    return ob.ArchiveRef(task_id=task, work_period=work, entry_period=entry, document=doc,
                         settled_at=settled_at if settled_at is not None else cp.period_end(work) + 600)


def _ledger(refs, **kwargs):
    kwargs.setdefault("ceilings", {TASK: [CEILING]})
    return ob.build_ledger(refs, **kwargs)


# -- identity -----------------------------------------------------------------------------------------------------


def test_the_id_is_a_hash_of_order_period_hotkey_unit_type_and_origin_entry():
    first = ob.obligation_id(TASK, WORK, "5Hot", ob.UNIT_OUTPUT_TOKENS, WORK + 1)
    assert first == ob.obligation_id(TASK, WORK, "5Hot", ob.UNIT_OUTPUT_TOKENS, WORK + 1)
    assert len(first) == 64 and int(first, 16) >= 0
    others = {ob.obligation_id("order-other", WORK, "5Hot", ob.UNIT_OUTPUT_TOKENS, WORK + 1),
              ob.obligation_id(TASK, WORK + 1, "5Hot", ob.UNIT_OUTPUT_TOKENS, WORK + 1),
              ob.obligation_id(TASK, WORK, "5Other", ob.UNIT_OUTPUT_TOKENS, WORK + 1),
              ob.obligation_id(TASK, WORK, "5Hot", ob.UNIT_OUTPUT_AND_PROMPT_TOKENS, WORK + 1),
              ob.obligation_id(TASK, WORK, "5Hot", ob.UNIT_OUTPUT_TOKENS, WORK + 3)}
    assert first not in others and len(others) == 5


def test_the_id_is_pinned():
    # A payer pays each id once: changing how ids are made would pay the same work again.
    assert ob.obligation_id("order-gen-x", 23817, "5Abc", ob.UNIT_OUTPUT_TOKENS, 23818) == (
        "809d92c9e4791bf5938beb2993441d70ed6fb78ef4b4c68b47a371e90236ccfc")


def test_an_unknown_unit_type_or_origin_is_refused():
    with pytest.raises(ValueError):
        ob.obligation_id(TASK, WORK, "5Hot", "tokens", WORK + 1)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError):
            ob.obligation_id(TASK, WORK, "5Hot", ob.UNIT_OUTPUT_TOKENS, bad)


def test_the_ledger_is_idempotent():
    refs = [_ref(_doc({"5A": 0.01, "5B": 0.02})),
            _ref(_doc({"5A": 0.005}), work=WORK + 1, entry=WORK + 2)]
    first, again = _ledger(refs), _ledger(list(reversed(refs)))
    assert first == again
    assert [o.obligation_id for o in first.obligations] == [o.obligation_id for o in again.obligations]
    assert len({o.obligation_id for o in first.obligations}) == 3


def test_reading_an_archive_twice_makes_no_second_obligation():
    ref = _ref(_doc({"5A": 0.01}))
    once = _ledger([ref])
    # The same key listed twice (a retried listing) is one archive, one obligation.
    twice = _ledger([ref, replace(ref)])
    assert twice == once and len(twice.obligations) == 1


# -- amounts --------------------------------------------------------------------------------------------------------


def test_amounts_are_whole_rao_rounded_down_exactly():
    share = 0.05141239341842233
    exact = Fraction(share) * Fraction(E) * 10**9
    assert ob.amount_rao(share, E) == math.floor(exact) == 7_588_511_631
    # Rounded down even a hair under the next rao, never up.
    assert ob.amount_rao(Fraction(2_999_999_999, 10**18), 1.0) == 2
    assert ob.amount_rao(0.0, E) == 0


def test_a_share_times_the_frozen_rate_is_verified_tokens_times_the_alpha_price():
    tokens = 1_000_000
    ledger = _ledger([_ref(_doc({"5A": PRICE * tokens}, budget=PRICE * tokens))])
    (obligation,) = ledger.obligations
    alpha_per_mtoken = PRICE * 1e6 * E
    # Down to the rao: never more than the price, less by under one rao.
    assert obligation.alpha == pytest.approx(alpha_per_mtoken * tokens / 1e6, abs=1e-9)
    assert obligation.alpha <= alpha_per_mtoken * tokens / 1e6
    assert obligation.units == tokens
    assert obligation.rate_source == ob.RATE_FROZEN and obligation.alpha_per_share == E


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), "0.1", True])
def test_a_reward_that_is_not_a_finite_non_negative_number_gives_no_obligation(bad):
    ledger = _ledger([_ref(_doc({"5A": 0.01, "5B": bad}))])
    assert ledger.obligations == ()
    assert [a["kind"] for a in ledger.anomalies] == ["invalid_archive"]


def test_less_than_one_rao_is_not_an_obligation():
    ledger = _ledger([_ref(_doc({"5A": 0.01, "5Dust": 1e-15}))])
    assert [o.hotkey for o in ledger.obligations] == ["5A"]
    assert ledger.anomalies[0]["kind"] == "under_one_rao"


# -- the period's budget and the order's -------------------------------------------------------------------------------


def test_an_archive_paying_past_its_budget_is_cut_to_it_and_never_owes_more():
    budget = 0.03
    ledger = _ledger([_ref(_doc({"5A": 0.02, "5B": 0.02}, budget=budget))])
    assert all(o.capped == ob.CAPPED_PERIOD_BUDGET for o in ledger.obligations)
    total = sum(o.amount_rao for o in ledger.obligations)
    assert total <= math.floor(Fraction(budget) * Fraction(E) * 10**9)
    assert total >= math.floor(Fraction(budget) * Fraction(E) * 10**9) - 2  # two floors lose under two rao
    a, b = ledger.obligations
    assert abs(a.amount_rao - b.amount_rao) <= 1
    assert ledger.anomalies[0]["kind"] == "over_period_budget"


def test_the_cap_binds_when_it_is_below_the_budget():
    ledger = _ledger([_ref(_doc({"5A": 0.05}, budget=0.05, cap=0.04))])
    (obligation,) = ledger.obligations
    assert obligation.capped == ob.CAPPED_PERIOD_BUDGET
    assert obligation.amount_rao == math.floor(Fraction(0.04) * Fraction(E) * 10**9)


def test_the_task_pay_ceiling_bounds_an_archive_as_the_weight_setter_does():
    ledger = _ledger([_ref(_doc({"5A": 0.05}, budget=0.05, cap=0.05))], pay_ceilings={TASK: 0.02})
    assert ledger.obligations[0].amount_rao == math.floor(Fraction(0.02) * Fraction(E) * 10**9)


def test_an_archive_within_its_budget_is_paid_as_settled():
    ledger = _ledger([_ref(_doc({"5A": 0.02, "5B": 0.005}, budget=0.03))])
    assert [o.capped for o in ledger.obligations] == [None, None]
    assert [o.amount_rao for o in ledger.obligations] == [ob.amount_rao(0.02, E), ob.amount_rao(0.005, E)]
    assert ledger.anomalies == ()


def test_shares_making_up_the_budget_in_floats_are_not_a_cut():
    # 0.02 + 0.01 is a hair over 0.03 as exact rationals of the floats: rounding, not overpay.
    assert Fraction(0.02) + Fraction(0.01) > Fraction(0.03)
    ledger = _ledger([_ref(_doc({"5A": 0.02, "5B": 0.01}, budget=0.03))])
    assert [o.capped for o in ledger.obligations] == [None, None] and ledger.anomalies == ()
    assert sum(o.amount_rao for o in ledger.obligations) <= math.floor(Fraction(0.03) * Fraction(E) * 10**9)
    assert [o.amount_rao for o in ledger.obligations] == pytest.approx(
        [ob.amount_rao(0.02, E), ob.amount_rao(0.01, E)], abs=1)


def test_a_late_part_of_a_period_gets_only_what_its_budget_has_left():
    first = _ref(_doc({"5A": 0.03}, budget=0.03))
    # Verdicts that arrived after the period was paid: settled again, entered later.
    late = _ref(_doc({"5A": 0.01, "5B": 0.01}, budget=0.04), entry=WORK + 3)
    ledger = _ledger([late, first])
    parts = sorted({(o.part, o.entry_period) for o in ledger.obligations})
    assert parts == [(0, WORK + 1), (1, WORK + 3)]
    total = sum(o.amount_rao for o in ledger.obligations)
    assert total <= math.floor(Fraction(0.04) * Fraction(E) * 10**9)
    late_ids = {o.obligation_id for o in ledger.obligations if o.part == 1}
    assert late_ids.isdisjoint({o.obligation_id for o in ledger.obligations if o.part == 0})


def test_an_order_budget_is_spent_in_settlement_order_so_later_parts_never_reclaim_it():
    share = 0.0025                     # 0.369 alpha at E
    budget = math.floor(Fraction(share) * Fraction(E) * 10**9 * 5 / 2)     # two and a half periods
    refs = [_ref(_doc({"5A": share}), work=WORK + i, entry=WORK + i + 1,
                 settled_at=cp.period_end(WORK + i) + 600) for i in range(4)]
    first = _ledger(refs, order_budgets_rao={TASK: budget})
    paid = {o.work_period: o.amount_rao for o in first.obligations}
    assert sum(paid.values()) <= budget and set(paid) == {WORK, WORK + 1, WORK + 2}
    # A late part of the first period settles after all of them: it takes nothing from what was allocated.
    late = _ref(_doc({"5B": share}, budget=1.0), work=WORK, entry=WORK + 9, settled_at=cp.period_end(WORK + 8))
    again = _ledger(refs + [late], order_budgets_rao={TASK: budget})
    assert {o.work_period: o.amount_rao for o in again.obligations if o.hotkey == "5A"} == paid
    assert not [o for o in again.obligations if o.hotkey == "5B"]
    assert {a["kind"] for a in again.anomalies} >= {"order_budget_spent"}


def test_a_late_part_zeroed_by_the_spent_period_budget_says_so():
    first = _ref(_doc({"5A": 0.05}, cap=0.05, price=None), entry=WORK + 1)
    late = _ref(_doc({"5B": 0.01}, cap=0.05, price=None), entry=WORK + 3)
    ledger = _ledger([first, late], rates=ob.RatePolicy(mode=ob.RATE_FIXED, alpha_per_period=E))
    assert [o.hotkey for o in ledger.obligations] == ["5A"]
    kinds = [a["kind"] for a in ledger.anomalies]
    assert "period_budget_spent" in kinds and "under_one_rao" not in kinds


def test_the_earn_instant_is_the_end_of_the_work_period_whatever_the_settlement_time():
    for settled in (cp.period_end(WORK) + 60, cp.period_end(WORK) + 9 * 86_400):
        (obligation,) = _ledger([_ref(_doc({"5A": 0.01}), settled_at=settled)]).obligations
        assert obligation.earn_at == cp.period_end(WORK) and obligation.earned_at == settled


def test_an_order_budget_stops_the_order_once_spent_period_by_period():
    budget_rao = ob.amount_rao(0.025, E)
    refs = [_ref(_doc({"5A": 0.01, "5B": 0.01}), work=WORK + i, entry=WORK + i + 1) for i in range(3)]
    ledger = _ledger(refs, order_budgets_rao={TASK: budget_rao})
    by_work = {}
    for o in ledger.obligations:
        by_work.setdefault(o.work_period, []).append(o)
    assert sum(o.amount_rao for o in ledger.obligations) <= budget_rao
    assert all(o.capped is None for o in by_work[WORK])
    # The period that crosses the budget is cut pro rata, the next one gets nothing.
    assert {o.capped for o in by_work[WORK + 1]} == {ob.CAPPED_ORDER_BUDGET}
    a, b = by_work[WORK + 1]
    assert abs(a.amount_rao - b.amount_rao) <= 1
    assert WORK + 2 not in by_work
    assert any(a["kind"] == "order_budget_spent" for a in ledger.anomalies)


# -- unit types and rates -------------------------------------------------------------------------------------------


def test_unit_types_follow_how_the_archive_was_priced():
    assert ob.unit_type_of(_doc({})) == ob.UNIT_OUTPUT_TOKENS
    assert ob.unit_type_of(_doc({}, max_price_input_per_token=3e-9)) == ob.UNIT_OUTPUT_AND_PROMPT_TOKENS
    assert ob.unit_type_of(_doc({}, price=None)) == ob.UNIT_PERIOD_SHARE
    assert ob.unit_type_of({"schema": ob.RL_ARCHIVE_SCHEMA, "mechanism": "rl-async"}) == ob.UNIT_RL_ROLLOUTS


def test_the_rl_archive_schema_is_the_rl_ledgers():
    from reliquary.validator.rl_ledger import RL_PERIOD_ARCHIVE_SCHEMA

    assert ob.RL_ARCHIVE_SCHEMA == RL_PERIOD_ARCHIVE_SCHEMA


def test_units_are_known_only_where_the_archive_splits_by_tokens():
    fixed = _ledger([_ref(_doc({"5A": 0.02, "5B": 0.01}, budget=0.03, tokens=3000))])
    assert [o.units for o in fixed.obligations] == [2000, 1000]
    auction = ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=PRICE,
                         pricing=ob.PRICING_AUCTION)
    priced = _ledger([_ref(_doc({"5A": 0.02}, tokens=3000))], ceilings={TASK: [auction]})
    assert priced.obligations[0].units is None
    unknown = ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=PRICE)
    assert _ledger([_ref(_doc({"5A": 0.02}))], ceilings={TASK: [unknown]}).obligations[0].units is None


def test_the_rate_is_the_ceiling_the_archive_was_paid_under_and_nothing_else_by_default():
    older = ob.Ceiling(miner_alpha_per_period=140.0, max_price_pool_share_per_token=PRICE / 2, set_at=1.0)
    newer = ob.Ceiling(miner_alpha_per_period=150.0, max_price_pool_share_per_token=PRICE, set_at=2.0)
    assert ob.resolve_rate(_doc({}, price=PRICE / 2), [older, newer], ob.RatePolicy()) == (140.0, ob.RATE_FROZEN)
    assert ob.resolve_rate(_doc({}, price=PRICE), [older, newer], ob.RatePolicy()) == (150.0, ob.RATE_FROZEN)
    # A price no known ceiling had: unvalued, unless the order's latest ceiling is opted into, and then said so.
    assert ob.resolve_rate(_doc({}, price=PRICE * 3), [older, newer], ob.RatePolicy()) == (None, "unvalued")
    assert ob.resolve_rate(_doc({}, price=PRICE * 3), [older, newer], ob.RatePolicy(fallbacks=("latest",))) == (
        150.0, ob.RATE_FROZEN_LATEST)


def test_a_re_priced_order_leaves_its_earlier_archives_unvalued_not_revalued():
    repriced = ob.Ceiling(miner_alpha_per_period=4.8, max_price_pool_share_per_token=PRICE * 7)
    strict = ob.build_ledger([_ref(_doc({"5A": 0.01}))], ceilings={TASK: [repriced]})
    assert strict.obligations == () and [a["kind"] for a in strict.anomalies] == ["unvalued"]
    # Its shares still count for the reconciliation, which needs no rate.
    assert strict.shares == ((TASK, WORK + 1, "5A", 0.01),)
    labelled = ob.build_ledger([_ref(_doc({"5A": 0.01}))], ceilings={TASK: [repriced]},
                               rates=ob.RatePolicy(fallbacks=("latest",)))
    assert labelled.obligations[0].rate_source == ob.RATE_FROZEN_LATEST
    assert [a["kind"] for a in labelled.anomalies] == ["valued_by_fallback"]


def test_without_a_ceiling_nothing_is_valued_unless_the_chain_rate_is_opted_into():
    rl = {"schema": ob.RL_ARCHIVE_SCHEMA, "mechanism": "rl-async", "rewards_by_hotkey": {"5A": 0.01},
          "cap": 0.1}
    assert ob.resolve_rate(rl, [CEILING], ob.RatePolicy(alpha_per_period=147.0)) == (None, "unvalued")
    assert ob.resolve_rate(rl, [CEILING], ob.RatePolicy(alpha_per_period=147.0, fallbacks=("chain",))) == (
        147.0, ob.RATE_CHAIN)
    with pytest.raises(ValueError):
        ob.RatePolicy(fallbacks=("chain",))
    with pytest.raises(ValueError):
        ob.RatePolicy(fallbacks=("whatever",))
    ledger = ob.build_ledger([_ref(rl)], ceilings={})
    assert ledger.obligations == () and ledger.anomalies[0]["kind"] == "unvalued"


def test_the_chain_and_fixed_modes_need_a_rate():
    with pytest.raises(ValueError):
        ob.RatePolicy(mode=ob.RATE_CHAIN)
    assert ob.resolve_rate(_doc({}), [CEILING], ob.RatePolicy(mode=ob.RATE_FIXED, alpha_per_period=100.0)) == (
        100.0, ob.RATE_FIXED)


# -- parts: late verdicts and requeued archives -----------------------------------------------------------------------


def test_an_earlier_part_absent_leaves_a_late_parts_ids_unchanged():
    first = _ref(_doc({"5A": 0.03}, budget=0.03))
    late = _ref(_doc({"5A": 0.01, "5B": 0.01}, budget=0.04), entry=WORK + 3)
    both = {o.obligation_id for o in _ledger([first, late]).obligations if o.entry_period == WORK + 3}
    alone = {o.obligation_id for o in _ledger([late]).obligations}
    # The late part's ids never collide with part 0's, present or not: nothing is paid twice or skipped.
    assert both == alone
    assert alone.isdisjoint({o.obligation_id for o in _ledger([first]).obligations})


def test_an_archive_requeued_twice_keeps_its_first_origin():
    original = _ledger([_ref(_doc({"5A": 0.01}), entry=WORK + 9)])
    twice = _ledger([_ref(_doc({"5A": 0.01}, replaces_entry_period=WORK + 5, origin_entry_period=WORK + 9),
                          entry=WORK + 2)])
    assert [o.obligation_id for o in twice.obligations] == [o.obligation_id for o in original.obligations]
    assert twice.obligations[0].origin_entry == WORK + 9


def test_a_requeued_archive_replaces_the_one_it_names_and_keeps_its_ids():
    original = _ref(_doc({"5A": 0.01}), entry=WORK + 9)
    before = _ledger([original])
    moved = _ref(_doc({"5A": 0.01}, replaces_entry_period=WORK + 9), entry=WORK + 2)
    both = _ledger([original, moved])
    after = _ledger([moved])
    assert [o.obligation_id for o in both.obligations] == [o.obligation_id for o in before.obligations]
    assert [o.obligation_id for o in after.obligations] == [o.obligation_id for o in before.obligations]
    assert both.obligations[0].entry_period == WORK + 2


# -- the J+2 calendar and the states ----------------------------------------------------------------------------------


def test_the_work_day_is_the_utc_day_a_period_ended():
    assert ob.work_day(WORK) == date(2026, 11, 26)
    assert ob.utc_day(cp.period_end(WORK)) == date(2026, 11, 26)
    # A period ending exactly at midnight (another genesis) belongs to the day before.
    genesis = _utc("2026-11-26T00:00:00") - cp.PERIOD_SECONDS
    assert ob.work_day(0, genesis) == date(2026, 11, 25)


def test_twenty_periods_end_on_every_utc_day():
    day = date(2026, 11, 26)
    first = cp.period_of(ob.day_start(day))
    ends = [p for p in range(first - 2, first + 25) if ob.work_day(p) == day]
    assert len(ends) == 20


def test_an_obligation_is_due_at_twelve_utc_two_days_after_its_work_day():
    terms = ob.PayTerms()
    settled = cp.period_end(WORK) + 900
    assert ob.due_at(date(2026, 11, 26), settled, terms) == _utc("2026-11-28T12:00:00")


def test_a_late_settlement_is_due_on_the_first_pass_after_it():
    terms = ob.PayTerms()
    day = date(2026, 11, 26)
    assert ob.due_at(day, _utc("2026-11-28T11:59:59"), terms) == _utc("2026-11-28T12:00:00")
    assert ob.due_at(day, _utc("2026-11-28T12:00:00"), terms) == _utc("2026-11-28T12:00:00")
    assert ob.due_at(day, _utc("2026-11-28T12:00:01"), terms) == _utc("2026-11-29T12:00:00")
    assert ob.due_at(day, _utc("2026-12-04T03:00:00"), terms) == _utc("2026-12-04T12:00:00")


def test_the_terms_are_settings():
    terms = ob.PayTerms(lag_days=1, pay_at_seconds=17 * 3600, hold_seconds=6 * 3600)
    day = date(2026, 11, 26)
    assert ob.due_at(day, _utc("2026-11-26T20:00:00"), terms) == _utc("2026-11-27T17:00:00")
    # Held at least six hours: a settlement at 12:00 misses that day's 17:00 pass.
    assert ob.due_at(day, _utc("2026-11-27T12:00:00"), terms) == _utc("2026-11-28T17:00:00")
    for bad in ({"lag_days": -1}, {"lag_days": 1.5}, {"pay_at_seconds": 86_400}, {"hold_seconds": -1},
                {"escalate_after_seconds": float("nan")}):
        with pytest.raises(ValueError):
            ob.PayTerms(**bad)


def test_states_go_earned_held_due_and_never_further():
    ledger = _ledger([_ref(_doc({"5A": 0.01}), settled_at=_utc("2026-11-26T15:00:00"))])
    (obligation,) = ledger.obligations
    assert obligation.state_at(_utc("2026-11-26T14:59:59")) is None
    assert obligation.state_at(_utc("2026-11-26T15:00:00")) == ob.EARNED
    assert obligation.state_at(_utc("2026-11-26T23:59:59")) == ob.EARNED
    assert obligation.state_at(_utc("2026-11-27T00:00:00")) == ob.HELD
    assert obligation.state_at(_utc("2026-11-28T11:59:59")) == ob.HELD
    assert obligation.state_at(_utc("2026-11-28T12:00:00")) == ob.DUE
    assert obligation.state_at(_utc("2027-01-01T00:00:00")) == ob.DUE
    assert set(ob.LEDGER_STATES) == {ob.EARNED, ob.HELD, ob.DUE}


def test_a_settlement_after_its_day_closed_is_held_at_once():
    ledger = _ledger([_ref(_doc({"5A": 0.01}), settled_at=_utc("2026-11-27T09:00:00"))])
    assert ledger.obligations[0].state_at(_utc("2026-11-27T09:00:00")) == ob.HELD


def test_an_unknown_settlement_time_is_taken_at_the_entry_period_start():
    ref = ob.ArchiveRef(task_id=TASK, work_period=WORK, entry_period=WORK + 5, document=_doc({"5A": 0.01}))
    (obligation,) = _ledger([ref]).obligations
    assert obligation.earned_at == cp.period_end(WORK + 4)


def test_only_archives_settled_by_then_are_visible():
    ledger = _ledger([_ref(_doc({"5A": 0.01}), settled_at=_utc("2026-11-26T15:00:00"))])
    assert ledger.visible_at(_utc("2026-11-26T14:00:00")) == []
    assert len(ledger.visible_at(_utc("2026-11-26T15:00:00"))) == 1


# -- work awaiting audit ------------------------------------------------------------------------------------------------


def test_a_period_settled_late_was_awaiting_audit_until_then():
    settled = _utc("2026-12-05T10:00:00")
    refs = [_ref(_doc({"5A": 0.01}), settled_at=settled)]
    before = ob.pending_from_archives(refs, _utc("2026-11-30T00:00:00"))
    assert [(p.work_period, p.reason) for p in before] == [(WORK, "settled_later")]
    assert ob.pending_from_archives(refs, settled) == []
    # Not due while pending: its obligation is not even visible.
    assert ob.build_ledger(refs, ceilings={TASK: [CEILING]}).visible_at(_utc("2026-11-30T00:00:00")) == []


def test_pending_work_is_never_cancelled_only_escalated():
    terms = ob.PayTerms()
    item = ob.PendingWork(job_id=TASK, task_id=TASK, work_period=WORK, since=cp.period_end(WORK),
                          reason="unsettled")
    assert item.escalate_at(cp.period_end(WORK) + 6 * 86_400, terms) is False
    assert item.escalate_at(cp.period_end(WORK) + 8 * 86_400, terms) is True
    # A year later it is still listed: nothing drops it.
    row = item.to_dict(cp.period_end(WORK) + 365 * 86_400, terms)
    assert row["escalate"] is True and row["work_period"] == WORK
    unknown = ob.PendingWork(job_id=TASK, reason="awaiting_audit", submissions_awaiting=44)
    assert unknown.escalate_at(0.0, terms) is None


def test_hotkey_totals_sum_obligations():
    ledger = _ledger([_ref(_doc({"5A": 0.01, "5B": 0.02})),
                      _ref(_doc({"5A": 0.005}), work=WORK + 1, entry=WORK + 2)])
    totals = ob.by_hotkey(ledger.obligations)
    assert totals["5A"] == ob.amount_rao(0.01, E) + ob.amount_rao(0.005, E)
    assert totals["5B"] == ob.amount_rao(0.02, E)


def test_the_body_task_id_never_overrides_the_key():
    ledger = _ledger([_ref(_doc({"5A": 0.01}, task_id="someone-else"))])
    assert ledger.obligations[0].task_id == TASK
    assert ledger.anomalies[0]["kind"] == "task_id_mismatch"


def test_due_dates_line_up_with_pass_days():
    terms = ob.PayTerms()
    at = _utc("2026-11-28T12:00:00")
    assert ob.pass_day(at, terms) == date(2026, 11, 28)
    assert ob.pass_day(at - 1, terms) == date(2026, 11, 27)
    assert ob.next_pass(at - 1, terms) == at and ob.next_pass(at + 1, terms) == at + 86_400
    assert ob.pay_instant(date(2026, 11, 28), terms) - ob.day_start(date(2026, 11, 28)) == 12 * 3600
    assert timedelta(seconds=terms.pay_at_seconds) == timedelta(hours=12)
