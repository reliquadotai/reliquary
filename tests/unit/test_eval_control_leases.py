"""Eval control audit leases: each test pins one behaviour of late answers, card paces, full-length leases
and their logs, and names the mutants it kills."""
from __future__ import annotations

import asyncio
import logging

import pytest

from reliquary.validator.corpus_audit_protocol import AuditResult
from reliquary.validator.corpus_audit_remote import LeaseRefused
from tests.unit.test_eval_control import (
    ITEMS, MODEL_A, PROOF, TOKENS, _Harness, _executor, _honest, _late, _paced, _run, _scored_after, _trained,
)

LOGGER = "reliquary.validator.eval_control"
# Two chunks for an item with one proof: the right count of scores, but a misfit.
MISFIT = AuditResult.model_validate({"scores": [
    {"status": "ok", "chunks": [[1, 1.0, 1.0], [1, 1.0, 1.0]]}, {"status": "ok", "chunks": [[1, 1.0, 1.0]]}]})
PARTLY = AuditResult.model_validate({"scores": [
    {"status": "ok", "chunks": [[1, 1.0, 1.0]]}, {"status": "error", "chunks": [], "detail": "boom"}]})


def _fresh(start=0.0):
    now = [start]
    h = _Harness(scorers=1)
    h.dispatcher._clock = lambda: now[0]
    return h, now


async def _lose(h, now, executor_id):
    """A lease to ``executor_id`` that expires and is swept; the lease."""
    scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
    await asyncio.sleep(0)
    lease = h.dispatcher.claim(_executor(executor_id))
    assert lease is not None
    now[0] = lease["expires_at"] + 1
    await h.dispatcher.sweep()
    scoring.cancel()
    await asyncio.sleep(0)
    return lease


# --- the take-back is the answered lease's own (D1, A7) ---------------------------------------------

def test_c_a_late_answer_before_the_sweep_takes_nothing_back():
    """Never swept, never struck: nothing to take back, even at zero strikes."""
    async def go():
        h, now = _fresh()
        await _trained(h, now)
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim(_executor("a"))
        now[0] = lease["expires_at"] + 1
        with pytest.raises(LeaseRefused):
            await h.dispatcher.result("a", lease["lease_id"], _honest())
        assert h.dispatcher.stats["answered_late"] == 1 and h.dispatcher.stats["strikes_taken_back"] == 0
        scoring.cancel()
        await asyncio.sleep(0)
    _run(go)


def test_c_a_late_answer_before_the_sweep_leaves_another_leases_strike():
    async def go():
        h, now = _fresh()
        h.dispatcher._strikes_limit = 3
        await _trained(h, now)
        await _lose(h, now, "a")
        assert h.dispatcher._strikes["a"] == 1
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        second = h.dispatcher.claim(_executor("a"))
        now[0] = second["expires_at"] + 1                 # expired, not swept yet
        with pytest.raises(LeaseRefused):
            await h.dispatcher.result("a", second["lease_id"], _honest())
        assert h.dispatcher._strikes["a"] == 1 and h.dispatcher.stats["strikes_taken_back"] == 0
        scoring.cancel()
        await asyncio.sleep(0)
    _run(go)


# --- what a late answer must score (S3, S5); on-time partly errored (R8) ---------------------------

@pytest.mark.parametrize("result", [MISFIT, PARTLY], ids=["misfit", "partly-errored"])
def test_c_a_late_answer_that_does_not_score_every_item_takes_nothing_back(result):
    async def go():
        h, now = _fresh()
        await _trained(h, now)
        await _late(h, now, "a", 100.0, result)
        assert h.dispatcher._strikes["a"] == 1 and h.dispatcher.stats["answered_late"] == 0
        assert "a" not in h.dispatcher._pace
    _run(go)


def test_c_an_on_time_partly_errored_result_is_neither_timed_nor_clean():
    async def go():
        h, now = _fresh()
        for _ in range(3):
            assert await _scored_after(h, now, "a", 1.0, PARTLY) == "requeued"
        assert not h.dispatcher._lease_times.get(MODEL_A) and "a" not in h.dispatcher._pace
        assert h.dispatcher._clean["a"] == 0
    _run(go)


# --- the clean streak and struck (CS2, CS3, CS4) ---------------------------------------------------

def test_c_a_clean_streak_clears_only_its_own_cards_expired_leases():
    from reliquary.validator.eval_control import CLEAN_STREAK

    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 3)(h, now)
        lost = await _lose(h, now, "b")                   # b: strike 1
        for _ in range(CLEAN_STREAK):                     # a's clean streak, not b's
            assert await _scored_after(h, now, "a", TOKENS) == "accepted"
        with pytest.raises(LeaseRefused):
            await h.dispatcher.result("b", lost["lease_id"], _honest())
        assert h.dispatcher._strikes["b"] == 0 and h.dispatcher.stats["strikes_taken_back"] == 1
    _run(go)


def test_c_on_time_results_short_of_a_streak_leave_the_strike_to_take_back():
    async def go():
        h, now = _fresh()
        h.dispatcher._strikes_limit = 5
        await _paced((1.0,) * 3)(h, now)
        lost = await _lose(h, now, "a")
        for _ in range(2):
            assert await _scored_after(h, now, "a", TOKENS) == "accepted"
        with pytest.raises(LeaseRefused):
            await h.dispatcher.result("a", lost["lease_id"], _honest())
        assert h.dispatcher._strikes["a"] == 0 and h.dispatcher.stats["strikes_taken_back"] == 1
    _run(go)


def test_c_after_a_clean_streak_the_old_lease_answered_late_is_still_timed():
    from reliquary.validator.eval_control import CLEAN_STREAK

    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 3)(h, now)
        lost = await _lose(h, now, "a")
        for _ in range(CLEAN_STREAK):
            assert await _scored_after(h, now, "a", TOKENS) == "accepted"
        times = len(h.dispatcher._lease_times[MODEL_A])
        with pytest.raises(LeaseRefused) as late:
            await h.dispatcher.result("a", lost["lease_id"], _honest())
        assert late.value.detail == "lease_expired"
        assert h.dispatcher.stats["answered_late"] == 1 and len(h.dispatcher._lease_times[MODEL_A]) == times + 1
        assert h.dispatcher._strikes["a"] == 0 and h.dispatcher.stats["strikes_taken_back"] == 0
    _run(go)


# --- the slowest recent pace (P1, P2, P6, P7) ------------------------------------------------------

def test_c_a_card_keeps_its_slowest_pace_after_several_cheap_batches():
    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 5)(h, now)
        await _late(h, now, "slow", 100.0)
        for _ in range(3):
            assert await _scored_after(h, now, "slow", 1.0 * TOKENS) == "accepted"
        assert h.dispatcher.lease_time(MODEL_A, TOKENS, "slow") >= 2 * 100.0
    _run(go)


def test_c_the_slowest_pace_counts_wherever_it_sits():
    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 5)(h, now)
        assert await _scored_after(h, now, "s", 1.0 * TOKENS) == "accepted"     # a cheap batch first
        await _late(h, now, "s", 100.0)
        assert await _scored_after(h, now, "s", 1.0 * TOKENS) == "accepted"
        assert h.dispatcher.lease_time(MODEL_A, TOKENS, "s") >= 2 * 100.0
    _run(go)


# --- the card's 1 h window, from below (P9, P10, C4) and its stamp (TM4, TM5) -----------------------

def test_c_a_slow_pace_still_counts_half_an_hour_later():
    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 5)(h, now)
        await _late(h, now, "slow", 100.0)
        now[0] += 1800.0
        await _paced((1.0,) * 3)(h, now)                  # fresh model samples
        assert h.dispatcher.lease_time(MODEL_A, TOKENS, "slow") >= 2 * 100.0
    _run(go)


def test_c_a_slow_pace_ages_out_after_an_hour_even_while_its_card_keeps_scoring():
    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 5)(h, now)
        await _late(h, now, "slow", 100.0)
        for _ in range(7):
            now[0] += 600.0
            assert await _scored_after(h, now, "slow", 1.0 * TOKENS) == "accepted"
        assert h.dispatcher.lease_time(MODEL_A, TOKENS, "slow") == h.dispatcher.lease_time(MODEL_A, TOKENS)
    _run(go)


def test_c_a_late_answers_pace_counts_for_an_hour_from_the_answer():
    async def go():
        h, now = _fresh()
        await _paced((1.0,) * 5)(h, now)
        await _late(h, now, "slow", 290.0)                # 41 s per token: a capped lease
        now[0] += 3500.0                                  # 3790 s after the grant, 3500 s after the answer
        await _paced((1.0,) * 3)(h, now)
        assert h.dispatcher.lease_time(MODEL_A, TOKENS, "slow") == h.dispatcher._lease_seconds
    _run(go)


# --- full length at the grant (F4, F6) --------------------------------------------------------------

def test_c_a_full_length_lease_stays_lost_whatever_the_float_rounding():
    async def go():
        h, now = _fresh(1764100000.0)                     # an epoch clock, as in production
        h.dispatcher._lease_seconds = 333.3
        for _ in range(2):
            lease, granted = await _late(h, now, "a", 700.0)
        assert h.quarantined == ["a"] and h.dispatcher.stats["strikes_taken_back"] == 0
    _run(go)


def test_c_a_lease_just_under_the_cap_is_not_full_length():
    async def go():
        h, now = _fresh()
        await _paced((125.0 / TOKENS,) * 3)(h, now)       # 2 x 125 s + 30 s: a 280 s lease
        lease, granted = await _late(h, now, "a", 290.0)
        assert lease["expires_at"] - granted == pytest.approx(280.0)
        assert h.dispatcher._strikes["a"] == 0 and h.dispatcher.stats["strikes_taken_back"] == 1
    _run(go)


# --- logs: the reuse expiry at info (RL1, RL2, RL3, W9); times and counts (W4, W7, W8, A16, F7, LD1, LD2)

def test_c_a_reuse_batch_expiry_is_logged_at_info_and_strikes_nothing(caplog):
    async def go():
        h, now = _fresh()
        h.dispatcher.reuse_only_jobs.add("j")
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF, job_id="j").score(ITEMS))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim(_executor("a"))
        assert lease is not None
        now[0] = lease["expires_at"] + 1
        await h.dispatcher.sweep()
        assert h.dispatcher._strikes["a"] == 0 and not h.dispatcher._expired
        scoring.cancel()
        await asyncio.sleep(0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(go)
    lines = [r for r in caplog.records if "of reuse batch" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "expired on executor a (no strike)" in lines[0].getMessage()


def test_c_expiry_and_late_answer_are_logged_with_their_times(caplog):
    async def go():
        h, now = _fresh()
        await _trained(h, now)
        await _late(h, now, "slow", 100.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run(go)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert ("(job , 2 items, 7 tokens) expired unanswered on executor slow, 60 s after its grant: strike 1 of 2"
            in text)
    assert "late, 100 s after its grant: timed, not a lost result" in text


# --- pre-existing (tests-9 of the 03fd1435f review): held tokens (K1) -------------------------------

def test_c_a_second_lease_counts_the_tokens_its_card_already_holds():
    async def go():
        h, now = _fresh()
        await _paced((5.0,) * 3)(h, now)
        view = h.dispatcher.view(*MODEL_A, PROOF)
        scorings = [asyncio.ensure_future(view.score(ITEMS)) for _ in range(2)]
        await asyncio.sleep(0)
        first = h.dispatcher.claim(_executor("a"))
        second = h.dispatcher.claim(_executor("a"))
        assert first["expires_at"] - now[0] == 2 * 5.0 * TOKENS + 30.0
        assert second["expires_at"] - now[0] == 2 * 5.0 * 2 * TOKENS + 30.0
        for s in scorings:
            s.cancel()
        await asyncio.sleep(0)
    _run(go)
