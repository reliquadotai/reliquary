"""Batches no audit card can return ("poison", 2026-11-24: every 9B card was set aside on trajectories past the 64
spans an audit item carries), parked per record, on evidence against the batch, for a bounded time; and parked records
leaving their period's close two periods after their receipt, the stake guarantee kept.

Every test drives the real ``PairedAuditDispatcher`` (claims, results, sweeps on a fake clock) or the real
``EvalAuditor``. Names missing on main are read with ``getattr``, so each test fails there on what it checks."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState, drawn
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc import ChunkResult
from reliquary.validator import corpus_periods as cp
from reliquary.validator import eval_control as ec
from reliquary.validator.corpus_audit_protocol import MAX_ITEM_SPANS, AuditResult
from reliquary.validator.corpus_audit_remote import LeaseRefused
from reliquary.validator.eval_control import BatchParked, PairedAuditDispatcher, eval_auditor

MODEL = ("org/a", "a" * 40)
JOB = "order-eval-poison"
BENCH = getattr(ec, "BENCH_SECONDS", 300.0)
RETRY = getattr(ec, "PARK_RETRY_SECONDS", 1800.0)
GIVE_UP = getattr(ec, "PARKED_CLOSE_GIVE_UP_SECONDS", 2 * 4320.0)
ItemsUnreadable = getattr(ec, "ItemsUnreadable", type("NotOnMain", (Exception,), {}))


def _doc(eid, provider=None, host=None):
    return {"executor_id": eid, "model_id": MODEL[0], "model_revision": MODEL[1],
            "provider_id": provider or f"prov-{eid}", "host": host or f"host-{eid}",
            "status": "active", "expires_at": 1e12}


def _items(n, tokens=4):
    return [{"tokens": list(range(1, tokens + 1)), "prompt_len": 1, "proofs": ["p"]} for _ in range(n)]


def _ok(lease):
    return AuditResult.model_validate({"scores": [
        {"status": "ok", "chunks": [[1, 1.0, 1.0]] * len(item["proofs"])} for item in lease["items"]]})


def _status(lease, status):
    return AuditResult.model_validate({"scores": [{"status": status, "chunks": []} for _ in lease["items"]]})


def _misfit(lease):
    return AuditResult.model_validate({"scores": [
        {"status": "ok", "chunks": [[1, 1.0, 1.0]] * (len(item["proofs"]) + 1)} for item in lease["items"]]})


class _Pool:
    """A real dispatcher on a fake clock (one scorer, as in production)."""

    def __init__(self, scorers=1):
        self.now = 1_000_000.0
        self.quarantined = []

        async def write(executor_id, reason):
            self.quarantined.append(executor_id)

        self.dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None),
                                                quarantine=write, scorers=scorers, clock=lambda: self.now)
        self.unparked = []
        if hasattr(self.dispatcher, "subscribe_unparked"):
            self.dispatcher.subscribe_unparked(lambda job, batch: self.unparked.append((job, batch)))

    def call(self, items, job=JOB):
        return asyncio.ensure_future(self.dispatcher.score(MODEL, PROOF, items, job))

    def claim(self, eid, **placement):
        """A lease, with the batch and job it is for (``batch``, ``job``), or None."""
        lease = self.dispatcher.claim(_doc(eid, **placement))
        if lease is not None:
            held = self.dispatcher.lease_of(lease["lease_id"]).batch
            lease = {**lease, "batch": held.id, "job": held.job_id}
        return lease

    async def answer(self, eid, lease, result=None):
        return await self.dispatcher.result(eid, lease["lease_id"], result or _ok(lease))

    async def expire(self, lease):
        self.now = max(self.now, lease["expires_at"] + 1)
        await self.dispatcher.sweep()

    def strikes(self, eid):
        return self.dispatcher._strikes[eid]


async def _witness(pool, eid, **placement):
    """``eid`` scores a batch of a job of its own, the only one it may take: a lease of it granted before is
    witnessed when lost."""
    pool.now += 1
    other = pool.call(_items(1), job=f"order-eval-witness-{eid}")
    await asyncio.sleep(0)
    lease = pool.claim(eid, **placement)
    assert lease is not None and lease["job"] == f"order-eval-witness-{eid}", "the witness took another batch"
    pool.now += 1
    assert await pool.answer(eid, lease) == "accepted"
    await _ended(other)


def _run(factory):
    return asyncio.run(asyncio.wait_for(factory(), timeout=30))


async def _ended(awaitable, seconds=1.0):
    """What a call gives once it ended (parked, scored, cancelled), at once on a fake clock: bounded, so a call that
    never ends fails the test instead of hanging it."""
    return await asyncio.wait_for(awaitable, timeout=seconds)


# -- validation when the batch is made -----------------------------------------


def _drawn_trajectory(turns):
    from reliquary.corpus.draw_audit import DrawSpec

    prompt = 4
    spans = [[prompt + 2 * t, prompt + 2 * t + 1] for t in range(turns)]
    return {"tokens": list(range(1, prompt + 2 * turns + 1)), "prompt_len": prompt, "proofs": [], "spans": spans,
            "draw": DrawSpec("ab" * 32, 0, "c" * 64, 0, "domain").to_wire()}


def test_items_no_executor_can_read_are_refused_before_any_batch_is_made():
    """2026-11-24: agentic trajectories of 95 turns, past the 64 spans an audit item carried then (#590 raised the
    bound); every executor refused the lease, and every card was set aside after two lost leases. Now the call is
    refused at once, by item, and nothing is leased."""
    async def go():
        pool = _Pool()
        items = [*_items(1), _drawn_trajectory(MAX_ITEM_SPANS + 1), *_items(1),
                 {**_items(1)[0], "tokens": [1] * 65_600}]
        call = pool.call(items)
        await asyncio.sleep(0)
        assert pool.dispatcher.claim(_doc("a"), ("reliquary.corpus-audit/v1", "reliquary.corpus-audit/v3")) is None, \
            "a batch no executor can read was leased"
        with pytest.raises(ItemsUnreadable) as refused:
            await _ended(call)
        assert sorted(refused.value.reasons) == [1, 3]
        assert "spans" in refused.value.reasons[1] and "65536" in refused.value.reasons[3]
        assert "1, 1, 1" not in str(refused.value)         # the why names the rule, not the miner's tokens
        assert pool.dispatcher.pending() == 0
        status = pool.dispatcher.status()
        assert status["jobs"][JOB]["needs_attention"] and status["stats"]["unreadable_items"] == 2
        assert "2 item(s) no executor can read" in status["jobs"][JOB]["reasons"][-1]
        fine = pool.call([_drawn_trajectory(MAX_ITEM_SPANS)])   # at the bound: leased as before
        await asyncio.sleep(0)
        assert pool.dispatcher.claim(_doc("a"), ("reliquary.corpus-audit/v3",)) is not None
        fine.cancel()
    _run(go)


class _Tokenizer:
    chat_template = "x"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 for c in text]


def _record(sid, mark, *, tokens=2, hotkey="5Hot", received_at=None):
    """A record whose completion ends with ``mark``: a fake remote tells records apart by it."""
    record = {"submission_id": sid, "rendered_prompt": "ab", "hotkey": hotkey, "token_count": tokens,
              "completions": [{"tokens": [1] * (tokens - 1) + [mark], "proofs": ["p"]}]}
    if received_at is not None:
        record["received_at"] = received_at
    return record


def test_the_record_with_an_unreadable_item_is_parked_alone_and_the_rest_of_its_call_is_scored():
    """Only the record at fault is parked, not its pass: the other records go to a card in the same call."""
    async def go():
        pool = _Pool()
        auditor = eval_auditor(job_id=JOB, records=None, tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                               remote=pool.dispatcher.view(*MODEL, PROOF, JOB))
        records = [_record("s1", 3), _record("s2", 4, tokens=65_600), _record("s3", 5)]
        outcomes = asyncio.ensure_future(auditor._audit_outcomes(records))
        for _ in range(5):
            await asyncio.sleep(0)
        lease = pool.claim("a")
        assert lease is not None and len(lease["items"]) == 2, "the unreadable record went to a card"
        assert await pool.answer("a", lease) == "accepted"
        first, parked, third = await outcomes
        assert first["passed"] and third["passed"] and first["scored_by"] == ["a"]
        assert not isinstance(parked, dict) and "65536" in parked
        assert auditor.parked_ids == {"s2"}
    _run(go)


# -- parking only on evidence against the batch --------------------------------


def test_a_batch_lost_on_two_providers_once_witnessed_is_parked_and_its_witnessed_strikes_taken_back():
    async def go():
        pool = _Pool()
        poison, good = pool.call(_items(1)), pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = pool.claim("a")                               # the poison batch, the oldest
        a2 = pool.claim("a")                               # and the good one
        pool.now += 1
        assert await pool.answer("a", a2) == "accepted"    # a scores: its loss of the poison is witnessed
        await _ended(good)
        await pool.expire(a1)
        assert pool.strikes("a") == 1 and not poison.done()    # struck, as #493 does; the batch still queued
        b1 = pool.claim("b")                               # another provider takes it, answers nothing
        assert b1["batch"] == a1["batch"]
        await pool.expire(b1)
        with pytest.raises(BatchParked) as parked:
            await _ended(poison)
        assert parked.value.indexes == [0] and parked.value.batch_ids == [a1["batch"]]
        assert pool.strikes("a") == 0                      # it works, the batch does not: taken back
        assert pool.strikes("b") == 1                      # unwitnessed: it stands
        status = pool.dispatcher.status()
        reason = status["jobs"][JOB]["reasons"][-1]
        assert "a (prov-a, witnessed)" in reason and "b (prov-b)" in reason and "2 providers" in reason
        assert status["parks_by_executor"] == {"a": 1, "b": 1} and status["stats"]["unanswered_parked"] == 1
        assert pool.quarantined == []
    _run(go)


def test_losses_on_one_provider_never_park_a_batch_and_its_cards_take_other_work():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        for eid in ("a", "b"):                             # two working cards of one provider, on two hosts
            lease = pool.claim(eid, provider="prov-x")
            await _witness(pool, eid, provider="prov-x")
            await pool.expire(lease)
            # Benched from the batch it lost: it takes other work instead of losing that one again at once.
            assert pool.claim(eid, provider="prov-x") is None
        assert not poison.done(), "one provider's losses parked a batch"
        nxt = pool.call(_items(1), job="order-eval-next")
        await asyncio.sleep(0)
        lease = pool.claim("b", provider="prov-x")
        assert lease["job"] == "order-eval-next" and await pool.answer("b", lease) == "accepted"
        await _ended(nxt)
        c = pool.claim("c")                                # a third provider: the evidence that was missing
        await pool.expire(c)
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_cards_that_answer_nothing_never_park_a_batch_and_are_still_set_aside():
    """#493: a card that answers nothing is set aside after two lost results. None of its losses is witnessed, so a dead
    pool parks nothing, and the first working card scores the batches."""
    async def go():
        pool = _Pool()
        calls = [pool.call(_items(1)) for _ in range(2)]
        await asyncio.sleep(0)
        for eid in ("dead1", "dead2"):
            first = pool.claim(eid)
            await pool.expire(first)
            second = pool.claim(eid)
            assert second["batch"] != first["batch"]      # benched from the batch it lost
            await pool.expire(second)
        assert pool.quarantined == ["dead1", "dead2"]
        assert not any(call.done() for call in calls)
        for _ in range(2):
            assert await pool.answer("good", pool.claim("good")) == "accepted"
        await _ended(asyncio.gather(*calls))
    _run(go)


def test_an_answer_with_no_measure_witnesses_nothing():
    """A result of statuses with no chunk (proof_undecodable) costs no GPU: it proves nothing about its card."""
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = pool.claim("a")
        pool.now += 1
        cheap = pool.call(_items(1), job="order-eval-cheap")
        await asyncio.sleep(0)
        a2 = pool.claim("a")
        assert await pool.answer("a", a2, _status(a2, "proof_undecodable")) == "accepted"
        await cheap
        await pool.expire(a1)
        await pool.expire(pool.claim("b"))
        assert not poison.done(), "an answer with no measure was taken for a witness"
        c1 = pool.claim("c")                               # a measured witness, then a loss: now it parks
        await _witness(pool, "c")
        await pool.expire(c1)
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_the_losses_of_a_card_quarantined_since_no_longer_count():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = pool.claim("a")
        await _witness(pool, "a")
        await pool.expire(a1)                              # a: one witnessed loss of the poison
        other = pool.call(_items(1), job="order-eval-other")
        await asyncio.sleep(0)
        a2 = pool.claim("a")                               # benched from the poison: a loses another batch
        assert a2["job"] == "order-eval-other"
        await pool.expire(a2)
        assert pool.quarantined == ["a"]
        b1 = pool.claim("b")
        b2 = pool.claim("b")
        assert (b1["job"], b2["job"]) == (JOB, "order-eval-other")
        pool.now += 1
        assert await pool.answer("b", b2) == "accepted"    # b works: its loss is witnessed
        await _ended(other)
        await pool.expire(b1)
        assert not poison.done(), "a quarantined card's loss still counted"
        c1 = pool.claim("c")
        await _witness(pool, "c")
        await pool.expire(c1)
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_a_cards_losses_count_towards_two_parks_an_hour_at_most():
    async def go():
        pool = _Pool()
        for k in range(3):
            call = pool.call(_items(1))
            await asyncio.sleep(0)
            m = pool.claim("m")                            # m drops the batch, but scores elsewhere
            await _witness(pool, "m")
            await pool.expire(m)
            await pool.expire(pool.claim(f"o{k}"))         # a fresh card of another provider loses it too
            if k < 2:
                with pytest.raises(BatchParked):
                    await _ended(call)
                assert pool.strikes("m") == 0
            else:
                assert not call.done(), "a third park in the hour counted m's losses"
                assert pool.strikes("m") == 1
                call.cancel()
        last = pool.call(_items(1))
        await asyncio.sleep(0)
        await pool.expire(pool.claim("m"))                 # past its cap its losses only strike
        assert "m" in pool.quarantined
        last.cancel()
    _run(go)


def test_witnessed_losses_count_against_their_card():
    """A card that scores some leases and drops others is still set aside after two lost results (#493's card on the
    2026-11-11 DNS fault delivered some results): a witnessed loss strikes as any loss, and resets its clean streak."""
    async def go():
        pool = _Pool()
        for k in range(2):
            call = pool.call(_items(1), job=f"order-eval-{k}")
            await asyncio.sleep(0)
            lost = pool.claim("flaky")
            await _witness(pool, "flaky")
            await pool.expire(lost)
            call.cancel()
        assert pool.quarantined == ["flaky"] and pool.dispatcher.stats["witnessed_losses"] == 2
    _run(go)


# -- one card cannot hold a batch; the 2 s before a sweep ------------------------


def test_an_error_answer_benches_its_card_from_that_batch():
    """Before: an ITEM_ERROR answered in time struck nothing and the card took the same batch again at once, so one
    card held the oldest batch, and the queue behind it, for ever."""
    async def go():
        pool = _Pool()
        held, behind = pool.call(_items(1)), pool.call(_items(1))
        await asyncio.sleep(0)
        first = pool.claim("e")
        assert await pool.answer("e", first, _status(first, "error")) == "requeued"
        assert pool.strikes("e") == 0                      # its own fault, nobody judged on it: no strike
        second = pool.claim("e")
        assert second["batch"] != first["batch"], "the card took the batch it answered with errors again"
        assert await pool.answer("e", second) == "accepted"
        await _ended(behind)
        assert pool.claim("e") is None                     # still benched from the first
        pool.now += BENCH + 1
        again = pool.claim("e")
        assert again["batch"] == first["batch"] and await pool.answer("e", again) == "accepted"
        await _ended(held)
    _run(go)


def test_a_result_that_does_not_fit_in_time_is_a_lost_result():
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("f")
        with pytest.raises(LeaseRefused) as refused:
            await pool.answer("f", lease, _misfit(lease))
        assert refused.value.status == 422
        assert pool.strikes("f") == 1 and pool.claim("f") is None
        call.cancel()
    _run(go)


@pytest.mark.parametrize("answer", ["error", "misfit"])
def test_an_answer_between_expiry_and_sweep_counts_as_the_sweep_would(answer):
    """Before: anything posted in the 2 s between a lease's expiry and the next sweep removed the lease and struck
    nothing; the executor knows its expires_at exactly."""
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("w")
        pool.now = lease["expires_at"] + 0.05              # expired, not swept yet
        with pytest.raises(LeaseRefused) as refused:
            await pool.answer("w", lease, _status(lease, "error") if answer == "error" else _misfit(lease))
        assert refused.value.status == 410
        assert pool.strikes("w") == 1 and pool.claim("w") is None
        call.cancel()
    _run(go)


# -- score() ends at the first park and cancels the call's other batches -------


def test_the_first_park_ends_the_call_at_once_cancels_its_other_batches_and_keeps_what_was_scored():
    async def go():
        pool = _Pool()
        call = pool.call(_items(129))                      # batches of 64, 64 and 1 items
        await asyncio.sleep(0)
        assert await pool.answer("c", pool.claim("c")) == "accepted"     # batch 0 decided
        for eid in ("a", "b"):
            middle = pool.claim(eid)                       # an error on batch 1 benches it from that batch
            assert len(middle["items"]) == 64
            assert await pool.answer(eid, middle, _status(middle, "error")) == "requeued"
            last = pool.claim(eid)                         # batch 2, lost on two providers, once witnessed
            assert len(last["items"]) == 1
            if eid == "a":
                await _witness(pool, "a")
                await pool.expire(last)
        pool.now += 5
        slow = pool.claim("slow")                          # batch 1, out for a while
        assert len(slow["items"]) == 64
        await pool.expire(last)
        assert pool.now < slow["expires_at"], "the slow lease must still be out"
        with pytest.raises(BatchParked) as parked:
            await _ended(call)        # at once: batch 1 is still leased
        assert parked.value.indexes == [128]
        assert sorted(parked.value.scored) == list(range(64)) and parked.value.scored[0][2] == ("c",)
        assert pool.dispatcher.lease_of(slow["lease_id"]).batch.future.cancelled()
        assert await pool.answer("slow", slow) == "unneeded"
        assert pool.dispatcher.pending() == 0
    _run(go)


# -- a parked batch that gets a scored answer is unparked ----------------------


def test_a_scored_late_answer_unparks_the_batch_and_takes_its_loss_back():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = pool.claim("a")
        await _witness(pool, "a")
        await pool.expire(a1)
        s1 = pool.claim("s")                               # slow: its lease expires while it scores
        await pool.expire(s1)
        with pytest.raises(BatchParked) as parked:
            await _ended(poison)
        assert pool.strikes("s") == 1
        pool.dispatcher._expired[s1["lease_id"]].full_length = False   # an adaptive lease, not the longest one
        with pytest.raises(LeaseRefused):
            await pool.answer("s", s1)                     # late, scored: the batch can be answered after all
        assert pool.unparked == [(JOB, parked.value.batch_ids[0])]
        assert pool.strikes("s") == 0 and pool.dispatcher.stats["unparked"] == 1
    _run(go)


# -- parked records: per record, a cool-down, a retry alone ---------------------


def _parked_error(message, indexes, batch_ids):
    """``BatchParked`` naming its batch's items; main's takes a message alone (its park took the whole call)."""
    try:
        return BatchParked(message, indexes=indexes, batch_ids=batch_ids)
    except TypeError:
        return BatchParked(message)


class _Remote:
    """A remote dispatcher for the auditor: records are told apart by the last token of their completion. A record
    marked ``lost`` has its batch parked; one marked ``unreadable`` has an item no executor can read."""

    def __init__(self):
        self.calls, self.unreadable, self.lost, self.unpark = [], set(), set(), None

    def connected(self):
        return True

    def subscribe(self, listener):
        pass

    def subscribe_unparked(self, listener):
        self.unpark = listener

    def quarantined(self):
        return set()

    async def score(self, items):
        marks = [item["tokens"][-1] for item in items]
        self.calls.append(marks)
        bad = {k: "spans: at most 64 items" for k, mark in enumerate(marks) if mark in self.unreadable}
        if bad:
            raise ItemsUnreadable(bad)
        lost = [k for k, mark in enumerate(marks) if mark in self.lost]
        if lost:
            raise _parked_error("its leases were lost", lost, [42])
        return [("ok", (ChunkResult(1, 1.0, 1.0),), ("x",)) for _ in items]


class _Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class _Records:
    def __init__(self, submissions):
        self.submissions, self.verdicts, self.voided = dict(submissions), {}, {}

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def list_submission_ids(self, job_id):
        return sorted(self.submissions)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def write_verdict(self, job_id, sid, verdict):
        if sid in self.verdicts:
            return False
        self.verdicts[sid] = verdict
        return True

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def write_voided(self, job_id, sid, document):
        self.voided[sid] = document

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return {}, None


def _order_auditor(records, clock, remote):
    """An order's auditor at q = 1: every record is audited."""
    auditor = eval_auditor(job_id=JOB, records=records, tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                           remote=remote, clock=clock)
    auditor._seeded = True
    return auditor


def test_only_the_parked_batchs_records_are_parked_and_the_rest_of_the_pass_is_judged():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({sid: _record(sid, mark, received_at=49_000.0)
                            for sid, mark in (("a", 3), ("p", 4), ("c", 5))})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        await auditor.judge_many(["a", "p", "c"])
        assert set(records.verdicts) == {"a", "c"} and auditor.parked_ids == {"p"}
    _run(go)


def test_a_parked_record_is_tried_again_alone_after_a_cool_down_that_doubles():
    """Before: a parked record was never tried again (until an operator acted, or a restart)."""
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0), "d": _record("d", 6, received_at=49_000.0)})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        await auditor.judge_many(["p"])
        assert auditor.parked_ids == {"p"}
        remote.calls.clear()
        clock.now += RETRY - 1
        await auditor.judge_many(["p"])
        assert remote.calls == [], "a parked record was tried again before its cool-down ended"
        clock.now += 1
        await auditor.judge_many(["p", "d"])
        assert sorted(remote.calls) == [[4], [6]]          # p tried again, alone: it parks nobody else
        assert "d" in records.verdicts and "p" not in records.verdicts
        remote.calls.clear()
        clock.now += RETRY                                 # parked again: its cool-down doubled
        await auditor.judge_many(["p"])
        assert remote.calls == []
        clock.now += RETRY
        remote.lost.clear()
        await auditor.judge_many(["p"])
        assert records.verdicts["p"]["passed"] and auditor.parked_ids == set()
    _run(go)


def test_a_batch_unparked_by_a_scored_answer_is_tried_again_at_once():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0)})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        await auditor.judge_many(["p"])
        assert auditor.parked_ids == {"p"} and remote.unpark is not None, "the auditor hears of no unpark"
        remote.lost.clear()
        remote.unpark(42)                                  # a card answered its batch: it can be scored after all
        await auditor.judge_many(["p"])
        assert records.verdicts["p"]["passed"] and auditor.parked_ids == set()
    _run(go)


def test_parked_records_are_no_validator_errors_however_often_they_park():
    """Before: each re-audit of a record already parked (a sibling at q < 1, after a restart) counted one validator
    error, and five in a row halted the order's auditor."""
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        auditor = _order_auditor(None, clock, remote)
        remote.lost.add(4)
        for _ in range(6):
            results, failed = await auditor._audit_records(["p"], [_record("p", 4)], {})
            assert results == [None] and failed == set()
            assert auditor._validator_errors == 0
            clock.now += 10 * RETRY
    _run(go)


def test_audit_ahead_parks_for_good_and_never_sends_a_parked_record_again():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0), "q": _record("q", 5, received_at=49_100.0)})
        auditor = _order_auditor(records, clock, remote)
        for sid, record in records.submissions.items():
            auditor._remember(sid, record)
        remote.lost.add(4)
        assert auditor._ahead_candidates(32) == ["p", "q"]
        await auditor._audit_ahead_batch(["p", "q"])
        assert auditor.parked_ids == {"p"} and "q" in auditor._ahead
        assert auditor._ahead_candidates(32) == []         # p parked, q audited ahead: nothing to send
    _run(go)


def test_a_parked_re_audit_keeps_its_passes_listed_and_tries_them_again():
    """Before: a re-audit call with a parked batch dropped every pass of it, standing and paid, in silence."""
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"s1": _record("s1", 4), "s2": _record("s2", 5)})
        auditor = _order_auditor(records, clock, remote)
        auditor._late_reaudits["x"] = asyncio.get_running_loop().create_future()   # no background retry here
        auditor._remote_scored["x"] = ["s1", "s2"]
        remote.lost.add(4)
        assert await auditor.reaudit_executor("x") == []
        assert auditor._remote_scored.get("x") == ["s1"]   # still listed; s2 re-audited, and it stands
        calls = len(remote.calls)
        await auditor.reaudit_executor("x")                # cooling down: not tried yet, still listed
        assert len(remote.calls) == calls and auditor._remote_scored["x"] == ["s1"]
        clock.now += RETRY
        remote.lost.clear()
        assert await auditor.reaudit_executor("x") == []
        assert not auditor._remote_scored.get("x")
        auditor._late_reaudits["x"].cancel()
    _run(go)


# -- leaving the close two periods after the receipt, at q < 1 -----------------


Q, HOLD, T0, HK = 0.5, 1000.0, 10_000.0, "5Hot"
RAND = "ab" * 32


class _States:
    def __init__(self, initial=None):
        self.states = dict(initial or {})

    async def get(self, hotkey):
        return self.states.get(hotkey, MinerState())

    async def update(self, hotkey, change, attempts=5):
        self.states[hotkey] = change(self.states.get(hotkey, MinerState()))
        return self.states[hotkey]

    async def update_many(self, changes, attempts=5):
        return {hotkey: await self.update(hotkey, change, attempts) for hotkey, change in changes.items()}


def _ids(want_drawn, n, start=0):
    found, i = [], start
    while len(found) < n:
        sid = f"{i:064x}"
        if drawn(RAND, sid, Q) == want_drawn:
            found.append(sid)
        i += 1
    return found


def _sampled_auditor(records, clock, remote):
    auditor = eval_auditor(job_id=JOB, records=records, tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                           params=AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10),
                           miner_states=_States({HK: MinerState(audited_passed=100)}), beacon=lambda r: RAND,
                           round_at=lambda t: int(t // 3) + 1, clock=clock, accept_slack_seconds=0.0,
                           remote=remote)
    # The corpus rule, as `wire_order_judge` sets it for an order priced on its prompts alone: an undrawn record is
    # paid unaudited once its hold is over and every sibling that could catch it is decided.
    auditor.stake_at_receipt = False
    auditor._seeded = True
    return auditor


def test_a_parked_record_leaves_the_close_and_its_siblings_two_periods_after_its_receipt_at_q_below_one():
    """At q < 1 a parked record stayed a sibling of its operator's later unaudited passes: re-audited at every pass,
    parked again, holding them, and through them the close, for ever (#589's review). It holds them until it is tried
    again, then two periods after its receipt at most; it is not re-scored while it cools down."""
    async def go():
        p = _ids(True, 1)[0]                               # drawn: audited, and no executor can read it
        x = _ids(False, 1, start=7000)[0]                  # undrawn: paid unaudited once its hold is over
        steady = _ids(False, 2, start=5000)                # keep the hotkey above 1/q at x's hold end
        records = _Records({p: _record(p, 4, received_at=T0), x: _record(x, 5, received_at=T0 + 10),
                            **{s: _record(s, 6, received_at=T0 + 500) for s in steady}})
        clock, remote = _Clock(T0 + 20), _Remote()
        remote.lost.add(4)
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p, x])
        assert auditor.parked_ids == {p}
        scored = len(remote.calls)
        clock.now = T0 + HOLD + 20                         # x's hold is over; p cools down
        await auditor.judge_many([p, x, *steady])
        assert x not in records.verdicts                   # p could still catch it
        assert len(remote.calls) == scored, "a parked record was scored again while it cools down"
        assert auditor.oldest_pending_received_at() == T0
        clock.now = T0 + GIVE_UP + 1
        later = _ids(False, 2, start=9000)                 # the miner keeps submitting: its hotkey stays sampled
        for sid in later:
            records.submissions[sid] = _record(sid, 7, received_at=clock.now - 500)
        await auditor.judge_many([p, x, *steady, *later])  # p is tried again, parks again: x is paid all the same
        assert x in records.verdicts and records.verdicts[x]["passed"] and not records.verdicts[x]["audited"]
        assert p not in records.verdicts and auditor.parked_ids == {p}
        oldest = auditor.oldest_pending_received_at()
        assert oldest is None or oldest > T0 + 10
    _run(go)


def test_the_give_up_counts_from_the_receipt_so_a_restart_does_not_start_it_again():
    async def go():
        p = _ids(True, 1)[0]
        records = _Records({p: _record(p, 4, received_at=T0)})
        clock, remote = _Clock(T0 + GIVE_UP + 60), _Remote()   # a restart long after p's receipt
        remote.lost.add(4)
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p])
        assert auditor.parked_ids == {p}
        assert auditor.oldest_pending_received_at() is None    # left out at once: its two periods are over
    _run(go)


def test_past_the_give_up_the_parked_operators_verdicts_of_its_period_wait_for_its_decision():
    """The stake guarantee (owner, 2026-11-26): once a parked record no longer holds its period's close, its operator's
    verdicts of that period stay unsettled until it is decided, so its failure could still forfeit them; every other
    operator is paid on time. What is held is paid late, within what is left of the period's budget."""
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler

    async def go():
        p = _ids(True, 1)[0]
        z = _ids(True, 1, start=3000)[0]                   # an audited pass of p's operator, before p, same period
        o = _ids(True, 1, start=6000)[0]                   # another operator's audited pass, same period
        assert cp.period_of(T0 - 30) == cp.period_of(T0)
        records = _Records({p: _record(p, 4, received_at=T0), z: _record(z, 5, received_at=T0 - 30),
                            o: _record(o, 6, received_at=T0 - 20, hotkey="5Other")})
        clock, remote = _Clock(T0 + 20), _Remote()
        remote.lost.add(4)
        auditor = _sampled_auditor(records, clock, remote)
        auditor._miner_states.states["5Other"] = MinerState(audited_passed=100)
        await auditor.judge_many([z, o, p])
        assert set(records.verdicts) == {z, o} and auditor.parked_ids == {p}

        class Archives:
            def __init__(self):
                self.docs = {}

            async def write(self, task_id, work, entry, document):
                self.docs[(work, entry)] = document

            async def read(self, task_id, work, entry):
                return self.docs.get((work, entry))

            async def list(self, task_id):
                return sorted(self.docs)

        state = {}

        async def write_settlement(job_id, document, etag):
            state.clear()
            state.update(document)
            return "etag"

        async def read_settlement(job_id):
            return dict(state), "etag"

        records.write_settlement, records.read_settlement = write_settlement, read_settlement
        archives = Archives()
        hold = getattr(auditor, "settle_hold", None)
        settler = CorpusPeriodSettler(task_id=JOB, job_id=JOB, cap=1.0, records=records, archives=archives,
                                      oldest_pending=auditor.oldest_pending_received_at, clock=clock,
                                      max_price=1e-3, **({"hold": hold} if hold is not None else {}))
        clock.now = T0 + GIVE_UP + 1 + cp.PERIOD_SECONDS
        await settler.settle_once()
        assert o in state["settled"] and z not in state["settled"]
        paid = {hk for doc in archives.docs.values() for hk in doc["rewards_by_hotkey"]}
        assert paid == {"5Other"}
        remote.lost.clear()                                # p decided at last: it passes
        clock.now += 10 * RETRY
        await auditor.judge_many([p])
        assert p in records.verdicts and auditor.parked_ids == set()
        await settler.settle_once()
        assert z in state["settled"] and p in state["settled"]
        late = [doc for doc in archives.docs.values() if HK in doc["rewards_by_hotkey"]]
        assert late and late[0]["rewards_by_hotkey"][HK] > 0  # a priced period: paid within its budget
    _run(go)


# -- the split judge's stream --------------------------------------------------


def test_the_stream_carries_unreadable_items_partial_parks_and_unparks():
    from reliquary.validator.order_judge import DispatchClient, DispatchServer

    class Dispatcher(PairedAuditDispatcher):
        outcome = "unreadable"

        async def score(self, model, proof, items, job_id="", judge=None, batches=None):
            if self.outcome == "unreadable":
                raise ItemsUnreadable({1: "spans: at most 64 items"})
            raise BatchParked("lost", indexes=[1], batch_ids=[7],
                              scored={0: ("ok", (ChunkResult(1, 0.5, 0.25),), ("e1",))})

    async def go():
        dispatcher = Dispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        w = SimpleNamespace(job=SimpleNamespace(checkpoint_repo=MODEL[0], checkpoint_revision=MODEL[1],
                                                sampling=SimpleNamespace(draw=None)), proof=PROOF)
        with tempfile.TemporaryDirectory(prefix="rq") as run_dir:
            path = Path(run_dir) / "dispatch.sock"
            server = DispatchServer(dispatcher, path, lambda job_id: w if job_id == JOB else None)
            client = DispatchClient(path, retry_seconds=0.02, renotify_seconds=0.2)
            await server.start()
            reader = asyncio.ensure_future(client.run())
            view = client.view(JOB)
            unparked = []
            view.subscribe_unparked(unparked.append)
            try:
                with pytest.raises(ItemsUnreadable) as refused:
                    await view.score(_items(2))
                assert refused.value.reasons == {1: "spans: at most 64 items"}
                dispatcher.outcome = "parked"
                with pytest.raises(BatchParked) as parked:
                    await view.score(_items(2))
                assert parked.value.indexes == [1] and parked.value.batch_ids == [7]
                assert parked.value.scored == {0: ("ok", (ChunkResult(1, 0.5, 0.25),), ("e1",))}
                dispatcher._unpark(SimpleNamespace(parked=True, job_id=JOB, id=7))
                for _ in range(100):
                    if unparked:
                        break
                    await asyncio.sleep(0.01)
                assert unparked == [7]
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
                await server.close()

    asyncio.run(asyncio.wait_for(go(), timeout=30))
