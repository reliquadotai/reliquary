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
        for _ in range(400):                               # the records are prepared in a thread: wait for the batch
            if pool.dispatcher.pending() or outcomes.done():
                break
            await asyncio.sleep(0.005)
        lease = pool.claim("a")
        assert lease is not None and len(lease["items"]) == 2, "the unreadable record went to a card"
        assert await pool.answer("a", lease) == "accepted"
        first, parked, third = await outcomes
        assert first["passed"] and third["passed"] and first["scored_by"] == ["a"]
        assert not isinstance(parked, dict) and "65536" in parked
        assert auditor.parked_ids == {"s2"}
    _run(go)


# -- parking only on evidence against the batch --------------------------------


async def _settled():
    """A few loop turns: a call whose batch was parked by a sweep has ended by then."""
    for _ in range(5):
        await asyncio.sleep(0)


async def _lose(pool, eid, *, witnessed=True, **placement):
    """``eid`` takes the oldest batch it may and loses it (its lease expires); witnessed when it scored another
    lease meanwhile. Returns the lost lease."""
    lease = pool.claim(eid, **placement)
    assert lease is not None
    if witnessed:
        await _witness(pool, eid, **placement)
    await pool.expire(lease)
    return lease


def test_a_batch_lost_by_working_cards_on_two_providers_is_parked_and_their_strikes_taken_back():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = await _lose(pool, "a")
        await _settled()
        assert pool.strikes("a") == 1 and not poison.done()    # struck, as #493 does; the batch still queued
        b1 = await _lose(pool, "b")                        # a working card of another provider loses it too
        assert b1["batch"] == a1["batch"]
        with pytest.raises(BatchParked) as parked:
            await _ended(poison)
        assert parked.value.indexes == [0] and parked.value.batch_ids == [a1["batch"]]
        assert pool.strikes("a") == 0 and pool.strikes("b") == 0   # they work, the batch does not
        status = pool.dispatcher.status()
        reason = status["jobs"][JOB]["reasons"][-1]
        assert "lost by working cards on 2 providers: a (prov-a), b (prov-b)" in reason
        assert status["parks_by_provider"] == {"prov-a": 1, "prov-b": 1}
        assert status["stats"]["unanswered_parked"] == 1 and status["jobs"][JOB]["parked_batches"] == 1
        assert pool.quarantined == []
    _run(go)


def test_an_unwitnessed_loss_is_no_evidence_however_old_or_fresh():
    """A loss by a card that scored nothing since (dead, slow, or long gone) proves nothing against the batch: it
    cannot complete another card's witnessed loss (the review of 8349fe16b borrowed such a loss to park)."""
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        h1 = await _lose(pool, "h", witnessed=False)       # an old loss, of a card that answered nothing
        pool.now += 11_000
        m1 = await _lose(pool, "m")                        # a working card of another provider, hours later
        assert m1["batch"] == h1["batch"]
        await _settled()
        assert not poison.done(), "a witnessed loss borrowed an unwitnessed one to park"
        assert pool.strikes("m") == 1
        await _lose(pool, "c")                             # a second working card: now the evidence holds
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_losses_on_one_provider_never_park_a_batch_and_its_cards_take_other_work():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        for eid in ("a", "b"):                             # two working cards of one provider, on two hosts
            await _lose(pool, eid, provider="prov-x")
            # Benched from the batch it lost: it takes other work instead of losing that one again at once.
            assert pool.claim(eid, provider="prov-x") is None
        await _settled()
        assert not poison.done(), "one provider's losses parked a batch"
        nxt = pool.call(_items(1), job="order-eval-next")
        await asyncio.sleep(0)
        lease = pool.claim("b", provider="prov-x")
        assert lease["job"] == "order-eval-next" and await pool.answer("b", lease) == "accepted"
        await _ended(nxt)
        await _lose(pool, "c")                             # a third provider: the evidence that was missing
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_cards_that_answer_nothing_never_park_a_batch_and_are_still_set_aside_as_before():
    """#493: a card that answers nothing is set aside after two lost results, as soon as before (it is not benched).
    None of its losses is witnessed, so a dead pool parks nothing and the first working card scores the batches."""
    async def go():
        pool = _Pool()
        calls = [pool.call(_items(1)) for _ in range(2)]
        await asyncio.sleep(0)
        for eid in ("dead1", "dead2"):
            first = pool.claim(eid)
            await pool.expire(first)
            second = pool.claim(eid)
            assert second["batch"] == first["batch"]       # not benched: it is a lost card, not a lost batch
            await pool.expire(second)
        assert pool.quarantined == ["dead1", "dead2"]
        await _settled()
        assert not any(call.done() for call in calls)
        for _ in range(2):
            assert await pool.answer("good", pool.claim("good")) == "accepted"
        await _ended(asyncio.gather(*calls))
    _run(go)


def test_an_answer_with_no_measure_witnesses_nothing_one_partly_measured_does():
    """A result of statuses with no chunk (proof_undecodable) costs no GPU: it proves nothing about its card. One
    measured item beside an undecodable proof (the miner's fault) does."""
    from reliquary.validator.corpus_audit_protocol import AuditResult as Result

    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        for eid, answer in (("a", lambda lease: _status(lease, "proof_undecodable")),
                            ("b", lambda lease: Result.model_validate({"scores": [
                                {"status": "ok", "chunks": [[1, 1.0, 1.0]]},
                                {"status": "proof_undecodable", "chunks": []}]}))):
            lost = pool.claim(eid)
            pool.now += 1
            cheap = pool.call(_items(2), job=f"order-eval-cheap-{eid}")
            await asyncio.sleep(0)
            other = pool.claim(eid)
            assert await pool.answer(eid, other, answer(other)) == "accepted"
            await _ended(cheap)
            await pool.expire(lost)
        await _settled()
        assert not poison.done(), "an answer with no measure was taken for a witness"
        assert pool.dispatcher.stats["witnessed_losses"] == 1  # b's only
        await _lose(pool, "c")                             # a second witnessed provider: b's loss and c's park it
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_a_measured_answer_witnesses_only_later_leases_and_only_for_an_undecided_batch():
    """A card that scored once and then died witnesses nothing; nor does an answer to a batch already decided or
    cancelled (it costs nothing: nobody reads it)."""
    async def go():
        pool = _Pool()
        await _witness(pool, "a")                          # a scores before it takes the poison
        pool.now += 1
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        await _lose(pool, "a", witnessed=False)
        gone = pool.call(_items(1), job="order-eval-gone")
        await asyncio.sleep(0)
        b1 = pool.claim("b")
        b2 = pool.claim("b")
        assert (b1["job"], b2["job"]) == (JOB, "order-eval-gone")
        gone.cancel()
        await asyncio.sleep(0)
        pool.now += 1
        assert await pool.answer("b", b2) == "unneeded"    # measured, but for a batch nobody waits for
        await pool.expire(b1)
        await _settled()
        assert not poison.done() and pool.dispatcher.stats["witnessed_losses"] == 0
        poison.cancel()
    _run(go)


def test_the_losses_of_a_card_quarantined_since_no_longer_count():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = await _lose(pool, "a")                        # a: one witnessed loss of the poison
        other = pool.call(_items(1), job="order-eval-other")
        await asyncio.sleep(0)
        a2 = pool.claim("a")                               # benched from the poison: a loses another batch
        assert a2["job"] == "order-eval-other"
        await pool.expire(a2)
        assert pool.quarantined == ["a"]
        b1 = pool.claim("b")
        b2 = pool.claim("b")
        assert (b1["batch"], b2["job"]) == (a1["batch"], "order-eval-other")
        pool.now += 1
        assert await pool.answer("b", b2) == "accepted"    # b works: its loss is witnessed
        await _ended(other)
        await pool.expire(b1)
        await _settled()
        assert not poison.done(), "a quarantined card's loss still counted"
        await _lose(pool, "c")
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_a_providers_losses_count_towards_two_parks_an_hour_at_most_whatever_its_card():
    """The cap is the provider's: a card rented again on it (a fresh executor id) does not start a fresh cap."""
    async def go():
        pool = _Pool()
        for k in range(3):
            call = pool.call(_items(1))
            await asyncio.sleep(0)
            await _lose(pool, f"m{k}", provider="prov-m")  # each time a fresh card of the same provider
            await _lose(pool, f"o{k}")                     # and a working card of another provider
            if k < 2:
                with pytest.raises(BatchParked):
                    await _ended(call)
                assert pool.strikes(f"m{k}") == 0
            else:
                await _settled()
                assert not call.done(), "a third park in the hour counted prov-m's losses"
                assert pool.strikes("m2") == 1
                call.cancel()
        assert pool.dispatcher.status()["parks_by_provider"]["prov-m"] == 2
        pool.now += 3601                                   # an hour later its losses count again
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        await _lose(pool, "m3", provider="prov-m")
        await _lose(pool, "o3")
        with pytest.raises(BatchParked):
            await _ended(call)
    _run(go)


def test_a_park_takes_back_only_the_strikes_its_batch_gave_and_none_a_clean_streak_cleared():
    async def go():
        pool = _Pool()
        pool.dispatcher._strikes_limit = 4
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        e1 = pool.claim("e")
        await _witness(pool, "e")
        with pytest.raises(LeaseRefused):
            await pool.answer("e", e1, _misfit(e1))        # a witnessed lost result: strike 1
        for _ in range(ec.CLEAN_STREAK):                   # a clean streak clears it
            await _witness(pool, "e")
        assert pool.strikes("e") == 0
        other = pool.call(_items(1), job="order-eval-other")
        await asyncio.sleep(0)
        e2 = pool.claim("e")
        assert e2["job"] == "order-eval-other"
        await pool.expire(e2)                              # a new loss elsewhere: strike 1
        assert pool.strikes("e") == 1
        other.cancel()
        await asyncio.sleep(0)
        await _lose(pool, "b")                             # a working card of another provider: parked
        with pytest.raises(BatchParked):
            await _ended(poison)
        assert pool.strikes("e") == 1, "the park took back a strike the clean streak had already cleared"
    _run(go)


def test_witnessed_losses_count_against_their_card_and_reset_its_clean_streak():
    """A card that scores some leases and drops others is still set aside after two lost results (#493's card on the
    2026-11-11 DNS fault delivered some results): a witnessed loss strikes as any loss, and resets its clean streak."""
    async def go():
        pool = _Pool()
        pool.dispatcher._strikes_limit = 3
        for k in range(3):
            call = pool.call(_items(1), job=f"order-eval-{k}")
            await asyncio.sleep(0)
            await _lose(pool, "flaky")
            call.cancel()
            for _ in range(k):                             # fewer than a clean streak between losses
                await _witness(pool, "flaky")
        assert pool.quarantined == ["flaky"] and pool.dispatcher.stats["witnessed_losses"] == 3
    _run(go)


def test_two_scorers_need_witnessed_losses_on_a_third_provider():
    async def go():
        pool = _Pool(scorers=2)
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1, b1 = pool.claim("a"), pool.claim("b")
        assert a1["batch"] == b1["batch"]
        witnesses = []

        async def witness(eid):                            # a measured answer to an undecided batch of its own
            pool.now += 1
            witnesses.append(pool.call(_items(1), job=f"order-eval-witness-{eid}"))
            await asyncio.sleep(0)
            lease = pool.claim(eid)                        # its own, or another's still needing a second scorer
            assert lease["job"].startswith("order-eval-witness-")
            pool.now += 1
            assert await pool.answer(eid, lease) == "accepted"

        await witness("a")
        await witness("b")
        await pool.expire(a1)
        await _settled()
        assert not poison.done(), "two providers parked a batch that needs two scorers"
        c1 = pool.claim("c")
        await witness("c")
        await pool.expire(c1)
        with pytest.raises(BatchParked):
            await _ended(poison)
        for call in witnesses:
            call.cancel()
    _run(go)


def test_a_witnessed_loss_older_than_the_park_window_is_no_evidence():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        await _lose(pool, "h")                             # an honest card's witnessed loss ...
        pool.now += getattr(ec, "PARK_WINDOW_SECONDS", 3600.0) + 60   # ... over an hour old
        await _lose(pool, "m")
        await _settled()
        assert not poison.done(), "a stale witnessed loss was borrowed to park"
        await _lose(pool, "c")
        with pytest.raises(BatchParked):
            await _ended(poison)
    _run(go)


def test_a_providers_park_cap_holds_for_the_whole_hour():
    async def go():
        pool = _Pool()
        for k in range(2):
            call = pool.call(_items(1))
            await asyncio.sleep(0)
            await _lose(pool, f"m{k}", provider="prov-m")
            await _lose(pool, f"o{k}")
            with pytest.raises(BatchParked):
                await _ended(call)
        pool.now += 1860                                   # 31 min: still inside the hour
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        await _lose(pool, "m2", provider="prov-m")
        await _lose(pool, "o2")
        await _settled()
        assert not call.done(), "the park cap lapsed before its hour"
        call.cancel()
    _run(go)


def test_a_scored_answer_to_the_longest_lease_between_expiry_and_sweep_is_no_evidence_against_its_batch():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        await _lose(pool, "a")                             # a witnessed loss on prov-a
        lease = pool.claim("w")
        assert lease is not None and pool.dispatcher.lease_of(lease["lease_id"]).full_length
        await _witness(pool, "w")                          # w works: its loss would be witnessed
        pool.now = lease["expires_at"] + 0.05              # expired, not swept yet
        with pytest.raises(LeaseRefused):
            await pool.answer("w", lease)                  # scored, late, to the longest lease
        await _settled()
        assert not poison.done(), "an answered lease counted as evidence and parked its batch"
        assert pool.dispatcher.status()["parks_by_provider"] == {} and pool.strikes("w") == 1
        poison.cancel()
    _run(go)


@pytest.mark.parametrize("full_length", [False, True])
def test_only_a_measured_late_answer_to_a_shorter_lease_lifts_the_bench(full_length):
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("w")
        await _witness(pool, "w")
        pool.dispatcher.lease_of(lease["lease_id"]).full_length = full_length
        await pool.expire(lease)                           # a witnessed loss: benched from the batch
        answer = _status(lease, "proof_undecodable") if not full_length else None
        with pytest.raises(LeaseRefused):
            await pool.answer("w", lease, answer)          # late: unmeasured (shorter lease) or measured (longest)
        assert pool.claim("w") is None, "the late answer lifted the card's bench"
        call.cancel()
    _run(go)


def test_an_error_answered_after_the_sweep_benches_its_card_as_before_it():
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("w")
        pool.dispatcher.lease_of(lease["lease_id"]).full_length = False
        await pool.expire(lease)                           # unwitnessed: not benched by the loss
        with pytest.raises(LeaseRefused):
            await pool.answer("w", lease, _status(lease, "error"))
        assert pool.strikes("w") == 1 and pool.claim("w") is None
        call.cancel()
    _run(go)


def test_an_unreadable_call_counts_as_a_park():
    async def go():
        pool = _Pool()
        items = _items(2)
        items[1] = {**items[1], "tokens": list(range(1, 65_600))}
        with pytest.raises(ItemsUnreadable):
            await pool.dispatcher.score(MODEL, PROOF, items, JOB)
        status = pool.dispatcher.status()
        assert status["jobs"][JOB]["parked_batches"] == 1 and status["stats"]["parked"] == 1
    _run(go)


# -- one card cannot hold a batch; the 2 s before a sweep ------------------------


def test_an_error_answer_benches_its_card_from_that_batch_doubling_up_to_an_hour():
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
        for seconds in (300, 600, 1200, 2400, 3600, 3600):
            pool.now += seconds - 1
            assert pool.claim("e") is None, f"benched less than {seconds} s"
            pool.now += 1.5
            again = pool.claim("e")
            assert again is not None and again["batch"] == first["batch"]
            if seconds == 3600 and pool.dispatcher.stats["benched"] >= 6:
                assert await pool.answer("e", again) == "accepted"
                break
            assert await pool.answer("e", again, _status(again, "error")) == "requeued"
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
@pytest.mark.parametrize("full_length", [True, False])
def test_an_answer_between_expiry_and_sweep_counts_as_the_sweep_would(answer, full_length):
    """Before: anything posted in the 2 s between a lease's expiry and the next sweep removed the lease and struck
    nothing; the executor knows its expires_at exactly."""
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("w")
        pool.dispatcher.lease_of(lease["lease_id"]).full_length = full_length
        pool.now = lease["expires_at"] + 0.05              # expired, not swept yet
        with pytest.raises(LeaseRefused) as refused:
            await pool.answer("w", lease, _status(lease, "error") if answer == "error" else _misfit(lease))
        assert refused.value.status == 410
        assert pool.strikes("w") == 1 and pool.claim("w") is None
        call.cancel()
    _run(go)


def test_a_scored_answer_between_expiry_and_sweep_to_the_longest_lease_stays_a_lost_result():
    async def go():
        pool = _Pool()
        call = pool.call(_items(1))
        await asyncio.sleep(0)
        lease = pool.claim("w")
        assert pool.dispatcher.lease_of(lease["lease_id"]).full_length
        pool.now = lease["expires_at"] + 0.05
        with pytest.raises(LeaseRefused):
            await pool.answer("w", lease)
        assert pool.strikes("w") == 1
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
            last = pool.claim(eid)                         # batch 2: lost by working cards on two providers
            assert len(last["items"]) == 1
            await _witness(pool, eid)
            if eid == "a":
                await pool.expire(last)
        pool.now += 5
        slow = pool.claim("slow")                          # batch 1, out for a while (a card slower than the rest)
        assert len(slow["items"]) == 64
        pool.dispatcher.lease_of(slow["lease_id"]).expires_at = slow["expires_at"] = pool.now + 3000
        await pool.expire(last)
        assert pool.now < slow["expires_at"], "the slow lease must still be out"
        with pytest.raises(BatchParked) as parked:
            await _ended(call)                             # at once: batch 1 is still leased
        assert parked.value.indexes == [128]
        assert sorted(parked.value.scored) == list(range(64)) and parked.value.scored[0][2] == ("c",)
        assert pool.dispatcher.lease_of(slow["lease_id"]).batch.future.cancelled()
        assert await pool.answer("slow", slow) == "unneeded"
        assert pool.dispatcher.pending() == 0
    _run(go)


# -- a parked batch that gets a measured answer is unparked --------------------


async def _parked_with_a_late_answerer(pool, *, full_length):
    """A batch parked on the witnessed losses of a and s; s still holds an expired lease of it, answerable late."""
    poison = pool.call(_items(1))
    await asyncio.sleep(0)
    await _lose(pool, "a")
    s1 = pool.claim("s")
    await _witness(pool, "s")
    await pool.expire(s1)
    with pytest.raises(BatchParked) as parked:
        await _ended(poison)
    pool.dispatcher._expired[s1["lease_id"]].full_length = full_length
    return s1, parked.value.batch_ids[0]


@pytest.mark.parametrize("full_length", [True, False])
def test_a_measured_late_answer_unparks_the_batch(full_length):
    async def go():
        pool = _Pool()
        s1, batch_id = await _parked_with_a_late_answerer(pool, full_length=full_length)
        with pytest.raises(LeaseRefused):
            await pool.answer("s", s1)                     # late, measured: the batch can be answered after all
        assert pool.unparked == [(JOB, batch_id)] and pool.dispatcher.stats["unparked"] == 1
    _run(go)


def test_a_late_answer_with_no_measure_unparks_nothing():
    async def go():
        pool = _Pool()
        s1, _ = await _parked_with_a_late_answerer(pool, full_length=False)
        with pytest.raises(LeaseRefused):
            await pool.answer("s", s1, _status(s1, "proof_undecodable"))
        assert pool.unparked == [] and pool.dispatcher.stats["unparked"] == 0
    _run(go)


def test_a_measured_late_answer_takes_its_cards_loss_out_of_the_evidence():
    async def go():
        pool = _Pool()
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        s1 = pool.claim("s")
        await _witness(pool, "s")
        await pool.expire(s1)                              # s: a witnessed loss
        with pytest.raises(LeaseRefused):
            await pool.answer("s", s1)                     # ... answered late and measured: slow, not lost
        await _lose(pool, "a")                             # a witnessed loss on a second provider
        await _settled()
        assert not poison.done(), "a card that answered the batch still counted as evidence against it"
        poison.cancel()
    _run(go)


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
        self.calls, self.unreadable, self.lost, self.failing, self.unpark = [], set(), set(), set(), None

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
        return [("ok", (ChunkResult(500 if mark in self.failing else 1, 1.0, 1.0),), ("x",)) for mark in marks]


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


async def _retried(auditor):
    """The parked records' retries, which run alone in the background, ended."""
    for _ in range(50):
        tasks = list(getattr(auditor, "_retries", {}).values())
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=1)


async def _judged_after_retry(auditor, sids):
    """A pass that starts the retries due, then the pass that takes what they decided."""
    await auditor.judge_many(sids)
    await _retried(auditor)
    await auditor.judge_many(sids)


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
        await _retried(auditor)
        assert sorted(remote.calls) == [[4], [6]]          # p tried again, alone: it parks nobody else
        assert "d" in records.verdicts and "p" not in records.verdicts
        remote.calls.clear()
        clock.now += RETRY                                 # parked again: its cool-down doubled
        await auditor.judge_many(["p"])
        assert remote.calls == []
        clock.now += RETRY
        remote.lost.clear()
        await _judged_after_retry(auditor, ["p"])
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
        await _judged_after_retry(auditor, ["p"])
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


def test_the_cool_down_doubles_up_to_four_hours():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0)})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        await auditor.judge_many(["p"])
        for backoff in (1800, 3600, 7200, 14400, 14400):
            remote.calls.clear()
            clock.now += backoff - 1
            await auditor.judge_many(["p"])
            assert remote.calls == [], f"tried before {backoff} s"
            clock.now += 1
            await auditor.judge_many(["p"])
            await _retried(auditor)
            assert remote.calls == [[4]], f"not tried at {backoff} s"
    _run(go)


class _RaceRemote(_Remote):
    """Parks the record marked 4 with its batch 42 on the first call; ``hook`` runs inside the next call."""

    def __init__(self):
        super().__init__()
        self.hook, self.before_park = None, None

    async def score(self, items):
        marks = [item["tokens"][-1] for item in items]
        if 4 in marks:
            self.calls.append(marks)
            if self.before_park is not None:
                self.before_park()
            raise _parked_error("lost", [marks.index(4)], [42])
        if self.hook is not None:
            self.hook()
        return await super().score(items)


def test_an_unpark_while_the_rest_of_the_call_is_scored_again_is_not_lost():
    """Parks are recorded when they are caught, before the rest of the call is scored again (the main path of a
    multi-batch pass): an unpark arriving meanwhile moves the record to now."""
    async def go():
        remote, clock = _RaceRemote(), _Clock(50_000.0)
        auditor = _order_auditor(None, clock, remote)
        remote.hook = lambda: remote.unpark(42)
        outcomes = await auditor._audit_outcomes([_record("a", 3), _record("p", 4)])
        assert outcomes[0]["passed"] and not isinstance(outcomes[1], dict)
        assert auditor._parked_wait("p", clock.now) is None, "the unpark that came during the relaunch was lost"
    _run(go)


def test_an_unpark_heard_before_its_park_is_recorded_still_counts():
    async def go():
        remote, clock = _RaceRemote(), _Clock(50_000.0)
        auditor = _order_auditor(None, clock, remote)
        remote.before_park = lambda: remote.unpark(42)    # the late answer lands before the auditor hears of the park
        await auditor._audit_outcomes([_record("p", 4)])
        assert auditor.parked_ids == {"p"} and auditor._parked_wait("p", clock.now) is None
    _run(go)


def test_a_failed_relaunch_keeps_what_was_parked():
    async def go():
        remote, clock = _RaceRemote(), _Clock(50_000.0)
        auditor = _order_auditor(None, clock, remote)

        def unwired():
            raise RuntimeError("the batch of order was cancelled")

        remote.hook = unwired
        results, _ = await auditor._audit_records(["a", "p"], [_record("a", 3), _record("p", 4)], {})
        assert results == [None, None] and auditor.parked_ids == {"p"}
        assert auditor._validator_errors == 1              # a's error only: p is parked, not an error
    _run(go)


def test_a_parked_record_cooling_down_is_never_scored_whoever_asks():
    """A pass that meets a parked record in an audit-ahead flight does not score it again at once."""
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        auditor = _order_auditor(None, clock, remote)
        remote.lost.add(4)
        await auditor._audit_outcomes([_record("p", 4)])
        calls = len(remote.calls)
        outcome, = await auditor._audit_outcomes([_record("p", 4)])
        assert not isinstance(outcome, dict) and len(remote.calls) == calls
        assert auditor._park_state["p"].tries == 0         # not parked again: its cool-down stands
    _run(go)


def test_an_unpark_reschedules_the_record_now():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0)})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        auditor._queued.add("p")
        await auditor.judge_many(["p"])
        auditor._reschedule(["p"])
        clock.now += 60
        auditor._queued.add("p")
        await auditor.judge_many(["p"])                    # cooling: due at its retry
        auditor._reschedule(["p"])
        assert auditor._due["p"] > clock.now + 600
        remote.unpark(42)
        assert auditor._due["p"] == clock.now
    _run(go)


def test_a_parked_re_audit_is_retried_in_the_background_after_its_cool_down():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"s1": _record("s1", 4)})
        auditor = _order_auditor(records, clock, remote)
        calls = []
        real = auditor.reaudit_executor

        async def counting(executor_id):
            calls.append(clock.now)
            return await real(executor_id)

        auditor.reaudit_executor = counting
        auditor._remote_scored["x"] = ["s1"]
        remote.lost.add(4)
        assert await auditor.reaudit_executor("x") == []
        state = auditor._reaudit_parked["s1"]
        clock.now = state.retry_at - 0.5                   # the background retry first sees it cooling
        task = auditor._late_reaudits.get("x")
        assert task is not None, "no background retry"
        await asyncio.sleep(0.3)
        clock.now = state.retry_at + 1
        remote.lost.clear()
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
        assert not auditor._remote_scored.get("x"), "the parked pass was never retried"
        assert len(calls) < 10, f"the background retry spins ({len(calls)} calls)"
    _run(go)


def test_a_pass_parked_in_a_re_audit_is_retried_alone():
    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        records = _Records({"s1": _record("s1", 4), "s2": _record("s2", 5)})
        auditor = _order_auditor(records, clock, remote)
        auditor._late_reaudits["x"] = asyncio.get_running_loop().create_future()
        auditor._remote_scored["x"] = ["s1"]
        remote.lost.add(4)
        assert await auditor.reaudit_executor("x") == []
        auditor._remote_scored["x"].append("s2")
        clock.now += RETRY
        remote.lost.clear()
        remote.calls.clear()
        await auditor.reaudit_executor("x")
        assert sorted(remote.calls) == [[4], [5]]
        auditor._late_reaudits["x"].cancel()
    _run(go)


def test_a_re_audit_in_slices_keeps_the_passes_parked_in_each_slice_in_flight(monkeypatch):
    """The re-audit reads and audits a quarantined executor's passes in slices, several at once, each slice's audit
    in a task of its own (`_reaudit_in_slices`). A pass parked in any of those slices is recorded in that task
    (`_record_park` reads `_REAUDIT_PARKED` there) and kept on the executor's list with its cool-down, neither
    voided nor charged, while a failure beside it is voided as usual. Once cooled down, each is tried again alone and
    leaves the list. Recorded outside the re-audit, a parked pass would leave the list unaudited and stay paid."""
    from reliquary.validator import corpus_auditor

    monkeypatch.setattr(corpus_auditor, "REAUDIT_SLICE_RECORDS", 3)
    monkeypatch.setattr(corpus_auditor, "REAUDIT_SLICES_IN_FLIGHT", 3)

    async def go():
        remote, clock = _Remote(), _Clock(50_000.0)
        sids = [f"s{k}" for k in range(9)]
        records = _Records({sid: _record(sid, 10 + k) for k, sid in enumerate(sids)})
        auditor = _order_auditor(records, clock, remote)
        auditor._late_reaudits["x"] = asyncio.get_running_loop().create_future()   # no background retry here
        auditor._remote_scored["x"] = list(sids)
        parked = ["s1", "s4", "s7"]
        remote.lost.update({11, 14, 17})
        remote.failing.add(15)                             # s5: a failure, beside s4 in the listing
        score, in_flight = remote.score, {"now": 0, "most": 0}

        async def overlapping(items):
            # Each call is answered later, as the executors do: the slices in flight overlap.
            in_flight["now"] += 1
            in_flight["most"] = max(in_flight["most"], in_flight["now"])
            try:
                await asyncio.sleep(0.01)
                return await score(items)
            finally:
                in_flight["now"] -= 1

        remote.score = overlapping
        assert await auditor.reaudit_executor("x") == ["s5"]
        assert in_flight["most"] >= 2, "the slices were not audited at once"
        holding = [k for k, marks in enumerate(remote.calls) if set(marks) & {11, 14, 17}]
        assert len(holding) >= 2, f"the parked passes were not spread over several slices: {remote.calls}"
        assert set(records.voided) == {"s5"}
        assert auditor._remote_scored.get("x") == parked and set(auditor._reaudit_parked) == set(parked)
        assert not set(parked) & auditor.parked_ids
        clock.now += RETRY
        remote.lost.clear()
        remote.calls.clear()
        assert await auditor.reaudit_executor("x") == []
        assert sorted(remote.calls) == [[11], [14], [17]]  # each alone
        assert not auditor._remote_scored.get("x") and not auditor._reaudit_parked
        assert set(records.voided) == {"s5"}
        auditor._late_reaudits["x"].cancel()
    _run(go)


def test_a_loss_resets_the_clean_streak():
    async def go():
        pool = _Pool()
        pool.dispatcher._strikes_limit = 3
        for k, delivered in enumerate((3, 2)):
            call = pool.call(_items(1), job=f"order-eval-lost-{k}")
            await asyncio.sleep(0)
            await pool.expire(pool.claim("c"))
            call.cancel()
            for _ in range(delivered):
                await _witness(pool, "c")
        assert pool.strikes("c") == 2                      # five deliveries, but never five in a row
    _run(go)


def test_a_park_takes_back_every_struck_lease_of_its_batch_and_only_those_once():
    async def go():
        pool = _Pool()
        pool.dispatcher._strikes_limit = 4
        poison = pool.call(_items(1))
        await asyncio.sleep(0)
        a1 = await _lose(pool, "a")                        # witnessed, strike 1, benched 300 s
        pool.now += BENCH + 1
        a2 = await _lose(pool, "a")                        # the same batch again: strike 2, benched again
        assert a2["batch"] == a1["batch"] and pool.strikes("a") == 2
        other = pool.call(_items(1), job="order-eval-other")
        await asyncio.sleep(0)
        a3 = pool.claim("a")
        assert a3["job"] == "order-eval-other"
        await pool.expire(a3)                              # another batch: strike 3
        other.cancel()
        await asyncio.sleep(0)
        await _lose(pool, "b")
        with pytest.raises(BatchParked):
            await _ended(poison)
        assert pool.strikes("a") == 1                      # the poison's two strikes back, not the other's
        pool.dispatcher._expired[a2["lease_id"]].full_length = False
        with pytest.raises(LeaseRefused):
            await pool.answer("a", a2)                     # a late answer has no strike left to take back
        assert pool.strikes("a") == 1
    _run(go)


def test_after_an_unpark_a_record_that_parks_again_waits_the_first_cool_down():
    class Remote(_Remote):
        """Each park names a new batch, as the dispatcher's batch ids do."""
        batch = 41

        async def score(self, items):
            marks = [item["tokens"][-1] for item in items]
            lost = [k for k, mark in enumerate(marks) if mark in self.lost]
            if lost:
                self.calls.append(marks)
                Remote.batch += 1
                raise _parked_error("its leases were lost", lost, [Remote.batch])
            return await super().score(items)

    async def go():
        remote, clock = Remote(), _Clock(50_000.0)
        records = _Records({"p": _record("p", 4, received_at=49_000.0)})
        auditor = _order_auditor(records, clock, remote)
        remote.lost.add(4)
        await auditor.judge_many(["p"])
        clock.now += 60
        remote.unpark(Remote.batch)                        # tried again at once ...
        await auditor.judge_many(["p"])
        await _retried(auditor)                            # ... alone, in a new batch, and parked again
        assert auditor._park_state["p"].retry_at == clock.now + RETRY
    _run(go)


def test_an_unpark_contradicts_the_park_so_the_record_holds_the_close_until_its_retry():
    """A park an unpark contradicts no longer counts: the record holds the close again until its retry decides it (or
    parks it alone again), within SOLO_RETRY_SECONDS of its next retry."""
    async def go():
        p, g = _ids(True, 2)
        records = _Records({p: _record(p, 4, received_at=T0), g: _record(g, 5, received_at=T0 + 10)})
        clock, remote = _Clock(T0 + GIVE_UP + 60), _HangingRemote()
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p, g])
        for sid in (p, g):
            auditor._park_state[sid].tries = 1             # as if each had parked again alone
        assert auditor.oldest_pending_received_at() is None
        remote.unpark(42)                                  # a card answered its batch after all
        assert auditor.oldest_pending_received_at() == T0
        remote.release.set()
    _run(go)


def _held_sibling_setup():
    p = _ids(True, 1)[0]                                   # drawn, parked
    x = _ids(False, 1, start=7000)[0]                      # undrawn sibling, payable once its hold is over
    steady = _ids(False, 2, start=5000)
    records = _Records({p: _record(p, 4, received_at=T0), x: _record(x, 5, received_at=T0 + 10),
                        **{s: _record(s, 6, received_at=T0 + 500) for s in steady}})
    clock, remote = _Clock(T0 + 20), _Remote()
    remote.lost.add(4)
    return p, x, steady, records, clock, remote


async def _x_held_by_cooling_p(auditor, p, x, steady, clock):
    auditor._queued.update([p, x])
    await auditor.judge_many([p, x])
    auditor._reschedule([p, x])
    clock.now = T0 + HOLD + 20                             # x's hold is over; p cools down
    auditor._queued.update([p, x, *steady])
    await auditor.judge_many([p, x, *steady])
    auditor._reschedule([p, x, *steady])
    retry = auditor._park_state[p].retry_at
    assert x not in auditor._judged and auditor._due[x] == retry > clock.now   # x waits for p's retry
    return retry


def test_an_unpark_wakes_the_passes_its_record_held():
    async def go():
        p, x, steady, records, clock, remote = _held_sibling_setup()
        auditor = _sampled_auditor(records, clock, remote)
        await _x_held_by_cooling_p(auditor, p, x, steady, clock)
        remote.unpark(42)
        assert auditor._due[x] == clock.now, "the unpark did not wake the pass p held"
    _run(go)


def test_deciding_a_parked_record_wakes_the_passes_it_held_and_only_those():
    async def go():
        p, x, steady, records, clock, remote = _held_sibling_setup()
        auditor = _sampled_auditor(records, clock, remote)
        await _x_held_by_cooling_p(auditor, p, x, steady, clock)
        later = {sid: auditor._due[sid] for sid in steady if sid in auditor._due}
        remote.lost.clear()
        auditor._park_state[p].retry_at = clock.now        # p comes due
        auditor._queued.add(p)
        await _judged_after_retry(auditor, [p])
        assert p in records.verdicts and p not in auditor.parked_ids
        assert auditor._due[x] == clock.now, "deciding p did not wake the pass it held"
        assert all(auditor._due.get(sid) == due for sid, due in later.items()), "it woke passes it did not hold"
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


def test_a_record_parked_alone_leaves_the_close_two_periods_after_its_receipt_even_after_a_restart():
    """A park that held its record's items alone names its culprit: it leaves the close at its receipt + 2 periods,
    whenever it was parked (the give-up counts from the receipt: a restart long after it starts no new wait)."""
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


class _HangingRemote(_Remote):
    """Parks the batch of the records marked 4 and 5 together on its first call; afterwards a call holding 4 never
    returns (a retry no card ever answers), and 5 passes."""

    def __init__(self):
        super().__init__()
        self.first, self.release = True, asyncio.Event()

    async def score(self, items):
        marks = [item["tokens"][-1] for item in items]
        self.calls.append(marks)
        if self.first and 4 in marks:
            self.first = False
            raise _parked_error("lost", [k for k, mark in enumerate(marks) if mark in (4, 5)], [42])
        if 4 in marks:
            await self.release.wait()
        return [("ok", (ChunkResult(1, 1.0, 1.0),), ("x",)) for _ in marks]


def test_a_record_parked_with_others_leaves_once_its_solo_retry_had_its_time_even_if_it_never_ends():
    """Parked with a batch-mate's items, it holds the close until its first retry alone had SOLO_RETRY_SECONDS (its
    honest batch-mate is decided first); a retry no card answers holds no pass, and the close is let go all the same."""
    async def go():
        p, g = _ids(True, 2)
        records = _Records({p: _record(p, 4, received_at=T0), g: _record(g, 5, received_at=T0 + 10)})
        clock, remote = _Clock(T0 + GIVE_UP + 60), _HangingRemote()
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p, g])
        assert auditor.parked_ids == {p, g}
        assert auditor.oldest_pending_received_at() == T0      # parked with each other: held, past the give-up
        clock.now += RETRY
        await _ended(auditor.judge_many([p, g]))               # p's retry hangs in the background: the pass ends
        for _ in range(100):
            if g not in auditor._retries:
                break
            await asyncio.sleep(0.01)
        assert p in auditor._retries                           # still out
        await auditor.judge_many([g])
        assert records.verdicts[g]["passed"]                   # the honest batch-mate, decided alone, before p leaves
        assert auditor.oldest_pending_received_at() == T0
        clock.now += getattr(ec, "SOLO_RETRY_SECONDS", 600.0) + 1
        assert auditor.oldest_pending_received_at() is None    # p's retry had its time: p lets go of the close
        remote.release.set()
        await _retried(auditor)
    _run(go)


def test_a_record_no_executor_can_read_leaves_the_close_two_periods_after_its_receipt():
    """Its own fault, alone in its park from the start: it leaves the close two periods after its receipt, to the second
    (the owner's two periods), whenever it was parked."""
    async def go():
        pool = _Pool()
        records = _Records({"p": _record("p", 4, tokens=65_600, received_at=T0)})
        clock = _Clock(T0 + 20)
        auditor = eval_auditor(job_id=JOB, records=records, tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                               remote=pool.dispatcher.view(*MODEL, PROOF, JOB), clock=clock)
        auditor._seeded = True
        await _ended(auditor.judge_many(["p"]))
        assert auditor.parked_ids == {"p"} and pool.dispatcher.pending() == 0
        assert GIVE_UP == 2 * cp.PERIOD_SECONDS
        clock.now = T0 + 2 * cp.PERIOD_SECONDS - 1
        assert auditor.oldest_pending_received_at() == T0
        clock.now = T0 + 2 * cp.PERIOD_SECONDS + 1
        assert auditor.oldest_pending_received_at() is None
    _run(go)


class _Archives:
    def __init__(self):
        self.docs = {}

    async def write(self, task_id, work, entry, document):
        self.docs[(work, entry)] = document

    async def read(self, task_id, work, entry):
        return self.docs.get((work, entry))

    async def list(self, task_id):
        return sorted(self.docs)


@pytest.mark.parametrize("parked_passes", [True, False])
def test_past_the_give_up_the_parked_operators_forfeitable_verdicts_wait_for_its_decision(parked_passes):
    """The stake guarantee (owner, 2026-11-26): once a parked record no longer holds its period's close, the verdicts of
    its operator received in its period up to its own receipt (what its failure can forfeit) stay unsettled until it is
    decided; every other verdict is paid on time. If it fails, they are forfeited, never paid. If it passes, they are
    paid late as any late verdict, and only the parked record's own late verdict is kept within the cap."""
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler

    async def go():
        p = _ids(True, 1)[0]                               # drawn, parked with its batch: hotkey 5Hot2, operator cold
        z = _ids(True, 1, start=3000)[0]                   # an audited pass of the same operator, before p
        w = _ids(True, 1, start=4000)[0]                   # one after p: p's failure cannot forfeit it
        o = _ids(True, 1, start=6000)[0]                   # another operator's
        assert cp.period_of(T0 - 30) == cp.period_of(T0 + 30)
        records = _Records({p: _record(p, 4, hotkey="5Hot2", received_at=T0), z: _record(z, 5, received_at=T0 - 30),
                            w: _record(w, 7, received_at=T0 + 30), o: _record(o, 6, hotkey="5Other",
                                                                             received_at=T0 - 20)})
        clock, remote = _Clock(T0 + 40), _Remote()
        remote.lost.add(4)
        sampled = MinerState(audited_passed=100)
        auditor = eval_auditor(job_id=JOB, records=records, tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                               params=AuditParams(q=Q, hold_seconds=HOLD, ban_after_failures=10),
                               miner_states=_States({HK: sampled, "5Hot2": sampled, "5Other": sampled}),
                               beacon=lambda r: RAND, round_at=lambda t: int(t // 3) + 1, clock=clock,
                               accept_slack_seconds=0.0, remote=remote,
                               operator_of=lambda hotkey: "cold" if hotkey in (HK, "5Hot2") else None)
        auditor.stake_at_receipt = False
        auditor._seeded = True
        await auditor.judge_many([z, w, o, p])
        assert set(records.verdicts) == {z, w, o} and auditor.parked_ids == {p}
        clock.now += RETRY
        await auditor.judge_many([p])                      # tried alone, parked again

        state = {}

        async def write_settlement(job_id, document, etag):
            state.clear()
            state.update(document)
            return "etag"

        async def read_settlement(job_id):
            return dict(state), "etag"

        records.write_settlement, records.read_settlement = write_settlement, read_settlement
        archives = _Archives()
        hooks = {name: getattr(auditor, attribute) for name, attribute in
                 (("hold", "settle_hold"), ("capped", "settle_capped")) if hasattr(auditor, attribute)}
        settler = CorpusPeriodSettler(task_id=JOB, job_id=JOB, cap=1.0, records=records, archives=archives,
                                      oldest_pending=auditor.oldest_pending_received_at, clock=clock, **hooks)
        clock.now = T0 + GIVE_UP + 1 + cp.PERIOD_SECONDS
        await settler.settle_once()
        assert {o, w} <= set(state["settled"]) and z not in state["settled"]
        assert {hk for doc in archives.docs.values() for hk in doc["rewards_by_hotkey"]} == {"5Other", HK}
        on_time = sum(doc["rewards_by_hotkey"].get(HK, 0.0) for doc in archives.docs.values())
        if not parked_passes:
            remote.lost.clear()
            remote.failing.add(4)                          # p decided at last: it fails, and its failure is confirmed
        else:
            remote.lost.clear()
        clock.now += 10 * RETRY
        await _judged_after_retry(auditor, [p])
        assert p in records.verdicts and auditor.parked_ids == set()
        await settler.settle_once()
        assert {z, p} <= set(state["settled"])
        late = sum(doc["rewards_by_hotkey"].get(HK, 0.0) for doc in archives.docs.values()) - on_time
        assert late == 0                                   # z: capped as p is (the period paid its cap), or forfeited
        assert not any("5Hot2" in doc["rewards_by_hotkey"] for doc in archives.docs.values())  # p: within the cap
        total = sum(v for doc in archives.docs.values() for v in doc["rewards_by_hotkey"].values())
        assert total == pytest.approx(1.0)                 # the period never pays past its cap
        if parked_passes:
            assert records.verdicts[p].get("left_out_of_close") is True   # durable: capped after a restart too
        else:
            assert z in records.voided and records.voided[z]["reason"] == "forfeited"
    _run(go)


def test_after_a_restart_the_parked_records_mark_still_caps_it_and_what_it_held():
    """A restart forgets which records left the close; the mark in the parked record's verdict still caps it, and the
    verdicts of its operator in its period received up to it, settled with it."""
    async def go():
        clock, remote = _Clock(T0 + GIVE_UP + 600), _Remote()
        auditor = _sampled_auditor(_Records({}), clock, remote)    # a fresh process
        period = cp.period_of(T0)
        released = {"p": {"hotkey": HK, "received_at": T0, "left_out_of_close": True},
                    "z": {"hotkey": HK, "received_at": T0 - 30}, "w": {"hotkey": HK, "received_at": T0 + 30},
                    "o": {"hotkey": "5Other", "received_at": T0 - 30}}
        assert cp.period_of(T0 - 30) == period
        assert auditor.settle_capped(released, cp.period_of) == {"p", "z"}
    _run(go)


def test_what_is_held_is_its_operators_verdicts_received_before_it_in_its_period_and_only_at_q_below_one():
    async def go():
        p = _ids(True, 1)[0]
        records = _Records({p: _record(p, 4, received_at=T0)})
        clock, remote = _Clock(T0 + 20), _Remote()
        remote.lost.add(4)
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p])
        clock.now += RETRY
        await auditor.judge_many([p])                      # alone, parked again
        clock.now = T0 + GIVE_UP + 1
        period = cp.period_of(T0)
        verdicts = {"before": (HK, period, T0 - 30), "after": (HK, period, T0 + 30),
                    "later_period": (HK, period + 1, T0 + cp.PERIOD_SECONDS), "other": ("5Other", period, T0 - 30)}
        assert auditor.settle_hold(verdicts, cp.period_of) == {"before"}
        # Capped once released: the parked record, any verdict marked so, and what the hold kept (owner, 2026-11-26).
        released = {p: {"hotkey": HK, "received_at": T0}, "marked": {"left_out_of_close": True},
                    **{sid: {"hotkey": hotkey, "received_at": at} for sid, (hotkey, _, at) in verdicts.items()}}
        assert auditor.settle_capped(released, cp.period_of) == {p, "marked", "before"}
        q_one = _order_auditor(_Records({"r": _record("r", 4, received_at=T0)}), clock, remote)
        clock.now = T0 + 20
        await q_one.judge_many(["r"])
        clock.now += RETRY
        await q_one.judge_many(["r"])
        clock.now = T0 + GIVE_UP + 1
        assert q_one.oldest_pending_received_at() is None
        assert q_one.settle_hold({"z": ("5Hot", period, T0 - 30)}, cp.period_of) == set()   # q = 1: no forfeit
    _run(go)


def test_the_hold_reaches_the_latest_parked_receipt_its_own_instant_and_verdicts_with_no_receipt():
    """`forfeits` voids up to the failure's receipt inclusive; a verdict with no receipt may be anything; two parked
    records of one operator and period hold up to the later receipt."""
    async def go():
        p1, p2 = _ids(True, 2)
        records = _Records({p1: _record(p1, 4, received_at=T0), p2: _record(p2, 8, received_at=T0 + 100)})
        clock, remote = _Clock(T0 + 120), _Remote()
        remote.lost.update({4, 8})
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p1, p2])
        clock.now += RETRY
        await auditor.judge_many([p1, p2])
        await _retried(auditor)                            # each alone, parked again
        assert auditor.parked_ids == {p1, p2}
        clock.now = T0 + 100 + GIVE_UP + 1
        assert auditor.oldest_pending_received_at() is None
        period = cp.period_of(T0)
        assert cp.period_of(T0 + 100) == period
        held = auditor.settle_hold({"same": (HK, period, T0), "none": (HK, period, None),
                                    "between": (HK, period, T0 + 50), "after": (HK, period, T0 + 150)},
                                   cp.period_of)
        assert held == {"same", "none", "between"}
    _run(go)


def test_a_parked_record_cooling_past_its_give_up_holds_no_sibling():
    async def go():
        p = _ids(True, 1)[0]
        x = _ids(False, 1, start=7000)[0]
        steady = _ids(False, 2, start=5000)
        records = _Records({p: _record(p, 4, received_at=T0), x: _record(x, 5, received_at=T0 + 10),
                            **{sid: _record(sid, 6, received_at=T0 + 500) for sid in steady}})
        clock, remote = _Clock(T0 + 20), _Remote()
        remote.lost.add(4)
        auditor = _sampled_auditor(records, clock, remote)
        await auditor.judge_many([p, x])
        assert auditor.parked_ids == {p}
        clock.now = T0 + GIVE_UP - 100                     # tried again just before its give-up, parked again
        auditor._park_state[p].retry_at = clock.now
        await auditor.judge_many([p])
        assert auditor._park_state[p].retry_at > T0 + GIVE_UP + 1 and x not in records.verdicts
        clock.now = T0 + GIVE_UP + 1                       # cooling, past its give-up
        later = _ids(False, 2, start=9000)
        for sid in later:
            records.submissions[sid] = _record(sid, 7, received_at=clock.now - 500)
        await auditor.judge_many([p, x, *steady, *later])
        assert x in records.verdicts and records.verdicts[x]["passed"], "a left-out record still held its sibling"
    _run(go)


def test_an_orders_settler_asks_its_auditor_what_to_hold_and_what_to_keep_within_the_cap():
    """``wire_order_judge`` wires both owner decisions into the order's settler; the corpus control's has neither."""
    from reliquary.validator.eval_control import wire_order_judge

    async def go():
        job = SimpleNamespace(job_id="order-gen-x", episode=None, prompt_source="x",
                              sampling=SimpleNamespace(draw=None), checkpoint_repo=MODEL[0],
                              checkpoint_revision=MODEL[1])
        w = SimpleNamespace(entry=SimpleNamespace(task_id="order-gen-x", params={"cap": 0.1}), cap=0.1, job=job,
                            tokenizer=_Tokenizer(), proof=PROOF, vocab_size=100,
                            stats=SimpleNamespace(observe=lambda *a, **k: None))
        wire_order_judge(w, records=_Records({}), job_store=None, remote=_Remote(), params=AuditParams(q=0.5),
                         miner_states=None, beacon=None, round_at=None)
        asked = []
        w.auditor.settle_hold = lambda verdicts, period_of: asked.append("hold") or {"z"}
        w.auditor.settle_capped = lambda verdicts, period_of: asked.append("capped") or {"p"}
        assert w.settler._hold({}, cp.period_of) == {"z"} and w.settler._capped({"p": {}}, cp.period_of) == {"p"}
        assert asked == ["hold", "capped"]
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
