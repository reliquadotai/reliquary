"""Restart and lost-ack recovery with real conditional object writes."""

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.infrastructure import corpus_executor_store
from reliquary.infrastructure.corpus_attempt_store import AttemptRefused, AttemptStore, digest
from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit_protocol import AuditLease, AuditResult
from reliquary.validator.corpus_audit_remote import ExecutorDirectory, RemoteAuditDispatcher
from tests.unit.test_corpus_audit_remote import _Clock, _directory, _doc
from tests.unit.test_corpus_grade_remote import (
    PACKAGE, VERSION, PASS, _dispatcher, _item, _result,
)
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

ITEMS = [{"tokens": [1, 2, 3], "prompt_len": 1, "proofs": ["abcd"]}]
ANSWER = AuditResult.model_validate({"scores": [{"status": "ok", "chunks": [[0, 0, 0]]}]})


@pytest.fixture
def attempts(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(corpus_executor_store, "get_s3_client", lambda **kw: client)
    return AttemptStore({"kind": "audit", "model": "frozen", "revision": "pin"})


async def audit(store, clock, *, docs=None, fraction=0, local=None):
    directory = _directory(docs or [_doc(), _doc("pod-2", "second-token")], clock)
    await directory.refresh()

    async def local_scores(items):
        return [("ok", (ChunkResult(0, 0.0, 0.0),)) for _ in items]

    d = RemoteAuditDispatcher(directory=directory, proof=SimpleNamespace(chunk_tokens=256, topk=16),
        local_scores=local or local_scores, clock=clock, recheck_fraction=fraction, attempt_store=store)
    return d


async def started(awaitable):
    task = asyncio.create_task(awaitable)
    await asyncio.sleep(0)
    assert not task.done()
    return task


async def cancel(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_claims_have_one_generation_and_submission_blocks_expired_takeover(attempts):
    key, credential = digest(ITEMS), digest("registration")
    leases = [{"lease_id": c * 32, "executor_id": "worker", "expires_at": 1100.0} for c in "ab"]
    heads = await asyncio.gather(*(attempts.claim(key, lease, credential, {}, now=1000) for lease in leases))
    assert sum(h is not None for h in heads) == 1
    head = next(h for h in heads if h)
    reserved = await attempts.reserve(key, head["lease_id"], "worker", credential, {"facts": 1}, {}, now=1001)
    assert reserved["status"] == "submitted"
    assert await attempts.claim(key, leases[1], credential, {}, now=2000) is None
    await attempts.finish(key, head["lease_id"], 1, {}, "accepted")
    replacement = await attempts.claim(key, {**leases[1], "lease_id": "c" * 32, "expires_at": 3000},
                                       credential, {}, now=2000)
    assert replacement["generation"] == 2
    assert (await attempts.lease(head["lease_id"]))["outcome"] == "accepted"
    with pytest.raises(AttemptRefused, match="lease_superseded"):
        await attempts.finish(key, head["lease_id"], 1, {}, "accepted")


@pytest.mark.asyncio
async def test_restart_retains_actual_lease_and_retries_are_idempotent(attempts):
    clock = _Clock()
    first = await audit(attempts, clock)
    old = await started(first.score(ITEMS))
    lease = await first.durable_claim("pod-1")
    AuditLease.model_validate(lease)  # Existing strict wire format, no new miner fields.
    await cancel(old)
    recovered = await audit(attempts, clock)
    pending = await started(recovered.score(ITEMS))
    assert list(recovered._leases) == [lease["lease_id"]]
    assert await recovered.durable_claim("pod-2") is None
    assert await recovered.durable_result("pod-1", lease["lease_id"], ANSWER) == "accepted"
    assert (await pending)[0][2] == "pod-1"
    assert await recovered.durable_result("pod-1", lease["lease_id"], ANSWER) == "accepted"
    changed = AuditResult.model_validate({"scores": [{"status": "ok", "chunks": [[1, 0, 0]]}]})
    with pytest.raises(AttemptRefused, match="attempt_result_conflict"):
        await recovered.durable_result("pod-1", lease["lease_id"], changed)
    assert recovered.stats["scored"] == 1


@pytest.mark.asyncio
async def test_result_reserved_before_crash_is_applied_after_restart_even_after_expiry(attempts, monkeypatch):
    clock = _Clock()
    first = await audit(attempts, clock)
    old = await started(first.score(ITEMS))
    lease = await first.durable_claim("pod-1")

    def crash(*args):
        raise RuntimeError("interrupted before native application")

    monkeypatch.setattr(first, "result", crash)
    with pytest.raises(RuntimeError):
        await first.durable_result("pod-1", lease["lease_id"], ANSWER)
    assert (await attempts.lease(lease["lease_id"]))["status"] == "submitted"
    await cancel(old)
    clock.now = lease["expires_at"] + 1
    recovered = await audit(attempts, clock)
    scores = await recovered.score(ITEMS)
    assert scores[0][2] == "pod-1"
    assert (await attempts.lease(lease["lease_id"]))["snapshot"]["decision"]["executor"] == "pod-1"


@pytest.mark.asyncio
async def test_lost_finish_ack_is_reconciled_without_applying_result_twice(attempts, monkeypatch):
    d = await audit(attempts, _Clock())
    pending = await started(d.score(ITEMS))
    lease = await d.durable_claim("pod-1")
    real = attempts.finish
    calls = 0

    async def fail_once(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("write unavailable")
        return await real(*args, **kw)

    monkeypatch.setattr(attempts, "finish", fail_once)
    with pytest.raises(RuntimeError):
        await d.durable_result("pod-1", lease["lease_id"], ANSWER)
    assert (await pending)[0][2] == "pod-1"
    assert await d.durable_result("pod-1", lease["lease_id"], ANSWER) == "accepted"
    assert d.stats["scored"] == 1


@pytest.mark.asyncio
async def test_uncertain_claim_write_cannot_bypass_ownership_through_local_fallback(attempts, monkeypatch):
    d = await audit(attempts, _Clock())
    pending = await started(d.score(ITEMS))
    real = attempts._write
    once = True

    async def lost_ack(key, document, etag):
        nonlocal once
        written = await real(key, document, etag)
        if once:
            once = False
            raise RuntimeError("claim acknowledgement lost")
        return written

    monkeypatch.setattr(attempts, "_write", lost_ack)
    with pytest.raises(RuntimeError):
        await d.durable_claim("pod-1")
    assert d._queue[0].attempt_uncertain
    await d.sweep()
    assert not pending.done() and not d._queue and d.stats["local"] == 0
    assert len(d._leases) == 1
    lease = next(iter(d._leases))
    assert await d.durable_claim("pod-2") is None
    await d.durable_result("pod-1", lease, ANSWER)
    assert (await pending)[0][2] == "pod-1"


@pytest.mark.asyncio
async def test_duplicate_audit_waiters_share_one_actual_attempt(attempts):
    d = await audit(attempts, _Clock())
    one = await started(d.score(ITEMS))
    two = await started(d.score(ITEMS))
    assert len(d._queue) == 1
    lease = await d.durable_claim("pod-1")
    await cancel(one)
    assert not two.done()
    await d.durable_result("pod-1", lease["lease_id"], ANSWER)
    assert (await two)[0][2] == "pod-1"
    assert d.stats["leased"] == 1


@pytest.mark.asyncio
async def test_expired_recovered_lease_counts_once_and_stale_result_is_fenced(attempts):
    clock = _Clock()
    first = await audit(attempts, clock)
    old = await started(first.score(ITEMS))
    lease = await first.durable_claim("pod-1")
    await cancel(old)
    clock.now = lease["expires_at"] + 1
    d = await audit(attempts, clock)
    pending = await started(d.score(ITEMS))
    new = await d.durable_claim("pod-2")
    assert new["lease_id"] != lease["lease_id"]
    work = d._leases[new["lease_id"]].work
    assert work.attempts == 1 and work.attempt_generation == 2
    with pytest.raises(AttemptRefused, match="lease_expired"):
        await d.durable_result("pod-1", lease["lease_id"], ANSWER)
    await d.durable_result("pod-2", new["lease_id"], ANSWER)
    await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"token_sha256": "b" * 64}, {"model_revision": "changed"},
                                   {"status": "revoked"}])
async def test_changed_executor_registration_does_not_restore_old_ownership(attempts, change):
    clock = _Clock()
    d = await audit(attempts, clock)
    old = await started(d.score(ITEMS))
    lease = await d.durable_claim("pod-1")
    await cancel(old)
    d = await audit(attempts, clock, docs=[_doc(**change), _doc("pod-2", "second-token")])
    pending = await started(d.score(ITEMS))
    assert lease["lease_id"] not in d._leases
    new = await d.durable_claim("pod-2")
    assert d._leases[new["lease_id"]].work.attempt_generation == 2
    await d.durable_result("pod-2", new["lease_id"], ANSWER)
    await pending


@pytest.mark.asyncio
async def test_grade_vote_and_draw_survive_restart_without_repeating_provider(attempts):
    clock = _Clock()
    store = AttemptStore({"kind": "grade", "package": PACKAGE, "version": VERSION})
    d = await _dispatcher(clock=clock, attempt_store=store, recheck=0, recheck_fraction=1)
    old = await started(d.decide(_item()))
    lease = await d.durable_claim("g0")
    old_lease_id = lease["lease_id"]
    await d.durable_result("g0", lease["lease_id"], _result(PASS))
    await asyncio.sleep(0)
    assert not old.done()
    await cancel(old)
    d = await _dispatcher(clock=clock, attempt_store=store, recheck_fraction=0)
    pending = await started(d.decide(_item()))
    assert await d.durable_claim("g0") is None
    lease = await d.durable_claim("g1")
    work = d._leases[lease["lease_id"]].work
    assert work.drawn is True and set(work.results) == {"g0"}
    # The prior generation's lost acknowledgement remains retryable after
    # the next provider has claimed, and never applies a second vote.
    assert await d.durable_result("g0", old_lease_id, _result(PASS)) == "accepted"
    await d.durable_result("g1", lease["lease_id"], _result(PASS))
    decision = await pending
    assert decision.graded_by == ("g0", "g1") and decision.providers == ("p0", "p1")


@pytest.mark.asyncio
async def test_a_changed_prior_grade_vote_registration_is_not_counted_after_restart(attempts):
    clock = _Clock()
    store = AttemptStore({"kind": "grade", "package": PACKAGE, "version": VERSION})
    d = await _dispatcher(clock=clock, attempt_store=store, recheck=0, recheck_fraction=1)
    old = await started(d.decide(_item()))
    lease = await d.durable_claim("g0")
    await d.durable_result("g0", lease["lease_id"], _result(PASS))
    lease = await d.durable_claim("g1")
    await cancel(old)
    d = await _dispatcher(clock=clock, attempt_store=store, recheck=0, recheck_fraction=1)
    d._directory.document("g0")["token_sha256"] = "e" * 64
    pending = await started(d.decide(_item()))
    work = d._leases[lease["lease_id"]].work
    assert not work.results and not work.providers
    await d.durable_result("g1", lease["lease_id"], _result(PASS))
    assert not work.future.done()
    new = await d.durable_claim("g2")
    await d.durable_result("g2", new["lease_id"], _result(PASS))
    assert (await pending).graded_by == ("g1", "g2")


@pytest.mark.asyncio
async def test_grade_reserved_vote_crash_and_lost_claim_report_preserve_real_counters(attempts, monkeypatch):
    clock = _Clock()
    store = AttemptStore({"kind": "grade", "package": PACKAGE, "version": VERSION})
    d = await _dispatcher(clock=clock, attempt_store=store, recheck=0, recheck_fraction=1)
    old = await started(d.decide(_item()))
    lease = await d.durable_claim("g0")

    def crash(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(d, "result", crash)
    with pytest.raises(RuntimeError):
        await d.durable_result("g0", lease["lease_id"], _result(PASS))
    await cancel(old)
    d = await _dispatcher(clock=clock, attempt_store=store, recheck=0, recheck_fraction=1)
    pending = await started(d.decide(_item()))
    lease = await d.durable_claim("g1")
    clock.now += 61
    d.heartbeat("g1", held_leases=[])
    new = await d.durable_claim("g2")
    work = d._leases[new["lease_id"]].work
    assert work.errors == 1 and work.timeouts == 0
    with pytest.raises(AttemptRefused, match="lease_unreported"):
        await d.durable_result("g1", lease["lease_id"], _result(PASS))
    await d.durable_result("g2", new["lease_id"], _result(PASS))
    assert (await pending).graded_by == ("g0", "g2")


@pytest.mark.asyncio
async def test_a_drawn_audit_recheck_is_restarted_instead_of_trusting_remote_ack(attempts):
    clock = _Clock()
    gate = asyncio.Event()

    async def delayed(items):
        await gate.wait()
        return [("ok", (ChunkResult(0, 0.0, 0.0),))]

    d = await audit(attempts, clock, fraction=1, local=delayed)
    old = await started(d.score(ITEMS))
    lease = await d.durable_claim("pod-1")
    assert await d.durable_result("pod-1", lease["lease_id"], ANSWER) == "accepted"
    await asyncio.sleep(0)
    assert not old.done()
    for task in list(d._background):
        await cancel(task)
    await cancel(old)
    d = await audit(attempts, clock, fraction=0)
    scores = await d.score(ITEMS)
    assert scores[0][2] is None  # Trusted recheck, with the persisted draw.
    assert d.stats["rechecks"] == 1


@pytest.mark.asyncio
async def test_binding_payload_and_snapshot_bounds_are_checked(attempts, monkeypatch):
    other = AttemptStore({"kind": "audit", "model": "frozen", "revision": "other"})
    key = digest(ITEMS)
    head = await attempts.claim(key, {"lease_id": "a" * 32, "executor_id": "worker", "expires_at": 2000},
                                digest("registration"), {}, now=1000)
    assert await other.read(key) is None
    assert await attempts.read(digest([{**ITEMS[0], "tokens": [1, 2, 4]}])) is None
    monkeypatch.setattr("reliquary.infrastructure.corpus_attempt_store.MAX_DOCUMENT_BYTES", 1024)
    with pytest.raises(ValueError, match="byte bound"):
        await attempts.finish(key, head["lease_id"], 1, {"huge": "x" * 2000}, "accepted")
