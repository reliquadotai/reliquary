"""R3: remote audit executors. Tokens are checked against the registry, results
stay provisional until a local recheck vouches for them, a lying executor is
quarantined and its batches re-queued, and with no executor the control audits
locally as before."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_audit import score_sequences
from reliquary.validator.corpus_audit_protocol import AuditResult
from reliquary.validator import corpus_audit_remote
from reliquary.validator.corpus_audit_remote import (
    ExecutorDirectory,
    LeaseRefused,
    RemoteAuditDispatcher,
    build_audit_executor_router,
    token_sha256,
)
from tests.unit.test_corpus_audit import _tiny
from tests.unit.test_corpus_auditor import _record, _Records, _Tokenizer

MODEL, REVISION = "org/Frozen", "abc123"
GOOD, OTHER = "good-token-" + "x" * 20, "other-token-" + "y" * 20


def _doc(executor_id="pod-1", token=GOOD, **kw):
    fields = dict(executor_id=executor_id, token_sha256=token_sha256(token), model_id=MODEL,
                  model_revision=REVISION, expires_at=10_000.0, status="active")
    fields.update(kw)
    return fields


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _directory(docs, clock):
    async def listed():
        return [dict(d) for d in docs]

    return ExecutorDirectory(model_id=MODEL, model_revision=REVISION, list_documents=listed,
                             clock=clock)


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------


def test_only_an_active_unexpired_token_for_this_model_is_accepted():
    clock = _Clock()
    docs = [_doc(), _doc("pod-2", OTHER, status="revoked"), _doc("pod-3", "t3" * 10, expires_at=999.0),
            _doc("pod-4", "t4" * 10, model_revision="other")]
    directory = _directory(docs, clock)
    asyncio.run(directory.refresh())
    assert directory.authenticate(GOOD)[0]["executor_id"] == "pod-1"
    assert directory.authenticate("nope")[1] == "unknown_token"
    assert directory.authenticate(OTHER)[1] == "revoked"
    assert directory.authenticate("t3" * 10)[1] == "expired"
    assert directory.authenticate("t4" * 10)[1] == "wrong_model"
    assert directory.authenticate(GOOD, "pod-2")[1] == "wrong_executor"
    assert directory.authenticate(None)[1] == "missing_token"


def test_the_registry_is_reread_every_30_seconds():
    clock = _Clock()
    docs = [_doc()]
    directory = _directory(docs, clock)
    asyncio.run(directory.maybe_refresh())
    docs[0]["status"] = "revoked"
    clock.now += 29
    asyncio.run(directory.maybe_refresh())
    assert directory.authenticate(GOOD)[0] is not None
    clock.now += 2
    asyncio.run(directory.maybe_refresh())
    assert directory.authenticate(GOOD)[1] == "revoked"


# --------------------------------------------------------------------------
# The dispatcher, with the tiny model as both the honest executor and the control
# --------------------------------------------------------------------------


def _items(model, n=3):
    records = [_record(model, rendered=f"prompt {k}", completion=list(range(100 + k, 160 + k)))
               for k in range(n)]
    from reliquary.corpus.encoding import prompt_token_ids

    items = []
    for record in records:
        prompt = prompt_token_ids(_Tokenizer(), record["rendered_prompt"])
        completion = record["completions"][0]
        items.append({"tokens": prompt + completion["tokens"], "prompt_len": len(prompt),
                      "proofs": completion["proofs"]})
    return items


def _wire(items):
    return [(i["tokens"], i["prompt_len"], i["proofs"]) for i in items]


def _honest_scores(model, lease):
    scores, _, _ = score_sequences(model, _wire(lease["items"]), chunk_tokens=lease["chunk_tokens"],
                                   topk=lease["topk"], batch_tokens=1 << 20)
    return AuditResult.model_validate({"scores": [
        {"status": status, "chunks": [[r.exp_mismatches, r.mant_err_mean, r.mant_err_median]
                                      for r in results]}
        for status, results in scores]})


def _lying_scores(lease):
    """Every chunk a perfect match: what an executor paid to pass everything sends."""
    return AuditResult.model_validate({"scores": [
        {"status": "ok", "chunks": [[0, 0.0, 0.0] for _ in item["proofs"]]}
        for item in lease["items"]]})


class _Draws:
    """A recheck draw per result, in order: 0.0 always rechecks, 1.0 never does."""

    def __init__(self, *values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 1.0


class _Harness:
    def __init__(self, model, *, fraction=0.0, docs=None, draws=None, gate=None):
        self.clock = _Clock()
        self.model = model
        self.directory = _directory(docs or [_doc(), _doc("pod-2", OTHER)], self.clock)
        self.local_calls = 0
        self.quarantined = []

        async def local_scores(items):
            self.local_calls += 1
            if gate is not None:
                await gate.wait()
            scores, _, _ = score_sequences(model, _wire(items), chunk_tokens=PROOF.chunk_tokens,
                                           topk=PROOF.topk, batch_tokens=1 << 20)
            return scores

        async def quarantine(executor_id, reason):
            self.quarantined.append((executor_id, reason))

        self.dispatcher = RemoteAuditDispatcher(
            directory=self.directory, proof=PROOF, local_scores=local_scores,
            quarantine=quarantine, clock=self.clock, rng=draws or _Draws(),
            recheck_fraction=fraction)


async def _drain(dispatcher, rounds=20):
    for _ in range(rounds):
        await asyncio.sleep(0)
        await dispatcher.sweep()
        await asyncio.sleep(0)


def test_an_undrawn_batch_is_resolved_from_its_executor_and_says_so():
    model = _tiny(0)

    async def go():
        h = _Harness(model, fraction=0.05, draws=_Draws(0.5))
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        assert h.dispatcher.connected()
        task = asyncio.ensure_future(h.dispatcher.score(_items(model)))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim("pod-1")
        assert h.dispatcher.result("pod-1", lease["lease_id"], _honest_scores(model, lease)) == "accepted"
        scores = await task
        assert [(s, by) for s, _, by in scores] == [("ok", "pod-1")] * 3
        assert h.local_calls == 0

    asyncio.run(go())


def test_a_drawn_batch_is_resolved_from_the_local_recheck():
    model = _tiny(0)

    async def go():
        h = _Harness(model, fraction=0.05, draws=_Draws(0.0))
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        task = asyncio.ensure_future(h.dispatcher.score(_items(model)))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim("pod-1")
        h.dispatcher.result("pod-1", lease["lease_id"], _honest_scores(model, lease))
        scores = await task
        assert [by for _, _, by in scores] == [None] * 3
        assert h.local_calls == 1 and h.quarantined == []

    asyncio.run(go())


def test_the_review_counterexample_is_a_divergence():
    """C1: a measure under-reported by just under one threshold flips the decision."""
    from dataclasses import replace

    from reliquary.protocol.toploc import ChunkResult
    from reliquary.validator.corpus_audit_remote import scores_agree

    proof = replace(PROOF, exp_mismatch_threshold=90, mant_mean_threshold=10.0,
                    mant_median_threshold=8.0, min_allowed_failures=0, ratio_allowed_failures=0.0)
    local = [("ok", (ChunkResult(170, 19.0, 15.0),))]
    assert not scores_agree([("ok", (ChunkResult(85, 9.5, 7.5),))], local, proof)
    # The same decision but a drift past the hardware tolerance is a divergence too.
    near = [("ok", (ChunkResult(10, 2.0, 2.0),))]
    assert not scores_agree([("ok", (ChunkResult(13, 2.0, 2.0),))], near, proof)
    assert not scores_agree([("ok", (ChunkResult(10, 3.1, 2.0),))], near, proof)
    # Hardware noise inside the tolerance agrees.
    assert scores_agree([("ok", (ChunkResult(12, 2.9, 2.7),))], near, proof)


def test_a_lying_executor_is_quarantined_and_the_rechecked_lie_is_never_written():
    model, other = _tiny(0), _tiny(1)

    async def go():
        h = _Harness(model, fraction=0.05, draws=_Draws(0.0))
        await h.directory.refresh()
        bad = _items(other, n=1)
        h.dispatcher.heartbeat("pod-1")
        listened = []

        async def listener(executor_id):
            listened.append(executor_id)

        h.dispatcher.subscribe(listener)
        task = asyncio.ensure_future(h.dispatcher.score(bad))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim("pod-1")
        h.dispatcher.result("pod-1", lease["lease_id"], _lying_scores(lease))
        scores = await task
        await _drain(h.dispatcher, rounds=2)
        from reliquary.validator.corpus_audit import outcome_from_scores

        assert all(not outcome_from_scores(s, c, PROOF).passed for s, c, _ in scores)
        assert [q[0] for q in h.quarantined] == ["pod-1"]
        assert listened == ["pod-1"]
        assert h.directory.authenticate(GOOD)[1] == "revoked"

    asyncio.run(go())


def test_concurrent_rechecks_never_vouch_for_each_other():
    """I3: an honest batch's recheck finishing first leaves the lying one held."""
    model, other = _tiny(0), _tiny(1)

    async def go():
        gate = asyncio.Event()
        h = _Harness(model, fraction=0.05, draws=_Draws(0.0, 0.0), gate=gate)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        lie = asyncio.ensure_future(h.dispatcher.score(_items(other, n=1)))
        honest = asyncio.ensure_future(h.dispatcher.score(_items(model, n=1)))
        await asyncio.sleep(0)
        lie_lease, honest_lease = h.dispatcher.claim("pod-1"), h.dispatcher.claim("pod-1")
        h.dispatcher.result("pod-1", lie_lease["lease_id"], _lying_scores(lie_lease))
        h.dispatcher.result("pod-1", honest_lease["lease_id"], _honest_scores(model, honest_lease))
        await asyncio.sleep(0)
        assert not lie.done() and not honest.done()
        gate.set()
        from reliquary.validator.corpus_audit import outcome_from_scores

        assert all(not outcome_from_scores(s, c, PROOF).passed for s, c, _ in await lie)
        assert [by for _, _, by in await honest] == [None]
        await _drain(h.dispatcher, rounds=2)
        assert [q[0] for q in h.quarantined] == ["pod-1"]

    asyncio.run(go())


def test_an_executor_holds_at_most_two_leases():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        tasks = [asyncio.ensure_future(h.dispatcher.score(_items(model, n=1))) for _ in range(3)]
        await asyncio.sleep(0)
        assert h.dispatcher.claim("pod-1") is not None
        assert h.dispatcher.claim("pod-1") is not None
        assert h.dispatcher.claim("pod-1") is None
        assert h.dispatcher.claim("pod-2") is not None
        for task in tasks:
            task.cancel()

    asyncio.run(go())


def test_three_expired_leases_in_a_row_bench_the_executor_not_quarantine_it():
    # Ruling P26's spirit: an expiry says the box died or fell behind, not that
    # its scores were wrong. Benched (no lease for a while), never quarantined.
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        tasks = []
        for _ in range(3):
            tasks.append(asyncio.ensure_future(h.dispatcher.score(_items(model, n=1))))
            await asyncio.sleep(0)
            h.dispatcher.heartbeat("pod-2")
            h.dispatcher.heartbeat("pod-1")
            assert h.dispatcher.claim("pod-1") is not None
            h.clock.now += 301
            h.dispatcher.heartbeat("pod-2")
            await h.dispatcher.sweep()
        assert h.quarantined == [] and "pod-1" not in h.dispatcher.quarantined
        tasks.append(asyncio.ensure_future(h.dispatcher.score(_items(model, n=1))))
        await asyncio.sleep(0)
        h.dispatcher.heartbeat("pod-1")
        assert h.dispatcher.claim("pod-1") is None          # benched
        h.clock.now += corpus_audit_remote.EXPIRY_BENCH_SECONDS + 1
        h.dispatcher.heartbeat("pod-1")
        assert h.dispatcher.claim("pod-1") is not None      # back after the bench
        for task in tasks:
            task.cancel()

    asyncio.run(go())


def test_the_lease_life_is_bounded(monkeypatch):
    from reliquary.validator.corpus_audit_remote import AUDIT_LEASE_SECONDS, _bounded_env

    monkeypatch.setenv("RELIQUARY_CORPUS_AUDIT_LEASE_SECONDS", "3600")
    with pytest.raises(ValueError):
        _bounded_env("RELIQUARY_CORPUS_AUDIT_LEASE_SECONDS", 300.0, 30.0, 600.0)
    assert 30.0 <= AUDIT_LEASE_SECONDS <= 600.0


def test_work_nobody_claims_within_a_minute_is_scored_locally():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        task = asyncio.ensure_future(h.dispatcher.score(_items(model, n=1)))
        await asyncio.sleep(0)
        h.clock.now += 61
        h.dispatcher.heartbeat("pod-1")
        await h.dispatcher.sweep()
        assert [by for _, _, by in await task] == [None]

    asyncio.run(go())


def test_an_expired_lease_is_requeued_and_its_late_result_refused():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        h.dispatcher.heartbeat("pod-2")
        task = asyncio.ensure_future(h.dispatcher.score(_items(model, n=1)))
        await asyncio.sleep(0)
        stale = h.dispatcher.claim("pod-1")
        h.clock.now += 301
        h.dispatcher.heartbeat("pod-2")
        await h.dispatcher.sweep()
        fresh = h.dispatcher.claim("pod-2")
        assert fresh is not None and fresh["items"] == stale["items"]
        with pytest.raises(LeaseRefused) as refused:
            h.dispatcher.result("pod-1", stale["lease_id"], _honest_scores(model, stale))
        assert refused.value.status == 410
        h.dispatcher.result("pod-2", fresh["lease_id"], _honest_scores(model, fresh))
        assert [s for s, _, _ in await task] == ["ok"]

    asyncio.run(go())


def test_a_result_that_does_not_fit_its_lease_is_refused_and_requeued():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        task = asyncio.ensure_future(h.dispatcher.score(_items(model, n=2)))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim("pod-1")
        short = AuditResult.model_validate({"scores": [{"status": "ok", "chunks": []}]})
        with pytest.raises(LeaseRefused) as refused:
            h.dispatcher.result("pod-1", lease["lease_id"], short)
        assert refused.value.status == 422
        assert h.dispatcher.claim("pod-1") is not None
        task.cancel()

    asyncio.run(go())


def test_with_no_executor_connected_the_control_scores_locally():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        assert not h.dispatcher.connected()
        task = asyncio.ensure_future(h.dispatcher.score(_items(model, n=2)))
        await _drain(h.dispatcher, rounds=2)
        assert [s for s, _, _ in await task] == ["ok", "ok"]
        assert h.local_calls == 1

    asyncio.run(go())


def test_an_executor_silent_past_the_live_window_is_not_connected():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        h.dispatcher.heartbeat("pod-1")
        assert h.dispatcher.connected()
        h.clock.now += 91
        assert not h.dispatcher.connected()

    asyncio.run(go())


def test_a_quarantine_that_fails_to_write_is_retried_until_it_lands():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        failures = [OSError("r2 down")]

        async def flaky(executor_id, reason):
            if failures:
                raise failures.pop()
            h.quarantined.append((executor_id, reason))

        h.dispatcher._quarantine_write = flaky
        await h.dispatcher.quarantine("pod-1", "test")
        assert h.quarantined == []
        await h.dispatcher.sweep()
        assert [q[0] for q in h.quarantined] == ["pod-1"]

    asyncio.run(go())


# --------------------------------------------------------------------------
# The auditor over a dispatcher: the decision stays on the control
# --------------------------------------------------------------------------


class _AlwaysConnected:
    def __init__(self, model, lie=False, lie_pass=False, executor="pod-1"):
        self.model, self.lie, self.lie_pass, self.calls = model, lie, lie_pass, 0
        self.executor = executor
        self.listeners = []

    def connected(self):
        return True

    def subscribe(self, listener):
        self.listeners.append(listener)

    async def score(self, items):
        from reliquary.protocol.toploc import ChunkResult

        self.calls += 1
        if self.lie:
            # Claims every proof fails, to get honest miners failed.
            return [("ok", tuple(ChunkResult(10_000, 1e9, 1e9) for _ in i["proofs"]), self.executor)
                    for i in items]
        if self.lie_pass:
            return [("ok", tuple(ChunkResult(0, 0.0, 0.0) for _ in i["proofs"]), self.executor)
                    for i in items]
        scores, _, _ = score_sequences(self.model, _wire(items), chunk_tokens=PROOF.chunk_tokens,
                                       topk=PROOF.topk, batch_tokens=1 << 20)
        return [(s, c, self.executor) for s, c in scores]


def test_the_auditor_judges_remote_scores_itself_and_records_who_scored():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model, other = _tiny(0), _tiny(1)
    sid_good, sid_bad = "a" * 64, "b" * 64
    records = _Records({sid_good: _record(model), sid_bad: _record(other)})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    asyncio.run(auditor.audit_many([sid_good, sid_bad]))
    assert records.verdicts[sid_good]["passed"] is True
    assert records.verdicts[sid_good]["scored_by"] == ["pod-1"]
    assert records.verdicts[sid_bad]["passed"] is False
    assert "scored_by" not in records.verdicts[sid_bad]  # confirmed on this GPU
    assert remote.calls == 1


def test_an_executor_alone_can_never_fail_an_honest_miner():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "a" * 64
    records = _Records({sid: _record(model)})
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=_AlwaysConnected(model, lie=True))
    asyncio.run(auditor.audit_many([sid]))
    assert records.verdicts[sid]["passed"] is True


class _VoidingRecords(_Records):
    def __init__(self, submissions):
        super().__init__(submissions)
        self.voided, self.settlement = {}, {}

    async def write_voided(self, job_id, sid, document):
        # Create-only, as `corpus_record_store.write_voided`: the first void of a submission stands.
        if sid in self.voided:
            return False
        self.voided[sid] = document
        return True

    async def read_voided(self, job_id, sid):
        return self.voided.get(sid)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return dict(self.settlement), None


class _States:
    def __init__(self):
        from reliquary.corpus.audit_policy import MinerState

        self.states = {}
        self._blank = MinerState
        # Writes of miners.json: one per update_many, even one that changes no hotkey (`MinerStates.update_many`).
        self.writes = 0

    async def get(self, hotkey):
        return self.states.get(hotkey, self._blank())

    async def update(self, hotkey, change):
        self.states[hotkey] = change(await self.get(hotkey))
        return self.states[hotkey]

    async def update_many(self, changes):
        self.writes += 1
        return {hotkey: await self.update(hotkey, change) for hotkey, change in changes.items()}


def test_a_quarantine_reaudits_every_pass_its_executor_scored_and_penalises_the_miner():
    """C1(c): collusion has a cost. Passes the liar got written are re-audited
    here; a failure is charged to the miner and the submission voided."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model, other = _tiny(0), _tiny(1)
    cheat, honest, old = "a" * 64, "b" * 64, "c" * 64
    records = _VoidingRecords({cheat: _record(other), honest: _record(model),
                               old: _record(other)})
    states = _States()
    remote = _AlwaysConnected(model, lie_pass=True)
    clock = _Clock(100_000.0)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote, miner_states=states, clock=clock)
    asyncio.run(auditor.audit_many([old]))
    clock.now += 5000  # `old` is settled and now outside the hold window
    asyncio.run(auditor.audit_many([cheat, honest]))
    assert all(records.verdicts[s]["passed"] for s in (cheat, honest, old))
    records.settlement = {"settled": [old]}
    voided = asyncio.run(remote.listeners[0]("pod-1"))
    assert voided == [cheat]
    assert set(records.voided) == {cheat}
    assert states.states["5Hot"].confirmed_failures  # the same path as a failed audit
    # The settler pays nothing for a voided pass.
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
    from tests.unit.test_corpus_settlement import _Archives

    records.verdicts = {cheat: records.verdicts[cheat]}
    records.state, records.etag = {}, None

    async def read_settlement(job_id):
        return dict(records.state), records.etag

    async def write_settlement(job_id, state, etag):
        records.state, records.etag = dict(state), "e"
        return "e"

    records.read_settlement, records.write_settlement = read_settlement, write_settlement
    archives = _Archives()
    # Every period of the verdict closed: genesis 0, the clock far past it.
    settler = CorpusPeriodSettler(task_id="t", job_id="j", cap=0.1, records=records,
                                  archives=archives, oldest_pending=lambda: None,
                                  genesis=lambda: 0.0, clock=lambda: 1e12)
    asyncio.run(settler.settle_once())
    assert archives.written == {} and records.state["settled"] == [cheat]


# A forced-draw refusal, charged (no "penalty" key); with "penalty": False, a kernel drift (`eval_draw.judge`).
_DRAW_REFUSAL = {"passed": False, "reason": "draw_mismatch", "forced_draw": {"max_gap": 5.3}}


def test_a_quarantine_voids_a_repeated_drift_without_charging_its_miner(monkeypatch):
    """A forced-draw refusal without penalty (kernel drift, ``eval_draw.judge``) that the second audit repeats
    is no confirmed failure, as in a judging pass: its pass is voided, as that pass would have refused it, and
    its miner is not charged. A refusal with penalty is still charged."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    drift, cheat = "a" * 64, "b" * 64
    records = _VoidingRecords({drift: {**_record(model), "hotkey": "5Drift"},
                               cheat: {**_record(model), "hotkey": "5Cheat"}})
    states = _States()
    remote = _AlwaysConnected(model, lie_pass=True)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote, miner_states=states, clock=_Clock(100_000.0))
    asyncio.run(auditor.audit_many([drift, cheat]))
    assert all(records.verdicts[s]["passed"] for s in (drift, cheat))
    audits = []

    def refuse(batch):
        audits.append([r["hotkey"] for r in batch])
        return [{**_DRAW_REFUSAL, **({"penalty": False} if r["hotkey"] == "5Drift" else {})} for r in batch]

    monkeypatch.setattr(auditor, "_judge_many", refuse)
    voided = asyncio.run(remote.listeners[0]("pod-1"))
    assert audits == [["5Drift", "5Cheat"], ["5Drift"], ["5Cheat"]]  # each refusal audited twice
    # Both withdrawn and returned; the eval auditor reopens their slots (`EvalAuditor._reopen_reaudited`).
    assert voided == [drift, cheat] and set(records.voided) == {drift, cheat}
    assert records.voided[drift]["penalty"] is False
    kept = states.states["5Drift"]
    assert kept.confirmed_failures == [] and kept.failure_ids == [] and kept.suspect_until is None
    charged = states.states["5Cheat"]
    assert charged.confirmed_failures and charged.suspect_until is not None and charged.audited_passed == 0


def _sampled_reaudit(monkeypatch, refusal, *, voids_listed=False, q=0.25, settled=False):
    """At q = 0.25, two hotkeys of one coldkey: a pass of 5Drift and one of 5Sib scored by pod-2, then a later
    pass of 5Drift scored by pod-1. pod-1 is quarantined, and both audits of its pass return ``refusal``.
    ``run.events``: from the re-audit on, each slot reopen and each void the status hears of, in order (with no
    slot store here, the spies stand for a draw job's ledger). ``voids_listed``: an earlier forfeit of this
    process has already listed the job's voids (`_void_forfeited`). ``q``: the job's share. ``settled``: pod-1's
    pass was settled (inside the hold window) before the re-audit."""
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    run = SimpleNamespace(earlier="a" * 64, sibling="b" * 64, drift="c" * 64, states=_States(), escalated=[],
                          events=[])
    run.records = _VoidingRecords({
        run.earlier: {**_record(model), "hotkey": "5Drift", "received_at": 99_000.0},
        run.sibling: {**_record(model), "hotkey": "5Sib", "received_at": 99_100.0},
        run.drift: {**_record(model), "hotkey": "5Drift", "received_at": 99_200.0}})
    remote = _AlwaysConnected(model, lie_pass=True, executor="pod-2")
    run.auditor = auditor = CorpusAuditor(
        job_id="j", records=run.records, model=model, tokenizer=_Tokenizer(), proof=PROOF, remote=remote,
        miner_states=run.states, clock=_Clock(100_000.0), params=AuditParams(q=q),
        operator_of={"5Drift": "5Cold", "5Sib": "5Cold"}.get,
        on_voided=lambda sid, document: run.events.append(("voided", sid)))
    if voids_listed:
        auditor._voided_ids = set()
    asyncio.run(auditor.audit_many([run.earlier, run.sibling]))
    remote.executor = "pod-1"
    asyncio.run(auditor.audit_many([run.drift]))
    assert all(run.records.verdicts[s]["passed"] for s in (run.earlier, run.sibling, run.drift))
    if settled:
        run.records.settlement = {"settled": [run.drift]}
    run.before, run.writes = dict(run.states.states), run.states.writes
    escalate = auditor._escalate

    async def spy(failures, *args, **kwargs):
        run.escalated.append(sorted(failures))
        return await escalate(failures, *args, **kwargs)

    async def reopen(submission_ids):
        if submission_ids:
            run.events.append(("reopen", sorted(submission_ids)))

    async def reopen_one(submission_id):
        await reopen([submission_id])

    monkeypatch.setattr(auditor, "_escalate", spy)
    monkeypatch.setattr(auditor, "_reopen_slots", reopen)
    monkeypatch.setattr(auditor, "_reopen_slot", reopen_one)
    monkeypatch.setattr(auditor, "_judge_many", lambda batch: [dict(refusal) for _ in batch])
    run.voided = asyncio.run(remote.listeners[0]("pod-1"))
    return run


def test_on_a_sampled_job_a_reaudited_drift_forfeits_nothing_and_moves_no_state(monkeypatch):
    """At q < 1 a charged failure forfeits the unpaid passes its hotkey and its operator's other hotkeys received
    up to it, and turns those other hotkeys suspect (`_escalate`). A re-audited drift is not charged: its pass
    alone is voided. With every refusal the re-audit repeats a drift, nothing is escalated and miners.json is not
    written."""
    run = _sampled_reaudit(monkeypatch, {**_DRAW_REFUSAL, "penalty": False})
    assert run.voided == [run.drift] and set(run.records.voided) == {run.drift}
    assert run.records.voided[run.drift]["penalty"] is False
    assert run.escalated == [] and run.states.writes == run.writes
    # Neither state moves: nothing is charged, and nothing is withdrawn either (5Drift keeps the probation credit
    # its voided pass earned when it was written, a known gap).
    assert run.states.states == run.before


def test_on_a_sampled_job_a_reaudited_charged_refusal_still_forfeits_its_operators_unpaid_passes(monkeypatch):
    """The same pass refused with penalty is charged: its hotkey's earlier pass and its operator's other hotkey's
    pass are forfeited."""
    run = _sampled_reaudit(monkeypatch, _DRAW_REFUSAL)
    assert run.voided == [run.drift] and run.escalated == [["5Drift"]]
    assert set(run.records.voided) == {run.earlier, run.sibling, run.drift}
    assert run.records.voided[run.earlier]["reason"] == run.records.voided[run.sibling]["reason"] == "forfeited"
    charged, sibling = run.states.states["5Drift"], run.states.states["5Sib"]
    assert charged.confirmed_failures and charged.forfeit_before == 99_200.0
    # Suspect and forfeiting, but no failure of its own (`after_operator_failure`).
    assert sibling.suspect_until is not None and sibling.forfeit_before == 99_200.0
    assert sibling.confirmed_failures == []
    # Its slot reopens with the forfeit of its own pass, as before: nothing reopens it a second time.
    assert [event for event in run.events if event[0] == "reopen"] == [
        ("reopen", sorted([run.earlier, run.sibling, run.drift]))]


def test_on_a_sampled_job_a_reaudited_drift_reopens_its_slot_at_once(monkeypatch):
    """On a draw job the drift's slot reopens during the re-audit, before the status hears of its void, as a
    judging pass reopens a refused sample's (`_write`). Its own forfeit used to reopen it (`_void_forfeited`); with
    none, it would wait for the next start's reconcile, its prompt full without the sample."""
    run = _sampled_reaudit(monkeypatch, {**_DRAW_REFUSAL, "penalty": False})
    assert run.events == [("reopen", [run.drift]), ("voided", run.drift)]


def test_a_later_forfeit_does_not_void_a_reaudited_drift_again(monkeypatch):
    """A drift the re-audit voided is out of the forfeit books: a later charged failure of its hotkey forfeits that
    hotkey's other unpaid pass and its operator's, and leaves the drift's void, slot and status alone. The job's
    voids were listed before the re-audit (listed after it, they would hold the drift's void anyway)."""
    run = _sampled_reaudit(monkeypatch, {**_DRAW_REFUSAL, "penalty": False}, voids_listed=True)
    run.events.clear()
    later = "d" * 64
    asyncio.run(run.auditor._escalate({"5Drift": [later]}, 100_000.0, {later: 99_300.0}))
    assert run.escalated == [["5Drift"]]
    assert run.records.voided[run.drift]["executor_id"] == "pod-1"  # the re-audit's void, not a forfeit's
    assert run.records.voided[run.earlier]["reason"] == run.records.voided[run.sibling]["reason"] == "forfeited"
    assert run.events == [("reopen", [run.earlier, run.sibling]), ("voided", run.earlier), ("voided", run.sibling)]


def test_an_eval_auditor_reopens_a_reaudited_drift_in_its_ledger_once(monkeypatch):
    """The eval auditor reopens every pass its re-audit voids in its own ledger (`EvalAuditor._reopen_reaudited`):
    the base re-audit's reopen of a drift adds no second ledger write there."""
    from reliquary.validator import corpus_service, eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    job, drift = draw_job(), "c" * 64
    records = _VoidingRecords({drift: {"hotkey": "5Drift", "token_count": 10, "prompt_index": job.prompt_start,
                                       "cursor": 0}})
    ledger = []

    async def record_prompt_failure(store, job, prompt_index, submission_id, **slot):
        ledger.append(submission_id)
        return True

    async def record_prompt_failures(store, job, entries):
        ledger.extend(entry[1] for entry in entries)
        return {entry[1]: True for entry in entries}

    monkeypatch.setattr(corpus_service, "record_prompt_failure", record_prompt_failure)
    monkeypatch.setattr(corpus_service, "record_prompt_failures", record_prompt_failures)
    auditor = eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                        vocab_size=10, job=job, job_store=object(), reopen_slots=True,
                                        draw_job=job)
    auditor._remote_scored["pod-1"] = [drift]

    async def refuse(batch, *, local=False):
        return [{**_DRAW_REFUSAL, "penalty": False} for _ in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", refuse)
    assert asyncio.run(auditor.reaudit_executor("pod-1")) == [drift]
    assert records.voided[drift]["penalty"] is False and ledger == [drift]


@pytest.mark.parametrize("fails", [True, False])
def test_an_eval_auditor_reopens_the_drifts_once_every_void_is_written(monkeypatch, fails):
    """Every void is written first: a drift's ledger write (a record read, a ledger read and rebuild, a CAS, up to 4
    tries) no longer holds back the next voids while the settler may pay those passes. The drifts' slots then reopen
    in the eval auditor's ledger, each before the status hears of its void; a charged refusal's is reported at once
    and reopens after, as before. ``fails``: the last void write raises (a store PUT failing after its retries),
    which stops the re-audit (the quarantine's listener only logs it): the drift voided before it still reopens, and
    does not wait for the next start's reconcile."""
    from reliquary.validator import corpus_service, eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    first, cheat, second = "c" * 64, "d" * 64, "e" * 64
    events = []

    class _FailingRecords(_VoidingRecords):
        async def write_voided(self, job_id, sid, document):
            if fails and sid == second:
                raise OSError("store PUT failed after its retries")
            events.append(("void", sid))
            return await super().write_voided(job_id, sid, document)

    records = _FailingRecords({sid: {"hotkey": hotkey, "token_count": 10, "prompt_index": job.prompt_start,
                                     "cursor": 0}
                               for sid, hotkey in ((first, "5Drift"), (cheat, "5Cheat"), (second, "5Drift"))})

    async def record_prompt_failure(store, job, prompt_index, submission_id, **slot):
        events.append(("reopen", submission_id))
        return True

    monkeypatch.setattr(corpus_service, "record_prompt_failure", record_prompt_failure)
    auditor = eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                        vocab_size=10, job=job, job_store=object(), reopen_slots=True,
                                        draw_job=job, on_voided=lambda sid, document: events.append(("voided", sid)))
    auditor._remote_scored["pod-1"] = [first, cheat, second]

    async def refuse(batch, *, local=False):
        return [{**_DRAW_REFUSAL, **({"penalty": False} if r["hotkey"] == "5Drift" else {})} for r in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", refuse)
    if fails:
        with pytest.raises(OSError):
            asyncio.run(auditor.reaudit_executor("pod-1"))
        assert events == [("void", first), ("void", cheat), ("voided", cheat), ("reopen", first), ("voided", first)]
        assert set(records.voided) == {first, cheat}
    else:
        assert asyncio.run(auditor.reaudit_executor("pod-1")) == [first, cheat, second]
        assert events == [("void", first), ("void", cheat), ("voided", cheat), ("void", second),
                          ("reopen", first), ("reopen", second), ("voided", first), ("voided", second),
                          ("reopen", cheat)]


def test_a_reaudit_charges_on_its_second_audit_run_alone(monkeypatch):
    """§7.2: the second audit of a re-audited refusal, run on this GPU and alone, decides whether it is charged.
    Here the first audit refuses 5Drift with a penalty and 5Cheat without; the second, the other way round:
    5Drift is voided uncharged and 5Cheat is charged."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    drift, cheat = "a" * 64, "b" * 64
    records = _VoidingRecords({drift: {**_record(model), "hotkey": "5Drift"},
                               cheat: {**_record(model), "hotkey": "5Cheat"}})
    states = _States()
    remote = _AlwaysConnected(model, lie_pass=True)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote, miner_states=states, clock=_Clock(100_000.0))
    asyncio.run(auditor.audit_many([drift, cheat]))
    audits = []

    def refuse(batch):
        audits.append([r["hotkey"] for r in batch])
        uncharged = "5Cheat" if len(audits) == 1 else "5Drift"
        return [{**_DRAW_REFUSAL, **({"penalty": False} if r["hotkey"] == uncharged else {})} for r in batch]

    monkeypatch.setattr(auditor, "_judge_many", refuse)
    assert asyncio.run(remote.listeners[0]("pod-1")) == [drift, cheat]
    assert audits == [["5Drift", "5Cheat"], ["5Drift"], ["5Cheat"]]
    assert records.voided[drift]["penalty"] is False and "penalty" not in records.voided[cheat]
    kept, charged = states.states["5Drift"], states.states["5Cheat"]
    assert kept.confirmed_failures == [] and kept.suspect_until is None
    assert charged.failure_ids == [cheat] and charged.suspect_until is not None


@pytest.mark.parametrize("q, settled", [(1.0, False), (0.25, True)])
def test_a_reaudited_drift_reopens_its_slot_where_no_forfeit_would(monkeypatch, q, settled):
    """At q = 1 nothing is forfeited, and a pass already settled (inside the hold window) never is
    (`_void_forfeited`): the re-audit reopens the drift's slot itself, before the status hears of its void."""
    run = _sampled_reaudit(monkeypatch, {**_DRAW_REFUSAL, "penalty": False}, q=q, settled=settled)
    assert run.voided == [run.drift] and run.escalated == []
    assert run.events == [("reopen", [run.drift]), ("voided", run.drift)]


_PASSING = {"passed": True, "worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0}


def _draw_reaudit(executors=("pod-1",)):
    """A forced-draw job at q = 1 whose other hotkeys passed 2,000 audits. ``run.ids``: three records of 5X, and
    ``run.cheat`` one of 5Cheat. A judging pass takes ``executors``' passing scores (an executor pair lists both);
    this GPU refuses 5X's as drifts and 5Cheat's with a penalty."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    run = SimpleNamespace(ids=[f"{i:064x}" for i in range(1, 4)], cheat="f" * 64, states=_States())
    run.records = _VoidingRecords({**{sid: {"hotkey": "5X", "token_count": 10} for sid in run.ids},
                                   run.cheat: {"hotkey": "5Cheat", "token_count": 10}})
    run.auditor = auditor = CorpusAuditor(job_id="j", records=run.records, model=None, tokenizer=None,
                                          proof=PROOF, miner_states=run.states, clock=_Clock(100_000.0))
    auditor._draw_job = object()
    for i in range(2000):
        auditor._count_refusal(f"h{i}", f"5Other{i % 50}", {"passed": True})

    async def forward(records, *, local=False):
        if local:
            return [{**_DRAW_REFUSAL, **({"penalty": False} if r["hotkey"] == "5X" else {})} for r in records]
        return [{**_PASSING, "scored_by": list(executors)} for _ in records]

    auditor._forward = forward
    return run


def test_a_reaudited_drift_counts_as_a_refusal_and_frequent_ones_are_charged():
    """A judging pass counts a drift as a refusal of its hotkey, and charges the one that leaves the hotkey refused
    far more often than the job's other hotkeys (`_count_refusal`). The lying executor's passes were counted as
    passing audits: the re-audit counts each drift it repeats as a refusal instead, once, and charges the third
    of 5X's three as that pass would. 5Cheat's penalised refusal is counted as a refusal too, as that pass counts
    it, and charged as before."""
    run = _draw_reaudit(executors=("pod-1", "pod-2"))
    auditor, records = run.auditor, run.records
    asyncio.run(auditor.audit_many([*run.ids, run.cheat]))
    assert auditor._draw_counts["5X"] == [3, 0] and auditor._draw_totals == [2004, 0]
    assert asyncio.run(auditor.reaudit_executor("pod-1")) == [*run.ids, run.cheat]
    assert auditor._draw_counts["5X"] == [3, 3] and auditor._draw_totals == [2004, 4]
    assert auditor._draw_counts["5Cheat"] == [1, 1]
    first, second, third = (records.voided[sid] for sid in run.ids)
    assert first["penalty"] is False and second["penalty"] is False
    assert third["penalty"] is True and third["charged_for"] == "frequent_refusals"
    assert "penalty" not in records.voided[run.cheat] and "charged_for" not in records.voided[run.cheat]
    charged = run.states.states["5X"]
    assert charged.failure_ids == [run.ids[2]] and charged.suspect_until is not None
    assert run.states.states["5Cheat"].failure_ids == [run.cheat]
    # The pair's other executor quarantined too: the same refusals are re-audited again, and counted once.
    asyncio.run(auditor.reaudit_executor("pod-2"))
    assert auditor._draw_counts["5X"] == [3, 3] and run.states.states["5X"].failure_ids == [run.ids[2]]
    assert auditor._draw_counts["5Cheat"] == [1, 1] and auditor._draw_totals == [2004, 4]


def test_after_a_restart_a_reaudited_drift_is_counted_in_full():
    """The counts are kept per process: a pass written before a restart was counted by no judging pass of this
    one. Its re-audited drift is counted as an audit and a refusal, as a judging pass counts one."""
    run = _draw_reaudit()
    # What `restore_scored` lists again after a restart: pod-1's passes, written by the process before.
    run.auditor._remote_scored["pod-1"] = list(run.ids)
    asyncio.run(run.auditor.reaudit_executor("pod-1"))
    assert run.auditor._draw_counts.get("5X") == [3, 3] and run.auditor._draw_totals == [2003, 3]
    assert run.records.voided[run.ids[2]]["charged_for"] == "frequent_refusals"
    assert run.states.states["5X"].failure_ids == [run.ids[2]]


def test_after_a_restart_the_passes_a_reaudit_confirms_are_counted_too(monkeypatch):
    """A judge process that restarted lists the quarantined executor's passes written before it again
    (`EvalAuditor.restore_scored`) and re-audits them, its counts started afresh. 200 passes of 5X, 3 refused now as
    drifts: the 197 it confirms are counted as passing audits, as the judging pass that wrote them counted them (one,
    refused by its first audit, is confirmed by its second, which decides, as there). 5X is then refused 3 times in
    200 audits, not far above the job's other hotkeys (0 in 2,000): nothing is charged, as in a process that never
    restarted. With its drifts counted alone, 5X was refused 3 times in 3 and charged."""
    from reliquary.validator import eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    ids = [f"{i:064x}" for i in range(1, 201)]
    records = _VoidingRecords({sid: {"hotkey": "5X", "token_count": 10, "prompt_index": job.prompt_start,
                                     "cursor": 0, "drift": k < 3, "flaky": k == 3} for k, sid in enumerate(ids)})
    for sid in ids:  # written before the restart, from pod-1's scores
        records.verdicts[sid] = {"submission_id": sid, "hotkey": "5X", "passed": True, "audited": True,
                                 "audited_at": 100_000.0, "received_at": 99_000.0, "scored_by": ["pod-1"]}
    states = _States()
    remote = SimpleNamespace(connected=lambda: True, subscribe=lambda listener: None, quarantined=lambda: {"pod-1"})
    auditor = eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                        vocab_size=10, job=job, job_store=object(), draw_job=job, remote=remote,
                                        miner_states=states, clock=_Clock(100_100.0), restores_scored=True)
    for i in range(2000):
        auditor._count_refusal(f"h{i}", f"5Other{i % 50}", {"passed": True})

    async def audit(batch, *, local=False):
        # The flaky record is refused in the batch of every restored pass, and passes its second audit, run alone.
        return [{**_DRAW_REFUSAL, "penalty": False} if r["drift"] or (r["flaky"] and len(batch) > 1)
                else dict(_PASSING) for r in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    assert asyncio.run(auditor.restore_scored()) == 200
    assert auditor._draw_counts["5X"] == [200, 3] and auditor._draw_totals == [2200, 3]
    assert sorted(records.voided) == ids[:3] and all(records.voided[sid]["penalty"] is False for sid in ids[:3])
    assert "5X" not in states.states  # no charge: no failure, no suspicion


def _frequent_drifts(monkeypatch, kind="base", *, q=1.0, settled=False, penalised=(), lying=True, reaudit=True):
    """A forced-draw job at share ``q`` whose other hotkeys passed 2,000 audits: three records of 5X, received 10 s
    apart, that this GPU refuses as drifts (with a penalty: the indices in ``penalised``). ``lying``: a judging pass
    took pod-1's passing scores for them, and pod-1's quarantine re-audits them (``run.voided``; not run without
    ``reaudit``); else that pass refused them itself. The third refusal crosses the frequent-refusals line. ``kind``:
    "base", a CorpusAuditor whose slot store the spies stand for, or "eval", an EvalAuditor and its ledger.
    ``settled``: the three were settled (inside the hold window) before the re-audit. ``run.events``: from the re-audit
    on, each slot reopen and each void the status hears of, in order."""
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.validator import corpus_service, eval_control
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    run = SimpleNamespace(ids=[f"{i:064x}" for i in range(1, 4)], states=_States(), events=[], voided=None)
    run.records = _VoidingRecords({sid: {"hotkey": "5X", "token_count": 10, "prompt_index": job.prompt_start,
                                         "cursor": 0, "received_at": 99_000.0 + 10 * k, "penalised": k in penalised}
                                   for k, sid in enumerate(run.ids)})
    common = dict(job_id=job.job_id, records=run.records, tokenizer=None, proof=PROOF, miner_states=run.states,
                  clock=_Clock(100_000.0), params=AuditParams(q=q), draw_job=job,
                  on_voided=lambda sid, document: run.events.append(("voided", sid)))

    async def reopen(submission_ids):
        if submission_ids:
            run.events.append(("reopen", sorted(submission_ids)))

    if kind == "base":
        run.auditor = CorpusAuditor(model=None, **common)

        async def reopen_one(submission_id):
            await reopen([submission_id])

        monkeypatch.setattr(run.auditor, "_reopen_slots", reopen)
        monkeypatch.setattr(run.auditor, "_reopen_slot", reopen_one)
    else:
        async def record_prompt_failure(store, job, prompt_index, submission_id, **slot):
            await reopen([submission_id])
            return True

        async def record_prompt_failures(store, job, entries, **kw):
            await reopen([entry[1] for entry in entries])
            return {entry[1]: True for entry in entries}

        monkeypatch.setattr(corpus_service, "record_prompt_failure", record_prompt_failure)
        monkeypatch.setattr(corpus_service, "record_prompt_failures", record_prompt_failures)
        run.auditor = eval_control.eval_auditor(vocab_size=10, job=job, job_store=object(), reopen_slots=True,
                                                **common)
    for i in range(2000):
        run.auditor._count_refusal(f"h{i}", f"5Other{i % 50}", {"passed": True})

    async def audit(batch, *, local=False):
        if lying and not local:
            return [{**_PASSING, "scored_by": ["pod-1"]} for _ in batch]
        return [{**_DRAW_REFUSAL, **({} if r["penalised"] else {"penalty": False})} for r in batch]

    monkeypatch.setattr(run.auditor, "_audit_outcomes", audit)
    asyncio.run(run.auditor.audit_many(run.ids))
    if lying:
        assert all(run.records.verdicts[sid]["passed"] for sid in run.ids)
        if settled:
            run.records.settlement = {"settled": list(run.ids)}
        run.events.clear()
        if reaudit:
            run.voided = asyncio.run(run.auditor.reaudit_executor("pod-1"))
    return run


@pytest.mark.parametrize("kind, q, settled", [("base", 1.0, False), ("base", 0.25, True),
                                              ("eval", 1.0, False), ("eval", 0.25, True)])
def test_a_drift_charged_for_frequent_refusals_reopens_its_slot_with_the_others(monkeypatch, kind, q, settled):
    """A drift the re-audit charges for frequent refusals is a refused sample like the others: its slot reopens with
    theirs once every void is written, before the status hears of its void, as a judging pass reopens every refused
    sample's (`_write`), charged or not. At q = 1, or settled inside the hold window, no forfeit reopens it; the base
    auditor used to leave it to the next start's reconcile, the eval auditor to reopen it after every void. Each slot
    reopens once: the eval auditor does not reopen the charged one again after the loop."""
    run = _frequent_drifts(monkeypatch, kind, q=q, settled=settled)
    assert run.voided == run.ids
    assert run.records.voided[run.ids[2]]["charged_for"] == "frequent_refusals"
    assert run.states.states["5X"].failure_ids == [run.ids[2]]
    assert run.events == [("reopen", [sid]) for sid in run.ids] + [("voided", sid) for sid in run.ids]


def test_a_reaudit_counts_a_penalised_refusal_with_its_hotkeys_drifts(monkeypatch):
    """5X's first record is refused with a penalty, the next two as drifts. A judging pass counts all three as
    refusals, and the third, 3 refused in 3 audits, is charged for frequent refusals. The re-audit of the same three
    counts and charges them alike: the penalised refusal is counted too, and comes back as it is."""
    judged = _frequent_drifts(monkeypatch, penalised=(0,), lying=False)
    run = _frequent_drifts(monkeypatch, penalised=(0,))
    assert run.voided == run.ids
    assert run.auditor._draw_counts["5X"] == judged.auditor._draw_counts["5X"] == [3, 3]
    assert run.auditor._draw_totals == judged.auditor._draw_totals == [2003, 3]
    assert run.states.states["5X"].failure_ids == judged.states.states["5X"].failure_ids == [run.ids[0], run.ids[2]]
    penalised, drift, charged = (run.records.voided[sid] for sid in run.ids)
    assert "penalty" not in penalised and "charged_for" not in penalised
    assert drift["penalty"] is False and charged["charged_for"] == "frequent_refusals"


def test_a_reaudits_own_forfeit_voids_its_failures_with_the_reaudits_documents(monkeypatch):
    """At q < 1 the drift charged for frequent refusals forfeits its hotkey's unpaid passes received up to it
    (`_escalate`) before the re-audit writes any void: its own and the two drifts received before it. The forfeit
    writes the re-audit's own document for each (`_void_forfeited`): it names pod-1 and its refusal, the two first
    uncharged and the third charged, so a judge's restart counts the refusals this re-audit counted
    (`_count_restored`); a "forfeited" void would hide them. Their slots reopen, and the re-audit's own creates are
    refused (create-only, as in production)."""
    landed = []
    run = _frequent_drifts(monkeypatch, q=0.25, reaudit=False)
    write = run.records.write_voided

    async def spy(job_id, sid, document):
        landed.append(await write(job_id, sid, document))
        return landed[-1]

    run.records.write_voided = spy
    assert asyncio.run(run.auditor.reaudit_executor("pod-1")) == run.ids
    voids = [run.records.voided[sid] for sid in run.ids]
    assert [(v["executor_id"], v["reason"], v["penalty"]) for v in voids] == [
        ("pod-1", "draw_mismatch", False), ("pod-1", "draw_mismatch", False), ("pod-1", "draw_mismatch", True)]
    assert voids[2]["charged_for"] == "frequent_refusals" and "forfeit_before" not in voids[2]
    assert landed == [True] * 3 + [False] * 3  # the forfeit's three creates, then the re-audit's own, refused
    charged = run.states.states["5X"]
    assert charged.failure_ids == [run.ids[2]] and charged.forfeit_before == 99_020.0
    assert {sid for what, sids in run.events if what == "reopen" for sid in sids} == set(run.ids)


def test_a_pass_forfeited_before_its_reaudit_keeps_the_forfeits_void(monkeypatch):
    """A later charged failure of 5X, from a judging pass, forfeits its three passes before pod-1 is quarantined.
    Voids are create-only, as in production (`corpus_record_store`): the re-audit's own void of each is refused, and
    the forfeit's stands."""
    run = _frequent_drifts(monkeypatch, q=0.25, reaudit=False)
    later = "f" * 64
    asyncio.run(run.auditor._escalate({"5X": [later]}, 100_000.0, {later: 99_100.0}))
    assert [run.records.voided[sid]["reason"] for sid in run.ids] == ["forfeited"] * 3
    assert asyncio.run(run.auditor.reaudit_executor("pod-1")) == run.ids
    assert [run.records.voided[sid]["reason"] for sid in run.ids] == ["forfeited"] * 3


def test_a_penalised_refusal_that_crosses_the_line_keeps_its_labels(monkeypatch):
    """Two drifts of 5X, then a refusal with a penalty: the third refusal crosses the frequent-refusals line, but the
    rule requalifies only a refusal without penalty (`_charged_if_frequent`). The penalised one is charged as it is,
    with no ``charged_for``, by a judging pass and by a re-audit alike."""
    judged = _frequent_drifts(monkeypatch, penalised=(2,), lying=False)
    run = _frequent_drifts(monkeypatch, penalised=(2,))
    for auditor in (judged.auditor, run.auditor):
        assert auditor._draw_counts["5X"] == [3, 3]
    verdict, void = judged.records.verdicts[judged.ids[2]], run.records.voided[run.ids[2]]
    assert verdict["reason"] == void["reason"] == "draw_mismatch"
    assert "charged_for" not in verdict
    assert "charged_for" not in void and "penalty" not in void
    assert judged.states.states["5X"].failure_ids == run.states.states["5X"].failure_ids == [run.ids[2]]


def test_a_reaudit_run_again_after_its_escalation_raised_still_charges(monkeypatch):
    """The re-audit counts 5X's three drifts and charges the third for frequent refusals; then its escalation raises
    (a miners.json write failing after its retries), and nothing is voided. Run again (`_reaudit_late` lists the
    passes again), it counts nothing twice and gets the charged outcome back (`_draw_recounted`): the third is
    escalated and voided charged, not uncharged as a drift seen a second time."""
    run = _frequent_drifts(monkeypatch, reaudit=False)
    escalate, calls = run.auditor._escalate, []

    async def flaky(failures, *args, **kwargs):
        calls.append(sorted(failures))
        if len(calls) == 1:
            raise OSError("miners.json write failed after its retries")
        return await escalate(failures, *args, **kwargs)

    monkeypatch.setattr(run.auditor, "_escalate", flaky)
    with pytest.raises(OSError):
        asyncio.run(run.auditor.reaudit_executor("pod-1"))
    assert run.records.voided == {} and run.states.states["5X"].failure_ids == []
    run.auditor._remote_scored["pod-1"] = list(run.ids)
    assert asyncio.run(run.auditor.reaudit_executor("pod-1")) == run.ids
    assert calls == [["5X"], ["5X"]] and run.auditor._draw_counts["5X"] == [3, 3]
    assert run.records.voided[run.ids[2]]["charged_for"] == "frequent_refusals"
    assert run.states.states["5X"].failure_ids == [run.ids[2]]


def test_a_refusal_its_judging_pass_counted_is_not_counted_again_by_its_reaudit(monkeypatch):
    """A judging pass refuses 5X's three records as drifts and counts them; their verdict writes fail, so they stay
    pending. A later pass takes pod-1's passing scores for them and writes passes, counting nothing again. pod-1's
    re-audit refuses them as drifts: each is still one audit refused once, 3 in 3, not 6 refusals in 3 audits (on which
    `frequent_refusals` charged whatever the job's rate: its binomial tail was empty). The job's other hotkeys are
    refused half the time, so 3 in 3 is no charge."""
    from reliquary.validator import eval_control
    from reliquary.validator.corpus_auditor import frequent_refusals
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    ids = [f"{i:064x}" for i in range(1, 4)]
    phase = {"now": "refuse"}

    class _Failing(_VoidingRecords):
        async def write_verdict(self, job_id, sid, verdict):
            if phase["now"] == "refuse":
                raise OSError("store PUT failed after its retries")
            return await super().write_verdict(job_id, sid, verdict)

    records = _Failing({sid: {"hotkey": "5X", "token_count": 10, "prompt_index": job.prompt_start, "cursor": 0,
                              "received_at": 99_000.0 + k} for k, sid in enumerate(ids)})
    auditor = eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                        vocab_size=10, job=job, job_store=object(), draw_job=job,
                                        miner_states=_States(), clock=_Clock(100_000.0))
    drift = {**_DRAW_REFUSAL, "penalty": False}
    for i in range(2000):
        auditor._count_refusal(f"h{i}", f"5Other{i % 50}", dict(drift) if i % 2 else {"passed": True})

    async def audit(batch, *, local=False):
        if phase["now"] == "pass" and not local:
            return [{**_PASSING, "scored_by": ["pod-1"]} for _ in batch]
        return [dict(drift) for _ in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    with pytest.raises(OSError):
        asyncio.run(auditor.audit_many(ids))
    assert auditor._draw_counts["5X"] == [3, 3] and records.verdicts == {}
    phase["now"] = "pass"
    asyncio.run(auditor.audit_many(ids))
    assert all(records.verdicts[sid]["passed"] for sid in ids) and auditor._draw_counts["5X"] == [3, 3]
    assert asyncio.run(auditor.reaudit_executor("pod-1")) == ids
    assert auditor._draw_counts["5X"] == [3, 3] and auditor._draw_totals == [2003, 1003]
    assert all(records.voided[sid]["penalty"] is False for sid in ids)
    # More refusals than audits is no count the auditor keeps: nothing is charged on it.
    assert not frequent_refusals(4, 2, 0, 0) and not frequent_refusals(6, 3, 1000, 2000)


def _restart(records, states, job, *, q=1.0, quarantined=("pod-1",), clock=100_100.0):
    """A judge process started afresh (`restores_scored`) over the verdicts, voids and miners of the one before it, its
    draw counts empty, with ``quarantined`` already quarantined."""
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.validator import eval_control

    remote = SimpleNamespace(connected=lambda: True, subscribe=lambda listener: None,
                             quarantined=lambda: set(quarantined))
    return eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF, vocab_size=10,
                                     job=job, job_store=object(), draw_job=job, remote=remote, miner_states=states,
                                     clock=_Clock(clock), params=AuditParams(q=q), restores_scored=True)


def _written(records, sid, hotkey, *, scored_by=None, received_at=99_000.0, **outcome):
    """A verdict a judging pass of an earlier process wrote."""
    records.submissions[sid] = {"hotkey": hotkey, "token_count": 10, "prompt_index": 0, "cursor": 0,
                                "received_at": received_at}
    records.verdicts[sid] = {"submission_id": sid, "hotkey": hotkey, "passed": True, "audited": True,
                             "audited_at": 100_000.0, "received_at": received_at,
                             **({"scored_by": list(scored_by)} if scored_by else {}), **outcome}


def test_a_restart_counts_every_audit_written_before_it_as_its_judging_pass_did(monkeypatch):
    """`restore_scored` first counts every verdict a judging pass counted (`_count_restored`): an audited pass, of any
    executor or audited here, settled or not, voided or not, or forfeited as it was written; a forced-draw refusal,
    penalised or not. Not an unaudited pass or ban, nor a failure of another kind. A voided audited pass an executor
    scored has its void read: a re-audit's (it names its executor) of a forced-draw refusal adds that refusal,
    penalised, charged for frequent refusals or not, and keeps its outcome; a forfeit's, or a re-audit's of another
    failure, adds nothing. Then pod-1's unvoided pass is re-audited, and confirmed: nothing more is counted."""
    from reliquary.validator.corpus_auditor import _FORFEITED
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    records, states = _VoidingRecords({}), _States()
    sid = iter(f"{i:064x}" for i in range(1, 100))
    p1, p2, p3, s1 = (next(sid) for _ in range(4))
    _written(records, p1, "5A", scored_by=["pod-1"])
    _written(records, p2, "5A", scored_by=["pod-2"])
    _written(records, p3, "5A")  # audited here
    _written(records, next(sid), "5A", **{**_DRAW_REFUSAL, "penalty": False})
    _written(records, s1, "5A", scored_by=["pod-1"], audited_at=1_000.0)  # settled outside the hold window
    records.settlement = {"settled": [s1]}
    _written(records, next(sid), "5B", **_DRAW_REFUSAL)
    _written(records, next(sid), "5B", passed=False, reason="bad_tokens")
    _written(records, next(sid), "5B", **{**_FORFEITED, "audited": True})
    _written(records, next(sid), "5B", audited=False)
    _written(records, next(sid), "5B", passed=False, audited=False, reason="banned")
    voids = {
        "drift": {"executor_id": "pod-1", **_DRAW_REFUSAL, "penalty": False},
        "penalised": {"executor_id": "pod-1", **_DRAW_REFUSAL},
        "charged": {"executor_id": "pod-1", **_DRAW_REFUSAL, "penalty": True, "charged_for": "frequent_refusals"},
        "other": {"executor_id": "pod-2", "passed": False, "reason": "bad_tokens"},
        "forfeit": {"reason": "forfeited", "forfeit_before": 99_500.0},
    }
    voided = {}
    for kind, document in voids.items():
        voided[kind] = next(sid)
        _written(records, voided[kind], "5C", scored_by=[document.get("executor_id", "pod-1")])
        records.voided[voided[kind]] = {"schema": "reliquary/corpus-voided/v1", "submission_id": voided[kind],
                                        "hotkey": "5C", "voided_at": 100_050.0, **document}
    read_voids = []
    read_voided = records.read_voided

    async def spy(job_id, submission_id):
        read_voids.append(submission_id)
        return await read_voided(job_id, submission_id)

    records.read_voided = spy
    auditor = _restart(records, states, job)
    reaudited = []

    async def confirm(batch, *, local=False):
        reaudited.extend(r["hotkey"] for r in batch)
        return [dict(_PASSING) for _ in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", confirm)
    assert asyncio.run(auditor.restore_scored()) == 2  # p1 for pod-1, p2 for pod-2
    assert reaudited == ["5A"] and auditor._remote_scored == {"pod-2": [p2]}
    assert auditor._draw_counts == {"5A": [5, 1], "5B": [2, 1], "5C": [5, 3]}
    assert auditor._draw_totals == [12, 5]
    assert sorted(read_voids) == sorted(voided.values())
    recounted = auditor._draw_recounted
    assert sorted(recounted) == sorted(voided[kind] for kind in ("drift", "penalised", "charged"))
    assert recounted[voided["drift"]]["penalty"] is False and "penalty" not in recounted[voided["penalised"]]
    assert recounted[voided["charged"]]["charged_for"] == "frequent_refusals"
    assert "executor_id" not in recounted[voided["drift"]] and recounted[voided["drift"]]["reason"] == "draw_mismatch"
    assert states.states == {}  # nothing charged


def _population(records, *, passes=2000, refusals=40, executor="pod-1", hotkeys=50):
    """The job's other hotkeys before a restart: ``passes`` passes scored by ``executor`` and ``refusals`` drifts."""
    for i in range(passes + refusals):
        _written(records, f"{0xA0000 + i:064x}", f"5Other{i % hotkeys}", scored_by=[executor],
                 **({} if i < passes else {**_DRAW_REFUSAL, "penalty": False}))


def test_after_a_restart_a_fresh_hotkey_is_judged_against_the_jobs_whole_rate(monkeypatch):
    """2,000 passes of pod-1 and 40 drifts written before a restart, pod-1 quarantined: its re-audit confirms the
    passes. Counted afresh, the job's totals read 2,000 passes and no refusal, and a fresh honest hotkey refused 3
    times in 50 audits was charged for frequent refusals. Its rate counted whole (2 %), it is not, as in a process
    that never restarted."""
    from tests.unit.test_eval_draw import _job as draw_job

    records = _VoidingRecords({})
    _population(records)
    auditor = _restart(records, _States(), draw_job())

    async def confirm(batch, *, local=False):
        return [dict(_PASSING) for _ in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", confirm)
    assert asyncio.run(auditor.restore_scored()) == 2000
    assert auditor._draw_totals == [2040, 40]
    drift = {**_DRAW_REFUSAL, "penalty": False}
    outcomes = [auditor._count_refusal(f"fresh{k}", "5Fresh", dict(drift) if k >= 47 else {"passed": True})
                for k in range(50)]
    assert auditor._draw_counts["5Fresh"] == [50, 3]
    assert not any(outcome.get("charged_for") for outcome in outcomes)


def test_after_a_restart_a_hotkeys_passes_of_a_healthy_executor_count_too(monkeypatch):
    """5Y's 100 passes scored by pod-2, healthy, and 3 more by pod-1, which its re-audit refuses as drifts. Counted
    afresh, only pod-1's passes were re-audited, so 5Y read 3 refusals in 3 audits and was charged. Its 103 audits
    counted, 3 refusals are not far above the job's 2 %: nothing is charged, as in a process that never restarted."""
    from tests.unit.test_eval_draw import _job as draw_job

    records, states = _VoidingRecords({}), _States()
    _population(records)
    drifts = [f"{0xB0000 + i:064x}" for i in range(3)]
    for i in range(100):
        _written(records, f"{0xC0000 + i:064x}", "5Y", scored_by=["pod-2"])
    for sid in drifts:
        _written(records, sid, "5Y", scored_by=["pod-1"])
    auditor = _restart(records, states, draw_job())

    async def audit(batch, *, local=False):
        return [{**_DRAW_REFUSAL, "penalty": False} if r["hotkey"] == "5Y" else dict(_PASSING) for r in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    asyncio.run(auditor.restore_scored())
    assert auditor._draw_counts["5Y"] == [103, 3]
    assert [records.voided[sid]["penalty"] for sid in drifts] == [False] * 3 and "5Y" not in states.states


def test_after_a_restart_the_reaudit_charges_the_latest_drift(monkeypatch):
    """Three passes of 5X scored by pod-1, received 10 s apart, their ids in the reverse order, and a fourth of 5X by
    pod-2 received between the last two. After a restart pod-1's passes are listed in id order; the re-audit counts
    its drifts in receipt order, as the judging passes took them, so the frequent-refusals charge lands on the latest
    drift, and at q < 1 its forfeit reaches the pod-2 pass. In id order it landed on the oldest, and forfeited only
    up to it."""
    from tests.unit.test_eval_draw import _job as draw_job

    records, states = _VoidingRecords({}), _States()
    _population(records, refusals=0)
    oldest, middle, latest, honest = "c" * 64, "b" * 64, "a" * 64, "d" * 64
    for k, sid in enumerate((oldest, middle, latest)):
        _written(records, sid, "5X", scored_by=["pod-1"], received_at=99_000.0 + 10 * k)
    _written(records, honest, "5X", scored_by=["pod-2"], received_at=99_015.0)
    auditor = _restart(records, states, draw_job(), q=0.25)

    async def audit(batch, *, local=False):
        return [{**_DRAW_REFUSAL, "penalty": False} if r["hotkey"] == "5X" else dict(_PASSING) for r in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    asyncio.run(auditor.restore_scored())
    assert [records.voided[sid].get("charged_for") for sid in (oldest, middle, latest)] == [
        None, None, "frequent_refusals"]
    assert states.states["5X"].failure_ids == [latest] and states.states["5X"].forfeit_before == 99_020.0
    assert records.voided[honest]["reason"] == "forfeited"


def test_a_reaudit_counts_no_pass_it_confirms_twice(monkeypatch):
    """Without a restart, every pass the re-audit confirms was counted by its judging pass, and stays counted once:
    5X's three passes stay three audits. Counted again (`_draw_counted` not checked where the re-audit counts them),
    they would dilute 5X's refusals: its three drifts judged next, 3 in 6 audits, are charged for frequent refusals,
    and would not be as 3 in 9."""
    world = _world([("5Other%d" % (i % 20), "drift" if i % 100 == 0 else "pass:pod-2", "pass") for i in range(400)]
                   + [("5X", "pass:pod-1", "pass")] * 3 + [("5X", "drift", "drift")] * 3)
    auditor = world.auditor()
    asyncio.run(auditor.audit_many(world.ids[:403]))
    counts = (dict(auditor._draw_counts), list(auditor._draw_totals))
    assert auditor._draw_counts["5X"] == [3, 0] and auditor._draw_totals == [403, 4]
    world.quarantined.add("pod-1")
    assert asyncio.run(auditor.reaudit_executor("pod-1")) == []
    assert (dict(auditor._draw_counts), list(auditor._draw_totals)) == counts
    asyncio.run(auditor.audit_many(world.ids[403:]))
    assert auditor._draw_counts["5X"] == [6, 3]
    assert world.records.verdicts[world.ids[-1]]["charged_for"] == "frequent_refusals"


_TAGGED = {"drift": {**_DRAW_REFUSAL, "penalty": False}, "penalised": dict(_DRAW_REFUSAL),
           "other": {"passed": False, "reason": "bad_tokens"}}


def _outcome(tag):
    if tag.startswith("pass"):
        executor = tag.partition(":")[2]
        return {**_PASSING, **({"scored_by": [executor]} if executor not in ("", "local") else {})}
    return dict(_TAGGED[tag])


def _world(rows, *, q=1.0):
    """A forced-draw job judged by a judge process (an EvalAuditor), one record per row: (hotkey, the outcome its
    judging pass takes, the outcome this GPU's re-audit of its pass finds), received 1 s apart. An outcome:
    "pass:<executor>" (scored by that executor; "pass:local", by none), "drift", "penalised", or "other" (a failure
    of another kind, charged and never counted). ``world.auditor()``: a judge process over the job's records and
    miners (``restores=True``: one that restarted); ``world.quarantined``: the executors quarantined."""
    from reliquary.corpus.audit_policy import AuditParams
    from reliquary.validator import eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    world = SimpleNamespace(job=job, ids=[f"{k + 1:064x}" for k in range(len(rows))], states=_States(),
                            quarantined=set())
    world.records = _VoidingRecords({
        sid: {"hotkey": hotkey, "token_count": 10, "prompt_index": job.prompt_start, "cursor": 0,
              "received_at": 99_000.0 + k, "judged": judged, "reaudit": reaudit}
        for k, (sid, (hotkey, judged, reaudit)) in enumerate(zip(world.ids, rows))})

    async def audit(batch, *, local=False):
        # A judging pass's audits take the row's judging outcome (its second audit of a failure too); this GPU's audit
        # of a pass already written, the re-audit's, takes its re-audit outcome.
        return [_outcome(r["reaudit"] if local and r["judged"].startswith("pass") else r["judged"]) for r in batch]

    def auditor(*, restores=False):
        remote = SimpleNamespace(connected=lambda: True, subscribe=lambda listener: None,
                                 quarantined=lambda: set(world.quarantined))
        made = eval_control.eval_auditor(job_id=job.job_id, records=world.records, tokenizer=None, proof=PROOF,
                                         vocab_size=10, job=job, job_store=object(), draw_job=job, remote=remote,
                                         miner_states=world.states, clock=_Clock(100_000.0), params=AuditParams(q=q),
                                         restores_scored=restores)
        made._audit_outcomes = audit
        return made

    world.auditor = auditor
    return world


def _judge_all(auditor, ids, size=100):
    for start in range(0, len(ids), size):
        asyncio.run(auditor.audit_many(ids[start:start + size]))


def _ending(world, auditor):
    """What a differential compares: the draw counts, each hotkey's charged failures, each void."""
    return (auditor._draw_counts, auditor._draw_totals,
            {hotkey: state.failure_ids for hotkey, state in world.states.states.items()},
            {sid: (doc.get("executor_id"), doc.get("reason"), doc.get("penalty"), doc.get("charged_for"))
             for sid, doc in world.records.voided.items()})


def _mixed_rows():
    """400 records of 20 other hotkeys (pod-1, pod-2 or audited here; 1 % refused as drifts, and one refusal with a
    penalty, which forfeits at q < 1), then the hotkeys pod-1's re-audit fails: 5X's three passes are drifts (the
    third crosses the line), 5P's pass a penalised refusal, 5O's a failure of another kind, 5M's one drift among
    two passes of pod-2 that stand."""
    executors = ("pass:pod-1", "pass:pod-2", "pass:local", "pass:pod-1")
    rows = [(f"5Other{i % 20}", "drift" if i % 100 == 50 else executors[i % 4], "pass") for i in range(400)]
    rows[123] = ("5Other3", "penalised", "pass")
    return rows + [("5X", "pass:pod-1", "drift")] * 3 + [
        ("5P", "pass:pod-1", "penalised"), ("5O", "pass:pod-1", "other"),
        ("5M", "pass:pod-2", "pass"), ("5M", "pass:pod-1", "drift"), ("5M", "pass:pod-2", "pass")]


@pytest.mark.parametrize("q", [1.0, 0.25])
@pytest.mark.parametrize("when", ["before", "after"])
def test_a_restarted_judge_ends_with_the_counts_of_one_that_never_restarted(q, when):
    """The same records judged by a process that never restarts, whose pod-1 is then quarantined and re-audited, and
    by one that restarts ``when`` that re-audit (its next start re-audits pod-1's passes, or finds them voided): both
    end with the same draw counts, charges and voids. At q < 1, 5X's third drift, charged, and 5P's penalised refusal
    forfeit their own passes (and 5X's two drifts before it) before the re-audit writes a void: those voids are the
    re-audit's own documents, and the restart counts their refusals. 5Other3's penalised refusal, from a judging pass,
    forfeits its earlier passes: voided, they are still counted. A penalised re-audit refusal counts as a refusal of
    its pass, like a drift, whether its void is the re-audit's or its forfeit's."""
    rows = _mixed_rows()
    never = _world(rows, q=q)
    auditor = never.auditor()
    _judge_all(auditor, never.ids)
    never.quarantined.add("pod-1")
    asyncio.run(auditor.reaudit_executor("pod-1"))
    expected = _ending(never, auditor)
    counts, totals, charged, voids = expected
    p, x, m = (never.ids[400 + k] for k in (3, 0, 5))
    assert counts["5X"] == [3, 3] and counts["5P"] == [1, 1] and counts["5O"] == [1, 0] and counts["5M"] == [3, 1]
    assert totals == [408, 10] and charged["5X"] == [never.ids[402]] and charged["5P"] == [p]
    assert voids[p][:2] == ("pod-1", "draw_mismatch") and voids[x][:3] == ("pod-1", "draw_mismatch", False)
    assert m not in voids and never.ids[406] in voids
    if q < 1:
        # 5Other3's passes written before its penalised refusal are voided by its forfeit, and the one judged with it
        # is forfeited as it is written: all were counted as passing audits.
        assert [voids[never.ids[i]] for i in (3, 23, 43, 63, 83)] == [(None, "forfeited", None, None)] * 5
        assert never.records.verdicts[never.ids[103]]["reason"] == "forfeited"

    world = _world(rows, q=q)
    first = world.auditor()
    _judge_all(first, world.ids)
    world.quarantined.add("pod-1")
    if when == "after":
        asyncio.run(first.reaudit_executor("pod-1"))
    restarted = world.auditor(restores=True)
    asyncio.run(restarted.restore_scored())
    assert _ending(world, restarted) == expected


def test_a_restart_counts_an_id_once_whoever_counts_it_first(monkeypatch):
    """A restarted judge process's passes run while its restore reads the verdicts. While the first slice is read, a
    pass judges a new record, Y; while a pre-restart pass, Z, is being read, something else counts it (standing for
    any count of an id whose read is in flight). A read then fails, and the restore is run again, as `_restore_behind`
    does: it now lists Y, and every id the first attempt counted. Each is counted once, where it is added: the process
    ends with the counts, charges and voids of one that never restarted and judged Y at the same point."""
    rows = _mixed_rows() + [("5Y", "pass:pod-2", "pass")]
    never = _world(rows)
    auditor = never.auditor()
    _judge_all(auditor, never.ids[:-1])
    asyncio.run(auditor.audit_many(never.ids[-1:]))
    never.quarantined.add("pod-1")
    asyncio.run(auditor.reaudit_executor("pod-1"))
    expected = _ending(never, auditor)

    world = _world(rows)
    first = world.auditor()
    _judge_all(first, world.ids[:-1])
    world.quarantined.add("pod-1")
    restarted = world.auditor(restores=True)
    y, z = world.ids[-1], world.ids[7]
    read, reads, failed = world.records.read_verdict, [], []

    async def read_verdict(job_id, sid):
        reads.append(sid)
        if len(reads) == 1:
            await restarted.audit_many([y])  # a judging pass, meanwhile
        if sid == z and len(reads) < 400:
            restarted._count_refusal(z, "5Other7", {"passed": True})  # counted while its read is in flight
        if len(reads) == 40 and not failed:
            failed.append(sid)
            raise OSError("store GET failed after its retries")
        return await read(job_id, sid)

    world.records.read_verdict = read_verdict
    with pytest.raises(OSError):
        asyncio.run(restarted.restore_scored())
    assert y in world.records.verdicts and restarted._draw_counts["5Y"] == [1, 0]
    asyncio.run(restarted.restore_scored())
    assert y in reads[40:] and z in reads[:40] and z in reads[40:]
    assert _ending(world, restarted) == expected


def test_a_restart_lists_no_pass_for_a_reaudit_before_every_verdict_is_counted(monkeypatch):
    """pod-1's three passes of 5Y are read first; its 100 passes of pod-2, healthy, after the job's other hotkeys'.
    A judging batch ends while the restore still reads (`_reaudit_quarantined`, which starts `_reaudit_late` for a
    quarantined executor with passes listed). Listed as they were read, pod-1's passes were re-audited then, on counts
    that held none of 5Y's pod-2 passes: 3 refusals in 3 audits, the third charged. Listed once every verdict is
    counted, they are re-audited on 5Y's whole counts, 3 in 103: nothing is charged, as in a process that never
    restarted."""
    from tests.unit.test_eval_draw import _job as draw_job

    records, states = _VoidingRecords({}), _States()
    drifts = [f"{i + 1:064x}" for i in range(3)]
    for sid in drifts:
        _written(records, sid, "5Y", scored_by=["pod-1"])
    _population(records, executor="pod-2")
    for i in range(100):
        _written(records, f"{0xC0000 + i:064x}", "5Y", scored_by=["pod-2"])
    auditor = _restart(records, states, draw_job())
    reaudited = []

    async def audit(batch, *, local=False):
        reaudited.extend(r["hotkey"] for r in batch)
        return [{**_DRAW_REFUSAL, "penalty": False} if r["hotkey"] == "5Y" else dict(_PASSING) for r in batch]

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    read, reads = records.read_verdict, []

    async def read_verdict(job_id, sid):
        reads.append(sid)
        if len(reads) == 40:
            auditor._reaudit_quarantined()  # the end of a judging batch (`_audit_records`)
            for _ in range(20):
                await asyncio.sleep(0)
            assert reaudited == [] and auditor._late_reaudits == {}
        return await read(job_id, sid)

    records.read_verdict = read_verdict
    asyncio.run(auditor.restore_scored())
    assert reaudited == ["5Y"] * 6  # each drift audited twice (§7.2), once every verdict was counted
    assert auditor._draw_counts["5Y"] == [103, 3]
    assert [records.voided[sid]["penalty"] for sid in drifts] == [False] * 3 and "5Y" not in states.states


def test_a_restart_charges_no_frequent_refusals_until_its_counts_are_whole(monkeypatch):
    """While the restore reads (`_draw_restoring`), judging passes count every audit but charge nothing for frequent
    refusals: 5Z, with 100 passes written before the restart, and 5C, a fresh hotkey, are refused three times each.
    Judged on the counts read so far, both were charged: 5Z's 100 passes were mostly unread. Both stay uncharged,
    counted; once every verdict is counted, 5C's fourth refusal in four audits, far above the job's 2 %, is charged,
    and 5Z, refused 3 times in 103 audits, is not."""
    from tests.unit.test_eval_draw import _job as draw_job

    records = _VoidingRecords({})
    _population(records, executor="pod-2")
    for i in range(100):
        _written(records, f"{0xD0000 + i:064x}", "5Z", scored_by=["pod-2"])
    auditor = _restart(records, _States(), draw_job())
    drift = {**_DRAW_REFUSAL, "penalty": False}
    read, reads, during = records.read_verdict, [], []

    async def read_verdict(job_id, sid):
        reads.append(sid)
        if len(reads) == 200:
            during.extend(auditor._count_refusal(f"{hotkey}{k}", hotkey, dict(drift))
                          for hotkey in ("5Z", "5C") for k in range(3))
        return await read(job_id, sid)

    records.read_verdict = read_verdict
    restoring = getattr(auditor, "_draw_restoring", None)
    asyncio.run(auditor.restore_scored())
    assert [outcome["penalty"] for outcome in during] == [False] * 6
    assert auditor._draw_counts["5Z"] == [103, 3] and auditor._draw_counts["5C"] == [3, 3]
    assert auditor._count_refusal("5C3", "5C", dict(drift))["charged_for"] == "frequent_refusals"
    assert auditor._count_refusal("5Z3", "5Z", {"passed": True}) == {"passed": True}
    assert "charged_for" not in auditor._count_refusal("5Z4", "5Z", dict(drift))
    assert restoring is True and auditor._draw_restoring is False


def test_without_a_connected_executor_the_auditor_runs_locally():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "a" * 64
    records = _Records({sid: _record(model)})
    remote = SimpleNamespace(connected=lambda: False, score=None)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    asyncio.run(auditor.audit_many([sid]))
    assert records.verdicts[sid]["passed"] is True


# --------------------------------------------------------------------------
# HTTP: the control routes and the executor client against them
# --------------------------------------------------------------------------


def _app(h):
    app = FastAPI()
    app.include_router(build_audit_executor_router(h.dispatcher, h.directory))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://control")


@pytest.mark.parametrize("token,detail", [("wrong", "unknown_token"), (OTHER, "revoked"),
                                          ("t3" * 10, "expired"), (None, "missing_token")])
def test_a_wrong_revoked_or_expired_token_is_refused(token, detail):
    model = _tiny(0)

    async def go():
        h = _Harness(model, docs=[_doc(), _doc("pod-2", OTHER, status="revoked"),
                                  _doc("pod-3", "t3" * 10, expires_at=999.0)])
        await h.directory.refresh()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with _app(h) as client:
            response = await client.post("/corpus/internal/audit/claim", headers=headers, json={
                "executor_id": "pod-1", "model_id": MODEL, "model_revision": REVISION})
        assert response.status_code == 401 and response.json()["detail"] == detail

    asyncio.run(go())


def test_the_executor_client_heartbeats_claims_scores_and_posts():
    from reliquary.validator.corpus_audit_executor import AuditExecutor

    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        async with _app(h) as client:
            executor = AuditExecutor(http=client, executor_id="pod-1", token=GOOD,
                                     load_model=lambda m, r: model, batch_tokens=1 << 20)
            await executor.start()
            assert (executor.model_id, executor.model_revision) == (MODEL, REVISION)
            assert await executor.step() is False  # no work: 204
            task = asyncio.ensure_future(h.dispatcher.score(_items(model, n=2)))
            await asyncio.sleep(0)
            assert await executor.step() is True
            assert [s for s, _, _ in await task] == ["ok", "ok"]
            # Quarantined: the next call is refused outright.
            h.directory.revoke_locally("pod-1")
            with pytest.raises(httpx.HTTPStatusError):
                await executor.step()

    asyncio.run(go())


def test_an_executor_for_another_model_than_registered_refuses_to_start():
    from reliquary.validator.corpus_audit_executor import AuditExecutor

    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        async with _app(h) as client:
            executor = AuditExecutor(http=client, executor_id="pod-1", token=GOOD,
                                     model_id=MODEL, model_revision="elsewhere",
                                     load_model=lambda m, r: pytest.fail("loaded"))
            with pytest.raises(RuntimeError, match="registered"):
                await executor.start()

    asyncio.run(go())


# --------------------------------------------------------------------------
# Wiring: the validator mounts the routes, the CLI starts the executor
# --------------------------------------------------------------------------


def test_the_validator_mounts_the_executor_routes_only_when_asked(
    seeded_job, fake_r2, wired_records, fixed_drand_chain, monkeypatch,  # noqa: F811
):
    import huggingface_hub
    import uvicorn

    import reliquary.corpus.encoding as encoding
    import reliquary.protocol.profiles as profiles
    import reliquary.shared.modeling as modeling
    from reliquary.infrastructure import corpus_executor_store
    from reliquary.validator import corpus_auditor, corpus_period_settlement
    from reliquary.validator.corpus_validator import run_corpus_validator
    from tests.unit.test_corpus_multi_job_validator import _Model, _entry
    from tests.unit.test_corpus_service import CHECKPOINT, _Tokenizer as _ServiceTokenizer
    from tests.unit.test_corpus_validator import _profile

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None: "/x")
    monkeypatch.setattr(encoding, "checkpoint_fingerprint", lambda d: CHECKPOINT)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: _ServiceTokenizer())
    monkeypatch.setattr(modeling, "load_text_only_model", lambda path, **kw: _Model())
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", _profile())
    monkeypatch.setattr(corpus_executor_store, "get_s3_client", lambda **kw: _R2())
    built = []

    async def idle(self):
        built.append(self)
        await asyncio.sleep(3600)

    async def settle_once(self):
        return None

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", settle_once)
    seen = {}

    class _Server:
        def __init__(self, config):
            self.app = config.app

        async def serve(self):
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                seen["claim"] = (await client.post(
                    "/corpus/internal/audit/claim", headers={"Authorization": "Bearer nope"},
                    json={"executor_id": "pod-1", "model_id": "org/Frozen",
                          "model_revision": "abc123"})).status_code
            raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(uvicorn, "Server", _Server)
    for remote_audit, expected in ((False, 404), (True, 401)):
        built.clear()
        with pytest.raises(_Stop):
            asyncio.run(run_corpus_validator(
                entry=_entry("corpus-math", "swe-v1"), cap=0.1, wallet=None, netuid=0,
                signer_client=None, http_host="127.0.0.1", http_port=0, set_weights=False,
                registration_gate=False, remote_audit=remote_audit))
        assert seen["claim"] == expected
        assert (built[0]._remote is not None) is remote_audit


class _R2:
    """An empty executor registry."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get_paginator(self, name):
        class _Paginator:
            def paginate(self, Bucket, Prefix=""):
                async def pages():
                    yield {"Contents": []}

                return pages()

        return _Paginator()


def test_the_audit_executor_command_needs_its_token(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.validator import corpus_audit_executor

    calls = []
    monkeypatch.setattr(corpus_audit_executor, "run_audit_executor", lambda **kw: calls.append(kw))
    monkeypatch.delenv("RELIQUARY_EXECUTOR_TOKEN", raising=False)
    argv = ["corpus", "audit-executor", "--control", "https://control", "--executor-id", "pod-1"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1 and "RELIQUARY_EXECUTOR_TOKEN" in result.output
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t" * 43)
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert calls == [{"control_url": "https://control", "executor_id": "pod-1",
                      "model_id": None, "model_revision": None}]


def test_remote_audit_is_an_operator_opt_in(monkeypatch):
    from reliquary.cli.main import _corpus_remote_audit_options

    monkeypatch.delenv("RELIQUARY_CORPUS_REMOTE_AUDIT", raising=False)
    assert _corpus_remote_audit_options() == {}
    monkeypatch.setenv("RELIQUARY_CORPUS_REMOTE_AUDIT", "1")
    assert _corpus_remote_audit_options() == {"remote_audit": True, "recheck_fraction": 0.05}
    monkeypatch.setenv("RELIQUARY_CORPUS_RECHECK_FRACTION", "0")
    with pytest.raises(ValueError):
        _corpus_remote_audit_options()


from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: E402,F401
from tests.unit.test_corpus_validator import fixed_drand_chain, wired_records  # noqa: E402,F401


# --------------------------------------------------------------------------
# Audit ahead: drawn records are scored before their pass reaches them
# --------------------------------------------------------------------------


def test_a_record_audited_ahead_is_judged_from_that_outcome_without_a_second_score():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model, other = _tiny(0), _tiny(1)
    sid_good, sid_bad = "a" * 64, "b" * 64
    records = _Records({sid_good: _record(model), sid_bad: _record(other)})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)

    async def go():
        await auditor._read_all([sid_good, sid_bad])
        # q = 1: every pending record is sure to be audited.
        assert sorted(auditor._ahead_candidates(8)) == [sid_good, sid_bad]
        await auditor._audit_ahead_batch([sid_good, sid_bad])
        assert set(auditor._ahead) == {sid_good, sid_bad} and remote.calls == 1
        assert auditor._ahead_candidates(8) == []
        await auditor.audit_many([sid_good, sid_bad])

    asyncio.run(go())
    assert remote.calls == 1  # the pass took the outcomes computed ahead
    assert records.verdicts[sid_good]["passed"] is True
    assert records.verdicts[sid_good]["scored_by"] == ["pod-1"]
    # A failure scored ahead is still confirmed by a second audit on this GPU before it counts.
    assert records.verdicts[sid_bad]["passed"] is False
    assert auditor._ahead == {}


def test_outcomes_audited_ahead_by_a_quarantined_executor_are_dropped():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "a" * 64
    records = _Records({sid: _record(model)})
    remote = _AlwaysConnected(model, lie_pass=True, executor="pod-9")
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)

    async def go():
        await auditor._audit_ahead_batch([sid])
        assert sid in auditor._ahead
        await auditor.reaudit_executor("pod-9")
        assert sid not in auditor._ahead

    asyncio.run(go())


def test_a_validator_error_ahead_is_left_for_the_pass():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "a" * 64
    records = _Records({sid: _record(model)})

    class _Broken(_AlwaysConnected):
        async def score(self, items):
            self.calls += 1
            raise RuntimeError("executor gone")

    remote = _Broken(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    asyncio.run(auditor._audit_ahead_batch([sid]))
    assert auditor._ahead == {} and auditor._ahead_flight == {}


def test_on_a_sampled_job_only_drawn_records_with_their_round_out_are_audited_ahead():
    from reliquary.corpus.audit_policy import AuditParams, drawn
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sids = [f"{i:x}" * 64 for i in range(1, 9)]
    records = _Records({sid: _record(model) for sid in sids})
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=_AlwaysConnected(model), params=AuditParams(q=0.5),
                            round_at=lambda t: 100)
    randomness = "c" * 64

    async def go():
        await auditor._read_all(sids)
        assert auditor._ahead_candidates(16) == []  # round 101 not out yet
        auditor._randomness[101] = randomness
        return auditor._ahead_candidates(16)

    chosen = asyncio.run(go())
    expected = {sid for sid in sids if drawn(randomness, sid, 0.5)}
    assert set(chosen) == expected and 0 < len(expected) < len(sids)


def test_the_audit_backlog_leaves_out_records_with_no_gpu_work_left():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    auditor = CorpusAuditor(job_id="j", records=_Records({}), model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=_AlwaysConnected(model))
    for sid in (a, b, c):
        auditor._schedule(sid, 0.0)
    assert auditor.audit_backlog() == 3
    auditor._undrawn.add(a)
    auditor._ahead[b] = {"passed": True}
    assert auditor.audit_backlog() == 1


def test_on_a_sampled_job_a_probation_hotkeys_records_are_audited_ahead():
    """Records sure to be audited because their hotkey is in probation (or too slow to be sampled) are sent ahead
    too, not only drawn ones: the card keeps working between passes (2026-11-18)."""
    from reliquary.corpus.audit_policy import AuditParams, MinerState
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sids = [f"{i:x}" * 64 for i in range(1, 5)]
    records = _Records({sid: _record(model) for sid in sids})
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=_AlwaysConnected(model), params=AuditParams(q=0.5),
                            round_at=lambda t: 100)

    async def go():
        await auditor._read_all(sids)
        assert auditor._ahead_candidates(16) == []          # standing unknown, round not out: nothing sure
        hotkey = auditor._meta[sids[0]][0]
        auditor._state_cache[hotkey] = MinerState()          # no audited pass yet: probation
        return auditor._ahead_candidates(16)

    assert set(asyncio.run(go())) == set(sids)


def test_a_record_its_pass_is_auditing_is_not_sent_ahead_again():
    """From the pass taking a record's outcome (ahead, or scored now) to its verdict's write, the record is in no
    other set; the ahead loop sent it again, the oldest drawn record, and threw the score away (2026-11-19: half
    the card's scores, the auction sized on them opening twice the leases the verdicts could take)."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sids = [c * 64 for c in "abcd"]
    during_write = []

    class _Watched(_Records):
        async def write_verdict(self, job_id, sid, verdict):
            during_write.append(auditor._ahead_candidates(8))
            return await super().write_verdict(job_id, sid, verdict)

    records = _Watched({sid: _record(model) for sid in sids})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)

    async def go():
        await auditor._read_all(sids)
        await auditor._audit_ahead_batch(sids[:2])  # a and b ahead; c and d scored by the pass itself
        await auditor.audit_many(sids)

    asyncio.run(go())
    assert during_write and all(candidates == [] for candidates in during_write)
    assert auditor._in_pass == set() and set(records.verdicts) == set(sids)
    assert remote.calls == 2 and auditor.ahead_wasted == 0


def test_a_pass_takes_the_outcome_of_an_ahead_batch_in_flight_when_it_took_the_record():
    """2026-11-19: the batch saw the record in its pass and dropped the outcome the pass was waiting for; the pass
    then scored it again (ahead used 1-5 a pass, 395 outcomes thrown away in 20 min)."""
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "f" * 64
    records = _Records({sid: _record(model)})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    release = asyncio.Event()
    score = remote.score

    async def held_score(items):
        await release.wait()
        return await score(items)

    remote.score = held_score

    async def go():
        await auditor._read_all([sid])
        flight = asyncio.ensure_future(auditor._audit_ahead_batch([sid]))
        auditor._ahead_flight[sid] = flight
        await asyncio.sleep(0)
        judging = asyncio.ensure_future(auditor.audit_many([sid]))
        await asyncio.sleep(0.05)
        release.set()
        await judging

    asyncio.run(go())
    assert remote.calls == 1 and auditor.ahead_wasted == 0
    assert records.verdicts[sid]["passed"] is True


def test_an_outcome_scored_ahead_for_a_record_judged_meanwhile_counts_as_wasted():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "e" * 64
    records = _Records({sid: _record(model)})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    score = remote.score

    async def judged_meanwhile(items):
        auditor._judged.add(sid)  # its verdict landed while the batch flew
        return await score(items)

    remote.score = judged_meanwhile

    async def go():
        await auditor._read_all([sid])
        await auditor._audit_ahead_batch([sid])

    asyncio.run(go())
    assert auditor._ahead == {} and auditor.ahead_wasted == 1
