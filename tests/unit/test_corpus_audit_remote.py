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


def test_three_expired_leases_in_a_row_quarantine_the_executor():
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
        assert [q[0] for q in h.quarantined] == ["pod-1"]
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
        self.voided[sid] = document
        return True

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return dict(self.settlement), None


class _States:
    def __init__(self):
        from reliquary.corpus.audit_policy import MinerState

        self.states = {}
        self._blank = MinerState

    async def get(self, hotkey):
        return self.states.get(hotkey, self._blank())

    async def update(self, hotkey, change):
        self.states[hotkey] = change(await self.get(hotkey))
        return self.states[hotkey]

    async def update_many(self, changes):
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
    from reliquary.validator.corpus_settlement import CorpusSettler

    class _Archives:
        written = {}

        async def other_max(self, task_id):
            return None

        async def write(self, task_id, window, data):
            self.written[window] = data

    records.verdicts = {cheat: records.verdicts[cheat]}
    records.state, records.etag = {}, None

    async def read_settlement(job_id):
        return dict(records.state), records.etag

    async def write_settlement(job_id, state, etag):
        records.state, records.etag = dict(state), "e"
        return "e"

    records.read_settlement, records.write_settlement = read_settlement, write_settlement
    settler = CorpusSettler(task_id="t", job_id="j", cap=0.1, records=records,
                            archives=_Archives(), clock=lambda: 0.0)
    asyncio.run(settler.settle_once())
    assert _Archives.written == {} and records.state["settled"] == [cheat]


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
    from reliquary.validator import corpus_auditor, corpus_settlement
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
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", settle_once)
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
