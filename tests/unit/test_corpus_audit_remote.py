"""R3: remote audit executors. Tokens are checked against the registry, results
stay provisional until a local recheck vouches for them, a lying executor is
quarantined and its batches re-queued, and with no executor the control audits
locally as before."""

from __future__ import annotations

import asyncio
import random
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


class _Harness:
    def __init__(self, model, *, fraction=0.0, docs=None, seed=0):
        self.clock = _Clock()
        self.model = model
        self.directory = _directory(docs or [_doc(), _doc("pod-2", OTHER)], self.clock)
        self.local_calls = 0
        self.quarantined = []

        async def local_scores(items):
            self.local_calls += 1
            scores, _, _ = score_sequences(model, _wire(items), chunk_tokens=PROOF.chunk_tokens,
                                           topk=PROOF.topk, batch_tokens=1 << 20)
            return scores

        async def quarantine(executor_id, reason):
            self.quarantined.append((executor_id, reason))

        self.dispatcher = RemoteAuditDispatcher(
            directory=self.directory, proof=PROOF, local_scores=local_scores,
            quarantine=quarantine, clock=self.clock, rng=random.Random(seed),
            recheck_fraction=fraction)


async def _drain(dispatcher, rounds=20):
    for _ in range(rounds):
        await asyncio.sleep(0)
        await dispatcher.sweep()
        await asyncio.sleep(0)


def test_an_honest_executor_scores_and_a_forced_recheck_vouches_for_it():
    model = _tiny(0)

    async def go():
        h = _Harness(model)
        await h.directory.refresh()
        items = _items(model)
        h.dispatcher.heartbeat("pod-1")
        assert h.dispatcher.connected()
        task = asyncio.ensure_future(h.dispatcher.score(items))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim("pod-1")
        assert h.dispatcher.claim("pod-1") is None
        assert h.dispatcher.result("pod-1", lease["lease_id"], _honest_scores(model, lease)) == "accepted"
        assert not task.done()  # provisional until a recheck vouches
        await _drain(h.dispatcher)
        scores = await task
        assert [s for s, _ in scores] == ["ok", "ok", "ok"]
        assert h.local_calls == 1 and h.quarantined == []

    asyncio.run(go())


def test_a_lying_executor_is_quarantined_and_its_batches_requeued():
    model, other = _tiny(0), _tiny(1)

    async def go():
        h = _Harness(model, fraction=0.0)
        await h.directory.refresh()
        # Proofs from another model: every item truly fails.
        bad = _items(other, n=2)
        h.dispatcher.heartbeat("pod-1")
        h.dispatcher.heartbeat("pod-2")
        first = asyncio.ensure_future(h.dispatcher.score(bad[:1]))
        second = asyncio.ensure_future(h.dispatcher.score(bad[1:]))
        await asyncio.sleep(0)
        for _ in range(2):
            lease = h.dispatcher.claim("pod-1")
            h.dispatcher.result("pod-1", lease["lease_id"], _lying_scores(lease))
        # A recheck of one of them diverges: both batches are back in play.
        h.dispatcher._fraction = 1.0
        await _drain(h.dispatcher, rounds=3)
        assert [q[0] for q in h.quarantined] == ["pod-1"]
        assert h.dispatcher.quarantined == {"pod-1"}
        assert h.directory.authenticate(GOOD)[1] == "revoked"
        # The re-queued batch goes to the honest executor.
        h.dispatcher._fraction = 0.0
        lease = h.dispatcher.claim("pod-2")
        if lease is not None:
            h.dispatcher.result("pod-2", lease["lease_id"], _honest_scores(model, lease))
        await _drain(h.dispatcher)
        results = [await first, await second]
        from reliquary.validator.corpus_audit import outcome_from_scores

        for scores in results:
            assert all(not outcome_from_scores(s, c, PROOF).passed for s, c in scores)

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
        await _drain(h.dispatcher)
        assert [s for s, _ in await task] == ["ok"]

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
        assert [s for s, _ in await task] == ["ok", "ok"]
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


# --------------------------------------------------------------------------
# The auditor over a dispatcher: the decision stays on the control
# --------------------------------------------------------------------------


class _AlwaysConnected:
    def __init__(self, model, lie=False):
        self.model, self.lie, self.calls = model, lie, 0

    def connected(self):
        return True

    async def score(self, items):
        self.calls += 1
        if self.lie:
            # Claims every proof fails, to get honest miners failed.
            from reliquary.protocol.toploc import ChunkResult

            return [("ok", tuple(ChunkResult(10_000, 1e9, 1e9) for _ in i["proofs"]))
                    for i in items]
        scores, _, _ = score_sequences(self.model, _wire(items), chunk_tokens=PROOF.chunk_tokens,
                                       topk=PROOF.topk, batch_tokens=1 << 20)
        return scores


def test_the_auditor_judges_remote_scores_itself():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model, other = _tiny(0), _tiny(1)
    sid_good, sid_bad = "a" * 64, "b" * 64
    records = _Records({sid_good: _record(model), sid_bad: _record(other)})
    remote = _AlwaysConnected(model)
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=remote)
    asyncio.run(auditor.audit_many([sid_good, sid_bad]))
    assert records.verdicts[sid_good]["passed"] is True
    assert records.verdicts[sid_bad]["passed"] is False
    assert remote.calls == 1  # the failure was confirmed on the control's own GPU


def test_an_executor_alone_can_never_fail_an_honest_miner():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    model = _tiny(0)
    sid = "a" * 64
    records = _Records({sid: _record(model)})
    auditor = CorpusAuditor(job_id="j", records=records, model=model, tokenizer=_Tokenizer(),
                            proof=PROOF, remote=_AlwaysConnected(model, lie=True))
    asyncio.run(auditor.audit_many([sid]))
    assert records.verdicts[sid]["passed"] is True


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
            await _drain(h.dispatcher)
            assert [s for s, _ in await task] == ["ok", "ok"]
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
