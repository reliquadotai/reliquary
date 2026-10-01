"""The GPU-less eval control (design v2, item 5): executor pairs on distinct
providers, a third on disagreement, a lone executor never decides, several
models in one process, prefix routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_audit_protocol import AuditResult
from reliquary.validator.corpus_audit_remote import LeaseRefused
from reliquary.validator.eval_control import (
    PairedAuditDispatcher,
    eval_auditor,
    eval_job_refusal,
)

MODEL_A = ("org/a", "a" * 40)
MODEL_B = ("org/b", "b" * 40)


def _executor(eid, model=MODEL_A, provider=None, host=None):
    return {"executor_id": eid, "model_id": model[0], "model_revision": model[1],
            "provider_id": provider or f"prov-{eid}", "host": host or f"host-{eid}",
            "status": "active", "expires_at": 1e12, "token_sha256": eid * 0}


ITEMS = [{"tokens": [1, 2, 3, 4], "prompt_len": 2, "proofs": ["p"]},
         {"tokens": [5, 6, 7], "prompt_len": 1, "proofs": ["q"]}]


def _honest(exp=1):
    return AuditResult.model_validate({"scores": [
        {"status": "ok", "chunks": [[exp, 1.0, 1.0]]}, {"status": "ok", "chunks": [[exp, 1.0, 1.0]]}]})


def _lying():
    # Under-reports nothing but flips the decision: far over the thresholds.
    return AuditResult.model_validate({"scores": [
        {"status": "ok", "chunks": [[500, 1.0, 1.0]]}, {"status": "ok", "chunks": [[1, 1.0, 1.0]]}]})


class _Harness:
    def __init__(self):
        self.quarantined = []
        self.directory = SimpleNamespace(revoke_locally=lambda e: None)

        async def write(executor_id, reason):
            self.quarantined.append(executor_id)

        self.dispatcher = PairedAuditDispatcher(directory=self.directory, quarantine=write)


def _run(coroutine_factory):
    return asyncio.run(coroutine_factory())


def test_two_executors_on_distinct_providers_decide_a_batch():
    async def go():
        h = _Harness()
        view = h.dispatcher.view(*MODEL_A, PROOF)
        scoring = asyncio.ensure_future(view.score(ITEMS))
        await asyncio.sleep(0)
        a = h.dispatcher.claim(_executor("a"))
        assert a is not None and a["chunk_tokens"] == PROOF.chunk_tokens
        # Same provider, or same host: never the second scorer.
        assert h.dispatcher.claim(_executor("a2", provider="prov-a")) is None
        assert h.dispatcher.claim(_executor("a3", host="host-a")) is None
        assert h.dispatcher.claim(_executor("a")) is None  # already on it
        await h.dispatcher.result("a", a["lease_id"], _honest())
        assert not scoring.done()  # one executor alone never decides
        b = h.dispatcher.claim(_executor("b"))
        assert b is not None and b["items"] == a["items"]
        await h.dispatcher.result("b", b["lease_id"], _honest(exp=2))
        scored = await scoring
        assert [s[2] for s in scored] == [("a", "b"), ("a", "b")]
        assert scored[0][0] == "ok" and h.quarantined == []
    _run(go)


def test_a_lone_executor_never_decides():
    async def go():
        h = _Harness()
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        lease = h.dispatcher.claim(_executor("a"))
        await h.dispatcher.result("a", lease["lease_id"], _honest())
        await h.dispatcher.sweep()
        await asyncio.sleep(0)
        assert not scoring.done() and h.dispatcher.pending(MODEL_A) == 1
        scoring.cancel()
    _run(go)


def test_disagreement_goes_to_a_third_and_the_minority_is_quarantined():
    async def go():
        h = _Harness()
        reaudited = []

        async def listener(executor_id):
            reaudited.append(executor_id)

        h.dispatcher.subscribe(listener)
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        a = h.dispatcher.claim(_executor("a"))
        b = h.dispatcher.claim(_executor("b"))
        await h.dispatcher.result("a", a["lease_id"], _honest())
        await h.dispatcher.result("b", b["lease_id"], _lying())
        assert not scoring.done()
        assert h.dispatcher.stats["disagreements"] == 1
        # Neither a nor b may take the third seat; c may.
        assert h.dispatcher.claim(_executor("a")) is None
        c = h.dispatcher.claim(_executor("c"))
        await h.dispatcher.result("c", c["lease_id"], _honest())
        scored = await scoring
        assert scored[0][2] == ("a", "c")
        await asyncio.sleep(0)
        assert h.quarantined == ["b"] and reaudited == ["b"]
        assert "b" in h.dispatcher.quarantined
    _run(go)


def test_pools_are_per_model():
    async def go():
        h = _Harness()
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_B, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        assert h.dispatcher.claim(_executor("a", model=MODEL_A)) is None
        assert h.dispatcher.claim(_executor("b", model=MODEL_B)) is not None
        scoring.cancel()
    _run(go)


def test_an_executor_without_a_placement_is_refused():
    h = _Harness()
    with pytest.raises(LeaseRefused) as refused:
        h.dispatcher.claim({**_executor("a"), "provider_id": None})
    assert refused.value.status == 409


def test_an_expired_lease_frees_the_seat():
    async def go():
        now = [0.0]
        h = _Harness()
        h.dispatcher._clock = lambda: now[0]
        scoring = asyncio.ensure_future(h.dispatcher.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        h.dispatcher.claim(_executor("a"))
        h.dispatcher.claim(_executor("b"))
        assert h.dispatcher.claim(_executor("c")) is None  # two seats taken
        now[0] = 10_000.0
        await h.dispatcher.sweep()
        assert h.dispatcher.claim(_executor("c")) is not None
        scoring.cancel()
    _run(go)


class _Tokenizer:
    chat_template = "x"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 for c in text]


def test_the_eval_auditor_scores_through_the_pair_and_names_both():
    async def go():
        h = _Harness()
        auditor = eval_auditor(job_id="order-eval-1", records=None, tokenizer=_Tokenizer(),
                               proof=PROOF, vocab_size=100,
                               remote=h.dispatcher.view(*MODEL_A, PROOF))
        record = {"rendered_prompt": "ab", "completions": [
            {"tokens": [1, 2], "proofs": ["p"]}], "hotkey": "h", "token_count": 2}
        judging = asyncio.ensure_future(auditor._forward([record], local=True))
        await asyncio.sleep(0.05)
        one = AuditResult.model_validate({"scores": [{"status": "ok", "chunks": [[1, 1.0, 1.0]]}]})
        for eid in ("a", "b"):
            lease = h.dispatcher.claim(_executor(eid))
            await h.dispatcher.result(eid, lease["lease_id"], one)
        judged = await judging
        assert judged[0]["passed"] is True and judged[0]["scored_by"] == ["a", "b"]
        # A token past the vocabulary fails the miner, no executor asked.
        bad = {**record, "completions": [{"tokens": [1, 999], "proofs": ["p"]}]}
        assert (await auditor._forward([bad]))[0]["passed"] is False
    _run(go)


def test_only_eval_jobs_are_served():
    entry = SimpleNamespace(job_id="code-v1", contract={})
    assert eval_job_refusal(entry, SimpleNamespace(prompt_source="x")) == "not an evaluation job"
    entry = SimpleNamespace(job_id="order-eval-1", contract={})
    assert "eval set" in eval_job_refusal(entry, SimpleNamespace(prompt_source="reliquary_logic_v2"))


def test_an_executor_registration_records_where_it_runs(monkeypatch):
    from reliquary.infrastructure import corpus_executor_store as store
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: fake)

    def register(**kw):
        return asyncio.run(store.register_executor(
            executor_id="e1", token_sha256="a" * 64, model_id="m", model_revision="r",
            expires_at=10.0, now=0.0, **kw))

    document, created = register(provider_id="lium-7", host="10.0.0.1")
    assert created and (document["provider_id"], document["host"]) == ("lium-7", "10.0.0.1")
    assert register(provider_id="lium-7", host="10.0.0.1")[1] is False
    with pytest.raises(store.ExecutorConflict):
        register(provider_id="lium-8", host="10.0.0.1")
    plain, _ = asyncio.run(store.register_executor(
        executor_id="e2", token_sha256="b" * 64, model_id="m", model_revision="r",
        expires_at=10.0, now=0.0))
    assert "provider_id" not in plain and "host" not in plain


def test_the_audit_executor_speaks_the_eval_prefix_when_told():
    import httpx

    from reliquary.validator.corpus_audit_executor import EVAL_AUDIT_PREFIX, AuditExecutor

    paths = []

    async def go():
        def handle(request):
            paths.append(request.url.path)
            if request.url.path.endswith("/claim"):
                return httpx.Response(204)
            return httpx.Response(200, json={"model_id": "m", "model_revision": "r"})

        async with httpx.AsyncClient(base_url="http://c",
                                     transport=httpx.MockTransport(handle)) as http:
            executor = AuditExecutor(http=http, executor_id="e", token="t", model_id="m",
                                     model_revision="r", prefix=EVAL_AUDIT_PREFIX)
            assert await executor.step() is False

    asyncio.run(go())
    assert paths == [f"{EVAL_AUDIT_PREFIX}/heartbeat", f"{EVAL_AUDIT_PREFIX}/claim"]


def test_settlement_archives_only_the_eval_tasks_this_process_serves():
    from reliquary.validator.eval_control import EvalArchives

    written = []

    async def upload(window, data, task_id):
        written.append((task_id, window))

    async def other_max(task_id):
        return 7

    archives = EvalArchives(served=lambda: {"order-eval-1", "code-v1"}, upload=upload,
                            other_max=other_max)
    asyncio.run(archives.write("order-eval-1", 7, {}))
    assert written == [("order-eval-1", 7)]
    for task_id in ("order-eval-2", "code-v1"):
        with pytest.raises(RuntimeError):
            asyncio.run(archives.write(task_id, 7, {}))
    assert asyncio.run(archives.other_max("order-eval-1")) == 7
