"""The eval control as one process: two eval jobs on two models, wired hot from
the registry, served under their prefix, executors and a qualification over
HTTP; a corpus job is never taken."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from reliquary.eval import prompt_source as ps
from reliquary.eval import qualification as qual
from reliquary.infrastructure import corpus_executor_store as executors
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as record_store
from tests.unit.test_admin_eval_jobs import THRESHOLDS
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
from tests.unit.test_eval_sets import opener
from tests.unit.test_jobs_cli import _rl_entry, registry  # noqa: F401

SAMPLING = {"temperature": 0.6, "top_p": 1.0, "top_k": 0}
MODELS = {"order-eval-a": ("customer/A", "a" * 40), "order-eval-b": ("customer/B", "b" * 40)}


class _Tokenizer:
    chat_template = "x"

    def apply_chat_template(self, messages, **kwargs):
        return f"<u>{messages[0]['content']}</u>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 90 for c in text]


@pytest.fixture
def world(tmp_path, monkeypatch, registry):  # noqa: F811
    from reliquary.admin.service import create_admin_app
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval.sets import build_set
    from reliquary.eval.storage import SubnetEvalStore, publish_set

    fake = _FakeMultiObjectR2()
    for module in (job_store, executors, record_store):
        monkeypatch.setattr(module, "get_s3_client", lambda **kw: fake)
    monkeypatch.setattr(ps, "_loaded", {})
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    build_set("logic", count=6, seed=1, out=tmp_path / "set", open_environment=opener(),
              clock=lambda: 1.0)
    asyncio.run(publish_set(tmp_path / "set", platform=LocalDirectorySink(tmp_path / "p"),
                            subnet=SubnetEvalStore()))
    store = qual.QualificationStore()
    admin = create_admin_app(secret=b"s" * 32, pool_max=0.3, models={}, records=object(),
                             current_round=lambda: 1)

    async def declare():
        for job_id, (model, revision) in MODELS.items():
            qid = f"order-q-{job_id[-1]}"
            record = qual.new_request(qualification_id=qid, model=model, revision=revision,
                                      set_id="logic-eval-s1-n6", problems=4, completions=2,
                                      sampling=SAMPLING, max_new_tokens=64, thinking=False)
            record.update(status=qual.QUALIFIED, result={
                "thresholds": THRESHOLDS, "architecture": "Qwen3ForCausalLM",
                "checkpoint_sha256": job_id[-1] * 64, "eos_token_id": 2})
            await store.write(record, None)
        transport = httpx.ASGITransport(app=admin)
        import secrets
        import time

        from reliquary.admin.auth import (
            NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request,
        )

        async with httpx.AsyncClient(transport=transport, base_url="http://admin") as client:
            for job_id, (model, _) in MODELS.items():
                body = json.dumps({"job_id": job_id, "model": model, "env": "logic",
                                   "prompt_count": 4, "samples_per_prompt": 2,
                                   "max_new_tokens": 64, "sampling": SAMPLING,
                                   "eval_set_id": "logic-eval-s1-n6",
                                   "qualification_id": f"order-q-{job_id[-1]}"}).encode()
                stamp, nonce = str(int(time.time())), secrets.token_hex(16)
                path = "/admin/v1/jobs"
                response = await client.post(path, content=body, headers={
                    TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                    SIGNATURE_HEADER: sign_request(b"s" * 32, stamp, nonce, "POST", path, body),
                    "content-type": "application/json"})
                assert response.status_code == 201, response.text

    asyncio.run(declare())
    return registry, tmp_path


def test_one_process_serves_eval_jobs_of_two_models(world, monkeypatch):
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_auditor, corpus_settlement
    from reliquary.validator.eval_control import (
        EvalExecutorDirectory,
        PairedAuditDispatcher,
        build_eval_control,
    )

    registry, tmp_path = world

    async def idle(self):
        await asyncio.sleep(3600)

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", lambda self: idle(self))
    loaded = []

    def tokenizer_for(repo, revision):
        loaded.append(repo)
        return _Tokenizer(), 100

    async def read_entries():
        return dict(registry["entries"])

    async def read_prompts(set_id):
        from reliquary.eval.storage import SubnetEvalStore, subnet_key

        return await SubnetEvalStore().get_bytes(subnet_key(set_id, "prompts.jsonl"))

    async def go():
        # Fresh: the control reads the set's prompts from the bucket itself.
        ps._loaded.clear()
        token = {"e1": "token-one", "e2": "token-two"}
        for eid, (provider, host) in {"e1": ("lium-1", "h1"), "e2": ("lium-2", "h2")}.items():
            await executors.register_executor(
                executor_id=eid, token_sha256=__import__("hashlib").sha256(
                    token[eid].encode()).hexdigest(),
                model_id="customer/A", model_revision="a" * 40, expires_at=1e12, now=0.0,
                provider_id=provider, host=host, scope="eval")
        # A corpus executor of the same model: never this control's.
        await executors.register_executor(
            executor_id="prod", token_sha256=__import__("hashlib").sha256(b"prod").hexdigest(),
            model_id="customer/A", model_revision="a" * 40, expires_at=1e12, now=0.0)
        directory = EvalExecutorDirectory()
        await directory.refresh()
        dispatcher = PairedAuditDispatcher(directory=directory)
        async def facts(repo, revision):
            return {"architecture": "Qwen3ForCausalLM", "eos_token_id": 2}

        queue = qual.QualificationQueue(store=qual.QualificationStore(),
                                        read_prompts=read_prompts, model_facts=facts)
        app, job_set = build_eval_control(
            store=BucketJobStore(), records=BucketRecordStore(), dispatcher=dispatcher,
            directory=directory, verify_signature=lambda request: True,
            tokenizer_for=tokenizer_for, qualifications=queue, read_entries=read_entries)
        await job_set.refresh()
        assert sorted(job_set.served) == ["order-eval-a", "order-eval-b"]
        assert sorted(loaded) == ["customer/A", "customer/B"]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval") as client:
            job = (await client.get("/corpus/jobs/order-eval-b/job")).json()
            assert job["checkpoint_repo"] == "customer/B"
            prompts = (await client.get("/corpus/jobs/order-eval-a/eval-prompts")).content
            source = ps.parse_eval_source(job["prompt_source"])
            assert ps.register_eval_prompts(source, prompts)  # hashes to the manifest
            contract = (await client.get("/corpus/jobs/order-eval-a/contract")).json()
            assert contract["model_id"] == "customer/A"
            assert (await client.get("/corpus/jobs/default/job")).status_code == 404
            assert (await client.get("/corpus/job")).status_code == 404  # no legacy paths
            headers = {"Authorization": "Bearer token-one"}
            claim = {"executor_id": "e1", "model_id": "customer/A", "model_revision": "a" * 40}
            first = await client.post("/corpus/internal/eval-audit/claim", json=claim,
                                      headers=headers)
            assert first.status_code == 204, first.text
            assert (await client.post("/corpus/internal/eval-audit/claim", json=claim,
                                      headers={"Authorization": "Bearer nope"})).status_code == 401
            beat = await client.post("/corpus/internal/eval-audit/heartbeat",
                                     json={"executor_id": "e1"}, headers=headers)
            assert beat.status_code == 200
            prod = await client.post("/corpus/internal/eval-audit/claim", headers={
                "Authorization": "Bearer prod"}, json={**claim, "executor_id": "prod"})
            assert prod.status_code == 401 and prod.json()["detail"] == "wrong_scope"
            # A qualification for model A, measured by both executors over HTTP.
            await qual.QualificationStore().write(qual.new_request(
                qualification_id="order-q-new", model="customer/A", revision="a" * 40,
                set_id="logic-eval-s1-n6", problems=4, completions=4, sampling=SAMPLING,
                max_new_tokens=64, thinking=False), None)
            await queue.refresh()
            result = {"type": "qualify", "chunks": [[3, 2.0, 1.0]] * 20, "completions": 4,
                      "failed_completions": 0, "completion_tokens": 100,
                      "decode_seconds": 1.0, "gpu_count": 1, "gpu": "H100",
                      "vllm_version": "0.1", "checkpoint_sha256": "a" * 64}
            outcomes = []
            for eid in ("e1", "e2"):
                auth = {"Authorization": f"Bearer {token[eid]}"}
                lease = await client.post("/corpus/internal/eval-audit/claim", headers=auth,
                                          json={**claim, "executor_id": eid, "kind": "qualify"})
                assert lease.status_code == 200 and lease.json()["type"] == "qualify"
                posted = await client.post(
                    f"/corpus/internal/eval-audit/{lease.json()['lease_id']}/result",
                    json=result, headers=auth)
                outcomes.append(posted.json()["outcome"])
            assert outcomes == ["pending", "qualified"]
            record, _ = await qual.QualificationStore().read("order-q-new")
            assert record["result"]["eos_token_id"] == 2
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())


def test_the_routing_document_names_both_prefixes():
    from pathlib import Path

    text = Path("docs/design/2026-10-01-evaluation-on-subnet-design.md").read_text()
    assert "^/corpus/jobs/order-eval-" in text and "^/corpus/internal/eval-audit/" in text


def test_a_drained_eval_job_is_unwired_and_releases_what_it_held(world, monkeypatch):
    from dataclasses import replace

    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_auditor, corpus_settlement
    from reliquary.validator.eval_control import (
        EvalExecutorDirectory,
        PairedAuditDispatcher,
        build_eval_control,
    )

    registry, _ = world

    async def idle(self):
        await asyncio.sleep(3600)

    async def nothing(self):
        return []

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "pending_ids", nothing)
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", lambda self: idle(self))

    async def read_entries():
        return dict(registry["entries"])

    async def go():
        dispatcher = PairedAuditDispatcher(directory=EvalExecutorDirectory(
            list_documents=lambda: asyncio.sleep(0, [])))
        app, job_set = build_eval_control(
            store=BucketJobStore(), records=BucketRecordStore(), dispatcher=dispatcher,
            directory=EvalExecutorDirectory(list_documents=lambda: asyncio.sleep(0, [])),
            verify_signature=lambda request: True,
            tokenizer_for=lambda repo, revision: (_Tokenizer(), 100), read_entries=read_entries)
        await job_set.refresh()
        assert sorted(app.state.eval_served) == ["order-eval-a", "order-eval-b"]
        listeners = len(dispatcher._listeners)
        assert listeners == 2
        for job_id in ("order-eval-a", "order-eval-b"):
            registry["entries"][job_id] = replace(registry["entries"][job_id], status="retired",
                                                  retired_at=5)
        await job_set.refresh()
        await job_set.refresh()
        assert job_set.served == {}
        assert app.state.eval_served == {} and app.state.eval_tokenizers == {}
        assert dispatcher._listeners == [] and ps._loaded == {}
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())
