"""v2 cross-side fixes: executor heartbeats with their detail, eval jobs that
stay completable (a failure reopens its slot), and the configurable prefix."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from reliquary.eval.prompt_source import eval_job_prefix, is_eval_job_id
from tests.unit.test_eval_prompt_source import _job


def test_the_eval_prefix_follows_the_admin_task_prefix(monkeypatch):
    monkeypatch.delenv("RELIQUARY_ADMIN_TASK_PREFIX", raising=False)
    assert eval_job_prefix() == "order-eval-" and is_eval_job_id("order-eval-1")
    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "acme-")
    assert eval_job_prefix() == "acme-eval-"
    assert is_eval_job_id("acme-eval-1") and not is_eval_job_id("order-eval-1")
    assert eval_job_prefix("x-") == "x-eval-"
    from reliquary.validator.corpus_hot_jobs import eval_entry_screen

    assert eval_entry_screen(SimpleNamespace(job_id="acme-eval-2")) is not None
    assert eval_entry_screen(SimpleNamespace(job_id="order-eval-2")) is None


def test_the_admin_takes_eval_ids_from_its_own_task_prefix(tmp_path, monkeypatch):
    import secrets
    import time

    from fastapi.testclient import TestClient

    from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
    from reliquary.admin.service import create_admin_app
    from reliquary.corpus.delivery import LocalDirectorySink

    app = create_admin_app(secret=b"s" * 32, pool_max=0.3, models={}, records=object(),
                           task_prefix="acme-", eval_store=LocalDirectorySink(tmp_path))
    client = TestClient(app)

    def post(body):
        data = json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        return client.post("/admin/v1/jobs", content=data, headers={
            TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
            SIGNATURE_HEADER: sign_request(b"s" * 32, stamp, nonce, "POST", "/admin/v1/jobs", data),
            "content-type": "application/json"})

    base = {"model": "m", "env": "logic", "prompt_count": 1, "samples_per_prompt": 1,
            "eval_set_id": "s", "qualification_id": "acme-q"}
    refused = post({**base, "job_id": "acme-7"})
    assert refused.status_code == 422 and "acme-eval-" in refused.text
    # The right prefix gets past the id rule (to the next refusal: the sampling).
    assert "acme-eval-" not in post({**base, "job_id": "acme-eval-7"}).text


# -- heartbeats ----------------------------------------------------------------


def test_eval_heartbeats_reach_the_registry_with_leases_and_loaded():
    from reliquary.validator.eval_control import PairedAuditDispatcher

    async def go():
        now = [0.0]
        written = []

        async def write(executor_id, at, detail):
            written.append((executor_id, at, detail))

        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None),
                                           record_heartbeat=write, clock=lambda: now[0])
        dispatcher.heartbeat("q", detail={"loaded": False})
        await dispatcher.write_heartbeats()
        now[0] = 5.0
        dispatcher.heartbeat("q", detail={"loaded": True, "leases": 1})
        await dispatcher.write_heartbeats()  # changed: written at once
        now[0] = 9.0
        dispatcher.heartbeat("q", detail={"loaded": True, "leases": 1})
        await dispatcher.write_heartbeats()  # same, too soon
        assert written == [("q", 0.0, {"leases": 0, "loaded": False}),
                           ("q", 5.0, {"leases": 1, "loaded": True})]
    asyncio.run(go())


def test_the_registry_stores_the_heartbeat(monkeypatch):
    from reliquary.infrastructure import corpus_executor_store as store
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: fake)

    async def go():
        await store.register_executor(executor_id="q", token_sha256="a" * 64, model_id="m",
                                      model_revision="r", expires_at=1e12, now=0.0,
                                      provider_id="p", host="h", scope="eval")
        await store.record_heartbeat("q", at=7.0, detail={"leases": 1, "loaded": True})
        document = await store.read_executor("q")
        assert document["last_heartbeat"] == 7.0
        assert document["heartbeat"] == {"leases": 1, "loaded": True}
    asyncio.run(go())


def test_the_qualify_executor_heartbeats_while_it_loads_and_qualifies():
    from reliquary.eval.qualify_executor import QualifyExecutor

    beats = []

    def handle(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/heartbeat"):
            beats.append(body["detail"])
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    http = httpx.Client(base_url="http://c", transport=httpx.MockTransport(handle))
    executor = QualifyExecutor(http=http, executor_id="q", token="t", model_id="m",
                               model_revision="r", run=lambda lease: {})
    heartbeats = executor.heartbeats(every=0.01)
    heartbeats.start()
    import time

    time.sleep(0.05)
    heartbeats.state["loaded"] = True
    time.sleep(0.05)
    heartbeats.stop()
    assert beats[0] == {"loaded": False, "leases": 0}
    assert {"loaded": True, "leases": 0} in beats


def test_qualify_contacts_count_as_heartbeats(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.validator.eval_control import PairedAuditDispatcher, build_eval_executor_router

    document = {"executor_id": "q", "model_id": "m", "model_revision": "r",
                "provider_id": "p", "host": "h", "scope": "eval"}
    directory = SimpleNamespace(authenticate=lambda token, eid=None: (document, None),
                                revoke_locally=lambda e: None)
    dispatcher = PairedAuditDispatcher(directory=directory)

    class _Queue:
        async def claim(self, executor):
            return None

        def lease_of(self, lease_id):
            return None

    app = FastAPI()
    app.include_router(build_eval_executor_router(dispatcher=dispatcher, directory=directory,
                                                  qualifications=_Queue()))
    client = TestClient(app)
    answer = client.post("/corpus/internal/eval-audit/claim", headers={"Authorization": "Bearer t"},
                         json={"executor_id": "q", "model_id": "m", "model_revision": "r",
                               "kind": "qualify"})
    assert answer.status_code == 204 and "q" in dispatcher._seen
    client.post("/corpus/internal/eval-audit/heartbeat", headers={"Authorization": "Bearer t"},
                json={"executor_id": "q", "detail": {"loaded": True, "leases": 0}})
    assert dispatcher.heartbeat_detail("q") == {"leases": 0, "loaded": True}


# -- completable eval jobs -------------------------------------------------------


def test_an_eval_jobs_status_says_complete_and_exhausted_prompts():
    from reliquary.corpus.slots import SlotLedger
    from reliquary.validator.corpus_job_status import JobStats, job_status

    job = _job("eval-set:s:3:" + "0" * 64, count=3)  # slots_per_prompt 1
    slots = SlotLedger(3, 1)
    slots.consume(0)
    for k in range(3):  # three attempts at prompt 1, each failed
        slots.consume(1)
        slots.record_failure(1, str(k) * 64)
    status = job_status(job_id=job.job_id, job=job, slots=slots, stats=JobStats(), settled=0,
                        totals=None, retired=False)
    assert (status["prompts_complete"], status["prompts_exhausted"]) == (1, 1)
    assert status["complete"] is False  # prompt 2 is open
    plain = job_status(job_id="code-v1", job=_job("fake-env"), slots=SlotLedger(10, 1),
                       stats=JobStats(), settled=0, totals=None, retired=False)
    assert "prompts_complete" not in plain and "complete" not in plain


class _Records:
    def __init__(self):
        self.subs, self.verdicts = {}, {}

    async def write_verdict(self, job_id, sid, verdict):
        if sid in self.verdicts:
            return False
        self.verdicts[sid] = verdict
        return True

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)


def test_a_failed_audit_reopens_its_prompts_slot(monkeypatch):
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.validator.corpus_service import rebuild_ledgers
    from reliquary.validator.eval_control import eval_auditor
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    store = BucketJobStore()
    job = _job("eval-set:s:4:" + "0" * 64, count=4)
    records = _Records()
    records.subs = {"a" * 64: {"prompt_index": 2}, "b" * 64: {"prompt_index": 3}}
    auditor = eval_auditor(job_id=job.job_id, records=records, tokenizer=None,
                           proof=TOPLOC_DEPLOYED_DEFAULTS, vocab_size=10,
                           remote=SimpleNamespace(subscribe=lambda l: None), job=job,
                           job_store=store)

    async def go():
        await auditor._write("a" * 64, {"passed": False})
        await auditor._write("b" * 64, {"passed": True})
        snapshot, _ = await store.read_ledgers(job.job_id)
        slots = rebuild_ledgers(job, snapshot).slots
        assert slots.failed_snapshot() == {2: ["a" * 12]}
        assert slots.capacity(2) == job.slots_per_prompt + 1
        # The reconcile at start records it again, idempotently, and any missed one.
        records.verdicts["c" * 64] = {"passed": False}
        records.subs["c" * 64] = {"prompt_index": 2}
        assert await auditor.reconcile_failures() == 2
        snapshot, _ = await store.read_ledgers(job.job_id)
        assert rebuild_ledgers(job, snapshot).slots.failed_snapshot() == {
            2: ["a" * 12, "c" * 12]}
    asyncio.run(go())


def test_grading_counts_exhausted_prompts_as_complete_failures():
    from reliquary.corpus.slots import SlotLedger
    from reliquary.eval.grading import exhausted_prompts, job_complete

    job = _job("eval-set:s:2:" + "0" * 64, count=2)  # one slot per prompt, samples 1
    slots = SlotLedger(2, 1)
    slots.consume(0)
    for k in range(3):
        if slots.remaining(1):
            slots.consume(1)
        slots.record_failure(1, str(k) * 64)
    collected = {"samples_by_prompt": {0: 1}}
    assert slots.prompt_state(1) == "exhausted"
    assert job_complete(job, collected, 1) is False
    assert job_complete(job, collected, 1, slots) is True
    assert exhausted_prompts(job, collected, 1, slots) == [1]
    assert job_complete(job, {"samples_by_prompt": {}}, 1, slots) is False
