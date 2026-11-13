"""Agentic eval orders: a forced-draw eval order on a SWE (or TMax) set carries the signed-sandbox episode that
plays its references; the admin checks it against the set and the model's renderer and records it, and the order
control serves the job only when its episode is the recorded one."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.eval import qualification as qual
from reliquary.eval.storage import SubnetEvalStore
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import SAMPLING, _pin, pins  # noqa: F401

SWE_SET = "swe-eval-r0-n4"
QWEN35 = "Qwen3_5ForConditionalGeneration"


def _swe_episode(**over):
    from tests.unit.test_corpus_job_signed_sandbox import signed_episode

    episode = signed_episode()
    episode["env"] = {**episode["env"], "split": "eval"}
    episode["max_tokens_per_turn"] = 1024
    episode.update(over)
    return episode


@pytest.fixture
def swe_set(admin, monkeypatch):  # noqa: F811
    """A SWE set of 4 references, built and published, its rows checked against a faked bridge."""
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.environment import agentic_swe, bridged_swe
    from reliquary.eval import rows as eval_rows
    from reliquary.eval.sets import build_source_set
    from reliquary.eval.storage import publish_set

    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda *a, **k: None)
    monkeypatch.setattr(agentic_swe, "load_swe_source", lambda *a, **k: [None] * 100)
    monkeypatch.setattr(bridged_swe, "split_length", lambda split: 100)
    monkeypatch.setattr(bridged_swe, "package_version", lambda: "fixture")
    monkeypatch.setattr(bridged_swe, "instance_id_of", lambda split, index: f"repo__x-{index}")
    monkeypatch.setattr(bridged_swe, "row_of", lambda split, index: (
        None, SimpleNamespace(instance_id=f"repo__x-{index}")))
    eval_rows._row_map.cache_clear()
    card = build_source_set("reliquary-swe", split="eval", count=4, set_id=SWE_SET, out=admin.root / "swe",
                     clock=lambda: 1.0)
    asyncio.run(publish_set(admin.root / "swe", platform=LocalDirectorySink(admin.root / "p"),
                            subnet=SubnetEvalStore()))
    yield card
    eval_rows._row_map.cache_clear()


@pytest.fixture
def qwen35(pins):  # noqa: F811
    from tests.unit.test_admin_eval_jobs import MODEL, REVISION

    pins[MODEL] = _pin(MODEL, REVISION, architecture=QWEN35, max_position_embeddings=262144)
    return pins


def _order(**kw):
    from tests.unit.test_admin_eval_jobs import MODEL, REVISION

    base = {"job_id": "order-eval-swe1", "model": MODEL, "revision": REVISION, "env": "reliquary-swe",
            "prompt_count": 3, "samples_per_prompt": 2, "max_new_tokens": 1024, "sampling": SAMPLING,
            "eval_set_id": SWE_SET, "draw": "gumbel_max", "episode": _swe_episode()}
    return {**base, **kw}


def _served(job_id="order-eval-swe1"):
    job, _ = asyncio.run(job_store.read_job(job_id))
    return job, asyncio.run(qual.EvalJobStore().read(job_id))


# -- the admin --------------------------------------------------------------------------------


def test_an_eval_order_on_a_swe_set_declares_its_signed_episode_job(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT

    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 201, response.text
    job, declared = _served()
    assert is_signed_sandbox(job) and job.renderer_id == _swe_episode()["renderer"]
    assert job.episode.env.split == "eval" and job.prompt_count == 3
    assert declared["episode"] == job.episode.to_contract() and declared["thinking"] is None
    entry = admin.registry["entries"]["order-eval-swe1"]
    assert list(entry.contract["environments"]) == [AGENTIC_SWE_ENVIRONMENT]


@pytest.mark.parametrize("change,word", [
    ({"episode": None}, "pass the episode"),
    ({"episode": _swe_episode(env={"package": "reliquary-swe", "version": "a" * 40, "split": "train",
                                   "num_images": 20})}, "split"),
    ({"episode": {k: v for k, v in _swe_episode().items() if k not in ("execution", "sandbox")}},
     "signed sandbox"),
    ({"thinking": True}, "thinking"),
    ({"max_new_tokens": 2048}, "max_tokens_per_turn"),
    ({"draw": None}, "forced-draw"),
])
def test_an_episode_order_unlike_its_set_is_refused(admin, swe_set, qwen35, change, word):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order(**change))
    assert response.status_code == 422 and word in response.text, response.text
    assert "order-eval-swe1" not in admin.registry["entries"]


def test_an_episode_order_on_a_model_without_a_renderer_is_refused(admin, swe_set, pins):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 422 and "episode_renderer_unavailable" in response.text, response.text


def test_a_single_turn_set_takes_no_episode(admin, qwen35):  # noqa: F811
    body = _order(env="logic", eval_set_id="logic-eval-s1-n8", max_new_tokens=512)
    response = admin("POST", "/admin/v1/jobs", body)
    assert response.status_code == 422 and "single-turn" in response.text, response.text


# -- the order control --------------------------------------------------------------------------


def test_the_order_control_serves_the_recorded_episode_only(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal, qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, declared = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    assert order_job_refusal(entry, job) is None
    assert qualification_refusal(entry, job, declared, None) is None
    longer = replace(job, episode=replace(job.episode, max_turns=job.episode.max_turns + 1))
    assert "episode" in qualification_refusal(entry, longer, declared, None)
    assert "episode" in qualification_refusal(entry, job, {**declared, "episode": None}, None)


def test_the_order_control_refuses_an_unsigned_or_unrendered_episode(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.corpus.job import EXECUTION_REPLAY
    from reliquary.validator.eval_control import order_job_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, _ = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    replayed = replace(job, episode=replace(job.episode, execution=EXECUTION_REPLAY, sandbox=None))
    assert "signed sandbox" in order_job_refusal(entry, replayed)
    other = replace(entry, contract={**entry.contract, "model_architecture": "Qwen3ForCausalLM"})
    assert "episode_renderer_unavailable" in order_job_refusal(other, job)


def test_a_qualified_order_never_runs_an_episode(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.validator.eval_control import qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, declared = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    qualified = {**declared, "schema": "reliquary/eval-job/v1", "qualification_id": "q1"}
    assert "forced-draw" in qualification_refusal(entry, job, qualified, {"qualification_id": "q1"})


# -- the order control wires and pays a signed episode order ------------------------------------


def _control(admin, tmp_path, monkeypatch, sandbox, resolver_factory=None):  # noqa: F811
    """The order control over the admin's bucket, its background loops idle (the test drives them)."""
    from reliquary.environment.agentic_swe import SignedSweSource
    from reliquary.eval.rows import RowMap
    from reliquary.infrastructure import corpus_executor_store as executors
    from reliquary.infrastructure import corpus_record_store as record_store
    from reliquary.infrastructure import drand
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_auditor, corpus_period_settlement, corpus_grading
    from reliquary.validator.eval_control import EvalExecutorDirectory, PairedAuditDispatcher, build_eval_control
    from tests.unit.test_eval_control_process import _Tokenizer

    async def idle(self):
        await asyncio.sleep(3600)

    for module in (record_store, executors):
        monkeypatch.setattr(module, "get_s3_client", lambda **kw: admin.bucket)
    monkeypatch.setattr(drand, "get_current_chain", lambda: {})
    # The fake bucket has no ranged reads: a record's metadata is the record.
    monkeypatch.setattr(BucketRecordStore, "read_submission_meta",
                        lambda self, job_id, sid: self.read_submission(job_id, sid))
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_grading.CorpusGrader, "run", idle)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", idle)
    source = SignedSweSource(RowMap.identity("eval", 0, 100), prompt_of=lambda s, i: "p",
                             row_of=lambda s, i: (None, SimpleNamespace(instance_id=f"repo__x-{i}")))

    async def read_entries():
        return dict(admin.registry["entries"])

    directory = EvalExecutorDirectory()
    return build_eval_control(
        store=BucketJobStore(), records=BucketRecordStore(), dispatcher=PairedAuditDispatcher(directory=directory),
        directory=directory, verify_signature=lambda request: True,
        tokenizer_for=lambda repo, revision: (_Tokenizer(), 100), read_entries=read_entries,
        sandbox=sandbox, sandbox_down="its sandbox services failed to start" if sandbox is None else None,
        model_dir_for=lambda repo, revision: str(tmp_path),
        signed_wiring={"intake_factory": lambda job, **kw: SimpleNamespace(source=source),
                       "resolver_factory": resolver_factory or (lambda split: SimpleNamespace(resolve=None))})


def _order_services(tmp_path, route_prefix):
    from pathlib import Path

    from reliquary.infrastructure.sandbox_store import MemorySessionStore
    from reliquary.validator.sandbox_wiring import SandboxValidatorConfig, build_sandbox_services
    from tests.unit.sandbox_fixtures import signer
    from tests.unit.test_signed_grading_and_wiring import VALIDATOR

    signer(tmp_path, "v", "v1")

    async def none(*args):
        return [] if not args else None

    return build_sandbox_services(SandboxValidatorConfig(Path(tmp_path / "v.pem"), "v1", {}, None),
                                  validator_hotkey=VALIDATOR, session_store=MemorySessionStore(),
                                  read_documents=none, fetch_report=none, route_prefix=route_prefix)


def test_an_episode_order_is_refused_without_the_controls_sandbox(admin, swe_set, qwen35, tmp_path,  # noqa: F811
                                                                  monkeypatch):
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    app, job_set = _control(admin, tmp_path, monkeypatch, None)

    async def go():
        await job_set.refresh()
        return dict(job_set.served)

    assert "order-eval-swe1" not in asyncio.run(go())
    reasons = app.state.dispatcher.attention["order-eval-swe1"]
    assert any("sandbox services failed to start" in r for r in reasons), reasons


def test_an_episode_order_is_wired_and_never_paid_before_its_grade(admin, swe_set, qwen35, tmp_path,  # noqa: F811
                                                                   monkeypatch):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2, BucketRecordStore
    from reliquary.validator import corpus_period_settlement
    from reliquary.validator.signed_grading import SignedEpisodeGrader
    from tests.unit.sandbox_fixtures import claims, signer, transcript
    from tests.unit.test_corpus_period_settlement import GENESIS, Archives, at, verdict
    from reliquary.protocol.sandbox_session import JOB_SCOPED_PREFIX
    from reliquary.validator import corpus_service

    settle = corpus_period_settlement.CorpusPeriodSettler.settle_once
    built, build_router = {}, corpus_service.build_corpus_router

    def spy(**kwargs):
        built[kwargs["job_id"]] = kwargs
        return build_router(**kwargs)

    monkeypatch.setattr(corpus_service, "build_corpus_router", spy)
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    services = _order_services(tmp_path, JOB_SCOPED_PREFIX)
    app, job_set = _control(admin, tmp_path, monkeypatch, services)
    job_id, sid = "order-eval-swe1", "a" * 64

    async def go():
        await job_set.refresh()
        assert job_id in job_set.served
        w = app.state.eval_served[job_id]
        assert isinstance(w.grader, SignedEpisodeGrader) and services.jobs[job_id] is w.sandbox_view
        assert built[job_id]["episode_intake"] is w.episode_intake
        import httpx

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://o") as client:
            # The session route is mounted (an unsigned open is refused, never 404).
            opened = await client.post(f"/corpus/jobs/{job_id}/sandbox/sessions", json={})
            assert opened.status_code not in (404, 405), opened.text
            assert (await client.post("/corpus/sandbox/sessions", json={})).status_code == 404
        # One passing, audited episode received in period 3; it is paid only once graded.
        records = BucketRecordStore()
        signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(), reward=1.0)
        await records.write_submission(job_id, sid, {
            "schema": RECORD_SCHEMA_V2, "hotkey": "5Hot", "prompt_index": 0, "received_at": at(3),
            "completions": [{"transcript": signed}]})
        await records.write_verdict(job_id, sid, verdict(sid, "5Hot", 10, at(3)))
        w.settler._genesis, w.settler._clock, w.settler._archives = (lambda: GENESIS), (lambda: at(6)), Archives()
        w.auditor.oldest_pending_received_at = lambda: None   # audited: only its grade is missing
        w.grader._enqueue = lambda submission_id: None        # listed, not graded yet
        await w.grader.rescan_once()
        unpaid = await settle(w.settler)
        assert w.settler._archives.docs == {}
        assert await w.grader.grade_one(sid) is not None
        paid = await settle(w.settler)
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()
        return unpaid, paid, w.settler._archives.docs

    unpaid, paid, docs = asyncio.run(go())
    assert unpaid is None and paid == 3
    assert docs[(3, 7)]["rewards_by_hotkey"] == {"5Hot": pytest.approx(0.02)}


def test_an_episode_order_is_held_from_miners_until_a_gateway_serves_its_env(admin, swe_set, qwen35,  # noqa: F811
                                                                            tmp_path, monkeypatch):
    pytest.importorskip("reliquary_sandbox.attest")
    import httpx

    from reliquary.protocol.sandbox_session import JOB_SCOPED_PREFIX
    from reliquary.sandbox.tasks import ResolvedTask

    digest = "d" * 64

    async def resolve(index):
        return ResolvedTask("img", env_options_sha256=digest)

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    services = _order_services(tmp_path, JOB_SCOPED_PREFIX)
    slots, asked = {"n": 0}, []

    def slots_serving_env(env, env_package, *, env_options_sha256=None, now=None):
        asked.append((env, env_package, env_options_sha256))
        return slots["n"]

    services.fleet.slots_serving_env = slots_serving_env
    app, job_set = _control(admin, tmp_path, monkeypatch, services,
                            resolver_factory=lambda split: SimpleNamespace(resolve=resolve))
    app.state.dispatcher.audit_ready = lambda *args: True
    job_id = "order-eval-swe1"

    async def go():
        await job_set.refresh()
        w = app.state.eval_served[job_id]
        w.auditor.seeded.set()
        spec = w.job.episode.sandbox
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://o") as client:
            async def listed():
                return (await client.get("/corpus/jobs")).json()

            # No gateway yet: listed as starting, not open; its status says why.
            held = await listed()
            assert held["jobs"] == [] and held["orders"][0]["state"] == "starting"
            assert (await app.state.control_status())["jobs"][job_id]["held"] == "no_gateway"
            assert (await client.get(f"/corpus/jobs/{job_id}/status")).json()["held"] == "no_gateway"
            assert asked[-1] == (str(spec.env), str(spec.env_package), digest)
            # A gateway serves it: open at once, and it stays open if the fleet drops again.
            slots["n"] = 4
            assert (await listed())["jobs"] == [job_id]
            assert "held" not in (await app.state.control_status())["jobs"][job_id]
            slots["n"] = 0
            assert (await listed())["jobs"] == [job_id]
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())


def test_an_episode_order_waits_for_its_audit_after_its_gateway(admin, swe_set, qwen35, tmp_path,  # noqa: F811
                                                                monkeypatch):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.protocol.sandbox_session import JOB_SCOPED_PREFIX

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    services = _order_services(tmp_path, JOB_SCOPED_PREFIX)
    services.fleet.slots_serving_env = lambda *a, **k: 2
    app, job_set = _control(admin, tmp_path, monkeypatch, services)
    job_id = "order-eval-swe1"

    async def go():
        await job_set.refresh()
        # The resolver fails here: held for a gateway at the package alone, never refused.
        assert app.state.eval_served[job_id].env_options_sha256 is None
        routes = job_set._routes
        assert routes.held_reason(job_id) == "awaiting_audit" and routes.open_jobs() == []
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())


def test_the_order_control_reads_only_its_own_sandbox_key(tmp_path):
    pytest.importorskip("reliquary_sandbox")
    from reliquary.infrastructure.sandbox_store import ORDER_SESSION_PREFIX
    from reliquary.validator.eval_control import order_sandbox_services

    built = []

    class Services:
        async def start(self):
            return None

    def build(config, **kwargs):
        built.append((config, kwargs))
        return Services()

    corpus_key = {"RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": "/corpus.pem", "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "v1",
                  "RELIQUARY_SANDBOX_VALIDATOR_HOTKEY": "5Corpus"}
    # The corpus validator's key is never this control's: without its own, no sandbox.
    assert asyncio.run(order_sandbox_services(environ=corpus_key, build=build)) == (None, None)
    own = {"RELIQUARY_ORDER_SANDBOX_KEY_FILE": str(tmp_path / "order.pem"),
           "RELIQUARY_ORDER_SANDBOX_KEY_ID": "order-1"}
    services, why = asyncio.run(order_sandbox_services(environ={**corpus_key, **own}, build=build))
    assert services is None and "RELIQUARY_ORDER_SANDBOX_HOTKEY" in why
    services, why = asyncio.run(order_sandbox_services(
        environ={**corpus_key, **own, "RELIQUARY_ORDER_SANDBOX_HOTKEY": "5Order"}, build=build))
    assert isinstance(services, Services) and why is None
    config, kwargs = built[-1]
    assert (str(config.key_file), config.key_id) == (own["RELIQUARY_ORDER_SANDBOX_KEY_FILE"], "order-1")
    assert kwargs["validator_hotkey"] == "5Order"
    # Its sessions are its own: the corpus validator never restores them, nor it the corpus validator's.
    assert kwargs["session_store"]._prefix == ORDER_SESSION_PREFIX
    assert kwargs["route_prefix"] == "/corpus/jobs/{job_id}"


def test_a_sandbox_that_fails_to_start_refuses_episode_orders_only():
    pytest.importorskip("reliquary_sandbox")
    from reliquary.validator.eval_control import order_sandbox_services

    class Broken:
        async def start(self):
            raise ConnectionError("restore failed")

    environ = {"RELIQUARY_ORDER_SANDBOX_KEY_FILE": "/o.pem", "RELIQUARY_ORDER_SANDBOX_KEY_ID": "o1",
               "RELIQUARY_ORDER_SANDBOX_HOTKEY": "5Order"}
    services, why = asyncio.run(order_sandbox_services(environ=environ, build=lambda c, **k: Broken(),
                                                       session_store=object()))
    assert services is None and "failed to start" in why


# -- the order's grading ----------------------------------------------------------------------


def _graded_episodes(admin, rewards, ungraded=()):  # noqa: F811
    """Passing, settled episodes of the order (2 per problem) and the grades its grader wrote."""
    grades = {}

    async def read_grade(job_id, sid):
        return grades.get(sid)

    async def list_grade_ids(job_id):
        return sorted(grades)

    admin.records.read_grade, admin.records.list_grade_ids = read_grade, list_grade_ids
    for prompt in range(3):
        for k in range(2):
            sid = f"{prompt * 2 + k:064x}"
            admin.records.subs[sid] = {"prompt_index": prompt, "hotkey": "hk1", "completions": [{"tokens": [5, 2]}]}
            admin.records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": "hk1", "token_count": 7}
            admin.records.settled.append(sid)
            if sid not in ungraded:
                reward = rewards[prompt]
                grades[sid] = {"status": "ok", "replay_certified": True, "grade": {"reward": reward},
                               "graded_success": reward == 1.0, "prompt_index": prompt, "hotkey": "hk1",
                               "instance_id": f"repo__x-{prompt}"}


def _grade(admin, eval_id="order-swe-e1", **kw):  # noqa: F811
    import time

    body = {"source": "job", "job_id": "order-eval-swe1", "set_ids": [SWE_SET], "problems_per_set": {SWE_SET: 3},
            "samples_per_set": {SWE_SET: 2}, **kw}
    for _ in range(200):
        response = admin("POST", f"/admin/v1/evaluations/{eval_id}/grade", body)
        if response.status_code != 202:
            return response
        time.sleep(0.02)
    return response


def test_an_episode_order_is_graded_from_its_graders_grades(admin, swe_set, qwen35):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    _graded_episodes(admin, rewards=[1.0, 0.0, 1.0])
    response = _grade(admin)
    assert response.status_code == 200, response.text
    report = json.loads((admin.root / "p" / "evaluations" / "order-swe-e1" / "report.json").read_text())
    verification = report["provenance"]["verification"]
    assert verification["scheme"] == "gumbel_max" and verification["grade"] == "signed-sandbox"
    assert report["provenance"]["job_complete"] is True
    assert json.dumps(report).count("0.6666") >= 1, report        # 4 of 6 episodes solved


def test_an_episode_order_with_ungraded_passes_waits_for_its_grader(admin, swe_set, qwen35):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    _graded_episodes(admin, rewards=[1.0, 1.0, 1.0], ungraded={f"{0:064x}"})
    response = _grade(admin)
    assert response.status_code == 409 and "job_not_graded" in response.text, response.text
