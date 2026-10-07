"""R4: `reliquary admin serve` — signed routes over the registry, the job
store, the executor registry and the platform bucket."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.infrastructure import corpus_executor_store as executors
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
from tests.unit.test_jobs_cli import _rl_entry, registry, stub_source_rows  # noqa: F401

SECRET = b"admin-secret"
MODEL = "Qwen/Qwen3.8-27B"
MODELS = {MODEL: {"revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                  "architecture": "Qwen3_5ForConditionalGeneration",
                  "checkpoint_sha256": "a" * 64, "eos_token_id": 151645}}
SOURCE = "openmathinstruct"


class _Records:
    def __init__(self):
        self.subs, self.verdicts, self.settlement = {}, {}, {}

    async def list_submission_ids(self, job_id):
        return sorted(self.subs)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)

    async def read_settlement(self, job_id):
        return dict(self.settlement), None


@pytest.fixture
def bucket(monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: fake)
    return fake


@pytest.fixture
def admin(bucket, registry, monkeypatch, tmp_path):  # noqa: F811
    from reliquary.corpus.delivery import LocalDirectorySink

    stub_source_rows(monkeypatch, SOURCE, 100_000)
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    records = _Records()
    app = create_admin_app(secret=SECRET, pool_max=0.3, models=MODELS, records=records,
                           task_prefix="math-",
                           deliveries=LocalDirectorySink(tmp_path / "platform"),
                           current_round=lambda: 777, work_dir=tmp_path / "work")
    client = TestClient(app)
    # One event loop for every request, as uvicorn has: an export outlives its request.
    client.__enter__()

    def call(method, path, body=None, *, timestamp=None, secret=SECRET, raw=None, nonce=None):
        data = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
        stamp = str(int(timestamp if timestamp is not None else time.time()))
        nonce = nonce or secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(secret, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return client.request(method, path, content=data, headers=headers)

    call.client, call.records, call.registry, call.bucket = client, records, registry, bucket
    call.platform = tmp_path / "platform"
    yield call
    client.__exit__(None, None, None)


def _job(job_id="math-a", cap=0.1, **kw):
    return {"job_id": job_id, "model": MODEL, "env": SOURCE, "prompt_start": 0,
            "prompt_count": 1000, "samples_per_prompt": 4, "cap": cap, **kw}


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------


def test_an_unsigned_request_is_refused(admin):
    assert admin.client.get("/admin/v1/executors/pod-1").status_code == 401


def test_signed_pause_resume_is_reversible_and_retirement_cannot_resume(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    before = admin("GET", "/admin/v1/tasks/math-a").json()["task"]
    path = "/admin/v1/tasks/math-a/admission"
    assert admin.client.post(path, json={"admission": "paused"}).status_code == 401
    assert admin("GET", "/admin/v1/task-catalog").json()["admission_controls_supported"] is True
    for admission in ("paused", "paused", "open"):
        response = admin("POST", path, {"admission": admission})
        assert response.status_code == 200
        assert response.json() == {"task_id": "math-a", "status": "active", "admission": admission}
        after = admin("GET", "/admin/v1/tasks/math-a").json()["task"]
        assert after["admission"] == admission
        assert after["profile_sha256"] == before["profile_sha256"]
        assert {k: v for k, v in after.items() if k != "admission"} == {
            k: v for k, v in before.items() if k != "admission"}
    assert admin("POST", path, {"admission": "open", "cap": 0}).status_code == 422
    assert admin("POST", "/admin/v1/tasks/math-a/retire", {}).status_code == 200
    assert admin("POST", path, {"admission": "open"}).status_code == 409


def test_a_replayed_request_is_refused(admin):
    path = "/admin/v1/executors/pod-1"
    assert admin("GET", path, nonce="ab" * 16).status_code == 404
    replay = admin("GET", path, nonce="ab" * 16)
    assert replay.status_code == 401 and replay.json()["detail"] == "replayed"


def test_two_identical_requests_in_one_second_both_pass_under_their_own_nonces(admin):
    stamp = time.time()
    for _ in range(2):
        assert admin("GET", "/admin/v1/executors/pod-1", timestamp=stamp).status_code == 404


def test_a_request_without_a_nonce_is_refused(admin):
    stamp = str(int(time.time()))
    path = "/admin/v1/executors/pod-1"
    headers = {TIMESTAMP_HEADER: stamp,
               SIGNATURE_HEADER: sign_request(SECRET, stamp, "", "GET", path, b"")}
    response = admin.client.get(path, headers=headers)
    assert response.status_code == 401 and response.json()["detail"] == "missing_signature"


def test_a_stale_timestamp_is_refused(admin):
    response = admin("GET", "/admin/v1/executors/pod-1", timestamp=time.time() - 301)
    assert response.status_code == 401 and response.json()["detail"] == "stale_timestamp"


def test_a_request_signed_with_another_secret_is_refused(admin):
    response = admin("GET", "/admin/v1/executors/pod-1", secret=b"other")
    assert response.status_code == 401 and response.json()["detail"] == "bad_signature"


# --------------------------------------------------------------------------
# Jobs and caps
# --------------------------------------------------------------------------


def test_create_job_writes_the_manifest_and_the_task_entry(admin):
    response = admin("POST", "/admin/v1/jobs", _job(thinking=True))
    assert response.status_code == 201, response.text
    assert response.json() == {"job_id": "math-a", "task_id": "math-a", "created": True,
                               "status": "active", "cap": 0.1}
    entry = admin.registry["entries"]["math-a"]
    assert entry.mechanism == "corpus-generation" and entry.job_id == "math-a"
    assert entry.contract["model_id"] == MODEL
    manifest = json.loads(admin.bucket.objects["reliquary/corpus/jobs/math-a.json"][0])
    assert manifest["renderer_id"] == "chat-template-thinking-v1"
    assert manifest["slots_per_prompt"] == 4 and manifest["checkpoint_repo"] == MODEL


def test_create_job_is_idempotent_on_the_job_id(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    again = admin("POST", "/admin/v1/jobs", _job())
    assert again.status_code == 200 and again.json()["created"] is False
    other = admin("POST", "/admin/v1/jobs", _job(prompt_count=500))
    assert other.status_code == 409


def test_signed_catalog_and_scoped_task_contracts(admin):
    from reliquary.protocol.release_contract import canonical_sha256

    assert admin.client.get("/admin/v1/task-catalog").status_code == 401
    catalog = admin("GET", "/admin/v1/task-catalog").json()
    assert catalog["schema"] == "subnet-task-catalog/v1"
    assert catalog["zero_cap_supported"] is True
    assert catalog["models"][0] == {"model": MODEL, **MODELS[MODEL]}
    source = next(e for e in catalog["environments"] if e["environment"] == SOURCE)
    assert source["contract_sha256"] == canonical_sha256(source["contract"])
    assert source["legacy_generation_supported"] is True
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    tasks = admin("GET", "/admin/v1/tasks").json()["tasks"]
    assert [e["task_id"] for e in tasks] == ["math-a"]
    task = admin("GET", "/admin/v1/tasks/math-a").json()["task"]
    assert task["profile_sha256"] == canonical_sha256(task["contract"])
    assert admin("GET", "/admin/v1/tasks/default").status_code == 409


def test_reviewed_zero_cap_manifest_is_read_only_then_created_and_replayed(admin):
    from reliquary.protocol.release_contract import canonical_sha256

    body = _job(cap=0.0, prompt_count=2, samples_per_prompt=1, max_new_tokens=128)
    before = dict(admin.bucket.objects)
    review = admin("POST", "/admin/v1/jobs/validate", body)
    assert review.status_code == 200, review.text
    contract = review.json()
    assert admin.bucket.objects == before
    assert list(admin.registry["entries"]) == ["default"]
    assert contract["manifest_sha256"] == canonical_sha256(contract["manifest"])
    assert contract["profile_sha256"] == canonical_sha256(contract["task"]["contract"])
    pinned = {**body, "manifest_sha256": contract["manifest_sha256"],
              "profile_sha256": contract["profile_sha256"]}
    assert admin("POST", "/admin/v1/jobs", pinned).status_code == 201
    assert admin("POST", "/admin/v1/jobs", pinned).status_code == 200
    entry = admin.registry["entries"]["math-a"]
    assert entry.params["cap"] == entry.params["floor"] == 0.0
    assert entry.params["min_incentive_share"] == 0.0
    status = admin("GET", "/admin/v1/jobs/math-a/status").json()
    assert status["manifest"] == contract["manifest"]
    assert status["manifest_sha256"] == contract["manifest_sha256"]
    assert status["profile_sha256"] == contract["profile_sha256"]


def test_review_hash_changes_are_refused_before_a_write(admin):
    body = _job(manifest_sha256="0" * 64)
    before = dict(admin.bucket.objects)
    response = admin("POST", "/admin/v1/jobs", body)
    assert response.status_code == 409
    assert response.json()["detail"] == "reviewed_contract_changed"
    assert admin.bucket.objects == before
    assert list(admin.registry["entries"]) == ["default"]


def test_zero_cap_review_and_create_do_not_scan_inherited_retired_jobs(admin, monkeypatch):
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.validator import corpus_job_status

    assert admin("POST", "/admin/v1/jobs", _job("math-old", cap=0.2)).status_code == 201
    assert admin("POST", "/admin/v1/tasks/math-old/retire", {}).status_code == 200
    before = dict(admin.registry["entries"])

    async def forbidden_read(*args, **kwargs):
        raise AssertionError("zero-cap operations must not read historical job counts")

    monkeypatch.setattr(corpus_job_status, "stored_job_counts", forbidden_read)
    app = create_admin_app(secret=SECRET, pool_max=0.0, models=MODELS, records=admin.records,
                           task_prefix="math-", deliveries=LocalDirectorySink(admin.platform))
    body = _job("math-zero", cap=0.0, prompt_count=2, samples_per_prompt=1, max_new_tokens=128)
    with TestClient(app) as client:
        def signed(path, body):
            data = json.dumps(body).encode()
            stamp, nonce = str(int(time.time())), secrets.token_hex(16)
            return client.post(path, content=data, headers={
                TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, "POST", path, data),
                "content-type": "application/json"})

        review = signed("/admin/v1/jobs/validate", body)
        assert review.status_code == 200, review.text
        pins = review.json()
        created = signed("/admin/v1/jobs", {**body,
            "manifest_sha256": pins["manifest_sha256"], "profile_sha256": pins["profile_sha256"]})
        assert created.status_code == 201, created.text
        assert signed("/admin/v1/tasks/math-zero/cap", {"cap": 0.0}).status_code == 200
    assert admin.registry["entries"]["default"] == before["default"]
    assert admin.registry["entries"]["math-zero"].params["cap"] == 0.0
    assert admin.registry["entries"]["math-old"] == before["math-old"]


def test_reusing_a_job_id_cannot_silently_change_the_emission_cap(admin):
    assert admin("POST", "/admin/v1/jobs", _job(cap=0.0)).status_code == 201
    response = admin("POST", "/admin/v1/jobs", _job(cap=0.01))
    assert response.status_code == 409
    assert response.json()["detail"] == "task_exists_with_another_contract"
    assert admin.registry["entries"]["math-a"].params["cap"] == 0.0


def test_create_job_resumes_a_declaration_whose_registry_write_was_lost(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    del admin.registry["entries"]["math-a"]
    again = admin("POST", "/admin/v1/jobs", _job())
    assert again.status_code == 201 and "math-a" in admin.registry["entries"]


def test_an_unqualified_model_or_episode_env_is_refused(admin):
    assert admin("POST", "/admin/v1/jobs", _job(model="org/Unknown")).status_code == 422
    response = admin("POST", "/admin/v1/jobs", _job(env="reliquary_stateful_tools_v1"))
    assert response.status_code == 422


def test_the_corpus_pool_limit_holds_and_keeps_the_manifest(admin):
    assert admin("POST", "/admin/v1/jobs", _job("math-a", cap=0.2)).status_code == 201
    over = admin("POST", "/admin/v1/jobs", _job("math-b", cap=0.15))
    assert over.status_code == 409 and "admin pool" in over.json()["detail"]
    assert "math-b" not in admin.registry["entries"]
    # Never deleted: a racing call's task may name it; an orphan is harmless.
    assert "reliquary/corpus/jobs/math-b.json" in admin.bucket.objects
    status = admin("GET", "/admin/v1/jobs/math-b/status").json()
    assert len(status["manifest_sha256"]) == 64
    assert status["profile_sha256"] is None
    assert status["tasks"] == status["task_contracts"] == []


def test_the_sum_of_active_caps_stays_within_one(admin):
    admin.registry["entries"]["logic"] = _rl_entry("logic", 0.45)
    over = admin("POST", "/admin/v1/jobs", _job("math-a", cap=0.1))
    assert over.status_code == 409


def test_set_cap_honours_the_limits_and_lowering_always_passes(admin):
    assert admin("POST", "/admin/v1/jobs", _job("math-a", cap=0.2)).status_code == 201
    assert admin("POST", "/admin/v1/tasks/math-a/cap", {"cap": 0.35}).status_code == 409
    response = admin("POST", "/admin/v1/tasks/math-a/cap", {"cap": 0.25})
    assert response.status_code == 200
    assert admin.registry["entries"]["math-a"].params["cap"] == 0.25
    assert admin.registry["entries"]["math-a"].params["floor"] == 0.25


def test_retire_stamps_the_current_round_and_is_idempotent(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    response = admin("POST", "/admin/v1/tasks/math-a/retire", {})
    assert response.json() == {"task_id": "math-a", "status": "retired", "retired_at": 777}
    assert admin.registry["entries"]["math-a"].status == "retired"
    assert admin("POST", "/admin/v1/tasks/math-a/retire", {"retired_at": 9}).json()["retired_at"] == 777
    assert admin("POST", "/admin/v1/tasks/math-nope/retire", {}).status_code == 404


def test_job_status_proxies_the_stored_counts_and_the_manifest(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    admin.records.subs = {"s1": {}, "s2": {}}
    admin.records.verdicts = {"s1": {"passed": True}}
    admin.records.settlement = {"settled": ["s1"], "pending": None, "last_window": 4}
    status = admin("GET", "/admin/v1/jobs/math-a/status").json()
    assert (status["submissions"], status["verdicts"], status["unaudited"], status["settled"],
            status["drained"], status["last_window"]) == (2, 1, 1, 1, False, 4)
    assert status["manifest"]["job_id"] == "math-a"
    assert status["tasks"] == [{"task_id": "math-a", "status": "active", "cap": 0.1}]
    assert admin("GET", "/admin/v1/jobs/math-ghost/status").status_code == 404


# --------------------------------------------------------------------------
# Executors
# --------------------------------------------------------------------------


def _executor(**kw):
    return {"executor_id": "pod-1", "token_sha256": hashlib.sha256(b"tok").hexdigest(),
            "model_id": MODEL, "model_revision": "r1", "expires_at": time.time() + 3600, **kw}


def test_register_read_and_revoke_an_executor(admin):
    created = admin("POST", "/admin/v1/executors", _executor())
    assert created.status_code == 201 and "token_sha256" not in created.json()
    again = admin("POST", "/admin/v1/executors", _executor(expires_at=created.json()["expires_at"]))
    assert again.status_code == 200
    asyncio.run(executors.record_heartbeat("pod-1", at=1234.0))
    seen = admin("GET", "/admin/v1/executors/pod-1").json()
    assert seen["status"] == "active" and seen["last_heartbeat"] == 1234.0
    revoked = admin("DELETE", "/admin/v1/executors/pod-1")
    assert revoked.status_code == 200 and revoked.json()["status"] == "revoked"
    assert admin("DELETE", "/admin/v1/executors/ghost").status_code == 404
    assert admin("POST", "/admin/v1/executors", _executor(executor_id="../x")).status_code == 422


# --------------------------------------------------------------------------
# Deliveries
# --------------------------------------------------------------------------


def test_a_delivery_runs_beside_the_request_and_returns_its_keys(admin, monkeypatch):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    from reliquary.validator import corpus_service

    class _Environment:
        def get_problem(self, index):
            return {"prompt": f"canonical question {index}?"}

    monkeypatch.setattr(corpus_service, "prompt_job_for_spec",
                        lambda job: corpus_service.SingleTurnPromptJob(job, _Environment()))
    sid = "1" * 64
    admin.records.verdicts = {sid: {"passed": True}}
    admin.records.subs = {sid: {"prompt_index": 3, "rendered_prompt": "q",
                                "completions": [{"text": "a", "tokens": [1]}]}}
    admin.records.settlement = {"settled": [sid], "pending": None}
    first = admin("POST", "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-1"})
    assert first.status_code == 202 and first.json()["state"] == "running"
    done = None
    for _ in range(100):
        response = admin("POST", "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-1"})
        if response.status_code == 200:
            done = response.json()
            break
        time.sleep(0.02)
    assert done is not None and done["state"] == "done" and done["rows"] == 1
    assert "deliveries/order-1/manifest.json" in done["keys"]
    assert (admin.platform / "deliveries" / "order-1" / "report.json").exists()
    assert "deliveries/order-1/instruction-00000.jsonl" in done["keys"]
    instruction = admin.platform / "deliveries" / "order-1" / "instruction-00000.jsonl"
    assert json.loads(instruction.read_text()) == {"prompt": "canonical question 3?",
                                                 "response": "a"}
    # Done stays done, from the bucket.
    again = admin("POST", "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-1"})
    assert again.status_code == 200 and again.json()["keys"] == done["keys"]


def test_a_delivery_of_an_unknown_job_is_404(admin):
    assert admin("POST", "/admin/v1/jobs/math-ghost/deliveries", {}).status_code == 404


def test_a_stored_delivery_cannot_be_returned_for_another_job(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    root = admin.platform / "deliveries" / "order-existing"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"job_id": "math-other", "keys": [],
                                                   "rows": 0}))
    response = admin("POST", "/admin/v1/jobs/math-a/deliveries",
                     {"delivery_id": "order-existing"})
    assert response.status_code == 409
    assert response.json()["detail"] == "delivery_belongs_to_another_job"


def test_cached_delivery_retry_checks_the_immutable_job_contract(admin, monkeypatch):
    from reliquary.corpus.delivery import LocalDirectorySink, export_delivery
    from reliquary.corpus.job import parse_job
    from reliquary.protocol.release_contract import canonical_sha256

    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    job = parse_job(admin("GET", "/admin/v1/jobs/math-a/status").json()["manifest"])
    pin = canonical_sha256(job.to_contract())
    root = admin.platform / "deliveries" / "order-existing"
    root.mkdir(parents=True)
    sink = LocalDirectorySink(admin.platform)
    for required, fields, accepted in (
        (False, {"job_manifest_sha256": pin}, True),
        (False, {"job_manifest_sha256": "b" * 64}, False),
        (False, {}, True),  # Historical storage exports predate the pin.
        (True, {"job_manifest_sha256": pin}, True),
        (True, {}, False),  # HTTP delivery recovery always requires it.
    ):
        monkeypatch.setattr(LocalDirectorySink, "requires_job_contract_pin", required,
                            raising=False)
        manifest = {"job_id": job.job_id, "keys": [], "rows": 1, **fields}
        (root / "manifest.json").write_text(json.dumps(manifest))
        response = admin("POST", "/admin/v1/jobs/math-a/deliveries",
                         {"delivery_id": "order-existing"})
        retry = export_delivery(job=job, records=None, sink=sink,
                                delivery_id="order-existing")
        if accepted:
            assert response.status_code == 200 and response.json()["state"] == "done"
            assert asyncio.run(retry) == manifest
        else:
            assert response.status_code == 409
            assert response.json()["detail"] == "delivery_belongs_to_another_contract"
            with pytest.raises(ValueError, match="another job contract"):
                asyncio.run(retry)


def test_a_bounded_delivery_sink_refuses_unsupported_namespaces_and_evaluation(admin, monkeypatch):
    from reliquary.corpus.delivery import LocalDirectorySink

    monkeypatch.setattr(LocalDirectorySink, "accepts_delivery_id", lambda self, value: False,
                        raising=False)
    monkeypatch.setattr(LocalDirectorySink, "evaluation_supported", False, raising=False)
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    refused = admin("POST", "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-other"})
    assert refused.status_code == 503
    assert refused.json()["detail"] == "delivery_namespace_not_configured"
    grade = admin("POST", "/admin/v1/evaluations/math-a/grade",
                  {"source": "job", "job_id": "math-a", "set_ids": ["math"],
                   "problems_per_set": {"math": 1}, "samples_per_set": {"math": 1}})
    assert grade.status_code == 503
    assert admin("GET", "/admin/v1/evaluations/math-a/files/report.json").status_code == 503


def test_a_missing_raw_source_preserves_the_original_admin_delivery(admin, monkeypatch):
    from reliquary.validator import corpus_service

    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201

    def missing(job):
        raise corpus_service.CorpusPromptSourceError("fixture source not installed")

    monkeypatch.setattr(corpus_service, "prompt_job_for_spec", missing)
    sid = "1" * 64
    admin.records.verdicts = {sid: {"passed": True}}
    admin.records.subs = {sid: {"prompt_index": 3, "rendered_prompt": "q",
                                "completions": [{"text": "a", "tokens": [1]}]}}
    admin.records.settlement = {"settled": [sid], "pending": None}
    path, body = "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-raw-missing"}
    assert admin("POST", path, body).status_code == 202
    for _ in range(100):
        response = admin("POST", path, body)
        if response.status_code == 200:
            break
        time.sleep(0.02)
    assert response.status_code == 200 and response.json()["rows"] == 1
    root = admin.platform / "deliveries" / "order-raw-missing"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["shards"] and manifest["instruction_shards"] == []
    assert manifest["instruction"]["omitted"] == {"prompt_source_unavailable": 1}


def test_a_bad_pool_is_refused_at_build():
    with pytest.raises(ValueError):
        create_admin_app(secret=SECRET, pool_max=1.5, models={})


def test_admin_serve_refuses_to_start_unconfigured_and_serves_once_configured(
    tmp_path, monkeypatch,
):
    import uvicorn
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    for name in ("RELIQUARY_ADMIN_SECRET", "RELIQUARY_ADMIN_POOL_MAX", "RELIQUARY_ADMIN_MODELS",
                 "RELIQUARY_PLATFORM_BUCKET", "RELIQUARY_PLATFORM_DELIVERY_URL",
                 "RELIQUARY_PLATFORM_DELIVERY_SECRET"):
        monkeypatch.delenv(name, raising=False)
    served = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kw: served.append((application, kw)))
    result = CliRunner().invoke(app, ["admin", "serve"])
    assert result.exit_code == 1 and "RELIQUARY_ADMIN_SECRET" in result.output
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    monkeypatch.setenv("RELIQUARY_ADMIN_SECRET", "x" * 32)
    monkeypatch.setenv("RELIQUARY_ADMIN_MODELS", str(models))
    result = CliRunner().invoke(app, ["admin", "serve"])
    assert result.exit_code == 1 and "RELIQUARY_ADMIN_POOL_MAX" in result.output
    monkeypatch.setenv("RELIQUARY_ADMIN_POOL_MAX", "0.3")
    result = CliRunner().invoke(app, ["admin", "serve", "--port", "9999"])
    assert result.exit_code == 0, result.output
    assert served and served[0][1]["port"] == 9999


def test_admin_environment_uses_the_bounded_http_delivery_sink(tmp_path, monkeypatch):
    from reliquary.cli import main
    from reliquary.corpus.delivery import HTTPDeliverySink

    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    monkeypatch.setenv("RELIQUARY_ADMIN_SECRET", "s" * 32)
    monkeypatch.setenv("RELIQUARY_ADMIN_MODELS", str(models))
    monkeypatch.setenv("RELIQUARY_ADMIN_POOL_MAX", "0")
    monkeypatch.setenv("RELIQUARY_PLATFORM_DELIVERY_URL", "https://example.test")
    monkeypatch.setenv("RELIQUARY_PLATFORM_DELIVERY_SECRET", "s" * 32)
    monkeypatch.setattr("reliquary.admin.service.create_admin_app", lambda **kwargs: kwargs)
    configured = main.build_admin_app_from_environment()
    assert isinstance(configured["deliveries"], HTTPDeliverySink)
    assert configured["deliveries"].max_file_bytes == 64 * 1024 * 1024
    monkeypatch.delenv("RELIQUARY_PLATFORM_DELIVERY_SECRET")
    with pytest.raises(ValueError, match="secret"):
        main.build_admin_app_from_environment()



# --------------------------------------------------------------------------
# Review fixes: I5, I6, I7 and the admin minors
# --------------------------------------------------------------------------


def test_a_create_losing_the_registry_race_answers_the_idempotent_200(admin, monkeypatch):
    """I5: the other call's task now names the job; the manifest stays."""
    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    real = store.create_task

    async def raced(entry, **kw):
        await real(entry, **kw)  # the concurrent identical call lands first
        raise RegistryError(f"task {entry.task_id!r} already exists")

    monkeypatch.setattr(store, "create_task", raced)
    response = admin("POST", "/admin/v1/jobs", _job())
    assert response.status_code == 200 and response.json()["created"] is False
    assert "reliquary/corpus/jobs/math-a.json" in admin.bucket.objects


def test_a_create_whose_manifest_write_races_an_identical_one_proceeds(admin, monkeypatch):
    """M6: the create-only manifest write lost to the same bytes is not a 409."""
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    del admin.registry["entries"]["math-a"]
    real = job_store.read_job
    calls = []

    async def stale(job_id, **kw):
        calls.append(job_id)
        if len(calls) == 1:
            return None, None  # read before the other call's write landed
        return await real(job_id, **kw)

    monkeypatch.setattr(job_store, "read_job", stale)
    response = admin("POST", "/admin/v1/jobs", _job())
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("task_id", ["math-rl"])
def test_cap_and_retire_refuse_a_task_that_is_not_a_corpus_task(admin, task_id):
    """I6: the platform reaches its own corpus jobs only."""
    admin.registry["entries"]["math-rl"] = _rl_entry("math-rl", 0.1)
    for path, body in ((f"/admin/v1/tasks/{task_id}/cap", {"cap": 0.0}),
                       (f"/admin/v1/tasks/{task_id}/retire", {})):
        response = admin("POST", path, body)
        assert response.status_code == 409 and response.json()["detail"] == "not_a_corpus_task"
    assert admin.registry["entries"][task_id].status == "active"


def test_a_retired_job_still_draining_holds_its_share_of_the_pool(admin):
    """I7: its cap is still paid until it drains."""
    assert admin("POST", "/admin/v1/jobs", _job("math-a", cap=0.2)).status_code == 201
    assert admin("POST", "/admin/v1/tasks/math-a/retire", {}).status_code == 200
    admin.records.subs = {"s1": {}}  # one submission still unaudited
    assert admin("POST", "/admin/v1/jobs", _job("math-b", cap=0.15)).status_code == 409
    admin.records.subs = {}
    assert admin("POST", "/admin/v1/jobs", _job("math-b", cap=0.15)).status_code == 201


def test_a_second_task_for_one_job_is_refused(admin):
    """M7."""
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    response = admin("POST", "/admin/v1/jobs", _job(task_id="math-a-again"))
    assert response.status_code == 409 and "math-a-again" not in admin.registry["entries"]


def test_a_delivery_of_a_job_not_yet_drained_is_refused(admin):
    """M8: a premature export would freeze a partial dataset under its id."""
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
    admin.records.subs = {"s1": {}}
    response = admin("POST", "/admin/v1/jobs/math-a/deliveries", {})
    assert response.status_code == 409 and response.json()["detail"] == "job_not_drained"


def test_a_retire_racing_another_answers_the_stored_stamp(admin, monkeypatch):
    """M9."""
    from dataclasses import replace

    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201

    async def raced(task_id, retired_at, **kw):
        entries = admin.registry["entries"]
        entries[task_id] = replace(entries[task_id], status="retired", retired_at=555)
        raise RegistryError("lost the race")

    monkeypatch.setattr(store, "retire_task_entry", raced)
    response = admin("POST", "/admin/v1/tasks/math-a/retire", {})
    assert response.status_code == 200 and response.json()["retired_at"] == 555


def test_a_non_ascii_signature_is_a_401_not_a_500(admin):
    """M4."""
    stamp = str(int(time.time()))
    headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: "ab" * 16,
               SIGNATURE_HEADER: ("é" * 64).encode("latin-1")}
    response = admin.client.get("/admin/v1/executors/pod-1", headers=headers)
    assert response.status_code == 401


def test_an_oversized_body_is_refused_before_it_is_read(admin):
    """M5."""
    response = admin("POST", "/admin/v1/jobs", raw=b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413



# --------------------------------------------------------------------------
# I6 residual: the admin scope is a task-id prefix
# --------------------------------------------------------------------------


def test_a_job_or_task_outside_the_admin_prefix_cannot_be_created(admin):
    for body in (_job("order-a"), _job("math-a", task_id="corpus-a")):
        response = admin("POST", "/admin/v1/jobs", body)
        assert response.status_code == 422
        assert response.json()["detail"] == "task_id_outside_admin_scope"
    assert set(admin.registry["entries"]) == {"default"}


@pytest.mark.parametrize("task_id", ["default", "logic", "corpus-code-v1"])
def test_operator_tasks_and_jobs_are_outside_the_admin_scope(admin, task_id):
    from dataclasses import replace

    admin.registry["entries"]["logic"] = _rl_entry("logic", 0.1)
    for method, path, body in (("POST", f"/admin/v1/tasks/{task_id}/cap", {"cap": 0.0}),
                               ("POST", f"/admin/v1/tasks/{task_id}/retire", {}),
                               ("GET", f"/admin/v1/jobs/{task_id}/status", None),
                               ("POST", f"/admin/v1/jobs/{task_id}/deliveries", {})):
        response = admin(method, path, body)
        assert response.status_code == 409, (path, response.text)
        assert response.json()["detail"] == "outside_admin_scope"
    assert admin.registry["entries"]["default"].status == "active"


def test_the_prefix_defaults_to_order_and_comes_from_the_environment(monkeypatch, tmp_path):
    from reliquary.cli.main import build_admin_app_from_environment

    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    monkeypatch.setenv("RELIQUARY_ADMIN_SECRET", "x" * 32)
    monkeypatch.setenv("RELIQUARY_ADMIN_MODELS", str(models))
    monkeypatch.setenv("RELIQUARY_ADMIN_POOL_MAX", "0.3")
    monkeypatch.delenv("RELIQUARY_PLATFORM_BUCKET", raising=False)
    monkeypatch.delenv("RELIQUARY_ADMIN_TASK_PREFIX", raising=False)
    assert build_admin_app_from_environment().state.task_prefix == "order-"
    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "ds-")
    assert build_admin_app_from_environment().state.task_prefix == "ds-"
    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "")
    with pytest.raises(ValueError):
        build_admin_app_from_environment()


# sha256 of the canonical (manifest, task contract, params) of `_job(thinking=True)`,
# pinned on 7753e5e4 before dataset orders on any model: an operator catalog job
# must stay byte-identical (tests/unit/test_order_settlement_identity.py).
CATALOG_JOB_GOLDEN = "821713d9f8458aaef010f4d70af62c075a39dffa8236afc18120f86ea086caf2"


def test_an_operator_catalog_job_is_declared_byte_identically(admin):
    assert admin("POST", "/admin/v1/jobs", _job(thinking=True)).status_code == 201
    entry = admin.registry["entries"]["math-a"]
    manifest = json.loads(admin.bucket.objects["reliquary/corpus/jobs/math-a.json"][0])
    document = {"manifest": manifest, "contract": entry.contract, "params": entry.params}
    digest = hashlib.sha256(json.dumps(document, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    assert digest == CATALOG_JOB_GOLDEN
