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


def test_a_delivery_runs_beside_the_request_and_returns_its_keys(admin):
    assert admin("POST", "/admin/v1/jobs", _job()).status_code == 201
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
    # Done stays done, from the bucket.
    again = admin("POST", "/admin/v1/jobs/math-a/deliveries", {"delivery_id": "order-1"})
    assert again.status_code == 200 and again.json()["keys"] == done["keys"]


def test_a_delivery_of_an_unknown_job_is_404(admin):
    assert admin("POST", "/admin/v1/jobs/math-ghost/deliveries", {}).status_code == 404


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
                 "RELIQUARY_PLATFORM_BUCKET"):
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
