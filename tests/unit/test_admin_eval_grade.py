"""`POST /admin/v1/evaluations/{eval_id}/grade`: signed, scoped, idempotent."""

from __future__ import annotations

import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.corpus.delivery import LocalDirectorySink
from tests.unit.test_eval_grading import GradingEnvironment, _fixture, _plain_scorer

PROVENANCE = {"model": "org/m", "revision": "a" * 40, "model_sha": "a" * 40,
              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7},
              "thinking": False, "max_new_tokens": 64, "vllm_version": "0.11",
              "gpu": "1x H100", "pod_provider_id": "lium-7"}


def _request(root):
    return {**_fixture(root), "provenance": PROVENANCE}

SECRET = b"admin-secret"


@pytest.fixture
def admin(tmp_path):
    app = create_admin_app(
        secret=SECRET, pool_max=0.3, models={}, records=object(), task_prefix="order-",
        deliveries=LocalDirectorySink(tmp_path / "platform"),
        eval_store=LocalDirectorySink(tmp_path / "subnet"),
        open_environment=lambda source, split: GradingEnvironment(source),
        grade_scorer=_plain_scorer, require_sandbox=lambda spec: None,
        work_dir=tmp_path / "work")
    client = TestClient(app)
    client.__enter__()

    def call(method, path, body=None, *, secret=SECRET):
        data = b"" if body is None else json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(secret, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return client.request(method, path, content=data, headers=headers)

    call.app, call.root = app, tmp_path
    yield call
    client.__exit__(None, None, None)


def _until_done(admin, path, body):
    for _ in range(200):
        response = admin("POST", path, body)
        if response.status_code != 202:
            return response
        time.sleep(0.02)
    raise AssertionError("grading never finished")


def test_grade_runs_beside_the_request_then_answers_its_keys(admin):
    request = _request(admin.root)
    path = "/admin/v1/evaluations/order-e1/grade"
    first = admin("POST", path, request)
    assert first.status_code == 202
    assert first.json() == {"state": "running", "eval_id": "order-e1"}
    done = _until_done(admin, path, request)
    assert done.status_code == 200, done.text
    assert done.json() == {"state": "done", "eval_id": "order-e1", "rows": 13,
                           "complete": False, "keys": [
        "evaluations/order-e1/graded.parquet", "evaluations/order-e1/report.json",
        "evaluations/order-e1/manifest.json"]}
    # Idempotent: the stored manifest answers, and another request is refused.
    assert admin("POST", path, request).json()["state"] == "done"
    other = {**request, "problems_per_set": {k: 1 for k in request["problems_per_set"]}}
    conflict = admin("POST", path, other)
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "grade_exists_with_another_request"


def test_an_unknown_set_is_404(admin):
    request = _request(admin.root)
    request["set_ids"][0] = "logic-eval-s9-n9"
    request["problems_per_set"] = {s: 1 for s in request["set_ids"]}
    request["samples_per_set"] = {s: 1 for s in request["set_ids"]}
    response = admin("POST", "/admin/v1/evaluations/order-e1/grade", request)
    assert response.status_code == 404 and response.json()["detail"] == "set_unknown"


def test_scope_signature_and_validation(admin):
    request = _request(admin.root)
    assert admin("POST", "/admin/v1/evaluations/math-e1/grade", request).status_code == 409
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", request,
                 secret=b"wrong").status_code == 401
    bad = {**request, "completion_keys": ["../etc/passwd"]}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", bad).status_code == 422
    too_many = {**request, "problems_per_set": {k: 99 for k in request["problems_per_set"]}}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", too_many).status_code == 422
    extra = {**request, "surprise": 1}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", extra).status_code == 422
    bare = {**request, "provenance": {"model": "org/m"}}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", bare).status_code == 422
    no_samples = {k: v for k, v in request.items() if k != "samples_per_set"}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", no_samples).status_code == 422


def test_two_concurrent_first_calls_start_one_grading(admin, monkeypatch):
    import asyncio

    import httpx

    from reliquary.eval import grading

    request = _request(admin.root)
    started = []
    original_load = grading.load_sets

    async def slow_load(*args, **kwargs):
        await asyncio.sleep(0.05)
        return await original_load(*args, **kwargs)

    async def fake_grade(**kwargs):
        started.append(kwargs["eval_id"])
        await asyncio.sleep(0.05)
        return {"keys": [], "rows": 0, "complete": True, "request_sha256": None}

    monkeypatch.setattr(grading, "load_sets", slow_load)
    monkeypatch.setattr(grading, "grade_evaluation", fake_grade)
    path = "/admin/v1/evaluations/order-e1/grade"
    data = json.dumps(request).encode()

    def headers():
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        return {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, "POST", path, data),
                "content-type": "application/json"}

    async def both():
        transport = httpx.ASGITransport(app=admin.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://admin") as client:
            return await asyncio.gather(client.post(path, content=data, headers=headers()),
                                        client.post(path, content=data, headers=headers()))

    answers = asyncio.run(both())
    assert [a.status_code for a in answers] == [202, 202]
    assert started == ["order-e1"]


def test_a_code_order_without_the_sandbox_is_refused(tmp_path, monkeypatch):
    from reliquary.corpus import export

    monkeypatch.setattr(export, "GRADER_SOCKET_PATH", str(tmp_path / "absent.sock"))
    app = create_admin_app(
        secret=SECRET, pool_max=0.3, models={}, records=object(), task_prefix="order-",
        deliveries=LocalDirectorySink(tmp_path / "platform"),
        eval_store=LocalDirectorySink(tmp_path / "subnet"),
        open_environment=lambda source, split: GradingEnvironment(source))
    client = TestClient(app)
    path = "/admin/v1/evaluations/order-e1/grade"
    data = json.dumps(_request(tmp_path)).encode()
    stamp, nonce = str(int(time.time())), secrets.token_hex(16)
    response = client.post(path, content=data, headers={
        TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
        SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, "POST", path, data),
        "content-type": "application/json"})
    assert response.status_code == 503
    assert response.json()["detail"] == "code_sandbox_unavailable"


def test_grading_needs_the_platform_bucket(tmp_path):
    app = create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=object(),
                           eval_store=LocalDirectorySink(tmp_path))
    client = TestClient(app)
    body = json.dumps({"set_ids": ["a"], "completion_keys": ["k"],
                       "problems_per_set": {"a": 1}, "samples_per_set": {"a": 1},
                       "provenance": PROVENANCE}).encode()
    stamp, nonce = str(int(time.time())), secrets.token_hex(16)
    path = "/admin/v1/evaluations/order-e1/grade"
    response = client.post(path, content=body, headers={
        TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
        SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, "POST", path, body),
        "content-type": "application/json"})
    assert response.status_code == 503
