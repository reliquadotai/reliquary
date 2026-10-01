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
from tests.unit.test_eval_grading import GradingEnvironment, _fixture

SECRET = b"admin-secret"


@pytest.fixture
def admin(tmp_path):
    app = create_admin_app(
        secret=SECRET, pool_max=0.3, models={}, records=object(), task_prefix="order-",
        deliveries=LocalDirectorySink(tmp_path / "platform"),
        eval_store=LocalDirectorySink(tmp_path / "subnet"),
        open_environment=lambda source, split: GradingEnvironment(source),
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
    request = _fixture(admin.root)
    path = "/admin/v1/evaluations/order-e1/grade"
    first = admin("POST", path, {**request, "provenance": {"model": "org/m"}})
    assert first.status_code == 202
    assert first.json() == {"state": "running", "eval_id": "order-e1"}
    done = _until_done(admin, path, {**request, "provenance": {"model": "org/m"}})
    assert done.status_code == 200, done.text
    assert done.json() == {"state": "done", "eval_id": "order-e1", "rows": 14, "keys": [
        "deliveries/order-e1/graded.parquet", "deliveries/order-e1/report.json",
        "deliveries/order-e1/manifest.json"]}
    # Idempotent: the stored manifest answers, and another request is refused.
    assert admin("POST", path, request).json()["state"] == "done"
    other = {**request, "problems_per_set": {k: 1 for k in request["problems_per_set"]}}
    conflict = admin("POST", path, other)
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "grade_exists_with_another_request"


def test_an_unknown_set_is_404(admin):
    request = _fixture(admin.root)
    request["set_ids"][0] = "logic-eval-s9-n9"
    request["problems_per_set"] = {s: 1 for s in request["set_ids"]}
    response = admin("POST", "/admin/v1/evaluations/order-e1/grade", request)
    assert response.status_code == 404 and response.json()["detail"] == "set_unknown"


def test_scope_signature_and_validation(admin):
    request = _fixture(admin.root)
    assert admin("POST", "/admin/v1/evaluations/math-e1/grade", request).status_code == 409
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", request,
                 secret=b"wrong").status_code == 401
    bad = {**request, "completion_keys": ["../etc/passwd"]}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", bad).status_code == 422
    too_many = {**request, "problems_per_set": {k: 99 for k in request["problems_per_set"]}}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", too_many).status_code == 422
    extra = {**request, "surprise": 1}
    assert admin("POST", "/admin/v1/evaluations/order-e1/grade", extra).status_code == 422


def test_grading_needs_the_platform_bucket(tmp_path):
    app = create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=object(),
                           eval_store=LocalDirectorySink(tmp_path))
    client = TestClient(app)
    body = json.dumps({"set_ids": ["a"], "completion_keys": ["k"],
                       "problems_per_set": {"a": 1}}).encode()
    stamp, nonce = str(int(time.time())), secrets.token_hex(16)
    path = "/admin/v1/evaluations/order-e1/grade"
    response = client.post(path, content=body, headers={
        TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
        SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, "POST", path, body),
        "content-type": "application/json"})
    assert response.status_code == 503
