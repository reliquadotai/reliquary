"""Per-pod trainer auth: a trainer executor's token opens its own task's trainer link."""
from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.infrastructure import corpus_executor_store as executors
from reliquary.validator.rl_async_service import build_rl_trainer_router, executor_trainer_auth
from tests.unit import test_admin_training as admin_training
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_admin_token_ceilings import rates  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_rl_async_service import TOKEN, components

POD = "pod-token-" + "a" * 20
OTHER = "pod-token-" + "b" * 20


class _Store:
    scope_of = staticmethod(executors.scope_of)

    def __init__(self, *documents):
        self.documents = list(documents)

    async def list_executors(self):
        return list(self.documents)


def _doc(token, *, task="task-a", status="active", expires=1000.0, scope="trainer"):
    return {"executor_id": "e-" + token[-3:], "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "scope": scope, "task_id": task, "status": status, "expires_at": expires}


def _client(tmp_path, *, auth=None, token=None):
    rl = components(tmp_path)
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=token, auth=auth))
    return TestClient(app), rl


def _status(client, token):
    return client.get("/rl/trainer/config", headers={"Authorization": f"Bearer {token}"}).status_code


def test_a_pod_token_opens_its_tasks_trainer_link(tmp_path):
    auth = executor_trainer_auth("task-a", store=_Store(_doc(POD)), clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth)
    assert _status(client, POD) == 200
    assert client.get("/rl/trainer/config").status_code == 401
    rl.close()


def test_a_pod_token_of_another_task_is_refused(tmp_path):
    auth = executor_trainer_auth("task-a", store=_Store(_doc(POD, task="task-b"), _doc(OTHER, scope="eval")),
                                 clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth)
    assert _status(client, POD) == 401 and _status(client, OTHER) == 401
    rl.close()


def test_a_retired_or_expired_trainer_executor_is_refused(tmp_path):
    now = [10.0]
    store = _Store(_doc(POD, status="revoked"), _doc(OTHER, expires=50.0))
    auth = executor_trainer_auth("task-a", store=store, clock=lambda: now[0])
    client, rl = _client(tmp_path, auth=auth)
    assert _status(client, POD) == 401
    assert _status(client, OTHER) == 200
    now[0] = 60.0                                   # expired, and the cache has aged out
    assert _status(client, OTHER) == 401
    rl.close()


def test_the_static_token_still_opens_the_link(tmp_path):
    auth = executor_trainer_auth("task-a", store=_Store(), clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth, token=TOKEN)
    assert _status(client, TOKEN) == 200 and _status(client, "x" * 32) == 401
    (tmp_path / "b").mkdir()
    only_static, rl2 = _client(tmp_path / "b", token=TOKEN)
    assert _status(only_static, TOKEN) == 200
    with pytest.raises(ValueError):
        build_rl_trainer_router(rl)
    rl.close()
    rl2.close()


def _trainer(**over):
    return {"executor_id": "pod-1", "token_sha256": hashlib.sha256(b"tok").hexdigest(),
            "model_id": "m", "model_revision": "r", "expires_at": 4e12, "scope": "trainer", **over}


def test_admin_registers_a_trainer_executor_with_its_task(request, monkeypatch, rates):  # noqa: F811
    monkeypatch.setenv(admin_training.training.TRAINING_ORDERS_ENV, "on")
    client = request.getfixturevalue("admin")
    assert client("POST", "/admin/v1/training", admin_training._order()).status_code == 201
    assert client("POST", "/admin/v1/executors", _trainer()).status_code == 422
    assert client("POST", "/admin/v1/executors", _trainer(task_id="default")).status_code == 422
    created = client("POST", "/admin/v1/executors", _trainer(task_id="order-train-t1"))
    assert created.status_code == 201 and created.json()["task_id"] == "order-train-t1"
    stored = asyncio.run(executors.read_executor("pod-1"))
    assert stored["scope"] == "trainer" and stored["task_id"] == "order-train-t1"


# --- heartbeats: the platform's readiness check reads the executor record's last_heartbeat ----------------------
class _BeatStore(_Store):
    def __init__(self, *documents, fail=False):
        super().__init__(*documents)
        self.beats = []
        self.fail = fail

    async def record_heartbeat(self, executor_id, *, at, detail=None):
        if self.fail:
            raise RuntimeError("store down")
        self.beats.append((executor_id, at, detail))


def _settle(client):
    """Let the fire-and-forget beat tasks run (the TestClient's loop is gone after each call)."""
    import time
    time.sleep(0.05)


def test_a_pod_token_request_records_a_heartbeat(tmp_path):
    store = _BeatStore(_doc(POD))
    auth = executor_trainer_auth("task-a", store=store, clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth)
    assert _status(client, POD) == 200
    _settle(client)
    assert [(e, a) for e, a, _ in store.beats] == [("e-" + POD[-3:], 10.0)]
    rl.close()


def test_two_requests_within_a_minute_record_one_heartbeat(tmp_path):
    now = [10.0]
    store = _BeatStore(_doc(POD))
    auth = executor_trainer_auth("task-a", store=store, clock=lambda: now[0])
    client, rl = _client(tmp_path, auth=auth)
    _status(client, POD)
    now[0] = 40.0
    _status(client, POD)
    _settle(client)
    assert len(store.beats) == 1
    now[0] = 71.0
    _status(client, POD)
    _settle(client)
    assert len(store.beats) == 2
    rl.close()


def test_a_static_token_request_records_no_heartbeat(tmp_path):
    store = _BeatStore(_doc(POD))
    auth = executor_trainer_auth("task-a", store=store, clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth, token=TOKEN)
    assert _status(client, TOKEN) == 200
    _settle(client)
    assert store.beats == []
    rl.close()


def test_a_heartbeat_store_failure_does_not_fail_the_request(tmp_path):
    store = _BeatStore(_doc(POD), fail=True)
    auth = executor_trainer_auth("task-a", store=store, clock=lambda: 10.0)
    client, rl = _client(tmp_path, auth=auth)
    assert _status(client, POD) == 200
    _settle(client)
    rl.close()
