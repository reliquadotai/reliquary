"""Evaluation grants stay inside the signed administrator corridor."""
from __future__ import annotations

import hashlib
import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.sandbox.evaluation_engagements import EvaluationSessionRequest, from_environment
from reliquary.eval.signed_agentic import checkpoint_identity

SECRET = b"evaluation-administrator-test"
TASK_ID = "f2603b85-0ba4-43fc-85a7-5b6cb43b78b2"
EVAL_ID = "order-eval-" + TASK_ID
PATH = f"/admin/v1/evaluations/{EVAL_ID}/sandbox-session"


def request_body():
    context = [TASK_ID, "f" * 64, "heldout-1", "problem-1", 0]
    return {"request_id": hashlib.sha256(json.dumps(context, separators=(",", ":")).encode()).hexdigest(),
            "eval_id": EVAL_ID, "task_id": TASK_ID, "set_id": "heldout-1",
            "problem_id": "problem-1", "sample_index": 0, "lease_fence": "f" * 64,
            "expires_at": int(time.time()) + 1800, "worker_id": "pod-1", "authority_kind": "leased_worker",
            "model": {"repo": "owner/model", "revision": "a" * 40}, "model_access": "public",
            "selection": {"env": "terminal", "problems": 1, "samples": 1},
            "protocol": {}, "protocol_json": "{}", "protocol_sha256": "b" * 64,
            "protocol_bindings": {"heldout-1": "c" * 64}, "set_protocol": {}}


class Sessions:
    def __init__(self):
        self.started, self.stopped, self.requests = 0, 0, []

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    async def open(self, request):
        self.requests.append(request)
        return {"grant": {"session_id": "s-1", "token": {"signature": "opaque"},
                           "gateway_url": "https://gateway.example", "expires_at": request.expires_at},
                "binding": {"worker_id": request.worker_id}}


def call(client, body, *, secret=SECRET, path=PATH, signed=True):
    raw = json.dumps(body).encode()
    timestamp, nonce = str(int(time.time())), secrets.token_hex(16)
    headers = {"Content-Type": "application/json"}
    if signed:
        headers.update({TIMESTAMP_HEADER: timestamp, NONCE_HEADER: nonce,
                        SIGNATURE_HEADER: sign_request(secret, timestamp, nonce, "POST", path, raw)})
    return client.post(path, content=raw, headers=headers)


def test_signed_route_checks_identity_and_never_exposes_an_unsigned_issuer():
    service = Sessions()
    app = create_admin_app(secret=SECRET, pool_max=0.2, models={}, records=object(),
                           eval_store=object(), evaluation_sessions=service)
    with TestClient(app) as client:
        body = request_body()
        assert call(client, body, signed=False).status_code == 401
        assert call(client, body, secret=b"wrong").status_code == 401
        assert call(client, {**body, "task_id": "operator-job"}).status_code == 422
        assert call(client, {**body, "task_id": "a2603b85-0ba4-43fc-85a7-5b6cb43b78b2"}).status_code == 409
        assert call(client, {**body, "eval_id": "order-other"}).status_code == 409
        response = call(client, body)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert len(service.requests) == 1
        assert service.requests[0].worker_id == "pod-1"
    assert (service.started, service.stopped) == (1, 1)


def test_request_does_not_accept_miner_impersonation_or_caller_budgets():
    body = request_body()
    for change in ({"authority_kind": "miner"}, {"budgets": {"wall_s": 900}},
                   {"model": {"repo": "owner/model", "revision": "main"}},
                   {"sample_index": True}):
        with pytest.raises(ValueError):
            EvaluationSessionRequest.model_validate({**body, **change})
    expected = hashlib.sha256(json.dumps({"schema": "hf-snapshot/v1", "repo": "owner/model",
                                         "revision": "a" * 40}, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    assert checkpoint_identity({"repo": "owner/model", "revision": "a" * 40}) == expected
    assert expected != "a" * 40 + "0" * 24


def test_unconfigured_evaluation_machines_do_not_construct_a_fleet():
    assert from_environment(subnet=object(), environ={}) is None
    assert from_environment(subnet=object(), environ={"RELIQUARY_EVAL_SANDBOX_MACHINE_IDS": " , "}) is None
