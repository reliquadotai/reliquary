"""The admin's sandbox machine routes: the platform registers, drains and reads the gateways it
rents, named with its prefix, and nothing else in the machine directory."""

from __future__ import annotations

import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.infrastructure import sandbox_store
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

SECRET = b"admin-secret"
PREFIX = "rlq-gw-test-"
KEY = "qn382fEFjFiqPt+otVIYgGBCnVW+CCydEoz6dDuDls4="
OTHER_KEY = "I6T1VLUI4qWPFcdNmZ1YsDhmD8hl413iCS0Z5Tviz/Y="


def make(monkeypatch, prefix=PREFIX):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(sandbox_store, "get_s3_client", lambda **kw: fake)
    app = create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=object(),
                           task_prefix="order-", sandbox_machine_prefix=prefix)
    client = TestClient(app)

    def call(method, path, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return client.request(method, path, content=data, headers=headers)

    return call


def registration(machine_id=PREFIX + "gw-0123456789abcdef0123", **kw):
    return {"machine_id": machine_id, "address": "https://sbx-eval2.reliqua.ai:20004",
            "provider": "targon", "capacity": 39, "key_id": machine_id + "-k1",
            "public_key_b64": KEY, "valid_from": 1_800_000_000, **kw}


def test_the_platform_registers_reads_and_drains_its_own_gateway(monkeypatch):
    admin = make(monkeypatch)
    body = registration()
    created = admin("POST", "/admin/v1/sandbox-machines", body)
    assert created.status_code == 201, created.text
    assert created.json()["status"] == "active" and created.json()["provider"] == "targon"
    # The same registration again is a replay, not a second machine.
    assert admin("POST", "/admin/v1/sandbox-machines", body).status_code == 200
    read = admin("GET", "/admin/v1/sandbox-machines/" + body["machine_id"])
    assert read.status_code == 200 and read.json()["last_heartbeat"] is None
    drained = admin("POST", f"/admin/v1/sandbox-machines/{body['machine_id']}/status",
                    {"status": "revoked", "reason": "destroyed"})
    assert drained.status_code == 200 and drained.json()["status"] == "revoked"


def test_another_registration_under_the_same_id_or_a_reused_key_is_refused(monkeypatch):
    admin = make(monkeypatch)
    assert admin("POST", "/admin/v1/sandbox-machines", registration()).status_code == 201
    again = admin("POST", "/admin/v1/sandbox-machines", registration(address="https://other.example:1"))
    assert again.status_code == 409
    # A public key belongs to one machine only.
    twin = registration(machine_id=PREFIX + "gw-ffffffffffffffffffff")
    assert admin("POST", "/admin/v1/sandbox-machines", twin).status_code == 409
    bad = registration(machine_id=PREFIX + "gw-1", public_key_b64=OTHER_KEY, address="ftp://x")
    assert admin("POST", "/admin/v1/sandbox-machines", bad).status_code == 422


@pytest.mark.parametrize("machine_id", ["sbx-gw-eval-01", "sbx-gw-01", "rlq-gw-prod-gw-1"])
def test_machines_outside_the_prefix_are_out_of_reach(monkeypatch, machine_id):
    admin = make(monkeypatch)
    assert admin("POST", "/admin/v1/sandbox-machines",
                 registration(machine_id=machine_id)).status_code == 409
    assert admin("GET", "/admin/v1/sandbox-machines/" + machine_id).status_code == 409
    assert admin("POST", f"/admin/v1/sandbox-machines/{machine_id}/status",
                 {"status": "revoked"}).status_code == 409


def test_an_unknown_machine_is_404_and_the_routes_are_off_without_a_prefix(monkeypatch):
    admin = make(monkeypatch)
    assert admin("GET", "/admin/v1/sandbox-machines/" + PREFIX + "gw-9").status_code == 404
    assert admin("POST", f"/admin/v1/sandbox-machines/{PREFIX}gw-9/status",
                 {"status": "draining"}).status_code == 404
    off = make(monkeypatch, prefix=None)
    assert off("POST", "/admin/v1/sandbox-machines", registration()).status_code == 404


def test_a_short_prefix_is_refused():
    with pytest.raises(ValueError):
        create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=object(),
                         sandbox_machine_prefix="gw-")


def test_the_order_control_reads_its_prefix_scope():
    pytest.importorskip("reliquary_sandbox")
    from reliquary.validator.sandbox_wiring import _machine_prefixes

    assert _machine_prefixes({}) is None
    assert _machine_prefixes({"RELIQUARY_SANDBOX_MACHINE_PREFIXES": '["rlq-gw-prod-"]'}) == {"rlq-gw-prod-"}
    for bad in ('[]', '"rlq-gw-prod-"', '["gw-"]'):
        with pytest.raises(ValueError):
            _machine_prefixes({"RELIQUARY_SANDBOX_MACHINE_PREFIXES": bad})
