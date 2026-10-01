"""The executor registry in R2: one object per audit executor, written by the
admin service (register, revoke) and the control (heartbeat, quarantine)."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from reliquary.infrastructure import corpus_executor_store as executors
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

TOKEN = "t" * 43
SHA = hashlib.sha256(TOKEN.encode()).hexdigest()


@pytest.fixture
def bucket(monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: fake)
    return fake


def _register(executor_id="pod-1", **kw):
    fields = dict(executor_id=executor_id, token_sha256=SHA, model_id="org/Frozen",
                  model_revision="abc123", expires_at=2_000_000_000.0)
    fields.update(kw)
    return asyncio.run(executors.register_executor(**fields, now=1000.0))


def test_a_registered_executor_reads_back_active(bucket):
    doc, created = _register()
    assert created is True
    stored = asyncio.run(executors.read_executor("pod-1"))
    assert stored == doc
    assert stored["status"] == "active" and stored["token_sha256"] == SHA
    assert "reliquary/corpus/executors/pod-1.json" in bucket.objects


def test_registering_the_same_executor_twice_is_idempotent_and_a_different_one_conflicts(bucket):
    first, _ = _register()
    again, created = _register()
    assert created is False and again == first
    with pytest.raises(executors.ExecutorConflict):
        _register(token_sha256="0" * 64)


def test_revoke_marks_it_revoked_and_heartbeat_keeps_the_status(bucket):
    _register()
    asyncio.run(executors.record_heartbeat("pod-1", at=1500.0, detail={"batches": 2}))
    revoked = asyncio.run(executors.set_executor_status("pod-1", "revoked", reason="pod destroyed"))
    assert revoked["status"] == "revoked" and revoked["status_reason"] == "pod destroyed"
    after = asyncio.run(executors.record_heartbeat("pod-1", at=1600.0))
    assert after["status"] == "revoked" and after["last_heartbeat"] == 1600.0


def test_an_unknown_executor_reads_none_and_cannot_be_revoked(bucket):
    assert asyncio.run(executors.read_executor("ghost")) is None
    assert asyncio.run(executors.set_executor_status("ghost", "revoked")) is None


def test_every_executor_is_listed(bucket):
    _register("pod-1")
    _register("pod-2", token_sha256="1" * 64)
    listed = asyncio.run(executors.list_executors())
    assert sorted(d["executor_id"] for d in listed) == ["pod-1", "pod-2"]


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "x" * 200, ".hidden"])
def test_an_executor_id_that_is_not_a_name_is_refused(bucket, bad):
    with pytest.raises(ValueError):
        _register(bad)


@pytest.mark.parametrize("field,value", [("token_sha256", "nothex"), ("expires_at", float("nan")),
                                         ("model_id", "")])
def test_a_registration_with_a_bad_field_is_refused(bucket, field, value):
    with pytest.raises(ValueError):
        _register(**{field: value})


def test_a_heartbeat_racing_a_revoke_never_reactivates(bucket, monkeypatch):
    _register()
    real_put = bucket.put_object
    raced = {"done": False}

    async def racing_put(Bucket, Key, Body, **condition):
        if not raced["done"]:
            raced["done"] = True
            await executors.set_executor_status("pod-1", "revoked")
        return await real_put(Bucket=Bucket, Key=Key, Body=Body, **condition)

    monkeypatch.setattr(bucket, "put_object", racing_put)
    doc = asyncio.run(executors.record_heartbeat("pod-1", at=1700.0))
    assert doc["status"] == "revoked" and doc["last_heartbeat"] == 1700.0
