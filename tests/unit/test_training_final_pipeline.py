"""Real local book -> frozen usage -> lossless checkpoint -> signed admin delivery, without GPU work."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.rl_orders import finalize as pipeline
from reliquary.rl_orders.supervisor import StoppedProcess
from reliquary.shared.ckpt_index import index_key
from reliquary.shared.task_registry import TaskEntry
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from reliquary.validator.rl_training_orders import _identity
from reliquary.validator.rl_training_usage import BILLING_BASIS, bind_runtime, quiesce_trainer, request_drain
from tests.unit.test_rl_auction import ENV, ask, auctioned_book
from tests.unit.test_rl_submit import NOW, group, service
from tests.unit.test_rl_training_checkpoint_delivery import context  # noqa: F401


@pytest.fixture
def prepared(context, monkeypatch):  # noqa: F811
    store, source, original, _, _ = context
    processes = tuple(StoppedProcess(role, pid, pid, role + "-start", 0, NOW, NOW + 10)
                      for role, pid in (("controller", 900000001), ("trainer", 900000002)))
    stopped = replace(original, started_at=NOW - 1, stopped_at=NOW + 12, processes=processes,
                      cgroup_empty_at=NOW + 11)
    index = json.loads(source.objects[index_key(stopped.run_id)])
    source.objects[index_key(stopped.run_id)] = canonical_json_bytes({**index, "published_at": NOW + 5})
    state = store.root / stopped.task_id
    state.mkdir(mode=0o700)
    controller = state / "controller"
    controller.mkdir(mode=0o700)
    book, auctions, _ = auctioned_book(controller)
    submissions = service(book, probation_groups=0, audit_rate=0)
    ledger = RlPayLedger(book, submissions, None, LedgerConfig(task_id=stopped.task_id, cap=.5))
    order = {"training_id": "example", "order_sha256": stopped.order_sha256,
             "quote_id": "quote-1", "quote_sha256": "1" * 64, "billing_basis": BILLING_BASIS}
    entry = TaskEntry(stopped.task_id, "p", "c" * 64, "rl-discovered-price", {"training_order": order}, "active", None)
    with book.transaction():
        book.db.executemany("INSERT INTO rl_lease_meta(key,value) VALUES(?,?)", (
            ("training_identity", _identity(entry)), ("training_task_id", entry.task_id),
            ("training_order_sha256", stopped.order_sha256)))
    bind_runtime(book, run_id=stopped.run_id, runtime_id=stopped.runtime_id)
    auctions.post_ask(ENV, "A", .5, 8)
    lease = book.grant([ask("A", [1])])[0].leases[0]
    submissions.submit(lease.lease_id, "A", group(8))
    submissions.report_training(lease.lease_id, rollout_match=[1.] * 16, accepted=True)
    ledger.ingest()
    book.adopt_checkpoint(1)
    book.adopt_checkpoint(2)
    rl = SimpleNamespace(book=book, buffer=submissions.buffer, submissions=submissions, ledger=ledger)
    request_drain(book, runtime_id=stopped.runtime_id)
    assert quiesce_trainer(rl, runtime_id=stopped.runtime_id, latest_checkpoint=2)["drained"]
    with sqlite3.connect(controller / "rl-async.sqlite3") as destination:
        book.db.backup(destination)
    book.close()
    return SimpleNamespace(store=store, source=source, stopped=stopped, entry=entry,
                           config=SimpleNamespace(runtime_dir=state), root=store.root)


def test_final_pipeline_is_bound_immutable_and_independent_of_live_source_after_completion(prepared):
    p = prepared
    result = pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store)
    final = pipeline.TrainingRuntimeStore(p.root, checkpoint_store=p.store).final(p.entry)
    assert final["usage"]["sha256"] == result.usage_sha256
    assert final["checkpoint"]["manifest_sha256"] == result.delivery_sha256
    assert final["usage"]["receipt"]["matched_tokens"] > 0
    assert final["usage"]["receipt"]["trainer_active_ms"] == 10_000
    p.source.objects.clear()
    assert pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store) == result
    with pytest.raises(ValueError, match="replay changed"):
        pipeline.finalize_training(p.entry, replace(p.stopped, runtime_id="other"), p.config, checkpoint_store=p.store)


def test_checkpoint_copy_failure_never_publishes_billable_completion(prepared):
    p = prepared
    key = next(key for key in p.source.objects if key.endswith(".safetensors"))
    p.source.objects[key] = b"corrupted checkpoint"
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store)
    assert not (p.config.runtime_dir / "completion.json").exists()
    assert not (p.config.runtime_dir / "usage.json").exists()


def test_post_copy_failure_can_replay_from_verified_cache_without_a_live_index(prepared, monkeypatch):
    p = prepared
    write = pipeline.write_final_usage
    monkeypatch.setattr(pipeline, "write_final_usage", lambda *a: (_ for _ in ()).throw(OSError("simulated write failure")))
    with pytest.raises(OSError):
        pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store)
    assert p.store.load_manifest(p.entry.task_id) is not None
    assert pipeline.TrainingRuntimeStore(p.root).final(p.entry) is None
    p.source.objects.clear()
    monkeypatch.setattr(pipeline, "write_final_usage", write)
    assert pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store).usage_sha256


def test_signed_admin_exposes_only_the_bound_final_manifest_and_verified_ranges(prepared, monkeypatch):
    p = prepared
    pipeline.finalize_training(p.entry, p.stopped, p.config, checkpoint_store=p.store)
    monkeypatch.setenv("RELIQUARY_TRAINING_ORDERS", "on")

    async def registry(*, strict=False):
        return {p.entry.task_id: p.entry}, None

    monkeypatch.setattr("reliquary.infrastructure.task_registry_store.read_registry", registry)
    secret = b"synthetic-private-admin-key"
    app = create_admin_app(secret=secret, pool_max=.5, models={},
                           training_runtime_store=pipeline.TrainingRuntimeStore(p.root))
    with TestClient(app) as client:
        def get(path, **extra):
            stamp, nonce = str(int(time.time())), secrets.token_hex(16)
            headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                       SIGNATURE_HEADER: sign_request(secret, stamp, nonce, "GET", path, b""), **extra}
            return client.get(path, headers=headers)

        base = "/admin/v1/training/" + p.entry.task_id
        assert client.get(base).status_code == 401
        status = get(base)
        assert status.status_code == 200 and status.json()["final_usage"]["receipt"]["final"] is True
        manifest = get(base + "/checkpoint").json()
        artifact = next(file for file in manifest["files"] if file["size"] >= 8)
        response = get(base + "/checkpoint/files/" + artifact["artifact_id"], Range="bytes=0-7")
        assert response.status_code == 206 and len(response.content) == 8
        assert response.headers["content-range"] == f"bytes 0-7/{artifact['size']}"
        assert str(p.root) not in status.text + json.dumps(manifest)
        assert get(base + "/checkpoint/files/" + "f" * 64).status_code == 404
        assert get(base + "/checkpoint/files/" + artifact["artifact_id"], Range="bytes=0-16777216").status_code == 416
        assert get("/admin/v1/training/default/checkpoint").status_code == 409
