"""The authenticated private runtime closes admission and drains without claiming GPU qualification."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.rl_orders.runtime import TrainerRuntime, supervised_runtime
from reliquary.validator.rl_async_service import build_rl_trainer_app
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from tests.unit.test_rl_auction import ENV, Clock, ask, auctioned_book
from tests.unit.test_rl_submit import service


@pytest.fixture
def managed(tmp_path):
    book, auctions, clock = auctioned_book(tmp_path)
    submissions = service(book, probation_groups=0, audit_rate=0)
    ledger = RlPayLedger(book, submissions, None, LedgerConfig(task_id="order-train-test", cap=.5))
    entry = SimpleNamespace(task_id="order-train-test", params={"training_order": {
        "order_sha256": "b" * 64, "deadline_at": clock() + 3600, "max_steps": 200}})
    settings = SimpleNamespace(task_id=entry.task_id, run="run-test", targets={ENV: 1}, pay_outage_fill=False)
    rl = SimpleNamespace(settings=settings, book=book, buffer=submissions.buffer, submissions=submissions,
                         ledger=ledger, board=None)
    monotonic = Clock(0)
    runtime = TrainerRuntime(entry, rl, runtime_id="a" * 32, clock=clock, monotonic=monotonic)
    rl.runtime = runtime
    auctions.admission = runtime.admission_reason
    with TestClient(build_rl_trainer_app(rl, token="test-runtime-token")) as client:
        yield SimpleNamespace(client=client, runtime=runtime, book=book, auctions=auctions, clock=clock,
                              monotonic=monotonic, identity=runtime.identity, rl=rl, entry=entry)
    book.close()


def post(managed, path, body=None):
    return managed.client.post("/rl/trainer/" + path, json=managed.identity if body is None else body,
                               headers={"Authorization": "Bearer test-runtime-token"})


def test_private_runtime_refuses_unauthenticated_and_cross_runtime_readiness(managed):
    assert managed.client.get("/rl/trainer/runtime").status_code == 401
    assert post(managed, "runtime/ready", {**managed.identity, "runtime_id": "c" * 32}).status_code == 422
    assert managed.runtime.snapshot()["trainer_ready"] is False
    assert managed.runtime.admission_reason() == "trainer_starting"
    assert post(managed, "runtime/ready").json()["trainer_ready"] is True
    assert managed.runtime.admission_reason() is None


def test_joined_worker_ack_is_distinct_from_drained_and_grace_release_is_neutral(managed):
    assert post(managed, "runtime/ready").status_code == 200
    managed.auctions.post_ask(ENV, "A", .5, 8)
    lease = managed.book.grant([ask("A", [1])])[0].leases[0]
    assert post(managed, "drain").json()["admission_closed"] is True
    ack = post(managed, "runtime/quiesced", {**managed.identity, "latest_checkpoint": 0})
    assert ack.status_code == 200
    assert ack.json()["trainer_quiesced"] is True and ack.json()["drained"] is False
    assert post(managed, "adopt", {"k": 1}).status_code == 409
    assert post(managed, "report", {}).status_code == 409
    managed.monotonic.t = 59
    managed.runtime.check_limits()
    assert managed.book.db.execute("SELECT state FROM rl_leases WHERE lease_id=?", (lease.lease_id,)).fetchone() == ("open",)
    managed.monotonic.t = 61
    managed.runtime.check_limits()
    assert managed.runtime.snapshot()["drained"] is True
    assert managed.book.db.execute("SELECT state FROM rl_leases WHERE lease_id=?", (lease.lease_id,)).fetchone() == ("released",)
    assert managed.auctions.take_debit(ENV, "A", 1.0) == 0
    assert post(managed, "runtime/quiesced", {**managed.identity, "latest_checkpoint": 0}).json()["drained"] is True


def test_drain_fences_new_batch_and_audit_takes_but_retains_config_reads(managed):
    post(managed, "runtime/ready")
    post(managed, "drain")
    for path, body in (("take", {"limit": 1}), ("take_older", {"below_k": 2, "limit": 1}),
                       ("take_exhausted", {"max_attempts": 1, "limit": 1}), ("audits", {"limit": 1})):
        reply = post(managed, path, body)
        assert reply.status_code == 200 and reply.json() == {"groups": []}
    assert managed.client.get("/rl/trainer/config", headers={"Authorization": "Bearer test-runtime-token"}).status_code == 200


def test_controller_deadline_enforcement_does_not_depend_on_a_supervisor_poll(managed):
    post(managed, "runtime/ready")
    managed.clock.t += 3601
    managed.runtime.check_limits()
    assert managed.runtime.drain_requested()
    assert managed.runtime.snapshot()["admission_closed"]
    assert post(managed, "runtime/ready").status_code == 422


def test_supervised_environment_is_exact_and_disallows_synthetic_outage_usage(managed):
    env = {"RELIQUARY_RL_RUNTIME_ID": "a" * 32, "RELIQUARY_RL_ORDER_SHA256": "b" * 64,
           "RELIQUARY_TASK_ID": managed.entry.task_id, "RELIQUARY_RL_RUN": "run-test"}
    assert supervised_runtime(managed.entry, managed.rl, environ={}) is None
    assert supervised_runtime(managed.entry, managed.rl, environ=env).identity == managed.identity
    with pytest.raises(ValueError, match="binding"):
        supervised_runtime(managed.entry, managed.rl, environ={**env, "RELIQUARY_TASK_ID": "other"})
    managed.rl.settings.pay_outage_fill = True
    with pytest.raises(ValueError, match="outage"):
        supervised_runtime(managed.entry, managed.rl, environ=env)
