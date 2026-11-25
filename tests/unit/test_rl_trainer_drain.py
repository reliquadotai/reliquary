"""Managed trainer shutdown never acknowledges unfinished reports, publications or ambiguous prefetched work."""
from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
from reliquary.trainer import async_cli
from reliquary.trainer.async_trainer import AsyncTrainer, AsyncTrainerConfig
from reliquary.trainer.rl_link import TrainerHeartbeat, TrainerLinkError, TrainerRuntime
from reliquary.validator.rl_buffer import SubmittedGroup, TrainingBuffer
from tests.unit.test_rl_async_trainer import Book, Subs

IDENTITY = {"task_id": "order-train-t1", "order_sha256": "a" * 64, "run_id": "r1", "runtime_id": "b" * 32}
PROFILE = {"profile_id": ACTIVE_PROTOCOL_PROFILE.profile_id, "contract": None}
ENV = {"RELIQUARY_TASK_ID": IDENTITY["task_id"], "RELIQUARY_RL_ORDER_SHA256": IDENTITY["order_sha256"],
       "RELIQUARY_RL_RUN": IDENTITY["run_id"], "RELIQUARY_RL_RUNTIME_ID": IDENTITY["runtime_id"]}


def snapshot(**fields):
    return {"schema": "reliquary/training-runtime/v1", **IDENTITY, "controller_ready": True,
            "trainer_ready": False, "trainer_quiesced": False, "drain_requested": False,
            "admission_closed": False, "drained": False, "latest_checkpoint": 3, "observed_at": 100, **fields}


def group(lease_id, arrived=0):
    return SubmittedGroup(lease_id, "miner", "math", 1, 3, "cd" * 32, arrived, (1.0, 0.0), (False, False),
                          (1,), ((2, 3), (4, 5)), (None, None))


class Model:
    def state_dict(self):
        return {"w": torch.ones(1)}


def accepted_step(model, batches, **kw):
    gate = kw["gate"]
    gate.accepted = [True] * len(batches[0])
    gate.forwarded = [1] * len(gate.forwarded)
    gate.admitted = [1] * len(gate.admitted)
    gate.tallies = [[2, 2, 2, 2] for _ in gate.tallies]
    return model


def make_trainer(*, drain, batch=2, publisher=None, step=accepted_step, sleep=lambda _: None, prefetch=False,
                 report_workers=1):
    book = Book(latest=3)
    buffer = TrainingBuffer(book)
    subs = Subs(buffer)
    trainer = AsyncTrainer(Model(), config=AsyncTrainerConfig(batch, audits_per_step=0, prefetch=prefetch,
                                                             report_workers=report_workers),
                           buffer=buffer, submissions=subs, book=book, publisher=publisher,
                           train_step_fn=step, drain_requested=drain, sleep=sleep)
    return trainer, book, buffer, subs


def test_a_partial_batch_is_reported_untrained_without_waiting_for_more_work():
    state = {"draining": False}
    trainer, book, buffer, subs = make_trainer(drain=lambda: state["draining"], batch=3,
                                              sleep=lambda _: state.update(draining=True),
                                              step=lambda *_a, **_k: pytest.fail("drained batch trained"))
    buffer.push(group("partial"))
    try:
        assert trainer.run_once() == "draining"
        trainer.quiesce()
        assert buffer.depth() == {}
        assert subs.reports["partial"] == {"rollout_match": [0.0, 0.0], "accepted": False,
                                           "failure_class": None, "judged": False}
        assert book.latest_checkpoint == 3 and trainer.published == []
    finally:
        trainer.close()
        book.db.close()


def test_a_stale_false_probe_cannot_undo_a_drain_observed_by_another_thread():
    entered, release = threading.Event(), threading.Event()
    answers, probes = [], []

    def drain():
        probes.append(threading.current_thread().name)
        if threading.current_thread().name == "stale-drain-probe":
            entered.set()
            assert release.wait(3)
            return False
        return True

    trainer, book, _, _ = make_trainer(drain=drain)
    thread = threading.Thread(target=lambda: answers.append(trainer.drain_requested()), name="stale-drain-probe")
    try:
        thread.start()
        assert entered.wait(1)
        assert trainer.drain_requested()
        release.set()
        thread.join(2)
        assert not thread.is_alive() and answers == [True]
        assert trainer.drain_requested() and len(probes) == 2
    finally:
        release.set()
        thread.join(2)
        trainer.close()
        book.db.close()


def test_an_inflight_step_finishes_and_quiescence_joins_its_publication_and_prefetch():
    state = {"draining": False}
    publishing, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Publisher:
        def publish(self, version, weights):
            assert version == 4
            publishing.set()
            assert release.wait(5)

    def step(model, batches, **kw):
        trainer._prefetched.result(timeout=3)
        state["draining"] = True
        return accepted_step(model, batches, **kw)

    trainer, book, buffer, subs = make_trainer(drain=lambda: state["draining"], batch=1,
                                              step=step, publisher=Publisher(), prefetch=True)
    buffer.push(group("trained", arrived=0))
    buffer.push(group("prefetched", arrived=1))
    errors = []

    def quiesce():
        try:
            trainer.quiesce()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=quiesce)
    try:
        assert trainer.run_once() == "trained"
        assert publishing.wait(3)
        thread.start()
        assert not finished.wait(0.05)
        release.set()
        thread.join(3)
        assert finished.is_set() and not errors
        assert subs.reports["trained"]["accepted"]
        assert not subs.reports["prefetched"]["accepted"]
        assert book.adopted == [4] and trainer.published == [4]
        assert buffer.depth() == {}
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(3)
        trainer.close()
        book.db.close()


def test_publication_failure_is_not_forgotten_by_a_second_close():
    class Publisher:
        def publish(self, *_):
            raise OSError("publication failed")

    trainer, book, _, _ = make_trainer(drain=lambda: True, publisher=Publisher())
    trainer._publish(4, {"w": torch.ones(1)})
    try:
        for _ in range(2):
            with pytest.raises(OSError, match="publication failed"):
                trainer.quiesce()
    finally:
        with pytest.raises(OSError):
            trainer.close()
        book.db.close()


def test_ambiguous_prefetch_cannot_be_acknowledged_as_quiesced():
    trainer, book, _, _ = make_trainer(drain=lambda: True)
    failed = Future()
    failed.set_exception(TrainerLinkError("lost take response"))
    trainer._prefetched = failed
    try:
        for _ in range(2):
            with pytest.raises(TrainerLinkError, match="lost take response"):
                trainer.quiesce()
    finally:
        trainer.close()
        book.db.close()


def test_managed_lifecycle_acknowledges_only_after_all_local_cleanup(monkeypatch):
    events = []
    trainer = SimpleNamespace(version=7, quiesce=lambda: events.append("joined"), close=lambda: events.append("closed"))
    heartbeat = SimpleNamespace(stop=lambda: events.append("heartbeat_joined"))
    runtime = SimpleNamespace(ready=lambda k: events.append(("ready", k)),
                              quiesced=lambda k: events.append(("quiesced", k)))
    monkeypatch.setattr(async_cli, "run_forever", lambda t: events.append("drained") or "draining")
    async_cli.run_trainer_lifecycle(trainer, heartbeat, runtime=runtime)
    assert events == [("ready", 7), "drained", "joined", "heartbeat_joined", "closed", ("quiesced", 7)]


@pytest.mark.parametrize("failed", ["run", "quiesce", "heartbeat", "close"])
def test_cleanup_or_execution_error_never_sends_a_quiesced_ack(monkeypatch, failed):
    events = []

    def event(name, result=None):
        def op(*_):
            events.append(name)
            if failed == name:
                raise RuntimeError(name)
            return result
        return op

    trainer = SimpleNamespace(version=7, quiesce=event("quiesce"), close=event("close"))
    runtime = SimpleNamespace(ready=event("ready"), quiesced=event("ack"))
    monkeypatch.setattr(async_cli, "run_forever", event("run", "draining"))
    with pytest.raises(RuntimeError, match=failed):
        async_cli.run_trainer_lifecycle(trainer, SimpleNamespace(stop=event("heartbeat")), runtime=runtime)
    assert "ack" not in events and "close" in events


def test_unexpected_managed_loop_exit_is_not_finality(monkeypatch):
    monkeypatch.setattr(async_cli, "run_forever", lambda t: None)
    trainer = SimpleNamespace(version=3, close=lambda: None)
    runtime = SimpleNamespace(ready=lambda k: None, quiesced=lambda k: pytest.fail("unexpected finality"))
    with pytest.raises(RuntimeError, match="without a drain"):
        async_cli.run_trainer_lifecycle(trainer, SimpleNamespace(stop=lambda: None), runtime=runtime)


def test_a_failed_parallel_report_still_joins_other_reports_before_exit():
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    trainer, book, buffer, subs = make_trainer(drain=lambda: False, report_workers=2)
    buffer.push(group("a-failed"))
    buffer.push(group("b-pending", arrived=1))
    normal_report = subs.report_training

    def report(lease_id, **kwargs):
        if lease_id == "a-failed":
            raise RuntimeError("report response unknown")
        entered.set()
        assert release.wait(5)
        normal_report(lease_id, **kwargs)

    subs.report_training = report
    errors, acknowledgements = [], []

    def run():
        try:
            async_cli.run_trainer_lifecycle(
                trainer, SimpleNamespace(stop=lambda: None),
                runtime=SimpleNamespace(ready=lambda _: None, quiesced=lambda _: acknowledgements.append(True)))
        except BaseException as exc:
            errors.append(exc)
        finally:
            exited.set()

    thread = threading.Thread(target=run)
    try:
        thread.start()
        assert entered.wait(3)
        assert not exited.wait(0.05)
        release.set()
        thread.join(3)
        assert exited.is_set() and len(errors) == 1 and str(errors[0]) == "report response unknown"
        assert not acknowledgements
        assert buffer.get("a-failed") is not None and buffer.get("b-pending") is None
    finally:
        release.set()
        thread.join(3)
        trainer.close()
        book.db.close()


def test_runtime_snapshots_are_fresh_and_bound_to_the_owned_execution():
    link = SimpleNamespace(runtime=lambda: snapshot())
    runtime = TrainerRuntime(link, IDENTITY, clock=lambda: 100)
    assert not runtime.drain_requested()
    for changes in ({"runtime_id": "other"}, {"observed_at": 89}, {"observed_at": 106},
                    {"drain_requested": 1}, {"latest_checkpoint": True}, {"observed_at": float("nan")},
                    {"drained": True}, {"trainer_quiesced": True}, {"drain_requested": True}):
        link.runtime = lambda changes=changes: snapshot(**changes)
        with pytest.raises(TrainerLinkError):
            runtime.snapshot()


def test_ready_and_quiesced_acknowledgements_require_the_actual_joined_checkpoint():
    calls = []
    link = SimpleNamespace(
        runtime_ready=lambda identity: calls.append(dict(identity)) or snapshot(trainer_ready=True),
        runtime_quiesced=lambda identity, k: calls.append((dict(identity), k)) or
        snapshot(drain_requested=True, admission_closed=True, trainer_quiesced=True))
    runtime = TrainerRuntime(link, IDENTITY, clock=lambda: 100)
    runtime.ready(3)
    runtime.quiesced(3)
    assert calls == [IDENTITY, (IDENTITY, 3)]
    with pytest.raises(TrainerLinkError, match="initialized"):
        runtime.ready(4)
    with pytest.raises(TrainerLinkError, match="quiescence"):
        runtime.quiesced(4)
    for invalid in (-1, True, None):
        with pytest.raises(TrainerLinkError, match="invalid"):
            runtime.ready(invalid)
        with pytest.raises(TrainerLinkError, match="invalid"):
            runtime.quiesced(invalid)


def test_managed_identity_requires_every_pin_and_matches_controller_config():
    task = {"task_id": IDENTITY["task_id"], "run": IDENTITY["run_id"]}
    assert async_cli.managed_runtime_identity({"RELIQUARY_TASK_ID": "legacy"}, task) is None
    assert async_cli.managed_runtime_identity(ENV, task) == IDENTITY
    for missing in ENV:
        with pytest.raises(ValueError):
            async_cli.managed_runtime_identity({k: v for k, v in ENV.items() if k != missing}, task)
    with pytest.raises(ValueError):
        async_cli.managed_runtime_identity(ENV, {**task, "run": "other"})


def test_heartbeat_must_join_before_the_worker_can_acknowledge_shutdown():
    entered, release = threading.Event(), threading.Event()

    def heartbeat():
        entered.set()
        release.wait(3)

    beat = TrainerHeartbeat(SimpleNamespace(heartbeat=heartbeat), every_s=0.01).start()
    try:
        assert entered.wait(1)
        with pytest.raises(TrainerLinkError, match="has not stopped"):
            beat.stop(timeout_s=0.01)
    finally:
        release.set()
        beat.stop(timeout_s=2)


def test_managed_worker_rejects_distributed_launch_before_initializing_it(monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("reliquary.trainer.distributed.trainer_backend", lambda: "fsdp2")
    monkeypatch.setattr("reliquary.trainer.cli._check_trainer_launch", lambda *_: pytest.fail("launch checked"))
    with pytest.raises(ValueError, match="single backend"):
        async_cli.run_async_train_worker()


@pytest.mark.parametrize(("profile", "error"), [
    (None, "pinned catalogue profile"),
    ({"profile_id": "other", "contract": None}, "pinned catalogue profile"),
    ({**PROFILE, "contract": {}}, "pinned catalogue profile"),
    (PROFILE, "verified local base"),
])
def test_managed_worker_checks_profile_model_and_runtime_before_initialization(tmp_path, monkeypatch, profile, error):
    events = []
    for key, value in {**ENV, "RELIQUARY_PROTOCOL_PROFILE": PROFILE["profile_id"],
                       "RELIQUARY_RL_TRAINER_URL": "http://127.0.0.1:1",
                       "RELIQUARY_RL_TRAINER_TOKEN": "private-fixture-token", "RELIQUARY_RL_VALIDATOR_HOTKEY": "h",
                       "RELIQUARY_RL_BASE_MODEL": "unresolved/model", "RELIQUARY_TRAINER_STATE_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("reliquary.trainer.distributed.trainer_backend", lambda: "single")
    monkeypatch.setattr("reliquary.trainer.cli._check_trainer_launch", lambda *_: None)
    link = SimpleNamespace(config=lambda: {"task_id": IDENTITY["task_id"], "run": IDENTITY["run_id"], "profile": profile},
                           close=lambda: events.append("link_closed"))
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerLink", lambda *_a, **_kw: link)
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerRuntime", lambda *_: SimpleNamespace(
        snapshot=lambda: events.append("runtime_checked") or snapshot()))
    monkeypatch.setattr(async_cli, "resume_weights", lambda *_a, **_kw: pytest.fail("weights initialized"))
    with pytest.raises(ValueError, match=error):
        async_cli.run_async_train_worker()
    assert events == (["runtime_checked", "link_closed"] if profile == PROFILE else ["link_closed"])


def test_worker_ready_follows_model_initialization_and_quiesced_follows_join(tmp_path, monkeypatch):
    events = []
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for key, value in {**ENV, "RELIQUARY_PROTOCOL_PROFILE": PROFILE["profile_id"],
                       "RELIQUARY_RL_TRAINER_URL": "http://127.0.0.1:1",
                       "RELIQUARY_RL_TRAINER_TOKEN": "private-fixture-token", "RELIQUARY_RL_VALIDATOR_HOTKEY": "h",
                       "RELIQUARY_RL_BASE_MODEL": str(model_dir),
                       "RELIQUARY_TRAINER_STATE_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("reliquary.trainer.distributed.trainer_backend", lambda: "single")
    monkeypatch.setattr("reliquary.trainer.cli._check_trainer_launch", lambda *_: None)
    link = SimpleNamespace(config=lambda: {"task_id": IDENTITY["task_id"], "run": IDENTITY["run_id"], "profile": PROFILE, "targets": {}},
                           requeue=lambda: 0, close=lambda: events.append("link_closed"))
    def trainer_link(*_a, **kw):
        assert kw == {"timeout_s": 10.0}
        return link

    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerLink", trainer_link)
    runtime = SimpleNamespace(snapshot=lambda: snapshot(), drain_requested=lambda: True,
                              ready=lambda k: events.append("ready"), quiesced=lambda k: events.append("ack"))
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerRuntime", lambda *_: runtime)
    monkeypatch.setattr(async_cli, "resume_weights", lambda *_a, **_kw: None)
    monkeypatch.setattr(async_cli, "_eos_ids", lambda *_: ())
    model = SimpleNamespace(to=lambda _: events.append("model_loaded") or model,
                            gradient_checkpointing_enable=lambda: None)
    monkeypatch.setattr("reliquary.shared.modeling.load_text_generation_model", lambda *_a, **_kw: model)
    monkeypatch.setattr("reliquary.infrastructure.storage.public_checkpoint_client", lambda: None)
    monkeypatch.setattr("reliquary.infrastructure.storage.public_checkpoint_bucket", lambda: "fixture")
    monkeypatch.setattr("reliquary.trainer.ckpt_publisher.DeltaPublisher", lambda **_: None)
    monkeypatch.setattr("reliquary.validator.training.reset_training_state", lambda: None)
    monkeypatch.setattr("reliquary.validator.telemetry.init", lambda *_a, **_kw: None)
    monkeypatch.setattr("reliquary.validator.telemetry.finish", lambda: events.append("telemetry_closed"))

    class Trainer:
        version = 3

        def __init__(self, *_a, **kw):
            assert kw["drain_requested"] == runtime.drain_requested
            events.append("trainer_initialized")

        def quiesce(self):
            events.append("joined")

        def close(self):
            events.append("closed")

    monkeypatch.setattr("reliquary.trainer.async_trainer.AsyncTrainer", Trainer)
    monkeypatch.setattr("reliquary.trainer.audits.TrainerAudits", lambda *_a, **_kw: None)
    heartbeat = SimpleNamespace(stop=lambda: events.append("heartbeat_joined"))
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerHeartbeat",
                        lambda *_: SimpleNamespace(start=lambda: heartbeat))
    monkeypatch.setattr(async_cli, "run_forever", lambda _: events.append("draining") or "draining")
    async_cli.run_async_train_worker()
    assert events == ["model_loaded", "trainer_initialized", "ready", "draining", "joined", "heartbeat_joined",
                      "closed", "ack", "telemetry_closed", "link_closed"]
