"""Private order telemetry is bounded, immutable and isolated from the legacy shared run."""

import hashlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from reliquary.shared.private_training_wandb import (
    PRIVATE_TRAINING_WANDB_SCHEMA,
    private_wandb_from_environment,
    private_wandb_run_id,
    validate_private_wandb_binding,
)
from reliquary.validator import telemetry
from reliquary.validator import private_training_telemetry as private_module
from reliquary.validator.private_training_telemetry import MAX_PENDING_ROWS, PrivateTrainingWriter

ORDER, TASK = "order-a", "training-task-a"


def identity():
    return {"order_id": ORDER, "task_id": TASK, "entity": "private-team",
            "project": "customer-training", "run_id": private_wandb_run_id(ORDER, TASK)}


def configuration():
    i = identity()
    return {"order_id": ORDER, "task_id": TASK,
            "wandb": {key: i[key] for key in ("entity", "project", "run_id")}}


def environment():
    return {"RELIQUARY_TRAINING_WANDB": "on", "RELIQUARY_ORDER_ID": ORDER,
            "RELIQUARY_TASK_ID": TASK, "WANDB_ENTITY": "private-team",
            "WANDB_PROJECT": "customer-training", "WANDB_API_KEY": "synthetic-writer-key"}


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    telemetry._reset_for_tests()
    for key in (*environment(), "WANDB_RUN_ID", "RELIQUARY_WANDB_VERSION", "WANDB_MODE", "WANDB_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    yield
    telemetry._reset_for_tests()


class FakeRun:
    def __init__(self, i, config, *, step=0, log_block=None, log_error=False):
        self.id, self.entity, self.project = i["run_id"], i["entity"], i["project"]
        self.config, self.step = dict(config), step
        self.log_block, self.log_error = log_block, log_error
        self.logs, self.finished = [], False
        self.logged = threading.Event()

    def log(self, values, *, step, commit):
        if self.log_block is not None:
            self.log_block.wait(3)
        if self.log_error:
            raise RuntimeError("synthetic SDK failure")
        self.logs.append((dict(values), step, commit))
        self.logged.set()

    def finish(self):
        self.finished = True


def sdk(monkeypatch, *, init_block=None, log_block=None, log_error=False, step=0, mismatch=False):
    fake = SimpleNamespace(init_calls=[], run=None, init_started=threading.Event(), initialized=threading.Event())

    def init(**kwargs):
        fake.init_calls.append(kwargs)
        fake.init_started.set()
        if init_block is not None:
            init_block.wait(3)
        fake.run = FakeRun(identity(), kwargs["config"], step=step, log_block=log_block, log_error=log_error)
        if mismatch:
            fake.run.config["reliquary_task_id"] = "another-task"
        fake.initialized.set()
        return fake.run

    fake.init = init
    fake.Settings = lambda **kwargs: SimpleNamespace(**kwargs)
    # Unexpected module-global logging would bypass the dedicated run binding.
    fake.log = lambda *a, **kw: pytest.fail("private metrics reached shared wandb.log")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def enable(monkeypatch):
    for key, value in environment().items():
        monkeypatch.setenv(key, value)


def test_dependency_free_identity_is_exact_deterministic_and_distinct():
    expected = "rl-" + hashlib.sha256(
        b"reliquary/private-training-wandb/v1\norder-a\ntraining-task-a"
    ).hexdigest()[:40]
    assert PRIVATE_TRAINING_WANDB_SCHEMA == "reliquary/private-training-wandb/v1"
    assert private_wandb_run_id(ORDER, TASK) == expected
    assert private_wandb_run_id("order-fixture", "order-train-fixture") == "rl-a9b9553b80dee043c95f217010b7836364809982"
    assert private_wandb_run_id("other-order", TASK) != expected
    assert private_wandb_run_id(ORDER, "other-task") != expected
    assert private_wandb_from_environment(configuration(), "trainer", environment()) == identity()


@pytest.mark.parametrize("order,task", [("a\nb", TASK), ("é", TASK), (ORDER, "task/other"),
                                        ("", TASK), (ORDER, ""), (None, TASK), (ORDER, True)])
def test_invalid_identity_cannot_select_a_run(order, task):
    with pytest.raises(ValueError):
        private_wandb_run_id(order, task)


@pytest.mark.parametrize("change", [{"RELIQUARY_ORDER_ID": "other"}, {"RELIQUARY_TASK_ID": "other"},
    {"WANDB_ENTITY": "other"}, {"WANDB_PROJECT": "other"}, {"WANDB_RUN_ID": "shared"},
    {"RELIQUARY_TRAINING_WANDB": "typo"}])
def test_authenticated_config_environment_and_derived_run_must_match(change):
    with pytest.raises(ValueError):
        private_wandb_from_environment(configuration(), "trainer", {**environment(), **change})


def test_binding_is_strict_and_optional_run_id_is_an_assertion_only():
    cfg = configuration()
    assert private_wandb_from_environment(cfg, "trainer", {**environment(), "WANDB_RUN_ID": identity()["run_id"]}) == identity()
    for binding in [None, {**cfg["wandb"], "run_id": "other"}, {**cfg["wandb"], "extra": True}]:
        with pytest.raises(ValueError):
            validate_private_wandb_binding(ORDER, TASK, binding)
    with pytest.raises(ValueError):
        private_wandb_from_environment(cfg, "controller", environment())


def test_private_flush_only_captures_and_post_publication_sink_copies_allowlisted_metrics(monkeypatch):
    enable(monkeypatch)
    fake = sdk(monkeypatch)
    telemetry.init("hotkey-ignored", configuration(), role="trainer", version="ignored-shared-version")
    telemetry.hold()
    telemetry.log_training_step({"train/entropy": 0.0, "train/kl": 0.2}, 3)
    metrics = telemetry.flush({"async/completion_tokens": 100, "async/published_checkpoint": 2,
                               "prompt": "never export", "train/secret": "never export"}, 3)
    assert fake.initialized.wait(1)
    assert fake.run.logs == []
    sink = telemetry.prepare_published_checkpoint(metrics, 3, 1000.5)
    metrics["train/entropy"] = 999
    telemetry.hold()
    telemetry.log_training_step({"train/entropy": 888}, 4)
    sink()
    assert fake.run.logged.wait(1)
    values, step, commit = fake.run.logs[0]
    assert values == {"train/entropy": 0.0, "train/kl": 0.2, "async/completion_tokens": 100.0,
                      "reliquary/order_id": ORDER, "reliquary/task_id": TASK,
                      "async/published_checkpoint": 3, "_timestamp": 1000.5}
    assert step == 3 and commit is True
    assert telemetry.flush({}, 4) == {"train/entropy": 888}
    assert len(fake.run.logs) == 1
    assert fake.init_calls[0]["id"] == identity()["run_id"]
    assert fake.init_calls[0]["resume"] == "allow"
    assert fake.init_calls[0]["save_code"] is False
    settings = fake.init_calls[0]["settings"]
    assert settings.console == "off" and settings.disable_git and settings.disable_code
    assert set(fake.init_calls[0]["config"]) == {
        "reliquary_order_id", "reliquary_task_id", "reliquary_checkpoint_key", "reliquary_writer_schema"}


def test_slow_initialization_and_log_have_a_bounded_queue_and_never_block_submit_or_close(monkeypatch):
    init_release, log_release = threading.Event(), threading.Event()
    fake = sdk(monkeypatch, init_block=init_release, log_block=log_release)
    writer = PrivateTrainingWriter(identity())
    try:
        assert fake.init_started.wait(1)
        started = time.monotonic()
        for checkpoint in range(MAX_PENDING_ROWS + 10):
            writer.prepare({"train/entropy": 1}, checkpoint, 1000)()
        assert time.monotonic() - started < 0.5
        assert writer._queue.qsize() == MAX_PENDING_ROWS
        assert writer._queue.queue[0][0] == 10
        init_release.set()
        assert fake.initialized.wait(1)
        started = time.monotonic()
        writer.close(timeout=0.01)
        assert time.monotonic() - started < 0.5
        assert writer._thread.daemon
    finally:
        init_release.set()
        log_release.set()
        writer.close()


def test_resume_skips_already_logged_checkpoints_and_keeps_exact_published_version(monkeypatch):
    fake = sdk(monkeypatch, step=8)
    writer = PrivateTrainingWriter(identity())
    try:
        writer.prepare({"train/entropy": 1}, 7, 1000)()
        writer.prepare({"train/entropy": 2}, 8, 1001)()
        writer.prepare({"train/entropy": 3}, 8, 1002)()
        assert fake.initialized.wait(1)
        assert fake.run.logged.wait(1)
        writer.close()
        assert [(row["async/published_checkpoint"], step) for row, step, _ in fake.run.logs] == [(8, 8)]
    finally:
        writer.close()


@pytest.mark.parametrize("change", [{"RELIQUARY_TRAINING_WANDB": "invalid"},
    {"RELIQUARY_ORDER_ID": "other"}, {"WANDB_RUN_ID": "public-run"}, {"WANDB_API_KEY": ""},
    {"WANDB_MODE": "offline"}, {"WANDB_MODE": "disabled"},
    {"WANDB_BASE_URL": "https://unqualified.example"}])
def test_invalid_private_opt_in_never_falls_back_to_shared_run(monkeypatch, change):
    enable(monkeypatch)
    for key, value in change.items():
        monkeypatch.setenv(key, value)
    fake = sdk(monkeypatch)
    telemetry.init("shared-hotkey", configuration(), role="trainer")
    telemetry.log_training_step({"train/entropy": 0.2}, 1)
    assert telemetry.prepare_published_checkpoint({"train/entropy": 0.2}, 1, 1000) is None
    assert not telemetry.is_active()
    assert fake.init_calls == []


def test_failed_reinitialization_cannot_reuse_an_old_private_writer(monkeypatch):
    enable(monkeypatch)
    fake = sdk(monkeypatch)
    telemetry.init("shared-hotkey", configuration(), role="trainer")
    assert fake.initialized.wait(1)
    telemetry.hold()
    telemetry.log_training_step({"train/entropy": 123}, 1)
    monkeypatch.setenv("RELIQUARY_ORDER_ID", "other-order")
    telemetry.init("shared-hotkey", configuration(), role="trainer")
    assert telemetry.prepare_published_checkpoint({"train/entropy": 1}, 10, 1000) is None
    assert telemetry.flush({}, 10) == {}
    assert len(fake.init_calls) == 1


def test_sdk_identity_mismatch_and_log_failure_are_fail_soft(monkeypatch):
    fake = sdk(monkeypatch, mismatch=True)
    writer = PrivateTrainingWriter(identity())
    writer.prepare({"train/entropy": 1}, 1, 1000)()
    assert fake.initialized.wait(1)
    writer.close()
    assert fake.run.logs == [] and not writer.active
    fake = sdk(monkeypatch, log_error=True)
    writer = PrivateTrainingWriter(identity())
    writer.prepare({"train/entropy": 1}, 2, 1000)()
    assert fake.initialized.wait(1)
    writer.close()
    assert fake.run.logs == [] and not writer.active


def test_sdk_init_failure_is_background_only_and_prepared_rows_cannot_change_identity(monkeypatch):
    failed = threading.Event()

    def fail_init(**kwargs):
        failed.set()
        raise RuntimeError("synthetic authentication failure")

    fake = SimpleNamespace(init=fail_init, Settings=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    writer = PrivateTrainingWriter(identity())
    try:
        assert failed.wait(1)
        with pytest.raises(TypeError):
            writer.identity["order_id"] = "another-order"
        writer.prepare({"train/entropy": 1}, 3, 1000)()
        assert not writer.active
    finally:
        writer.close()


def test_transient_sdk_init_failure_retries_same_identity_and_delivers_queued_committed_row(monkeypatch):
    fake = sdk(monkeypatch)
    original = fake.init
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("synthetic network outage")
        return original(**kwargs)

    monkeypatch.setattr(fake, "init", flaky)
    monkeypatch.setattr(private_module, "INITIAL_RETRY_SECONDS", 0.001)
    writer = PrivateTrainingWriter(identity())
    try:
        writer.prepare({"train/entropy": 0.5}, 9, 1000)()
        assert fake.initialized.wait(1) and fake.run.logged.wait(1)
        assert len(attempts) == 2
        assert attempts[0]["id"] == attempts[1]["id"] == identity()["run_id"]
        assert fake.run.logs[0][0]["async/published_checkpoint"] == 9
    finally:
        writer.close()


def test_close_interrupts_init_retry_backoff_and_identity_mismatch_is_not_retried(monkeypatch):
    attempted = threading.Event()

    def fail(**kwargs):
        attempted.set()
        raise RuntimeError("synthetic unavailable SDK")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail, Settings=lambda **kw: kw))
    monkeypatch.setattr(private_module, "INITIAL_RETRY_SECONDS", 60)
    writer = PrivateTrainingWriter(identity())
    assert attempted.wait(1)
    started = time.monotonic()
    writer.close()
    assert time.monotonic() - started < 0.5 and not writer._thread.is_alive()
    fake = sdk(monkeypatch, mismatch=True)
    writer = PrivateTrainingWriter(identity())
    assert fake.initialized.wait(1)
    writer._thread.join(1)
    assert not writer._thread.is_alive() and len(fake.init_calls) == 1
    writer.close()


@pytest.mark.parametrize("failure", [None, "trainer", "backup"])
def test_cli_always_closes_optional_writer_even_after_failed_publication(monkeypatch, failure):
    from reliquary.trainer.async_cli import _close_async_trainer

    events = []

    def close(name):
        def run(**kwargs):
            events.append(name)
            if failure == name:
                raise RuntimeError(name)
        return run

    monkeypatch.setattr(telemetry, "finish", close("telemetry"))
    trainer, backup = SimpleNamespace(close=close("trainer")), SimpleNamespace(close=close("backup"))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            _close_async_trainer(trainer, backup)
    else:
        _close_async_trainer(trainer, backup)
    assert events == ["trainer", "backup", "telemetry"]
