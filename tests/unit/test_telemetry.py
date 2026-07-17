"""Validator-side wandb telemetry — fail-soft, opt-in, no-op by default."""

import pytest

from reliquary.validator import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry(monkeypatch):
    """Ensure each test starts with WANDB_API_KEY unset and module state
    fresh — the module caches `_enabled` / `_run` as globals."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("RELIQUARY_WANDB_VERSION", raising=False)
    telemetry._reset_for_tests()
    yield
    telemetry._reset_for_tests()


def test_is_active_false_before_init():
    assert telemetry.is_active() is False


def test_init_noop_when_no_api_key():
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    assert telemetry.is_active() is False


def test_log_training_step_is_safe_when_disabled():
    telemetry.log_training_step({"train/lr": 1e-5}, step=3)  # must not raise


def test_finish_is_safe_when_disabled():
    telemetry.finish()  # must not raise


from unittest.mock import MagicMock


def test_init_happy_path_with_mocked_wandb(monkeypatch):
    """With WANDB_API_KEY set and wandb.init mocked, telemetry activates
    and passes the expected kwargs."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    fake_wandb = MagicMock()
    fake_wandb.init.return_value = MagicMock(id="fakerun")
    # Inject fake wandb into sys.modules so the lazy `import wandb` inside
    # init() picks it up.
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    telemetry.init(hotkey_ss58=hotkey, config={"learning_rate": 1e-5})

    assert telemetry.is_active() is True
    fake_wandb.init.assert_called_once()
    kwargs = fake_wandb.init.call_args.kwargs
    assert kwargs["project"] == "reliquary-validator"
    assert kwargs["id"] == f"{hotkey[:8]}-v1"
    assert kwargs["resume"] == "allow"
    assert kwargs["config"]["learning_rate"] == 1e-5


def test_init_reads_wandb_project_env_override(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_PROJECT", "my-custom-project")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert fake_wandb.init.call_args.kwargs["project"] == "my-custom-project"


def test_init_reads_wandb_entity_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert fake_wandb.init.call_args.kwargs["entity"] == "my-team"


def test_init_version_env_override_changes_run_id(monkeypatch):
    """RELIQUARY_WANDB_VERSION overrides the constant, producing a
    different run id — this is how the operator starts a new run."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.setenv("RELIQUARY_WANDB_VERSION", "v9")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    telemetry.init(hotkey_ss58=hotkey, config={})

    assert fake_wandb.init.call_args.kwargs["id"] == f"{hotkey[:8]}-v9"


def test_init_fail_soft_on_wandb_exception(monkeypatch, caplog):
    """If wandb.init raises (network, auth, quota), telemetry stays
    disabled and the validator keeps running. No exception propagates."""
    import logging as _logging
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    fake_wandb = MagicMock()
    fake_wandb.init.side_effect = RuntimeError("network down")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})  # must not raise

    assert telemetry.is_active() is False
    assert any("init failed" in rec.message for rec in caplog.records)


def test_init_fail_soft_when_wandb_not_installed(monkeypatch, caplog):
    """If `import wandb` raises ImportError (package not installed),
    telemetry stays disabled. No exception propagates."""
    import builtins
    import logging as _logging

    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    assert telemetry.is_active() is False
    assert any("init failed" in rec.message for rec in caplog.records)


def test_log_training_step_forwards_and_commits_metrics(monkeypatch):
    """When active, log_training_step calls wandb.log with the same dict
    and explicitly commits the numbered step."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    assert telemetry.is_active() is True

    metrics = {"train/lr": 1e-5, "train/ppo_loss": 0.42}
    telemetry.log_training_step(metrics, step=7)

    fake_wandb.log.assert_called_once_with(metrics, step=7, commit=True)


def test_log_training_step_accepts_none_step(monkeypatch):
    """step=None is valid — wandb picks its own counter."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.log_training_step({"x": 1}, step=None)

    fake_wandb.log.assert_called_once_with({"x": 1}, step=None, commit=True)


def test_log_training_step_fail_soft_on_log_exception(monkeypatch, caplog):
    """If wandb.log raises, the call returns silently (first warning
    only) — the validator must keep training."""
    import logging as _logging
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    fake_wandb.log.side_effect = RuntimeError("network down")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})

    caplog.set_level(_logging.WARNING, logger="reliquary.validator.telemetry")
    telemetry.log_training_step({"x": 1}, step=1)  # must not raise
    telemetry.log_training_step({"x": 2}, step=2)  # must not raise

    warnings = [rec for rec in caplog.records if "log failed" in rec.message]
    assert len(warnings) == 1  # second failure is suppressed


def test_update_config_records_runtime_resolved_values(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    fake_run = MagicMock(id="fakerun")
    fake_wandb.init.return_value = fake_run
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={"kl_beta": 0.04})
    telemetry.update_config({"kl_reference_revision": "a" * 40})

    fake_run.config.update.assert_called_once_with(
        {"kl_reference_revision": "a" * 40},
        allow_val_change=True,
    )


def test_finish_calls_wandb_finish_when_active(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.finish()

    fake_wandb.finish.assert_called_once()
    assert telemetry.is_active() is False  # state cleared


def test_finish_fail_soft_on_exception(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    fake_wandb = MagicMock()
    fake_wandb.finish.side_effect = RuntimeError("boom")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    telemetry.init(hotkey_ss58="5abc" + "0" * 44, config={})
    telemetry.finish()  # must not raise
    assert telemetry.is_active() is False
