"""Weight submission is optional for applications, required in weight-only mode."""

from types import SimpleNamespace

import pytest
from rich.text import Text
from typer.testing import CliRunner

import reliquary.cli.main as cli


@pytest.fixture
def runtime(monkeypatch):
    import bittensor
    import reliquary.constants as constants
    import reliquary.infrastructure.chain as chain
    import reliquary.infrastructure.task_registry_store as registry
    import reliquary.shared.modeling as modeling
    import reliquary.validator.corpus_periods as periods
    import reliquary.validator.observed_proof_rollout as observed
    import reliquary.validator.remote_proof as remote
    import reliquary.validator.rl_async_service as async_service
    import reliquary.validator.service as service
    import reliquary.validator.task_config as task_config
    import reliquary.validator.weight_only as weight_only

    events, targets = [], []
    monkeypatch.delenv("RELIQUARY_SIGNER_URL", raising=False)
    monkeypatch.setenv("RELIQUARY_RL_MODE", "window")
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(constants, "TASK_ID", "default")
    monkeypatch.setattr(constants, "TASK_IDS", ("default",))
    monkeypatch.setattr(constants, "DETACHED_TRAINER", True)
    monkeypatch.setattr(constants, "KL_BASE_MODEL", "")

    async def subtensor():
        return SimpleNamespace()

    async def read_registry():
        return {}, None

    config = SimpleNamespace(
        entry=SimpleNamespace(env_split=()), emission_cap=1.0,
        service_contract=None, price_params=None, env_caps=None,
    )
    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    monkeypatch.setattr(registry, "read_registry", read_registry)
    monkeypatch.setattr(task_config, "legacy_task_config", lambda: config)
    monkeypatch.setattr(periods, "is_period_task", lambda entry: False)
    monkeypatch.setattr(cli, "_resolve_cli_environment_mix", lambda envs: [("openmathinstruct", 1.0)])
    monkeypatch.setattr(cli, "_v3_activation_checkpoint_revision", lambda *args: None)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *args, **kw: SimpleNamespace())
    monkeypatch.setattr(remote, "executor_mode", lambda: "remote")
    monkeypatch.setattr(observed, "observed_live_requested", lambda: False)
    pool = SimpleNamespace(
        start=lambda: None, dispatch_devices=("cuda:0",),
        proxies=lambda: {"cuda:0": SimpleNamespace()}, qualify=lambda revision: {},
    )
    monkeypatch.setattr(remote.RemoteProofPool, "from_environment", lambda **kw: pool)

    async def run_application(*args, **kw):
        events.append("application")

    async def mount(*args, **kw):
        pass

    monkeypatch.setattr(service, "ValidationService", lambda *args, **kw: SimpleNamespace(
        server=SimpleNamespace(), run=run_application,
    ))
    monkeypatch.setattr(cli, "mount_corpus_service", mount)
    monkeypatch.setattr(async_service, "run_rl_async_validator", run_application)

    class WeightOnly:
        def __init__(self, **kw):
            events.append("weight-created")

        async def run(self):
            events.append("weight-run")

    class Thread:
        def __init__(self, *, target, name, daemon):
            assert name == "weight-setter" and daemon
            targets.append(target)

        def start(self):
            events.append("weight-thread-start")

    monkeypatch.setattr(weight_only, "WeightOnlyValidator", WeightOnly)
    monkeypatch.setattr(cli.threading, "Thread", Thread)
    return events, targets


@pytest.mark.parametrize("mode", ["window", "async"])
@pytest.mark.parametrize("options, weights", [
    ([], True), (["--set-weights"], True), (["--no-set-weights"], False),
])
def test_rl_application_weight_choice(monkeypatch, runtime, mode, options, weights):
    monkeypatch.setenv("RELIQUARY_RL_MODE", mode)
    events, targets = runtime

    result = CliRunner().invoke(cli.app, ["validate", *options])

    assert result.exit_code == 0, (result.output, result.exception)
    assert events == (["weight-thread-start", "application"] if weights else ["application"])
    assert len(targets) == int(weights)
    if weights:
        targets[0]()  # Run the actual thread target outside the application's event loop.
        assert events[-2:] == ["weight-created", "weight-run"]


@pytest.mark.parametrize("options", [[], ["--set-weights"]])
def test_weight_only_mode_keeps_submitting(runtime, options):
    events, targets = runtime
    result = CliRunner().invoke(cli.app, ["validate", "--no-train", *options])
    assert result.exit_code == 0, (result.output, result.exception)
    assert events == ["weight-created", "weight-run"]
    assert targets == []


@pytest.mark.parametrize("force_color", [False, True])
def test_weight_only_opt_out_refuses_before_bootstrap(monkeypatch, force_color):
    def forbidden(coro):
        coro.close()
        raise AssertionError("invalid options must refuse before wallet, signer or chain setup")

    monkeypatch.setattr(cli, "_run_validator_event_loop", forbidden)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", force_color)
    result = CliRunner().invoke(
        cli.app, ["validate", "--no-train", "--no-set-weights"], color=force_color,
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert ("\x1b[" in result.output) is force_color
    assert "--no-set-weights requires --train" in Text.from_ansi(result.output).plain
