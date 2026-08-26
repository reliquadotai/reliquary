"""Tests for the ``RELIQUARY_ENVIRONMENTS`` env-var override on the
``mine`` / ``validate`` CLI commands.

The ``--environments`` option accepts a comma-separated list of environment
names. The default is OpenMath-only so OpenCode code execution is explicit
opt-in. Setting ``RELIQUARY_ENVIRONMENTS`` lets operators flip or restrict
environments with just a restart, no code push.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


class _FakeCuda:
    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA H100 80GB HBM3"

    @staticmethod
    def get_device_properties(index):
        assert index == 0
        return type("Properties", (), {"uuid": "GPU-EXACT"})()


class _FakeTorch:
    cuda = _FakeCuda()


def _reload_cli_main():
    """Reload the CLI module so typer.Option defaults are re-evaluated
    against the current process environment. The Option default value
    is captured at function-decoration time (module load), so the only
    way to test different env-var states is to reload the module.
    """
    # Drop the cached module so the next import re-runs the decorators.
    sys.modules.pop("reliquary.cli.main", None)
    return importlib.import_module("reliquary.cli.main")


def _get_environments_option_default(cli_module, command_name: str) -> str:
    """Reach into the typer command's parameter list and pull out the
    Option default for ``--environments``. Typer stores the click params
    on the registered command's ``params`` list; the Option default is
    on the ``default`` attribute of the matching one.
    """
    for cmd in cli_module.app.registered_commands:
        if cmd.callback.__name__ == command_name:
            import inspect
            sig = inspect.signature(cmd.callback)
            return sig.parameters["environments"].default.default
    raise AssertionError(f"command {command_name!r} not found in app")


def test_mine_environments_defaults_to_safe_default_when_unset(monkeypatch):
    """When ``RELIQUARY_ENVIRONMENTS`` is not set, the miner stays on the
    safe OpenMath-only default."""
    monkeypatch.delenv("RELIQUARY_ENVIRONMENTS", raising=False)
    cli = _reload_cli_main()
    from reliquary.constants import DEFAULT_ENVIRONMENTS
    expected = DEFAULT_ENVIRONMENTS
    assert _get_environments_option_default(cli, "mine") == expected


def test_validate_environments_defaults_to_safe_default_when_unset(monkeypatch):
    """Same fallback on the trainer/validator subcommand."""
    monkeypatch.delenv("RELIQUARY_ENVIRONMENTS", raising=False)
    cli = _reload_cli_main()
    from reliquary.constants import DEFAULT_ENVIRONMENTS
    expected = DEFAULT_ENVIRONMENTS
    assert _get_environments_option_default(cli, "validate") == expected


def test_mine_environments_picks_up_env_var(monkeypatch):
    """Setting ``RELIQUARY_ENVIRONMENTS=openmathinstruct`` makes the miner
    CLI default to ``openmathinstruct``."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENTS", "openmathinstruct")
    cli = _reload_cli_main()
    assert _get_environments_option_default(cli, "mine") == "openmathinstruct"


def test_validate_environments_picks_up_env_var(monkeypatch):
    """Same on the trainer/validator subcommand."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENTS", "openmathinstruct")
    cli = _reload_cli_main()
    assert _get_environments_option_default(cli, "validate") == "openmathinstruct"


def test_env_var_takes_precedence_over_safe_default(monkeypatch):
    """If the operator provides a custom value it wins over the computed
    default — otherwise the override would be useless."""
    monkeypatch.setenv("RELIQUARY_ENVIRONMENTS", "openmathinstruct,opencodeinstruct")
    cli = _reload_cli_main()
    assert _get_environments_option_default(cli, "mine") == "openmathinstruct,opencodeinstruct"
    assert _get_environments_option_default(cli, "validate") == "openmathinstruct,opencodeinstruct"


def test_miner_never_requires_grader(monkeypatch):
    """Miners never grade — opencode reward is validator-authoritative — so the
    reference miner never launches the gVisor grader."""
    cli = _reload_cli_main()
    assert cli._miner_requires_grader(["opencodeinstruct"]) is False
    assert cli._miner_requires_grader(["openmathinstruct"]) is False
    assert cli._miner_requires_grader(["openmathinstruct", "opencodeinstruct"]) is False


def test_cli_environment_resolution_is_fail_closed(monkeypatch):
    monkeypatch.delenv("RELIQUARY_PROTOCOL_PROFILE", raising=False)
    cli = _reload_cli_main()

    assert cli._resolve_cli_environment_mix("openmathinstruct") == [
        ("openmathinstruct", cli.B_BATCH)
    ]
    with pytest.raises(ValueError, match="at least one"):
        cli._resolve_cli_environment_mix("")
    with pytest.raises(ValueError, match="duplicate"):
        cli._resolve_cli_environment_mix(
            "openmathinstruct,openmathinstruct"
        )
    with pytest.raises(ValueError, match="Unknown environment"):
        cli._resolve_cli_environment_mix("missing")
    # Installed does not imply eligible: the default v2 profile does not
    # declare the new environment and must never fall back to Math+Code.
    with pytest.raises(ValueError, match="not declared"):
        cli._resolve_cli_environment_mix("reliquaryverifiable_v1")


def test_v2_ignores_stale_proof_device_configuration(monkeypatch):
    monkeypatch.setenv("RELIQUARY_PROOF_DEVICES", "cuda:0")
    cli = _reload_cli_main()
    monkeypatch.setattr(cli, "PROTOCOL_VERSION", 2)

    assert cli._configured_proof_device_identities(_FakeTorch()) == ()


def test_v3_requires_and_resolves_explicit_physical_proof_devices(monkeypatch):
    cli = _reload_cli_main()
    monkeypatch.setattr(cli, "PROTOCOL_VERSION", 3)
    monkeypatch.setenv("RELIQUARY_PROOF_DEVICES", "cuda:00")

    identities = cli._configured_proof_device_identities(_FakeTorch())

    assert len(identities) == 1
    assert identities[0].device_id == "cuda:0"
    assert identities[0].device_uuid == "gpu-exact"

    monkeypatch.delenv("RELIQUARY_PROOF_DEVICES")
    with pytest.raises(RuntimeError, match="requires explicit proof replicas"):
        cli._configured_proof_device_identities(_FakeTorch())


def test_v3_activation_requires_pinned_model_and_stamped_resume(monkeypatch):
    cli = _reload_cli_main()
    monkeypatch.setattr(cli, "PROTOCOL_VERSION", 3)
    monkeypatch.setattr(cli, "PROTOCOL_MODEL_ID", "Qwen/Qwen3.5-4B")
    monkeypatch.setattr(
        cli,
        "PROTOCOL_MODEL_REVISION",
        "a" * 40,
    )

    assert (
        cli._v3_activation_checkpoint_revision(
            "Qwen/Qwen3.5-4B",
            "sha:" + "b" * 40,
        )
        == "b" * 40
    )

    with pytest.raises(RuntimeError, match="must bootstrap"):
        cli._v3_activation_checkpoint_revision(
            "Qwen/Qwen3.5-2B",
            "sha:" + "b" * 40,
        )
    with pytest.raises(RuntimeError, match="stamped-40-char-checkpoint"):
        cli._v3_activation_checkpoint_revision(
            "Qwen/Qwen3.5-4B",
            "",
        )


@pytest.fixture(autouse=True)
def _cleanup_module_cache():
    """Make sure each test re-imports cleanly — leaving a side-effecting
    typer.Option in sys.modules across tests would let one test's env-var
    setting leak into another."""
    yield
    sys.modules.pop("reliquary.cli.main", None)
