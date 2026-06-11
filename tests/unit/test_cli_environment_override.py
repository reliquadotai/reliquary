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


@pytest.fixture(autouse=True)
def _cleanup_module_cache():
    """Make sure each test re-imports cleanly — leaving a side-effecting
    typer.Option in sys.modules across tests would let one test's env-var
    setting leak into another."""
    yield
    sys.modules.pop("reliquary.cli.main", None)
