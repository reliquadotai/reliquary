"""Weight submission is optional for applications, required in weight-only mode."""

from types import SimpleNamespace

import pytest
from rich.text import Text
from typer.testing import CliRunner

import reliquary.cli.main as cli


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
