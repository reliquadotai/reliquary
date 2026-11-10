from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_validator_event_loop_propagates_nonfatal_errors():
    from reliquary.cli.main import _run_validator_event_loop

    async def fail() -> None:
        raise ValueError("ordinary failure")

    with pytest.raises(ValueError, match="ordinary failure"):
        _run_validator_event_loop(fail())


def test_validator_event_loop_raises_soft_open_file_limit_to_hard(monkeypatch):
    # A 1024 soft limit killed the V1 controller (EMFILE) on 2026-09-22.
    import reliquary.cli.main as cli

    calls = []
    monkeypatch.setattr(cli.resource, "getrlimit", lambda _kind: (1024, 524288))
    monkeypatch.setattr(cli.resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))

    async def ok() -> None:
        return None

    cli._run_validator_event_loop(ok())
    assert calls == [(cli.resource.RLIMIT_NOFILE, (524288, 524288))]


def test_open_file_limit_raise_is_best_effort(monkeypatch):
    import reliquary.cli.main as cli

    monkeypatch.setattr(cli.resource, "getrlimit", lambda _kind: (1024, cli.resource.RLIM_INFINITY))

    def refuse(_kind, _limits):
        raise ValueError("not permitted")

    monkeypatch.setattr(cli.resource, "setrlimit", refuse)
    cli._raise_open_file_limit()  # must not raise
