from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_and_service_share_the_fatal_proof_error_type():
    from reliquary.cli.main import FatalProofPlaneError as CliFatalError
    from reliquary.validator.service import (
        FatalProofPlaneError as ServiceFatalError,
    )

    assert CliFatalError is ServiceFatalError


def test_validator_event_loop_propagates_nonfatal_errors():
    from reliquary.cli.main import _run_validator_event_loop

    async def fail() -> None:
        raise ValueError("ordinary failure")

    with pytest.raises(ValueError, match="ordinary failure"):
        _run_validator_event_loop(fail())


def test_fatal_proof_error_hard_exits_despite_blocking_shutdown_thread():
    script = textwrap.dedent(
        """
        import sys
        import threading

        from reliquary.cli.main import _run_validator_event_loop
        from reliquary.validator.errors import FatalProofPlaneError

        blocker = threading.Event()
        started = threading.Event()

        def block_interpreter_shutdown():
            started.set()
            blocker.wait()

        threading.Thread(
            target=block_interpreter_shutdown,
            name="simulated-stuck-native-shutdown",
            daemon=False,
        ).start()
        assert started.wait(1.0)
        print("blocking-shutdown-thread-started", file=sys.stderr, flush=True)

        async def fail():
            try:
                raise FatalProofPlaneError("simulated active proof timeout")
            finally:
                print("simulated-service-cleanup-completed", file=sys.stderr, flush=True)

        _run_validator_event_loop(fail())
        """
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), existing_pythonpath) if part
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 1
    assert "blocking-shutdown-thread-started" in completed.stderr
    assert "forcing process exit for supervisor restart" in completed.stderr
    assert completed.stderr.index(
        "simulated-service-cleanup-completed"
    ) < completed.stderr.index("forcing process exit for supervisor restart")


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
