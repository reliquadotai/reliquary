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
