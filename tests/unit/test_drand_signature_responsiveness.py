"""Independent signature fetches cannot hold the serving interpreter's GIL."""

import asyncio
import ctypes
import hashlib
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from reliquary.infrastructure import drand
from reliquary.validator.service import ValidationService

SIGNATURE = "ab" * 48
RANDOMNESS = hashlib.sha256(bytes.fromhex(SIGNATURE)).hexdigest()


@pytest.mark.asyncio
async def test_native_signature_fetch_cannot_stall_parent_heartbeat(monkeypatch):
    def native_fetch(round_number):
        # PyDLL deliberately retains the GIL, like a synchronous native binding.
        ctypes.PyDLL(None).usleep(350_000)
        return SIGNATURE

    monkeypatch.setitem(sys.modules, "bittensor_drand", SimpleNamespace(get_signature_for_round=native_fetch))
    monkeypatch.setattr(
        drand,
        "_SIGNATURE_FETCH_CODE",
        "import ctypes,sys; ctypes.PyDLL(None).usleep(350_000); sys.stdout.write('ab'*48)",
        raising=False,
    )
    gaps = []
    done = threading.Event()

    async def heartbeat():
        previous = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - previous)
            previous = now

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.01)
    try:
        assert await asyncio.to_thread(drand.verify_beacon_signature, "synthetic", 42, RANDOMNESS, SIGNATURE)
    finally:
        done.set()
        await pulse
    assert max(gaps) < 0.15


@pytest.mark.parametrize("canonical,expected", [(SIGNATURE, True), (SIGNATURE.upper(), True), ("cd" * 48, False)])
def test_cross_check_keeps_exact_signature_comparison(monkeypatch, canonical, expected):
    monkeypatch.setattr(drand.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=canonical))
    assert drand.verify_beacon_signature("synthetic", 42, RANDOMNESS, SIGNATURE) is expected


def test_wrong_randomness_rejects_without_starting_child(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid randomness started an independent fetch")

    monkeypatch.setattr(drand.subprocess, "run", unexpected)
    assert not drand.verify_beacon_signature("synthetic", 42, "00" * 32, SIGNATURE)


def test_child_has_exact_round_deadline_and_no_runtime_credentials(monkeypatch):
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "synthetic-private-value")
    monkeypatch.setenv("HF_TOKEN", "synthetic-private-value")
    monkeypatch.setenv("SSL_CERT_FILE", "/synthetic-trust.pem")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=SIGNATURE)

    monkeypatch.setattr(drand.subprocess, "run", run)
    assert drand.verify_beacon_signature("synthetic", 42, RANDOMNESS, SIGNATURE)
    argv, options = calls[0]
    assert argv[:4] == [sys.executable, "-I", "-B", "-c"]
    assert options["input"] == "42" and options["timeout"] == 30
    assert options["check"] is True and options["stderr"] == subprocess.DEVNULL
    assert options["env"]["SSL_CERT_FILE"] == "/synthetic-trust.pem"
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "R2_SECRET_ACCESS_KEY" not in options["env"] and "HF_TOKEN" not in options["env"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "malformed"])
def test_child_failure_remains_fail_closed_without_raw_output(monkeypatch, caplog, failure):
    def run(argv, **kwargs):
        if failure == "exit":
            raise subprocess.CalledProcessError(2, argv, stderr="synthetic-private-value")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 30, stderr="synthetic-private-value")
        return SimpleNamespace(stdout="synthetic-private-value")

    monkeypatch.setattr(drand.subprocess, "run", run)
    assert not drand.verify_beacon_signature("synthetic", 42, RANDOMNESS, SIGNATURE)
    assert "synthetic-private-value" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_owned_child_is_reaped_on_timeout_even_if_verification_cancelled(tmp_path, monkeypatch, cancel):
    pid_path = tmp_path / "child.pid"
    code = f"import os,time; from pathlib import Path; Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(5)"
    monkeypatch.setattr(drand, "_SIGNATURE_FETCH_CODE", code)
    monkeypatch.setattr(drand, "_SIGNATURE_FETCH_TIMEOUT_SECONDS", 0.3)
    svc = SimpleNamespace(_window_n=41)
    batcher = SimpleNamespace(beacon_invalid=False)
    task = asyncio.create_task(ValidationService._verify_beacon_async(svc, [batcher], "synthetic", 42, RANDOMNESS, SIGNATURE))
    deadline = time.monotonic() + 2
    while not pid_path.exists() and not task.done() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert pid_path.exists()
    pid = int(pid_path.read_text())
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
        assert batcher.beacon_invalid
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("owned native helper was not reaped within its bounded deadline")
