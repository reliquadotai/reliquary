"""Private CPU rehearsal and harmless process-lifecycle checks; no GPU jobs."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from reliquary.integrations import affine_scoring as scoring
from reliquary.integrations.affine import AffineRuntimeError
from reliquary.integrations.affine_scoring_fixture import rehearse
from reliquary.protocol.release_contract import canonical_sha256


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    root = tmp_path_factory.mktemp("scoring-rehearsal") / "case"
    rehearse(root)
    return root


@pytest.fixture
def case(rehearsal, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(rehearsal / "bundle", bundle)
    output = scoring.read_json(rehearsal / "work/native/dots-0.json")
    return bundle, tmp_path / "work", output


def _rewrite_job(bundle, change):
    job = scoring.read_json(bundle / "job.json")
    change(job)
    for field, value in (
        ("sourceClosureDigest", job["sourceFiles"]),
        ("runtimeDigest", job["runtime"]),
        ("inputInventoryDigest", job["inputFiles"]),
        ("checkpointInventoryDigest", job["checkpointFiles"]),
    ):
        job[field] = canonical_sha256(value)
    job["jobDigest"] = canonical_sha256(
        {k: v for k, v in job.items() if k != "jobDigest"}
    )
    scoring.write_json(bundle / "job.json", job, replace=True)
    return job


def _child(monkeypatch, workspace, program):
    """Replace only the command; preserve the runner's real Popen and cleanup."""
    original = subprocess.Popen
    launched = []

    def launch(_command, *args, **kwargs):
        process = original(
            [sys.executable, "-c", program, str(workspace)], *args, **kwargs
        )
        launched.append(process)
        return process

    monkeypatch.setattr(scoring.subprocess, "Popen", launch)
    return launched


def _output_program(output):
    return (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1],'native','dots-0.json').write_text("
        + repr(json.dumps(output))
        + ")"
    )


def _alive(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _stopped(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.02)
    assert not _alive(pid)


def _cli(*arguments):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "reliquary.integrations.affine_scoring_cli",
            *map(str, arguments),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_actual_cli_rehearse_verify_export_and_idempotency(tmp_path, monkeypatch):
    root = tmp_path / "cli"
    report = _cli("rehearse", "--out", root)
    assert report["stage"] == "rehearsal_completed" and report["assurance"] == "fixture"
    assert not report["gpuQualified"] and not report["operatorAccepted"]
    assert not report["nativeRewardVerified"] and not report["chainTransactions"]
    bundle, work = root / "bundle", root / "work"
    assert _cli("check", "--bundle", bundle)["stage"] == "metadata_checked"
    verified = _cli("verify", "--bundle", bundle, "--workspace", work)
    exported = tmp_path / "private-result.zip"
    assert (
        _cli("export", "--bundle", bundle, "--workspace", work, "--out", exported)[
            "stage"
        ]
        == "exported"
    )
    with zipfile.ZipFile(exported) as archive:
        assert set(archive.namelist()) == {"compute-result.json", "dots-0.json"}
    assert exported.stat().st_mode & 0o777 == 0o600

    def unexpected(*_args, **_kwargs):
        raise AssertionError("completed job launched another process")

    monkeypatch.setattr(scoring.subprocess, "Popen", unexpected)
    assert scoring.run(bundle, work, fixture=True) == verified
    with pytest.raises(AffineRuntimeError, match="private runtime file"):
        scoring.export(bundle, work, exported)


@pytest.mark.parametrize("target", ["source", "input", "job"])
def test_frozen_tampering_refused_before_launch(case, monkeypatch, target):
    bundle, work, _ = case
    if target == "source":
        (bundle / "source/grad_docs_v9.py").write_text("changed source")
    elif target == "input":
        (bundle / "inputs/todo.json").write_text("[]")
    else:
        job = scoring.read_json(bundle / "job.json")
        job["epoch"] += 1
        scoring.write_json(bundle / "job.json", job, replace=True)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("tampered job reached process launch")

    monkeypatch.setattr(scoring.subprocess, "Popen", unexpected)
    with pytest.raises(ValueError):
        scoring.run(bundle, work, fixture=True)
    assert not (work / "compute-result.json").exists()


def test_fixture_and_native_modes_cannot_be_crossed(case, monkeypatch):
    bundle, work, _ = case
    with pytest.raises(ValueError, match="cannot be interchanged"):
        scoring.run(bundle, work)
    source = scoring.read_json(bundle / "job.json")["sourceFiles"]
    monkeypatch.setattr(
        scoring, "NATIVE_SOURCE", {k: v["sha256"] for k, v in source.items()}
    )

    def native(job):
        job["assurance"] = "historical-v9-compute"
        job["runtime"]["gpu"] = {"name": "NVIDIA H200", "sm": [9, 0]}
        job["runtime"]["packages"] = {
            k: "synthetic"
            for k in ("torch", "numpy", "transformers", "safetensors", "tokenizers")
        }
        job["checkpointFiles"]["model.safetensors"] = {
            "sha256": canonical_sha256("synthetic"),
            "bytes": 1,
        }

    _rewrite_job(bundle, native)
    with pytest.raises(ValueError, match="cannot be interchanged"):
        scoring.run(bundle, work, fixture=True)


@pytest.mark.parametrize("uid,euid", [(0, 10001), (10001, 0)])
def test_direct_native_execution_refuses_root_before_asset_probes(
    monkeypatch, uid, euid
):
    monkeypatch.setattr(scoring.os, "getuid", lambda: uid)
    monkeypatch.setattr(scoring.os, "geteuid", lambda: euid)
    with pytest.raises(ValueError, match="unprivileged UID"):
        scoring._runtime({}, None, None)


@pytest.mark.parametrize("kind", ["dimensions", "nonfinite"])
def test_invalid_real_child_output_never_completes(case, monkeypatch, kind):
    bundle, work, output = case
    if kind == "dimensions":
        output["docs"][0]["dots"].pop()
    else:
        output["docs"][0]["dots"][0][0] = float("nan")
    _child(monkeypatch, work, _output_program(output))
    with pytest.raises(ValueError):
        scoring.run(bundle, work, fixture=True)
    assert not (work / "compute-result.json").exists()
    assert scoring.read_json(work / "state.json")["stage"] != "compute_completed"
    with pytest.raises(ValueError, match="bounded regular JSON"):
        scoring.verify(bundle, work)


def test_validated_output_tampering_refuses_verify_and_export(rehearsal, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(rehearsal, root)
    with (root / "work/native/dots-0.json").open("a") as stream:
        stream.write("\n")
    for action in (
        lambda: scoring.verify(root / "bundle", root / "work"),
        lambda: scoring.export(root / "bundle", root / "work", tmp_path / "bad.zip"),
    ):
        with pytest.raises(ValueError, match="bytes differ"):
            action()
    assert not (tmp_path / "bad.zip").exists()


def test_comparison_tolerance_and_oracle_binding(rehearsal, tmp_path):
    bundle, work = rehearsal / "bundle", rehearsal / "work"
    oracle = scoring.read_json(work / "native/dots-0.json")
    oracle["docs"][0]["dots"][0][0] = 0.125
    reference = tmp_path / "reference.json"
    scoring.write_json(reference, oracle)
    assert not scoring.compare(bundle, work, reference, atol=0.124, rtol=0)[
        "comparisonPassed"
    ]
    from reliquary.integrations.affine_scoring_cli import main

    assert (
        main(
            [
                "compare",
                "--bundle",
                str(bundle),
                "--workspace",
                str(work),
                "--reference",
                str(reference),
                "--atol",
                "0.124",
                "--rtol",
                "0",
            ]
        )
        == 1
    )
    result = scoring.compare(bundle, work, reference, atol=0.125, rtol=0)
    assert result["comparisonPassed"] and result["values"] == 16
    assert result["maxAbsError"] == 0.125 and not result["independentReferenceVerified"]
    assert not result["gpuQualified"] and not result["nativeRewardVerified"]
    assert _cli(
        "compare",
        "--bundle",
        bundle,
        "--workspace",
        work,
        "--reference",
        reference,
        "--atol",
        "0.125",
        "--rtol",
        "0",
    )["comparisonPassed"]
    for atol, rtol in [(float("nan"), 0), (0, float("inf")), (-1, 0), (0, True)]:
        with pytest.raises(ValueError, match="finite tolerances"):
            scoring.compare(bundle, work, reference, atol=atol, rtol=rtol)
    oracle["base"] = canonical_sha256("different synthetic oracle")
    scoring.write_json(reference, oracle, replace=True)
    with pytest.raises(ValueError, match="mismatch"):
        scoring.compare(bundle, work, reference, atol=0.125, rtol=0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group runner")
def test_leader_exit_cleans_inherited_stdout_child(case, monkeypatch):
    bundle, work, _ = case
    program = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1],'child.pid').write_text(str(child.pid))"
    )
    processes = _child(monkeypatch, work, program)
    with pytest.raises(ValueError, match="native output"):
        scoring.run(bundle, work, fixture=True, max_seconds=10)
    assert len(processes) == 1 and processes[0].returncode == 0
    monkeypatch.undo()
    _stopped(int((work / "child.pid").read_text()))
    assert not (work / "compute-result.json").exists()


@pytest.mark.parametrize(
    "reason",
    ["cancelled", "budget_exhausted", "log_limit_exceeded", "storage_limit_exceeded"],
)
def test_real_child_resource_stops(case, monkeypatch, reason):
    bundle, work, _ = case
    _rewrite_job(
        bundle,
        lambda job: job["limits"].update(
            finalizationReserveSeconds=1,
            maxLogBytes=1024,
            maxWorkspaceBytes=8192
            if reason == "storage_limit_exceeded"
            else 16 * 1024**2,
        ),
    )
    program = "import pathlib,sys,time; "
    if reason == "log_limit_exceeded":
        program += "sys.stdout.write('x'*131072); sys.stdout.flush(); "
    if reason == "storage_limit_exceeded":
        program += (
            "pathlib.Path(sys.argv[1],'native','large').write_bytes(b'x'*32768); "
        )
    program += "time.sleep(30)"
    processes = _child(monkeypatch, work, program)
    stop = threading.Event() if reason == "cancelled" else None
    timer = threading.Timer(0.2, stop.set) if stop else None
    if timer:
        timer.start()
    try:
        with pytest.raises(ValueError, match=reason):
            scoring.run(
                bundle,
                work,
                fixture=True,
                max_seconds=2 if reason == "budget_exhausted" else 10,
                stop_event=stop,
            )
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert len(processes) == 1 and processes[0].poll() is not None
    assert scoring.read_json(work / "state.json")["stage"] == reason
    assert not (work / "compute-result.json").exists()
    assert (work / "runtime.log").stat().st_size <= 1024


def test_output_byte_limit_before_completion(case, monkeypatch):
    bundle, work, output = case
    _rewrite_job(bundle, lambda job: job["limits"].update(maxOutputBytes=1024))
    output["padding"] = "x" * 4096
    _child(monkeypatch, work, _output_program(output))
    with pytest.raises(ValueError, match="bounded native output"):
        scoring.run(bundle, work, fixture=True)
    assert not (work / "compute-result.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group runner")
def test_reconcile_never_kills_an_existing_process_group(case, monkeypatch):
    bundle, work, _ = case
    work.mkdir(mode=0o700)
    other = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    original = os.killpg
    calls = []

    def observe(pid, sig):
        calls.append(sig)
        return original(pid, sig)

    monkeypatch.setattr(scoring.os, "killpg", observe)
    try:
        job = scoring.read_json(bundle / "job.json")
        scoring.write_json(
            work / "state.json", {**scoring.summary(job, "running"), "pid": other.pid}
        )
        with pytest.raises(ValueError, match="still exists"):
            scoring.reconcile(work)
        assert other.poll() is None and calls == [0]
        with pytest.raises(ValueError, match="reconciliation"):
            scoring.run(bundle, work, fixture=True)
    finally:
        original(other.pid, signal.SIGKILL)
        other.wait(timeout=5)
    assert scoring.reconcile(work) == {"stage": "reconciled", "rerunAllowed": False}
