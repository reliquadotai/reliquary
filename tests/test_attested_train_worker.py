"""The opt-in boundary has no dependency on the optional GPU trainer runtime."""

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.trainer import attested


def job_data(tmp_path):
    return {
        "version": "reliquary/attested-training-job/v1",
        "runtime_python": str(tmp_path / "runtime/bin/python"),
        **{name: str(tmp_path / name) for name in attested._PATH_OPTIONS},
    }


def write_job(tmp_path, **changes):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({**job_data(tmp_path), **changes}))
    path.chmod(0o600)
    return path


def test_opt_in_dispatch_does_not_start_legacy_worker(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(attested, "run_attested_worker", calls.append)
    result = CliRunner().invoke(
        app, ["train-worker", "--attested-job", str(tmp_path / "job")]
    )
    assert result.exit_code == 0, result.output
    assert calls == [tmp_path / "job"]


def test_default_and_shadow_keep_legacy_dispatch(monkeypatch):
    from reliquary.trainer import cli

    calls = []
    monkeypatch.setattr(cli, "run_train_worker", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()
    assert runner.invoke(app, ["train-worker"]).exit_code == 0
    assert runner.invoke(app, ["train-worker", "--shadow"]).exit_code == 0
    assert calls == [{"shadow": False}, {"shadow": True}]


def test_shadow_cannot_be_combined_with_attested_job(monkeypatch, tmp_path):
    monkeypatch.setattr(
        attested, "run_attested_worker", lambda _: pytest.fail("launched")
    )
    result = CliRunner().invoke(
        app, ["train-worker", "--shadow", "--attested-job", str(tmp_path / "job")]
    )
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


@pytest.mark.parametrize(
    "change",
    [
        {"runtime_python": "relative/python"},
        {"config": "/state/\0config"},
        {"max_steps": 0},
        {"max_steps": True},
        {"max_steps": 100001},
        {"idle_seconds": -1},
        {"lease_seconds": 3601},
        {"device": "cpu"},
        {"shell": "arbitrary-command"},
    ],
)
def test_job_rejects_unknown_or_unbounded_inputs(tmp_path, change):
    with pytest.raises(ValueError):
        attested.load_job(write_job(tmp_path, **change))


def test_job_rejects_duplicate_fields_and_oversized_input(tmp_path):
    path = write_job(tmp_path)
    path.write_text('{"version":"one","version":"two"}')
    with pytest.raises(ValueError, match="duplicate"):
        attested.load_job(path)
    path.write_bytes(b" " * 16385)
    with pytest.raises(ValueError, match="16 KiB"):
        attested.load_job(path)


def test_job_requires_regular_owner_controlled_file(tmp_path):
    path = write_job(tmp_path)
    path.chmod(0o666)
    with pytest.raises(ValueError, match="owner-controlled"):
        attested.load_job(path)
    path.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(OSError):
        attested.load_job(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        attested.load_job(fifo)


def test_handoff_has_fixed_argv_and_executes_in_place(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        attested.os, "execv", lambda binary, argv: seen.append((binary, argv))
    )
    attested.run_attested_worker(
        write_job(tmp_path, device="cpu-development", max_steps=2)
    )
    binary, argv = seen[0]
    assert binary == str(tmp_path / "runtime/bin/python")
    assert argv[:3] == [binary, "-I", "-c"]
    assert argv[4:7] == [
        attested.TEE_RUNTIME_URL,
        attested.TEE_RUNTIME_COMMIT,
        attested.TEE_RUNTIME_VERSION,
    ]
    assert argv[7] == sys.prefix
    assert argv[8:] == attested.load_job(tmp_path / "job.json").trainer_args()
    assert "--device" in argv and "cpu-development" in argv


@pytest.fixture
def isolated_runtime(tmp_path):
    """A deliberately tiny distribution fixture, never a hardware/TEE backend."""
    root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / "bin/python"
    site = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    package = site / "reliquary_tee"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import json, sys\n"
        "def main():\n"
        "    print(json.dumps(sys.argv))\n"
        "    raise SystemExit(7)\n"
    )
    metadata = site / f"reliquary_tee-{attested.TEE_RUNTIME_VERSION}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: reliquary-tee\nVersion: {attested.TEE_RUNTIME_VERSION}\n"
    )
    direct_url = metadata / "direct_url.json"
    source = {
        "url": attested.TEE_RUNTIME_URL,
        "vcs_info": {"vcs": "git", "commit_id": attested.TEE_RUNTIME_COMMIT},
    }
    direct_url.write_text(json.dumps(source))
    return python, direct_url, source


def bootstrap(python, *args, env=None, parent_prefix=sys.prefix):
    return subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            attested._BOOTSTRAP,
            attested.TEE_RUNTIME_URL,
            attested.TEE_RUNTIME_COMMIT,
            attested.TEE_RUNTIME_VERSION,
            parent_prefix,
            *args,
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
        check=False,
    )


def test_installed_runtime_pin_and_exit_status(isolated_runtime, tmp_path):
    python, _, _ = isolated_runtime
    hostile = tmp_path / "reliquary_tee"
    hostile.mkdir()
    (hostile / "__init__.py").write_text(
        "raise RuntimeError('PYTHONPATH was imported')"
    )
    result = bootstrap(
        python,
        "trainer",
        "--max-steps",
        "2",
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == ["reliquary-tee", "trainer", "--max-steps", "2"]


@pytest.mark.parametrize(
    ("commit", "returncode"),
    [
        ("51452c8d877987d8a6735aeaf3fa872d27773e59", 1),
        ("2aa9dd968c1b58b611b37c56d3fbdd6389857315", 7),
    ],
    ids=["previous-runtime-refused", "clock-wait-runtime-accepted"],
)
def test_runtime_revision_upgrade_keeps_exact_pin(isolated_runtime, commit, returncode):
    python, path, source = isolated_runtime
    source["vcs_info"].update(commit_id=commit, requested_revision=commit)
    path.write_text(json.dumps(source))

    result = bootstrap(python, "trainer", "--max-steps", "1")

    assert result.returncode == returncode, result.stderr
    if returncode == 7:
        assert json.loads(result.stdout) == [
            "reliquary-tee",
            "trainer",
            "--max-steps",
            "1",
        ]
    else:
        assert "does not match the pinned Git revision" in result.stderr
        assert result.stdout == ""


@pytest.mark.parametrize("change", ["commit", "url", "editable", "missing"])
def test_runtime_refuses_unpinned_installation(isolated_runtime, change):
    python, path, source = isolated_runtime
    if change == "commit":
        source["vcs_info"]["commit_id"] = "a" * 40
    elif change == "url":
        source["url"] = "https://example.invalid/other.git"
    elif change == "editable":
        source["dir_info"] = {"editable": True}
    path.write_text(json.dumps(source))
    if change == "missing":
        path.unlink()
    result = bootstrap(python, "trainer")
    assert result.returncode == 1
    assert "does not match the pinned Git revision" in result.stderr
    assert result.stdout == ""


def test_global_interpreter_is_rejected():
    result = bootstrap(sys._base_executable, "trainer")
    assert result.returncode == 1
    assert "separate virtual environment" in result.stderr


def test_same_core_virtual_environment_is_rejected(isolated_runtime):
    python, _, _ = isolated_runtime
    result = bootstrap(python, "trainer", parent_prefix=str(python.parent.parent))
    assert result.returncode == 1
    assert "separate virtual environment" in result.stderr


def test_install_requirement_matches_runtime_pin():
    root = Path(__file__).resolve().parents[1]
    requirement = (root / "requirements/attested-trainer.txt").read_text()
    assert (
        f"reliquary-tee[gpu] @ git+{attested.TEE_RUNTIME_URL}@{attested.TEE_RUNTIME_COMMIT}"
        in requirement.splitlines()
    )
    assert (
        f"/blob/{attested.TEE_RUNTIME_COMMIT}/docs/async-rl.md"
        in (root / "docs/attested-training.md").read_text()
    )
