"""Build provenance must describe the bytes sent to Docker, including staged work."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("role", ["cpu_executor", "signer"])
@pytest.mark.parametrize("change", ["staged", "unstaged", "wrong_revision", "untracked"])
def test_artifact_build_is_bound_to_committed_source(tmp_path, role, change):
    repo = tmp_path / "source"
    (repo / "scripts").mkdir(parents=True)
    (repo / "reliquary").mkdir()
    script = repo / "scripts" / f"build_{role}_artifact.sh"
    shutil.copy2(ROOT / "scripts" / script.name, script)
    source = repo / "reliquary" / "__init__.py"
    source.write_text("# committed source\n")
    (repo / ".gitignore").write_text(".env\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    if change in {"staged", "unstaged"}:
        source.write_text("# uncommitted source\n")
        if change == "staged":
            git("add", "reliquary")
    elif change == "wrong_revision":
        revision = "0" * 40
    else:
        (repo / "reliquary" / "scratch.py").write_text("# not release source\n")
        (repo / "reliquary" / ".env").write_text("TEST_ONLY=not-a-credential\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    context = tmp_path / "context.tar"
    for command in ("docker", "jq", "sha256sum", "gzip"):
        executable = fake_bin / command
        executable.write_text(
            '#!/bin/sh\ncat > "$CAPTURE_CONTEXT"\nexit 90\n'
            if command == "docker" else "#!/bin/sh\nexit 91\n"
        )
        executable.chmod(0o755)
    result = subprocess.run(
        [str(script), str(tmp_path / "artifact"), revision],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}",
             "CAPTURE_CONTEXT": str(context)},
        text=True, capture_output=True, timeout=15,
    )
    if change != "untracked":
        assert result.returncode == 2, result.stderr
        assert not context.exists()
    else:
        assert result.returncode == 90, result.stderr
        with tarfile.open(context) as archive:
            assert "reliquary/__init__.py" in archive.getnames()
            assert "reliquary/scratch.py" not in archive.getnames()
            assert "reliquary/.env" not in archive.getnames()
            assert archive.extractfile("reliquary/__init__.py").read() == b"# committed source\n"
