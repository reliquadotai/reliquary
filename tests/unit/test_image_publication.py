"""A code merge must not advance an image tag watched by running services."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.parametrize("event,ref,promote,latest", [
    ("push", "refs/heads/main", "false", False),
    ("push", "refs/heads/main", "true", False),
    ("workflow_dispatch", "refs/heads/main", "false", False),
    ("workflow_dispatch", "refs/heads/main", "true", True),
    ("workflow_dispatch", "refs/heads/feature", "true", False),
])
def test_latest_requires_explicit_main_promotion(tmp_path, event, ref, promote, latest):
    workflow = yaml.safe_load((Path(__file__).resolve().parents[2] /
                               ".github/workflows/docker-image.yml").read_text())
    step = next(s for s in workflow["jobs"]["build-push"]["steps"]
                if s.get("name") == "Resolve image tags")
    output = tmp_path / "output"
    subprocess.run(["bash", "-eu", "-c", step["run"]], check=True, env={
        **os.environ, "GITHUB_EVENT_NAME": event, "GITHUB_REF": ref,
        "PROMOTE_LATEST": promote, "GITHUB_SHA": "a" * 40,
        "IMAGE": "example/validator", "GITHUB_OUTPUT": str(output),
    })
    tags = output.read_text().splitlines()[1:-1]
    assert tags == (["example/validator:latest"] if latest else []) + [
        "example/validator:sha-aaaaaaa"
    ]
