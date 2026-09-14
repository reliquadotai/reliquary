"""Main publishes the complete runtime to latest; feature builds do not."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml


@pytest.mark.parametrize("event,ref,latest,logic", [
    ("push", "refs/heads/main", True, "true"),
    ("workflow_dispatch", "refs/heads/main", False, "false"),
    ("workflow_dispatch", "refs/heads/main", True, "true"),
    ("workflow_dispatch", "refs/heads/feature", False, "false"),
    ("workflow_dispatch", "refs/heads/feature", False, "true"),
])
def test_latest_publishes_complete_main_runtime(tmp_path, event, ref, latest, logic):
    workflow = yaml.safe_load((Path(__file__).resolve().parents[2] /
                               ".github/workflows/docker-image.yml").read_text())
    step = next(s for s in workflow["jobs"]["build-push"]["steps"]
                if s.get("name") == "Resolve image tags")
    output = tmp_path / "output"
    result = subprocess.run(["bash", "-eu", "-c", step["run"]], check=False, env={
        **os.environ, "GITHUB_EVENT_NAME": event, "GITHUB_REF": ref,
        "GITHUB_SHA": "a" * 40,
        "IMAGE": "example/validator", "GITHUB_OUTPUT": str(output),
        "INCLUDE_LOGIC": logic,
    })
    assert result.returncode == 0
    tags = output.read_text().splitlines()[1:-1]
    assert tags == (["example/validator:latest"] if latest else []) + [
        "example/validator:sha-aaaaaaa" + ("-logic" if logic == "true" else "")
    ]
