from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliquary.environment.agentic.goldens import episode_golden_row
from reliquary.environment.agentic.suite import BUILTIN_EPISODE_ENVIRONMENTS


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "reliquary"
    / "environment"
    / "manifests"
    / "goldens"
)


@pytest.mark.parametrize("environment", BUILTIN_EPISODE_ENVIRONMENTS)
def test_episode_golden_fixture_is_immutable(environment: str):
    path = FIXTURE_ROOT / f"{environment}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["index"] for row in rows] == [0, 1, 2, 7]
    assert rows == [
        episode_golden_row(environment, row["index"])
        for row in rows
    ]
