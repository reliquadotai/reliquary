"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_reliquary_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep durable-state tests isolated from host/container defaults."""

    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
