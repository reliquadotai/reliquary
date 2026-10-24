"""Shared test fixtures."""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    # A command-line option, not an env var: suites that run pytest in a subprocess must not inherit the shard.
    parser.addoption("--test-shard", default=None, metavar="I/N",
                     help="run only the tests whose node id hashes to shard I of N (1-based), for CI matrices")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = config.getoption("--test-shard")
    if not shard:
        return
    index, count = (int(part) for part in shard.split("/"))
    if not 1 <= index <= count:
        raise pytest.UsageError(f"--test-shard {shard}: need 1 <= I <= N")
    keep, drop = [], []
    for item in items:
        (keep if zlib.crc32(item.nodeid.encode()) % count == index - 1 else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


@pytest.fixture(autouse=True)
def isolated_reliquary_state_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep durable-state tests isolated from host/container defaults."""

    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
