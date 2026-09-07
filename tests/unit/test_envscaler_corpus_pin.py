"""The EnvScaler corpus must be rebuildable, byte for byte, from its pin.

The 2026-09-02 measurement ran against a loose directory identified only by
the sha256 of its two files. That directory is gone and those bytes were a
re-serialisation, so nothing tied the numbers to a fetchable input. These
tests pin the recipe rather than the directory: decode the two fields
upstream ships as JSON strings, serialise canonically, and you land on the
digests `scripts/fetch_envscaler_corpus.py` declares.

Order is the corpus identity — the loader addresses scenarios by position —
so a normalisation that reordered records would be a silent corpus swap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "fetch_envscaler_corpus.py"
SPEC = importlib.util.spec_from_file_location("fetch_envscaler_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _spec(**overrides) -> dict:
    spec = {"decode": ("tools",), "expected_length": 2}
    spec.update(overrides)
    return spec


def _upstream(records: list[dict]) -> bytes:
    return json.dumps(records).encode("utf-8")


def test_the_string_encoded_field_is_decoded() -> None:
    """Upstream ships `tools` as a JSON string; the loader wants a list."""
    raw = _upstream([
        {"env_id": "a", "tools": '[{"type": "function"}]'},
        {"env_id": "b", "tools": "[]"},
    ])

    records = json.loads(MODULE.normalize(raw, _spec()))

    assert records[0]["tools"] == [{"type": "function"}]
    assert records[1]["tools"] == []


def test_record_order_survives_normalisation() -> None:
    """Sorting records would renumber every scenario the loader indexes."""
    raw = _upstream([
        {"env_id": "z", "tools": "[]"},
        {"env_id": "a", "tools": "[]"},
    ])

    records = json.loads(MODULE.normalize(raw, _spec()))

    assert [record["env_id"] for record in records] == ["z", "a"]


def test_a_short_corpus_is_refused() -> None:
    """A truncated fetch silently shifts every index; fail instead."""
    raw = _upstream([{"env_id": "a", "tools": "[]"}])

    with pytest.raises(SystemExit, match="expected 2"):
        MODULE.normalize(raw, _spec())


def test_normalisation_is_deterministic() -> None:
    """Two runs must produce the same bytes, or the digest pin is useless."""
    raw = _upstream([
        {"tools": "[]", "env_id": "a"},
        {"env_id": "b", "tools": "[]"},
    ])

    first = MODULE.normalize(raw, _spec())
    second = MODULE.normalize(raw, _spec())

    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_every_source_declares_a_normalized_digest() -> None:
    """A source without an output pin can drift without anyone noticing."""
    assert set(MODULE.SOURCES) == set(MODULE.NORMALIZED_SHA256)
    for name, spec in MODULE.SOURCES.items():
        assert len(spec["sha256"]) == 64, name
        assert len(spec["revision"]) == 40, name
        assert len(MODULE.NORMALIZED_SHA256[name]) == 64, name


def test_the_loader_reads_the_filenames_this_script_writes() -> None:
    """The two names are a contract between the script and the environment."""
    loader = (
        Path(__file__).parents[2]
        / "reliquary/environment/agentic/envs/envscaler_tools_v1/environment.py"
    ).read_text(encoding="utf-8")

    for name in MODULE.SOURCES:
        assert f'"{name}"' in loader, name
