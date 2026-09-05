from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualify_environment_suite.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "qualify_environment_suite", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_environment_qualification_passes_development_override():
    report = _module().qualify(
        sample_count=250,
        profile_id="qwen3-4b-reliquary-verifiable-v6-dev1",
        allow_dirty=True,
    )
    assert report["passed"] is True
    assert report["counts"]["unique_ids"] == 250
    assert report["counts"]["malformed_inputs"] == 10_000
    assert report["reward_frontier"]["minimum_mixed_population_sigma"] > 0.24
    assert len(report["report_sha256"]) == 64
