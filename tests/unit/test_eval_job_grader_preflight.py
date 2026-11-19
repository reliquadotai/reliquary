"""CLI evaluation declarations refuse an incompatible frozen grader before either write."""
from __future__ import annotations

import asyncio
import copy
import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.eval import prompt_source, protocol
from reliquary.eval.sets import build_source_set
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_eval_validator_mode import _eval_args
from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle, opener
from tests.unit.test_jobs_cli import _rl_entry, bucket, registry  # noqa: F401


def _published(tmp_path, monkeypatch, *, grader, legacy=False):
    directory = tmp_path / "verifiers-fake-r0-n5"
    with patch.object(protocol, "grader_sha256", return_value=grader):
        card = build_source_set("verifiers:fake", out=directory,
                                open_taskset=opener(fake_handle([FakeTask(i) for i in range(5)])))
    if legacy:
        card = {key: value for key, value in card.items() if not key.startswith("protocol")}
        (directory / "set.json").write_text(json.dumps(card))
    monkeypatch.setattr(prompt_source, "_loaded", {})
    monkeypatch.setattr(prompt_source, "FETCHERS", [prompt_source._from_directory])
    monkeypatch.setenv(prompt_source.SETS_DIR_ENV, str(tmp_path))
    return directory


def test_cli_refuses_grader_drift_before_manifest_or_registry_write(bucket, registry, tmp_path, monkeypatch):
    directory = _published(tmp_path, monkeypatch, grader="a" * 64)
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    before = copy.deepcopy(bucket.objects), copy.deepcopy(registry["entries"])
    frozen = {path.name: path.read_bytes() for path in directory.iterdir()}

    result = CliRunner().invoke(app, _eval_args())

    assert result.exit_code != 0, result.output
    assert "set protocol grader drift" in result.output
    assert (bucket.objects, registry["entries"]) == before
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == frozen


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_still_declares_current_and_unsealed_legacy_sets(bucket, registry, tmp_path, monkeypatch, legacy):
    _published(tmp_path, monkeypatch, grader=protocol.grader_sha256(), legacy=legacy)
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _eval_args())
    assert result.exit_code == 0, result.output
    assert asyncio.run(job_store.read_job("eval-teutonic-aime26"))[0] is not None
    assert "eval-teutonic-aime26" in registry["entries"]
