"""`reliquary eval build-set | publish-set | run`."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.corpus.delivery import LocalDirectorySink
from tests.unit.test_eval_sets import FakeEnvironment


def test_build_then_publish_a_set(tmp_path, monkeypatch):
    from reliquary.corpus import delivery
    from reliquary.eval import sets, storage

    monkeypatch.setattr(sets, "open_source", lambda source, split: FakeEnvironment(source))
    runner = CliRunner()
    built = runner.invoke(app, ["eval", "build-set", "--env", "logic", "--count", "3",
                                "--seed", "4", "--out", str(tmp_path / "set")])
    assert built.exit_code == 0, built.output
    assert json.loads(built.output)["set_id"] == "logic-eval-s4-n3"
    monkeypatch.setattr(delivery.R2DeliverySink, "from_environment",
                        classmethod(lambda cls: LocalDirectorySink(tmp_path / "p")))
    monkeypatch.setattr(storage, "SubnetEvalStore", lambda: LocalDirectorySink(tmp_path / "s"))
    published = runner.invoke(app, ["eval", "publish-set", str(tmp_path / "set")])
    assert published.exit_code == 0, published.output
    assert len(json.loads(published.output)["written"]) == 5
    again = runner.invoke(app, ["eval", "build-set", "--env", "logic", "--count", "3",
                                "--seed", "4", "--out", str(tmp_path / "set")])
    assert again.exit_code == 1


def test_build_set_refuses_an_unknown_env(tmp_path):
    result = CliRunner().invoke(app, ["eval", "build-set", "--env", "telecom", "--count", "3",
                                      "--seed", "4", "--out", str(tmp_path / "x")])
    assert result.exit_code == 1 and "no held-out region" in result.output


def test_run_needs_the_executor_token(monkeypatch, tmp_path):
    monkeypatch.delenv("RELIQUARY_EXECUTOR_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["eval", "run", "--platform", "http://127.0.0.1:9",
                                      "--executor-id", "pod", "--work-dir", str(tmp_path)])
    assert result.exit_code == 1 and "RELIQUARY_EXECUTOR_TOKEN" in result.output
