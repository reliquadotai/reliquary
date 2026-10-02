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


def test_build_a_set_from_a_catalog_range(tmp_path, monkeypatch):
    from reliquary.eval import sets

    monkeypatch.setattr(sets, "open_source", lambda source, split: FakeEnvironment(source))
    result = CliRunner().invoke(app, ["eval", "build-set", "--source", "reliquary_dapo_math_v1",
                                      "--split", "eval", "--start", "10", "--count", "4",
                                      "--out", str(tmp_path / "set")])
    assert result.exit_code == 0, result.output
    card = json.loads(result.output)
    assert card["set_id"] == "reliquary_dapo_math_v1-eval-r10-n4"
    assert card["index_range"] == [10, 14] and card["disjointness"]["disjoint"] is True


def test_build_set_takes_one_kind_of_source(tmp_path):
    runner = CliRunner()
    both = runner.invoke(app, ["eval", "build-set", "--preset", "logic", "--source", "x",
                               "--out", str(tmp_path / "a")])
    assert both.exit_code == 1 and "exactly one of --preset and --source" in both.output
    partial = runner.invoke(app, ["eval", "build-set", "--preset", "logic", "--count", "3",
                                  "--out", str(tmp_path / "b")])
    assert partial.exit_code == 1 and "--count and --seed only" in partial.output
    args = runner.invoke(app, ["eval", "build-set", "--source", "verifiers:x",
                               "--taskset-args", "[1]", "--out", str(tmp_path / "c")])
    assert args.exit_code == 1 and "JSON object" in args.output


def test_create_needs_the_admin_secret_and_a_full_revision(monkeypatch):
    monkeypatch.delenv("RELIQUARY_ADMIN_SECRET", raising=False)
    base = ["eval", "create", "--set", "s", "--samples", "2", "--max-new-tokens", "64",
            "--temperature", "0.6"]
    runner = CliRunner()
    branch = runner.invoke(app, base + ["--model", "org/m@main"])
    assert branch.exit_code == 1 and "40-hex" in branch.output
    unsigned = runner.invoke(app, base + ["--model", "org/m@" + "a" * 40])
    assert unsigned.exit_code == 1 and "RELIQUARY_ADMIN_SECRET" in unsigned.output


def test_create_hands_the_operator_flow_its_arguments(monkeypatch):
    from reliquary.eval import operator

    monkeypatch.setenv("RELIQUARY_ADMIN_SECRET", "s" * 32)
    monkeypatch.setattr(operator, "read_set_card", lambda set_id: {"set_id": set_id})
    seen = {}

    def create(client, **kwargs):
        seen.update(kwargs)
        return [{"job_id": "order-eval-x", "set_id": c["set_id"], "qualification_id": "order-q-1"}
                for c in kwargs["cards"]]

    monkeypatch.setattr(operator, "create_evaluations", create)
    result = CliRunner().invoke(app, [
        "eval", "create", "--set", "aime26", "--set", "gpqa", "--model", "org/m@" + "a" * 40,
        "--samples", "8", "--max-new-tokens", "32768", "--thinking", "--temperature", "0.6",
        "--top-p", "0.95"])
    assert result.exit_code == 0, result.output
    assert [c["set_id"] for c in seen["cards"]] == ["aime26", "gpqa"]
    assert (seen["model"], seen["revision"], seen["thinking"]) == ("org/m", "a" * 40, True)
    assert seen["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 0}
    assert seen["prefix"] == "order-"
    assert len(json.loads(result.stdout)) == 2


def test_compare_prints_the_difference(tmp_path):
    from tests.unit.test_eval_operator import write_grading

    write_grading(tmp_path / "a", n_problems=2, correct={"p0": [False, False]})
    write_grading(tmp_path / "b", n_problems=2, correct={"p0": [True, True]})
    result = CliRunner().invoke(app, ["eval", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["envs"]["math"]["diff"] == 0.5
