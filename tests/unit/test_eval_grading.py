"""Grading an evaluation: every row graded or flagged, a report with known counts."""

from __future__ import annotations

import asyncio
import json
from math import comb

import pyarrow.parquet as pq
import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval.grading import (
    GradeRequestError,
    SetUnknown,
    answer_text,
    format_failed,
    grade_evaluation,
)
from reliquary.eval.sets import HELD_OUT, build_set
from reliquary.eval.storage import publish_set


class GradingEnvironment:
    """Prompt ``<source> problem <i>``; a completion holding ``=<i>`` is right,
    ``half`` scores 0.5, ``CRASH`` breaks the grader."""

    def __init__(self, name):
        self.name = name

    def __len__(self):
        held = {h.source: h.source_length for h in HELD_OUT.values()}.get(self.name)
        return held or 1000

    def get_problem(self, index):
        return {"prompt": f"{self.name} problem {index}", "index": index}

    def compute_reward(self, problem, completion):
        if "CRASH" in completion:
            raise RuntimeError("sandbox died")
        if f"={problem['index']}" in completion:
            return 1.0
        return 0.5 if "half" in completion else 0.0


def open_grading(source, split):
    return GradingEnvironment(source)


def _publish(tmp_path, env, count, seed):
    directory = tmp_path / f"build-{env}"
    card = build_set(env, count=count, seed=seed, out=directory,
                     open_environment=lambda s, sp: GradingEnvironment(s), clock=lambda: 1.0)
    asyncio.run(publish_set(directory, platform=LocalDirectorySink(tmp_path / "platform"),
                            subnet=LocalDirectorySink(tmp_path / "subnet")))
    grading = [json.loads(l) for l in (directory / "grading.jsonl").read_text().splitlines()]
    return card, grading


def _line(problem_id, index, text, *, finish="stop", tokens=10):
    return json.dumps({"problem_id": problem_id, "sample_index": index, "completion": text,
                       "completion_tokens": tokens, "finish_reason": finish}) + "\n"


def _fixture(tmp_path):
    """logic: 3 of 4 problems ordered, 4 samples each: p0 4/4 right, p1 2 right,
    one unparseable, one missing; p2 3 wrong and a grader crash. code: 1 problem,
    2 samples, one half, one right. Plus noise to count, never grade."""
    logic, logic_rows = _publish(tmp_path, "logic", 4, 1)
    code, code_rows = _publish(tmp_path, "code", 2, 3)
    right = lambda g: f'```json\n{{"a": "={g["source_index"]}"}}\n```'  # noqa: E731
    lines = []
    p0, p1, p2 = logic_rows[:3]
    for j in range(4):
        lines.append(_line(p0["problem_id"], j, right(p0), finish="length" if j == 0 else "stop"))
    lines += [_line(p1["problem_id"], 0, right(p1)), _line(p1["problem_id"], 1, right(p1)),
              _line(p1["problem_id"], 2, "no json at all")]
    lines += [_line(p2["problem_id"], j, '```json\n{"a": "wrong"}\n```') for j in range(3)]
    lines.append(_line(p2["problem_id"], 3, "CRASH"))
    lines.append(_line(logic_rows[3]["problem_id"], 0, right(logic_rows[3])))  # not ordered
    lines.append(_line(p0["problem_id"], 0, "repeat"))  # duplicate
    lines.append(_line(p0["problem_id"], 4, right(p0)))  # out of range: padding
    lines.append(_line(p0["problem_id"], -1, right(p0)))  # out of range
    c0 = code_rows[0]
    lines += [_line(c0["problem_id"], 0, "```python\nhalf\n```"),
              _line(c0["problem_id"], 1, f"```python\n={c0['source_index']}\n```")]
    platform = tmp_path / "platform" / "evaluations" / "order-e1"
    platform.mkdir(parents=True)
    (platform / "completions-00000.jsonl").write_text("".join(lines[:6]))
    (platform / "completions-00001.jsonl").write_text("".join(lines[6:]))
    request = {"set_ids": [logic["set_id"], code["set_id"]],
               "completion_keys": ["evaluations/order-e1/completions-00000.jsonl",
                                   "evaluations/order-e1/completions-00001.jsonl"],
               "problems_per_set": {logic["set_id"]: 3, code["set_id"]: 1},
               "samples_per_set": {logic["set_id"]: 4, code["set_id"]: 2}}
    return request


PROVENANCE = {"model": "org/m", "revision": "a" * 40, "pod_provider_id": "lium-7"}


def _plain_scorer(spec, environment):
    return environment.compute_reward


def _grade(tmp_path, request, **kw):
    kw.setdefault("open_environment", open_grading)
    kw.setdefault("scorer_for", _plain_scorer)
    kw.setdefault("require_sandbox", lambda spec: None)
    return asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"), work_dir=tmp_path / "work",
        clock=lambda: 5.0, provenance=PROVENANCE, **request, **kw))


def test_the_report_from_a_fixture_with_known_counts(tmp_path):
    request = _fixture(tmp_path)
    manifest = _grade(tmp_path, request)
    out = tmp_path / "platform" / "evaluations" / "order-e1"
    assert manifest["keys"] == ["evaluations/order-e1/graded.parquet",
                                "evaluations/order-e1/report.json",
                                "evaluations/order-e1/manifest.json"]
    assert not (tmp_path / "platform" / "deliveries").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["complete"] is False and manifest["complete"] is False
    logic = report["envs"]["logic"]
    assert logic["n_problems"] == 3 and logic["samples"] == 4
    assert (logic["expected_rows"], logic["graded_rows"], logic["ungraded_rows"],
            logic["missing_rows"]) == (12, 10, 1, 1)
    assert logic["duplicate_rows"] == 1 and logic["out_of_range_rows"] == 2
    assert logic["complete"] is False
    # Headline: missing and ungraded count as failures, c/n = 4/4, 2/4, 0/4.
    assert logic["pass@1"]["value"] == pytest.approx((1 + 0.5 + 0) / 3)
    low, high = logic["pass@1"]["ci95"]
    assert low <= logic["pass@1"]["value"] <= high
    assert logic["pass@k"]["2"]["value"] == pytest.approx(
        (1 + (1 - comb(2, 2) / comb(4, 2)) + 0) / 3)
    assert logic["pass@k"]["4"] == {"value": pytest.approx(2 / 3), "n_problems": 3}
    # Beside it, the variant over graded samples only: 4/4, 2/3, 0/3.
    excluded = logic["excluding_ungraded_and_missing"]
    assert excluded["pass@1"]["value"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert excluded["pass@k"]["4"]["n_problems"] == 1
    assert logic["truncation_rate"] == pytest.approx(1 / 11)
    assert logic["format_failure_rate"] == pytest.approx(2 / 11)  # "no json", "CRASH"
    code = report["envs"]["code"]
    assert code["pass@1"]["value"] == pytest.approx(0.5) and code["complete"] is True
    assert code["mean_score"] == pytest.approx(0.75)
    assert report["macro"]["pass@1"] == pytest.approx(0.5)
    assert report["macro"]["envs"] == 2 and report["macro"]["envs_without_graded_rows"] == []
    assert set(report["macro"]["pass@k"]) == {"1", "2"}
    assert report["counts"] == {"unexpected_rows": 1, "malformed_rows": 0}
    assert report["provenance"]["pod_provider_id"] == "lium-7"
    sets_ = {s["env"]: s for s in report["provenance"]["sets"]}
    assert "inflated" in sets_["code"]["contamination_note"]
    assert sets_["logic"]["contamination_note"] is None
    assert sets_["code"]["rl_disjointness"].startswith("ruled")
    table = pq.read_table(out / "graded.parquet").to_pylist()
    assert len(table) == 13  # every accepted row, each graded or flagged
    crashed = [r for r in table if r["completion"] == "CRASH"]
    assert crashed[0]["score"] is None and crashed[0]["correct"] is None
    assert crashed[0]["grader_detail"].startswith("grader_error: RuntimeError")
    assert all(r["score"] is not None or r["grader_detail"] for r in table)
    stored = json.loads((out / "manifest.json").read_text())
    import hashlib

    for entry in stored["files"]:
        assert hashlib.sha256((out / entry["name"]).read_bytes()).hexdigest() == entry["sha256"]


def test_an_env_with_no_graded_row_stays_in_the_macro_as_zero(tmp_path):
    request = _fixture(tmp_path)

    class Broken(GradingEnvironment):
        def compute_reward(self, problem, completion):
            if self.name == "reliquary_code_v1":
                raise RuntimeError("down")
            return super().compute_reward(problem, completion)

    _grade(tmp_path, request, open_environment=lambda s, sp: Broken(s))
    report = json.loads((tmp_path / "platform" / "evaluations" / "order-e1" /
                         "report.json").read_text())
    code = report["envs"]["code"]
    assert code["no_graded_rows"] is True and code["pass@1"]["value"] == 0.0
    assert code["excluding_ungraded_and_missing"]["pass@1"] is None
    assert report["macro"]["envs"] == 2
    assert report["macro"]["envs_without_graded_rows"] == ["code"]
    assert report["macro"]["pass@1"] == pytest.approx(0.5 / 2)


def test_a_code_set_needs_the_sandbox(tmp_path, monkeypatch):
    from reliquary.corpus import export
    from reliquary.eval.grading import SandboxUnavailable

    request = _fixture(tmp_path)
    monkeypatch.setattr(export, "GRADER_SOCKET_PATH", str(tmp_path / "absent.sock"))
    with pytest.raises(SandboxUnavailable, match="sandboxed grading service"):
        asyncio.run(grade_evaluation(
            eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
            subnet=LocalDirectorySink(tmp_path / "subnet"), open_environment=open_grading,
            provenance=PROVENANCE, **request))


def test_code_is_scored_by_the_sandboxed_scorer_not_the_package(tmp_path, monkeypatch):
    from reliquary.environment.grader_client import GraderInfrastructureError

    request = _fixture(tmp_path)
    seen = []

    class Code(GradingEnvironment):
        def compute_reward(self, problem, completion):
            if self.name == "reliquary_code_v1":
                raise AssertionError("executed outside the sandbox")
            return super().compute_reward(problem, completion)

        def admission_reward_cases(self, problem):
            return [{"case": problem["index"]}]

    def sandbox(problem, texts, materials=None):
        seen.append(materials)
        raise GraderInfrastructureError("unreachable")

    import reliquary.validator.admission as admission

    monkeypatch.setattr(admission, "_score_opencode_adapter", sandbox)
    _grade(tmp_path, request, open_environment=lambda s, sp: Code(s),
           scorer_for=__import__("reliquary.eval.grading", fromlist=["x"])._default_scorer)
    table = pq.read_table(tmp_path / "platform" / "evaluations" / "order-e1" /
                          "graded.parquet").to_pylist()
    code_rows = [r for r in table if r["env"] == "code"]
    assert len(code_rows) == 2 and len(seen) == 2
    assert all(r["score"] is None and "GraderInfrastructureError" in r["grader_detail"]
               for r in code_rows)


def test_grading_is_idempotent(tmp_path):
    request = _fixture(tmp_path)
    first = _grade(tmp_path, request)
    calls = []

    def opening(source, split):
        calls.append(source)
        return GradingEnvironment(source)

    second = _grade(tmp_path, request, open_environment=opening)
    assert second == first and calls == []


def test_an_unknown_set_is_named(tmp_path):
    request = _fixture(tmp_path)
    request["set_ids"][0] = "logic-eval-s9-n9"
    request["problems_per_set"] = {request["set_ids"][0]: 1, request["set_ids"][1]: 1}
    request["samples_per_set"] = {request["set_ids"][0]: 1, request["set_ids"][1]: 1}
    with pytest.raises(SetUnknown):
        _grade(tmp_path, request)


def test_more_problems_than_the_set_is_refused(tmp_path):
    request = _fixture(tmp_path)
    request["problems_per_set"][request["set_ids"][0]] = 5
    with pytest.raises(GradeRequestError):
        _grade(tmp_path, request)
    request["problems_per_set"][request["set_ids"][0]] = 1
    request["samples_per_set"][request["set_ids"][0]] = 0
    with pytest.raises(GradeRequestError):
        _grade(tmp_path, request)


def test_a_drifted_source_flags_rows_instead_of_scoring_them(tmp_path):
    request = _fixture(tmp_path)

    class Drifted(GradingEnvironment):
        def get_problem(self, index):
            return {"prompt": "something else", "index": index}

    _grade(tmp_path, request, open_environment=lambda s, sp: Drifted(s))
    table = pq.read_table(tmp_path / "platform" / "evaluations" / "order-e1" /
                          "graded.parquet").to_pylist()
    assert all(r["score"] is None and r["grader_detail"].startswith("source_drift")
               for r in table)


def test_free_text_is_graded_after_the_reasoning():
    assert answer_text("text", "<think>plan</think>The answer.") == "The answer."
    assert answer_text("text", "<think>never closed") == ""
    assert answer_text("json", "<think>x</think>y") == "<think>x</think>y"
    assert format_failed("text", "") and not format_failed("text", "ok")
    assert format_failed("boxed", "no box") and not format_failed("boxed", "\\boxed{3}")
    assert format_failed("fenced_python", "def f(): pass")
    assert not format_failed("fenced_python", "```python\ndef f(): pass\n```")
    assert format_failed("json", "nothing") and not format_failed("json", '{"a": 1}')
