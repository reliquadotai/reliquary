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
    """logic: 3 of 4 problems ordered, 4 samples, c = 4, 2, 0, one grader crash
    on the third problem; code: 1 problem, 2 samples, one half, one right."""
    logic, logic_rows = _publish(tmp_path, "logic", 4, 1)
    code, code_rows = _publish(tmp_path, "code", 2, 3)
    right = lambda g: f'```json\n{{"a": "={g["source_index"]}"}}\n```'  # noqa: E731
    lines = []
    p0, p1, p2 = logic_rows[:3]
    for j in range(4):
        lines.append(_line(p0["problem_id"], j, right(p0), finish="length" if j == 0 else "stop"))
    lines += [_line(p1["problem_id"], 0, right(p1)), _line(p1["problem_id"], 1, right(p1)),
              _line(p1["problem_id"], 2, "no json at all"),
              _line(p1["problem_id"], 3, '```json\n{"a": 0}\n```')]
    lines += [_line(p2["problem_id"], j, '```json\n{"a": "wrong"}\n```') for j in range(3)]
    lines.append(_line(p2["problem_id"], 3, "CRASH"))
    # Noise the grader must count, not grade: a 4th problem not ordered, a repeat.
    lines.append(_line(logic_rows[3]["problem_id"], 0, right(logic_rows[3])))
    lines.append(_line(p0["problem_id"], 0, "repeat"))
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
               "problems_per_set": {logic["set_id"]: 3, code["set_id"]: 1}}
    return request


def _grade(tmp_path, request, **kw):
    return asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"), open_environment=open_grading,
        work_dir=tmp_path / "work", clock=lambda: 5.0,
        provenance={"model": "org/m", "pod_provider_id": "lium-7"}, **request, **kw))


def test_the_report_from_a_fixture_with_known_counts(tmp_path):
    request = _fixture(tmp_path)
    manifest = _grade(tmp_path, request)
    out = tmp_path / "platform" / "deliveries" / "order-e1"
    assert manifest["keys"] == ["deliveries/order-e1/graded.parquet",
                                "deliveries/order-e1/report.json",
                                "deliveries/order-e1/manifest.json"]
    report = json.loads((out / "report.json").read_text())
    logic = report["envs"]["logic"]
    assert logic["n_problems"] == 3 and logic["samples"] == 4 and logic["rows"] == 12
    assert logic["grader_errors"] == 1 and logic["rows_graded"] == 11
    assert logic["duplicate_rows"] == 1
    # c/n = 4/4, 2/4, 0/3 (the crash leaves 3 graded samples).
    assert logic["pass@1"]["value"] == pytest.approx((1 + 0.5 + 0) / 3)
    low, high = logic["pass@1"]["ci95"]
    assert low <= logic["pass@1"]["value"] <= high
    expected2 = (1 + (1 - comb(2, 2) / comb(4, 2)) + 0) / 3
    assert logic["pass@k"]["2"]["value"] == pytest.approx(expected2)
    # pass@4 only over problems with 4 graded samples.
    assert logic["pass@k"]["4"] == {"value": pytest.approx(1.0), "n_problems": 2}
    assert logic["truncation_rate"] == pytest.approx(1 / 12)
    assert logic["format_failure_rate"] == pytest.approx(2 / 12)  # "no json", "CRASH"
    assert logic["mean_completion_tokens"] == pytest.approx(10)
    code = report["envs"]["code"]
    assert code["pass@1"]["value"] == pytest.approx(0.5)
    assert code["mean_score"] == pytest.approx(0.75)
    assert report["macro"]["pass@1"] == pytest.approx((logic["pass@1"]["value"] + 0.5) / 2)
    assert set(report["macro"]["pass@k"]) == {"1", "2"}
    assert report["counts"] == {"unexpected_rows": 1, "malformed_rows": 0}
    assert report["provenance"]["pod_provider_id"] == "lium-7"
    assert {s["set_id"] for s in report["provenance"]["sets"]} == set(request["set_ids"])
    table = pq.read_table(out / "graded.parquet").to_pylist()
    assert len(table) == 14  # every ordered row, each graded or flagged
    crashed = [r for r in table if r["completion"] == "CRASH"]
    assert crashed[0]["score"] is None and crashed[0]["correct"] is None
    assert crashed[0]["grader_detail"].startswith("grader_error: RuntimeError")
    assert all(r["score"] is not None or r["grader_detail"] for r in table)
    stored = json.loads((out / "manifest.json").read_text())
    import hashlib

    for entry in stored["files"]:
        assert hashlib.sha256((out / entry["name"]).read_bytes()).hexdigest() == entry["sha256"]


def test_grading_is_idempotent(tmp_path):
    request = _fixture(tmp_path)
    first = _grade(tmp_path, request)
    calls = []

    def opening(source, split):
        calls.append(source)
        return GradingEnvironment(source)

    second = asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"), open_environment=opening, **request))
    assert second == first and calls == []


def test_an_unknown_set_is_named(tmp_path):
    request = _fixture(tmp_path)
    request["set_ids"][0] = "logic-eval-s9-n9"
    request["problems_per_set"] = {request["set_ids"][0]: 1, request["set_ids"][1]: 1}
    with pytest.raises(SetUnknown):
        _grade(tmp_path, request)


def test_more_problems_than_the_set_is_refused(tmp_path):
    request = _fixture(tmp_path)
    request["problems_per_set"][request["set_ids"][0]] = 5
    with pytest.raises(GradeRequestError):
        _grade(tmp_path, request)


def test_a_drifted_source_flags_rows_instead_of_scoring_them(tmp_path):
    request = _fixture(tmp_path)

    class Drifted(GradingEnvironment):
        def get_problem(self, index):
            return {"prompt": "something else", "index": index}

    asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"),
        open_environment=lambda s, sp: Drifted(s), work_dir=tmp_path / "w", **request))
    table = pq.read_table(tmp_path / "platform" / "deliveries" / "order-e1" /
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
