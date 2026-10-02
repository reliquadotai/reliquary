"""Grading a Verifiers set: each completion scored by its task, after the reasoning."""

from __future__ import annotations

import asyncio
import json

import pyarrow.parquet as pq

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval import verifiers_source as vs
from reliquary.eval.grading import grade_evaluation
from reliquary.eval.sets import build_source_set
from reliquary.eval.storage import publish_set
from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle
from tests.unit.test_eval_verifiers_source import opener as taskset_opener


def _publish(tmp_path, tasks):
    directory = tmp_path / "vset"
    card = build_source_set("verifiers:fake", out=directory, clock=lambda: 1.0,
                            open_taskset=taskset_opener(fake_handle(tasks)))
    asyncio.run(publish_set(directory, platform=LocalDirectorySink(tmp_path / "platform"),
                            subnet=LocalDirectorySink(tmp_path / "subnet")))
    grading = [json.loads(line) for line in (directory / "grading.jsonl").read_text().splitlines()]
    return card, grading


def _line(problem_id, index, text):
    return json.dumps({"problem_id": problem_id, "sample_index": index, "completion": text,
                       "completion_tokens": 10, "finish_reason": "stop"}) + "\n"


def _grade(tmp_path, card, lines, handle, problems, provenance=None):
    platform = tmp_path / "platform" / "evaluations" / "order-e1"
    platform.mkdir(parents=True, exist_ok=True)
    (platform / "completions-00000.jsonl").write_text("".join(lines))
    opened = []

    def open_taskset(name, args):
        opened.append((name, args))
        return handle

    manifest = asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"), work_dir=tmp_path / "work",
        clock=lambda: 5.0, provenance=provenance or {"model": "org/m"},
        set_ids=[card["set_id"]],
        completion_keys=["evaluations/order-e1/completions-00000.jsonl"],
        problems_per_set={card["set_id"]: problems}, samples_per_set={card["set_id"]: 2},
        open_environment=lambda s, sp: (_ for _ in ()).throw(AssertionError("catalog")),
        require_sandbox=lambda spec: (_ for _ in ()).throw(AssertionError("sandbox")),
        open_taskset=open_taskset))
    report = json.loads((platform / "report.json").read_text())
    rows = pq.read_table(platform / "graded.parquet").to_pylist()
    return manifest, report, sorted(rows, key=lambda r: (r["problem_id"], r["sample_index"])), \
        opened


def test_a_verifiers_set_is_graded_by_its_tasks(tmp_path):
    tasks = [FakeTask(i) for i in range(4)]
    card, grading = _publish(tmp_path, tasks)
    p = [g["problem_id"] for g in grading]
    lines = [
        _line(p[0], 0, "<think>it is 7, no, 0</think>0"),     # right, after the reasoning
        _line(p[0], 1, "<think>the answer is 0"),             # never closed: nothing to read
        _line(p[1], 0, "1"),                                   # right, no reasoning
        _line(p[1], 1, "<think>1</think>2"),                   # wrong
        _line(p[2], 0, "2"),
    ]
    manifest, report, rows, opened = _grade(tmp_path, card, lines, fake_handle(tasks), 3)
    assert opened == [("fake", {})]  # once per set
    assert [(r["problem_id"][-1], r["sample_index"], r["score"], r["grader_detail"])
            for r in rows] == [("0", 0, 1.0, ""), ("0", 1, 0.0, "format_failure"),
                               ("1", 0, 1.0, ""), ("1", 1, 0.0, ""), ("2", 0, 1.0, "")]
    assert report["envs"]["verifiers:fake"]["format_failure_rate"] == 0.2
    env = report["envs"]["verifiers:fake"]
    assert env["n_problems"] == 3 and env["missing_rows"] == 1 and env["graded_rows"] == 5
    sets = report["provenance"]["sets"][0]
    assert sets["source_kind"] == "verifiers"
    assert sets["training_overlap"]["external_benchmark"] is True
    assert sets["taskset"]["id"] == "fake"


def test_a_drifted_task_is_flagged_not_scored(tmp_path):
    tasks = [FakeTask(i) for i in range(2)]
    card, grading = _publish(tmp_path, tasks)
    moved = [FakeTask(0, prompt="a reworded question"), FakeTask(1)]
    lines = [_line(grading[0]["problem_id"], 0, "0"), _line(grading[1]["problem_id"], 0, "1")]
    _, report, rows, _ = _grade(tmp_path, card, lines, fake_handle(moved), 2)
    assert rows[0]["score"] is None and rows[0]["grader_detail"].startswith("source_drift")
    assert rows[1]["score"] == 1.0
    assert report["envs"]["verifiers:fake"]["ungraded_rows"] == 1


def test_a_task_needing_a_missing_runtime_is_ungraded_not_zero(tmp_path):
    tasks = [FakeTask(0)]
    card, grading = _publish(tmp_path, tasks)
    handle = fake_handle(tasks)

    def no_docker(task, frozen, answer):
        raise vs.RuntimeUnavailable("no Docker runtime here")

    handle.score_trace = no_docker
    _, _, rows, _ = _grade(tmp_path, card, [_line(grading[0]["problem_id"], 0, "0")], handle, 1)
    assert rows[0]["score"] is None
    assert "RuntimeUnavailable" in rows[0]["grader_detail"]


def test_a_system_turn_is_part_of_the_checked_prompt(tmp_path):
    tasks = [FakeTask(0, system="be brief")]
    card, grading = _publish(tmp_path, tasks)
    moved = [FakeTask(0, system="be verbose")]
    _, _, rows, _ = _grade(tmp_path, card, [_line(grading[0]["problem_id"], 0, "0")],
                           fake_handle(moved), 1)
    assert rows[0]["grader_detail"].startswith("source_drift")


def test_with_thinking_on_a_completion_cut_before_closing_is_not_an_answer(tmp_path):
    """The template opened the reasoning block in the prompt: a completion cut at
    its budget carries no tag, and grading its reasoning would score a guess."""
    tasks = [FakeTask(0)]
    card, grading = _publish(tmp_path, tasks)
    lines = [_line(grading[0]["problem_id"], 0, "0"),             # no tag at all: cut short
             _line(grading[0]["problem_id"], 1, "hmm</think>0")]  # closed: an answer
    _, _, rows, _ = _grade(tmp_path, card, lines, fake_handle(tasks), 1,
                           provenance={"model": "org/m", "thinking": True})
    assert [(r["score"], r["grader_detail"]) for r in rows] == [
        (0.0, "format_failure"), (1.0, "")]
