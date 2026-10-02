"""Real Verifiers tasks as eval-set sources; skipped without `verifiers`."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.eval import verifiers_source as vs
from reliquary.eval.sets import build_source_set


vf = pytest.importorskip("verifiers.v1")


class TinyData(vf.TaskData):
    answer: str


class TinyConfig(vf.TaskConfig):
    pass


class TinyTask(vf.Task[TinyData, vf.State, TinyConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 1.0 if trace.last_reply.strip().endswith(self.data.answer) else 0.0


class JudgedConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig()


class JudgedTask(vf.Task[TinyData, vf.State, JudgedConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 0.0


class RuntimeTask(vf.Task[TinyData, vf.State, TinyConfig]):
    @vf.reward(weight=1.0)
    async def passed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return 1.0


async def deterministic(task, trace) -> float:
    return 1.0


def test_a_real_task_is_frozen_and_scored():
    task = TinyTask(TinyData(idx=0, prompt="2+2?", answer="4"), TinyConfig())
    assert vs.refusal(task) is None
    frozen = vs.freeze(task)
    assert (frozen.prompt, frozen.system) == ("2+2?", None)
    assert vs.score_trace(task, frozen, "it is 4") == 1.0
    assert vs.score_trace(task, frozen, "it is 5") == 0.0
    assert vs.needs_runtime(task) is False


def test_a_real_system_prompt_reaches_the_trace():
    task = TinyTask(TinyData(idx=0, prompt="q", system_prompt="be brief", answer="4"),
                    TinyConfig())
    trace = vs._trace_for(task, vs.freeze(task), "4")
    assert [m.role for m in trace.messages] == ["system", "user", "assistant"]
    assert trace.last_reply == "4"


def test_a_judged_reward_is_refused_until_replaced():
    task = JudgedTask(TinyData(idx=0, prompt="q", answer="a"), JudgedConfig())
    assert "may call the LLM judge" in vs.refusal(task)
    replaced = JudgedConfig(rewards={"correct": {"fn": f"{__name__}:deterministic"}})
    assert vs.refusal(JudgedTask(TinyData(idx=0, prompt="q", answer="a"), replaced)) is None


def test_a_reward_needing_a_runtime_is_known():
    assert vs.needs_runtime(RuntimeTask(TinyData(idx=0, prompt="q", answer="a"), TinyConfig()))


def test_the_gpqa_reward_is_deterministic():
    pytest.importorskip("gpqa")
    from reliquary.eval.verifiers_rewards import gpqa_letter

    task = SimpleNamespace(answer="C")
    assert asyncio.run(gpqa_letter(task, SimpleNamespace(last_reply="The answer is (C)."))) == 1.0
    assert asyncio.run(gpqa_letter(task, SimpleNamespace(last_reply="no idea"))) == 0.0


def test_aime26_as_a_set(tmp_path):
    pytest.importorskip("aime26")
    card = build_source_set("verifiers:aime26", count=3, out=tmp_path / "s")
    assert card["count"] == 3 and card["taskset"]["package"] == "aime26"
    assert card["needs_runtime"] is False
    handle = vs.open_taskset("aime26", {})
    grading = rows(tmp_path / "s", "grading.jsonl")[0]
    answer = handle.task(grading["task_key"]).data.answer
    frozen = grading["prompt_sha256"]
    assert vs.score_answer(handle, grading["task_key"], frozen, f"so \\boxed{{{answer}}}") == 1.0
    assert vs.score_answer(handle, grading["task_key"], frozen, "\\boxed{1000}") == 0.0


def rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_aime26_graded_end_to_end(tmp_path):
    pytest.importorskip("aime26")
    import pyarrow.parquet as pq

    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval.grading import grade_evaluation
    from reliquary.eval.storage import publish_set

    card = build_source_set("verifiers:aime26", count=2, out=tmp_path / "s")
    asyncio.run(publish_set(tmp_path / "s", platform=LocalDirectorySink(tmp_path / "platform"),
                            subnet=LocalDirectorySink(tmp_path / "subnet")))
    handle = vs.open_taskset("aime26", {})
    grading = rows(tmp_path / "s", "grading.jsonl")
    answers = [handle.task(g["task_key"]).data.answer for g in grading]
    lines = [
        {"problem_id": grading[0]["problem_id"], "sample_index": 0,
         "completion": f"<think>maybe \\boxed{{1}}</think>So \\boxed{{{answers[0]}}}"},
        {"problem_id": grading[1]["problem_id"], "sample_index": 0,
         "completion": f"<think>it is \\boxed{{{answers[1]}}}"},  # never closed
    ]
    platform = tmp_path / "platform" / "evaluations" / "order-e1"
    platform.mkdir(parents=True)
    (platform / "c.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    asyncio.run(grade_evaluation(
        eval_id="order-e1", platform=LocalDirectorySink(tmp_path / "platform"),
        subnet=LocalDirectorySink(tmp_path / "subnet"), work_dir=tmp_path / "w",
        provenance={"model": "m"}, set_ids=[card["set_id"]],
        completion_keys=["evaluations/order-e1/c.jsonl"],
        problems_per_set={card["set_id"]: 2}, samples_per_set={card["set_id"]: 1}))
    graded = sorted(pq.read_table(platform / "graded.parquet").to_pylist(),
                    key=lambda r: r["problem_id"])
    assert [(r["score"], r["grader_detail"]) for r in graded] == [
        (1.0, ""), (0.0, "format_failure")]
