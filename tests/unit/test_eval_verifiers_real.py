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
    prompt = rows(tmp_path / "s", "prompts.jsonl")[0]["messages"][-1]["content"]
    answer = handle.task(grading["task_key"]).data.answer
    assert vs.score_answer(handle, grading["task_key"], None, prompt,
                           f"so \\boxed{{{answer}}}") == 1.0
    assert vs.score_answer(handle, grading["task_key"], None, prompt, "\\boxed{1000}") == 0.0


def rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]
