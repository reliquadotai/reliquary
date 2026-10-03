"""Verifiers tasksets as eval-set sources: freezing, refusals, scoring.

The first half needs no `verifiers`: a fake taskset stands behind the handle.
The second half builds real Verifiers tasks and is skipped without the package.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from reliquary.eval import verifiers_source as vs
from reliquary.eval.sets import build_source_set


class FakeTask:
    def __init__(self, index, prompt=None, system=None):
        self.key = f"key{index:03d}"
        self.data = SimpleNamespace(idx=index, name=f"t{index}", image=None,
                                    system_prompt=system,
                                    prompt=prompt if prompt is not None else f"question {index}",
                                    answer=str(index))

    def hooks(self, attr):
        return [lambda task, trace: 1.0]


def fake_handle(tasks, *, refuse=lambda task: None, scored=None):
    def score(task, frozen, answer):
        if scored is not None:
            scored.append((task.key, frozen, answer))
        return 1.0 if answer.strip() == task.data.answer else 0.0

    return vs.TasksetHandle("fake", {}, tasks, package="fake-pkg", package_version="1.0",
                            verifiers_version="0.3.1", score_trace=score, refuse=refuse)


def opener(handle):
    calls = []

    def open_taskset(name, args):
        calls.append((name, args))
        return handle

    open_taskset.calls = calls
    return open_taskset


def rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_the_source_name():
    assert vs.is_verifiers_source("verifiers:aime26")
    assert not vs.is_verifiers_source("reliquary_dapo_math_v1")
    assert vs.taskset_id("verifiers:aime26") == "aime26"
    with pytest.raises(ValueError):
        vs.taskset_id("verifiers:")


def test_build_a_whole_taskset(tmp_path):
    tasks = [FakeTask(i) for i in range(5)]
    open_taskset = opener(fake_handle(tasks))
    card = build_source_set("verifiers:fake", out=tmp_path / "s", open_taskset=open_taskset,
                            clock=lambda: 2.0)
    assert open_taskset.calls == [("fake", {})]
    assert card["set_id"] == "verifiers-fake-r0-n5"
    assert card["source_kind"] == "verifiers" and card["env"] == "verifiers:fake"
    assert card["split"] is None and card["count"] == 5
    assert card["taskset"] == {"id": "fake", "args": {}, "package": "fake-pkg",
                               "package_version": "1.0", "verifiers_version": "0.3.1"}
    assert card["disjointness"]["external_benchmark"] is True
    prompts, grading = rows(tmp_path / "s", "prompts.jsonl"), rows(tmp_path / "s", "grading.jsonl")
    assert prompts[2]["messages"] == [{"role": "user", "content": "question 2"}]
    assert grading[2] == {"problem_id": "verifiers-fake-r0-n5-000002", "source": "verifiers:fake",
                          "source_index": 2, "task_key": "key002",
                          "prompt_sha256": vs.prompt_digest(None, "question 2")}


def test_a_system_prompt_becomes_the_system_turn(tmp_path):
    tasks = [FakeTask(0, system="You write Python.")]
    build_source_set("verifiers:fake", out=tmp_path / "s", open_taskset=opener(fake_handle(tasks)))
    prompts, grading = rows(tmp_path / "s", "prompts.jsonl"), rows(tmp_path / "s", "grading.jsonl")
    assert prompts[0]["messages"] == [{"role": "system", "content": "You write Python."},
                                      {"role": "user", "content": "question 0"}]
    assert grading[0]["prompt_sha256"] == vs.prompt_digest("You write Python.", "question 0")


def test_taskset_args_are_frozen_and_named(tmp_path):
    handle = fake_handle([FakeTask(0)])
    handle.args = {"diamond": True}
    card = build_source_set("verifiers:fake", out=tmp_path / "s",
                            taskset_args={"diamond": True}, open_taskset=opener(handle))
    assert card["taskset"]["args"] == {"diamond": True}
    assert card["set_id"].startswith("verifiers-fake-a") and card["set_id"].endswith("-r0-n1")


def test_a_range_and_a_sample(tmp_path):
    tasks = [FakeTask(i) for i in range(20)]
    card = build_source_set("verifiers:fake", start=5, count=10, sample=4, seed=9,
                            out=tmp_path / "s", open_taskset=opener(fake_handle(tasks)))
    assert card["set_id"] == "verifiers-fake-r5-n10-k4-s9"
    indices = [g["source_index"] for g in rows(tmp_path / "s", "grading.jsonl")]
    assert indices == random.Random(9).sample(range(5, 15), 4)


def test_a_range_past_the_taskset_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\[3, 8\) runs past 'verifiers:fake' \(5 tasks\)"):
        build_source_set("verifiers:fake", start=3, count=5, out=tmp_path / "s",
                         open_taskset=opener(fake_handle([FakeTask(i) for i in range(5)])))


def test_a_split_is_refused_for_a_taskset(tmp_path):
    with pytest.raises(ValueError, match="no split"):
        build_source_set("verifiers:fake", split="eval", out=tmp_path / "s",
                         open_taskset=opener(fake_handle([FakeTask(0)])))


def test_one_refused_task_refuses_the_set(tmp_path):
    tasks = [FakeTask(i) for i in range(3)]
    handle = fake_handle(tasks, refuse=lambda t: "it gives the model tools" if t.key == "key001"
                         else None)
    with pytest.raises(ValueError, match="task 1 cannot be a set problem: it gives the model tools"):
        build_source_set("verifiers:fake", out=tmp_path / "s", open_taskset=opener(handle))
    assert not (tmp_path / "s" / "set.json").exists()


@pytest.mark.parametrize("prompt", [
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
     {"role": "user", "content": "c"}],
    [{"role": "assistant", "content": "a"}],
])
def test_a_conversation_in_the_prompt_is_refused(prompt):
    with pytest.raises(ValueError, match="not one user turn"):
        vs.freeze(FakeTask(0, prompt=prompt))


def test_a_message_list_prompt_is_read():
    frozen = vs.freeze(FakeTask(0, prompt=[{"role": "system", "content": "s"},
                                           {"role": "user", "content": "u"}]))
    assert (frozen.system, frozen.prompt) == ("s", "u")


def test_an_image_is_refused():
    task = FakeTask(0)
    task.data.image = "x.png"
    with pytest.raises(ValueError, match="image"):
        vs.freeze(task)


def test_score_an_answer():
    scored = []
    handle = fake_handle([FakeTask(i) for i in range(3)], scored=scored)
    frozen = vs.prompt_digest(None, "question 2")
    assert vs.score_answer(handle, "key002", frozen, "2") == 1.0
    assert vs.score_answer(handle, "key002", frozen, "7") == 0.0
    assert scored[0][0] == "key002" and scored[0][2] == "2"


def test_a_drifted_prompt_is_not_scored():
    handle = fake_handle([FakeTask(0)])
    with pytest.raises(LookupError, match="source_drift"):
        vs.score_answer(handle, "key000", vs.prompt_digest(None, "the frozen question"), "0")


def test_a_task_gone_from_the_taskset_is_drift():
    with pytest.raises(LookupError, match="source_drift"):
        vs.score_answer(fake_handle([FakeTask(0)]), "nope", vs.prompt_digest(None, "q"), "0")
