"""`reliquary eval run` end to end, against the fake platform and a fake generator."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from reliquary.eval.platform_client import LeaseLost, PlatformError
from reliquary.eval.runner import (
    Completion,
    EvalRunner,
    normalized_sampling,
    problem_seed,
    render_prompt,
    resolved_revision,
)
from tests.unit.test_eval_platform_client import FakePlatform, _task


class FakeGenerator:
    def __init__(self, *, fail=False) -> None:
        self.loads, self.generated, self.fail = 0, [], fail

    def load(self, *, repo, revision, gpu_count, max_new_tokens, rows, thinking):
        self.loads += 1
        self.load_rows = len(rows)
        return {"vllm_version": "0.0-fake", "gpu": f"{gpu_count}x FakeGPU", "model_sha": revision}

    def render(self, row, *, thinking):
        return row["messages"][0]["content"] + ("/think" if thinking else "")

    def generate(self, prompts, *, samples, seeds, sampling, max_new_tokens):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.generated.append(list(prompts))
        return [[Completion(text=f"{p}|{seed}|{j}", tokens=3 + j,
                            finish_reason="length" if j == 1 else "stop") for j in range(n)]
                for p, n, seed in zip(prompts, samples, seeds)]


def _prompts():
    rows = [{"problem_id": f"logic-eval-s1-n4-{i:06d}", "env": "logic",
             "set_id": "logic-eval-s1-n4", "samples": 2,
             "messages": [{"role": "user", "content": f"logic {i}"}]} for i in range(3)]
    rows += [{"problem_id": f"math-eval-s1-n9-{i:06d}", "env": "math",
              "set_id": "math-eval-s1-n9", "samples": 4,
              "messages": [{"role": "user", "content": f"math {i}"}]} for i in range(2)]
    return rows


def _platform(**kw):
    return FakePlatform(tasks=[_task()], prompts=_prompts(), part_size=100, **kw)


def _lines(platform, keys):
    return [json.loads(line) for key in keys for line in platform.objects[key].splitlines()]


def test_the_runner_generates_uploads_and_reports(tmp_path):
    platform, generator = _platform(), FakeGenerator()
    result = EvalRunner(platform.client(), generator, work_dir=tmp_path,
                        chunk_problems=2).run()
    assert generator.loads == 1
    assert result["rows"] == 3 * 2 + 2 * 4
    assert result["completion_keys"] == [f"evaluations/ev-1/completions-{i:05d}.jsonl"
                                         for i in range(3)]
    assert platform.results == [result]
    assert result["vllm_version"] == "0.0-fake" and result["model_sha"] == "a" * 40
    lines = _lines(platform, result["completion_keys"])
    assert len(lines) == 14
    assert set(lines[0]) == {"problem_id", "sample_index", "completion", "completion_tokens",
                             "finish_reason"}
    assert [(l["problem_id"][-1], l["sample_index"]) for l in lines[:2]] == [("0", 0), ("0", 1)]
    kinds = [e["kind"] for e in platform.events]
    assert kinds == ["model_loaded", "progress", "progress", "progress"]
    assert platform.events[-1]["detail"] == {"done": 5, "total": 5}


def test_no_task_means_nothing_runs(tmp_path):
    generator = FakeGenerator()
    assert EvalRunner(FakePlatform().client(), generator, work_dir=tmp_path).run() is None
    assert generator.loads == 0


def test_a_crash_resumes_from_the_uploaded_chunks(tmp_path):
    reference = _platform()
    clean = EvalRunner(reference.client(), FakeGenerator(), work_dir=tmp_path / "ref",
                       chunk_problems=2).run()
    parts = [-(-len(reference.objects[k]) // 100) for k in clean["completion_keys"]]
    assert parts[1] >= 2
    platform, generator = _platform(), FakeGenerator()
    # Die after chunk 0 and one part of chunk 1.
    platform.fail_puts_after = parts[0] + 1
    with pytest.raises(PlatformError):
        EvalRunner(platform.client(retries=0), generator, work_dir=tmp_path / "run",
                   chunk_problems=2).run()
    assert platform.results == []
    generated_before, puts_before = len(generator.generated), platform.puts
    assert generated_before == 2
    platform.fail_puts_after = None
    platform.tasks = [platform.claimed]  # the platform offers the task again
    second = FakeGenerator()
    result = EvalRunner(platform.client(), second, work_dir=tmp_path / "run",
                        chunk_problems=2).run()
    # Chunk 0 not regenerated nor re-sent; chunk 1 sent as written; chunk 2 new.
    assert len(second.generated) == 1 and second.generated[0][0].startswith("math 1")
    assert len(_lines(platform, result["completion_keys"])) == 14
    sizes = [len(platform.objects[k]) for k in result["completion_keys"]]
    assert platform.puts == sum(-(-s // 100) for s in sizes)
    assert puts_before == parts[0] + 1


def test_a_refused_lease_stops_the_work(tmp_path):
    platform, generator = _platform(), FakeGenerator()
    platform.heartbeat_status = 409
    runner = EvalRunner(platform.client(), generator, work_dir=tmp_path, chunk_problems=1,
                        heartbeat_seconds=0.01)
    original = generator.generate

    def slow(*args, **kwargs):
        import time

        time.sleep(0.05)
        return original(*args, **kwargs)

    generator.generate = slow
    with pytest.raises(LeaseLost):
        runner.run()
    assert platform.results == []
    assert len(generator.generated) < 5


def test_a_generator_failure_is_reported(tmp_path):
    platform = _platform()
    with pytest.raises(RuntimeError):
        EvalRunner(platform.client(), FakeGenerator(fail=True), work_dir=tmp_path).run()
    assert platform.events[-1]["kind"] == "error"
    assert "out of memory" in platform.events[-1]["detail"]["message"]


def test_seeds_do_not_depend_on_chunking(tmp_path):
    a, b = _platform(), _platform()
    ra = EvalRunner(a.client(), FakeGenerator(), work_dir=tmp_path / "a", chunk_problems=1).run()
    rb = EvalRunner(b.client(), FakeGenerator(), work_dir=tmp_path / "b", chunk_problems=5).run()
    assert _lines(a, ra["completion_keys"]) == _lines(b, rb["completion_keys"])
    assert problem_seed(7, "p") == problem_seed(7, "p") != problem_seed(8, "p")


class LlamaStyleTokenizer:
    """A Llama-3 style tokenizer: BOS id 1, written as ``<s>`` by the chat
    template itself, and prepended again by ``add_special_tokens=True``."""

    BOS = 1

    def __init__(self, template="llama"):
        self.chat_template, self.calls = template, []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return f"<s>[INST] {messages[0]['content']} [/INST]"

    def __call__(self, text, add_special_tokens=True):
        ids = [self.BOS] if add_special_tokens else []
        rest = text
        while rest:
            if rest.startswith("<s>"):
                ids.append(self.BOS)
                rest = rest[3:]
            else:
                ids.append(100 + ord(rest[0]))
                rest = rest[1:]
        return {"input_ids": ids}


def test_rendering_a_chat_template_never_doubles_the_bos():
    row = {"problem_id": "p", "messages": [{"role": "user", "content": "q"}]}
    tokenizer = LlamaStyleTokenizer()
    ids = render_prompt(tokenizer, row, thinking=True)
    assert ids.count(LlamaStyleTokenizer.BOS) == 1 and ids[0] == LlamaStyleTokenizer.BOS
    assert tokenizer.calls == [{"tokenize": False, "add_generation_prompt": True,
                                "enable_thinking": True}]


def test_raw_text_gets_its_special_tokens():
    plain = LlamaStyleTokenizer(template=None)
    ids = render_prompt(plain, {"problem_id": "p", "text": "raw"}, thinking=False)
    assert ids == [1, 100 + ord("r"), 100 + ord("a"), 100 + ord("w")]
    messages = {"problem_id": "p", "messages": [{"role": "user", "content": "q"}]}
    assert render_prompt(plain, messages, thinking=False) == [1, 100 + ord("q")]


def test_sampling_defaults_fill_absent_and_null_fields():
    assert normalized_sampling({"temperature": 0.6, "seed": None}) == {
        "temperature": 0.6, "top_p": 1.0, "top_k": 0, "seed": 0}
    assert normalized_sampling(None)["temperature"] == 1.0


def test_the_model_sha_is_the_commit_hugging_face_served():
    sha = "ab" * 20
    assert resolved_revision(f"/cache/models--o--m/snapshots/{sha}", sha) == sha
    assert resolved_revision(f"/cache/snapshots/{sha}", "abab") == sha
    with pytest.raises(RuntimeError, match="served"):
        resolved_revision(f"/cache/snapshots/{sha}", "cd" * 20)
    with pytest.raises(RuntimeError):
        resolved_revision("/cache/snapshots/main", "main")


def test_a_task_with_a_null_seed_runs(tmp_path):
    task = _task()
    task["sampling"]["seed"] = None
    platform = FakePlatform(tasks=[task], prompts=_prompts(), part_size=1000)
    result = EvalRunner(platform.client(), FakeGenerator(), work_dir=tmp_path).run()
    assert result["rows"] == 14


def test_the_runner_imports_without_vllm():
    code = ("import sys, reliquary.eval.runner, reliquary.eval.grading; "
            "assert 'vllm' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
