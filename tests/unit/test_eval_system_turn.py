"""A set row's optional system turn, carried to the one chat-template renderer."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from reliquary.environment.agentic.types import EpisodeTask
from reliquary.eval import prompt_source as ps
from reliquary.eval.qualify_executor import qualify_lease
from reliquary.eval.qualify_protocol import QualifyLease
from reliquary.validator.corpus_service import ChatTemplatePromptRenderer, SingleTurnPromptJob


def body(*rows) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)


def row(i, messages):
    return {"problem_id": f"s-{i:06d}", "env": "verifiers:x", "set_id": "s", "messages": messages}


SYSTEM_ROW = row(0, [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q0"}])
USER_ROW = row(1, [{"role": "user", "content": "q1"}])


def source_for(data: bytes, count: int) -> ps.EvalSource:
    return ps.eval_source_for("s", data, count)


def test_a_system_and_a_user_turn_are_accepted():
    data = body(SYSTEM_ROW, USER_ROW)
    rows = ps.register_eval_prompts(source_for(data, 2), data)
    assert rows[0]["messages"][0]["role"] == "system"


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
    [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
    [{"role": "system", "content": "a"}],
    [{"role": "system", "content": 3}, {"role": "user", "content": "b"}],
    [{"role": "system", "content": ""}, {"role": "user", "content": "b"}],
])
def test_anything_else_is_refused(messages):
    data = body(row(0, messages))
    with pytest.raises(ValueError, match="is not one user turn"):
        ps.register_eval_prompts(source_for(data, 1), data)


def test_the_environment_hands_the_system_turn_over():
    data = body(SYSTEM_ROW, USER_ROW)
    source = source_for(data, 2)
    environment = ps.EvalSetEnvironment(source, ps.register_eval_prompts(source, data))
    assert environment.get_problem(0)["system"] == "be brief"
    assert environment.get_problem(0)["prompt"] == "q0"
    assert "system" not in environment.get_problem(1)


def job(prompt_source, count=2):
    return SimpleNamespace(job_id="j", prompt_source=prompt_source, prompt_start=0,
                           prompt_count=count, prompt_end=count,
                           owns=lambda i: 0 <= i < count)


def test_an_eval_job_puts_the_system_turn_in_the_task():
    data = body(SYSTEM_ROW, USER_ROW)
    source = source_for(data, 2)
    environment = ps.EvalSetEnvironment(source, ps.register_eval_prompts(source, data))
    prompts = SingleTurnPromptJob(job(source.name), environment)
    assert prompts.task_for(0).metadata == {"system": "be brief"}
    assert prompts.task_for(1).metadata == {}


def test_a_catalog_job_never_takes_a_system_key():
    class Environment:
        def get_problem(self, index):
            return {"prompt": "p", "system": "smuggled"}

    task = SingleTurnPromptJob(job("reliquary_dapo_math_v1"), Environment()).task_for(0)
    assert task.metadata == {}


class Tokenizer:
    chat_template = "x"

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append((messages, kwargs))
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_the_renderer_adds_the_system_turn_only_when_there_is_one():
    tokenizer = Tokenizer()
    renderer = ChatTemplatePromptRenderer(tokenizer, thinking=True)
    plain = EpisodeTask(id="a", prompt="q", tools=())
    system = EpisodeTask(id="b", prompt="q", tools=(), metadata={"system": "be brief"})
    assert renderer.initial_text(plain) == "<user>q<assistant>"
    assert renderer.initial_text(system) == "<system>be brief<user>q<assistant>"
    assert tokenizer.seen[0] == ([{"role": "user", "content": "q"}],
                                 {"tokenize": False, "add_generation_prompt": True,
                                  "enable_thinking": True})


def test_the_qualification_lease_carries_the_system_turn():
    lease = {"protocol": "reliquary.corpus-audit/v1", "type": "qualify", "lease_id": "a" * 32,
             "qualification_id": "order-q1", "model_id": "m", "model_revision": "r" * 40,
             "chunk_tokens": 32, "topk": 128, "expires_at": 1e12,
             "prompts": [{"problem_id": "p0", "text": "q0", "system": "be brief"},
                         {"problem_id": "p1", "text": "q1"}],
             "completions": 2, "sampling": {"temperature": 1.0}, "max_new_tokens": 16,
             "thinking": False}
    QualifyLease.model_validate(lease)
    tokenizer = Tokenizer()

    class Generator:
        def generate_many(self, prompts, ns):
            return [[SimpleNamespace(tokens=[1], proofs=["p"]) for _ in range(n)] for n in ns]

    qualify_lease(lease, tokenizer=tokenizer, generator=Generator(),
                  score=lambda items: [("proof_undecodable", ()) for _ in items],
                  model_info={"gpu_count": 1, "gpu": "H100", "vllm_version": "0.1",
                              "checkpoint_sha256": "d" * 64}, clock=lambda: 0.0)
    assert [m for m, _ in tokenizer.seen] == [
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q0"}],
        [{"role": "user", "content": "q1"}]]


def test_qualification_prompts_from_set_rows():
    from reliquary.eval.qualification import lease_prompt

    assert lease_prompt(SYSTEM_ROW) == {"problem_id": "s-000000", "text": "q0",
                                        "system": "be brief"}
    assert lease_prompt(USER_ROW) == {"problem_id": "s-000001", "text": "q1"}


def test_a_system_row_passes_fidelity_and_a_dropped_system_turn_fails(tmp_path, monkeypatch):
    """The miner renders what the validator expects; a miner leaving the system
    turn out (an easier prompt) is refused."""
    from dataclasses import replace

    from reliquary.eval.sets import build_source_set
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job
    from reliquary.validator.corpus_text import check_prompt_fidelity
    from tests.unit.test_corpus_export import _job_spec
    from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle
    from tests.unit.test_eval_verifiers_source import opener as taskset_opener

    card = build_source_set("verifiers:fake", out=tmp_path / "vset", open_taskset=taskset_opener(
        fake_handle([FakeTask(0, system="be brief"), FakeTask(1)])))
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [ps._from_directory])
    monkeypatch.setenv(ps.SETS_DIR_ENV, str(tmp_path))
    (tmp_path / card["set_id"]).symlink_to(tmp_path / "vset")
    data = (tmp_path / "vset" / "prompts.jsonl").read_bytes()
    source = ps.eval_source_for(card["set_id"], data, 2)
    job = replace(_job_spec(job_id="order-eval-1", prompt_source=source.name, prompt_count=2),
                  renderer_id="chat-template-thinking-v1")
    tokenizer = Tokenizer()
    renderer = renderer_for_job(job, None, tokenizer=tokenizer)
    prompts = prompt_job_for_spec(job)
    miner_text = renderer.initial_text(prompts.task_for(0))
    assert miner_text == "<system>be brief<user>question 0<assistant>"
    assert check_prompt_fidelity(miner_text, job=prompts, prompt_index=0, renderer=renderer).ok
    easier = "<user>question 0<assistant>"
    assert not check_prompt_fidelity(easier, job=prompts, prompt_index=0, renderer=renderer).ok
