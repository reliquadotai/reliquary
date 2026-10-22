"""The CPU operator owns the remote model process and never requests HF weights."""

import hashlib
import json
import time

import pytest

from reliquary.eval import protocol
from reliquary.eval.runner import CHUNK_PROBLEMS, Completion, EvalRunner, VLLMGenerator
from tests.unit.test_eval_platform_client import FakePlatform, _task
from tests.unit.test_eval_protocol import envelope, request_protocol
from tests.unit.test_eval_signed_policy import build


class OperatorModel:
    def load(self, **kwargs):
        assert kwargs["model_access"] == "private"
        assert "hf_token" not in kwargs
        assert kwargs["max_total_tokens"] == 65536
        return {"vllm_version": "expected:fixture", "gpu": "expected:fixture",
                "model_sha": kwargs["revision"]}

    def generate_turn(self, input_ids, **kwargs):
        assert input_ids == [1] and kwargs["max_new_tokens"] == 1
        return {"token_ids": [2], "finish_reason": "stop", "message": {"content": "done"}}


@pytest.fixture
def signed_task(tmp_path, monkeypatch):
    monkeypatch.setattr("reliquary.eval.verifiers_source.installed_artifacts", lambda _: "9" * 64)
    monkeypatch.setattr(protocol, "renderer_sha256", lambda: "2" * 64)
    card = build(tmp_path)
    body = {**request_protocol(), "harness": "signed-bash-v2", "interaction": "signed_episode",
            "max_turns": 48,
            "sets": [{"set_id": card["set_id"], "env": card["env"], "problems": 1, "samples": 1}],
            "protocol_bindings": {card["set_id"]: card["protocol_sha256"]}}
    task = {**_task(), **envelope(body), "model_access": "private", "envs": body["sets"],
            "set_protocols": {card["set_id"]: card}}
    prompts = [json.loads(line) for line in (tmp_path / "set/prompts.jsonl").read_text().splitlines()]
    return task, prompts


def test_remote_operator_uses_owned_process_without_private_weight_credential(
        tmp_path, monkeypatch, signed_task):
    task, prompts = signed_task
    platform = FakePlatform(tasks=[task], prompts=prompts)
    client = platform.client()

    def no_credential(*args, **kwargs):
        pytest.fail("the CPU operator must not obtain the private checkpoint token")

    monkeypatch.setattr(client, "model_credential", no_credential)
    runner = EvalRunner(client, VLLMGenerator(), work_dir=tmp_path / "run",
                        remote_model_factory=OperatorModel)
    local_plan = hashlib.sha256(json.dumps([
        CHUNK_PROBLEMS, [(row["problem_id"], 1) for row in prompts],
        task["model"], "private", task["protocol_sha256"], task["protocol_bindings"],
        task["sampling"], task["max_new_tokens"], task["thinking"]], sort_keys=True).encode()).hexdigest()
    runner._save_state({"task_id": task["task_id"], "plan": local_plan,
        "chunks": {"completions-00000.jsonl": {"key": "must-not-reuse", "rows": 1}},
        "info": {"vllm_version": "local", "gpu": "local", "model_sha": task["model"]["revision"]},
        "seconds": 0})

    def episode(task, row, sample_index, heartbeat):
        runner._generator.begin_episode()
        result = runner._generator.generate_turn([1], max_new_tokens=1, seed=0,
            sampling={}, absolute_deadline=time.monotonic() + 5)
        return Completion(result["message"]["content"], len(result["token_ids"]), "stop")

    monkeypatch.setattr(runner, "_episode", episode)
    result = runner.run()
    assert result["rows"] == 1 and len(platform.results) == 1
    assert result["gpu"] == "expected:fixture"
    assert json.loads(runner._state_path(task["task_id"]).read_text())["plan"] != local_plan
    assert not runner._generator._process.is_alive()
    assert not any(path.endswith("model-credential") for _, path in platform.calls)


@pytest.mark.parametrize("invalid", ["unfrozen", "changed-policy", "custom-generator"])
def test_remote_operator_rejects_invalid_task_before_starting_a_process(
        tmp_path, monkeypatch, signed_task, invalid):
    task, prompts = signed_task
    if invalid == "unfrozen":
        task = _task()
    elif invalid == "changed-policy":
        task["protocol"]["max_turns"] = 49
    import reliquary.eval.model_process as owned

    class ForbiddenProcess(owned.ModelProcess):
        def __init__(self, *args, **kwargs):
            pytest.fail("must not spawn")

    monkeypatch.setattr(owned, "ModelProcess", ForbiddenProcess)
    platform = FakePlatform(tasks=[task], prompts=prompts)
    generator = object() if invalid == "custom-generator" else VLLMGenerator()
    with pytest.raises(ValueError):
        EvalRunner(platform.client(), generator, work_dir=tmp_path / "run",
                   remote_model_factory=OperatorModel).run()
