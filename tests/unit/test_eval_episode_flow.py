"""One existing runner/upload/grader flow with deterministic CPU tools, no model."""

import asyncio
import json
import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval import agentic, protocol
from reliquary.eval.grading import grade_evaluation
from reliquary.eval.runner import EvalRunner
from reliquary.eval.sets import build_episode_set
from reliquary.eval.storage import publish_set
from tests.unit.test_eval_agentic import TelecomTools, CALL, FINAL, encode, decode
from tests.unit.test_eval_platform_client import FakePlatform, _task
from tests.unit.test_eval_runner import FakeGenerator


@pytest.mark.parametrize("tampered_accounting", [False, True])
def test_episode_uses_existing_runner_upload_and_authoritative_grader(tmp_path, monkeypatch, tampered_accounting):
    class Environment(TelecomTools):
        def __len__(self):
            return 1
    from reliquary.environment.registry import get_environment_spec
    source = agentic.TelecomBenchmark(Environment(),
        get_environment_spec(agentic.TELECOM_ENVIRONMENT).environment_manifest_sha256)
    monkeypatch.setattr(agentic, "load_telecom_benchmark", lambda expected: source)
    card = build_episode_set(agentic.TELECOM_ENVIRONMENT, out=tmp_path / "set",
        interaction="episode", count=1, seed=7, open_episode=lambda *_: source)
    prompts = [json.loads(line) for line in (tmp_path / "set/prompts.jsonl").read_text().splitlines()]
    body = {"schema": protocol.REQUEST_PROTOCOL,
        "sets": [{"set_id": card["set_id"], "env": card["env"], "problems": 1, "samples": 1}],
        "protocol_bindings": {card["set_id"]: card["protocol_sha256"]},
        "sampling": {"seed": 9, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "thinking": True, "max_new_tokens": 512, "harness": "telecom-episode-v1",
        "max_turns": 40, "max_episode_tokens": 4096, "episode_timeout_seconds": 60,
        "interaction": "episode"}
    task = {**_task(), **{key: body[key] for key in ("sampling", "thinking", "max_new_tokens", "protocol_bindings")},
        "protocol": body, "protocol_json": protocol.canonical(body).decode(), "protocol_sha256": protocol.digest(body),
        "envs": [{"set_id": card["set_id"], "env": card["env"], "samples": 1, "problem_ids_count": 1}],
        "set_protocols": {card["set_id"]: {key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")}}}
    class Generator(FakeGenerator):
        calls = 0
        encode = staticmethod(encode)
        decode = staticmethod(decode)
        def load(self, **kwargs):
            kwargs.pop("max_total_tokens")
            return super().load(**kwargs)
        def begin_episode(self):
            pass
        def generate_turn(self, tokens, **kwargs):
            text = (CALL, FINAL)[self.calls]
            self.calls += 1
            return {"text": text, "token_ids": encode(text), "finish_reason": "stop"}
    platform, generator = FakePlatform(tasks=[task], prompts=prompts), Generator()
    result = EvalRunner(platform.client(), generator, work_dir=tmp_path / "work").run()
    assert generator.loads == 1 and generator.calls == 2 and result["rows"] == 1
    public, subnet = LocalDirectorySink(tmp_path / "p"), LocalDirectorySink(tmp_path / "s")
    asyncio.run(publish_set(tmp_path / "set", platform=None, subnet=subnet))
    for key in result["completion_keys"]:
        uploaded = platform.objects[key]
        if tampered_accounting:
            row = json.loads(uploaded)
            row["completion_tokens"] = 999999
            row["episode"].update(turns=999, tool_calls=999, transcript_tokens=999, stop_reason="invented")
            uploaded = (json.dumps(row) + "\n").encode()
        asyncio.run(public.put_bytes(key, uploaded))
    provenance = {**{key: task[key] for key in ("protocol", "protocol_json", "protocol_sha256",
        "protocol_bindings", "sampling", "thinking", "max_new_tokens")},
        "model": task["model"]["repo"], "revision": task["model"]["revision"], "model_sha": result["model_sha"]}
    asyncio.run(grade_evaluation(eval_id="fixture", set_ids=[card["set_id"]],
        completion_keys=result["completion_keys"], problems_per_set={card["set_id"]: 1},
        samples_per_set={card["set_id"]: 1}, platform=public, subnet=subnet,
        provenance=provenance, require_sandbox=lambda _: None, work_dir=tmp_path / "grade"))
    report = json.loads((tmp_path / "p/evaluations/fixture/report.json").read_text())
    env = report["envs"][card["env"]]
    assert report["complete"] and env["mean_score"] == 1 and env["n_problems_graded"] == 1
    assert env["mean_turns"] == 2 and env["tool_calls_total"] == 1
    assert env["stop_reasons"] == {"finished": 1} and generator.calls == 2
    assert env["mean_completion_tokens"] == len(encode(CALL)) + len(encode(FINAL))
