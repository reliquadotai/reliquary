"""The signed miner's episode loop, recorded at the generate endpoint.

A scenario is a scripted model (completion texts, one per turn, tokenized with the real
Qwen tokenizer) served by a fake `/inference/v1/generate` engine, played against
reliquary-sandbox's real gateway over FakeBoxes. What a scenario records is everything
the validator and the engine's session log can see of the loop: each generate request
(prompt ids, sampling params, model), whether every turn named one session, the stop,
the gateway's call records, the final status and the router's refusals.

`tests/fixtures/harness_goldens/miner-loop.<tokenizer>.json` holds what verifiers b2e4e81's
bash harness (reliquary_sandbox_verifiers.SandboxEpisodeRunner under verifiers' train
client, as `SignedSweEpisodeRunner` built it on feat/rl-async a5df544f) produced for
every scenario. Written with

    RUN_VERIFIERS_BRIDGE=1 python -m tests.unit.harness_episode_parity --write TOKENIZER_DIR

(needs verifiers and uv with network: verifiers prepares its program with uv)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

from tests.unit.harness_goldens import FIXTURES, dump, tokenizer_digest

MODEL = "policy"
MAX_TOKENS = 512
TEMPERATURE, TOP_P = 0.7, 0.95
PROMPT = "Write 42 to /work/answer.txt, then check it."


def _call(name: str, **params: str) -> str:
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _turn(reasoning: str, *calls: str, content: str = "") -> str:
    head = f"{reasoning}\n</think>\n\n" + (f"{content}\n\n" if content and calls else content)
    return head + "\n".join(calls) + "<|im_end|>"


SCENARIOS: dict[str, dict[str, Any]] = {
    "honest": dict(env="fake-env", max_turns=6, turns=[
        _turn("Write it, say hi, and look it up.", _call("bash", command="write /work/answer.txt 42"),
              _call("bash", command="echo hi"), _call("search", query="answer 42")),
        _turn("Check the file, then fix its trailing text.", _call("bash", command="cat /work/answer.txt"),
              _call("edit", path="/work/answer.txt", old_str="42", new_str="42"), content="Checking."),
        _turn("It is there.", content="The answer is written."),
    ]),
    "max-turns": dict(env="fake-env", max_turns=2, turns=[
        _turn("One.", _call("bash", command="echo one")),
        _turn("Two.", _call("bash", command="write /work/answer.txt 42"), _call("bash", command="echo two")),
        _turn("Never asked.", content="unreachable"),
    ]),
    "episode-closed": dict(env="fake-env", max_turns=6, budgets={"cpu_s": 1}, turns=[
        _turn("Burn the budget.", _call("bash", command="echo hi"), _call("bash", command="burn 1500"),
              _call("bash", command="echo never")),
        _turn("Never asked.", content="unreachable"),
    ]),
    "context-length": dict(env="fake-env", max_turns=6, room_after_first=8, turns=[
        _turn("Write it.", _call("bash", command="write /work/answer.txt 42")),
        _turn("Never asked.", content="unreachable"),
    ]),
    "capped": dict(env="fake-env", max_turns=6, turns=[
        _turn("Write it.", _call("bash", command="write /work/answer.txt 42")),
        "I am still thinking about whether the file needs a trailing newline, and",
    ]),
    "bash-only": dict(env="fake-bash", max_turns=6, turns=[
        _turn("Edit is not mine here.", _call("edit", path="/work/answer.txt", old_str="a", new_str="b")),
        _turn("Bash then.", _call("bash", command="write /work/answer.txt 42")),
        _turn("Done.", content="Written."),
    ]),
}


def bash_only_task(split: str, index: int):
    """The fake env, offering bash only (TMax's tool set)."""
    from reliquary_sandbox_service.episodes.testing import fake_sandbox_task

    return dataclasses.replace(fake_sandbox_task(split, index), tools=("bash",))


class FakeEngine:
    """`/inference/v1/generate` answering each session's n-th turn with the script's n-th
    completion (encoded with the tokenizer), and `/v1/models` naming `max_model_len`."""

    def __init__(self, tokenizer, turns: list[str], stop_ids, max_model_len: int) -> None:
        self.requests: list[dict] = []
        self.sessions: list[str | None] = []
        self.completions: list[list[int]] = []
        self.max_model_len = max_model_len
        self.app = app = FastAPI()
        encoded = [[int(t) for t in tokenizer.encode(text, add_special_tokens=False)] for text in turns]
        stops = frozenset(int(t) for t in stop_ids)

        @app.post("/inference/v1/generate")
        async def generate(request: Request) -> dict:
            body = await request.json()
            self.requests.append(body)
            self.sessions.append(request.headers.get("X-Session-ID"))
            ids = encoded[min(len(self.requests) - 1, len(encoded) - 1)]
            self.completions.append(ids)
            return {"request_id": uuid.uuid4().hex, "choices": [{
                "index": 0, "token_ids": ids,
                "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": -0.5} for t in ids]},
                "finish_reason": "stop" if ids and ids[-1] in stops else "length"}]}

        @app.get("/v1/models")
        async def models() -> dict:
            return {"object": "list", "data": [{"id": MODEL, "object": "model",
                                                "max_model_len": self.max_model_len}]}


def _budgets(**overrides):
    from reliquary_sandbox.attest import Budgets

    values = dict(max_calls=8, per_call_timeout_s=30, cpu_s=600, wall_s=3600,
                  memory_bytes=2 * 1024**3, pids=256, disk_bytes=1024**3)
    values.update(overrides)
    return Budgets(**values)


@asynccontextmanager
async def verifiers_runner(*, gateway_url: str, engine_url: str, tokenizer_dir: str, max_turns: int):
    """verifiers' bash harness, built as `SignedSweEpisodeRunner` built it (a5df544f)."""
    from renderers.configs import Qwen38RendererConfig
    from reliquary_sandbox_verifiers import SandboxEpisodeRunner
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import TrainClientConfig
    from verifiers.v1.types import Sampling

    os.environ.setdefault("RELIQUARY_GENERATE_KEY", "local")
    ctx = ModelContext(model=MODEL,
                       client=TrainClientConfig(base_url=f"{engine_url.rstrip('/')}/v1",
                                                api_key_var="RELIQUARY_GENERATE_KEY",
                                                renderer=Qwen38RendererConfig(),
                                                renderer_model_name=tokenizer_dir),
                       sampling=Sampling(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS))
    async with SandboxEpisodeRunner(ctx=ctx, gateway_url=gateway_url, max_turns=max_turns) as runner:
        yield runner


@asynccontextmanager
async def harness_runner(*, gateway_url: str, engine_url: str, tokenizer_dir: str, max_turns: int):
    """The miner's own loop (reliquary-harness), as `SignedSweEpisodeRunner` builds it."""
    from types import SimpleNamespace

    from reliquary.miner.harness_episode import HarnessEpisodeRunner

    async with HarnessEpisodeRunner(gateway_url=gateway_url, model=MODEL, renderer_dir=tokenizer_dir,
                                    generate_url=engine_url,
                                    sampling=SimpleNamespace(temperature=TEMPERATURE, top_p=TOP_P),
                                    max_tokens=MAX_TOKENS, max_turns=max_turns) as runner:
        yield runner


def _tokenizer(tokenizer_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir)


def _initial_length(tokenizer_dir: str, tools) -> int:
    from reliquary.environment.agentic_swe import load_turn_renderer

    return len(load_turn_renderer(tokenizer_dir, tools=tools).initial_ids(PROMPT))


async def play(name: str, tokenizer_dir: str, workdir: Path, runner_factory=verifiers_runner) -> dict:
    """One scenario through `runner_factory` (an async context manager taking
    gateway_url, engine_url, tokenizer_dir and max_turns, yielding a runner with
    `run(token=, prompt=, on_trace=)`); what the loop showed the outside world."""
    from reliquary_sandbox_service.episodes import testing
    from reliquary_sandbox_service.episodes.local_gateway import serve_app

    scenario = SCENARIOS[name]
    tokenizer = _tokenizer(tokenizer_dir)
    tools = ("bash",) if scenario["env"] == "fake-bash" else ("bash", "edit")
    from reliquary.environment.agentic_swe import load_turn_renderer

    renderer = load_turn_renderer(tokenizer_dir, tools=tools)
    max_model_len = 1 << 20
    if "room_after_first" in scenario:
        first = tokenizer.encode(scenario["turns"][0], add_special_tokens=False)
        max_model_len = len(renderer.initial_ids(PROMPT)) + len(first) + scenario["room_after_first"]
    engine = FakeEngine(tokenizer, scenario["turns"], renderer.stop_ids, max_model_len)
    envs = {"fake-env": testing.FAKE_ENV_ENTRY, "fake-bash": "tests.unit.harness_episode_parity:bash_only_task"}
    seen: list[str] = []
    with testing.fake_gateway(workdir, envs=envs) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env=scenario["env"], split="train", index=3, image=testing.FAKE_IMAGE,
                              budgets=_budgets(**scenario.get("budgets", {})))
        async with runner_factory(gateway_url=gateway.url, engine_url=engine_url, tokenizer_dir=tokenizer_dir,
                                  max_turns=scenario["max_turns"]) as runner:
            result = await runner.run(token=token, prompt=PROMPT, on_trace=lambda trace: seen.append(trace.id))
        verified = gateway.verify(result.transcript, require_graded=False)
    records = [record["body"] for record in result.transcript["records"]]
    sessions = set(engine.sessions)
    return {
        "scenario": name, "tokenizer": tokenizer_digest(tokenizer_dir),
        "requests": [{"token_ids": body["token_ids"], "sampling_params": body["sampling_params"],
                      "model": body.get("model")} for body in engine.requests],
        "completions": engine.completions,
        # One session id on every turn, and the one `on_trace` reported before the first turn.
        "one_session": len(sessions) == 1 and None not in sessions and seen == list(sessions),
        "stop": getattr(result.trace, "stop_condition", None),
        "error": result.error is not None,
        "final": result.final["body"]["status"],
        "transcript_verified": verified.ok,
        "tools": records[0]["tools"],
        "calls": [[body["turn"], body["k"], body["tool"], body["arguments"]] for body in records
                  if body["kind"] == "call"],
        "refusals": [list(refusal) for refusal in result.refusals],
    }


def fixture_path(tokenizer: str) -> Path:
    return FIXTURES / f"miner-loop.{tokenizer}.json"


def record(tokenizer_dir: str, workdir: Path, runner_factory=verifiers_runner) -> dict:
    async def all_scenarios():
        return {name: await play(name, tokenizer_dir, workdir / name, runner_factory) for name in SCENARIOS}

    return asyncio.run(all_scenarios())


def main(argv: list[str]) -> int:
    import tempfile

    if len(argv) < 2 or argv[0] != "--write":
        print(__doc__)
        return 2
    for directory in argv[1:]:
        with tempfile.TemporaryDirectory() as workdir:
            recorded = record(directory, Path(workdir))
        path = fixture_path(tokenizer_digest(directory))
        path.write_text(dump(recorded), encoding="utf-8")
        print(path, {name: (row["stop"], row["final"], len(row["requests"])) for name, row in recorded.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
