"""The translated harnesses end to end (`openhands-xml@1`, `claude-code@1`, `codex@1`): the miner's own loop plays
each against reliquary-sandbox's real gateway (FakeBoxes; the file tools' Python programs really run, in a
temporary directory). Every call reaches the gateway as a canonical `bash` call, the model is shown the harness's
own results, the episode ends the harness's way, and the validator's signed parse admits it under that harness's
renderer only."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_harness import spec as harness_spec  # noqa: E402
from reliquary_harness.spec import BASH_VF_B2E4E81  # noqa: E402
from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402
from reliquary_sandbox_service.task_runtime import ExecResult  # noqa: E402

from reliquary.corpus.trajectory_parse import TrajectoryRefused  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402
from tests.unit.test_harness_variants_e2e import _parse, _renderer  # noqa: E402

pytestmark = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
MARKS = ("# openhands:", "# claude-code:", "# codex:")


def _xml(name, **params):
    return f"<function={name}>\n" + "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in params.items()) + "</function>"


def _tool(name, **params):
    return "<tool_call>\n" + _xml(name, **{k: f"\n{v}\n" for k, v in params.items()}) + "\n</tool_call>"


def _scripts(work):
    target = f"{work}/notes.txt"
    return {
        "openhands-xml": (
            ["Write it.\n</think>\n\n" + _xml("execute_bash", command="write /work/answer.txt 42") + "<|im_end|>",
             "Create notes.\n</think>\n\n" + _xml("str_replace_editor", command="create", path=target,
                                                    file_text="hello") + "<|im_end|>",
             "No call.\n</think>\n\nI think I am done.<|im_end|>",
             "Finish.\n</think>\n\n" + _xml("finish", message="Done.") + "<|im_end|>"],
            ["EXECUTION RESULT of [execute_bash]:", f"File created successfully at: {target}",
             "Please continue working on the task"]),
        "claude-code": (
            ["Write it.\n</think>\n\n" + _tool("Bash", command="write /work/answer.txt 42") + "<|im_end|>",
             "Notes.\n</think>\n\n" + _tool("Write", file_path=target, content="hello") + "\n"
             + _tool("Read", file_path=target) + "<|im_end|>",
             "Done.\n</think>\n\nI wrote the answer and the notes.<|im_end|>"],
            [f"File created successfully at: {target}", "1\thello"]),
        "codex": (
            ["Write it.\n</think>\n\n" + _tool("exec_command", cmd="write /work/answer.txt 42") + "<|im_end|>",
             "Notes.\n</think>\n\n" + _tool("apply_patch", input=f"*** Begin Patch\n*** Add File: {target}\n+hello\n"
                                                                  "*** End Patch") + "<|im_end|>",
             "Done.\n</think>\n\nI wrote the answer and the notes.<|im_end|>"],
            ["Process exited with code 0\nOutput:", "Success. Updated the following files:"]),
    }


def _spec(key):
    return next(spec for spec in harness_spec.SPECS.values() if spec.id == key)


async def _play(tmp_path, key):
    spec = _spec(key)
    work = tmp_path / "box"
    work.mkdir()
    turns, seen_texts = _scripts(work)[key]
    directory = next(iter(TOKENIZERS.values()))
    tokenizer = parity._tokenizer(directory)
    renderer = _renderer(spec, 0)
    engine = parity.FakeEngine(tokenizer, turns, renderer.stop_ids, 1 << 20)
    with testing.fake_gateway(tmp_path / "gw") as (gateway, boxes), serve_app(engine.app) as engine_url:
        scripted = boxes.bash

        def bash(box_id, call_id, command, **kw):
            if command.startswith(MARKS):        # a file tool's program: really run, in the temporary directory
                done = subprocess.run(["bash", "-c", command], capture_output=True, cwd=work)
                return ExecResult(done.stdout, done.stderr, done.returncode, False, False)
            return scripted(box_id, call_id, command, **kw)

        boxes.bash = bash
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets())
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=directory,
                                        generate_url=engine_url, sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                        max_tokens=512, max_turns=8, spec=spec) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT)
    return spec, result, engine, tokenizer, seen_texts, work


@pytest.mark.parametrize("key", ["openhands-xml", "claude-code", "codex"])
def test_a_translated_harness_runs_on_canonical_bash_and_is_admitted_under_itself_only(tmp_path, key):
    spec, result, engine, tokenizer, seen_texts, work = asyncio.run(_play(tmp_path, key))
    assert result.error is None and result.trace.stop_condition == "agent_completed", result.error
    assert result.final["body"]["status"] == "graded" and result.final["body"]["reward"] == 1.0
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    assert {c["tool"] for c in calls} == {"bash"}                       # the gateway only ever ran canonical bash
    assert calls[0]["arguments"]["command"] == "write /work/answer.txt 42"
    assert any(c["arguments"]["command"].startswith(MARKS) for c in calls)
    assert (work / "notes.txt").read_text() in ("hello", "hello\n")     # the file tool really wrote it
    shown = "".join(tokenizer.decode(b["token_ids"][len(a["token_ids"]):], skip_special_tokens=False)
                    for a, b in zip(engine.requests, engine.requests[1:]))
    for text in seen_texts:
        assert text in shown, (text, shown[-2000:])
    renderer = _renderer(spec, 0)
    assert engine.requests[0]["token_ids"] == renderer.initial_ids(parity.PROMPT)
    _parse(renderer, result, engine)                                     # the validator admits it under its harness
    with pytest.raises(TrajectoryRefused):
        _parse(_renderer(BASH_VF_B2E4E81, 0), result, engine)
