"""mini-swe@2 end to end: the miner's own loop plays mini-swe-agent's official SWE-bench harness against
reliquary-sandbox's real gateway (FakeBoxes): its bash tool calls run as canonical bash calls (several a turn), a
turn with no call or a call to another tool is answered with mini's format error and the episode goes on, the
submit command ends it graded; the validator's signed parse admits it under mini's renderer only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_harness import mini  # noqa: E402
from reliquary_harness.spec import BASH_VF_B2E4E81, MINI_SWE_V2  # noqa: E402
from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.corpus.trajectory_parse import TrajectoryRefused  # noqa: E402
from reliquary.environment import agentic_swe  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.harness_tokenizers import TOKENIZERS  # noqa: E402
from tests.unit.test_harness_variants_e2e import _parse, _renderer  # noqa: E402

pytestmark = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


def _call(name, **params):
    return parity._call(name, **params)


SCRIPT = [
    parity._turn("Write it and read it back.", _call("bash", command="write /work/answer.txt 42"),
                 _call("bash", command="cat /work/answer.txt"), content="THOUGHT: write the answer, then check it."),
    parity._turn("Thinking.", content="I believe this is done."),
    parity._turn("Read a file.", _call("cat", path="/work/answer.txt")),
    parity._turn("Done.", _call("bash", command=mini.SUBMIT + " && cat patch.txt"), content="THOUGHT: submit."),
]


async def _play(tmp_path):
    directory = next(iter(TOKENIZERS.values()))
    tokenizer = parity._tokenizer(directory)
    renderer = _renderer(MINI_SWE_V2, 0)
    engine = parity.FakeEngine(tokenizer, SCRIPT, renderer.stop_ids, 1 << 20)
    with testing.fake_gateway(tmp_path) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets())
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=directory,
                                        generate_url=engine_url, sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                        max_tokens=256, max_turns=8, spec=MINI_SWE_V2) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT)
    return result, engine, tokenizer


def test_a_mini_episode_runs_graded_and_is_admitted_under_mini_only(tmp_path):
    result, engine, tokenizer = asyncio.run(_play(tmp_path))
    assert result.error is None and result.trace.stop_condition == "agent_completed"
    assert result.final["body"]["status"] == "graded" and result.final["body"]["reward"] == 1.0
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    assert [(c["tool"], c["arguments"]["command"]) for c in calls] == \
        [("bash", "write /work/answer.txt 42"), ("bash", "cat /work/answer.txt")]   # two calls in one turn, both run
    # The turn without a call and the turn calling `cat` are answered by mini's format error, nothing runs.
    assert result.refusals == ((1, 0, "not_offered"), (2, 0, "not_offered"))
    seen = [tokenizer.decode(b["token_ids"][len(a["token_ids"]):], skip_special_tokens=False)
            for a, b in zip(engine.requests, engine.requests[1:])]
    assert seen[0].count("<tool_response>") == 2 and "<returncode>0</returncode>\n<output>\n42\n</output>" in seen[0]
    assert "<|im_start|>user\nTool call error:" in seen[1] and "No tool calls found" in seen[1]
    assert "<tool_response>" not in seen[1] and "Unknown tool 'cat'" in seen[2]
    parsed = _parse(_renderer(MINI_SWE_V2, 0), result, engine)
    assert [a.tool for a in parsed.actions] == ["bash", "bash", "?format", "?format"]
    with pytest.raises(TrajectoryRefused):
        _parse(_renderer(BASH_VF_B2E4E81, 0), result, engine)


def test_any_task_can_play_mini():
    pin = {"id": MINI_SWE_V2.id, "version": MINI_SWE_V2.version, "digest": MINI_SWE_V2.digest}
    assert agentic_swe.harness_pin_refusal(pin, variants=False) is None     # corpus, eval, RL window
    assert agentic_swe.harness_pin_refusal(pin, variants=True) is None      # RL lease
