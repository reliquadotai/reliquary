"""mini-swe@1 end to end: the miner's own loop plays mini-swe-agent's text harness against
reliquary-sandbox's real gateway (FakeBoxes): fenced commands run as canonical bash calls, a
badly formatted turn is answered with mini's error and the episode goes on, the submit command
ends it graded; the validator's signed parse admits it under mini's renderer only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_harness import mini  # noqa: E402
from reliquary_harness.spec import BASH_VF_B2E4E81, MINI_SWE_V1  # noqa: E402
from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.corpus.trajectory_parse import TrajectoryRefused  # noqa: E402
from reliquary.environment import agentic_swe  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402
from tests.unit.test_harness_variants_e2e import _parse, _renderer  # noqa: E402

pytestmark = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


def _block(command):
    return f"```mswea_bash_command\n{command}\n```"


SCRIPT = [
    "Write the answer.\n</think>\n\nTHOUGHT: write it.\n\n" + _block("write /work/answer.txt 42") + "<|im_end|>",
    "Two at once.\n</think>\n\n" + _block("echo a") + "\n" + _block("echo b") + "<|im_end|>",
    "Check it.\n</think>\n\nTHOUGHT: read it back.\n\n" + _block("cat /work/answer.txt") + "<|im_end|>",
    "Done.\n</think>\n\nTHOUGHT: submit.\n\n" + _block(mini.SUBMIT) + "<|im_end|>",
]


async def _play(tmp_path):
    directory = next(iter(TOKENIZERS.values()))
    tokenizer = parity._tokenizer(directory)
    renderer = _renderer(MINI_SWE_V1, 0)
    engine = parity.FakeEngine(tokenizer, SCRIPT, renderer.stop_ids, 1 << 20)
    with testing.fake_gateway(tmp_path) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets())
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=directory,
                                        generate_url=engine_url, sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                        max_tokens=256, max_turns=8, spec=MINI_SWE_V1) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT)
    return result, engine, tokenizer


def test_a_mini_episode_runs_graded_and_is_admitted_under_mini_only(tmp_path):
    result, engine, tokenizer = asyncio.run(_play(tmp_path))
    assert result.error is None and result.trace.stop_condition == "agent_completed"
    assert result.final["body"]["status"] == "graded" and result.final["body"]["reward"] == 1.0
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    assert [(c["tool"], c["arguments"]["command"]) for c in calls] == \
        [("bash", "write /work/answer.txt 42"), ("bash", "cat /work/answer.txt")]
    assert result.refusals == ((1, 0, "not_offered"),)       # the two-block turn, answered by mini's error
    seen = [tokenizer.decode(b["token_ids"][len(a["token_ids"]):], skip_special_tokens=False)
            for a, b in zip(engine.requests, engine.requests[1:])]
    assert "<|im_start|>user\n<returncode>0</returncode>\n<output>\n</output>" in seen[0]
    assert "found 2 actions" in seen[1] and "<tool_response>" not in "".join(seen)
    assert "<|im_start|>user\n<returncode>0</returncode>\n<output>\n42</output>" in seen[2]
    parsed = _parse(_renderer(MINI_SWE_V1, 0), result, engine)
    assert [a.tool for a in parsed.actions] == ["bash", "?format", "bash"]
    with pytest.raises(TrajectoryRefused):
        _parse(_renderer(BASH_VF_B2E4E81, 0), result, engine)


def test_any_task_can_play_mini():
    pin = {"id": MINI_SWE_V1.id, "version": MINI_SWE_V1.version, "digest": MINI_SWE_V1.digest}
    assert agentic_swe.harness_pin_refusal(pin, variants=False) is None     # corpus, eval, RL window
    assert agentic_swe.harness_pin_refusal(pin, variants=True) is None      # RL lease
