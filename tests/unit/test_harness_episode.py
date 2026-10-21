"""`HarnessEpisodeRunner`: how the signed miner builds it, and how a run that fails still
finishes its episode. (What it plays is `test_harness_episode_parity`.)"""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from reliquary.miner.signed_episode import SignedSweEpisodeRunner  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402

needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


def test_the_signed_miner_plays_its_episodes_with_its_own_loop():
    runner = SignedSweEpisodeRunner.__new__(SignedSweEpisodeRunner)
    runner._model_name, runner._renderer_dir, runner._generate_url = "policy", "/ckpt", "http://127.0.0.1:8013"
    runner._sampling = SimpleNamespace(temperature=0.6, top_p=0.9)
    runner._job = SimpleNamespace(episode=SimpleNamespace(max_tokens_per_turn=4096, max_turns=40))
    built = runner._default_runner("http://gateway")
    assert isinstance(built, HarnessEpisodeRunner)
    assert (built._url, built._model, built._renderer_dir, built._generate_url, built._max_turns) == \
        ("http://gateway", "policy", "/ckpt", "http://127.0.0.1:8013", 40)
    assert built._sampling == {"temperature": 0.6, "top_p": 0.9, "max_tokens": 4096}


def _closed_port() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def _directory() -> str:
    return next(iter(TOKENIZERS.values()))


async def _play(tmp_path, *, generate_url=None, wall_s=3600, script=None):
    tokenizer = parity._tokenizer(_directory())
    engine = parity.FakeEngine(tokenizer, script or parity.SCENARIOS["honest"]["turns"], {248046, 248044}, 1 << 20)
    seen = []
    with testing.fake_gateway(tmp_path) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets(wall_s=wall_s))
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=_directory(),
                                        generate_url=generate_url or engine_url,
                                        sampling=SimpleNamespace(temperature=1.0, top_p=1.0), max_tokens=64,
                                        max_turns=4) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT, on_trace=seen.append)
    return result, seen


@needs_tokenizer
def test_an_unreachable_engine_is_an_error_and_the_episode_is_still_finished(tmp_path):
    result, seen = asyncio.run(_play(tmp_path, generate_url=_closed_port()))
    assert result.error and "Connect" in result.error
    assert result.trace.stop_condition is None and [t.id for t in seen] == [result.trace.id]
    assert result.transcript["records"][-1]["body"]["kind"] == "final"


@needs_tokenizer
def test_a_run_past_the_wall_budget_stops_as_an_error(tmp_path, monkeypatch):
    import reliquary.miner.harness_episode as module

    # A 61 s wall budget leaves the agent 1 s (FINISH_MARGIN_S = 60); the engine waits 3.
    monkeypatch.setattr(module, "FINISH_MARGIN_S", 60.0)
    original = parity.FakeEngine.__init__

    def slow(self, *args, **kwargs):
        original(self, *args, **kwargs)
        routes = [r for r in self.app.router.routes if getattr(r, "path", "") == "/inference/v1/generate"]
        endpoint = routes[0].endpoint

        async def wait_then(request):
            await asyncio.sleep(3)
            return await endpoint(request)

        routes[0].endpoint = wait_then
        routes[0].dependant.call = wait_then

    monkeypatch.setattr(parity.FakeEngine, "__init__", slow)
    result, _ = asyncio.run(_play(tmp_path, wall_s=61))
    assert result.error == "stopped before the episode's wall budget ran out"
    assert result.transcript["records"][-1]["body"]["kind"] == "final"
