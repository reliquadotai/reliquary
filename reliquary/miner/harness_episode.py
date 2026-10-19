"""One signed episode played by the miner's own loop (reliquary-harness), in place of verifiers'
bash harness: the miner does inference only.

Per episode: open the token's session on the gateway, check record 0's tools version, read
its offered tools, then `reliquary_harness.loop.play` with

* the bash harness spec's Qwen3.8 turn renderer for those tools (the validator's own,
  `agentic_swe.load_turn_renderer`),
* `reliquary_harness.engine.GenerateEngine` on the local generate endpoint (every turn
  named by one session id, reported through `on_trace` before the first turn, so the
  engine's session log holds the episode's tokens and proofs as before),
* every call answered by `reliquary_sandbox.tool_routing.ToolRouter` over record 0's tools
  (the signed record's rendering, or a refusal's text),

then finish the episode and fetch its transcript. What goes on the wire, turn by turn, is
what verifiers b2e4e81's bash harness under its train client sent
(`tests/unit/test_harness_episode_parity.py` holds the two equal at the generate
endpoint); the result has `SandboxEpisodeResult`'s shape, so `SignedSweEpisodeRunner`
settles it unchanged.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

FINISH_MARGIN_S = 60.0
"""The agent stops this long before the episode's wall deadline, so `finish` is in time."""
CANCELLED_FINISH_TIMEOUT_S = 120.0
"""When the run is cancelled, `finish` (which frees the box) gets at most this long before
the cancellation propagates; the gateway's own deadline closes the episode otherwise."""
API_KEY_VAR = "RELIQUARY_GENERATE_KEY"


@dataclass(frozen=True)
class HarnessEpisodeResult:
    """One episode's outcome. `final`, `state` and `transcript` are the gateway's; `trace`
    has the session `id` the engine logged the turns under and the loop's `stop_condition`.
    `error` set means the run did not complete cleanly (an engine, renderer or routing
    failure, or the wall budget): do not submit the episode."""

    episode_id: str
    trace: Any
    final: dict[str, Any]
    state: bytes | None
    transcript: dict[str, Any]
    refusals: tuple[tuple[int, int, str], ...]
    error: str | None


class HarnessEpisodeRunner:
    """Episodes on one gateway. `sampling` has the job's `temperature` and `top_p`;
    `max_tokens` is the job's per-turn cap."""

    def __init__(self, *, gateway_url: str, model: str, renderer_dir: str, generate_url: str,
                 sampling, max_tokens: int, max_turns: int, load_renderer=None, client=None) -> None:
        self._url = gateway_url
        self._model = model
        self._renderer_dir = renderer_dir
        self._generate_url = generate_url
        self._sampling = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                          "max_tokens": int(max_tokens)}
        self._max_turns = int(max_turns)
        self._load_renderer = load_renderer
        self._client = client
        self._renderers: dict[tuple[str, ...], Any] = {}
        self._renderers_lock = asyncio.Lock()

    async def __aenter__(self) -> HarnessEpisodeRunner:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _engine_client(self):
        if self._client is None:
            from reliquary_harness.engine import openai_client

            self._client = openai_client(f"{self._generate_url.rstrip('/')}/v1",
                                         os.environ.get(API_KEY_VAR) or "EMPTY")
        return self._client

    async def _renderer(self, tools: tuple[str, ...]):
        key = tuple(sorted(tools))
        async with self._renderers_lock:
            if key not in self._renderers:
                load = self._load_renderer
                if load is None:
                    from reliquary.environment.agentic_swe import load_turn_renderer as load
                self._renderers[key] = await asyncio.to_thread(load, self._renderer_dir, tools=key)
            return self._renderers[key]

    async def run(self, *, token: dict[str, Any], prompt: str, on_trace=None) -> HarnessEpisodeResult:
        """Open the token's episode and play it. The offered tools are record 0's (signed),
        for both the renderer's tool list and the router's refusals."""
        from reliquary_harness.engine import GenerateEngine
        from reliquary_harness.loop import play
        from reliquary_sandbox.episode_client import EpisodeClient
        from reliquary_sandbox.observation import require_tools_version
        from reliquary_sandbox.tool_routing import ToolRouter

        loop = asyncio.get_running_loop()
        # Taken before the open: the gateway's wall time starts no earlier (when the box is
        # prepared), so this deadline is never later than the gateway's.
        started = loop.time()
        trace = SimpleNamespace(id=uuid.uuid4().hex, stop_condition=None)
        async with EpisodeClient(self._url) as client:
            episode = await client.open(token)
            router, error = None, None
            try:
                body = episode.record0["body"]
                deadline = started + max(1.0, body["budgets"]["wall_s"] - FINISH_MARGIN_S)
                require_tools_version(body["tools_version"])
                tools = tuple(body["tools"])
                router = ToolRouter(episode, offered=tools)
                renderer = await self._renderer(tools)
                engine = GenerateEngine(renderer, client=self._engine_client(), model=self._model,
                                        sampling=self._sampling)
                if on_trace is not None:
                    on_trace(trace)

                async def generate(prompt_ids, bridged_from):
                    return await engine.turn(prompt_ids, session_id=trace.id, bridged_from=bridged_from)

                async with asyncio.timeout_at(deadline):
                    outcome = await play(renderer=renderer, prompt=prompt, max_turns=self._max_turns,
                                         generate=generate, answer=router.answer,
                                         closed=lambda: router.final is not None)
                trace.stop_condition = outcome.stop
            except TimeoutError:
                error = "stopped before the episode's wall budget ran out"
            except Exception as failure:  # the episode is still finished below
                error = f"{type(failure).__name__}: {failure}"
            except BaseException:
                # Cancelled or interrupted: try to free the box, briefly, then propagate.
                with suppress(Exception):
                    async with asyncio.timeout(CANCELLED_FINISH_TIMEOUT_S):
                        await episode.finish()
                raise
            finished = await episode.finish()
            transcript = (await episode.transcript())["transcript"]
        refusals = tuple(router.refusals) if router is not None else ()
        return HarnessEpisodeResult(episode.id, trace, finished.record, finished.state, transcript,
                                    refusals, error)


__all__ = ["CANCELLED_FINISH_TIMEOUT_S", "FINISH_MARGIN_S", "HarnessEpisodeResult", "HarnessEpisodeRunner"]
