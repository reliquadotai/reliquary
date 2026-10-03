"""One SWE-smith episode, run by `verifiers` in-process against the miner's
loopback generate endpoint. `verifiers`, `renderers` and `reliquary_swe` are
imported here only.

The env config is the one gate qualification ran (bash harness, edit on,
search off, docker runtime with egress blocked after setup), with the train
client and the job's renderer, sampling and turn limit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    session_id: str | None
    final_diff: str
    stop: str | None
    ok: bool
    reward: float | None
    error: str | None = None


def env_config(episode, *, harness_env: dict | None = None) -> dict:
    return {
        "taskset": {"id": "reliquary-swe", "split": episode.env.split,
                    "num_images": episode.env.num_images},
        "agent": {"harness": {"id": "bash", "edit": True, "search": False,
                              "env": dict(harness_env or {})},
                  "runtime": {"type": "docker", "allow": [], "block": ["*"]},
                  "max_turns": episode.max_turns},
    }


def _errors(errors) -> str:
    return "; ".join(f"{type(e).__name__}: {getattr(e, 'message', e)}" for e in errors)


class SweEpisodeRunner:
    def __init__(self, *, episode, model_name: str, renderer_model_dir: str, generate_url: str,
                 sampling, harness_env: dict | None = None) -> None:
        from renderers.configs import Qwen38RendererConfig
        from verifiers.v1.clients import ModelContext
        from verifiers.v1.configs.client import TrainClientConfig
        from verifiers.v1.types import Sampling
        from verifiers.v1.utils.loaders import load_environment, resolve_env_config

        self._episode = episode
        self._env = load_environment(resolve_env_config(env_config(episode, harness_env=harness_env)))
        self._ctx = ModelContext(
            model=model_name,
            client=TrainClientConfig(base_url=f"{generate_url.rstrip('/')}/v1",
                                     api_key_var="RELIQUARY_GENERATE_KEY",
                                     renderer=Qwen38RendererConfig(),
                                     renderer_model_name=renderer_model_dir),
            sampling=Sampling(temperature=sampling.temperature, top_p=sampling.top_p,
                              max_tokens=episode.max_tokens_per_turn),
        )
        self._rows = None
        self._serving = None

    def task(self, index: int):
        from reliquary_swe import corpus
        from reliquary_swe.taskset import task_for

        if self._rows is None:
            self._rows = corpus.load_swesmith_rows(self._episode.env.num_images)
        # The taskset's own task config, as SweTaskset.load builds every task.
        return task_for(self._rows[index], index, self._episode.env.split,
                        self._env.taskset.config.task)

    async def __aenter__(self) -> "SweEpisodeRunner":
        self._serving = self._env.serving()
        await self._serving.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._serving.__aexit__(*exc)

    async def run(self, index: int) -> EpisodeResult:
        episode = await self._env.run_episode(self.task(index), self._ctx)
        if not episode.traces:
            return EpisodeResult(None, "", None, False, None,
                                 error=_errors(episode.errors) or "no trace")
        trace = episode.traces[0]
        # The episode's ok also covers finalize(), which captures the diff:
        # a trace whose diff was never captured is not a trajectory to sign.
        ok = bool(episode.ok and trace.ok)
        return EpisodeResult(
            session_id=trace.id, final_diff=str(trace.info.get("patch") or ""),
            stop=trace.stop_condition, ok=ok,
            reward=trace.reward if trace.rewards else None,
            error=None if ok else (_errors(list(trace.errors) + list(episode.errors))
                                   or "episode not ok"))


__all__ = ["EpisodeResult", "SweEpisodeRunner", "env_config"]
