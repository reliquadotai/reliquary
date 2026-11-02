"""New weights reach the episode engine with turns in flight (rl_async_episodes.EpisodeEngineHost.push,
corpus_generate_server.GenerateEngine.between_steps)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from reliquary.miner.corpus_generate_server import GenerateEngine
from reliquary.miner.rl_async_episodes import EpisodeEngineHost, EpisodeWeights


class Llm:
    def __init__(self):
        self.calls, self.thread = [], None

    def collective_rpc(self, fn, args=()):
        self.thread = threading.current_thread().name
        self.calls.append(("load", len(args[0])))

    def reset_prefix_cache(self, reset_running_requests=False):
        self.calls.append(("reset", reset_running_requests))
        return True


class Core:
    def __init__(self, unfinished=False):
        self._llm, self.unfinished = Llm(), unfinished

    def has_unfinished(self):
        return self.unfinished

    def step(self):
        return []

    def add(self, *a, **k):
        pass


def _host(core):
    host = EpisodeEngineHost(make_core=None, release_core=None, serve=None, draws=None, model_name="m", port=1,
                             max_total_tokens=64, max_tokens_per_turn=16, stop_ids=lambda: [1], sampling=None,
                             proof=None)
    host.core, host.engine = core, GenerateEngine(core, max_total_tokens=64, max_tokens_per_turn=16)
    host.engine.start()
    return host


def test_the_engine_thread_runs_a_call_between_steps_and_hands_back_its_result_or_error():
    engine = GenerateEngine(Core(), max_total_tokens=64, max_tokens_per_turn=16)
    with pytest.raises(RuntimeError):
        engine.between_steps(lambda: 1)                              # not running
    engine.start()
    try:
        assert engine.between_steps(lambda: threading.current_thread().name) == "generate-engine"
        with pytest.raises(ValueError):
            engine.between_steps(lambda: (_ for _ in ()).throw(ValueError("bad")))
        assert engine.between_steps(lambda: 2) == 2                  # the thread survived the error
    finally:
        engine.stop()


def test_a_push_keeps_the_engine_and_resumes_the_turns_in_flight_on_the_new_weights(tmp_path):
    core = Core(unfinished=True)
    host = _host(core)
    engine = host.engine
    try:
        host.push({"w": torch.zeros(2)}, tmp_path / "scratch")
        assert host.engine is engine and engine._thread.is_alive()   # sessions and waiters kept
        assert core._llm.calls == [("load", 1), ("reset", True)] and core._llm.thread == "generate-engine"
        assert not (tmp_path / "scratch").exists()
    finally:
        engine.stop()


def test_a_failed_push_with_turns_in_flight_is_raised_not_restarted(tmp_path):
    core = Core(unfinished=True)
    host = _host(core)
    host.restart = lambda *a: pytest.fail("a restart would end the episodes in flight")
    core._llm.collective_rpc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("push failed"))
    weights = EpisodeWeights(host, base_model_dir=str(tmp_path), work_dir=str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="push failed"):
            weights({"w": torch.zeros(2)}, 3)
    finally:
        host.engine.stop()
