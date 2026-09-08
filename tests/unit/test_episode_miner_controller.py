from __future__ import annotations

from types import SimpleNamespace

from reliquary.constants import M_ROLLOUTS
from reliquary.environment.agentic.types import GeneratedAction
from reliquary.environment.registry import get_environment_spec
from reliquary.miner.engine import MiningEngine
from reliquary.protocol.profiles import resolve_protocol_profile


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_episode_controller_builds_complete_flattened_rollouts(monkeypatch):
    env = get_environment_spec("reliquary_retrieval_tools_v1").create()
    reference = env.get_task(0).private["reference_actions"]

    class _Policy:
        def __init__(self, **kwargs):
            del kwargs
            self.actions = iter(reference)

        def generate(self, **kwargs):
            del kwargs
            return GeneratedAction(text=next(self.actions).to_json())

    monkeypatch.setattr(
        "reliquary.miner.episode_policy.HFEpisodePolicy", _Policy
    )
    monkeypatch.setattr(
        "reliquary.miner.engine.ACTIVE_PROTOCOL_PROFILE",
        resolve_protocol_profile("qwen3-4b-reliquary-episode-v7-dev1"),
    )
    engine = object.__new__(MiningEngine)
    engine.tokenizer = _Tokenizer()
    engine.vllm_model = object()
    engine.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="qualification-miner")
    )

    generations = engine._generate_m_episode_rollouts(
        env,
        "ab" * 32,
        prompt_idx=0,
        checkpoint_hash="checkpoint",
    )
    assert len(generations) == M_ROLLOUTS
    for generation in generations:
        trace = generation["trace"]
        assert generation["prompt_length"] == trace.assistant_spans[0][0]
        assert generation["tokens"] == list(trace.tokens)
        assert trace.reward is not None and trace.reward.reward == 1.0
        assert len(trace.assistant_spans) == len(reference)
