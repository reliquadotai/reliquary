"""A weighted mix of harnesses on one RL episode env: the block's `harness` is a list of `{id, version, digest,
weight}`, and each lease plays one of them, drawn from the lease's nonce (the validator's secret-derived) by both
sides alone, then that harness's surface variant. One harness keeps its block, its bytes and its draw; a window
or a corpus job, with no validator seed, refuses a mix."""

from __future__ import annotations

import asyncio
import collections
import logging
from types import SimpleNamespace

import pytest
from reliquary_harness.spec import BASH_SURFACE_V1, BASH_VF_B2E4E81, MINI_SWE_V2

from reliquary.environment import agentic_swe
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError
from reliquary.validator.rl_async_episodes import (
    async_episode_policy, load_episode_environment, validate_episode_block,
)
from tests.unit.episode_v2_fixtures import EPISODE, episode_block, episode_contract_dict
from tests.unit.test_harness_goldens import TOKENIZERS


def _pin(spec, **extra):
    return {"id": spec.id, "version": spec.version, "digest": spec.digest, **extra}


MIX = [_pin(MINI_SWE_V2, weight=3), _pin(BASH_SURFACE_V1, weight=1)]


def _policy(harness):
    block = episode_block() if harness is None else episode_block(harness=harness)
    return async_episode_policy(EPISODE, validate_episode_block(EPISODE, {"episode": block}))


def _leases(count, prompt_idx=3):
    return [SimpleNamespace(nonce=f"{i:064x}", lease_id=f"L{i}", prompt_idx=prompt_idx) for i in range(count)]


def test_a_block_may_name_a_weighted_mix_and_its_policy_carries_it():
    policy = _policy(MIX)
    assert policy.harness_choices() == ((_pin(MINI_SWE_V2), 3), (_pin(BASH_SURFACE_V1), 1))
    with pytest.raises(ValueError, match="harness mix drawn per lease"):
        policy.harness_pin()
    assert agentic_swe.policy_default_harness(policy) is MINI_SWE_V2
    window = ServiceContract.from_dict(episode_contract_dict(episode=episode_block(harness=MIX)))
    assert window.episode_policy(EPISODE).harness_choices() == policy.harness_choices()
    single = _policy(_pin(MINI_SWE_V2))
    assert single.harness_mix is None and single.harness_choices() == ((_pin(MINI_SWE_V2), 1),)
    assert _policy(None).harness_choices() == ((None, 1),)


@pytest.mark.parametrize("bad, why", [
    ([_pin(MINI_SWE_V2, weight=1)], "2 to 8"),
    ([_pin(MINI_SWE_V2, weight=1), _pin(MINI_SWE_V2, weight=2)], "twice"),
    ([_pin(MINI_SWE_V2, weight=0), _pin(BASH_SURFACE_V1, weight=1)], "weight"),
    ([_pin(MINI_SWE_V2, weight=1.5), _pin(BASH_SURFACE_V1, weight=1)], "weight"),
    ([_pin(MINI_SWE_V2, weight=True), _pin(BASH_SURFACE_V1, weight=1)], "weight"),
    ([_pin(MINI_SWE_V2), _pin(BASH_SURFACE_V1, weight=1)], "weight"),
    ([_pin(MINI_SWE_V2, weight=1), {"id": "bash-surface", "version": "1", "weight": 1}], r"\{id, version, digest\}"),
])
def test_a_malformed_mix_is_refused(bad, why):
    with pytest.raises((ValueError, ServiceContractError), match=why):
        validate_episode_block(EPISODE, {"episode": episode_block(harness=bad)})


def test_each_lease_draws_one_harness_of_the_mix_at_its_weight():
    policy = _policy(MIX)
    leases = _leases(4000)
    drawn = [agentic_swe.lease_surface(policy, lease) for lease in leases]
    assert drawn == [agentic_swe.lease_surface(policy, lease) for lease in leases]    # deterministic
    counts = collections.Counter(spec.key for spec, _ in drawn)
    assert set(counts) == {"mini-swe@2", "bash-surface@1"}
    assert 0.70 < counts["mini-swe@2"] / len(leases) < 0.80                          # weight 3 of 4
    # A mini lease has no variant; a bash-surface lease draws its own, all four reached.
    assert {v for spec, v in drawn if spec is MINI_SWE_V2} == {0}
    assert {v for spec, v in drawn if spec is BASH_SURFACE_V1} == {0, 1, 2, 3}
    for (spec, variant), lease in zip(drawn, leases):
        assert variant == agentic_swe.lease_variant(spec, lease)
    # Another prompt, lease or nonce is another draw; other weights are another mix.
    other = _policy([_pin(MINI_SWE_V2, weight=1), _pin(BASH_SURFACE_V1, weight=1)])
    assert [agentic_swe.lease_surface(other, lease)[0] for lease in leases[:200]] != \
        [spec for spec, _ in drawn[:200]]


def test_one_harness_keeps_its_draw():
    for harness, spec in [(None, BASH_VF_B2E4E81), (_pin(BASH_SURFACE_V1), BASH_SURFACE_V1),
                          (_pin(MINI_SWE_V2), MINI_SWE_V2)]:
        policy = _policy(harness)
        for lease in _leases(64):
            assert agentic_swe.lease_surface(policy, lease) == (spec, agentic_swe.lease_variant(spec, lease))


def test_a_mix_this_build_does_not_fully_serve_is_not_loaded():
    stale = [_pin(MINI_SWE_V2, weight=1), _pin(BASH_SURFACE_V1, weight=1, digest="0" * 64)]
    block = validate_episode_block(EPISODE, {"episode": episode_block(harness=stale)})
    with pytest.raises(ValueError, match="this build serves"):
        load_episode_environment(EPISODE, block)
    assert "this build serves" in agentic_swe.policy_harness_refusal(async_episode_policy(EPISODE, block))
    assert agentic_swe.policy_harness_refusal(_policy(MIX)) is None


def test_a_window_refuses_a_mix():
    from reliquary.miner.episode_mining import _episode_policies
    from reliquary.validator.service import ValidationService

    contract = ServiceContract.from_dict(episode_contract_dict(episode=episode_block(harness=MIX)))
    with pytest.raises(ValueError, match="an RL lease's"):
        _episode_policies(contract, [EPISODE], lambda policy: None)
    with pytest.raises(ValueError, match="an RL lease's"):
        ValidationService._default_episode_renderer(SimpleNamespace(tokenizer=None), contract.episode_policy(EPISODE))


def test_tasks_create_writes_a_weighted_mix():
    from reliquary.cli.main import with_episode_harnesses

    envs = {"swe_v1": {"episode": episode_block()}, "tmax_v1": {"episode": episode_block()}}
    out = with_episode_harnesses(envs, "swe_v1=mini-swe@2:3+bash-surface@1, tmax_v1=bash-surface@1")
    assert out["swe_v1"]["episode"]["harness"] == [_pin(MINI_SWE_V2, weight=3), _pin(BASH_SURFACE_V1, weight=1)]
    assert out["tmax_v1"]["episode"]["harness"] == _pin(BASH_SURFACE_V1)      # one harness: its block as before
    for name, block in out.items():
        validate_episode_block(name, block)
    for bad, why in [("swe_v1=mini-swe@2:2", "at least two"), ("swe_v1=mini-swe@2:x+bash-surface@1", "weight"),
                     ("swe_v1=mini-swe@2+mini-swe@2", "twice"), ("swe_v1=mini-swe@2+openhands@1", "unknown harness"),
                     ("swe_v1=mini-swe@2:0+bash-surface@1", "weight")]:
        with pytest.raises(ValueError, match=why):
            with_episode_harnesses(envs, bad)


class _Renderer:
    def __init__(self, spec, variant):
        self.spec, self.variant = spec, variant

    def initial_ids(self, prompt):
        return [hash(self.spec.key) % 997, self.variant, len(prompt)]


def test_the_validator_grades_each_lease_in_the_surface_it_draws(caplog):
    from reliquary.validator.rl_episode import LeaseEpisodeChecker
    from reliquary.validator.rl_submit import Graded

    policy = _policy(MIX)
    built = []

    def renderer_of(variant, spec=None):
        built.append(((spec or MINI_SWE_V2).key, variant))
        return _Renderer(spec or MINI_SWE_V2, variant)

    checker = LeaseEpisodeChecker(policy=policy, renderer=_Renderer(MINI_SWE_V2, 0),
                                  source=SimpleNamespace(prompt=lambda i: "p" * i), directory=lambda now: None,
                                  token_verifier=None, renderer_of=renderer_of)
    for lease in _leases(64):
        spec, variant = checker.lease_surface(lease)
        assert (spec, variant) == agentic_swe.lease_surface(policy, lease)
        renderer = checker.surface_renderer(spec, variant)
        assert (renderer.spec, renderer.variant) == (spec, variant)
        assert checker.prompt_ids(3, variant, spec) == tuple(_Renderer(spec, variant).initial_ids("ppp"))
    assert sorted(set(built)) == [("bash-surface@1", v) for v in range(4)]     # mini's is the default renderer

    def grade(lease, rollouts, *, received, surface):
        return Graded((1,), reject="episode_parse"), {"rollout": 2, "check": "bad_turns"}

    checker._grade = grade
    with caplog.at_level(logging.INFO, logger="reliquary.validator.rl_episode"):
        for lease in _leases(8):
            checker.grade(lease, [], received=0.0)
    assert sum(checker.outcomes.values()) == 8
    assert {key for key, _ in checker.outcomes} <= {"mini-swe@2", "bash-surface@1"}
    assert "bad_turns" in caplog.text and "mini-swe@2" in caplog.text


def test_the_miner_plays_each_lease_in_the_harness_the_validator_draws(monkeypatch):
    from reliquary.miner import episode_commit, rl_async_episodes
    from reliquary.miner.forced_draw import ForcedDraws
    from reliquary.miner.rl_async_episodes import EpisodeLeaseGenerator
    from tests.unit.test_rl_async_episodes import NONCE, _Runner

    def admissible(outcome, *, policy, renderer, prompt, lease=False):
        tokens = [7, 7, 7] + [outcome.seed_index] * 4 + [5]
        return episode_commit.AdmissibleEpisode(tokens, [(3, 8)], episode_commit.AdmittedStop("agent_completed")), None

    monkeypatch.setattr(episode_commit, "episode_admissible", admissible)
    monkeypatch.setattr(rl_async_episodes, "OPEN_RETRY_S", 0.0)
    played, rendered = [], []

    def factory(policy, held, text, variant=0, harness=None):
        played.append((held.lease_id, harness.key, variant))
        return _Runner()

    def renderer_for(policy, variant=0, spec=None):
        rendered.append((spec.key, variant))
        return SimpleNamespace(initial_ids=lambda t: [7, 7, 7])

    class Engine:
        def take_session(self, session_id):
            return SimpleNamespace(session_id=session_id)

    policy = _policy(MIX)
    from reliquary.miner.rl_async import HeldLease

    leases = [HeldLease(f"lease{i}", EPISODE, 4, 2, f"{i:02x}" * 32 if i else NONCE, 1e12) for i in range(12)]
    generator = EpisodeLeaseGenerator(policies={EPISODE: policy}, runner_factory=factory, engine=Engine(),
                                      draws=ForcedDraws(), renderer_for=renderer_for,
                                      task_prompt=lambda env, idx: "task")
    groups = asyncio.run(generator.agenerate(leases))
    assert all(len(group) == 16 for group in groups)
    expected = [(lease.lease_id, *(lambda s: (s[0].key, s[1]))(agentic_swe.lease_surface(policy, lease)))
                for lease in leases]
    assert sorted(played) == sorted(expected)
    assert {key for _, key, _ in played} == {"mini-swe@2", "bash-surface@1"}
    assert set(rendered) == {(key, variant) for _, key, variant in expected}


@pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
def test_the_validators_renderers_render_each_harness_of_the_mix():
    from reliquary.validator.rl_async_service import renderer_for_checkpoint

    renderer_for = renderer_for_checkpoint(next(iter(TOKENIZERS.values())))
    policy = _policy(MIX)
    default = renderer_for(policy)
    assert default.spec is MINI_SWE_V2
    seen = {}
    for lease in _leases(64):
        spec, variant = agentic_swe.lease_surface(policy, lease)
        renderer = renderer_for(policy, variant=variant, spec=spec)
        assert renderer.spec is spec and renderer is renderer_for(policy, variant=variant, spec=spec)
        seen[spec.key, variant] = renderer.initial_ids("Fix it.")
    assert seen["mini-swe@2", 0] == default.initial_ids("Fix it.")
    assert len(set(map(tuple, seen.values()))) == len(seen)          # every surface renders its own prompt
