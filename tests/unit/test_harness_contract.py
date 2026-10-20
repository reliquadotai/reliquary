"""The harness in the contract: a corpus job's `episode.harness_spec`, an RL episode env's
`episode.harness`. Absent, nothing changes (the compatibility harness, same contract bytes);
present, it is `{id, version, digest}` and a binary that does not serve it refuses."""

from __future__ import annotations

import pytest
from reliquary_harness.spec import BASH_SURFACE_V1, BASH_VF_B2E4E81

from reliquary.corpus.job import JobError, parse_job
from reliquary.environment import agentic_swe
from reliquary.validator.rl_async_episodes import (
    async_episode_policy, load_episode_environment, validate_episode_block,
)
from tests.unit.episode_v2_fixtures import EPISODE, episode_block
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_corpus_job_signed_sandbox import _sha, signed_episode


def _pin(spec, **overrides):
    return {"id": spec.id, "version": spec.version, "digest": spec.digest, **overrides}


def test_a_job_without_a_harness_spec_keeps_its_contract_bytes():
    raw = _manifest(episode=signed_episode())
    job = parse_job(raw)
    assert job.episode.harness_spec is None
    assert "harness_spec" not in job.to_contract()["episode"]
    assert _sha(job.to_contract()) == _sha(raw)


def test_a_job_pinning_a_harness_carries_it_in_its_contract():
    raw = _manifest(episode=signed_episode(harness_spec=_pin(BASH_VF_B2E4E81)))
    job = parse_job(raw)
    assert job.episode.harness_spec == _pin(BASH_VF_B2E4E81)
    assert job.to_contract()["episode"]["harness_spec"] == _pin(BASH_VF_B2E4E81)
    assert parse_job(job.to_contract()) == job
    assert _sha(job.to_contract()) == _sha(raw) != _sha(_manifest(episode=signed_episode()))


@pytest.mark.parametrize("bad", [
    "bash-vf-b2e4e81", {"id": "bash-vf-b2e4e81", "version": "1"},
    _pin(BASH_VF_B2E4E81, digest="x" * 64), _pin(BASH_VF_B2E4E81, id="Bash VF"),
    {**_pin(BASH_VF_B2E4E81), "variant": 0},
])
def test_a_malformed_harness_spec_is_refused(bad):
    with pytest.raises(JobError, match="harness_spec"):
        parse_job(_manifest(episode=signed_episode(harness_spec=bad)))


def test_a_corpus_job_cannot_pin_a_harness_with_variants_nor_another_digest():
    surface = parse_job(_manifest(episode=signed_episode(harness_spec=_pin(BASH_SURFACE_V1))))
    refusal = agentic_swe.sandbox_support_refusal(surface.episode)
    assert "an RL lease's, not a corpus job's" in refusal
    stale = parse_job(_manifest(episode=signed_episode(harness_spec=_pin(BASH_VF_B2E4E81, digest="0" * 64))))
    assert "this build serves" in agentic_swe.episode_support_refusal(stale.episode, need_verifiers=False)


def test_an_rl_episode_block_may_name_its_harness_and_its_policy_carries_it():
    plain = validate_episode_block(EPISODE, {"episode": episode_block()})
    assert async_episode_policy(EPISODE, plain).harness_pin() is None
    block = validate_episode_block(EPISODE, {"episode": {**episode_block(), "harness": _pin(BASH_SURFACE_V1)}})
    policy = async_episode_policy(EPISODE, block)
    assert policy.harness_pin() == _pin(BASH_SURFACE_V1)
    assert agentic_swe.harness_of(policy.harness_pin()) is BASH_SURFACE_V1
    with pytest.raises(ValueError, match="harness"):
        validate_episode_block(EPISODE, {"episode": {**episode_block(), "harness": {"id": "bash-surface"}}})


def test_an_rl_env_whose_harness_this_build_does_not_serve_is_not_loaded():
    block = validate_episode_block(EPISODE, {"episode": {**episode_block(),
                                                         "harness": _pin(BASH_SURFACE_V1, digest="0" * 64)}})
    with pytest.raises(ValueError, match="this build serves"):
        load_episode_environment(EPISODE, block)


def test_the_validator_grades_a_lease_in_the_variant_it_draws_for_it():
    from types import SimpleNamespace

    from reliquary.validator.rl_episode import LeaseEpisodeChecker

    class Renderer:
        def __init__(self, variant):
            self.variant = variant

        def initial_ids(self, prompt):
            return [self.variant, len(prompt)]

    block = validate_episode_block(EPISODE, {"episode": {**episode_block(), "harness": _pin(BASH_SURFACE_V1)}})
    policy = async_episode_policy(EPISODE, block)
    built = []
    checker = LeaseEpisodeChecker(policy=policy, renderer=Renderer(0), source=SimpleNamespace(prompt=lambda i: "p" * i),
                                  directory=lambda now: None, token_verifier=None,
                                  renderer_of=lambda v: built.append(v) or Renderer(v))
    leases = [SimpleNamespace(nonce=f"{i:064x}", lease_id="L", prompt_idx=3) for i in range(32)]
    for lease in leases:
        variant = checker.lease_variant(lease)
        assert variant == agentic_swe.lease_variant(BASH_SURFACE_V1, lease)
        assert checker.prompt_ids(lease.prompt_idx, variant) == (variant, 3)
    assert sorted(set(built)) == [1, 2, 3]          # one renderer per variant, built once


def test_tasks_create_pins_the_harness_each_episode_env_plays():
    from reliquary.cli.main import with_episode_harnesses
    from reliquary_harness.spec import MINI_SWE_V1

    envs = {"swe_v1": {"episode": episode_block()}, "tmax_v1": {"episode": episode_block()}}
    pinned = with_episode_harnesses(envs, "swe_v1=mini-swe@1, tmax_v1=bash-surface@1")
    assert pinned["swe_v1"]["episode"]["harness"] == _pin(MINI_SWE_V1)
    assert pinned["tmax_v1"]["episode"]["harness"] == _pin(BASH_SURFACE_V1)
    assert "harness" not in envs["swe_v1"]["episode"]                  # the file's blocks are not edited
    assert with_episode_harnesses(envs, None) is envs
    for bad, why in [("swe_v1", "env=id@version"), ("other=mini-swe@1", "not an --rl-episode-envs env"),
                     ("swe_v1=mini-swe@1,swe_v1=bash-surface@1", "twice"), ("swe_v1=openhands@1", "unknown harness")]:
        with pytest.raises(ValueError, match=why):
            with_episode_harnesses(envs, bad)
    for name, block in pinned.items():
        validate_episode_block(name, block)
