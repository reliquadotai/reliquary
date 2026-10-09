"""The signed_episode interaction mode (plan 2C, Task 6)."""
import dataclasses

import pytest

from reliquary.environment import registry
from reliquary.environment.base import Environment
from reliquary.environment.registry import get_environment_spec
from reliquary.environment.signed_episode import SignedEpisodeEnvironment
from tests.unit.episode_v2_fixtures import EPISODE, PROMPT_TEXT, episode_spec, register_episode_env


def test_the_episode_spec_creates_the_adapter_and_names_its_mode(monkeypatch):
    register_episode_env(monkeypatch)
    spec = get_environment_spec(EPISODE)
    env = spec.create()
    assert isinstance(env, SignedEpisodeEnvironment) and isinstance(env, Environment)
    assert env.name == EPISODE and len(env) == 1000
    problem = env.get_problem(3)
    assert problem["prompt"] == PROMPT_TEXT and problem["task_index"] == 3 and len(problem["id"]) == 16
    manifest = spec.consensus_manifest()
    assert manifest["interaction_mode"] == "signed_episode"
    assert manifest["episode_schema"] == "reliquary/signed-episode/v1"


def test_a_signed_episode_is_never_scored_from_text(monkeypatch):
    register_episode_env(monkeypatch)
    spec = get_environment_spec(EPISODE)
    with pytest.raises(TypeError):
        spec.score_many({"prompt": "x"}, ["done"])
    with pytest.raises(TypeError):
        spec.create().compute_reward({"prompt": "x"}, "done")


@pytest.mark.parametrize("change", [
    dict(admission_resource_class="cpu"), dict(validator_authoritative_reward=False),
    dict(renderer_id="reliquary-chatml-tools-v1"),
])
def test_a_signed_episode_spec_must_be_sandboxed_and_validator_scored(change):
    with pytest.raises(ValueError):
        dataclasses.replace(episode_spec(), **change)


def test_the_installed_manifests_keep_their_bytes():
    for name, spec in registry.ENVIRONMENT_SPECS.items():
        assert spec.consensus_manifest().get("interaction_mode") in (None, "episode"), name
