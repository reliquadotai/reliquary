"""A signed-sandbox SWE job reads its prompt, task id, image and limits from the same
bridged env a gateway serves (reliquary-swe through the sandbox's verifiers bridge)."""

import asyncio
import types

import pytest

from reliquary.environment import bridged_swe


class FakeEnv:
    def __init__(self):
        self.asked = []

    def task(self, split, index):
        self.asked.append((split, index))
        return types.SimpleNamespace(data=types.SimpleNamespace(prompt=f"prompt {index}",
                                                                instance_id=f"inst-{index}"))

    def sandbox_task(self, split, index):
        limits = types.SimpleNamespace(memory_bytes=4 << 30, disk_bytes=10 << 30, pids=1024,
                                       wall_s=3600, per_call_timeout_s=600, max_calls=None)
        return types.SimpleNamespace(image="r@sha256:" + "a" * 64, limits=limits)


@pytest.fixture
def env(monkeypatch):
    fake = FakeEnv()
    monkeypatch.setattr(bridged_swe, "bridged_env", lambda: fake)
    return fake


def test_the_prompt_and_the_task_id_come_from_the_bridged_task(env):
    assert bridged_swe.prompt_of("train", 3) == "prompt 3"
    assert bridged_swe.row_of("train", 4)[1].instance_id == "inst-4"
    assert env.asked == [("train", 3), ("train", 4)]


def test_the_signed_source_reads_the_bridged_env_by_default(env):
    from reliquary.environment.agentic_swe import SignedSweSource

    source = SignedSweSource("train")
    assert source.prompt(2) == "prompt 2" and source.instance_id(2) == "inst-2"


def test_the_resolver_reads_the_bridged_sandbox_task(env):
    from reliquary.sandbox.tasks import SweTaskResolver

    resolved = asyncio.run(SweTaskResolver("train").resolve(1))
    assert resolved.image == "r@sha256:" + "a" * 64
    assert resolved.limits["per_call_timeout_s"] == 600 and "max_calls" not in resolved.limits


def test_the_identity_is_the_bridged_one(monkeypatch):
    episodes = pytest.importorskip("reliquary_sandbox_service.episodes")
    monkeypatch.setattr(episodes, "bridged_env_package_of", lambda name: f"{name}-bridged")
    assert bridged_swe.installed_env_package("reliquary-swe") == "reliquary_swe-bridged"


def test_a_signed_job_names_the_split_the_gateway_serves():
    from reliquary.corpus.job import sandbox_split

    assert sandbox_split(types.SimpleNamespace(env=types.SimpleNamespace(num_images=20))) == "train"
    with pytest.raises(ValueError, match="20"):
        sandbox_split(types.SimpleNamespace(env=types.SimpleNamespace(num_images=10)))


def test_the_bridged_env_is_built_with_the_gateway_options(monkeypatch):
    loading = pytest.importorskip("reliquary_sandbox_verifiers.loading")
    built = []
    monkeypatch.setattr(loading, "VerifiersEnv", lambda entry, options: built.append((entry, options)))
    monkeypatch.setattr(bridged_swe.importlib.metadata, "version", lambda name: "0.3.0")
    bridged_swe._env.cache_clear()
    try:
        bridged_swe.bridged_env()
    finally:
        bridged_swe._env.cache_clear()
    options = loading.VerifiersEnvOptions.from_mapping(built[0][1])
    assert built[0][0] == "verifiers:reliquary-swe==0.3.0"
    assert set(options.splits) == {"train", "r2e", "polyglot"}
    assert options.tools == ("bash", "edit") and options.defaults.per_call_timeout_s == 600
