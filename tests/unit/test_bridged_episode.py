"""Any verifiers env served through the sandbox bridge, as a signed-episode env, described by a public
catalogue in its order entry."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from reliquary.environment import bridged_episode, registry
from reliquary.environment.bridged_episode import (
    BridgedEnvConfig,
    BridgedEpisodeSource,
    bridged_manifest_sha256,
    build_environment,
    catalogue_spec,
    describe,
    image_lock_sha256,
)
from reliquary.environment.signed_episode import SignedEpisodeEnvironment
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError
from reliquary.sandbox.tasks import ResolvedTask
from tests.unit.episode_v2_fixtures import (
    BUDGETS,
    CATALOGUE,
    CATALOGUE_IMAGES,
    ENV_PACKAGE,
    EPISODE,
    episode_contract,
    episode_contract_dict,
)

IMAGE = "registry.example/tmax@sha256:" + "a" * 64
IDENTITY = "reliquary-terminal==0.3.0+g827598fa320ad040.vfb2"
OPTIONS = {"tools": ["bash"], "splits": {"tmax": {"split": "tmax"}}}
CONFIG = BridgedEnvConfig(environment="tmax_v2", package="reliquary-terminal==0.3.0", rl_split="tmax",
                          options=OPTIONS)
GIB = 1024**3


class FakeEnv:
    """What BridgedEpisodeSource reads of a VerifiersEnv."""

    def __init__(self, rows=5, sized=True, images=None, split="tmax"):
        self.rows, self.sized, self.split = rows, sized, split
        self.lock = SimpleNamespace(images=images or {"tmax-base": IMAGE})
        self.sandbox_calls = 0

    def taskset(self, split):
        assert split == self.split
        return list(range(self.rows)) if self.sized else (i for i in range(self.rows))

    def task(self, split, index):
        if not 0 <= index < self.rows:
            raise IndexError(index)
        return SimpleNamespace(data=SimpleNamespace(prompt=f"task {index}"))

    def sandbox_task(self, split, index):
        self.sandbox_calls += 1
        limits = SimpleNamespace(cpus=4.0, memory_bytes=2 * GIB, disk_bytes=None, pids=512, wall_s=None,
                                 per_call_timeout_s=None, max_calls=None)
        return SimpleNamespace(image=IMAGE, limits=limits, prepare_timeout_s=900.0)


def source(env=None):
    pytest.importorskip("reliquary_sandbox_verifiers")
    return BridgedEpisodeSource(CONFIG, env=env or FakeEnv(), identity=IDENTITY)


def test_the_source_serves_prompts_rows_and_resolved_tasks_with_their_options_digest():
    src = source()
    assert len(src) == 5 and src.prompt(2) == "task 2" and src.task_image(2) == IMAGE
    resolved = asyncio.run(src.resolve(2))
    assert resolved == ResolvedTask(IMAGE, {"memory_bytes": 2 * GIB, "pids": 512},
                                    env_options_sha256=src.options_sha256, prepare_timeout_s=900.0,
                                    cpu_millis=4000)
    assert len(src.options_sha256) == 64


def test_the_options_digest_is_the_gateways():
    """The boot check reads ``source.options_sha256``: the digest a gateway serving these options publishes."""
    loading = pytest.importorskip("reliquary_sandbox_verifiers.loading")
    assert source().options_sha256 == loading.options_sha256(OPTIONS)
    other = BridgedEpisodeSource(BridgedEnvConfig(**{**CONFIG.__dict__, "options": {"tools": ["bash", "edit"]}}),
                                 env=FakeEnv(), identity=IDENTITY)
    assert other.options_sha256 != source().options_sha256


def test_a_taskset_without_len_is_counted_once():
    assert len(source(FakeEnv(rows=7, sized=False))) == 7


def test_resolutions_are_cached():
    env = FakeEnv()
    src = source(env)
    asyncio.run(src.resolve(1))
    asyncio.run(src.resolve(1))
    src.task_image(1)
    assert env.sandbox_calls == 1


def test_the_lock_digest_is_the_sandboxs_canonical_one():
    images = {"b": IMAGE, "a": IMAGE}
    expected = hashlib.sha256(
        json.dumps({"version": 1, "images": dict(sorted(images.items()))}, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    assert image_lock_sha256(images) == expected
    image_lock = pytest.importorskip("reliquary_sandbox_verifiers.image_lock")
    if hasattr(image_lock.ImageLock, "sha256"):          # after the sandbox re-pin
        assert image_lock.ImageLock(images=images).sha256() == expected


def test_the_lock_digest_matches_the_sandboxs_golden_vector():
    """The sandbox's own test pins this exact hex for this lock: both sides hash the same bytes."""
    images = {"base": "registry.example/task@sha256:" + "a" * 64,
              "other": "registry.example/other@sha256:" + "b" * 64}
    assert image_lock_sha256(images) == "e5459c4580f905f85e61aa8114aec37194919bbf2484a2443357edb429330731"
    assert image_lock_sha256(dict(reversed(list(images.items())))) == image_lock_sha256(images)


def test_the_manifest_follows_identity_options_lock_split_and_rows():
    src = source()
    base = {"identity": IDENTITY, "options_sha256": src.options_sha256, "lock_sha256": src.lock_sha256,
            "rl_split": "tmax", "rows": 5}
    assert src.manifest_sha256() == bridged_manifest_sha256(**base)
    for field, value in (("identity", IDENTITY + "x"), ("rows", 6), ("rl_split", "rl"), ("lock_sha256", "0" * 64),
                         ("options_sha256", "0" * 64)):
        assert bridged_manifest_sha256(**{**base, field: value}) != src.manifest_sha256()


def test_a_prompt_that_is_not_text_is_refused():
    env = FakeEnv()
    env.task = lambda split, index: SimpleNamespace(data=SimpleNamespace(prompt=[{"role": "user"}]))
    with pytest.raises(TypeError):
        source(env).prompt(0)


def test_the_environment_and_its_spec(monkeypatch):
    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})
    pytest.importorskip("reliquary_sandbox_verifiers")
    environment = build_environment(CONFIG, env=FakeEnv(), identity=IDENTITY)
    assert isinstance(environment, SignedEpisodeEnvironment) and environment.name == "tmax_v2"
    assert len(environment) == 5 and environment.get_problem(1)["prompt"] == "task 1"
    spec = catalogue_spec("tmax_v2", environment.source.manifest_sha256())
    registry.register_catalogue_spec(spec)
    assert registry.get_environment_spec("tmax_v2") is spec is registry.find_environment_spec("tmax_v2")
    assert spec.interaction_mode == "signed_episode"
    assert spec.environment_manifest_sha256 == environment.source.manifest_sha256()
    assert "tmax_v2" in registry.catalogue_environment_names()
    registry.register_catalogue_spec(spec)                                    # the same spec again: fine
    with pytest.raises(ValueError, match="another"):
        registry.register_catalogue_spec(catalogue_spec("tmax_v2", "0" * 64))
    with pytest.raises(ValueError, match="installed"):
        registry.register_catalogue_spec(catalogue_spec(next(iter(registry.ENVIRONMENT_SPECS)), "0" * 64))
    with pytest.raises(TypeError):
        spec.create()                                       # built by build_environment, never by its spec
    from reliquary.services.schedule import default_installed_version
    assert default_installed_version("tmax_v2") == environment.source.manifest_sha256()
    with pytest.raises(ValueError, match="Unknown environment"):
        registry.get_environment_spec("not_an_env")


def test_an_order_entry_carries_its_public_catalogue():
    contract = episode_contract(catalogue=CATALOGUE)
    assert contract.catalogue_of(EPISODE) == CATALOGUE
    assert contract.catalogue_of("missing") is None
    assert BridgedEnvConfig.from_catalogue(EPISODE, CATALOGUE) == BridgedEnvConfig(
        EPISODE, CATALOGUE["package"], CATALOGUE["rl_split"], CATALOGUE["options"], CATALOGUE["image_lock_sha256"])
    conformed = {**CATALOGUE, "conformance_sha256": "e" * 64}
    assert episode_contract(catalogue=conformed).catalogue_of(EPISODE) == conformed


def test_a_catalogue_lock_that_is_not_the_installed_packages_is_refused():
    pytest.importorskip("reliquary_sandbox_verifiers")
    pinned = BridgedEnvConfig(environment="tmax_v2", package="reliquary-terminal==0.3.0", rl_split="tmax",
                              options=OPTIONS, image_lock_sha256=image_lock_sha256({"tmax-base": IMAGE}))
    assert BridgedEpisodeSource(pinned, env=FakeEnv(), identity=IDENTITY).lock_sha256 == pinned.image_lock_sha256
    typo = BridgedEnvConfig(**{**pinned.__dict__, "image_lock_sha256": "0" * 64})
    with pytest.raises(ValueError, match="image lock"):
        BridgedEpisodeSource(typo, env=FakeEnv(), identity=IDENTITY)


def test_an_order_without_a_catalogue_keeps_its_bytes():
    plain = episode_contract()
    assert b"catalogue" not in plain.canonical and plain.catalogue_of(EPISODE) is None
    assert ServiceContract.from_dict(episode_contract_dict()).sha256 == plain.sha256
    assert episode_contract(catalogue=CATALOGUE).sha256 != plain.sha256


@pytest.mark.parametrize("change, why", [
    ({"package": "reliquary-swe"}, "name==version"),
    ({"package": "reliquary-other==0.1.0a1"}, "bridged identity"),
    ({"rl_split": "train:other"}, "split"),
    ({"image_lock_sha256": "x"}, "image_lock_sha256"),
    ({"conformance_sha256": "x"}, "conformance_sha256"),
    ({"options": {"x": "y" * 20_000}}, "options"),
    ({"options": []}, "options"),
    ({"options": {"api_key": "x"}}, "credential"),                        # the catalogue is public
    ({"options": {"splits": {"tmax": {"client_secret": "x"}}}}, "credential"),
    ({"options": {"auth": [{"Bearer": "x"}]}}, "credential"),
    ({"extra": 1}, "catalogue"),
])
def test_a_malformed_catalogue_is_refused(change, why):
    with pytest.raises(ServiceContractError, match=why):
        episode_contract(catalogue={**CATALOGUE, **change})


def test_a_catalogue_needs_a_bridged_identity():
    plain_identity = {**episode_contract_dict()["environments"][EPISODE]["episode"],
                      "env_package": "reliquary-swe==0.1.0a1+g0123456789abcdef"}
    with pytest.raises(ServiceContractError, match="bridged identity"):
        episode_contract(episode=plain_identity, catalogue=CATALOGUE)


def test_a_catalogue_needs_an_episode_block():
    value = episode_contract_dict()
    name = next(n for n in value["environments"] if "episode" not in value["environments"][n])
    value["environments"][name]["catalogue"] = dict(CATALOGUE)          # on the order's single-turn env
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(value)


def test_the_validator_builds_its_launch_episode_envs_from_the_orders_catalogue():
    from reliquary.validator.service import ValidationService

    me = SimpleNamespace(_service_runtime=SimpleNamespace(contract=episode_contract(catalogue=CATALOGUE)))
    assert ValidationService._launch_bridged_configs(me) == {EPISODE: BridgedEnvConfig.from_catalogue(EPISODE, CATALOGUE)}
    me = SimpleNamespace(_service_runtime=SimpleNamespace(contract=episode_contract()))
    assert ValidationService._launch_bridged_configs(me) == {}
    assert ValidationService._launch_bridged_configs(SimpleNamespace(_service_runtime=None)) == {}


def test_the_validator_checks_the_installed_identity_and_registers_the_spec(monkeypatch):
    pytest.importorskip("reliquary_sandbox_verifiers")
    from reliquary.validator.service import ValidationService

    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})
    contract = episode_contract(catalogue=CATALOGUE)
    me = SimpleNamespace(_service_runtime=SimpleNamespace(contract=contract))
    config = BridgedEnvConfig.from_catalogue(EPISODE, CATALOGUE)
    installed = {"identity": ENV_PACKAGE}
    real_build = bridged_episode.build_environment
    monkeypatch.setattr(bridged_episode, "build_environment", lambda config: real_build(
        config, env=FakeEnv(images=CATALOGUE_IMAGES, split=CATALOGUE["rl_split"]), identity=installed["identity"]))
    environment = ValidationService._build_bridged(me, config)
    assert environment.name == EPISODE
    assert registry.find_environment_spec(EPISODE).environment_manifest_sha256 == environment.source.manifest_sha256()
    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})
    installed["identity"] = ENV_PACKAGE.replace(".vfb2", ".vfb3")
    with pytest.raises(ValueError, match="the order pins"):
        ValidationService._build_bridged(me, config)
    assert registry.find_environment_spec(EPISODE) is None


def test_a_launch_catalogue_env_is_checked_when_built_not_at_task_config():
    """The task config cannot know a bridged env's manifest (it is built from the catalogue at start): its cap
    follows its share, and the validator compares the built manifest with the order's version."""
    from reliquary.validator.task_config import TaskConfigError, _service_env_caps
    from tests.unit.service_v2_fixtures import MATH

    entry, shares = SimpleNamespace(task_id="rl"), {MATH: 0, EPISODE: 10000}
    caps = _service_env_caps(entry, cap=0.5, contract=episode_contract(catalogue=CATALOGUE, shares=shares))
    assert caps == {MATH: 0.0, EPISODE: 0.5}
    with pytest.raises(TaskConfigError, match="version mismatch"):              # no catalogue: the registry's check
        _service_env_caps(entry, cap=0.5, contract=episode_contract(shares=shares))


def test_describe_gives_an_entry_and_a_catalogue_the_order_accepts():
    pytest.importorskip("reliquary_sandbox_verifiers")
    episode = {"sandbox_env": "reliquary-swe", "tools": ["bash", "edit"], "max_turns": 8,
               "max_tokens_per_turn": 512, "max_episode_tokens": 4096, "budgets": dict(BUDGETS)}
    block = describe(name=EPISODE, package=CATALOGUE["package"], rl_split=CATALOGUE["rl_split"], options={},
                     episode=episode, cooldown_windows=10, attestation=b"report",
                     env=FakeEnv(images=CATALOGUE_IMAGES, split=CATALOGUE["rl_split"]), identity=ENV_PACKAGE)
    assert block["catalogue"] == {**CATALOGUE, "conformance_sha256": hashlib.sha256(b"report").hexdigest()}
    value = episode_contract_dict()
    share = value["environments"][EPISODE]["share_bps"]
    value["environments"][EPISODE] = {**block["environment"], "share_bps": share, "catalogue": block["catalogue"]}
    contract = ServiceContract.from_dict(value)
    assert contract.episode_policy(EPISODE).env_package == ENV_PACKAGE
    assert contract.environments[EPISODE]["dataset"]["rows"] == 5
