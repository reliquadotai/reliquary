"""Miners build their prompts from the announced catalogue, and every miner-side reader of the announcement
serves the envs its schedule added."""
import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox_verifiers")
bt = pytest.importorskip("bittensor")

from reliquary.environment.bridged_episode import BridgedEpisodeSource  # noqa: E402
from reliquary.environment.registry import ENVIRONMENT_SPECS  # noqa: E402
from reliquary.miner import episode_mining as em  # noqa: E402
from reliquary.protocol.service_schedule import ScheduleError, next_schedule  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    ADDED,
    ADDED_IDENTITY,
    CATALOGUE,
    CATALOGUE_IMAGES,
    CATALOGUE_SPLIT,
    ENV_PACKAGE,
    EPISODE,
    PROMPT_TEXT,
    WINDOW_BEACON,
    added_block,
    episode_contract,
    episode_runtime,
)
from tests.unit.service_v2_fixtures import MATH  # noqa: E402

IMAGE = CATALOGUE_IMAGES["base"]
SHARES = {MATH: 4000, EPISODE: 3000, ADDED: 3000}


class FakeVerifiersEnv:
    def __init__(self, split, rows=40, text=PROMPT_TEXT):
        self.split, self.rows, self.text = split, rows, text
        self.lock = SimpleNamespace(images=dict(CATALOGUE_IMAGES))     # the catalogue's lock digest

    def taskset(self, split):
        assert split == self.split
        return list(range(self.rows))

    def task(self, split, index):
        assert split == self.split
        return SimpleNamespace(data=SimpleNamespace(prompt=self.text))

    def sandbox_task(self, split, index):
        return SimpleNamespace(image=IMAGE, limits=SimpleNamespace(cpus=None, memory_bytes=None))


def build_source(identities):
    def build(config):
        return BridgedEpisodeSource(config, env=FakeVerifiersEnv(config.rl_split),
                                    identity=identities[config.environment])
    return build


def forbidden(name):
    raise AssertionError(f"the static registry was used for {name}")


def with_added(contract, schedule, *, active=True):
    if not active:
        return next_schedule(contract, schedule, add={ADDED: added_block()})
    return next_schedule(contract, schedule, add={ADDED: added_block()}, active=list(SHARES), shares=SHARES)


def announcement(tmp_path, *, added=False, catalogue=CATALOGUE):
    contract = episode_contract(catalogue=catalogue) if catalogue is not None else episode_contract()
    rt = episode_runtime(tmp_path, contract)
    value = rt.announcement(window=1, randomness=WINDOW_BEACON)
    return value if not added else dict(value, schedule=with_added(contract, rt.schedule).to_dict())


# -- the prompts --------------------------------------------------------------------------------------------


def test_a_miner_builds_its_prompts_from_the_announced_catalogue(tmp_path):
    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden, build_source=build_source({EPISODE: ENV_PACKAGE}))
    assert prompts.size(EPISODE) == 0                               # unknown until announced
    prompts.bind(announcement(tmp_path))
    assert prompts(EPISODE, 3) == PROMPT_TEXT and prompts.size(EPISODE) == 40
    assert isinstance(prompts.sources[EPISODE], BridgedEpisodeSource)
    assert prompts.sources[EPISODE].config.rl_split == CATALOGUE_SPLIT


def test_an_install_that_is_not_the_orders_pin_refuses_to_start(tmp_path):
    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden,
                                    build_source=build_source({EPISODE: "reliquary-swe==0.1.0a1+g0.vfb2"}))
    with pytest.raises(ValueError, match="the order pins"):
        prompts.bind(announcement(tmp_path))
    assert prompts.sources == {}


def test_an_installed_lock_that_is_not_the_catalogues_refuses_to_start(tmp_path):
    def build(config):
        env = FakeVerifiersEnv(config.rl_split)
        env.lock = SimpleNamespace(images={"base": IMAGE.replace("a" * 64, "b" * 64)})
        return BridgedEpisodeSource(config, env=env, identity=ENV_PACKAGE)

    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden, build_source=build)
    with pytest.raises(ValueError, match="image lock"):
        prompts.bind(announcement(tmp_path))


def test_an_env_added_to_the_running_run_is_mined_from_the_schedules_catalogue(tmp_path):
    prompts = em.EpisodeTaskPrompts([EPISODE, ADDED], load=forbidden,
                                    build_source=build_source({EPISODE: ENV_PACKAGE, ADDED: ADDED_IDENTITY}))
    plain = announcement(tmp_path / "plain")
    prompts.bind(plain)                                             # not announced yet: nothing for it
    assert ADDED not in prompts.sources and prompts.size(ADDED) == 0
    prompts.bind(announcement(tmp_path / "added", added=True))
    assert prompts.sources[ADDED].config.package == "reliquary-terminal==0.3.0"
    assert prompts.sources[ADDED].config.rl_split == "tmax"
    assert prompts.size(ADDED) == 40 and prompts(ADDED, 0) == PROMPT_TEXT


def test_a_catalogue_that_did_not_change_is_not_rebuilt(tmp_path):
    built = []

    def counting(config):
        built.append(config.environment)
        return build_source({EPISODE: ENV_PACKAGE})(config)

    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden, build_source=counting)
    value = announcement(tmp_path)
    prompts.bind(value)
    prompts.bind(value)
    assert built == [EPISODE]


def test_an_env_without_a_catalogue_keeps_the_registry_path(tmp_path, monkeypatch):
    from tests.unit.episode_v2_fixtures import register_episode_env

    register_episode_env(monkeypatch)
    prompts = em.EpisodeTaskPrompts([EPISODE])
    prompts.bind(announcement(tmp_path, catalogue=None))
    assert prompts(EPISODE, 0) == PROMPT_TEXT and prompts.size(EPISODE) == 1000


def test_an_announced_env_with_neither_a_catalogue_nor_a_registration_refuses_to_start(tmp_path):
    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden)
    with pytest.raises(ValueError, match="no catalogue"):
        prompts.bind(announcement(tmp_path, catalogue=None))


def test_the_episode_envs_option_accepts_names_the_announcement_will_resolve():
    assert em.parse_episode_environments(f"{EPISODE},{ADDED}") == (EPISODE, ADDED)
    assert em.parse_episode_environments(f" {ADDED} ,{ADDED}") == (ADDED,)
    single_turn = next(name for name, spec in ENVIRONMENT_SPECS.items() if spec.interaction_mode != "signed_episode")
    with pytest.raises(ValueError, match="not a signed-episode"):
        em.parse_episode_environments(single_turn)
    with pytest.raises(ValueError, match="not an environment id"):
        em.parse_episode_environments("not an id!")
    with pytest.raises(ValueError, match="names no environment"):
        em.parse_episode_environments(" , ")
    with pytest.raises(ValueError, match="not a signed-episode"):
        em.EpisodeTaskPrompts([MATH])


def test_the_legacy_miner_never_takes_a_catalogue_env(monkeypatch):
    from reliquary.cli import main as cli

    with pytest.raises(ValueError, match="mine-episodes"):
        cli._resolve_legacy_miner_mix(f"openmathinstruct,{ADDED}")
    # An installed env keeps the registry's own answer.
    assert cli._resolve_legacy_miner_mix("openmathinstruct") == cli._resolve_cli_environment_mix("openmathinstruct")
    with pytest.raises(ValueError, match="duplicate") as caught:
        cli._resolve_legacy_miner_mix("openmathinstruct,openmathinstruct")
    assert "mine-episodes" not in str(caught.value)


# -- the stack and the loop: an added env through every miner-side reader --------------------------------


def mining_world(tmp_path):
    from tests.unit.test_episode_mining import World, miner_state

    w = World(tmp_path)
    state = miner_state(w.rt)
    view = em.episode_window(state)
    added = dataclasses.replace(view, announcement=dict(
        view.announcement, schedule=with_added(w.rt.contract, w.rt.schedule).to_dict()))
    return w, view, added


def test_the_stack_binds_the_prompts_and_serves_an_added_env(tmp_path):
    from tests.unit.test_episode_mining import MINER, WALLET, config

    w, view, added = mining_world(tmp_path)
    cfg = config(environments=(EPISODE, ADDED))
    bound = []

    class Prompts:
        def bind(self, value):
            bound.append(value)

        def __call__(self, environment, task):
            return PROMPT_TEXT

    stack = asyncio.run(em.build_episode_stack(added, config=cfg, wallet=WALLET, hotkey=MINER.ss58_address,
                                               sign_binding=lambda b: "", task_prompt=Prompts(), deps=w.deps))
    asyncio.run(stack.close())
    assert bound == [added.announcement]
    # The plain announcement does not name the added env: refusal to start, before anything loads.
    w.events.clear()
    with pytest.raises(ValueError, match="not in this order"):
        asyncio.run(em.build_episode_stack(view, config=cfg, wallet=WALLET, hotkey=MINER.ss58_address,
                                           sign_binding=lambda b: "", task_prompt=Prompts(), deps=w.deps))
    assert w.events == []


def test_an_install_that_is_not_the_pin_refuses_before_anything_loads(tmp_path):
    from tests.unit.test_episode_mining import MINER, WALLET, World, config, miner_state

    w = World(tmp_path)
    view = em.episode_window(miner_state(w.rt))
    view = dataclasses.replace(view, announcement=announcement(tmp_path / "cat"))
    prompts = em.EpisodeTaskPrompts([EPISODE], load=forbidden,
                                    build_source=build_source({EPISODE: "reliquary-swe==0.1.0a1+g0.vfb2"}))
    with pytest.raises(ValueError, match="the order pins"):
        asyncio.run(em.build_episode_stack(view, config=config(), wallet=WALLET, hotkey=MINER.ss58_address,
                                           sign_binding=lambda b: "", task_prompt=prompts, deps=w.deps))
    assert w.downloads == [] and w.events == []


def test_an_added_env_with_other_caps_needs_a_miner_process_of_its_own(tmp_path):
    from tests.unit.test_episode_mining import MINER, WALLET, World, config, miner_state

    w = World(tmp_path)
    view = em.episode_window(miner_state(w.rt))
    block = added_block(episode={"max_tokens_per_turn": 256})
    schedule = next_schedule(w.rt.contract, w.rt.schedule, add={ADDED: block}, active=list(SHARES), shares=SHARES)
    view = dataclasses.replace(view, announcement=dict(view.announcement, schedule=schedule.to_dict()))
    with pytest.raises(ValueError, match="miner process of its own"):
        asyncio.run(em.build_episode_stack(view, config=config(environments=(EPISODE, ADDED)), wallet=WALLET,
                                           hotkey=MINER.ss58_address, sign_binding=lambda b: "",
                                           task_prompt=lambda e, t: "", deps=w.deps))
    assert w.downloads == [] and w.events == []
    # Alone, it has its own engine.
    stack = asyncio.run(em.build_episode_stack(view, config=config(environments=(ADDED,)), wallet=WALLET,
                                               hotkey=MINER.ss58_address, sign_binding=lambda b: "",
                                               task_prompt=lambda e, t: "", deps=w.deps))
    assert stack.engine_caps["max_tokens_per_turn"] == 256
    asyncio.run(stack.close())


def test_the_loop_sizes_an_added_env_from_its_schedule_row(tmp_path):
    w, view, added = mining_world(tmp_path)

    async def never():
        return None

    loop = em.EpisodeMiningLoop(environments=(EPISODE, ADDED), read_state=never, build_stack=None,
                                sizes=lambda name: 100_000)
    rows = int(w.rt.contract.environments[EPISODE]["dataset"]["rows"])
    assert loop._sizes_of(added) == {EPISODE: min(rows, 100_000), ADDED: 2588}      # the added row's dataset
    assert loop._sizes_of(view) == {EPISODE: min(rows, 100_000)}


def test_the_preflight_judges_the_added_envs_of_the_new_announcement(tmp_path, monkeypatch):
    from tests.unit.test_episode_mining import WALLET, config

    w, view, added = mining_world(tmp_path)
    captured = {}

    class Loop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self, *, stop=None):
            return None

    monkeypatch.setattr(em, "EpisodeMiningLoop", Loop)

    async def read_state():
        return None

    asyncio.run(em.run_episode_miner(config=config(environments=(EPISODE, ADDED)), wallet=WALLET, deps=w.deps,
                                     task_prompts=em.EpisodeTaskPrompts([EPISODE, ADDED], load=forbidden),
                                     read_state=read_state))
    preflight = captured["preflight"]
    assert preflight(added) is None
    assert "not in this order" in preflight(view)


# -- the group miner -----------------------------------------------------------------------------------------


def test_the_group_miner_mines_an_added_env(tmp_path):
    from reliquary.constants import M_ROLLOUTS
    from reliquary.protocol.seed_pool import pool_from_service_policy
    from reliquary.protocol.service_episode import EpisodePrecommit
    from tests.unit.test_episode_group_miner import Clock, alternating, world

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), clock=Clock())
    plain = w.rt.announcement(window=1, randomness=WINDOW_BEACON)
    value = dict(plain, schedule=with_added(w.rt.contract, w.rt.schedule).to_dict())

    def mine(announcement_):
        return asyncio.run(w.miner.mine_task(announcement=announcement_, randomness="cd" * 32, window=1,
                                             environment=ADDED, task_index=3, open_until=None))

    with pytest.raises(ValueError, match="not in this order"):
        mine(plain)
    mine(value)
    (body,) = w.miner._sessions.bodies
    precommit = EpisodePrecommit.from_dict(body["precommit"])
    pool = pool_from_service_policy(value, environment=ADDED, prompt_idx=3,
                                    checkpoint_hash=plain["checkpoint"]["revision"])
    assert (precommit.environment, precommit.task_index) == (ADDED, 3)
    assert precommit.pool_sha256 == pool.sha256 and precommit.order == w.rt.contract.sha256


# -- the legacy engine's two readers ------------------------------------------------------------------------


def test_the_legacy_engine_reads_an_announcement_with_an_added_env(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from reliquary.miner.engine import MiningEngine, maybe_pull_checkpoint

    # The order's checkpoint policy is task-scoped: the runtime that reads it is too.
    monkeypatch.setenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "1")
    monkeypatch.setenv("RELIQUARY_TASK_ID", "next-rl")
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", "run-1")
    rt = episode_runtime(tmp_path)
    plain = rt.announcement(window=1, randomness=WINDOW_BEACON)
    value = dict(plain, schedule=with_added(rt.contract, rt.schedule).to_dict())
    malformed = dict(plain, schedule=with_added(rt.contract, rt.schedule).to_dict())
    malformed["schedule"]["environments"][ADDED]["added"]["environment"]["episode"]["max_turns"] = 99

    def pull(policy):
        state = SimpleNamespace(service_policy=policy, checkpoint_n=0, checkpoint_repo_id=None,
                                checkpoint_revision=None)
        return asyncio.run(maybe_pull_checkpoint(state, 0, "", "", "cached", download_fn=AsyncMock(),
                                                 load_fn=MagicMock()))

    assert pull(value) == pull(plain) == (0, "", "", "cached")
    with pytest.raises(ScheduleError):
        pull(malformed)

    engine = object.__new__(MiningEngine)
    engine.max_new_tokens = 2
    engine.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[0]))
    env = SimpleNamespace(name=MATH, compute_reward=lambda problem, text: None)
    generations = [{"tokens": [99, 0], "prompt_length": 1}] * 8
    binding = engine._service_submission_binding(value, generations, {}, env)
    assert binding == engine._service_submission_binding(plain, generations, {}, env)
    assert binding["contract_sha256"] == rt.contract.sha256
    with pytest.raises(ScheduleError):
        engine._service_submission_binding(malformed, generations, {}, env)
