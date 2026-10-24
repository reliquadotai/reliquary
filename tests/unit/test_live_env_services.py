"""The RL services take an env added to the running run: its task source, checker, served pin, policy, quota
names, reservation tier and allocation answer, the difficulty auction, and the runtime's per-env loops; a window
frozen before the add never sees it."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.environment import bridged_episode, registry  # noqa: E402
from reliquary.protocol.service_schedule import next_schedule  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    ADDED, ADDED_IDENTITY, EPISODE, POOL, added_block, make_test_episode_env,
)
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402
from tests.unit.test_trajectory_parse import EOT, TERM  # noqa: E402

SHARES = {MATH: 4000, EPISODE: 3000, ADDED: 3000}


@pytest.fixture(autouse=True)
def fresh_catalogue(monkeypatch):
    monkeypatch.setattr(registry, "_CATALOGUE_SPECS", {})


def serve(services, policy, **kwargs):
    """As the validator does: the check in a worker thread (it builds the renderer), then the add on the loop."""
    services.environment_refusal(policy)
    services.add_environment(ADDED, make_test_episode_env(), policy, **kwargs)


def added_policy(rt):
    view = next_schedule(rt.order_contract, rt.schedule, add={ADDED: added_block()}).contract_view(rt.order_contract)
    return view.episode_policy(ADDED)


def add_to_the_run(rt, *, active=False, window=1, request_id="add-1"):
    """Apply a schedule that adds ADDED (inactive unless ``active``); returns its policy from the runtime's view."""
    if active:
        schedule = next_schedule(rt.order_contract, rt.schedule, add={ADDED: added_block()},
                                 active=[MATH, EPISODE, ADDED], shares=SHARES)
    else:
        schedule = next_schedule(rt.order_contract, rt.schedule, add={ADDED: added_block()})
    rt.apply_schedule(schedule, request_id=request_id, window=window)
    return rt.contract.episode_policy(ADDED)


def test_the_rl_services_serve_an_added_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = added_policy(rt)
    assert services.environment_refusal(policy) is None
    serve(services, policy)
    assert ADDED in services.environments and ADDED in services.intake.environments
    assert services.issuer._engagements["rl_precommit"]._environments(ADDED).policy == policy
    assert services.views[ADDED].policy == policy and services.policies[ADDED] == policy
    assert services.served[ADDED][1] == ADDED_IDENTITY
    serve(services, policy)        # idempotent
    assert services.environments.count(ADDED) == 1


def test_an_env_the_proof_could_never_judge_is_refused(tmp_path):
    services, rt = _services(tmp_path)
    services.proof_stop_ids = frozenset({EOT})                             # TERM, a renderer stop, is outside it
    assert services.environment_refusal(added_policy(rt)) is not None


def test_an_env_whose_tools_the_renderer_does_not_support_is_refused(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})

    def no_renderer(policy):
        raise ValueError(f"no renderer for tools {policy.tools}")

    services.renderer_for = no_renderer
    assert "renderer" in services.environment_refusal(added_policy(rt))


def test_added_envs_run_the_difficulty_auction():
    from reliquary.validator.batcher import _difficulty_auction_environment

    assert not _difficulty_auction_environment(ADDED)
    registry.register_catalogue_spec(bridged_episode.catalogue_spec(ADDED, "e" * 64))
    assert _difficulty_auction_environment(ADDED)
    from reliquary.constants import DIFFICULTY_AUCTION_ENVIRONMENTS

    assert all(_difficulty_auction_environment(name) for name in DIFFICULTY_AUCTION_ENVIRONMENTS)   # as before


def test_an_env_the_services_cannot_serve_is_never_half_added(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    services.proof_stop_ids = frozenset({EOT})                             # TERM outside the proof's stop set
    before = (dict(services.views), dict(services.policies), dict(services.served), services.environments)
    with pytest.raises(ValueError, match="outside the proof's stop set"):
        serve(services, added_policy(rt))
    assert (dict(services.views), dict(services.policies), dict(services.served), services.environments) == before
    assert ADDED not in services.intake.environments


def test_a_larger_seed_pool_than_the_services_were_sized_for_is_refused(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = added_policy(rt)
    assert services.pool_seeds == policy.pool_seeds
    bigger = dataclasses.replace(policy, pool_seeds=policy.pool_seeds + 2)
    assert "seed pool" in services.environment_refusal(bigger)


def test_a_name_with_another_group_size_is_refused(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    launched = services.policies[EPISODE]
    # The added env under the launch env's sandbox env name, with another M: the quota could not tell them apart.
    clash = dataclasses.replace(added_policy(rt), sandbox_env=launched.sandbox_env,
                                pool_seeds=launched.pool_seeds - 2)
    assert "group sizes" in services.environment_refusal(clash)


def test_the_same_name_with_another_policy_is_never_taken_as_already_added(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = added_policy(rt)
    serve(services, policy)
    with pytest.raises(ValueError, match="another policy"):
        serve(services, dataclasses.replace(policy, max_turns=1))


def test_the_policy_must_be_the_runtime_views_once_the_schedule_names_the_env(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = add_to_the_run(rt)
    with pytest.raises(ValueError, match="schedule"):
        serve(services, dataclasses.replace(policy, max_turns=1))
    serve(services, policy)
    assert services.policies[ADDED] is policy


def test_the_yield_quota_tiers_and_allocation_cover_an_added_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    policy = added_policy(rt)
    m = policy.pool_seeds // 2
    serve(services, policy)
    quota = services.quota
    # M by both names, and the per-env cap's canonical name
    assert quota._group_sizes[ADDED] == quota._group_sizes[policy.sandbox_env] == m
    assert quota._env_names[ADDED] == quota._env_names[policy.sandbox_env] == policy.sandbox_env
    # the fleet size and the saturation view count its machines at its own reservation, once its tier is settled
    assert services.tiers._policies is services.policies
    assert ADDED in services.tier_pending
    assert (policy.sandbox_env, policy.env_package) not in services.reservations()
    asyncio.run(services._seed_added(ADDED))
    assert (policy.sandbox_env, policy.env_package) in services.reservations()
    assert ADDED in services.tiers.view()
    # what a miner is told about its groups names the added env
    assert services.pool_sizes[ADDED] == policy.pool_seeds
    # every task the added env's view resolves is observed by the tiers
    seen = []
    services.tiers.observe = lambda env, task: seen.append(env)
    asyncio.run(services.views[ADDED].resolve_task(0))
    assert seen == [ADDED] and services.resolvers[ADDED] is services.views[ADDED].resolve_task
    assert services.task_counts[ADDED] == len(make_test_episode_env().source)
    services.ledger.close()


def test_the_allocation_route_answers_for_an_added_env():
    from reliquary.validator.rl_sandbox_wiring import _allocation_of

    quota = SimpleNamespace(capacity_known=lambda: True, sessions_allowed=lambda hotkey: 48,
                            lane=lambda hotkey: "main")
    sizes = {EPISODE: 16}
    allocation_of = _allocation_of(quota, sizes, clock=lambda: 0.0, ttl_s=0.0)
    assert set(allocation_of("5Hot")["groups"]) == {EPISODE}
    sizes[ADDED] = 16                                                       # added while the route serves
    answer = allocation_of("5Hot")
    assert answer["groups"] == {EPISODE: 3, ADDED: 3} and answer["pool_seeds"] == sizes


async def test_an_added_envs_reservation_tier_is_seeded_from_its_own_tasks(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    services._loop = asyncio.get_running_loop()
    serve(services, added_policy(rt))
    await asyncio.gather(*list(services._tasks))
    assert services.tiers.seeded(ADDED)
    services.ledger.close()


def test_the_saturation_round_watches_an_added_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    serve(services, added_policy(rt))
    asked = []
    services.fleet.directory_ready = lambda *a: True
    services.fleet.serving = lambda sandbox_env, package, **k: asked.append((sandbox_env, package)) or ["m"]
    services.check_saturation()
    policy = services.policies[ADDED]
    assert (policy.sandbox_env, policy.env_package) in asked
    services.ledger.close()


# --- an added env inactive, or in a window frozen before the add ---

def test_a_window_frozen_before_the_add_never_takes_a_precommit_of_it(tmp_path):
    from reliquary.protocol.service_episode import EpisodePrecommit
    from reliquary.services.runtime import EpisodePrecommitRefused
    from tests.unit.episode_v2_fixtures import episode_precommit

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = add_to_the_run(rt, active=True)                                # applied for the NEXT window
    serve(services, policy)
    precommit = EpisodePrecommit.from_dict({**episode_precommit(rt.contract, hotkey="5Hot").to_dict(),
                                            "environment": ADDED})
    with pytest.raises(EpisodePrecommitRefused) as refused:
        rt.record_episode_precommit(precommit)                              # window 1 was frozen before the add
    assert refused.value.reason == "environment_not_active"


def test_an_inactive_added_env_gets_no_pool_and_no_precommit(tmp_path):
    from reliquary.protocol.service_episode import EpisodePrecommit
    from reliquary.services.runtime import EpisodePrecommitRefused, protocol_slot_geometry
    from tests.unit.episode_v2_fixtures import episode_precommit

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    serve(services, add_to_the_run(rt))
    assert rt.schedule.share_bps(ADDED) == 0 and ADDED not in rt.schedule.active_environments()
    picks, slots = protocol_slot_geometry()
    with pytest.raises(ValueError, match="active"):                        # no pool may name it
        rt.open_window(2, pools={**{n: POOL for n in rt.schedule.active_environments()}, ADDED: 0.0},
                       picks_target=picks, batch_slots=slots)
    envelope = rt.open_window(2, pools={n: POOL for n in rt.schedule.active_environments()}, picks_target=picks,
                              batch_slots=slots)
    assert ADDED not in envelope["pools"]
    precommit = EpisodePrecommit.from_dict({**episode_precommit(rt.contract, hotkey="5Hot", window=2).to_dict(),
                                            "environment": ADDED})
    with pytest.raises(EpisodePrecommitRefused) as refused:
        rt.record_episode_precommit(precommit)
    assert refused.value.reason == "environment_not_active"


async def test_the_intake_judges_a_group_by_its_own_windows_frozen_contract(tmp_path):
    from reliquary.protocol.service_catalogue import announced_contract
    from reliquary.protocol.submission import RejectReason
    from tests.unit.episode_v2_fixtures import WINDOW_BEACON

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    frozen = announced_contract(rt.announcement(window=1, randomness=WINDOW_BEACON))
    serve(services, add_to_the_run(rt, active=True))
    assert frozen.environments.get(ADDED) is None
    request = SimpleNamespace(window_start=1, miner_hotkey="5Hot", rollouts=[])
    prepared = SimpleNamespace(request=request)
    prepared, claim = await services.intake.admit(environment=ADDED, prepared=prepared, received=0.0,
                                                  contract=frozen)
    assert claim is None
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.GENERATION_CONTRACT_MISMATCH,
                                                               "episode_policy_stale")


def test_the_runtimes_per_env_loops_keep_an_inactive_added_env_consistent(tmp_path):
    from types import SimpleNamespace as G

    from tests.unit.episode_v2_fixtures import episode_runtime

    def journal(rt):
        for key, prompts in ((0, [1]), (1, [2]), (2, [1, 2]), (3, [3])):
            rt.record_training_journal(key, b"payload-%d" % key, is_tombstone=False, stride=2,
                                       batches={MATH: [G(prompt_idx=p) for p in prompts]})

    plain = episode_runtime(tmp_path / "plain")
    journal(plain)
    expected = plain.record_consumption(3)
    assert expected                                                     # measured from the journal
    rt = episode_runtime(tmp_path / "added")
    add_to_the_run(rt)
    journal(rt)
    measured = rt.record_consumption(3)
    assert measured == {**expected, ADDED: 0.0}                         # measured, never invented, never paid
    populations = {name: int(spec["dataset"]["rows"]) for name, spec in rt.contract.environments.items()}
    advice = rt.refresh_cooldown_advice(window=1, populations=populations)
    assert advice[ADDED]["current_windows"] == rt.schedule.cooldown_windows(ADDED) == 10
    assert advice[ADDED]["status"] != "error"
    for name in (MATH, EPISODE):                                        # the order's envs as without the add
        assert advice[name] == plain.refresh_cooldown_advice(window=1, populations={
            n: int(s["dataset"]["rows"]) for n, s in plain.contract.environments.items()})[name]
    assert set(rt.cooldown_advice()) == {MATH, EPISODE, ADDED}


def test_the_quota_refuses_a_name_it_knows_with_another_size_or_env_and_changes_nothing():
    from reliquary.sandbox.rl_quota import YieldQuota

    quota = YieldQuota(ledger=None, pool_seeds=16, group_size=8, capacity=lambda: 0,
                       group_sizes={EPISODE: 8, "swe": 8}, env_names={EPISODE: "swe", "swe": "swe"})
    with pytest.raises(ValueError, match="two group sizes"):
        quota.add_environment({ADDED: 8, "swe": 4}, {ADDED: "swe"})
    with pytest.raises(ValueError, match="names two envs"):
        quota.add_environment({ADDED: 8}, {ADDED: "terminal", EPISODE: "terminal"})
    assert ADDED not in quota._group_sizes and ADDED not in quota._env_names
    quota._capacity_cache = (0.0, 5)
    quota.add_environment({ADDED: 8, "terminal": 8}, {ADDED: "terminal", "terminal": "terminal"})
    assert quota._group_sizes[ADDED] == 8 and quota._env_names[ADDED] == "terminal"
    assert quota._capacity_cache is None and quota._dirty


# --- review fixes ---

def second_block(**episode):
    return added_block(catalogue={"package": "reliquary-other==0.1.0"},
                       episode={"sandbox_env": "reliquary-other",
                                "env_package": "reliquary-other==0.1.0+g0123456789abcdef.vfb2", **episode})


def test_an_added_sandbox_env_named_like_a_launch_env_is_refused_by_the_schedule(tmp_path):
    from reliquary.protocol.service_schedule import ScheduleError
    from tests.unit.episode_v2_fixtures import episode_runtime

    rt = episode_runtime(tmp_path)
    with pytest.raises((ScheduleError, ValueError), match="name of environment"):
        next_schedule(rt.order_contract, rt.schedule, add={ADDED: added_block(episode={"sandbox_env": EPISODE})})


def test_an_added_env_named_like_an_earlier_adds_sandbox_env_or_the_reverse_is_refused(tmp_path):
    from tests.unit.episode_v2_fixtures import episode_runtime

    rt = episode_runtime(tmp_path)
    first = next_schedule(rt.order_contract, rt.schedule, add={ADDED: added_block()})
    with pytest.raises(ValueError, match="name of environment"):
        next_schedule(rt.order_contract, first, add={"tmax_v3": second_block(sandbox_env=ADDED)})
    renamed = next_schedule(rt.order_contract, rt.schedule,
                            add={ADDED: added_block(episode={"sandbox_env": "tmax_v3"})})
    with pytest.raises(ValueError, match="sandbox env of environment"):
        next_schedule(rt.order_contract, renamed, add={"tmax_v3": second_block()})


@pytest.mark.parametrize("first_sandbox, second_sandbox", [("reliquary-terminal", ADDED), ("tmax_v3", None)])
def test_two_adds_of_one_request_never_share_a_name(tmp_path, first_sandbox, second_sandbox):
    from tests.unit.episode_v2_fixtures import episode_runtime

    rt = episode_runtime(tmp_path)
    second = second_block() if second_sandbox is None else second_block(sandbox_env=second_sandbox)
    with pytest.raises(ValueError, match="name of environment|sandbox env of environment"):
        next_schedule(rt.order_contract, rt.schedule,
                      add={ADDED: added_block(episode={"sandbox_env": first_sandbox}), "tmax_v3": second})


def test_the_refusal_covers_the_quotas_name_map_and_the_add_agrees(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    clash = dataclasses.replace(added_policy(rt), sandbox_env=EPISODE)        # the launch env's name
    assert "already names env" in services.environment_refusal(clash)
    with pytest.raises(ValueError, match="already names env"):
        serve(services, clash)
    assert ADDED not in services.policies and ADDED not in services.quota._env_names
    services.ledger.close()


def test_add_environment_never_builds_a_renderer_on_the_event_loop(tmp_path):
    import dataclasses

    from tests.unit.test_trajectory_parse import FakeRenderer

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    built = []

    def loading(policy):                                                    # a tokenizer load in production
        built.append(tuple(policy.tools))
        return FakeRenderer()

    services.renderer_for = loading
    policy = dataclasses.replace(added_policy(rt), tools=("bash", "python"))   # a tool set not built yet
    with pytest.raises(ValueError, match="no renderer was built"):
        services.add_environment(ADDED, make_test_episode_env(), policy)
    assert built == [] and ADDED not in services.views
    assert services.environment_refusal(policy) is None                    # the worker thread's check builds it
    assert built == [("bash", "python")]
    assert services.environment_refusal(policy) is None                    # once per tool set
    assert built == [("bash", "python")]
    services.add_environment(ADDED, make_test_episode_env(), policy)
    assert built == [("bash", "python")] and ADDED in services.views        # reused, never rebuilt


def test_a_renderer_built_off_the_loop_can_be_handed_over(tmp_path):
    import dataclasses

    from tests.unit.test_trajectory_parse import FakeRenderer

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    services.renderer_for = lambda policy: pytest.fail("built on the loop")
    policy = dataclasses.replace(added_policy(rt), tools=("python",))
    renderer = FakeRenderer()
    services.add_environment(ADDED, make_test_episode_env(), policy, renderer=renderer)
    assert services.intake._checkers[ADDED]._renderer is renderer


async def test_an_unseeded_added_env_leaves_the_launch_capacity_unchanged_and_takes_no_group(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    before = services.reservations()
    services._loop = asyncio.get_running_loop()
    serve(services, added_policy(rt))
    assert services.reservations() == before                               # its whole-box tier never counts
    value = SimpleNamespace(environment=ADDED, task_index=0)
    assert await services.precommit_admission(value) == ("fleet_unavailable",
                                                         {"environment": ADDED, "why": "tier_unseeded"})
    assert ADDED not in services.allocation_pool_sizes()
    await asyncio.gather(*list(services._tasks))
    assert not services.tier_pending and services.tiers.seeded(ADDED)
    assert set(services.reservations()) == set(before) | {(services.policies[ADDED].sandbox_env,
                                                           services.policies[ADDED].env_package)}
    assert ADDED in services.allocation_pool_sizes()
    services.ledger.close()


def test_only_the_current_windows_envs_count(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    policy = added_policy(rt)
    serve(services, policy)
    asyncio.run(services._seed_added(ADDED))
    added_key = (policy.sandbox_env, policy.env_package)
    launch = services.policies[EPISODE]
    launch_key = (launch.sandbox_env, launch.env_package)
    assert services.counted_envs() is None and set(services.reservations()) == {launch_key, added_key}
    asked = []
    services.fleet.session_room = lambda now=None, *, reservations=None: asked.append(reservations) or (0, 0)
    services.set_active_envs([EPISODE])                                     # the window's frozen pools
    assert set(services.reservations()) == {launch_key}
    assert services.allocation_pool_sizes() == {EPISODE: launch.pool_seeds}
    services.quota._capacity_cache = None
    services.quota._capacity_of()                                            # C: the quota's own fleet size
    assert set(asked[-1]) == {launch_key}
    services.free_fraction()
    assert set(asked[-1]) == {launch_key}
    services.set_active_envs(None)                                          # never called: every env, as before
    assert set(services.reservations()) == {launch_key, added_key}
    services.ledger.close()


def test_the_window_batcher_runs_the_auction_on_an_added_env():
    from tests.unit.test_grpo_window_batcher import FakeEnv, _make_batcher

    class Added(FakeEnv):
        name = ADDED

    registry.register_catalogue_spec(bridged_episode.catalogue_spec(ADDED, "e" * 64))
    assert _make_batcher(env=Added()).difficulty_auction_enabled is True


def test_seeding_an_added_env_never_moves_a_launch_envs_frozen_tier(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    asyncio.run(services.seed_tiers())
    services.tiers._seeded[EPISODE]["cpu_millis"] = 123_456                  # frozen at start
    serve(services, added_policy(rt))
    asyncio.run(services._seed_added(ADDED))
    assert services.tiers._seeded[EPISODE]["cpu_millis"] == 123_456
    services.ledger.close()


def test_a_second_add_of_the_same_env_changes_nothing(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = added_policy(rt)
    serve(services, policy)
    checker, view = services.intake._checkers[ADDED], services.views[ADDED]
    serve(services, policy)
    assert services.intake._checkers[ADDED] is checker and services.views[ADDED] is view


def test_a_remote_proof_worker_with_another_stop_set_refuses_the_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    services.remote_stop_ids = frozenset({TERM})
    assert "remote proof worker" in services.environment_refusal(added_policy(rt))


@pytest.mark.parametrize("watching", [True, False])
def test_an_env_added_while_no_machine_serves_it_is_alerted(tmp_path, caplog, watching):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, boot_wait_s=1.0 if watching else None)
    services.fleet.directory_ready = lambda *a: True
    services.fleet.serving = lambda *a, **k: []
    services.fleet.reported_packages = lambda env: []
    with caplog.at_level("ERROR"):
        serve(services, added_policy(rt))
    assert any("ALERT" in r.message and ADDED in r.message for r in caplog.records)
    assert (ADDED in services.unserved_envs) is watching


def test_a_source_failing_while_it_is_read_leaves_nothing_half_added(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    policy = added_policy(rt)
    services.environment_refusal(policy)
    environment = make_test_episode_env()

    class Broken(type(environment.source)):
        def __len__(self):
            raise RuntimeError("the dataset is gone")

    environment.source.__class__ = Broken
    with pytest.raises(ValueError, match="the dataset is gone"):
        services.add_environment(ADDED, environment, policy)
    assert ADDED not in services.policies and ADDED not in services.served and ADDED not in services.pool_sizes
    assert ADDED not in services.quota._env_names
    services.ledger.close()


@pytest.mark.parametrize("error", [ConnectionError("tokenizer download failed"), TimeoutError("hub timed out")])
def test_a_renderer_that_cannot_be_loaded_now_is_raised_not_answered(tmp_path, error):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})

    def loading(policy):
        raise error

    services.renderer_for = loading
    with pytest.raises(type(error)):
        services.environment_refusal(added_policy(rt))      # the caller waits on it: no verdict on the env


# ---- the dry run (``add_check``): what ``add_environment`` would refuse, refused first, nothing changed ----

def broken_env(**methods):
    environment = make_test_episode_env()
    environment.source.__class__ = type("Broken", (type(environment.source),), methods)
    return environment


def raising(error):
    def method(self, *a):
        raise error
    return property(method) if isinstance(error, KeyError) else method


@pytest.mark.parametrize("source", ["len", "options", "renderer", "policy", "stops", "names"])
def test_the_dry_run_refuses_each_thing_the_add_would(tmp_path, source):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    policy, environment = added_policy(rt), make_test_episode_env()
    if source == "len":
        environment = broken_env(__len__=raising(RuntimeError("the dataset is gone")))
    elif source == "options":
        environment = broken_env(options_sha256=raising(KeyError("options")))
    elif source == "renderer":
        services.renderer_for = None                          # no renderer can be built for its tools
    elif source == "policy":
        import dataclasses

        add_to_the_run(rt)                                    # the runtime's view names it with its own policy
        policy = dataclasses.replace(policy, max_turns=policy.max_turns - 1)
    elif source == "stops":
        services.remote_stop_ids = frozenset({TERM})
    else:
        import dataclasses

        policy = dataclasses.replace(policy, sandbox_env=EPISODE)      # its sandbox env is another env's name
    why = services.add_check(ADDED, environment, policy)
    assert why is not None, source
    with pytest.raises(ValueError):
        services.add_environment(ADDED, environment, policy)
    assert ADDED not in services.policies and ADDED not in services.views
    services.ledger.close()


def test_the_dry_run_passes_what_the_add_takes(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy, environment = added_policy(rt), make_test_episode_env()
    assert services.add_check(ADDED, environment, policy) is None
    assert ADDED not in services.views                       # a dry run changes nothing
    services.add_environment(ADDED, environment, policy)
    assert services.add_check(ADDED, environment, policy) is None      # served with this policy


def test_a_transient_source_error_in_the_dry_run_is_raised(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    environment = broken_env(__len__=raising(TimeoutError("the dataset read timed out")))
    with pytest.raises(TimeoutError):
        services.add_check(ADDED, environment, added_policy(rt))
