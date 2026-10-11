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
    services.add_environment(ADDED, make_test_episode_env(), policy)
    assert ADDED in services.environments and ADDED in services.intake.environments
    assert services.issuer._engagements["rl_precommit"]._environments(ADDED).policy == policy
    assert services.views[ADDED].policy == policy and services.policies[ADDED] == policy
    assert services.served[ADDED][1] == ADDED_IDENTITY
    services.add_environment(ADDED, make_test_episode_env(), policy)        # idempotent
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
        services.add_environment(ADDED, make_test_episode_env(), added_policy(rt))
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
    services.add_environment(ADDED, make_test_episode_env(), policy)
    with pytest.raises(ValueError, match="another policy"):
        services.add_environment(ADDED, make_test_episode_env(), dataclasses.replace(policy, max_turns=1))


def test_the_policy_must_be_the_runtime_views_once_the_schedule_names_the_env(tmp_path):
    import dataclasses

    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT})
    policy = add_to_the_run(rt)
    with pytest.raises(ValueError, match="schedule"):
        services.add_environment(ADDED, make_test_episode_env(), dataclasses.replace(policy, max_turns=1))
    services.add_environment(ADDED, make_test_episode_env(), policy)
    assert services.policies[ADDED] is policy


def test_the_yield_quota_tiers_and_allocation_cover_an_added_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    policy = added_policy(rt)
    m = policy.pool_seeds // 2
    services.add_environment(ADDED, make_test_episode_env(), policy)
    quota = services.quota
    # M by both names, and the per-env cap's canonical name
    assert quota._group_sizes[ADDED] == quota._group_sizes[policy.sandbox_env] == m
    assert quota._env_names[ADDED] == quota._env_names[policy.sandbox_env] == policy.sandbox_env
    # the fleet size and the saturation view count its machines at its own reservation
    assert services.tiers._policies is services.policies
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
    services.add_environment(ADDED, make_test_episode_env(), added_policy(rt))
    await asyncio.gather(*list(services._tasks))
    assert services.tiers.seeded(ADDED)
    services.ledger.close()


def test_the_saturation_round_watches_an_added_env(tmp_path):
    services, rt = _services(tmp_path, proof_stop_ids={TERM, EOT}, yield_path=tmp_path / "rl-yield.sqlite3")
    services.add_environment(ADDED, make_test_episode_env(), added_policy(rt))
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
    services.add_environment(ADDED, make_test_episode_env(), policy)
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
    services.add_environment(ADDED, make_test_episode_env(), add_to_the_run(rt))
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
    services.add_environment(ADDED, make_test_episode_env(), add_to_the_run(rt, active=True))
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
