"""Envs added to a running run, carried by the schedule."""
import json

import pytest

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_catalogue import (
    CataloguedContract,
    announced_contract,
    catalogued,
    distribution_of,
    validate_added,
)
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError
from reliquary.protocol.service_schedule import (
    ScheduleError,
    ServiceSchedule,
    initial_schedule,
    next_schedule,
)
from tests.unit.episode_v2_fixtures import (
    ADDED,
    ADDED_IDENTITY,
    ENV_PACKAGE,
    EPISODE,
    SANDBOX_ENV,
    added_block,
    episode_contract,
)
from tests.unit.service_v2_fixtures import MATH, contract_v2

CONTRACT = episode_contract()


def without_episode():
    block = added_block()
    block["environment"].pop("episode")
    return block


def with_added(*, active=None, shares=None, block=None):
    return next_schedule(CONTRACT, initial_schedule(CONTRACT), add={ADDED: block or added_block()},
                         active=active, shares=shares)


def test_an_added_env_enters_inactive_and_the_view_serves_its_policy_and_catalogue():
    schedule = with_added()
    assert ADDED in schedule.to_dict()["environments"] and ADDED not in schedule.active_environments()
    assert schedule.added_environments() == {ADDED: added_block()}
    view = schedule.contract_view(CONTRACT)
    assert isinstance(view, CataloguedContract)
    assert view.sha256 == CONTRACT.sha256 and view.canonical == CONTRACT.canonical
    assert view.to_dict() == CONTRACT.to_dict()
    assert set(view.environments) == set(CONTRACT.environments) | {ADDED}
    assert set(view.order_environments) == set(CONTRACT.environments)
    assert view.episode_policy(ADDED).env_package == ADDED_IDENTITY
    assert view.episode_environments == tuple(sorted((EPISODE, ADDED)))
    assert view.catalogue_of(ADDED) == added_block()["catalogue"]


def test_an_added_env_can_be_switched_on_in_the_same_request():
    schedule = with_added(active=[MATH, EPISODE, ADDED], shares={MATH: 4000, EPISODE: 3000, ADDED: 3000})
    assert ADDED in schedule.active_environments() and schedule.share_bps(ADDED) == 3000


def test_a_schedule_without_added_rows_keeps_its_bytes_and_its_contract():
    plain = initial_schedule(CONTRACT)
    assert b"added" not in plain.canonical
    assert ServiceSchedule.from_dict(plain.to_dict(), CONTRACT).canonical == plain.canonical
    assert type(plain.contract_view(CONTRACT)) is ServiceContract
    assert type(catalogued(CONTRACT, plain.to_dict())) is ServiceContract
    # A schedule change without an addition keeps the pre-2D row shape and bytes.
    changed = next_schedule(CONTRACT, plain, cooldowns={MATH: 3})
    expected = plain.to_dict()
    expected["revision"], expected["environments"][MATH]["cooldown_windows"] = 1, 3
    assert changed.canonical == canonical_json_bytes(expected) and b"added" not in changed.canonical
    assert type(changed.contract_view(CONTRACT)) is ServiceContract


def test_rows_accumulate_and_a_later_schedule_keeps_the_added_row():
    schedule = with_added(active=[MATH, EPISODE, ADDED], shares={MATH: 4000, EPISODE: 3000, ADDED: 3000})
    off = next_schedule(CONTRACT, schedule, active=[MATH, EPISODE], shares={MATH: 5000, EPISODE: 5000})
    assert ADDED in off.added_environments() and ADDED not in off.active_environments()
    with pytest.raises(ScheduleError, match="already"):
        next_schedule(CONTRACT, off, add={ADDED: added_block()})


def test_a_schedule_with_an_added_row_round_trips_and_never_changes_the_order_hash():
    schedule = with_added()
    again = ServiceSchedule.from_dict(json.loads(schedule.canonical), CONTRACT)
    assert again.canonical == schedule.canonical and again.order_sha256 == CONTRACT.sha256
    # The view validates the same schedule (its order envs are the order's, not the overlay).
    assert ServiceSchedule.from_dict(schedule.to_dict(), schedule.contract_view(CONTRACT)).canonical == \
        schedule.canonical
    # Removing the added row is refused: rows only accumulate through next_schedule, and a row without its
    # block is an unknown env.
    value = schedule.to_dict()
    value["environments"][ADDED].pop("added")
    with pytest.raises(ScheduleError, match="added to it"):
        ServiceSchedule.from_dict(value, CONTRACT)


@pytest.mark.parametrize("block, why", [
    (added_block(share_bps=100), "share"),
    (without_episode(), "episode"),
    (added_block(catalogue={"package": "reliquary-terminal"}), "name==version"),
    (added_block(catalogue={"package": "reliquary-other==1.0"}), "bridged identity"),
    (added_block(catalogue={"rl_split": "rl"}), "split"),
    (added_block(catalogue={"image_lock_sha256": "x"}), "image_lock_sha256"),
    (added_block(catalogue={"conformance_sha256": ""}), "conformance_sha256"),
    (added_block(missing_box="uncertain"), "graded"),
    (added_block(episode={"max_episode_tokens": 8192}), "max_episode_tokens"),       # above the order's bound
    (added_block(episode={"max_turns": 64}), "max_turns"),
    (added_block(episode={"sandbox_env": SANDBOX_ENV}), "sandbox env"),               # a new sandbox env name
    (added_block(episode={"env_package": ENV_PACKAGE},
                 catalogue={"package": ENV_PACKAGE.split("+")[0]}), "distribution"),  # a new distribution
    # The launch catalogue's own checks apply unchanged: split and tools consistency, secrets.
    (added_block(catalogue={"options": {"tools": ["bash"], "splits": {"other": {"split": "other"}}}}),
     "not a split the options serve"),
    (added_block(catalogue={"options": {"tools": ["bash", "edit"], "splits": {"tmax": {"split": "tmax"}}}}),
     "options.tools"),
    (added_block(catalogue={"options": {"tools": ["bash"], "splits": {"tmax": {"split": "tmax"}},
                                        "api_key": "x"}}), "credential"),
    (added_block(catalogue={"options": {"tools": ["bash"], "splits": {"tmax": {"split": "tmax"}},
                                        "endpoint": "https://user:hunter2@example.com/v1"}}), "credential"),
    (added_block(catalogue_x=1), "expected fields"),
])
def test_a_malformed_added_env_is_refused(block, why):
    with pytest.raises((ScheduleError, ServiceContractError), match=why):
        with_added(block=block)


def test_a_conformance_digest_is_required_of_an_added_env():
    block = added_block()
    del block["catalogue"]["conformance_sha256"]
    with pytest.raises((ScheduleError, ServiceContractError), match="conformance"):
        with_added(block=block)


def test_the_catalogue_of_an_added_env_is_the_blocks_not_the_entrys():
    block = added_block()
    block["environment"]["catalogue"] = dict(block["catalogue"])
    with pytest.raises(ScheduleError, match="the block's"):
        with_added(block=block)


def test_an_added_env_cannot_shadow_an_env_of_the_order():
    with pytest.raises(ServiceContractError, match="already an environment of the order"):
        validate_added(EPISODE, added_block(), CONTRACT.environments, CONTRACT.environments)
    with pytest.raises(ScheduleError, match="already in the schedule"):
        next_schedule(CONTRACT, initial_schedule(CONTRACT), add={EPISODE: added_block()})


def test_an_installed_env_name_cannot_be_added():
    # As for a launch catalogue env: a bridged env never takes the name of an env of this build.
    with pytest.raises(ScheduleError, match="installed in this build"):
        next_schedule(CONTRACT, initial_schedule(CONTRACT), add={"openmathinstruct": added_block()})


def test_two_added_envs_need_their_own_sandbox_env_and_distribution():
    first = with_added()
    clash = added_block()
    with pytest.raises(ScheduleError, match="sandbox env"):
        next_schedule(CONTRACT, first, add={"tmax_v3": clash})
    other_name = added_block(episode={"sandbox_env": "reliquary-terminal-v3"})
    with pytest.raises(ScheduleError, match="distribution"):
        next_schedule(CONTRACT, first, add={"tmax_v3": other_name})
    fresh = added_block(episode={"sandbox_env": "reliquary-terminal-v3",
                                 "env_package": "reliquary_terminal_v3==0.4.0+gabcdef0123456789.vfb2"},
                        catalogue={"package": "reliquary_terminal_v3==0.4.0"})
    both = next_schedule(CONTRACT, first, add={"tmax_v3": fresh})
    assert set(both.added_environments()) == {ADDED, "tmax_v3"}
    assert set(both.contract_view(CONTRACT).episode_environments) == {EPISODE, ADDED, "tmax_v3"}


def test_a_malformed_added_row_beside_a_valid_one_is_a_schedule_error():
    value = with_added().to_dict()
    value["environments"]["zzz_broken"] = {"active": 0, "share_bps": 0, "cooldown_windows": 1,
                                           "added": {"environment": {"episode": 1}, "catalogue": {}}}
    with pytest.raises(ScheduleError):
        ServiceSchedule.from_dict(value, CONTRACT)


def test_the_distribution_name_is_normalised():
    assert distribution_of("Reliquary_Terminal==0.3.0+g0.vfb2") == "reliquary-terminal"


def test_a_run_without_an_episode_env_cannot_add_one():
    single_turn = contract_v2()
    with pytest.raises((ScheduleError, ServiceContractError), match="no episode environment"):
        next_schedule(single_turn, initial_schedule(single_turn), add={ADDED: added_block()})


def test_the_announced_contract_carries_the_added_env():
    schedule = with_added(active=[MATH, EPISODE, ADDED], shares={MATH: 4000, EPISODE: 3000, ADDED: 3000})
    value = {"contract": CONTRACT.to_dict(), "schedule": schedule.to_dict()}
    assert announced_contract(value).episode_policy(ADDED).split == "tmax"
    assert announced_contract(value).catalogue_of(ADDED)["package"] == "reliquary-terminal==0.3.0"
    plain = {"contract": CONTRACT.to_dict(), "schedule": initial_schedule(CONTRACT).to_dict()}
    assert type(announced_contract(plain)) is ServiceContract
    assert type(announced_contract({"contract": CONTRACT.to_dict()})) is ServiceContract


def test_a_view_rebuilt_from_its_own_bytes_is_equal():
    view = with_added().contract_view(CONTRACT)
    again = CataloguedContract(view.canonical, view.added)
    block = added_block()
    assert again == view and json.loads(view.added) == {ADDED: {**block["environment"], "catalogue": block["catalogue"]}}
    # A view of a view is the same view (the overlay is rebuilt from the order's bytes).
    assert catalogued(view, with_added().to_dict()) == view


def test_a_view_refuses_bad_overlay_bytes():
    view = with_added().contract_view(CONTRACT)
    with pytest.raises(ServiceContractError, match="canonical"):
        CataloguedContract(view.canonical, json.dumps(json.loads(view.added), indent=1).encode())
    with pytest.raises(ServiceContractError, match="already an environment of the order"):
        CataloguedContract(view.canonical, canonical_json_bytes({EPISODE: CONTRACT.environments[EPISODE]}))
    with pytest.raises(ServiceContractError, match="added environments required"):
        CataloguedContract(view.canonical, b"{}")


def test_the_view_requires_the_added_envs_capabilities():
    view = with_added().contract_view(CONTRACT)
    supported = {"environment-reward/v1", "trainer-driven/v1", "exploration-first-scan/v1",
                 "in-zone-rotation/v1", "public-seed-pool/v3", "legacy/v1", "task-scoped/v1",
                 "signed-sandbox-episode/v1"}
    view.require_capabilities(supported)
    with pytest.raises(ServiceContractError, match="capabilities"):
        view.require_capabilities(supported - {"signed-sandbox-episode/v1"})
