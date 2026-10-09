"""Episode environments in service-contract/v2 (plan 2C, Task 1)."""
import pytest

from reliquary.protocol.service_contract import (
    EPISODE_BUDGET_CEILINGS, EPISODE_BUDGET_FIELDS, EPISODE_CAPABILITY, MAX_EPISODE_TOKENS, SUPPORTED_V2_CAPABILITIES, ServiceContract, ServiceContractError,
    supported_v2_capabilities,
)
from reliquary.protocol.submission import ServicePolicyAnnouncement
from reliquary.services.admission_policy import parse_service_announcement
from tests.unit.episode_v2_fixtures import (
    BUDGETS, ENV_PACKAGE, EPISODE, SANDBOX_ENV, SPLIT, WINDOW_BEACON, episode_block, episode_contract,
    episode_contract_dict, episode_runtime,
)
from tests.unit.service_v2_fixtures import MATH, contract_v2


def test_an_episode_environment_parses_and_exposes_its_policy():
    contract = episode_contract()
    policy = contract.episode_policy(EPISODE)
    assert (policy.environment, policy.sandbox_env, policy.split) == (EPISODE, SANDBOX_ENV, SPLIT)
    assert policy.env_package == ENV_PACKAGE and policy.tools == ("bash", "edit")
    assert (policy.max_turns, policy.max_tokens_per_turn, policy.max_episode_tokens) == (8, 512, 4096)
    assert policy.budgets_dict() == BUDGETS
    assert policy.pool_seeds == 2 * contract.environment(EPISODE)["sampling"]["group_size"]
    assert contract.episode_policy(MATH) is None
    assert contract.episode_environments == (EPISODE,)


def test_a_single_turn_order_has_no_episode_and_announces_exactly_the_v2_capabilities():
    contract = contract_v2()
    assert contract.episode_environments == ()
    assert supported_v2_capabilities(contract) == SUPPORTED_V2_CAPABILITIES
    assert b"episode" not in contract.canonical


def test_an_episode_order_requires_and_announces_the_episode_capability(tmp_path):
    contract = episode_contract()
    with pytest.raises(ServiceContractError, match="capabilities"):
        contract.require_capabilities(set(SUPPORTED_V2_CAPABILITIES))
    assert supported_v2_capabilities(contract) == SUPPORTED_V2_CAPABILITIES | {EPISODE_CAPABILITY}
    rt = episode_runtime(tmp_path)
    announcement = rt.announcement(window=1, randomness=WINDOW_BEACON)
    assert announcement["supported_capabilities"] == sorted(SUPPORTED_V2_CAPABILITIES | {EPISODE_CAPABILITY})
    ServicePolicyAnnouncement(**announcement)
    parsed, schedule = parse_service_announcement(announcement)
    assert parsed.sha256 == contract.sha256 and EPISODE in schedule.active_environments()


@pytest.mark.parametrize("field, value", [
    ("kind", "signed-sandbox-episode/v2"), ("sandbox_env", ""), ("sandbox_env", "a b"),
    ("split", "x" * 80), ("env_package", "reliquary-swe"), ("tools", []), ("tools", ["edit", "bash"]),
    ("tools", ["bash", "bash"]), ("tools", ["python"]), ("max_turns", 0), ("max_turns", 257),
    ("max_tokens_per_turn", 0), ("max_episode_tokens", 1), ("max_tokens_per_turn", 4096),
    ("budgets", {**BUDGETS, "max_calls": 0}),
    ("budgets", {k: v for k, v in BUDGETS.items() if k != "pids"}),
])
def test_a_malformed_episode_block_is_refused(field, value):
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(episode_contract_dict(episode=episode_block(**{field: value})))


def test_an_episode_block_with_an_unknown_field_is_refused():
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(episode_contract_dict(episode={**episode_block(), "judge": "llm"}))


def test_an_episode_environment_must_draw_from_the_public_seed_pool():
    value = episode_contract_dict()
    value["environments"][EPISODE]["sampling"] = {"kind": "legacy/v1"}
    with pytest.raises(ServiceContractError, match="public seed pool"):
        ServiceContract.from_dict(value)


def test_an_episode_reward_is_graded_never_uncertain():
    value = episode_contract_dict()
    value["environments"][EPISODE]["missing_box"] = "uncertain"
    with pytest.raises(ServiceContractError, match="graded"):
        ServiceContract.from_dict(value)


@pytest.mark.parametrize("field", EPISODE_BUDGET_FIELDS)
def test_each_session_budget_is_bounded_by_its_ceiling(field):
    ceiling = EPISODE_BUDGET_CEILINGS[field]
    ServiceContract.from_dict(episode_contract_dict(episode=episode_block(budgets={**BUDGETS, field: ceiling})))
    with pytest.raises(ServiceContractError, match=field):
        ServiceContract.from_dict(
            episode_contract_dict(episode=episode_block(budgets={**BUDGETS, field: ceiling + 1})))


def test_the_budget_ceilings_cover_every_budget_field_and_fit_the_fixture():
    assert set(EPISODE_BUDGET_CEILINGS) == set(EPISODE_BUDGET_FIELDS)
    assert all(BUDGETS[name] <= EPISODE_BUDGET_CEILINGS[name] for name in BUDGETS)


def test_an_episode_is_bounded_by_the_corpus_trajectory_cap():
    assert MAX_EPISODE_TOKENS == 60000
    ServiceContract.from_dict(episode_contract_dict(episode=episode_block(max_episode_tokens=60000)))
    with pytest.raises(ServiceContractError, match="max_episode_tokens"):
        ServiceContract.from_dict(episode_contract_dict(episode=episode_block(max_episode_tokens=60001)))
