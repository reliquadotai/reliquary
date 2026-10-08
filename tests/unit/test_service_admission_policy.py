from types import SimpleNamespace

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.seed_pool import SeedPool, pool_from_service_policy
from reliquary.protocol.service_contract import SUPPORTED_V2_CAPABILITIES
from reliquary.protocol.service_schedule import initial_schedule, next_schedule
from reliquary.protocol.service_submission import ServiceBinding
from reliquary.services.admission_policy import validate_submission_policy
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2

CHECKPOINT = {"checkpoint_n": 3, "repo": "models/test", "revision": "a" * 40, "sha256": "e" * 64}


def announcement(contract, schedule=None):
    return {"contract": contract.to_dict(), "schedule": (schedule or initial_schedule(contract)).to_dict(),
            "checkpoint": CHECKPOINT, "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES),
            "pool_epoch": 5, "pool_randomness": "ab" * 32}


def request(contract, *, env=MATH, purpose="training", candidate=1, checkpoint="a" * 40, pool_env=None):
    pool = SeedPool.from_contract(contract, environment=pool_env or env, prompt_idx=7, checkpoint_hash=checkpoint,
                                  pool_epoch=5, randomness="ab" * 32)
    intent = ServiceBinding(contract.sha256, purpose)
    selection = pool.selection(candidate)
    rollouts = [SimpleNamespace(env_name=env, commit={"rollout": {
        "service_binding": intent.rollout_binding(i), "seed_pool": selection.rollout_binding(i)}})
        for i in range(M_ROLLOUTS)]
    return SimpleNamespace(service_binding=intent.to_dict(), pool_selection=selection.to_dict(),
                           rollouts=rollouts, checkpoint_hash=checkpoint, prompt_idx=7)


def test_any_active_env_of_the_order_is_accepted():
    contract = contract_v2()
    for env in (MATH, CODE):
        assert validate_submission_policy(request(contract, env=env), announcement(contract)) == contract


def test_inactive_env_is_refused():
    contract = contract_v2()
    schedule = next_schedule(contract, initial_schedule(contract), active=(MATH,), shares={MATH: 10000})
    with pytest.raises(ValueError, match="not active"):
        validate_submission_policy(request(contract, env=CODE), announcement(contract, schedule))


def test_pool_of_another_env_is_refused():
    contract = contract_v2()
    with pytest.raises(ValueError):
        validate_submission_policy(request(contract, env=MATH, pool_env=CODE), announcement(contract))


def test_binding_is_to_the_order_not_the_checkpoint():
    contract = contract_v2()
    ann = announcement(contract)
    ann["checkpoint"] = {**CHECKPOINT, "revision": "b" * 40}
    with pytest.raises(ValueError, match="checkpoint"):
        validate_submission_policy(request(contract), ann)
    assert validate_submission_policy(request(contract, checkpoint="b" * 40), ann) == contract


def test_exploration_refused_where_the_env_disables_it():
    contract = contract_v2(exploration=0)
    with pytest.raises(ValueError, match="exploration"):
        validate_submission_policy(request(contract, purpose="exploration"), announcement(contract))


def test_pool_resolution_needs_the_env():
    contract = contract_v2()
    pool = pool_from_service_policy(announcement(contract), environment=CODE, prompt_idx=7, checkpoint_hash="a" * 40)
    assert pool.environment == CODE and pool.pool_groups == 2


def test_v1_announcement_is_refused():
    import json
    from pathlib import Path
    from reliquary.protocol.service_contract import ServiceContract
    v1 = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    contract = ServiceContract.from_dict(v1)
    with pytest.raises(ValueError):
        validate_submission_policy(SimpleNamespace(service_binding=None, pool_selection=None, rollouts=[]),
                                   {"contract": contract.to_dict(), "schedule": {}, "checkpoint": CHECKPOINT,
                                    "supported_capabilities": ["legacy/v1"], "pool_epoch": 0, "pool_randomness": ""})
