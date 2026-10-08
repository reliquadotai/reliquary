# tests/unit/test_service_contract_v2.py
import pytest

from reliquary.protocol.release_contract import canonical_json_bytes

from reliquary.protocol.service_contract import (
    SUPPORTED_V2_CAPABILITIES, ServiceContract, ServiceContractError,
)
from reliquary.protocol.service_schedule import ScheduleError, ServiceSchedule, initial_schedule, next_schedule
from tests.unit.service_v2_fixtures import CODE, MATH, SCIENCE, contract_v2, contract_v2_dict


def test_v2_round_trips_canonically_and_lists_envs():
    contract = contract_v2()
    assert contract.version == 2
    assert set(contract.environments) == {MATH, CODE}
    assert contract.environment(MATH)["missing_box"] == "uncertain"
    assert contract.environment(CODE)["missing_box"] == "graded"
    assert ServiceContract.from_dict(contract.to_dict()) == contract
    contract.require_capabilities(set(SUPPORTED_V2_CAPABILITIES))


def test_v2_has_no_checkpoint_bound_context():
    with pytest.raises(ServiceContractError, match="run-wide"):
        contract_v2().context_sha256


@pytest.mark.parametrize("mutate, message", [
    (lambda v: v.update(service_kind="dataset_mapping"), "adaptive_training"),
    (lambda v: v["environments"][MATH].update(share_bps=1), "sum to 10000"),
    (lambda v: v["environments"][MATH].update(missing_box="always"), "missing_box"),
    (lambda v: v["environments"][MATH].update(exploration=2), "exploration"),
    (lambda v: v["policies"]["reward"].update(cap_bps=10001), "cap_bps"),
    (lambda v: v["policies"].update(eligibility={"kind": "dataset-epoch/v1"}), "policies"),
    (lambda v: v["environments"].update({"Bad Name": v["environments"][MATH]}), "environment id"),
    (lambda v: v["environments"][MATH]["sampling"].update(kind="public-draw-pool/v1"), "sampling"),
    (lambda v: v["environments"][MATH]["sampling"].update(renewal_windows=2), "renews every window"),
    (lambda v: v["environments"][MATH]["sampling"].update(renewal_windows=1000000), "renews every window"),
])
def test_v2_rejects_invalid_orders(mutate, message):
    value = contract_v2_dict()
    mutate(value)
    with pytest.raises(ServiceContractError, match=message):
        ServiceContract.from_dict(value)


def test_v1_adaptive_contract_still_parses_for_backward_reading():
    import json
    from pathlib import Path
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    assert ServiceContract.from_dict(value).version == 1


def test_initial_schedule_activates_positive_shares():
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 6000, CODE: 4000, SCIENCE: 0})
    schedule = initial_schedule(contract)
    assert schedule.revision == 0
    assert schedule.active_environments() == tuple(sorted((MATH, CODE)))
    assert schedule.cooldown_windows(MATH) == 50
    assert schedule.order_sha256 == contract.sha256


def test_next_schedule_requires_shares_summing_over_active_envs():
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 6000, CODE: 4000, SCIENCE: 0})
    current = initial_schedule(contract)
    changed = next_schedule(contract, current, active=(MATH, SCIENCE), shares={MATH: 7000, SCIENCE: 3000},
                            cooldowns={SCIENCE: 900})
    assert changed.revision == 1
    assert set(changed.active_environments()) == {MATH, SCIENCE}
    assert changed.share_bps(CODE) == 0 and changed.cooldown_windows(SCIENCE) == 900
    with pytest.raises(ScheduleError, match="sum to 10000"):
        next_schedule(contract, current, active=(MATH, SCIENCE), shares={MATH: 7000, SCIENCE: 2000})
    with pytest.raises(ScheduleError, match="unknown environment"):
        next_schedule(contract, current, cooldowns={"reliquary_other_v1": 3})


def test_cooldown_only_change_keeps_shares():
    contract = contract_v2()
    changed = next_schedule(contract, initial_schedule(contract), cooldowns={CODE: 7})
    assert changed.cooldown_windows(CODE) == 7
    assert changed.share_bps(MATH) == contract.environment(MATH)["share_bps"]


def test_schedule_from_dict_refuses_another_order():
    contract, other = contract_v2(), contract_v2(cooldown_windows=51)
    with pytest.raises(ScheduleError, match="order"):
        ServiceSchedule.from_dict(initial_schedule(contract).to_dict(), other)


def test_public_seed_pool_renews_every_window_and_nothing_else_is_refused_for_it():
    value = contract_v2_dict()
    assert value["environments"][MATH]["sampling"]["renewal_windows"] == 1
    assert ServiceContract.from_dict(value).environment(MATH)["sampling"]["renewal_windows"] == 1
    value["environments"][MATH]["sampling"]["renewal_windows"] = 0
    with pytest.raises(ServiceContractError, match="renewal_windows"):
        ServiceContract.from_dict(value)


def test_schedule_refuses_active_without_shares_and_inactive_with_a_share():
    contract = contract_v2(envs=(MATH, CODE, SCIENCE), shares={MATH: 6000, CODE: 4000, SCIENCE: 0})
    current = initial_schedule(contract)
    with pytest.raises(ScheduleError, match="explicit shares"):
        next_schedule(contract, current, active=(MATH, SCIENCE))
    # An inactive env (science) given a nonzero share, rebalancing the others to keep the sum.
    with pytest.raises(ScheduleError, match="inactive environment .* share 0"):
        next_schedule(contract, current, shares={MATH: 5000, CODE: 4000, SCIENCE: 1000})
    raw = current.to_dict()
    raw["environments"][SCIENCE]["active"] = 1  # active, share 0
    with pytest.raises(ScheduleError, match="needs a positive share"):
        ServiceSchedule.from_dict(raw, contract)


@pytest.mark.parametrize("path, bad", [
    (("visibility",), ["task"]),
    (("visibility",), {"a": 1}),
])
def test_v2_unhashable_enums_are_contract_errors_not_type_errors(path, bad):
    value = contract_v2_dict()
    value[path[0]] = bad
    with pytest.raises(ServiceContractError, match="visibility"):
        ServiceContract(canonical_json_bytes(value))


def test_v2_unhashable_sampling_kind_is_a_contract_error():
    value = contract_v2_dict()
    value["environments"][MATH]["sampling"]["kind"] = ["legacy/v1"]
    with pytest.raises(ServiceContractError, match="sampling"):
        ServiceContract(canonical_json_bytes(value))
