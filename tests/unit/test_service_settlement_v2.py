import math

import pytest

from reliquary.protocol.service_schedule import initial_schedule
from reliquary.services.settlement import SettlementError, settle_window, validate_service_archive_v2
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2

PICKS, SLOTS = 2, 4


def envelope(contract, pools):
    schedule = initial_schedule(contract)
    return {"order_sha256": contract.sha256, "schedule": schedule.to_dict(), "schedule_sha256": schedule.sha256,
            "pools": pools, "picks_target": PICKS, "batch_slots": SLOTS, "price_bps": 1500, "cap_bps": 1000,
            "checkpoint": {"checkpoint_n": 1, "repo": "models/test", "revision": "d" * 40, "sha256": "b" * 64}}


def archive(rows):
    return {"window_start": 3, "window_status": "complete",
            "batch": [{"hotkey": h, "env_name": e} for h, e in rows], "rewards_by_hotkey": {}}


def test_unfilled_window_pays_exploration_without_touching_training():
    contract = contract_v2()
    pools = {MATH: 0.4, CODE: 0.4}
    g = 0.4 / (PICKS * SLOTS)
    result = settle_window(archive=archive([("a", MATH)] * 3), envelope=envelope(contract, pools),
                           exploration={MATH: {"x": 0.15 * g}}, aborted=False)
    assert result["service_scale_by_environment"][MATH] == 1.0
    assert "service_training_scale_by_environment" not in result
    assert result["rewards_by_hotkey"] == {"a": pytest.approx(3 * g), "x": pytest.approx(0.15 * g)}
    validate_service_archive_v2(result, contract, cap=1.0)


def test_full_window_scales_training_and_exploration_by_the_same_factor():
    contract = contract_v2()
    pools = {MATH: 0.4, CODE: 0.4}
    g = 0.4 / (PICKS * SLOTS)
    x = 2 * 0.15 * g
    result = settle_window(archive=archive([("a", MATH)] * (PICKS * SLOTS)), envelope=envelope(contract, pools),
                           exploration={MATH: {"x": x}}, aborted=False)
    scale = result["service_scale_by_environment"][MATH]
    assert scale == pytest.approx(0.4 / (0.4 + x))
    assert scale < 1.0
    assert result["service_training_by_environment"][MATH]["a"] == pytest.approx(0.4 * scale)
    # nominal exploration is archived unscaled; replay applies the scale
    assert result["service_exploration_by_environment"][MATH] == {"x": x}
    assert result["rewards_by_hotkey"]["x"] == pytest.approx(x * scale)
    total = result["rewards_by_hotkey"]["a"] + result["rewards_by_hotkey"]["x"]
    assert math.isclose(total, 0.4, rel_tol=0, abs_tol=1e-12)
    validate_service_archive_v2(result, contract, cap=1.0)


def test_scale_never_below_one_over_one_point_one_at_the_exploration_cap():
    contract = contract_v2()
    pools = {MATH: 0.4, CODE: 0.4}
    g = 0.4 / (PICKS * SLOTS)
    x = 5 * 0.15 * g  # the most whole entitlements under the cap (0.04)
    assert x <= 0.4 * 1000 / 10000 < 6 * 0.15 * g
    result = settle_window(archive=archive([("a", MATH)] * (PICKS * SLOTS)), envelope=envelope(contract, pools),
                           exploration={MATH: {"x": x}}, aborted=False)
    assert result["service_scale_by_environment"][MATH] >= 1 / 1.1 - 1e-12
    validate_service_archive_v2(result, contract, cap=1.0)


def test_aborted_window_pays_no_exploration():
    contract = contract_v2()
    result = settle_window(archive=archive([]), envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}),
                           exploration={MATH: {"x": 0.001}}, aborted=True)
    assert result["service_exploration_by_environment"] == {}
    assert result["rewards_by_hotkey"] == {}
    assert result["window_status"] == "aborted"
    validate_service_archive_v2(result, contract, cap=1.0)
    result["service_exploration_by_environment"] = {MATH: {"x": 0.15 * 0.05}}
    with pytest.raises(ValueError, match="abort"):
        validate_service_archive_v2(result, contract, cap=1.0)


def test_exploration_over_cap_is_refused():
    contract = contract_v2()
    with pytest.raises(SettlementError, match="cap"):
        settle_window(archive=archive([]), envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}),
                      exploration={MATH: {"x": 0.05}}, aborted=False)


def test_validator_refuses_exploration_over_cap():
    contract = contract_v2()
    g = 0.4 / (PICKS * SLOTS)
    record = settle_window(archive=archive([]), envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}),
                           exploration={MATH: {"x": 0.15 * g}}, aborted=False)
    over = 0.15 * g * 8  # 0.06 > cap 0.04, whole entitlements
    record["service_exploration_by_environment"][MATH]["x"] = over
    record["rewards_by_hotkey"]["x"] = over
    with pytest.raises(ValueError, match="cap"):
        validate_service_archive_v2(record, contract, cap=1.0)


def test_validator_accepts_its_own_settlement_and_rejects_tampering():
    contract = contract_v2()
    g = 0.25 / (PICKS * SLOTS)
    record = settle_window(archive=archive([("a", MATH), ("b", CODE)]), envelope=envelope(contract, {MATH: 0.25, CODE: 0.25}),
                           exploration={CODE: {"x": 0.15 * g}}, aborted=False)
    validate_service_archive_v2(record, contract, cap=0.5)
    tampered = {**record, "rewards_by_hotkey": {**record["rewards_by_hotkey"], "a": record["rewards_by_hotkey"]["a"] * 2}}
    with pytest.raises(ValueError):
        validate_service_archive_v2(tampered, contract, cap=0.5)
    with pytest.raises(ValueError, match="pool"):
        validate_service_archive_v2(record, contract, cap=0.4)


def test_forged_self_consistent_training_map_is_refused():
    contract = contract_v2()
    record = settle_window(archive=archive([("a", MATH), ("b", MATH)]),
                           envelope=envelope(contract, {MATH: 0.25, CODE: 0.25}), exploration={}, aborted=False)
    forged = dict(record)
    amounts = record["service_training_by_environment"][MATH]
    forged["service_training_by_environment"] = {**record["service_training_by_environment"],
                                                 MATH: {"a": amounts["a"] * 2, "b": 0.0}}
    forged["rewards_by_hotkey"] = {"a": amounts["a"] * 2, "b": 0.0}  # consistent with its own training map
    with pytest.raises(ValueError, match="batch"):
        validate_service_archive_v2(forged, contract, cap=0.5)


def test_forged_scale_is_refused():
    contract = contract_v2()
    g = 0.4 / (PICKS * SLOTS)
    record = settle_window(archive=archive([("a", MATH)] * (PICKS * SLOTS)),
                           envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}),
                           exploration={MATH: {"x": 2 * 0.15 * g}}, aborted=False)
    record["service_scale_by_environment"][MATH] = 1.0
    with pytest.raises(ValueError, match="scale"):
        validate_service_archive_v2(record, contract, cap=1.0)


def test_price_scaled_pool_validates():
    contract = contract_v2()
    pools = {MATH: 0.25 * 0.6, CODE: 0.25}
    g = pools[MATH] / (PICKS * SLOTS)
    record = settle_window(archive=archive([("a", MATH)]), envelope=envelope(contract, pools),
                           exploration={MATH: {"x": 0.15 * g}}, aborted=False)
    validate_service_archive_v2(record, contract, cap=0.5)


def test_exploration_amount_must_be_whole_entitlements():
    contract = contract_v2()
    g = 0.25 / (PICKS * SLOTS)
    record = settle_window(archive=archive([]), envelope=envelope(contract, {MATH: 0.25, CODE: 0.25}),
                           exploration={MATH: {"x": 0.15 * g}}, aborted=False)
    record["service_exploration_by_environment"][MATH]["x"] = 0.1 * g
    record["rewards_by_hotkey"]["x"] = 0.1 * g
    with pytest.raises(ValueError, match="entitlement"):
        validate_service_archive_v2(record, contract, cap=0.5)
