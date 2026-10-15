import copy
import json
import math

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.service_schedule import initial_schedule
from reliquary.services.exploration import exploration_cap, exploration_price
from reliquary.services.settlement import SettlementError, settle_window, validate_service_archive_v2
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, contract_v2_dict

PICKS, SLOTS = 2, 4
T = PICKS * SLOTS


def envelope(contract, pools, *, picks=PICKS, slots=SLOTS):
    schedule = initial_schedule(contract)
    return {"order_sha256": contract.sha256, "schedule": schedule.to_dict(), "schedule_sha256": schedule.sha256,
            "pools": pools, "picks_target": picks, "batch_slots": slots,
            "checkpoint": {"checkpoint_n": 1, "repo": "models/test", "revision": "d" * 40, "sha256": "b" * 64}}


def archive(rows):
    return {"window_start": 3, "window_status": "complete",
            "batch": [{"hotkey": h, "env_name": e, "prompt_idx": i} for i, (h, e) in enumerate(rows)],
            "rewards_by_hotkey": {}}


def settle(rows, pools, exploration, *, contract=None, aborted=False, picks=PICKS, slots=SLOTS):
    contract = contract or contract_v2()
    return settle_window(archive=archive(rows), envelope=envelope(contract, pools, picks=picks, slots=slots),
                         contract=contract, exploration=exploration, aborted=aborted)


def validate(record, contract=None, *, cap=1.0, picks=PICKS, slots=SLOTS):
    validate_service_archive_v2(record, contract or contract_v2(), cap=cap, picks_target=picks, batch_slots=slots)


def test_unfilled_window_pays_exploration_without_touching_training():
    g = 0.4 / T
    result = settle([("a", MATH)] * 3, {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 1}})
    assert result["service_scale_by_environment"] == {CODE: 1.0, MATH: 1.0}
    assert "service_training_scale_by_environment" not in result
    assert result["service_exploration_by_environment"] == {MATH: {"x": 1}}
    assert type(result["service_exploration_by_environment"][MATH]["x"]) is int
    assert result["rewards_by_hotkey"] == {"a": pytest.approx(3 * g), "x": pytest.approx(0.15 * g)}
    validate(result)


def test_full_window_scales_training_and_exploration_by_the_same_factor():
    g = 0.4 / T
    x = 2 * 0.15 * g
    result = settle([("a", MATH)] * T, {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 2}})
    scale = result["service_scale_by_environment"][MATH]
    assert scale == pytest.approx(0.4 / (0.4 + x)) and scale < 1.0
    assert result["service_training_by_environment"][MATH]["a"] == pytest.approx(0.4 * scale)
    # whole entitlement counts are archived; replay prices and scales them
    assert result["service_exploration_by_environment"][MATH] == {"x": 2}
    assert result["rewards_by_hotkey"]["x"] == pytest.approx(x * scale)
    total = result["rewards_by_hotkey"]["a"] + result["rewards_by_hotkey"]["x"]
    assert math.isclose(total, 0.4, rel_tol=0, abs_tol=1e-12)
    validate(result)


def test_scale_is_exactly_one_over_one_point_one_at_the_exploration_cap():
    # T = 48 slots: 32 entitlements x 0.15 / 48 = exactly 10 % of the pool, the cap.
    picks, slots = 6, 8
    pools = {MATH: 0.4, CODE: 0.4}
    result = settle([("a", MATH)] * 48, pools, {MATH: {"x": 20, "y": 12}}, picks=picks, slots=slots)
    scale = result["service_scale_by_environment"][MATH]
    assert scale == pytest.approx(1 / 1.1, rel=1e-12)
    rewards = result["rewards_by_hotkey"]
    assert rewards["a"] == pytest.approx(0.4 / 1.1, rel=1e-12)            # training: -9.09 %
    assert rewards["x"] == pytest.approx(20 * 0.15 * 0.4 / 48 / 1.1, rel=1e-12)
    assert rewards["x"] + rewards["y"] == pytest.approx(0.04 / 1.1, rel=1e-12)
    assert math.fsum(rewards.values()) == pytest.approx(0.4, abs=1e-12)
    validate(result, picks=picks, slots=slots)
    with pytest.raises(SettlementError, match="cap"):                      # one more entitlement is over the cap
        settle([("a", MATH)] * 48, pools, {MATH: {"x": 20, "y": 13}}, picks=picks, slots=slots)


def test_aborted_window_pays_no_exploration():
    result = settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 1}}, aborted=True)
    assert result["service_exploration_by_environment"] == {}
    assert result["rewards_by_hotkey"] == {}
    assert result["window_status"] == "aborted"
    validate(result)
    result["service_exploration_by_environment"] = {MATH: {"x": 1}}
    result["rewards_by_hotkey"] = {"x": 0.15 * 0.4 / T}
    with pytest.raises(SettlementError, match="abort"):
        validate(result)


def test_exploration_over_cap_is_refused():
    # T = 8: 5 entitlements = 9.375 % fit, 6 = 11.25 % do not
    settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 3, "y": 2}})
    with pytest.raises(SettlementError, match="cap"):
        settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 3, "y": 3}})


def test_validator_refuses_exploration_over_cap():
    g = 0.4 / T
    record = settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 5}})
    validate(record)
    record["service_exploration_by_environment"][MATH]["x"] = 6
    record["rewards_by_hotkey"]["x"] = 6 * 0.15 * g
    with pytest.raises(SettlementError, match="cap"):
        validate(record)


def test_validator_accepts_its_own_settlement_and_rejects_tampering():
    contract = contract_v2()
    record = settle([("a", MATH), ("b", CODE)], {MATH: 0.25, CODE: 0.25}, {CODE: {"x": 1}}, contract=contract)
    validate(record, contract, cap=0.5)
    tampered = {**record, "rewards_by_hotkey": {**record["rewards_by_hotkey"], "a": record["rewards_by_hotkey"]["a"] * 2}}
    with pytest.raises(SettlementError, match="rewards_by_hotkey"):
        validate(tampered, contract, cap=0.5)
    more = copy.deepcopy(record)
    more["service_exploration_by_environment"][CODE]["x"] = 2      # one more entitlement than was paid
    with pytest.raises(SettlementError, match="rewards_by_hotkey"):
        validate(more, contract, cap=0.5)
    with pytest.raises(SettlementError, match="pool"):
        validate(record, contract, cap=0.4)
    with pytest.raises(SettlementError, match="another order"):
        validate(record, contract_v2(cooldown_windows=51), cap=0.5)


def test_forged_self_consistent_training_map_is_refused():
    record = settle([("a", MATH), ("b", MATH)], {MATH: 0.25, CODE: 0.25}, {})
    forged = dict(record)
    amounts = record["service_training_by_environment"][MATH]
    forged["service_training_by_environment"] = {**record["service_training_by_environment"],
                                                 MATH: {"a": amounts["a"] * 2, "b": 0.0}}
    forged["rewards_by_hotkey"] = {"a": amounts["a"] * 2, "b": 0.0}  # consistent with its own training map
    with pytest.raises(SettlementError, match="batch"):
        validate(forged, cap=0.5)


def test_forged_scale_is_refused():
    record = settle([("a", MATH)] * T, {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 2}})
    record["service_scale_by_environment"][MATH] = 1.0
    with pytest.raises(SettlementError, match="scale"):
        validate(record)


def test_forged_slot_geometry_is_refused():
    """picks_target=1, batch_slots=10 in the archive would price a group at a tenth of the pool:
    ten rows would take it all. The geometry comes from the caller, never from the archive."""
    contract = contract_v2()
    pools = {MATH: 0.4, CODE: 0.4}
    forged = settle([("thief", MATH)] * 10, pools, {}, contract=contract, picks=1, slots=10)
    assert forged["rewards_by_hotkey"]["thief"] == pytest.approx(0.4)       # self-consistent under its own geometry
    validate(forged, contract, picks=1, slots=10)
    with pytest.raises(SettlementError, match="geometry"):
        validate(forged, contract, picks=7, slots=16)
    honest = settle([("a", MATH)] * 10, pools, {}, contract=contract, picks=7, slots=16)
    assert honest["rewards_by_hotkey"]["a"] == pytest.approx(10 * 0.4 / 112)
    validate(honest, contract, picks=7, slots=16)
    for field, value in (("service_picks_target", 1), ("service_batch_slots", 1), ("service_picks_target", 7.0),
                         ("service_batch_slots", True), ("service_picks_target", None)):
        with pytest.raises(SettlementError, match="geometry"):
            validate({**honest, field: value}, contract, picks=7, slots=16)
    with pytest.raises(TypeError):                                           # the geometry is not optional
        validate_service_archive_v2(honest, contract, cap=1.0)
    for picks, slots in ((0, 16), (7, 0), (7.0, 16), (True, 16), (None, 16)):
        with pytest.raises(SettlementError, match="geometry"):
            validate(honest, contract, picks=picks, slots=slots)


def test_price_scaled_pool_validates_and_pays_from_the_scaled_pool():
    pools = {MATH: 0.25 * 0.6, CODE: 0.25}
    g = pools[MATH] / T
    record = settle([("a", MATH), ("a", CODE)], pools, {MATH: {"x": 2}})
    assert record["service_pools_by_environment"] == {CODE: 0.25, MATH: 0.15}
    assert record["service_training_by_environment"] == {CODE: {"a": pytest.approx(0.25 / T)}, MATH: {"a": pytest.approx(g)}}
    assert record["rewards_by_hotkey"] == {"a": pytest.approx(g + 0.25 / T), "x": pytest.approx(2 * 0.15 * g)}
    assert record["service_scale_by_environment"] == {CODE: 1.0, MATH: 1.0}
    validate(record, cap=0.5)
    with pytest.raises(SettlementError, match="share"):          # 0.25 > 0.4 x 50 %
        validate(record, cap=0.4)


def test_zero_pool_settles_validates_and_has_no_exploration():
    record = settle([("a", MATH), ("b", CODE)], {MATH: 0.0, CODE: 0.4}, {})
    assert record["rewards_by_hotkey"] == {"a": 0.0, "b": pytest.approx(0.4 / T)}
    assert record["service_scale_by_environment"][MATH] == 1.0
    validate(record)
    with pytest.raises(SettlementError, match="zero pool"):
        settle([], {MATH: 0.0, CODE: 0.4}, {MATH: {"x": 1}})
    record["service_exploration_by_environment"] = {MATH: {"x": 1}}
    record["rewards_by_hotkey"]["x"] = 0.0
    with pytest.raises(SettlementError, match="zero pool"):
        validate(record)
    record["service_exploration_by_environment"] = {MATH: {}}    # an empty map is no exploration
    del record["rewards_by_hotkey"]["x"]
    validate(record)
    both = settle([], {MATH: 0.0, CODE: 0.0}, {})
    assert both["rewards_by_hotkey"] == {}
    validate(both)


def test_more_training_rows_than_slots_never_pays_more_than_the_pool():
    record = settle([("a", MATH)] * (T + 2) + [("b", MATH)] * 2, {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 1}})
    g = 0.4 / T
    scale = record["service_scale_by_environment"][MATH]
    assert scale == pytest.approx(0.4 / ((T + 4) * g + 0.15 * g))
    assert math.fsum(record["rewards_by_hotkey"].values()) == pytest.approx(0.4, abs=1e-12)
    assert record["rewards_by_hotkey"]["a"] == pytest.approx((T + 2) * g * scale)
    validate(record)


def test_a_row_outside_the_envelope_is_refused_by_settle_and_validate():
    with pytest.raises(SettlementError, match="outside"):
        settle([("a", "reliquary_other_v1")], {MATH: 0.4, CODE: 0.4}, {})
    with pytest.raises(SettlementError, match="outside"):
        settle([("a", MATH)], {MATH: 0.4, CODE: 0.4}, {"reliquary_other_v1": {"x": 1}})
    record = settle([("a", MATH)], {MATH: 0.4, CODE: 0.4}, {})
    record["batch"].append({"hotkey": "a", "env_name": "reliquary_other_v1"})
    with pytest.raises(SettlementError, match="outside"):
        validate(record)
    record = settle([("a", MATH)], {MATH: 0.4, CODE: 0.4}, {})
    record["service_exploration_by_environment"] = {"reliquary_other_v1": {"x": 1}}
    with pytest.raises(SettlementError, match="outside"):
        validate(record)


def test_archive_validates_after_a_json_round_trip_and_replays_to_the_same_bits():
    pools = {MATH: 0.4 / 3, CODE: 0.7 / 3}                       # values with no short decimal form
    record = settle([("a", MATH)] * 7 + [("b", MATH)] + [("b", CODE)] * 3, pools, {MATH: {"x": 3, "y": 2}, CODE: {"x": 1}})
    assert record["service_scale_by_environment"][MATH] < 1.0
    again = json.loads(json.dumps(record))
    assert again == record                                       # counts stay integers, floats keep their bits
    assert type(again["service_exploration_by_environment"][MATH]["x"]) is int
    validate(again)
    # replay: the archive fields alone give back the same map, bit for bit
    contract = contract_v2()
    replayed = settle_window(
        archive={"batch": again["batch"]},
        envelope={"order_sha256": again["service_order_sha256"], "schedule": again["service_schedule"],
                  "schedule_sha256": again["service_schedule_sha256"], "pools": again["service_pools_by_environment"],
                  "picks_target": again["service_picks_target"], "batch_slots": again["service_batch_slots"]},
        contract=contract, exploration=again["service_exploration_by_environment"], aborted=False)
    assert replayed["rewards_by_hotkey"] == record["rewards_by_hotkey"]
    assert replayed["service_scale_by_environment"] == record["service_scale_by_environment"]
    assert list(record["rewards_by_hotkey"]) == ["b", "x", "a", "y"]     # env by env (code first), training then exploration


def test_price_and_cap_come_from_the_contract_in_settle_and_validate():
    """No envelope copy: an envelope that carries other bps changes nothing, another contract does."""
    pools = {MATH: 0.4, CODE: 0.4}
    contract = contract_v2()
    env = {**envelope(contract, pools), "price_bps": 9000, "cap_bps": 10000}
    record = settle_window(archive=archive([]), envelope=env, contract=contract, exploration={MATH: {"x": 1}}, aborted=False)
    assert record["rewards_by_hotkey"]["x"] == pytest.approx(0.15 * 0.4 / T)
    with pytest.raises(SettlementError, match="cap"):
        settle_window(archive=archive([]), envelope=env, contract=contract, exploration={MATH: {"x": 6}}, aborted=False)
    richer = contract_v2_dict()
    richer["policies"]["reward"].update(price_bps=3000, cap_bps=5000)
    richer = ServiceContract.from_dict(richer)
    paid = settle([], pools, {MATH: {"x": 13}}, contract=richer)             # 13 x 0.30 / 8 = 48.75 % <= 50 %
    assert paid["rewards_by_hotkey"]["x"] == pytest.approx(13 * 0.30 * 0.4 / T)
    validate(paid, richer)
    with pytest.raises(SettlementError, match="cap"):
        settle([], pools, {MATH: {"x": 14}}, contract=richer)
    with pytest.raises(SettlementError, match="another order"):
        settle_window(archive=archive([]), envelope=envelope(contract, pools), contract=richer, exploration={}, aborted=False)


def test_ledger_cap_and_settlement_cap_are_the_same_test(tmp_path):
    """What the ledger admits, settlement accepts: fill a ledger to its cap and settle its counts."""
    import sqlite3

    from reliquary.services.exploration import ExplorationLedger
    for picks, slots, pool in ((6, 8, 0.4), (7, 16, 0.32), (2, 4, 1e-7), (3, 5, 0.123456789)):
        book = ExplorationLedger(sqlite3.connect(":memory:"), order_sha256="a" * 64)
        price = exploration_price(pool, picks_target=picks, batch_slots=slots, price_bps=1500)
        cap = exploration_cap(pool, cap_bps=1000)
        i = 0
        with book.db:
            while book.reserve(window=1, environment=MATH, observation_id=f"{i:064x}", hotkey=f"h{i % 3}", prompt_idx=i,
                               amount=price, cap=cap, draw_round=5, new_hotkey_audit_groups=0) is not None:
                i += 1
            book.resolve_draws(1, environment=MATH, beacon_for_round=lambda r: "ab" * 32, audit_bps=0, run_salt=b"s" * 32)
            book.finalize_window(1, environment=MATH)
        assert i == 1000 * picks * slots // 1500                 # 10 % of the pool in 15 % units
        counts = book.payable(1, environment=MATH)
        assert sum(counts.values()) == i
        record = settle([("a", MATH)] * (picks * slots), {MATH: pool, CODE: pool}, {MATH: counts}, picks=picks, slots=slots)
        validate(record, picks=picks, slots=slots)
        assert math.fsum(v for v in record["service_training_by_environment"][MATH].values()) + math.fsum(
            n * price * record["service_scale_by_environment"][MATH] for n in counts.values()) == pytest.approx(pool, abs=1e-12)


def good_record():
    return settle([("a", MATH)] * 2 + [("b", CODE)], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": 2}})


def put(*path_and_value):
    *path, value = path_and_value

    def change(record):
        target = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


MALFORMED = {
    # the six named triggers
    "string in the scale map": put("service_scale_by_environment", MATH, "1.0"),
    "None in a training map": put("service_training_by_environment", MATH, "a", None),
    "string in rewards": put("rewards_by_hotkey", "a", "0.1"),
    "list as an env's exploration map": put("service_exploration_by_environment", MATH, ["x", 2]),
    "non-dict batch row": put("batch", 0, "a"),
    "non-list batch": put("batch", {"hotkey": "a", "env_name": MATH}),
    # counts must be whole positive integers
    "NaN count": put("service_exploration_by_environment", MATH, "x", float("nan")),
    "inf count": put("service_exploration_by_environment", MATH, "x", float("inf")),
    "negative count": put("service_exploration_by_environment", MATH, "x", -1),
    "bool count": put("service_exploration_by_environment", MATH, "x", True),
    "zero count": put("service_exploration_by_environment", MATH, "x", 0),
    "float count": put("service_exploration_by_environment", MATH, "x", 2.0),
    "fractional count": put("service_exploration_by_environment", MATH, "x", 1.5),
    "string count": put("service_exploration_by_environment", MATH, "x", "2"),
    "None count": put("service_exploration_by_environment", MATH, "x", None),
    "huge count": put("service_exploration_by_environment", MATH, "x", 10**400),
    "float amount instead of a count": put("service_exploration_by_environment", MATH, "x", 2 * 0.15 * 0.05),
    # every other map and container
    "exploration is a list": put("service_exploration_by_environment", [[MATH, {"x": 2}]]),
    "exploration is missing": put("service_exploration_by_environment", None),
    "an env's exploration is None": put("service_exploration_by_environment", MATH, None),
    "an env's exploration is a string": put("service_exploration_by_environment", MATH, "x"),
    "training is a list": put("service_training_by_environment", []),
    "an env's training is a list": put("service_training_by_environment", MATH, [0.1]),
    "an env's training is None": put("service_training_by_environment", MATH, None),
    "NaN in a training map": put("service_training_by_environment", MATH, "a", float("nan")),
    "bool in a training map": put("service_training_by_environment", MATH, "a", True),
    "scale is a list": put("service_scale_by_environment", [1.0, 1.0]),
    "None in the scale map": put("service_scale_by_environment", MATH, None),
    "NaN in the scale map": put("service_scale_by_environment", MATH, float("nan")),
    "rewards is a list": put("rewards_by_hotkey", [0.1]),
    "rewards is missing": put("rewards_by_hotkey", None),
    "None in rewards": put("rewards_by_hotkey", "a", None),
    "inf in rewards": put("rewards_by_hotkey", "a", float("inf")),
    "negative in rewards": put("rewards_by_hotkey", "a", -0.1),
    "list in rewards": put("rewards_by_hotkey", "a", [0.1]),
    "batch is a string": put("batch", "ab"),
    "batch is a number": put("batch", 3),
    "batch row is None": put("batch", 0, None),
    "batch row is a list": put("batch", 0, ["a", MATH]),
    "batch row env is a list": put("batch", 0, {"hotkey": "a", "env_name": [MATH]}),
    "batch row env is a dict": put("batch", 0, {"hotkey": "a", "env_name": {}}),
    "batch row hotkey is a number": put("batch", 0, {"hotkey": 5, "env_name": MATH}),
    "batch row has no hotkey": put("batch", 0, {"env_name": MATH}),
    "pools is a list": put("service_pools_by_environment", [0.4, 0.4]),
    "string pool": put("service_pools_by_environment", MATH, "0.4"),
    "None pool": put("service_pools_by_environment", MATH, None),
    "NaN pool": put("service_pools_by_environment", MATH, float("nan")),
    "negative pool": put("service_pools_by_environment", MATH, -0.4),
    "bool pool": put("service_pools_by_environment", MATH, True),
    "schedule is a string": put("service_schedule", "schedule"),
    "schedule is a list": put("service_schedule", [1]),
    "schedule is missing": put("service_schedule", None),
    "schedule digest is a number": put("service_schedule_sha256", 5),
    "geometry is a string": put("service_picks_target", "2"),
    "payment policy is a list": put("service_payment_policy", ["service-first-scan-exploration/v1"]),
    "order digest is a dict": put("service_order_sha256", {}),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_every_malformed_archive_raises_settlement_error(name):
    record = good_record()
    validate(record)
    MALFORMED[name](record)
    with pytest.raises(SettlementError) as caught:
        validate(record)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("record", [None, [], "archive", 5])
def test_an_archive_that_is_not_a_map_raises_settlement_error(record):
    with pytest.raises(SettlementError):
        validate(record)


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -0.1, 1.5, "1", None, True])
def test_a_bad_cap_argument_raises_settlement_error(cap):
    with pytest.raises(SettlementError, match="cap"):
        validate(good_record(), cap=cap)


BAD_COUNTS = [float("nan"), float("inf"), -1, 0, True, False, 1.0, 1.5, "1", None, [1], 10**400]


@pytest.mark.parametrize("count", BAD_COUNTS)
def test_settle_checks_every_exploration_count(count):
    with pytest.raises(SettlementError, match="whole positive"):
        settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": count}})
    with pytest.raises(SettlementError, match="whole positive"):   # a bad map is refused even when aborted
        settle([], {MATH: 0.4, CODE: 0.4}, {MATH: {"x": count}}, aborted=True)


@pytest.mark.parametrize("exploration", [None, [], "x", {MATH: None}, {MATH: ["x"]}, {MATH: "x"}, {MATH: {5: 1}}, {5: {"x": 1}}])
def test_settle_checks_the_shape_of_its_exploration_input(exploration):
    with pytest.raises(SettlementError):
        settle([], {MATH: 0.4, CODE: 0.4}, exploration)


@pytest.mark.parametrize("change", [
    lambda a, e: a.update(batch="rows"),
    lambda a, e: a.update(batch=[None]),
    lambda a, e: a.update(batch=[{"hotkey": None, "env_name": MATH}]),
    lambda a, e: a.update(rewards_by_hotkey={"a": "1"}),
    lambda a, e: a.update(rewards_by_hotkey=[1]),
    lambda a, e: e.pop("pools"),
    lambda a, e: e.pop("picks_target"),
    lambda a, e: e.pop("schedule_sha256"),
    lambda a, e: e.update(pools=[0.4]),
    lambda a, e: e.update(pools={MATH: "0.4", CODE: 0.4}),
    lambda a, e: e.update(pools={MATH: float("nan"), CODE: 0.4}),
    lambda a, e: e.update(picks_target=0),
    lambda a, e: e.update(batch_slots=4.0),
    lambda a, e: e.update(batch_slots=True),
])
def test_settle_raises_settlement_error_on_malformed_archive_or_envelope(change):
    contract = contract_v2()
    a, e = archive([("a", MATH)]), envelope(contract, {MATH: 0.4, CODE: 0.4})
    settle_window(archive=dict(a), envelope=dict(e), contract=contract, exploration={}, aborted=False)
    change(a, e)
    with pytest.raises(SettlementError):
        settle_window(archive=a, envelope=e, contract=contract, exploration={}, aborted=False)
    for bad in (None, "yes", 1):
        with pytest.raises(SettlementError, match="disposition"):
            settle_window(archive=archive([]), envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}), contract=contract,
                          exploration={}, aborted=bad)


# ---- N4: no integer of any size can raise anything but SettlementError ----

HUGE = [10**400, -10**400, 10**309, 10**20]
HUGE_IDS = ["400 digits", "-400 digits", "309 digits", "10**20"]


@pytest.mark.parametrize("huge", HUGE, ids=HUGE_IDS)
@pytest.mark.parametrize("target", [
    ("service_pools_by_environment", MATH), ("service_scale_by_environment", MATH),
    ("service_training_by_environment", MATH, "a"), ("rewards_by_hotkey", "a"),
    ("service_exploration_by_environment", MATH, "x"), ("service_picks_target",), ("service_batch_slots",),
], ids=lambda t: "/".join(t))
def test_validate_raises_only_settlement_error_on_a_huge_integer(target, huge):
    record = good_record()
    validate(record)
    put(*target, huge)(record)
    record = json.loads(json.dumps(record))                      # JSON-representable, as on the wire
    with pytest.raises(SettlementError):
        validate(record)


@pytest.mark.parametrize("huge", HUGE, ids=HUGE_IDS)
@pytest.mark.parametrize("what", ["pool", "picks", "slots", "caller_rewards", "count", "cap_arg", "geometry_arg"])
def test_settle_and_validate_raise_only_settlement_error_on_a_huge_integer(what, huge):
    contract = contract_v2()
    a, e, x = archive([("a", MATH)]), envelope(contract, {MATH: 0.4, CODE: 0.4}), {MATH: {"x": 1}}
    if what == "pool":
        e["pools"][MATH] = huge
    elif what == "picks":
        e["picks_target"] = huge
    elif what == "slots":
        e["batch_slots"] = huge
    elif what == "caller_rewards":
        a["rewards_by_hotkey"] = {"a": huge}
    elif what == "count":
        x = {MATH: {"x": huge}}
    if what in ("cap_arg", "geometry_arg"):
        with pytest.raises(SettlementError):
            validate(good_record(), cap=huge) if what == "cap_arg" else validate(good_record(), picks=huge)
        return
    with pytest.raises(SettlementError):
        settle_window(archive=a, envelope=e, contract=contract, exploration=x, aborted=False)


# ---- N9: settle checks its envelope like validate does ----

@pytest.mark.parametrize("pools", [
    {MATH: 0.4}, {MATH: 0.4, CODE: 0.4, "ghost": 0.1}, {MATH: 0.4, "ghost": 0.4}, {},
], ids=["env missing", "extra env", "wrong env", "no pool"])
def test_settle_refuses_pools_that_are_not_exactly_the_active_envs(pools):
    contract = contract_v2()
    with pytest.raises(SettlementError, match="active"):
        settle_window(archive=archive([]), envelope=envelope(contract, pools), contract=contract,
                      exploration={}, aborted=False)


@pytest.mark.parametrize("contract", [None, "contract", {}, 5])
def test_settle_and_validate_refuse_a_missing_contract(contract):
    good = contract_v2()
    with pytest.raises(SettlementError):
        settle_window(archive=archive([]), envelope=envelope(good, {MATH: 0.4, CODE: 0.4}), contract=contract,
                      exploration={}, aborted=False)
    with pytest.raises(SettlementError):
        validate_service_archive_v2(good_record(), contract, cap=1.0, picks_target=PICKS, batch_slots=SLOTS)


def test_settle_refuses_an_envelope_whose_schedule_digest_is_not_its_schedule():
    contract = contract_v2()
    e = envelope(contract, {MATH: 0.4, CODE: 0.4})
    e["schedule_sha256"] = "0" * 64
    with pytest.raises(SettlementError, match="digest"):
        settle_window(archive=archive([]), envelope=e, contract=contract, exploration={}, aborted=False)


# ---- round 3: m5, m9 ----

def test_m9_every_paid_row_needs_an_integer_prompt_idx_at_settle_and_at_validate():
    good = settle([("a", MATH)], {MATH: 0.4, CODE: 0.4}, {})
    validate(good)
    for value in (None, "0", True, 0.0, [0]):
        record = json.loads(json.dumps(good))
        if value is None:
            del record["batch"][0]["prompt_idx"]
        else:
            record["batch"][0]["prompt_idx"] = value
        with pytest.raises(SettlementError, match="prompt_idx"):
            validate(record)
        contract = contract_v2()
        with pytest.raises(SettlementError, match="prompt_idx"):
            settle_window(archive={"batch": record["batch"]}, envelope=envelope(contract, {MATH: 0.4, CODE: 0.4}),
                          contract=contract, exploration={}, aborted=False)


def test_m5_an_unexpected_error_inside_settlement_is_logged_with_its_traceback(monkeypatch, caplog):
    from reliquary.services import settlement

    def defect(*args, **kwargs):
        return {}["our own bug"]
    monkeypatch.setattr(settlement, "_compute", defect)
    with caplog.at_level("ERROR", logger="reliquary.services.settlement"):
        with pytest.raises(SettlementError, match="KeyError"):
            settle([("a", MATH)], {MATH: 0.4, CODE: 0.4}, {})
    (record,) = [r for r in caplog.records if r.name == "reliquary.services.settlement"]
    assert record.exc_info is not None and record.exc_info[0] is KeyError and "defect" in caplog.text
    caplog.clear()
    with caplog.at_level("ERROR", logger="reliquary.services.settlement"):   # a plain refusal is not an exception log
        with pytest.raises(SettlementError):
            settle([("a", "reliquary_other_v1")], {MATH: 0.4, CODE: 0.4}, {})
    assert caplog.records == []
