import json
import math

import pytest

from reliquary.services.cooldown_advisor import recommend_cooldown

POLICY = {"kind": "in-zone-rotation/v1", "margin_bps": 10000, "min_windows": 1, "max_windows": 1000,
          "smoothing_bps": 10000, "hysteresis_windows": 1, "max_change_windows": 50, "min_first_scans": 100}


def call(policy=POLICY, population=10000, first_scans=200, in_zone_first=60, consumption=100.0, previous=None):
    return recommend_cooldown(policy=policy, population=population, first_scans=first_scans,
                              in_zone_first=in_zone_first, consumption=consumption, previous=previous)


def test_formula_is_n_p_over_q_times_margin():
    result = call()
    assert result["status"] == "ok" and result["p"] == pytest.approx(0.3)
    assert result["raw_windows"] == 30 and result["recommended_windows"] == 30
    doubled = call(policy={**POLICY, "margin_bps": 20000})
    assert doubled["raw_windows"] == 60


def test_insufficient_first_scans_or_no_consumption_says_so():
    few = call(first_scans=99, in_zone_first=30, consumption=10.0)
    assert few["status"] == "insufficient_data" and "first_scans_below_minimum" in few["reasons"]
    none = call(first_scans=500, in_zone_first=30, consumption=0.0)
    assert none["status"] == "insufficient_data" and "no_consumption" in none["reasons"]
    empty = call(population=0)
    assert empty["status"] == "insufficient_data" and "empty_population" in empty["reasons"]


def test_bounds_ema_hysteresis_and_rate_limit():
    policy = {**POLICY, "smoothing_bps": 5000, "max_change_windows": 5}
    first = call(policy=policy)
    second = call(policy=policy, first_scans=400, in_zone_first=240, previous=first)
    assert second["ema_windows"] == pytest.approx((30 + 60) / 2)
    assert second["recommended_windows"] == 35 and "rate_limited" in second["reasons"]
    same = call(policy=policy, first_scans=400, in_zone_first=124,
                previous={**first, "recommended_windows": 30, "ema_windows": 30.0})
    assert same["recommended_windows"] == 30 and "hysteresis" in same["reasons"]
    # A zero change is not damping: no hysteresis reason.
    unchanged = call(policy=policy, previous={**first, "recommended_windows": 30, "ema_windows": 30.0})
    assert unchanged["recommended_windows"] == 30 and "hysteresis" not in unchanged["reasons"]
    capped = call(policy={**POLICY, "max_windows": 20})
    assert capped["recommended_windows"] == 20 and "bounded" in capped["reasons"]


def test_lower_bound_applies():
    result = call(policy={**POLICY, "min_windows": 7}, in_zone_first=1)
    assert result["raw_windows"] == 1 and result["recommended_windows"] == 7 and "bounded" in result["reasons"]


def test_zero_in_zone_is_ok_and_clamped_to_minimum():
    result = call(in_zone_first=0)
    assert result["status"] == "ok" and result["p"] == 0.0 and result["raw_windows"] == 0
    assert result["recommended_windows"] == POLICY["min_windows"] and "no_in_zone_first_scans" in result["reasons"]


def test_in_zone_rate_one_and_zero_scans():
    one = call(first_scans=200, in_zone_first=200)
    assert one["p"] == 1.0 and one["raw_windows"] == 100
    zero = call(first_scans=0, in_zone_first=0)
    assert zero["status"] == "insufficient_data" and zero["p"] is None
    assert zero["recommended_windows"] is None


def test_insufficient_data_keeps_previous_recommendation_and_reports_it():
    kept = call(first_scans=10, in_zone_first=5, previous={"ema_windows": 12.5, "recommended_windows": 13})
    assert kept["status"] == "insufficient_data"
    assert kept["recommended_windows"] == 13 and kept["ema_windows"] == 12.5


def test_tiny_consumption_does_not_overflow():
    # 5e-324 is the smallest float: the quotient really reaches inf before the ceiling clamp.
    result = call(consumption=5e-324, population=10**15)
    assert result["status"] == "ok" and result["recommended_windows"] == POLICY["max_windows"]


def test_note_states_the_upper_envelope_caveat_once():
    for result in (call(), call(first_scans=1, in_zone_first=1)):
        assert result["note"].count("upper envelope") == 1
        assert "recommendation only" in result["note"].lower()


@pytest.mark.parametrize("kwargs", [
    {"consumption": float("nan")}, {"consumption": float("inf")}, {"consumption": -1.0},
    {"population": -1}, {"first_scans": -1}, {"in_zone_first": -1},
    {"first_scans": 10, "in_zone_first": 11}, {"population": 1.5}, {"first_scans": True},
    {"consumption": "3"},
])
def test_invalid_inputs_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)


@pytest.mark.parametrize("bad", [
    {"margin_bps": 0}, {"min_windows": 0}, {"min_windows": 5, "max_windows": 4}, {"smoothing_bps": 0},
    {"smoothing_bps": 10001}, {"hysteresis_windows": -1}, {"max_change_windows": 0}, {"min_first_scans": 0},
])
def test_invalid_policy_raises_value_error(bad):
    with pytest.raises(ValueError):
        call(policy={**POLICY, **bad})
    with pytest.raises(ValueError):
        call(policy={k: v for k, v in POLICY.items() if k != "margin_bps"})


def test_invalid_previous_raises_value_error():
    with pytest.raises(ValueError):
        call(previous={"ema_windows": float("nan"), "recommended_windows": 3})
    with pytest.raises(ValueError):
        call(previous=[1])


def test_results_are_finite_and_json_safe():
    result = call()
    assert all(math.isfinite(v) for v in (result["p"], result["q"], result["ema_windows"]))
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("field", ["population", "first_scans", "in_zone_first"])
def test_four_hundred_digit_ints_are_value_errors_not_overflow(field):
    huge = 10 ** 400
    kwargs = {"population": 10000, "first_scans": 200, "in_zone_first": 60, field: huge}
    if field == "first_scans":
        kwargs["in_zone_first"] = 60
    with pytest.raises(ValueError):
        call(**kwargs)
    with pytest.raises(ValueError):
        call(policy={**POLICY, "max_windows": huge})
    with pytest.raises(ValueError):
        call(previous={"ema_windows": 1.0, "recommended_windows": huge})
    with pytest.raises(ValueError):
        call(previous={"ema_windows": huge, "recommended_windows": 3})
    with pytest.raises(ValueError):
        call(consumption=huge)
