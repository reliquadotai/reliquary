import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.cooldown_policy import propose_cooldown
from reliquary.services.observations import observation_id


def contract(**overrides):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["eligibility"] = {
        "kind": "dataset-epoch/v1", "coverage_bps": 8000, "refresh_windows": 1, "max_epoch_windows": 100,
    }
    value["policies"]["cooldown"] = {
        "kind": "adaptive-rotation/v1", "min_windows": 1, "max_windows": 1000,
        "margin_bps": 10000, "min_panel_groups": 2, "coverage_bps": 10000,
        "freshness_windows": 5, "smoothing_bps": 10000,
        "hysteresis_windows": 0, "max_change_windows": 1000, "fallback_windows": 7,
        **overrides,
    }
    return ServiceContract.from_dict(value)


def population(ordered, *, size=1000, active=False):
    dataset = ordered.to_dict()["dataset"]
    return {"id": "active-slice" if active else dataset["id"],
            "kind": "active" if active else "source",
            "sha256": "f" * 64 if active else dataset["sha256"], "size": size}


def panel(ordered, pop, rewards, *, window=10, identifier="panel-1"):
    rows = [{
        "schema": "prompt-observation/v1", "context_sha256": ordered.context_sha256,
        "row_id": f"row-{i}", "group_id": "draw-0", "expected_samples": 4,
        "sample_ids": [f"sample-{j}" for j in range(len(scores))], "rewards_bps": list(scores),
        "tokens": [1] * len(scores), "window": window,
        "verification": {"generation": "verified", "sampling": "verified", "grading": "graded"},
        "source_sha256": "a" * 64,
    } for i, scores in enumerate(rewards)]
    return {"panel_id": identifier, "context_sha256": ordered.context_sha256,
            "population": dict(pop), "group_size": 4, "declared_window": window,
            "expected_group_ids": [observation_id(row) for row in rows],
            "observations": rows, "independent": True, "qualified": True}


MIXED = [0, 0, 10000, 10000]
UNIFORM = [0, 0, 0, 0]


def suggest(ordered, pop, samples, *, window=10, q=10, previous=None):
    return propose_cooldown(ordered, pop, samples, window=window,
                            distinct_groups_per_window=q, previous=previous)


def test_rotation_has_window_units_and_matching_source_or_active_denominator():
    ordered = contract(margin_bps=8500)
    source, active = population(ordered), population(ordered, size=500, active=True)
    source_panel = panel(ordered, source, [MIXED, MIXED, UNIFORM, UNIFORM])
    source_result = suggest(ordered, source, source_panel)
    active_result = suggest(ordered, active, panel(ordered, active, [MIXED] * 4))
    assert source_result["windows"] == active_result["windows"] == 43  # ceil(1000*.5/10*.85)
    assert suggest(ordered, source, source_panel, q=20)["windows"] == 22
    with pytest.raises(ValueError, match="population mismatch"):
        suggest(ordered, active, source_panel)  # Do not apply a source probability to a curated count.
    assert suggest(ordered, source, source_panel) == source_result


def test_fractional_scores_use_actual_sigma_gate():
    ordered, rewards = contract(), [[1000, 1000, 2000, 2000], [0, 5000, 10000, 0]]
    pop = population(ordered)
    result = suggest(ordered, pop, panel(ordered, pop, rewards))
    assert result["measurement"]["in_zone_groups"] == 1
    assert result["windows"] == 50


@pytest.mark.parametrize("flag,reason", [("independent", "selective_panel"), ("qualified", "unqualified_panel")])
def test_selective_or_unqualified_panel_cannot_control_cooldown(flag, reason):
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    samples[flag] = False
    result = suggest(ordered, pop, samples)
    assert result["fallback"] and result["windows"] == 7
    assert reason in result["reasons"] and result["state"]["ema_windows_bps"] is None


def test_nonresponses_and_unknown_scores_remain_visible_and_are_not_zero_signal():
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED, [0, None], UNIFORM])
    missing = samples["expected_group_ids"][2]
    samples["observations"].pop()
    result = suggest(ordered, pop, samples)
    measurement = result["measurement"]
    assert measurement["completed_groups"] == 1
    assert measurement["nonresponse_group_ids"] == [missing]
    assert measurement["unusable_group_ids"] == [samples["expected_group_ids"][1]]
    assert measurement["p_sample"] == {"numerator": 1, "denominator": 1}
    assert measurement["p_assignment_bounds"] == {"low": 1, "high": 3, "denominator": 3}
    assert result["fallback"] and {"insufficient_groups", "insufficient_coverage"} <= set(result["reasons"])
    samples["observations"] = []
    result = suggest(ordered, pop, samples)
    assert result["measurement"]["p_sample"] is None
    assert "zero_signal" not in result["reasons"]


def test_coverage_and_minimum_groups_are_separate_exact_gates():
    ordered = contract(coverage_bps=7500, min_panel_groups=3)
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 4)
    samples["observations"].pop()
    result = suggest(ordered, pop, samples)
    assert not result["fallback"] and result["measurement"]["coverage_bps"] == 7500
    samples["observations"].pop()
    result = suggest(ordered, pop, samples)
    assert result["fallback"] and {"insufficient_groups", "insufficient_coverage"} <= set(result["reasons"])


@pytest.mark.parametrize("change", ["assignment", "response", "unexpected", "incomparable", "future", "context", "unverified"])
def test_bad_panel_identity_or_sampling_never_increases_qualified_coverage(change):
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    if change == "assignment":
        samples["expected_group_ids"].append(samples["expected_group_ids"][0])
    elif change == "response":
        samples["observations"].append(deepcopy(samples["observations"][0]))
    elif change == "unexpected":
        samples["observations"][0]["group_id"] = "other-draw"
    elif change == "incomparable":
        samples["observations"][0]["expected_samples"] = 8
    elif change == "future":
        samples["observations"][0]["window"] += 1
    elif change == "context":
        samples["context_sha256"] = "b" * 64
    else:
        samples["observations"][0]["verification"]["sampling"] = "unverified"
        result = suggest(ordered, pop, samples)
        assert result["fallback"] and "unverified_panel" in result["reasons"]
        return
    with pytest.raises(ValueError):
        suggest(ordered, pop, samples)


def test_freshness_boundary_zero_supply_and_zero_consumption_use_frozen_fallback():
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    assert not suggest(ordered, pop, samples, window=15)["fallback"]
    stale = suggest(ordered, pop, samples, window=16)
    assert stale["windows"] == 7 and "stale_panel" in stale["reasons"]
    zero = suggest(ordered, pop, panel(ordered, pop, [UNIFORM] * 2))
    assert zero["windows"] == 7 and "zero_signal" in zero["reasons"]
    stopped = suggest(ordered, pop, samples, q=0)
    assert stopped["windows"] == 7 and "zero_consumption" in stopped["reasons"]


def test_bounds_ema_hysteresis_and_change_cap_have_explicit_state():
    ordered = contract(smoothing_bps=5000, max_change_windows=10, hysteresis_windows=2)
    pop = population(ordered)
    previous = suggest(ordered, pop, panel(ordered, pop, [MIXED] * 2))
    next_panel = panel(ordered, pop, [MIXED, UNIFORM], window=11, identifier="panel-2")
    next_result = suggest(ordered, pop, next_panel, window=11, previous=previous)
    assert next_result["state"]["ema_windows_bps"] == 750000
    assert next_result["windows"] == 90 and "rate_limited" in next_result["reasons"]
    # Reusing an unchanged panel must not repeatedly apply its evidence to EMA.
    reused = suggest(ordered, pop, next_panel, window=12, previous=next_result)
    assert reused["state"]["ema_windows_bps"] == 750000 and reused["windows"] == 80
    assert "panel_reused" in reused["reasons"]
    assert suggest(ordered, pop, next_panel, window=12, previous=reused) == reused
    near = deepcopy(next_result)
    near["windows"] = 76
    held = suggest(ordered, pop, next_panel, window=12, previous=near)
    assert held["windows"] == 76 and "hysteresis" in held["reasons"]
    with pytest.raises(ValueError, match="different evidence"):
        suggest(ordered, pop, next_panel, window=11, q=9, previous=next_result)
    upper_contract, lower_contract = contract(max_windows=80), contract(min_windows=3)
    upper_pop, lower_pop = population(upper_contract), population(lower_contract, size=1)
    upper = suggest(upper_contract, upper_pop, panel(upper_contract, upper_pop, [MIXED] * 2))
    lower = suggest(lower_contract, lower_pop, panel(lower_contract, lower_pop, [MIXED] * 2))
    assert upper["windows"] == 80 and lower["windows"] == 3
    assert "bounded_rotation" in upper["reasons"] and "bounded_rotation" in lower["reasons"]


def test_previous_state_cannot_cross_population_or_contract_and_inputs_stay_unchanged():
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    before = deepcopy((pop, samples))
    previous = suggest(ordered, pop, samples)
    assert (pop, samples) == before
    active = population(ordered, active=True)
    with pytest.raises(ValueError, match="another contract/population"):
        suggest(ordered, active, panel(ordered, active, [MIXED] * 2, window=11), window=11, previous=previous)
    changed = contract(margin_bps=8000)
    with pytest.raises(ValueError, match="another contract/population"):
        suggest(changed, pop, panel(changed, pop, [MIXED] * 2, window=11), window=11, previous=previous)


def test_fallback_clears_stale_ema_and_valid_recovery_is_rate_limited():
    ordered = contract(smoothing_bps=1000, max_change_windows=5)
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    previous = suggest(ordered, pop, samples)
    stale = suggest(ordered, pop, samples, window=16, previous=previous)
    assert stale["windows"] == 7 and stale["state"]["ema_windows_bps"] is None
    fresh = panel(ordered, pop, [MIXED, UNIFORM], window=17, identifier="fresh-panel")
    recovered = suggest(ordered, pop, fresh, window=17, previous=stale)
    assert recovered["state"]["ema_windows_bps"] == 500000  # No inherited stale estimate.
    assert recovered["windows"] == 12 and "rate_limited" in recovered["reasons"]


@pytest.mark.parametrize("parameter", ["window", "q"])
def test_boolean_or_negative_window_units_are_rejected(parameter):
    ordered = contract()
    pop = population(ordered)
    samples = panel(ordered, pop, [MIXED] * 2)
    for bad in (True, -1, 1.5):
        with pytest.raises(ValueError):
            suggest(ordered, pop, samples, **{parameter: bad})
