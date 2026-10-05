"""Pure cooldown proposals from a declared, qualified observation panel."""

from __future__ import annotations

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.protocol.service_contract import ServiceContract, _identifier, _integer, _object, _sha
from reliquary.services.observations import observation_id, validate_observation
from reliquary.services.scoring import classify_signal


def propose_cooldown(contract: ServiceContract, population: dict, panel: dict, *,
                     window: int, distinct_groups_per_window: int,
                     previous: dict | None = None) -> dict:
    """Return a suggestion and resumable state; never change eligibility.

    Population and panel must describe the same source or active slice. The
    caller supplies the independent collector's qualification, not a claim
    that voluntarily submitted/cherry-picked groups represent that population.
    EMA operates on bounded rotation windows, stored as integer basis points.
    Invalid/stale/zero-supply measurements use the contract's static fallback
    immediately and clear the EMA; subsequent valid proposals remain rate-limited.
    """
    value = contract.to_dict()
    policy = value["policies"]["cooldown"]
    if policy["kind"] != "adaptive-rotation/v1":
        raise ValueError("an adaptive-rotation contract is required")
    _integer(window, "window", 0)
    _integer(distinct_groups_per_window, "distinct_groups_per_window", 0)
    _object(population, {"id", "kind", "sha256", "size"}, "population")
    _identifier(population["id"], "population.id")
    _sha(population["sha256"], "population.sha256")
    _integer(population["size"], "population.size", 0)
    if not isinstance(population["kind"], str) or population["kind"] not in {"source", "active"}:
        raise ValueError("population must be source or active")
    if population["kind"] == "source" and (
        population["id"] != value["dataset"]["id"]
        or population["sha256"] != value["dataset"]["sha256"]
    ):
        raise ValueError("source population does not match the ordered dataset")
    _object(panel, {"panel_id", "context_sha256", "population", "group_size", "declared_window",
                    "expected_group_ids", "observations", "independent", "qualified"}, "panel")
    _identifier(panel["panel_id"], "panel_id")
    if (panel["context_sha256"] != contract.context_sha256
        or not isinstance(panel["population"], dict)
        or canonical_sha256(panel["population"]) != canonical_sha256(population)):
        raise ValueError("panel context/population mismatch")
    group_size = _integer(panel["group_size"], "group_size", 2, 65536)
    declared = _integer(panel["declared_window"], "declared_window", 0, window)
    if any(type(panel[key]) is not bool for key in ("independent", "qualified")):
        raise ValueError("panel qualification must be explicit booleans")
    expected, rows = panel["expected_group_ids"], panel["observations"]
    if not isinstance(expected, list) or not isinstance(rows, list):
        raise ValueError("panel assignments and observations must be arrays")
    _integer(len(expected), "expected groups", 1, value["limits"]["max_groups"])
    for identifier in expected:
        _sha(identifier, "expected group id")
    expected_ids = set(expected)
    if len(expected_ids) != len(expected):
        raise ValueError("duplicate panel assignment")
    seen, completed, unusable = set(), set(), set()
    in_zone, tokens, stale, unverified = 0, 0, False, False
    for raw in rows:
        row = validate_observation(raw, contract)
        identifier = observation_id(row)
        if identifier not in expected_ids or identifier in seen:
            raise ValueError("unexpected or duplicate panel response")
        seen.add(identifier)
        if row["expected_samples"] != group_size:
            raise ValueError("panel groups are not comparable")
        _integer(row["window"], "observation window", declared, window)
        stale |= window - row["window"] > policy["freshness_windows"]
        tokens += sum(row["tokens"])
        if tokens > value["limits"]["max_tokens"]:
            raise ValueError("panel exceeds its token budget")
        verification = row["verification"]
        verified = all(verification[key] == "verified" for key in ("generation", "sampling"))
        unverified |= not verified
        rewards = [r / 10000 if r is not None else None for r in row["rewards_bps"]]
        signal = classify_signal(rewards, expected=group_size,
                                 sigma_min_bps=value["scoring"]["sigma_min_bps"])
        if not verified or verification["grading"] != "graded" or signal.category == "unknown":
            unusable.add(identifier)
        else:
            completed.add(identifier)
            in_zone += signal.in_zone
    count = len(completed)
    coverage_bps = count * 10000 // len(expected_ids)
    panel_sha256 = canonical_sha256(panel)
    prior_ema, prior_windows = None, None
    if previous is not None:
        if (not isinstance(previous, dict)
            or previous.get("schema") != "cooldown-suggestion/v1"
            or previous.get("contract_sha256") != contract.sha256
            or canonical_sha256(previous.get("population")) != canonical_sha256(population)):
            raise ValueError("previous cooldown belongs to another contract/population")
        _integer(previous.get("window"), "previous window", 0, window)
        prior_windows = _integer(previous.get("windows"), "previous windows",
                                 policy["min_windows"], policy["max_windows"])
        state = _object(previous.get("state"), {"ema_windows_bps", "panel_sha256"}, "previous state")
        _sha(state["panel_sha256"], "previous panel digest")
        prior_ema = state["ema_windows_bps"]
        if prior_ema is not None:
            _integer(prior_ema, "previous EMA", policy["min_windows"] * 10000,
                     policy["max_windows"] * 10000)
        if previous["window"] == window:
            if state["panel_sha256"] != panel_sha256 or previous.get("distinct_groups_per_window") != distinct_groups_per_window:
                raise ValueError("cooldown window already has different evidence")
            return previous
    reasons = []
    if not panel["independent"]:
        reasons.append("selective_panel")
    if not panel["qualified"]:
        reasons.append("unqualified_panel")
    if unverified:
        reasons.append("unverified_panel")
    if stale:
        reasons.append("stale_panel")
    if count < policy["min_panel_groups"]:
        reasons.append("insufficient_groups")
    if count * 10000 < policy["coverage_bps"] * len(expected_ids):
        reasons.append("insufficient_coverage")
    if not population["size"]:
        reasons.append("empty_population")
    if not distinct_groups_per_window:
        reasons.append("zero_consumption")
    if count and not in_zone:
        reasons.append("zero_signal")
    fallback = bool(reasons)
    ema = None
    if fallback:
        windows = policy["fallback_windows"]
    else:
        # N*p/Q*margin, in 1/10000-window units. Integer ceil preserves even
        # rare positive signal; do not quantize the probability to integer BPS.
        denominator = count * distinct_groups_per_window
        numerator = population["size"] * in_zone * policy["margin_bps"]
        rotation_bps = (numerator + denominator - 1) // denominator
        bounded = max(policy["min_windows"] * 10000,
                      min(policy["max_windows"] * 10000, rotation_bps))
        if bounded != rotation_bps:
            reasons.append("bounded_rotation")
        if prior_ema is None:
            ema = bounded
        elif previous["state"]["panel_sha256"] == panel_sha256 and previous["distinct_groups_per_window"] == distinct_groups_per_window:
            ema = prior_ema
            reasons.append("panel_reused")
        else:
            alpha = policy["smoothing_bps"]
            ema = (alpha * bounded + (10000 - alpha) * prior_ema + 5000) // 10000
            reasons.append("smoothed")
        windows = (ema + 9999) // 10000
        if prior_windows is not None:
            if abs(windows - prior_windows) <= policy["hysteresis_windows"]:
                windows = prior_windows
                reasons.append("hysteresis")
            elif abs(windows - prior_windows) > policy["max_change_windows"]:
                direction = 1 if windows > prior_windows else -1
                windows = prior_windows + direction * policy["max_change_windows"]
                reasons.append("rate_limited")
    return {
        "schema": "cooldown-suggestion/v1", "contract_sha256": contract.sha256,
        "context_sha256": contract.context_sha256, "population": dict(population),
        "window": window, "distinct_groups_per_window": distinct_groups_per_window,
        "windows": windows, "fallback": fallback, "reasons": reasons,
        "measurement": {
            "panel_id": panel["panel_id"], "panel_sha256": panel_sha256,
            "independent": panel["independent"], "qualified": panel["qualified"],
            "declared_window": declared,
            "group_size": group_size, "expected_groups": len(expected_ids),
            "completed_groups": count, "in_zone_groups": in_zone,
            "coverage_bps": coverage_bps,
            "p_sample": {"numerator": in_zone, "denominator": count} if count else None,
            "p_assignment_bounds": {"low": in_zone, "high": in_zone + len(expected_ids) - count,
                                    "denominator": len(expected_ids)},
            "completed_group_ids": sorted(completed),
            "nonresponse_group_ids": sorted(expected_ids - seen),
            "unusable_group_ids": sorted(unusable),
        },
        "state": {"ema_windows_bps": ema, "panel_sha256": panel_sha256},
    }
