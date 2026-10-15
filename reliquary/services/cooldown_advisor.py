"""Cooldown recommendation (decision E): ``cooldown = N * p / Q * margin``, RECOMMENDATION ONLY.

``p`` is the in-zone rate of FIRST scans (always reported, since the first scan is paid in both
lanes), ``N`` the env's prompt population and ``Q`` the smoothed number of groups the trainer
consumed per window. The result is bounded, smoothed (EMA), damped (hysteresis) and rate limited.

This module is pure: it reads nothing, writes nothing, never closes admission and never changes
a live cooldown. An operator applies the number by hand. With too little data it says so.

Caveat: miners choose any M of the 2M public seeds per prompt, so the observed in-zone rate is an
upper envelope of the true per-group rate (a miner can keep a subset that is in zone); it is not
corrected for here.
"""
from __future__ import annotations

import math

NOTE = ("Recommendation only, never applied automatically; the observed in-zone rate is an upper "
        "envelope of the true per-group rate because miners pick any M of the 2M public seeds.")

_POLICY_KEYS = ("margin_bps", "min_windows", "max_windows", "smoothing_bps", "hysteresis_windows",
                "max_change_windows", "min_first_scans")
_RAW_CEILING = 2 ** 53
_INT_CEILING = 2 ** 63  # a 400-digit int would otherwise reach float arithmetic and raise OverflowError


def _int(name: str, value, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _INT_CEILING:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63], got {str(value)[:40]!r}")
    return value


def _checked_policy(policy) -> dict:
    if not isinstance(policy, dict):
        raise ValueError("policy must be a dict")
    missing = [key for key in _POLICY_KEYS if key not in policy]
    if missing:
        raise ValueError(f"policy is missing {missing}")
    checked = {
        "margin_bps": _int("margin_bps", policy["margin_bps"], 1),
        "min_windows": _int("min_windows", policy["min_windows"], 1),
        "max_windows": _int("max_windows", policy["max_windows"], 1),
        "smoothing_bps": _int("smoothing_bps", policy["smoothing_bps"], 1),
        "hysteresis_windows": _int("hysteresis_windows", policy["hysteresis_windows"], 0),
        "max_change_windows": _int("max_change_windows", policy["max_change_windows"], 1),
        "min_first_scans": _int("min_first_scans", policy["min_first_scans"], 1),
    }
    if checked["min_windows"] > checked["max_windows"]:
        raise ValueError("min_windows must not exceed max_windows")
    if checked["smoothing_bps"] > 10000:
        raise ValueError("smoothing_bps must be at most 10000")
    return checked


def _checked_previous(previous) -> tuple[float | None, int | None]:
    if previous is None:
        return None, None
    if not isinstance(previous, dict):
        raise ValueError("previous must be a dict or None")
    ema, last = previous.get("ema_windows"), previous.get("recommended_windows")
    if ema is not None:
        if isinstance(ema, bool) or not isinstance(ema, (int, float)) or abs(ema) > _INT_CEILING \
                or not math.isfinite(ema) or ema < 0:
            raise ValueError(f"previous ema_windows invalid: {ema!r}")
        ema = float(ema)
    if last is not None:
        last = _int("previous recommended_windows", last, 0)
    return ema, last


def recommend_cooldown(*, policy: dict, population: int, first_scans: int, in_zone_first: int,
                       consumption: float, previous: dict | None) -> dict:
    """Return the recommendation dict. Valid but empty data gives ``status="insufficient_data"``
    (keeping the previous recommendation); invalid inputs raise ``ValueError``."""
    policy = _checked_policy(policy)
    population = _int("population", population)
    first_scans = _int("first_scans", first_scans)
    in_zone_first = _int("in_zone_first", in_zone_first)
    if in_zone_first > first_scans:
        raise ValueError("in_zone_first cannot exceed first_scans")
    if isinstance(consumption, bool) or not isinstance(consumption, (int, float)) \
            or abs(consumption) > _INT_CEILING or not math.isfinite(consumption) or consumption < 0:
        raise ValueError(f"consumption must be a finite number >= 0, got {consumption!r}")
    consumption = float(consumption)
    prior_ema, last = _checked_previous(previous)

    base = {"first_scans": first_scans, "in_zone_first": in_zone_first, "q": consumption,
            "p": (in_zone_first / first_scans) if first_scans else None, "note": NOTE}
    reasons: list[str] = []
    if first_scans < policy["min_first_scans"]:
        reasons.append("first_scans_below_minimum")
    if consumption <= 0:
        reasons.append("no_consumption")
    if population <= 0:
        reasons.append("empty_population")
    if reasons:
        return {**base, "status": "insufficient_data", "reasons": reasons, "raw_windows": None,
                "ema_windows": prior_ema, "recommended_windows": last}

    if in_zone_first == 0:
        reasons.append("no_in_zone_first_scans")
    exact = population * in_zone_first * policy["margin_bps"] / (first_scans * 10000 * consumption)
    raw = math.ceil(min(exact - 1e-9, _RAW_CEILING)) if exact > 1e-9 else 0
    raw = max(raw, 0)
    bounded = min(max(raw, policy["min_windows"]), policy["max_windows"])
    if bounded != raw:
        reasons.append("bounded")
    alpha = policy["smoothing_bps"] / 10000
    ema = float(bounded) if prior_ema is None else alpha * bounded + (1 - alpha) * prior_ema
    target = math.ceil(ema - 1e-9)
    if last is not None:
        delta = target - last
        if abs(delta) <= policy["hysteresis_windows"]:
            target = last
            if delta:
                reasons.append("hysteresis")
        elif abs(delta) > policy["max_change_windows"]:
            target = last + (policy["max_change_windows"] if delta > 0 else -policy["max_change_windows"])
            reasons.append("rate_limited")
    return {**base, "status": "ok", "reasons": reasons, "raw_windows": raw, "ema_windows": ema,
            "recommended_windows": int(target)}
