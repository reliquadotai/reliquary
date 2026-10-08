"""Per-window service settlement (decision B money), pure functions.

Training is recomputed per env from the paid ``batch`` rows; first-scan exploration is added; when an
env's nominal training plus nominal exploration would exceed its pool, BOTH are scaled by the same
factor (user ruling: proportional split, exploration is not paid first).

Per env ``e`` with pool ``P_e``::

    price_e   = P_e / (picks_target * batch_slots)        # nominal training group
    nominal_e = (paid training rows of e) * price_e
    X_e       = sum of NOMINAL exploration amounts of e
    scale_e   = 1                       if nominal_e + X_e <= P_e (+ 1e-12)
              = P_e / (nominal_e + X_e) otherwise
    training row paid = price_e * scale_e ;  exploration amount paid = amount * scale_e
    burned_e  = P_e - training_e - exploration_e   (0 when scale_e < 1)

Float identity is a money invariant: settle and validate share ``_compute`` so the operations and their
order are the same (envs sorted, hotkeys sorted, ``math.fsum`` for sums, per-hotkey totals accumulated
env by env: training first, then exploration). Tolerance is the absolute 1e-12 used by
``reliquary/services/runtime.py``.
"""
from __future__ import annotations

import math

from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.service_schedule import ServiceSchedule
from reliquary.services.exploration import exploration_cap, exploration_price, training_group_price

SERVICE_PAYMENT_POLICY_V2 = "service-first-scan-exploration/v1"
_EPS = 1e-12


class SettlementError(ValueError):
    pass


def _count_rows(batch, pools: dict[str, float]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {env: {} for env in pools}
    for row in batch or []:
        env, hotkey = row.get("env_name"), row.get("hotkey")
        if env not in pools or not isinstance(hotkey, str):
            raise SettlementError("paid group outside the window envelope")
        counts[env][hotkey] = counts[env].get(hotkey, 0) + 1
    return counts


def _compute(pools: dict[str, float], picks: int, slots: int, counts: dict[str, dict[str, int]],
             nominal_exploration: dict[str, dict[str, float]]):
    """Return (training, scales, exploration_paid, rewards). Shared by settle and validate."""
    training: dict[str, dict[str, float]] = {}
    paid_exploration: dict[str, dict[str, float]] = {}
    scales: dict[str, float] = {}
    rewards: dict[str, float] = {}
    for env in sorted(pools):
        pool = pools[env]
        price = training_group_price(pool, picks_target=picks, batch_slots=slots)
        rows = counts.get(env, {})
        nominal = sum(rows.values()) * price
        amounts = nominal_exploration.get(env, {})
        spent = math.fsum(amounts[h] for h in sorted(amounts))
        total = nominal + spent
        scale = 1.0 if total <= pool + _EPS else pool / total
        scales[env] = scale
        training[env] = {h: rows[h] * price * scale for h in sorted(rows)}
        paid_exploration[env] = {h: amounts[h] * scale for h in sorted(amounts)}
        for h, v in training[env].items():
            rewards[h] = rewards.get(h, 0.0) + v
        for h, v in paid_exploration[env].items():
            rewards[h] = rewards.get(h, 0.0) + v
    return training, scales, paid_exploration, rewards


def settle_window(*, archive: dict, envelope: dict, exploration: dict[str, dict[str, float]], aborted: bool) -> dict:
    pools = {env: float(envelope["pools"][env]) for env in sorted(envelope["pools"])}
    picks, slots = int(envelope["picks_target"]), int(envelope["batch_slots"])
    counts = _count_rows(archive.get("batch"), pools)
    # Archived exploration is the NOMINAL per-hotkey amount; the scale is archived separately and
    # replay applies it, so the whole-entitlement check stays exact.
    explored = {} if aborted else {env: dict(m) for env, m in sorted(exploration.items()) if m}
    for env, amounts in explored.items():
        if env not in pools:
            raise SettlementError("exploration outside the window envelope")
        if math.fsum(amounts[h] for h in sorted(amounts)) > exploration_cap(
                pools[env], cap_bps=int(envelope["cap_bps"])) + _EPS:
            raise SettlementError("exploration exceeds its per-env cap")
    training, scales, _, rewards = _compute(pools, picks, slots, counts, explored)
    price = {env: training_group_price(pools[env], picks_target=picks, batch_slots=slots) for env in pools}
    unscaled: dict[str, float] = {}
    for env in sorted(counts):
        for h, n in counts[env].items():
            unscaled[h] = unscaled.get(h, 0.0) + n * price[env]
    caller = archive.get("rewards_by_hotkey") or {}
    delta = max((abs(unscaled.get(k, 0.0) - float(caller.get(k, 0.0))) for k in set(caller) | set(unscaled)),
                default=0.0)
    out = {**archive,
           "service_payment_policy": SERVICE_PAYMENT_POLICY_V2,
           "service_order_sha256": envelope["order_sha256"],
           "service_schedule": envelope["schedule"],
           "service_schedule_sha256": envelope["schedule_sha256"],
           "service_pools_by_environment": pools,
           "service_picks_target": picks,
           "service_batch_slots": slots,
           "service_training_by_environment": training,
           "service_exploration_by_environment": explored,
           "service_scale_by_environment": scales,
           "service_training_recomputed_delta": delta,
           "rewards_by_hotkey": rewards}
    if aborted:
        out["window_status"] = "aborted"
    return out


def _same_map(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= _EPS for k in a)


def validate_service_archive_v2(record: dict, contract: ServiceContract, *, cap: float) -> None:
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 <= cap <= 1 + _EPS:
        raise ValueError("invalid service archive cap")
    if record.get("service_payment_policy") != SERVICE_PAYMENT_POLICY_V2:
        raise ValueError("service archive payment policy is missing or unsupported")
    if record.get("service_order_sha256") != contract.sha256:
        raise ValueError("service archive belongs to another order")
    schedule = ServiceSchedule.from_dict(record.get("service_schedule") or {}, contract)
    if schedule.sha256 != record.get("service_schedule_sha256"):
        raise ValueError("service schedule digest mismatch")
    pools = record.get("service_pools_by_environment")
    if not isinstance(pools, dict) or set(pools) != set(schedule.active_environments()):
        raise ValueError("service pools must cover exactly the active environments")
    picks, slots = record.get("service_picks_target"), record.get("service_batch_slots")
    if type(picks) is not int or type(slots) is not int or picks < 1 or slots < 1:
        raise ValueError("invalid service slot geometry")
    for env in sorted(pools):
        pool = pools[env]
        if type(pool) not in (int, float) or not math.isfinite(pool) or pool < 0:
            raise ValueError("invalid service pool")
        if pool > cap * schedule.share_bps(env) / 10000 + _EPS:
            raise ValueError("service pool exceeds the env's share of the task cap")
    pools = {env: float(pools[env]) for env in sorted(pools)}
    training = record.get("service_training_by_environment")
    exploration = record.get("service_exploration_by_environment")
    scales = record.get("service_scale_by_environment")
    rewards = record.get("rewards_by_hotkey")
    if not all(isinstance(m, dict) for m in (training, exploration, scales, rewards)):
        raise ValueError("service archive reward maps are required")
    if not set(training) <= set(pools) or not set(exploration) <= set(pools):
        raise ValueError("service archive maps name an env outside the pools")
    if record.get("window_status") == "aborted" and any(exploration.values()):
        raise ValueError("aborted service window cannot award exploration")
    policy = contract.reward_policy
    nominal: dict[str, dict[str, float]] = {}
    for env in sorted(pools):
        amounts = exploration.get(env) or {}
        price = exploration_price(pools[env], picks_target=picks, batch_slots=slots, price_bps=policy["price_bps"])
        for hotkey in sorted(amounts):
            amount = amounts[hotkey]
            if not isinstance(hotkey, str) or type(amount) not in (int, float) or not math.isfinite(amount):
                raise ValueError("invalid exploration amount")
            units = amount / price if price > 0 else 0
            if price <= 0 or abs(units - round(units)) > 1e-6 or round(units) < 1:
                raise ValueError("exploration amount is not a whole number of entitlements")
        if math.fsum(amounts[h] for h in sorted(amounts)) > exploration_cap(
                pools[env], cap_bps=policy["cap_bps"]) + _EPS:
            raise ValueError("exploration exceeds its per-env cap")
        nominal[env] = {h: float(amounts[h]) for h in amounts}
    try:
        counts = _count_rows(record.get("batch"), pools)
    except SettlementError as exc:
        raise ValueError(str(exc)) from exc
    want_training, want_scales, want_paid, want_rewards = _compute(pools, picks, slots, counts, nominal)
    if not _same_map(scales, want_scales):
        raise ValueError("service scale differs from the recomputed scale")
    for env in sorted(pools):
        if not _same_map(training.get(env) or {}, want_training[env]):
            raise ValueError("service training amounts differ from the batch rows")
        paid = math.fsum(list(want_training[env].values()) + list(want_paid[env].values()))
        if paid > pools[env] + _EPS or (want_scales[env] < 1 and abs(paid - pools[env]) > _EPS):
            raise ValueError("service env pool is not conserved")
    if not _same_map(rewards, want_rewards):
        raise ValueError("rewards_by_hotkey differs from the recomputed service lanes")
