"""Per-window service settlement (decision B money), pure functions.

Training is recomputed per env from the paid ``batch`` rows; first-scan exploration is added; when an
env's nominal training plus nominal exploration would exceed its pool, BOTH are scaled by the same
factor (user ruling: proportional split, exploration is not paid first).

Per env ``e`` with pool ``P_e`` and slot geometry ``T = picks_target * batch_slots``::

    price_e   = P_e / T                                   # nominal training group
    xprice_e  = price_e * price_bps / 10000               # nominal exploration entitlement
    nominal_e = (paid training rows of e) * price_e
    X_e       = sum over hotkeys of count_h * xprice_e    # counts are whole entitlements (integers)
    scale_e   = 1                       if nominal_e + X_e <= P_e (+ 1e-12)
              = P_e / (nominal_e + X_e) otherwise
    training paid to h    = rows_h  * price_e  * scale_e
    exploration paid to h = count_h * xprice_e * scale_e
    burned_e  = P_e - training_e - exploration_e   (0 when scale_e < 1)

The archive carries the integer entitlement COUNTS (``service_exploration_by_environment``), never
float amounts. ``price_bps`` / ``cap_bps`` come from the contract in settle and in validate alike;
the slot geometry comes from the window envelope in settle and from the caller's protocol constants
in validate, which refuses an archive that claims another geometry. A zero pool has no exploration.

Float identity is a money invariant: settle and validate share ``_check_exploration`` and
``_compute``, so the checks, the operations and their order are the same (envs sorted, hotkeys
sorted, ``math.fsum`` for sums, per-hotkey totals accumulated env by env: training first, then
exploration). Comparisons use the absolute 1e-12 of ``reliquary/services/runtime.py``.

Aborted windows: ``settle_window(aborted=True)`` pays no exploration and stamps
``window_status = "aborted"``; validation refuses an aborted archive that awards exploration. By the
existing convention the reward map of an aborted window is NOT paid by weight replay, so the
training amounts it still carries are informational.

Every malformed input, archive or argument, raises ``SettlementError`` (a ``ValueError``).
"""
from __future__ import annotations

import math

from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.service_schedule import ServiceSchedule
from reliquary.services.exploration import (
    exploration_cap, exploration_price, exploration_within_cap, training_group_price,
)

SERVICE_PAYMENT_POLICY_V2 = "service-first-scan-exploration/v1"
_EPS = 1e-12
_MAX_COUNT = 2**53  # far above any cap; keeps count * price inside float range


class SettlementError(ValueError):
    pass


def _number(value, what: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise SettlementError(f"invalid {what}")
    return float(value)


def _amounts(value, what: str) -> dict[str, float]:
    """A ``{str: finite non-negative number}`` map, or SettlementError."""
    if not isinstance(value, dict):
        raise SettlementError(f"{what} must be a map")
    out: dict[str, float] = {}
    for key, amount in value.items():
        if not isinstance(key, str):
            raise SettlementError(f"{what} has a non-string key")
        out[key] = _number(amount, what)
    return out


def _geometry(picks, slots) -> tuple[int, int]:
    if type(picks) is not int or type(slots) is not int or picks < 1 or slots < 1:
        raise SettlementError("invalid service slot geometry")
    return picks, slots


def _reward_bps(contract: ServiceContract) -> tuple[int, int]:
    try:
        policy = contract.reward_policy
        price_bps, cap_bps = policy["price_bps"], policy["cap_bps"]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SettlementError("contract has no exploration reward policy") from exc
    if type(price_bps) is not int or type(cap_bps) is not int or price_bps < 0 or cap_bps < 0:
        raise SettlementError("invalid exploration reward policy")
    return price_bps, cap_bps


def _count_rows(batch, pools: dict[str, float]) -> dict[str, dict[str, int]]:
    if batch is None:
        batch = []
    if not isinstance(batch, list):
        raise SettlementError("batch must be a list of paid groups")
    counts: dict[str, dict[str, int]] = {env: {} for env in pools}
    for row in batch:
        if not isinstance(row, dict):
            raise SettlementError("batch row must be a map")
        env, hotkey = row.get("env_name"), row.get("hotkey")
        if not isinstance(env, str) or env not in pools or not isinstance(hotkey, str):
            raise SettlementError("paid group outside the window envelope")
        counts[env][hotkey] = counts[env].get(hotkey, 0) + 1
    return counts


def _check_exploration(exploration, pools: dict[str, float], picks: int, slots: int,
                       price_bps: int, cap_bps: int) -> dict[str, dict[str, int]]:
    """THE per-entry check of exploration counts, shared by settle and validate.

    Returns ``{env: {hotkey: count}}`` without the empty env maps, envs and hotkeys sorted.
    """
    if not isinstance(exploration, dict):
        raise SettlementError("exploration must be a map of envs")
    for env in exploration:
        if not isinstance(env, str) or env not in pools:
            raise SettlementError("exploration outside the window envelope")
    out: dict[str, dict[str, int]] = {}
    for env in sorted(exploration):
        entries = exploration[env]
        if not isinstance(entries, dict):
            raise SettlementError("an env's exploration must be a map of hotkeys")
        for hotkey, count in entries.items():
            if not isinstance(hotkey, str):
                raise SettlementError("exploration hotkey must be a string")
            if type(count) is not int or not 1 <= count <= _MAX_COUNT:
                raise SettlementError("exploration must be a whole positive number of entitlements")
        if not entries:
            continue
        total = sum(entries.values())
        if pools[env] <= 0:
            raise SettlementError("a zero pool has no exploration")
        if total > _MAX_COUNT or not exploration_within_cap(
                total, price=exploration_price(pools[env], picks_target=picks, batch_slots=slots, price_bps=price_bps),
                cap=exploration_cap(pools[env], cap_bps=cap_bps)):
            raise SettlementError("exploration exceeds its per-env cap")
        out[env] = {hotkey: entries[hotkey] for hotkey in sorted(entries)}
    return out


def _compute(pools: dict[str, float], picks: int, slots: int, price_bps: int,
             counts: dict[str, dict[str, int]], entitlements: dict[str, dict[str, int]]):
    """Return (training, scales, exploration_paid, rewards). Shared by settle and validate."""
    training: dict[str, dict[str, float]] = {}
    paid_exploration: dict[str, dict[str, float]] = {}
    scales: dict[str, float] = {}
    rewards: dict[str, float] = {}
    for env in sorted(pools):
        pool = pools[env]
        price = training_group_price(pool, picks_target=picks, batch_slots=slots)
        xprice = exploration_price(pool, picks_target=picks, batch_slots=slots, price_bps=price_bps)
        rows = counts.get(env, {})
        units = entitlements.get(env, {})
        nominal = sum(rows.values()) * price
        spent = math.fsum(units[h] * xprice for h in sorted(units))
        total = nominal + spent
        scale = 1.0 if total <= pool + _EPS else pool / total
        scales[env] = scale
        training[env] = {h: rows[h] * price * scale for h in sorted(rows)}
        paid_exploration[env] = {h: units[h] * xprice * scale for h in sorted(units)}
        for h, v in training[env].items():
            rewards[h] = rewards.get(h, 0.0) + v
        for h, v in paid_exploration[env].items():
            rewards[h] = rewards.get(h, 0.0) + v
    return training, scales, paid_exploration, rewards


def settle_window(*, archive: dict, envelope: dict, contract: ServiceContract,
                  exploration: dict[str, dict[str, int]], aborted: bool) -> dict:
    """Stamp the service money on a window archive.

    ``exploration`` is ``{env: ExplorationLedger.payable(window, environment=env)}``: whole
    entitlement counts per hotkey. ``contract`` supplies ``price_bps`` / ``cap_bps``.
    """
    if not isinstance(archive, dict) or not isinstance(envelope, dict):
        raise SettlementError("archive and envelope must be maps")
    if type(aborted) is not bool:
        raise SettlementError("invalid service settlement disposition")
    for key in ("order_sha256", "schedule", "schedule_sha256", "pools", "picks_target", "batch_slots"):
        if key not in envelope:
            raise SettlementError(f"window envelope has no {key}")
    if envelope["order_sha256"] != contract.sha256:
        raise SettlementError("window envelope belongs to another order")
    price_bps, cap_bps = _reward_bps(contract)
    pools = _amounts(envelope["pools"], "service pool")
    pools = {env: pools[env] for env in sorted(pools)}
    picks, slots = _geometry(envelope["picks_target"], envelope["batch_slots"])
    counts = _count_rows(archive.get("batch"), pools)
    # Checked even when aborted: a bad map is a caller bug whatever the disposition.
    explored = _check_exploration(exploration, pools, picks, slots, price_bps, cap_bps)
    if aborted:
        explored = {}
    training, scales, _, rewards = _compute(pools, picks, slots, price_bps, counts, explored)
    price = {env: training_group_price(pools[env], picks_target=picks, batch_slots=slots) for env in pools}
    unscaled: dict[str, float] = {}
    for env in sorted(counts):
        for h, n in counts[env].items():
            unscaled[h] = unscaled.get(h, 0.0) + n * price[env]
    caller = _amounts(archive.get("rewards_by_hotkey") or {}, "caller rewards_by_hotkey")
    delta = max((abs(unscaled.get(k, 0.0) - caller.get(k, 0.0)) for k in set(caller) | set(unscaled)),
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


def validate_service_archive_v2(record: dict, contract: ServiceContract, *, cap: float,
                                picks_target: int, batch_slots: int) -> None:
    """Refuse (``SettlementError``) an archive whose service money cannot be recomputed.

    ``cap`` is the task's emission cap; ``picks_target`` / ``batch_slots`` are the slot geometry the
    CALLER expects from the protocol constants. The archive's own copy must agree: it is never the
    source, otherwise ``picks_target=1`` would let a handful of rows take a whole pool.
    """
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 <= cap <= 1 + _EPS:
        raise SettlementError("invalid service archive cap")
    picks, slots = _geometry(picks_target, batch_slots)
    if not isinstance(record, dict):
        raise SettlementError("service archive must be a map")
    price_bps, cap_bps = _reward_bps(contract)
    if record.get("service_payment_policy") != SERVICE_PAYMENT_POLICY_V2:
        raise SettlementError("service archive payment policy is missing or unsupported")
    if record.get("service_order_sha256") != contract.sha256:
        raise SettlementError("service archive belongs to another order")
    try:
        schedule = ServiceSchedule.from_dict(record.get("service_schedule") or {}, contract)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SettlementError(f"invalid service schedule: {exc}") from exc
    if schedule.sha256 != record.get("service_schedule_sha256"):
        raise SettlementError("service schedule digest mismatch")
    pools = record.get("service_pools_by_environment")
    if not isinstance(pools, dict) or set(pools) != set(schedule.active_environments()):
        raise SettlementError("service pools must cover exactly the active environments")
    if _geometry(record.get("service_picks_target"), record.get("service_batch_slots")) != (picks, slots):
        raise SettlementError("service slot geometry differs from the protocol's")
    pools = _amounts(pools, "service pool")
    pools = {env: pools[env] for env in sorted(pools)}
    for env, pool in pools.items():
        if pool > cap * schedule.share_bps(env) / 10000 + _EPS:
            raise SettlementError("service pool exceeds the env's share of the task cap")
    training = record.get("service_training_by_environment")
    if not isinstance(training, dict) or not set(training) <= set(pools):
        raise SettlementError("service training map is missing or names an env outside the pools")
    training = {env: _amounts(training[env], "service training amount") for env in training}
    scales = _amounts(record.get("service_scale_by_environment"), "service scale")
    rewards = _amounts(record.get("rewards_by_hotkey"), "rewards_by_hotkey")
    entitlements = _check_exploration(record.get("service_exploration_by_environment"), pools, picks, slots,
                                      price_bps, cap_bps)
    if record.get("window_status") == "aborted" and entitlements:
        raise SettlementError("aborted service window cannot award exploration")
    counts = _count_rows(record.get("batch"), pools)
    want_training, want_scales, want_paid, want_rewards = _compute(pools, picks, slots, price_bps, counts,
                                                                   entitlements)
    if not _same_map(scales, want_scales):
        raise SettlementError("service scale differs from the recomputed scale")
    for env in sorted(pools):
        if not _same_map(training.get(env, {}), want_training[env]):
            raise SettlementError("service training amounts differ from the batch rows")
        paid = math.fsum(list(want_training[env].values()) + list(want_paid[env].values()))
        if paid > pools[env] + _EPS or (want_scales[env] < 1 and abs(paid - pools[env]) > _EPS):
            raise SettlementError("service env pool is not conserved")
    if not _same_map(rewards, want_rewards):
        raise SettlementError("rewards_by_hotkey differs from the recomputed service lanes")
