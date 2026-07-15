#!/usr/bin/env python3
"""Replay the difficulty-auction counterfactual from window archives.

The report is deliberately explicit about population coverage. Historical R2
archives contain fully validated batch entries and runners-up, but not payloads
rejected before validation or as ``batch_filled``. Those missing submissions
cannot be treated as if their difficulty distribution were observed.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DELTAS = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_DEADLINES = (120.0, 180.0, 300.0, 360.0)
MAX_UNIT_INTERVAL_STD = 0.5 + 1e-12


@dataclass(frozen=True)
class Candidate:
    hotkey: str
    prompt_idx: int
    drand_round: int
    selection_digest: bytes
    mean_reward: float
    reward_std: float
    reward_count: int
    arrival_age_seconds: float | None
    response_time: float | None
    production_selected: bool
    production_rewarded: bool
    operator_id: str | None = None


def _finite_unit(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _non_negative_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _normalized_identity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _reward_stats(rewards: Iterable[Any]) -> tuple[float, float, int] | None:
    values = []
    for reward in rewards:
        number = _finite_unit(reward)
        if number is None:
            return None
        values.append(number)
    if not values:
        return None
    mean_reward = sum(values) / len(values)
    variance = sum(
        (reward - mean_reward) ** 2 for reward in values
    ) / len(values)
    return mean_reward, variance**0.5, len(values)


def _decode_reward_vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        compact = value.strip()
        if compact and all(char in "01" for char in compact):
            return [float(char) for char in compact]
        return None
    if isinstance(value, (list, tuple)):
        rewards = [_finite_unit(item) for item in value]
        if rewards and all(item is not None for item in rewards):
            return [float(item) for item in rewards]
    return None


def _entry_stats(entry: dict[str, Any]) -> tuple[float, float, int] | None:
    mean_reward = _finite_unit(entry.get("difficulty_auction_mean_reward"))
    try:
        reward_std = float(
            entry.get("difficulty_auction_reward_std", entry.get("sigma"))
        )
    except (TypeError, ValueError):
        reward_std = float("nan")
    reward_count = _positive_int(
        entry.get("difficulty_auction_reward_count", 8)
    )
    if (
        mean_reward is not None
        and math.isfinite(reward_std)
        and 0.0 <= reward_std <= MAX_UNIT_INTERVAL_STD
        and reward_count is not None
    ):
        return mean_reward, reward_std, reward_count

    rollouts = entry.get("rollouts")
    if isinstance(rollouts, list):
        stats = _reward_stats(
            rollout.get("reward")
            for rollout in rollouts
            if isinstance(rollout, dict)
        )
        if stats is not None:
            return stats

    rewards = _decode_reward_vector(entry.get("reward_vector"))
    return _reward_stats(rewards) if rewards is not None else None


def _digest_bytes(value: Any, fallback: str) -> bytes:
    if isinstance(value, str):
        try:
            decoded = bytes.fromhex(value)
        except ValueError:
            decoded = b""
        if len(decoded) == 32:
            return decoded
    return hashlib.sha256(fallback.encode()).digest()


def _candidate_from_entry(entry: dict[str, Any]) -> Candidate | None:
    stats = _entry_stats(entry)
    if stats is None:
        return None
    try:
        raw_hotkey = entry["hotkey"]
        if raw_hotkey is None:
            return None
        hotkey = str(raw_hotkey).strip()
        prompt_idx = int(entry["prompt_idx"])
        drand_round = int(
            entry.get("submitted_drand_round")
            or entry.get("drand_round")
            or 0
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not hotkey or prompt_idx < 0 or drand_round < 0:
        return None
    digest = _digest_bytes(
        entry.get("selection_digest") or entry.get("merkle_root"),
        f"{hotkey}:{prompt_idx}:{drand_round}",
    )
    mean_reward, reward_std, reward_count = stats

    return Candidate(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        drand_round=drand_round,
        selection_digest=digest,
        mean_reward=mean_reward,
        reward_std=reward_std,
        reward_count=reward_count,
        arrival_age_seconds=_non_negative_float(
            entry.get("arrival_age_seconds")
        ),
        response_time=_non_negative_float(entry.get("response_time")),
        production_selected=bool(entry.get("selected_for_batch", False)),
        production_rewarded=bool(entry.get("rewarded", False)),
        operator_id=_normalized_identity(
            entry.get("difficulty_auction_operator_id")
        ),
    )


def extract_window_candidates(
    archive: dict[str, Any],
    *,
    environment: str,
) -> tuple[list[Candidate], int]:
    candidates: list[Candidate] = []
    missing_scores = 0
    seen: set[tuple[str, int, bytes]] = set()
    for field in ("batch", "runners_up"):
        entries = archive.get(field, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("env_name", archive.get("environment", ""))) != environment:
                continue
            candidate = _candidate_from_entry(entry)
            if candidate is None:
                missing_scores += 1
                continue
            key = (
                candidate.hotkey,
                candidate.prompt_idx,
                candidate.selection_digest,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates, missing_scores


def difficulty_value(candidate: Candidate, delta: float) -> float:
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and non-negative")
    return candidate.reward_std * (1.0 - candidate.mean_reward) ** delta


def _canonical_key(candidate: Candidate) -> bytes:
    digest = hashlib.sha256()
    digest.update(candidate.hotkey.encode())
    digest.update(candidate.prompt_idx.to_bytes(8, "big", signed=False))
    digest.update(candidate.selection_digest)
    return digest.digest()


def select_candidates(
    candidates: Iterable[Candidate],
    *,
    delta: float,
    batch_size: int,
    operator_of: dict[str, str] | None = None,
    max_slots_per_operator: int | None = None,
) -> list[Candidate]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if max_slots_per_operator is not None and max_slots_per_operator <= 0:
        raise ValueError("operator slot cap must be positive")
    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if difficulty_value(candidate, delta) > 0.0
        ),
        key=lambda candidate: (
            -difficulty_value(candidate, delta),
            candidate.drand_round,
            _canonical_key(candidate),
        ),
    )
    cap_applies = _operator_cap_applies(
        ranked,
        operator_of=operator_of,
        max_slots_per_operator=max_slots_per_operator,
    )
    selected: list[Candidate] = []
    claimed_prompts: set[int] = set()
    operator_slots: Counter[str] = Counter()
    for candidate in ranked:
        if len(selected) >= batch_size:
            break
        if candidate.prompt_idx in claimed_prompts:
            continue
        operator = _candidate_operator(candidate, operator_of)
        if (
            cap_applies
            and operator is not None
            and operator_slots[operator] >= max_slots_per_operator
        ):
            continue
        selected.append(candidate)
        claimed_prompts.add(candidate.prompt_idx)
        if operator is not None:
            operator_slots[operator] += 1
    return selected


def _operator_cap_applies(
    eligible: Iterable[Candidate],
    *,
    operator_of: dict[str, str] | None,
    max_slots_per_operator: int | None,
) -> bool:
    materialized = list(eligible)
    return (
        max_slots_per_operator is not None
        and max_slots_per_operator > 0
        and operator_of is not None
        and bool(materialized)
        and all(
            _candidate_operator(candidate, operator_of) is not None
            for candidate in materialized
        )
    )


def _operator_id(
    operator_of: dict[str, str] | None,
    hotkey: str,
) -> str | None:
    if operator_of is None:
        return None
    return _normalized_identity(operator_of.get(hotkey))


def _candidate_operator(
    candidate: Candidate,
    operator_of: dict[str, str] | None,
) -> str | None:
    """Resolve ownership at the window, with current-chain data as fallback."""
    return candidate.operator_id or _operator_id(operator_of, candidate.hotkey)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _share_summary(identities: Iterable[str]) -> dict[str, float | int | None]:
    counts = Counter(identities)
    total = sum(counts.values())
    ordered = sorted(counts.values(), reverse=True)
    return {
        "distinct": len(counts),
        "top1_share": ordered[0] / total if total else None,
        "top5_share": sum(ordered[:5]) / total if total else None,
    }


def _reward_histogram(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts = Counter(
        f"{candidate.mean_reward:.3f}" for candidate in candidates
    )
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _candidate_arrival_age(candidate: Candidate) -> tuple[float | None, str]:
    if candidate.arrival_age_seconds is not None:
        return candidate.arrival_age_seconds, "http_arrival"
    if candidate.response_time is not None:
        return candidate.response_time, "acceptance_proxy"
    return None, "missing"


def _summarize_deadlines(
    candidate_counts_by_deadline: dict[float, list[int]],
    distinct_counts_by_deadline: dict[float, list[int]],
    *,
    batch_size: int,
) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for deadline, candidate_counts in candidate_counts_by_deadline.items():
        distinct_counts = distinct_counts_by_deadline[deadline]
        filled = sum(count >= batch_size for count in distinct_counts)
        summary[f"{deadline:g}"] = {
            "windows": len(candidate_counts),
            "mean_validated_candidates_by_deadline": _mean(candidate_counts),
            "median_validated_candidates_by_deadline": _median(candidate_counts),
            "mean_distinct_prompts_by_deadline": _mean(distinct_counts),
            "windows_with_at_least_batch_size_distinct": filled,
            "fraction_with_at_least_batch_size_distinct": (
                filled / len(distinct_counts) if distinct_counts else None
            ),
        }
    return summary


def replay_archives(
    archives: Iterable[dict[str, Any]],
    *,
    environment: str,
    deltas: Iterable[float],
    batch_size: int,
    operator_of: dict[str, str] | None = None,
    max_slots_per_operator: int | None = None,
    deadlines: Iterable[float] = DEFAULT_DEADLINES,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if max_slots_per_operator is not None and max_slots_per_operator <= 0:
        raise ValueError("operator slot cap must be positive")
    windows = []
    missing_scores = 0
    batch_filled_rejects = 0
    total_archives = 0
    total_candidates = 0
    production_selected_all: list[Candidate] = []
    all_candidates: list[Candidate] = []
    exact_arrival_count = 0
    proxy_arrival_count = 0
    missing_arrival_count = 0
    deadline_values = tuple(float(deadline) for deadline in deadlines)
    if any(
        not math.isfinite(deadline) or deadline < 0.0
        for deadline in deadline_values
    ):
        raise ValueError("deadlines must be finite and non-negative")
    deadline_candidate_counts: dict[float, list[int]] = {
        deadline: [] for deadline in deadline_values
    }
    deadline_distinct_counts: dict[float, list[int]] = {
        deadline: [] for deadline in deadline_values
    }
    exact_deadline_candidate_counts: dict[float, list[int]] = {
        deadline: [] for deadline in deadline_values
    }
    exact_deadline_distinct_counts: dict[float, list[int]] = {
        deadline: [] for deadline in deadline_values
    }
    per_delta: dict[str, dict[str, Any]] = {}
    delta_values = tuple(float(delta) for delta in deltas)
    if not delta_values or any(
        not math.isfinite(delta) or delta < 0.0 for delta in delta_values
    ):
        raise ValueError("deltas must be non-negative finite numbers")
    selected_by_delta: dict[float, list[Candidate]] = {
        delta: [] for delta in delta_values
    }
    overlap_by_delta: dict[float, list[float]] = {
        delta: [] for delta in delta_values
    }
    cap_status_by_delta: dict[float, list[str]] = {
        delta: [] for delta in delta_values
    }

    for archive in archives:
        if not isinstance(archive, dict):
            continue
        total_archives += 1
        candidates, missing = extract_window_candidates(
            archive, environment=environment
        )
        missing_scores += missing
        reject_summary = archive.get("reject_summary", {})
        if isinstance(reject_summary, dict):
            batch_filled = _non_negative_int(
                reject_summary.get("batch_filled", 0)
            )
            if batch_filled is not None:
                batch_filled_rejects += batch_filled
        if not candidates:
            continue
        total_candidates += len(candidates)
        all_candidates.extend(candidates)
        arrival_rows = []
        for candidate in candidates:
            age, basis = _candidate_arrival_age(candidate)
            arrival_rows.append((candidate, age, basis))
            if basis == "http_arrival":
                exact_arrival_count += 1
            elif basis == "acceptance_proxy":
                proxy_arrival_count += 1
            else:
                missing_arrival_count += 1
        exact_http_window = all(
            basis == "http_arrival" for _candidate, _age, basis in arrival_rows
        )
        production = [
            candidate for candidate in candidates if candidate.production_selected
        ]
        production_selected_all.extend(production)
        window_record = {
            "window_start": archive.get("window_start"),
            "candidate_count": len(candidates),
            "production_selected_count": len(production),
            "distinct_prompts": len({candidate.prompt_idx for candidate in candidates}),
            "exact_http_arrival_complete": exact_http_window,
        }
        for deadline in deadline_values:
            by_deadline = [
                candidate
                for candidate, age, _basis in arrival_rows
                if age is not None and age <= deadline
            ]
            deadline_candidate_counts[deadline].append(len(by_deadline))
            deadline_distinct_counts[deadline].append(
                len({candidate.prompt_idx for candidate in by_deadline})
            )
            if exact_http_window:
                exact_deadline_candidate_counts[deadline].append(
                    len(by_deadline)
                )
                exact_deadline_distinct_counts[deadline].append(
                    len({candidate.prompt_idx for candidate in by_deadline})
                )
        production_keys = {
            (candidate.hotkey, candidate.prompt_idx, candidate.selection_digest)
            for candidate in production
        }
        for delta in delta_values:
            eligible = [
                candidate
                for candidate in candidates
                if difficulty_value(candidate, delta) > 0.0
            ]
            if max_slots_per_operator is None:
                cap_status = "not_requested"
            elif not eligible:
                cap_status = "not_applicable"
            elif _operator_cap_applies(
                eligible,
                operator_of=operator_of,
                max_slots_per_operator=max_slots_per_operator,
            ):
                cap_status = "applied"
            else:
                cap_status = "incomplete_mapping"
            cap_status_by_delta[delta].append(cap_status)
            selected = select_candidates(
                candidates,
                delta=delta,
                batch_size=batch_size,
                operator_of=operator_of,
                max_slots_per_operator=max_slots_per_operator,
            )
            selected_by_delta[delta].extend(selected)
            shadow_keys = {
                (candidate.hotkey, candidate.prompt_idx, candidate.selection_digest)
                for candidate in selected
            }
            union = production_keys | shadow_keys
            jaccard = (
                len(production_keys & shadow_keys) / len(union)
                if union else 1.0
            )
            overlap_by_delta[delta].append(jaccard)
            window_record[f"delta_{delta:g}"] = {
                "selected_count": len(selected),
                "selection_jaccard": jaccard,
                "mean_reward": _mean(
                    candidate.mean_reward for candidate in selected
                ),
            }
        windows.append(window_record)

    candidate_hotkeys = {candidate.hotkey for candidate in all_candidates}
    eligible_hotkeys = {
        candidate.hotkey
        for candidate in all_candidates
        if candidate.reward_std > 0.0
    }
    mapped_hotkeys = {
        candidate.hotkey
        for candidate in all_candidates
        if _candidate_operator(candidate, operator_of) is not None
    }
    mapped_eligible_hotkeys = {
        candidate.hotkey
        for candidate in all_candidates
        if candidate.reward_std > 0.0
        and _candidate_operator(candidate, operator_of) is not None
    }
    eligible_candidates = [
        candidate for candidate in all_candidates if candidate.reward_std > 0.0
    ]
    operator_mapping_complete = (
        bool(eligible_candidates)
        and all(
            _candidate_operator(candidate, operator_of) is not None
            for candidate in eligible_candidates
        )
    )
    archived_operator_candidates = sum(
        candidate.operator_id is not None for candidate in all_candidates
    )
    fallback_operator_candidates = sum(
        candidate.operator_id is None
        and _operator_id(operator_of, candidate.hotkey) is not None
        for candidate in all_candidates
    )
    operator_conflicts = sum(
        candidate.operator_id is not None
        and (external := _operator_id(operator_of, candidate.hotkey)) is not None
        and external != candidate.operator_id
        for candidate in all_candidates
    )

    production_summary = {
        "selected_count": len(production_selected_all),
        "mean_reward": _mean(
            candidate.mean_reward for candidate in production_selected_all
        ),
        "mean_std": _mean(
            candidate.reward_std for candidate in production_selected_all
        ),
        "hotkey_concentration": _share_summary(
            candidate.hotkey for candidate in production_selected_all
        ),
        "mean_reward_histogram": _reward_histogram(production_selected_all),
    }
    if any(
        _candidate_operator(candidate, operator_of) is not None
        for candidate in production_selected_all
    ):
        production_summary["operator_concentration"] = _share_summary(
            operator
            for candidate in production_selected_all
            if (operator := _candidate_operator(candidate, operator_of))
            is not None
        )
    for delta in delta_values:
        selected = selected_by_delta[delta]
        cap_statuses = cap_status_by_delta[delta]
        cap_applied_windows = cap_statuses.count("applied")
        cap_skipped_windows = cap_statuses.count("incomplete_mapping")
        cap_not_applicable_windows = cap_statuses.count("not_applicable")
        summary = {
            "selected_count": len(selected),
            "mean_reward": _mean(candidate.mean_reward for candidate in selected),
            "mean_std": _mean(candidate.reward_std for candidate in selected),
            "mean_selection_jaccard": _mean(overlap_by_delta[delta]),
            "hotkey_concentration": _share_summary(
                candidate.hotkey for candidate in selected
            ),
            "mean_reward_histogram": _reward_histogram(selected),
            "operator_cap_requested": max_slots_per_operator,
            "operator_cap_applied": (
                max_slots_per_operator is not None
                and cap_applied_windows > 0
                and cap_skipped_windows == 0
            ),
            "operator_cap_applied_windows": cap_applied_windows,
            "operator_cap_skipped_windows": cap_skipped_windows,
            "operator_cap_not_applicable_windows": (
                cap_not_applicable_windows
            ),
            "operator_mapping_complete": operator_mapping_complete,
        }
        if any(
            _candidate_operator(candidate, operator_of) is not None
            for candidate in selected
        ):
            summary["operator_concentration"] = _share_summary(
                operator
                for candidate in selected
                if (operator := _candidate_operator(candidate, operator_of))
                is not None
            )
        per_delta[f"{delta:g}"] = summary

    deadline_summary = _summarize_deadlines(
        deadline_candidate_counts,
        deadline_distinct_counts,
        batch_size=batch_size,
    )
    exact_deadline_summary = _summarize_deadlines(
        exact_deadline_candidate_counts,
        exact_deadline_distinct_counts,
        batch_size=batch_size,
    )

    return {
        "schema_version": 1,
        "environment": environment,
        "population": "archived_fully_validated_batch_plus_runners_up",
        "population_limitations": [
            "excludes_pre_validation_rejects",
            "excludes_batch_filled_payloads",
            "historical_runners_use_binary_reward_vector_when_rollouts_absent",
        ],
        "archive_count": total_archives,
        "windows_with_candidates": len(windows),
        "candidate_count": total_candidates,
        "candidate_entries_missing_reward_data": missing_scores,
        "observed_batch_filled_reject_count": batch_filled_rejects,
        "arrival_timing_coverage": {
            "http_arrival": exact_arrival_count,
            "acceptance_proxy": proxy_arrival_count,
            "missing": missing_arrival_count,
            "warning": (
                "historical response_time includes validator processing and "
                "is only an upper-bound proxy for HTTP arrival"
            ),
        },
        "deadline_counterfactual_basis": "best_available_per_candidate",
        "deadline_counterfactual": deadline_summary,
        "deadline_counterfactual_exact_http_arrival": exact_deadline_summary,
        "operator_mapping": {
            "candidate_hotkeys": len(candidate_hotkeys),
            "mapped_candidate_hotkeys": len(mapped_hotkeys),
            "eligible_hotkeys": len(eligible_hotkeys),
            "mapped_eligible_hotkeys": len(mapped_eligible_hotkeys),
            "complete_for_cap": operator_mapping_complete,
            "archived_operator_candidates": archived_operator_candidates,
            "fallback_operator_candidates": fallback_operator_candidates,
            "archived_external_conflicts": operator_conflicts,
            "resolution_order": "archived_window_snapshot_then_external_fallback",
        },
        "production": production_summary,
        "counterfactual_by_delta": per_delta,
        "windows": windows,
    }


def _load_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_paths(paths: Iterable[str]) -> list[dict[str, Any]]:
    archives = []
    for raw_path in paths:
        path = Path(raw_path)
        matches = sorted(path.glob("window-*.json*")) if path.is_dir() else [path]
        archives.extend(_load_file(match) for match in matches)
    return archives


async def _load_r2(current_window: int, n: int) -> list[dict[str, Any]]:
    from reliquary.infrastructure.storage import list_recent_datasets

    return await list_recent_datasets(current_window=current_window, n=n)


async def _load_chain_operator_map(netuid: int) -> dict[str, str]:
    from reliquary.infrastructure import chain

    subtensor = None
    try:
        subtensor = await chain.get_subtensor()
        metagraph = await chain.get_metagraph(subtensor, netuid)
        hotkeys = list(getattr(metagraph, "hotkeys", []))
        coldkeys = list(getattr(metagraph, "coldkeys", []))
        if len(hotkeys) != len(coldkeys):
            raise RuntimeError("metagraph hotkey/coldkey lengths differ")
        operator_map = {}
        for hotkey, coldkey in zip(hotkeys, coldkeys):
            if hotkey is None or coldkey is None:
                continue
            normalized_hotkey = str(hotkey).strip()
            normalized_coldkey = str(coldkey).strip()
            if normalized_hotkey and normalized_coldkey:
                operator_map[normalized_hotkey] = normalized_coldkey
        return operator_map
    finally:
        await chain.close_subtensor(subtensor)


def _parse_deltas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("deltas must be non-negative finite numbers")
    return values


def _positive_cli_int(raw: str) -> int:
    value = _positive_int(raw)
    if value is None:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _non_negative_cli_int(raw: str) -> int:
    value = _non_negative_int(raw)
    if value is None:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        )
    return value


def _load_operator_map(path: str | None) -> dict[str, str] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("operator map must be a JSON object: hotkey -> coldkey")
    operator_map = {}
    for hotkey, coldkey in payload.items():
        if hotkey is None or coldkey is None:
            continue
        normalized_hotkey = str(hotkey).strip()
        normalized_coldkey = str(coldkey).strip()
        if normalized_hotkey and normalized_coldkey:
            operator_map[normalized_hotkey] = normalized_coldkey
    return operator_map


def _print_human(report: dict[str, Any]) -> None:
    print("Difficulty auction replay")
    print(f"environment: {report['environment']}")
    print(
        "coverage: "
        f"{report['windows_with_candidates']}/{report['archive_count']} windows, "
        f"{report['candidate_count']} validated candidates, "
        f"{report['candidate_entries_missing_reward_data']} missing scores"
    )
    print(
        "unobserved pressure: "
        f"{report['observed_batch_filled_reject_count']} archived batch_filled rejects "
        "whose payload difficulty is unavailable"
    )
    timing = report["arrival_timing_coverage"]
    print(
        "arrival timing: "
        f"exact={timing['http_arrival']} proxy={timing['acceptance_proxy']} "
        f"missing={timing['missing']}"
    )
    operator_mapping = report["operator_mapping"]
    print(
        "operator mapping: "
        f"eligible={operator_mapping['eligible_hotkeys']} "
        f"mapped={operator_mapping['mapped_eligible_hotkeys']} "
        f"complete_for_cap={operator_mapping['complete_for_cap']} "
        f"archived={operator_mapping['archived_operator_candidates']} "
        f"fallback={operator_mapping['fallback_operator_candidates']} "
        f"conflicts={operator_mapping['archived_external_conflicts']}"
    )
    for deadline, summary in report["deadline_counterfactual"].items():
        print(
            f"deadline_best_available={deadline}s: mean_distinct="
            f"{summary['mean_distinct_prompts_by_deadline']} "
            f"fill_rate={summary['fraction_with_at_least_batch_size_distinct']}"
        )
    for deadline, summary in report[
        "deadline_counterfactual_exact_http_arrival"
    ].items():
        if summary["windows"] == 0:
            continue
        print(
            f"deadline_exact={deadline}s: windows={summary['windows']} "
            f"mean_distinct={summary['mean_distinct_prompts_by_deadline']} "
            f"fill_rate={summary['fraction_with_at_least_batch_size_distinct']}"
        )
    production = report["production"]
    production_operator_top1 = (
        production.get("operator_concentration", {}).get("top1_share")
    )
    print(
        "production: "
        f"n={production['selected_count']} "
        f"mean_reward={production['mean_reward']} "
        f"top1={production['hotkey_concentration']['top1_share']} "
        f"operator_top1={production_operator_top1}"
    )
    for delta, summary in report["counterfactual_by_delta"].items():
        operator_top1 = (
            summary.get("operator_concentration", {}).get("top1_share")
        )
        if summary["operator_cap_requested"] is None:
            cap_windows = "not_requested"
        else:
            applied = summary["operator_cap_applied_windows"]
            total = (
                applied
                + summary["operator_cap_skipped_windows"]
                + summary["operator_cap_not_applicable_windows"]
            )
            cap_windows = f"{applied}/{total}"
        print(
            f"delta={delta}: n={summary['selected_count']} "
            f"mean_reward={summary['mean_reward']} "
            f"mean_jaccard={summary['mean_selection_jaccard']} "
            f"top1={summary['hotkey_concentration']['top1_share']} "
            f"operator_top1={operator_top1} "
            f"operator_cap_applied={summary['operator_cap_applied']} "
            f"cap_windows={cap_windows}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="window JSON/GZ files or directories")
    parser.add_argument("--from-r2", action="store_true")
    parser.add_argument("--current-window", type=_non_negative_cli_int)
    parser.add_argument("--n", type=_positive_cli_int, default=250)
    parser.add_argument("--environment", default="openmathinstruct")
    parser.add_argument(
        "--deltas",
        type=_parse_deltas,
        default=DEFAULT_DELTAS,
        help="comma-separated values (default: 0,0.5,1,1.5,2)",
    )
    parser.add_argument("--batch-size", type=_positive_cli_int, default=8)
    parser.add_argument("--operator-map")
    parser.add_argument("--operator-map-from-chain", action="store_true")
    parser.add_argument("--netuid", type=_non_negative_cli_int, default=81)
    parser.add_argument(
        "--max-slots-per-operator", type=_positive_cli_int
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.from_r2:
        if args.current_window is None:
            parser.error("--from-r2 requires --current-window")
        archives = asyncio.run(_load_r2(args.current_window, args.n))
    else:
        if not args.paths:
            parser.error("provide archive paths or --from-r2")
        archives = _load_paths(args.paths)

    if args.operator_map and args.operator_map_from_chain:
        parser.error("choose --operator-map or --operator-map-from-chain")
    operator_map = _load_operator_map(args.operator_map)
    if args.operator_map_from_chain:
        operator_map = asyncio.run(_load_chain_operator_map(args.netuid))

    report = replay_archives(
        archives,
        environment=args.environment,
        deltas=args.deltas,
        batch_size=args.batch_size,
        operator_of=operator_map,
        max_slots_per_operator=args.max_slots_per_operator,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
