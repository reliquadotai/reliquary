#!/usr/bin/env python3
"""Replay market-shape metrics from public validator window archives.

The operator-round counterfactual assumes every candidate not observed failing
proof would pass. It is a ranking-shape bound, not a production outcome claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


def _paths(inputs: Iterable[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            found.update(path.glob("*.json"))
            found.update(path.glob("*.json.gz"))
        elif path.is_file():
            found.add(path)
    return sorted(found)


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: archive must be an object")
    data = value.get("data", value)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: wrapped archive data must be an object")
    return data


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _shares(counts: Counter[str]) -> tuple[float | None, float | None, float]:
    total = sum(counts.values())
    if total <= 0:
        return None, None, 0.0
    shares = sorted((value / total for value in counts.values()), reverse=True)
    top1 = shares[0]
    hhi = sum(share * share for share in shares)
    weighted = sum(
        (index + 1) * value
        for index, value in enumerate(sorted(counts.values()))
    )
    n = len(counts)
    gini = (2 * weighted) / (n * total) - (n + 1) / n
    return top1, hhi, gini


def _candidate_digest(window: int, environment: str, row: dict[str, Any]) -> bytes:
    value = (
        f"{window}:{environment}:{row.get('operator_id', '')}:"
        f"{row.get('prompt_idx', '')}:{row.get('selection_digest', '')}"
    )
    return hashlib.sha256(value.encode("utf-8")).digest()


def _operator_digest(window: int, environment: str, operator: str) -> bytes:
    return hashlib.sha256(
        f"{window}:{environment}:{operator}".encode("utf-8")
    ).digest()


def operator_round_counterfactual(
    candidates: list[dict[str, Any]],
    *,
    window: int,
    environment: str,
    target: int,
) -> list[dict[str, Any]]:
    """Utility first, operator rounds inside exact ties, unique prompt/content."""
    eligible = [
        row for row in candidates
        if row.get("operator_id")
        and _finite(row.get("value")) is not None
        and _finite(row.get("value")) > 0.0
        and row.get("proof_passed") is not False
    ]
    by_value: dict[float, list[dict[str, Any]]] = {}
    for row in eligible:
        by_value.setdefault(float(row["value"]), []).append(row)
    ranked: list[dict[str, Any]] = []
    for value in sorted(by_value, reverse=True):
        by_operator: dict[str, list[dict[str, Any]]] = {}
        for row in by_value[value]:
            by_operator.setdefault(str(row["operator_id"]), []).append(row)
        for queue in by_operator.values():
            queue.sort(key=lambda row: _candidate_digest(window, environment, row))
        operators = sorted(
            by_operator,
            key=lambda operator: (
                _operator_digest(window, environment, operator),
                operator,
            ),
        )
        round_index = 0
        while True:
            added = False
            for operator in operators:
                queue = by_operator[operator]
                if round_index < len(queue):
                    ranked.append(queue[round_index])
                    added = True
            if not added:
                break
            round_index += 1

    selected: list[dict[str, Any]] = []
    prompts: set[int] = set()
    contents: set[str] = set()
    for row in ranked:
        prompt = int(row.get("prompt_idx", -1))
        content = str(row.get("prompt_content_sha256", ""))
        if prompt in prompts or (content and content in contents):
            continue
        selected.append(row)
        prompts.add(prompt)
        if content:
            contents.add(content)
        if len(selected) == target:
            break
    return selected


def _completion_tokens(row: dict[str, Any]) -> int:
    total = 0
    for rollout in row.get("rollouts") or []:
        length = _finite(rollout.get("completion_length"))
        if length is not None and length > 0:
            total += int(length)
    return total


def _positive_count(row: dict[str, Any]) -> int | None:
    vector = row.get("reward_vector")
    if isinstance(vector, str) and vector and set(vector) <= {"0", "1"}:
        return vector.count("1")
    rewards = [
        _finite(rollout.get("reward"))
        for rollout in row.get("rollouts") or []
    ]
    clean = [reward for reward in rewards if reward is not None]
    return sum(reward > 0.0 for reward in clean) if clean else None


def summarize(paths: Iterable[str | Path], *, target: int = 16) -> dict[str, Any]:
    archives = [_load(path) for path in _paths(paths)]
    environments: dict[str, dict[str, Any]] = {}
    names = sorted({
        str(name)
        for archive in archives
        for name in archive.get("environments", [])
    })
    for environment in names:
        selected_rows: list[dict[str, Any]] = []
        selected_offsets: list[float] = []
        production_ops: Counter[str] = Counter()
        token_ops: Counter[str] = Counter()
        counterfactual_ops: Counter[str] = Counter()
        k_counts: Counter[int] = Counter()
        counterfactual_jaccards: list[float] = []
        candidate_count = 0
        counterfactual_windows = 0
        for archive in archives:
            window = int(archive.get("window_start", -1))
            batch = [
                row for row in archive.get("batch", [])
                if str(row.get("env_name", "")) == environment
            ]
            selected_rows.extend(batch)
            opened = _finite(
                (archive.get("window_opened_wall_ts_by_environment") or {}).get(
                    environment
                )
            )
            for row in batch:
                operator = str(row.get("difficulty_auction_operator_id", ""))
                if operator:
                    production_ops[operator] += 1
                    token_ops[operator] += _completion_tokens(row)
                k = _positive_count(row)
                if k is not None:
                    k_counts[k] += 1
                arrival = _finite(row.get("precommit_arrival_ts"))
                if opened is not None and arrival is not None and arrival >= opened:
                    selected_offsets.append(arrival - opened)

            auction = (archive.get("difficulty_auction") or {}).get(
                environment, {}
            )
            candidates = list(auction.get("candidates") or [])
            candidate_count += len(candidates)
            counterfactual = operator_round_counterfactual(
                candidates,
                window=window,
                environment=environment,
                target=target,
            )
            if counterfactual:
                counterfactual_windows += 1
                for row in counterfactual:
                    counterfactual_ops[str(row["operator_id"])] += 1
                production_ids = {
                    str(row.get("selection_digest", ""))
                    for row in candidates if row.get("selected")
                }
                counterfactual_ids = {
                    str(row.get("selection_digest", ""))
                    for row in counterfactual
                }
                union = production_ids | counterfactual_ids
                counterfactual_jaccards.append(
                    len(production_ids & counterfactual_ids) / len(union)
                    if union else 1.0
                )

        lengths = [
            float(rollout["completion_length"])
            for row in selected_rows
            for rollout in row.get("rollouts") or []
            if _finite(rollout.get("completion_length")) is not None
        ]
        flat_top1, flat_hhi, flat_gini = _shares(production_ops)
        token_top1, token_hhi, token_gini = _shares(token_ops)
        cf_top1, cf_hhi, cf_gini = _shares(counterfactual_ops)
        total_tokens = sum(token_ops.values())
        environments[environment] = {
            "windows": len(archives),
            "candidate_rows": candidate_count,
            "selected_groups": len(selected_rows),
            "selected_completion_tokens": total_tokens,
            "completion_length_mean": (
                statistics.fmean(lengths) if lengths else None
            ),
            "completion_length_p50": _quantile(lengths, 0.50),
            "completion_length_p95": _quantile(lengths, 0.95),
            "selected_precommit_offset_p50_s": _quantile(
                selected_offsets, 0.50
            ),
            "selected_precommit_offset_p95_s": _quantile(
                selected_offsets, 0.95
            ),
            "positive_reward_count_distribution": dict(sorted(k_counts.items())),
            "flat_selected_slot_operator_concentration": {
                "operators": len(production_ops),
                "top1_share": flat_top1,
                "hhi": flat_hhi,
                "gini": flat_gini,
            },
            "gross_completion_token_operator_concentration": {
                "operators": len(token_ops),
                "top1_share": token_top1,
                "hhi": token_hhi,
                "gini": token_gini,
            },
            "operator_round_counterfactual_assuming_unfailed_proofs_pass": {
                "windows": counterfactual_windows,
                "operators": len(counterfactual_ops),
                "top1_share": cf_top1,
                "hhi": cf_hhi,
                "gini": cf_gini,
                "mean_jaccard_vs_archived_selection": (
                    statistics.fmean(counterfactual_jaccards)
                    if counterfactual_jaccards else None
                ),
            },
        }
    return {
        "schema_version": 1,
        "scope": "read-only public archive shape replay",
        "archives": len(archives),
        "environments": environments,
        "limitations": [
            "archives omit work rejected before durable candidate telemetry",
            "operator-round replay assumes candidates not observed failing proof pass",
            "completion tokens are not a direct training-value label",
            "no profitability conclusion is made without device cost telemetry",
        ],
        "production_activation_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="archive files or directories")
    parser.add_argument("--target", type=int, default=16)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("target must be positive")
    print(json.dumps(summarize(args.paths, target=args.target), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
