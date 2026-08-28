#!/usr/bin/env python3
"""Deterministic synthetic capacity replay for checkpoint-epoch review.

This is a protocol-shape comparison, not production telemetry or an economic
forecast. It makes its candidate-supply assumptions explicit in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from reliquary.shared.checkpoint_epoch import (
    CHECKPOINT_EPOCH_ADMISSION_POLICY,
    CHECKPOINT_EPOCH_FINALIZATION_POLICY,
    CHECKPOINT_EPOCH_RANKING_POLICY,
    CHECKPOINT_EPOCH_REWARD_POLICY,
    EpochAdmissionCommitment,
    select_epoch_reveals,
)
from reliquary.protocol.profiles import resolve_protocol_profile
from reliquary.validator.batch_selection import throughput_rank


_CURRENT_PROFILE = resolve_protocol_profile(
    "qwen3-4b-base-dapo-reasoning-v5"
)
_ROLLOUTS = _CURRENT_PROFILE.sampling.rollouts
_THROUGHPUT = _CURRENT_PROFILE.throughput_tiebreak


@dataclass(frozen=True)
class Candidate:
    environment: str
    lane: int
    operator: int
    prompt: int
    difficulty: float
    tokens: int
    gpu_seconds: float
    prepared_at_open: bool
    valid: bool
    stale: bool


def _gini(values: Iterable[int]) -> float:
    ordered = sorted(value for value in values if value >= 0)
    total = sum(ordered)
    if not ordered or total == 0:
        return 0.0
    n = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def _tiebreak(seed: int, candidate: Candidate) -> bytes:
    value = (
        f"{seed}:{candidate.environment}:{candidate.lane}:"
        f"{candidate.operator}:{candidate.prompt}"
    )
    return hashlib.sha256(value.encode("utf-8")).digest()


def _population(
    *,
    rng: random.Random,
    horizon: int,
    candidate_limit: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for environment in ("math", "code"):
        validity = 0.86 if environment == "math" else 0.78
        for lane in range(horizon):
            for index in range(candidate_limit):
                gpu_seconds = rng.lognormvariate(math.log(45.0), 0.35)
                candidates.append(Candidate(
                    environment=environment,
                    lane=lane,
                    operator=rng.randrange(12),
                    prompt=lane * 10_000 + index,
                    difficulty=round(rng.random(), 1),
                    tokens=rng.randint(2_000, 8_000) * _ROLLOUTS,
                    gpu_seconds=gpu_seconds,
                    prepared_at_open=rng.random() < 0.75,
                    valid=rng.random() < validity,
                    stale=rng.random() < 0.015,
                ))
    return candidates


def _run_policy(
    *,
    name: str,
    seed: int,
    population: list[Candidate],
    horizon: int,
    target: int,
    candidate_supply: int,
    reveal_limit: int,
    epoch_mode: bool,
) -> dict[str, object]:
    policy_population: list[Candidate] = []
    selected: list[Candidate] = []
    underfill_by_environment = Counter()
    first_lane_underfill = 0
    for environment in ("math", "code"):
        for lane in range(horizon):
            lane_candidates = [
                candidate
                for candidate in population
                if candidate.environment == environment
                and candidate.lane == lane
            ][:candidate_supply]
            policy_population.extend(lane_candidates)
            if epoch_mode:
                commitments = [
                    EpochAdmissionCommitment(
                        commitment_id=(
                            f"{candidate.environment}:{candidate.lane}:"
                            f"{candidate.operator}:{candidate.prompt}"
                        ),
                        operator_id=str(candidate.operator),
                        window_number=candidate.lane,
                        environment=candidate.environment,
                        prompt_idx=candidate.prompt,
                        payload_sha256=hashlib.sha256(
                            repr(candidate).encode("utf-8")
                        ).hexdigest(),
                    )
                    for candidate in lane_candidates
                ]
                selected_ids = set(select_epoch_reveals(
                    commitments,
                    admission_randomness=f"{seed:064x}",
                    epoch_id=hashlib.sha256(b"synthetic-epoch").hexdigest(),
                    manifest_sha256_hex=hashlib.sha256(
                        b"synthetic-manifest"
                    ).hexdigest(),
                    commitment_set_sha256_hex=hashlib.sha256(
                        b"synthetic-commitment-set"
                    ).hexdigest(),
                    limit=reveal_limit,
                    per_prompt_limit=10,
                ))
                lane_candidates = [
                    candidate
                    for candidate, commitment in zip(lane_candidates, commitments)
                    if commitment.commitment_id in selected_ids
                ]
                lane_candidates.sort(key=lambda candidate: (
                    -candidate.difficulty,
                    _tiebreak(seed, candidate),
                ))
            else:
                throughput = _THROUGHPUT
                if throughput is None:
                    raise RuntimeError("active profile has no throughput tie-break")
                lane_candidates.sort(key=lambda candidate: (
                    -candidate.difficulty,
                    throughput_rank(
                        candidate.tokens,
                        arrival_round=int(candidate.gpu_seconds // 3.0),
                        window_open_round=0,
                        token_cap=throughput.token_cap * _ROLLOUTS,
                        bucket_tokens_per_round=throughput.bucket_tokens_per_round,
                    ),
                    int(candidate.gpu_seconds // 3.0),
                    _tiebreak(seed, candidate),
                ))
            lane_candidates = [
                candidate
                for candidate in lane_candidates
                if candidate.valid and not candidate.stale
            ]
            winners = lane_candidates[:target]
            selected.extend(winners)
            missing = target - len(winners)
            underfill_by_environment[environment] += missing
            if lane == 0:
                first_lane_underfill += missing

    generated_seconds = sum(candidate.gpu_seconds for candidate in policy_population)
    selected_tokens = sum(candidate.tokens for candidate in selected)
    operator_counts = Counter(candidate.operator for candidate in selected)
    shares = [count / max(len(selected), 1) for count in operator_counts.values()]
    return {
        "policy": name,
        "candidate_supply_per_environment_lane": candidate_supply,
        "selected_reveal_limit_per_environment_lane": reveal_limit,
        "generated_candidates": len(policy_population),
        "valid_candidates_available": sum(
            candidate.valid and not candidate.stale for candidate in policy_population
        ),
        "selected_training_groups": len(selected),
        "underfill_by_environment": dict(sorted(underfill_by_environment.items())),
        "burned_slot_share": round(
            sum(underfill_by_environment.values())
            / (2 * horizon * target),
            6,
        ),
        "generated_gpu_seconds": round(generated_seconds, 3),
        "gpu_seconds_per_selected_group": round(
            generated_seconds / max(len(selected), 1),
            3,
        ),
        "accepted_training_tokens_per_gpu_hour": round(
            selected_tokens / max(generated_seconds / 3_600.0, 1e-9),
            3,
        ),
        "first_lane_warmup_loss_groups": first_lane_underfill,
        "selected_operator_hhi": round(sum(share * share for share in shares), 6),
        "selected_operator_gini": round(_gini(operator_counts.values()), 6),
        "selected_mean_difficulty": round(
            sum(candidate.difficulty for candidate in selected)
            / max(len(selected), 1),
            6,
        ),
        "selected_distinct_prompt_share": round(
            len({
                (candidate.environment, candidate.prompt)
                for candidate in selected
            })
            / max(len(selected), 1),
            6,
        ),
        "open_edge_submission_burst": sum(
            epoch_mode and candidate.prepared_at_open
            for candidate in policy_population
        ),
        "open_edge_payload_burst": 0,
        "post_selection_payload_burst_upper_bound": (
            2 * horizon * reveal_limit if epoch_mode else 0
        ),
        "stale_discarded_work": sum(
            candidate.stale for candidate in policy_population
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19_871)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--target", type=int, default=16)
    parser.add_argument("--current-candidates", type=int, default=64)
    parser.add_argument("--epoch-candidates", type=int, default=32)
    parser.add_argument("--epoch-commitments", type=int, default=64)
    args = parser.parse_args()
    if min(
        args.horizon,
        args.target,
        args.current_candidates,
        args.epoch_candidates,
        args.epoch_commitments,
    ) < 1:
        parser.error("all sizing arguments must be positive")
    if min(args.current_candidates, args.epoch_candidates) < args.target:
        parser.error("candidate limits must be at least the target")
    if args.epoch_commitments < args.epoch_candidates:
        parser.error("epoch commitments must cover the reveal cohort")

    population = _population(
        rng=random.Random(args.seed),
        horizon=args.horizon,
        candidate_limit=max(args.current_candidates, args.epoch_commitments),
    )
    report = {
        "scope": "synthetic capacity-envelope replay; not production telemetry",
        "assumptions": {
            "environments": ["math", "code"],
            "operators": 12,
            "epoch_prepared_at_open_probability": 0.75,
            "epoch_commitments_are_selected_before_payload_reveal": True,
            "epoch_admission_policy": CHECKPOINT_EPOCH_ADMISSION_POLICY,
            "epoch_ranking_policy": CHECKPOINT_EPOCH_RANKING_POLICY,
            "epoch_reward_policy": CHECKPOINT_EPOCH_REWARD_POLICY,
            "epoch_finalization_policy": CHECKPOINT_EPOCH_FINALIZATION_POLICY,
            "reward_or_profit_claim": False,
        },
        "seed": args.seed,
        "current": _run_policy(
            name="current_per_window",
            seed=args.seed,
            population=population,
            horizon=args.horizon,
            target=args.target,
            candidate_supply=args.current_candidates,
            reveal_limit=args.current_candidates,
            epoch_mode=False,
        ),
        "checkpoint_epoch": _run_policy(
            name="checkpoint_epoch",
            seed=args.seed,
            population=population,
            horizon=args.horizon,
            target=args.target,
            candidate_supply=args.epoch_commitments,
            reveal_limit=args.epoch_candidates,
            epoch_mode=True,
        ),
        "unmeasurable_without_authenticated_telemetry": [
            "real generation cost",
            "real profitability",
            "real underfill",
            "real operator concentration",
            "real selection bias",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
