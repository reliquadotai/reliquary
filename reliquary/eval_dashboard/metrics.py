"""Evaluation aggregation and web-dashboard compatibility payloads."""

from __future__ import annotations

import math
from typing import Iterable

from reliquary.eval_dashboard.models import (
    CheckpointResult,
    DomainResult,
    TaskResult,
)


def wilson_interval(
    successes: float, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = min(1.0, max(0.0, successes / n))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def summarize_domain(domain: str, tasks: list[TaskResult]) -> DomainResult:
    if not tasks:
        raise ValueError("cannot summarize an empty domain")
    sample_count = len(tasks[0].samples)
    if sample_count <= 0 or any(len(task.samples) != sample_count for task in tasks):
        raise ValueError("every task must carry the same non-zero sample count")

    prompt_means = [
        sum(sample.reward for sample in task.samples) / sample_count for task in tasks
    ]
    first_samples = [task.samples[0].reward for task in tasks]
    prompt_bests = [max(sample.reward for sample in task.samples) for task in tasks]
    samples = [sample for task in tasks for sample in task.samples]
    # Keep the legacy dashboard contract: pass@1 is the fixed first draw,
    # pass@k is best-of-k, and pass_avg averages every sampled completion.
    # The fixed per-task seeds make pass@1 a stable paired comparison.
    pass_at_1 = sum(first_samples) / len(first_samples)
    pass_at_k = sum(prompt_bests) / len(prompt_bests)
    pass_avg = sum(prompt_means) / len(prompt_means)
    trunc_pct = 100.0 * sum(not sample.terminated for sample in samples) / len(samples)
    forced_pct = None
    if domain == "math":
        forced_pct = 100.0 * sum(sample.forced for sample in samples) / len(samples)

    return DomainResult(
        domain=domain,
        n_prompts=len(tasks),
        samples_per_prompt=sample_count,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        pass_avg=pass_avg,
        trunc_pct=trunc_pct,
        forced_pct=forced_pct,
        pass1_ci95=wilson_interval(pass_at_1 * len(tasks), len(tasks)),
        tasks=tasks,
    )


def _point(
    result: CheckpointResult,
    domain: str,
    base_checkpoint_n: int,
    publish_interval_windows: int,
) -> dict:
    value = result.domains[domain]
    point = {
        "checkpoint_n": result.checkpoint_n,
        "step": max(0, result.checkpoint_n - base_checkpoint_n)
        * publish_interval_windows,
        "pass@1": value.pass_at_1,
        "pass@k": value.pass_at_k,
        "pass_avg": value.pass_avg,
        "trunc_pct": value.trunc_pct,
        "n_prompts": value.n_prompts,
        "pass1_ci95": list(value.pass1_ci95),
        "model_revision": result.model_revision,
    }
    if value.forced_pct is not None:
        point["forced_pct"] = value.forced_pct
    return point


def _summary(
    points: list[dict], *, base_checkpoint_n: int, latest_checkpoint_n: int
) -> dict:
    expected = max(0, latest_checkpoint_n - base_checkpoint_n + 1)
    if not points:
        return {
            "has_data": False,
            "n_tested": 0,
            "n_expected": expected,
            "latest_publishable_checkpoint_n": latest_checkpoint_n,
            "coverage": 0.0,
        }
    base = next(
        (point for point in points if point["checkpoint_n"] == base_checkpoint_n), None
    )
    latest = points[-1]
    best = max(points, key=lambda point: point["pass@1"])
    base_pass1 = base["pass@1"] if base else None
    return {
        "has_data": True,
        "base_checkpoint_n": base_checkpoint_n,
        "base_pass1": base_pass1,
        "latest_checkpoint_n": latest["checkpoint_n"],
        "latest_pass1": latest["pass@1"],
        "latest_passk": latest["pass@k"],
        "latest_trunc_pct": latest["trunc_pct"],
        "delta_pass1": (
            latest["pass@1"] - base_pass1 if base_pass1 is not None else None
        ),
        "best_checkpoint_n": best["checkpoint_n"],
        "best_pass1": best["pass@1"],
        "latest_publishable_checkpoint_n": latest_checkpoint_n,
        "n_tested": len(points),
        "n_expected": expected,
        "coverage": len(points) / expected if expected else 0.0,
    }


def _trend(points: list[dict]) -> dict:
    n = len(points)
    if n < 3:
        return {"has_fit": False, "n_points": n}
    xs = [float(point["step"]) for point in points]
    ys = [float(point["pass@1"]) for point in points]
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0.0:
        return {"has_fit": False, "n_points": n}
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sxx
    intercept = y_bar - slope * x_bar
    fitted = [intercept + slope * x for x in xs]
    residual_ss = sum((y - fit) ** 2 for y, fit in zip(ys, fitted))
    total_ss = sum((y - y_bar) ** 2 for y in ys)
    df = n - 2
    stderr = math.sqrt((residual_ss / df) / sxx) if df > 0 else float("inf")
    ci = None
    t_value = 0.0
    if math.isfinite(stderr) and stderr > 0.0:
        t_value = slope / stderr
        ci = [(slope - 1.96 * stderr) * 100.0, (slope + 1.96 * stderr) * 100.0]
    elif stderr == 0.0:
        t_value = math.copysign(float("inf"), slope) if slope else 0.0
        ci = [slope * 100.0, slope * 100.0]

    slope_100 = slope * 100.0
    if ci is not None and ci[0] > 0.0:
        verdict = "significant improvement"
    elif ci is not None and ci[1] < 0.0:
        verdict = "significant regression"
    else:
        verdict = "flat (within noise)"
    return {
        "has_fit": True,
        "n_points": n,
        "intercept": intercept,
        "slope_per_step": slope,
        "slope_per_100steps": slope_100,
        "slope_per_100steps_ci95": ci,
        "r2": 1.0 - residual_ss / total_ss if total_ss > 0.0 else 0.0,
        "t": t_value if math.isfinite(t_value) else 1e308 * (1 if t_value > 0 else -1),
        "df": df,
        # Normal approximation. The manifest identifies this implementation;
        # dashboard decisions use the confidence interval, not this display value.
        "p_value": math.erfc(abs(t_value) / math.sqrt(2.0))
        if math.isfinite(t_value)
        else 0.0,
        "verdict": verdict,
    }


def _milestones(
    math_points: list[dict], code_points: list[dict], limit: int = 8
) -> list[dict]:
    by_checkpoint: dict[int, dict[str, float]] = {}
    steps: dict[int, int] = {}
    for domain, points in (("math", math_points), ("code", code_points)):
        for point in points:
            by_checkpoint.setdefault(point["checkpoint_n"], {})[domain] = point[
                "pass@1"
            ]
            steps[point["checkpoint_n"]] = point["step"]
    checkpoints = sorted(by_checkpoint)
    if len(checkpoints) > limit:
        selected = {
            checkpoints[round(i * (len(checkpoints) - 1) / (limit - 1))]
            for i in range(limit)
        }
        checkpoints = sorted(selected)
    if not checkpoints:
        return []
    return [
        {
            "checkpoint_n": checkpoint,
            "step": steps[checkpoint],
            "label": f"ckpt {checkpoint}",
            "math_pass1": by_checkpoint[checkpoint].get("math"),
            "code_pass1": by_checkpoint[checkpoint].get("code"),
            "is_base": checkpoint == checkpoints[0],
            "is_latest": checkpoint == checkpoints[-1],
        }
        for checkpoint in checkpoints
    ]


def build_dashboard(
    results: Iterable[CheckpointResult],
    *,
    generated_at: float,
    generated_at_iso: str,
    evidence_completed_at: float,
    evidence_completed_at_iso: str,
    config_hash: str,
    config_sha256: str,
    lineage_id: str,
    checkpoint_repo_id: str,
    base_checkpoint_n: int,
    latest_checkpoint_n: int,
    latest_checkpoint_revision: str,
    latest_model_repo_id: str,
    publish_interval_windows: int,
) -> dict:
    ordered = sorted(results, key=lambda result: result.checkpoint_n)
    math_points = [
        _point(result, "math", base_checkpoint_n, publish_interval_windows)
        for result in ordered
    ]
    code_points = [
        _point(result, "code", base_checkpoint_n, publish_interval_windows)
        for result in ordered
    ]
    latest_window = max((result.observed_window for result in ordered), default=0)
    expected = max(0, latest_checkpoint_n - base_checkpoint_n + 1)
    return {
        "schema_version": "2",
        "generated_at": generated_at,
        "generated_at_iso": generated_at_iso,
        "evidence_completed_at": evidence_completed_at,
        "evidence_completed_at_iso": evidence_completed_at_iso,
        "config_hash": config_hash,
        "config_sha256": config_sha256,
        "lineage_id": lineage_id,
        "checkpoint_repo_id": checkpoint_repo_id,
        "latest_model_repo_id": latest_model_repo_id,
        "latest_checkpoint_revision": latest_checkpoint_revision,
        "base_checkpoint_n": base_checkpoint_n,
        "publish_interval_windows": publish_interval_windows,
        "latest_window": latest_window,
        "latest_checkpoint_n": latest_checkpoint_n,
        "n_windows_cached": 0,
        "n_checkpoints": expected,
        "n_evaluated": len(ordered),
        "eval_summary": _summary(
            math_points,
            base_checkpoint_n=base_checkpoint_n,
            latest_checkpoint_n=latest_checkpoint_n,
        ),
        "eval_stats": _trend(math_points),
        "eval_summary_code": _summary(
            code_points,
            base_checkpoint_n=base_checkpoint_n,
            latest_checkpoint_n=latest_checkpoint_n,
        ),
        "eval_stats_code": _trend(code_points),
        "training": [],
        "eval": math_points,
        "eval_code": code_points,
        "milestones": _milestones(math_points, code_points),
    }
