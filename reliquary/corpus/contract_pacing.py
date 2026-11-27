"""Rates for a contracted draw order. Inputs are measured at the order control."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def audit_rate(clean_audited_tokens: int | None, *, q_min: float, trust_tokens: int,
               evaluation: bool = False, required: bool = False) -> float:
    """Confidence from verified audits of one coldkey. Missing history requires a full audit.

    The caller still runs audit_policy.decision for its compulsory checks (probation,
    suspect, sparse submissions and missing randomness). No card-load input can alter q.
    """
    if not 0.0 < q_min <= 1.0 or isinstance(q_min, bool):
        raise ValueError("q_min must be in (0, 1]")
    if type(trust_tokens) is not int or trust_tokens <= 0:
        raise ValueError("trust_tokens must be a positive number of tokens")
    if clean_audited_tokens is None or evaluation or required:
        return 1.0
    if type(clean_audited_tokens) is not int or clean_audited_tokens < 0:
        raise ValueError("clean_audited_tokens must be a nonnegative number of tokens")
    return q_min + (1.0 - q_min) * math.exp(-clean_audited_tokens / trust_tokens)


def decay_counts(delivered: float, committed: float, elapsed_seconds: float) -> tuple[float, float]:
    """Age the coldkey's existing totals before adding its most recent completed interval."""
    _nonnegative(delivered, "delivered")
    _nonnegative(committed, "committed")
    _nonnegative(elapsed_seconds, "elapsed_seconds")
    decay = 0.5 ** (elapsed_seconds / (7 * 86400))
    return delivered * decay, committed * decay


def reliability(delivered: float, committed: float) -> float:
    """Start at 0.8 on 10M tokens. Only work within an active commitment earns reliability."""
    _nonnegative(delivered, "delivered")
    _nonnegative(committed, "committed")
    return (min(delivered, committed) + 8_000_000) / (committed + 10_000_000)


@dataclass(frozen=True, slots=True)
class PaceState:
    sampled_at: float
    average_card_rate: float
    rate: float
    last_cut_at: float


def paced_rate(previous: PaceState | None, *, now: float, card_tokens_per_second: float,
               queued_audit_tokens: float, audit_fraction: float, offered_rate: float,
               needed_rate: float) -> PaceState:
    """Target generated tokens/s for the supplied share of the cards.

    Card rate and backlog are in audited tokens/s and audited tokens. For a shared
    card, pass the capacity and backlog allocated to this order. A backlog at 30
    minutes pauses grants immediately. Below the bound, use the 20-minute mean,
    correct toward 15 minutes of work over 15 minutes, and allow ordinary cuts
    of at most 10% in five minutes. Supply and the order's need are hard bounds.
    """
    for name, value in (("now", now), ("card_tokens_per_second", card_tokens_per_second),
                        ("queued_audit_tokens", queued_audit_tokens),
                        ("offered_rate", offered_rate), ("needed_rate", needed_rate)):
        _nonnegative(value, name)
    if isinstance(audit_fraction, bool) or not 0 < audit_fraction <= 1:
        raise ValueError("audit_fraction must be in (0, 1]")
    if previous is None:
        average, last_cut = card_tokens_per_second, now
    else:
        elapsed = _nonnegative(now - previous.sampled_at, "time since the last sample")
        weight = -math.expm1(-elapsed / 1200.0)
        average = previous.average_card_rate + weight * (card_tokens_per_second - previous.average_card_rate)
        last_cut = previous.last_cut_at

    if average <= 0 or queued_audit_tokens >= 1800.0 * average:
        return PaceState(now, average, 0.0, now)
    desired = max(0.0, average + (900.0 * average - queued_audit_tokens) / 900.0) / audit_fraction
    if previous is not None and previous.rate > 0:
        if abs(desired - previous.rate) <= 0.1 * previous.rate:
            desired = previous.rate
        elif desired < previous.rate:
            if now - last_cut < 300.0:
                desired = previous.rate
            else:
                desired = max(desired, 0.9 * previous.rate)
                last_cut = now
    return PaceState(now, average, min(desired, offered_rate, needed_rate), last_cut)


def tail_rescue_eligible(*, remaining_slots: int, total_slots: int, expired: bool, reliable: bool) -> bool:
    """Whether a second miner may race an expired holder for its one remaining result."""
    if type(remaining_slots) is not int or type(total_slots) is not int or not 0 <= remaining_slots <= total_slots:
        raise ValueError("remaining_slots must be between zero and total_slots")
    return bool(expired and reliable and remaining_slots > 0 and 50 * remaining_slots <= total_slots)


def contract_lease_seconds(job_seconds: float, *, expected_seconds: float | None = None,
                           episode_floor_seconds: float = 0.0) -> float:
    """Start at at most 15 minutes, or allow twice the observed completion time.

    A signed episode has its own minimum runtime in the session contract.
    """
    if _nonnegative(job_seconds, "job_seconds") == 0:
        raise ValueError("job_seconds must be positive")
    _nonnegative(episode_floor_seconds, "episode_floor_seconds")
    target = 900.0
    if expected_seconds is not None:
        target = min(target, max(60.0, 2 * _nonnegative(expected_seconds, "expected_seconds")))
    return max(episode_floor_seconds, min(target, job_seconds))
