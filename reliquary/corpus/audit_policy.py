"""Who is audited, when, and what a confirmed failure costs. Pure: no I/O, no
clock; the caller passes `now` and the drand randomness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib

AUDIT_PARAM_KEYS = {
    "audit_q": "q", "audit_probation_submissions": "probation_submissions",
    "audit_hold_seconds": "hold_seconds", "audit_suspect_seconds": "suspect_seconds",
    "audit_ban_after_failures": "ban_after_failures",
    "audit_ban_window_seconds": "ban_window_seconds", "audit_ban_seconds": "ban_seconds",
}
_INTS = {"probation_submissions", "ban_after_failures"}
MANT_HISTORY = 200


@dataclass(frozen=True)
class AuditParams:
    q: float = 1.0
    probation_submissions: int = 100
    hold_seconds: float = 4320.0
    suspect_seconds: float = 86400.0
    ban_after_failures: int = 3
    ban_window_seconds: float = 86400.0
    ban_seconds: float = 604800.0

    @classmethod
    def from_params(cls, params: Mapping) -> "AuditParams":
        validate_audit_params(params)
        return cls(**{attr: params[key] for key, attr in AUDIT_PARAM_KEYS.items() if key in params})


def validate_audit_params(params: Mapping) -> None:
    for key, attr in AUDIT_PARAM_KEYS.items():
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number, got {value!r}")
        if attr in _INTS and (not float(value).is_integer() or value < 1):
            raise ValueError(f"{key} must be a whole number >= 1, got {value}")
        if attr == "q" and not 0.0 < value <= 1.0:
            raise ValueError(f"audit_q must be in (0, 1], got {value}")
        if attr.endswith("seconds") and value < 0:
            raise ValueError(f"{key} must not be negative, got {value}")


@dataclass
class MinerState:
    state: str = "probation"
    audited_passed: int = 0
    confirmed_failures: list = field(default_factory=list)
    suspect_until: float | None = None
    banned_until: float | None = None
    mant_mean_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "MinerState":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def effective_state(m: MinerState, now: float, params: AuditParams | None = None) -> str:
    probation = (params or AuditParams()).probation_submissions
    if m.banned_until is not None:
        return "banned" if now < m.banned_until else "probation"
    if m.suspect_until is not None and now < m.suspect_until:
        return "suspect"
    return "sampled" if m.audited_passed >= probation else "probation"


def drawn(randomness_hex: str, submission_id: str, q: float) -> bool:
    digest = hashlib.sha256(bytes.fromhex(randomness_hex) + bytes.fromhex(submission_id)).digest()
    return int.from_bytes(digest, "big") < int(q * 2**256)


def decision(m, *, params, now, received_at, recent_submissions, randomness_hex,
             submission_id: str = "") -> str:
    state = effective_state(m, now, params)
    if state == "banned":
        return "void_banned"
    if state in ("probation", "suspect") or params.q >= 1.0:
        return "audit"
    # The slow hotkey: below 1/q per hold a draw may never land before payment.
    if recent_submissions < 1.0 / params.q:
        return "audit"
    if randomness_hex is None:
        return "audit"
    if drawn(randomness_hex, submission_id, params.q):
        return "audit"
    return "pass_unaudited" if now >= received_at + params.hold_seconds else "wait"


def after_pass(m: MinerState, params: AuditParams, mant_mean: float) -> MinerState:
    history = (list(m.mant_mean_history) + [float(mant_mean)])[-MANT_HISTORY:]
    return replace(m, audited_passed=m.audited_passed + 1, mant_mean_history=history)


def after_confirmed_failure(m: MinerState, params: AuditParams, now: float) -> MinerState:
    failures = [t for t in m.confirmed_failures if now - t <= params.ban_window_seconds] + [now]
    if len(failures) >= params.ban_after_failures:
        return replace(m, confirmed_failures=failures, suspect_until=None,
                       banned_until=now + params.ban_seconds, audited_passed=0)
    return replace(m, confirmed_failures=failures, suspect_until=now + params.suspect_seconds)
