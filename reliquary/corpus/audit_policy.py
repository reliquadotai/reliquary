"""Who is audited, when, and what a confirmed failure costs. Pure: no I/O, no
clock; the caller passes `now` and the drand randomness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import math

AUDIT_PARAM_KEYS = {
    "audit_q": "q", "audit_probation_submissions": "probation_submissions",
    "audit_hold_seconds": "hold_seconds", "audit_suspect_seconds": "suspect_seconds",
    "audit_ban_after_failures": "ban_after_failures",
    "audit_ban_window_seconds": "ban_window_seconds", "audit_ban_seconds": "ban_seconds",
}
_INTS = {"probation_submissions", "ban_after_failures"}
_POSITIVE_SECONDS = {"suspect_seconds", "ban_seconds", "ban_window_seconds"}
_HEX_DIGITS = set("0123456789abcdef")
MANT_HISTORY = 200
# Submissions whose confirmed failure is already counted; bounded, and far
# longer than one judging pass's retries need.
FAILURE_IDS = 256
# Submissions whose audited pass is already counted: a write that landed but
# whose response was lost is retried, and must not count them twice. The
# auditor writes at most this many passes per hotkey per write.
PASS_IDS = 32


@dataclass(frozen=True)
class AuditParams:
    q: float = 1.0
    probation_submissions: int = 100
    hold_seconds: float = 4320.0
    suspect_seconds: float = 86400.0
    ban_after_failures: int = 3
    ban_window_seconds: float = 604800.0
    ban_seconds: float = 604800.0

    @classmethod
    def from_params(cls, params: Mapping) -> "AuditParams":
        validate_audit_params(params)
        return cls(**{attr: params[key] for key, attr in AUDIT_PARAM_KEYS.items() if key in params})


_DEFAULTS = AuditParams()


def validate_audit_params(params: Mapping) -> None:
    for key, attr in AUDIT_PARAM_KEYS.items():
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number, got {value!r}")
        # json.loads happily parses NaN/Infinity; unchecked they disable a
        # hold/ban/window (NaN comparisons are always false) or end a ban
        # instantly (now < nan is false).
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value!r}")
        if attr in _INTS and (not float(value).is_integer() or value < 1):
            raise ValueError(f"{key} must be a whole number >= 1, got {value}")
        if attr == "q" and not 0.0 < value <= 1.0:
            raise ValueError(f"audit_q must be in (0, 1], got {value}")
        if attr.endswith("seconds") and value < 0:
            raise ValueError(f"{key} must not be negative, got {value}")
        # Zero makes suspect or a ban end the instant it starts, or a failure
        # leave the ban window before the next one is counted.
        if attr in _POSITIVE_SECONDS and value == 0:
            raise ValueError(f"{key} must be > 0, got {value}")
    # A zero hold with q < 1 pays an unaudited "sampled" submission the
    # instant it is not drawn, defeating the backward audit on a later
    # confirmed failure (§7.4); harmless at q = 1, which never takes that path.
    q = params.get("audit_q", _DEFAULTS.q)
    hold = params.get("audit_hold_seconds", _DEFAULTS.hold_seconds)
    if q < 1.0 and hold == 0:
        raise ValueError("audit_hold_seconds must not be 0 when audit_q < 1")


@dataclass
class MinerState:
    audited_passed: int = 0
    confirmed_failures: list = field(default_factory=list)
    suspect_until: float | None = None
    banned_until: float | None = None
    mant_mean_history: list = field(default_factory=list)
    failure_ids: list = field(default_factory=list)
    pass_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "MinerState":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def effective_state(m: MinerState, now: float, params: AuditParams) -> str:
    # The only authority on state; nothing persists a "state" string, so no
    # stale copy can disagree with what `now` and these fields say (§5).
    if m.banned_until is not None:
        return "banned" if now < m.banned_until else "probation"
    if m.suspect_until is not None and now < m.suspect_until:
        return "suspect"
    return "sampled" if m.audited_passed >= params.probation_submissions else "probation"


def _require_hex64(value: str, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX_DIGITS:
        raise ValueError(f"{label} must be exactly 64 lowercase hex characters, got {value!r}")
    return bytes.fromhex(value)


def drawn(randomness_hex: str, submission_id: str, q: float) -> bool:
    digest = hashlib.sha256(_require_hex64(randomness_hex, "randomness_hex") +
                            _require_hex64(submission_id, "submission_id")).digest()
    return int.from_bytes(digest, "big") < int(q * 2**256)


def decision(m, *, params, now, received_at, recent_submissions, randomness_hex,
             submission_id: str, slack_seconds: float = 0.0) -> str:
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
    # The slack covers a sibling received inside the hold whose record is still being written.
    payable_at = received_at + params.hold_seconds + slack_seconds
    return "pass_unaudited" if now >= payable_at else "wait"


def after_pass(m: MinerState, params: AuditParams, mant_mean: float,
               submission_id: str) -> MinerState:
    if submission_id in m.pass_ids:
        return m
    history = (list(m.mant_mean_history) + [float(mant_mean)])[-MANT_HISTORY:]
    ids = (list(m.pass_ids) + [submission_id])[-PASS_IDS:]
    return replace(m, audited_passed=m.audited_passed + 1, mant_mean_history=history,
                   pass_ids=ids)


def after_confirmed_failure(m: MinerState, params: AuditParams, now: float,
                            submission_id: str) -> MinerState:
    # Any confirmed failure resets audited_passed: probation (and the ban
    # that follows suspect) is left only with no confirmed failure (§5), so
    # a hotkey coming out of suspect always starts a fresh probation count.
    # Idempotent per submission: the state is written before the verdict, so a
    # retry after a crash between the two must not count the failure twice.
    if submission_id in m.failure_ids:
        return m
    ids = (list(m.failure_ids) + [submission_id])[-FAILURE_IDS:]
    failures = [t for t in m.confirmed_failures if now - t <= params.ban_window_seconds] + [now]
    if len(failures) >= params.ban_after_failures:
        return replace(m, confirmed_failures=failures, suspect_until=None,
                       banned_until=now + params.ban_seconds, audited_passed=0, failure_ids=ids)
    return replace(m, confirmed_failures=failures, suspect_until=now + params.suspect_seconds,
                   audited_passed=0, failure_ids=ids)
