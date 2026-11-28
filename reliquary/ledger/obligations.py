"""Miner pay obligations, due two UTC days after the work.

A settled period archive (``corpus_period_settlement``, or the RL ledger's,
of the same shape) says what each hotkey earned in that period of work:
``rewards_by_hotkey``, in shares of the miners' emission per period, already
held to the period's budget, min(cap, price x verified units). An order froze,
with its price ceiling, the miner alpha per period its prices were converted
at (``pay.price_ceiling.miner_alpha_per_period``). A share times that rate is
alpha at the order's own prices: what its verified units came to under the
settler's rules (output tokens at the output price; with an input price,
prompt tokens at the input price too; on an auction, each lease at its own
price).

Each (order, period, hotkey, unit type) is one obligation:

- ``amount_rao``: whole rao of alpha, rounded down, never more than the
  period's budget, nor an optional budget for the whole order;
- ``obligation_id``: a hash of (order, period, hotkey, unit type) and the
  archive's origin entry (the entry it was first settled under), the same
  every time it is computed and whatever other archives are present, so a
  payer can pay each one once;
- states: ``earned`` (the period settled, so its audit is final) while its UTC
  day is still open, ``held`` (the day closed: the dispute window), then
  ``due`` on the first pass at or after J+2 12:00 UTC, or the first pass after
  a late settlement. ``reserved``, ``submitted``, ``finalized`` and
  ``reconciled`` belong to a payer: this ledger stops at ``due``;
- a period that ended unsettled owes nothing yet: it is listed as awaiting
  audit, with its age, and never cancelled.

The order is the archive's task id (its bucket prefix, as the weight setter
trusts it). A work period settled in several archives (verdicts that arrived
after it was paid) gives one part per archive, each keyed by its origin entry.
An archive requeued to an earlier entry (``replaces_entry_period``) hides the
one it replaces and keeps its origin (``origin_entry_period``), so its
obligations keep their ids.

Valuation is strict: in ``frozen`` mode an archive is valued only at the
ceiling its own price was set with. With no such ceiling it is ``unvalued``
(an anomaly, no obligation) unless a fallback (the order's latest ceiling, the
chain's rate) is opted into, and then each obligation says so in
``rate_source``. An order budget is spent in settlement order, so a part that
settles later never takes budget from work already allocated.

Two instants per obligation: ``earned_at``, when its archive settled (the due
date counts from it), and ``earn_at``, the end of its work period, a protocol
fact no copy or migration moves: a payer looks the hotkey's owner up there.

Pure: no I/O and no clock. ``sources`` reads, ``cli`` runs. Nothing here holds
a key or moves alpha.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.validator import corpus_periods as cp

RAO_PER_ALPHA = 10**9
OBLIGATION_ID_SCHEMA = "reliquary/obligation/v2"

# What an obligation pays for, from what its archive was priced on.
UNIT_OUTPUT_TOKENS = "output_tokens"
# Input-priced: output tokens at the output price and prompt tokens at the input price,
# split together by value (the archive keeps no per-hotkey split of the two).
UNIT_OUTPUT_AND_PROMPT_TOKENS = "output_and_prompt_tokens"
# An RL period: accepted rollouts at their lease price and first scans at their own rule,
# settled together by the RL ledger (the archive keeps no per-hotkey split of the two).
UNIT_RL_ROLLOUTS = "rl_rollouts"
# No price ceiling: the period's cap split by verified tokens.
UNIT_PERIOD_SHARE = "period_share"
UNIT_TYPES = (UNIT_OUTPUT_TOKENS, UNIT_OUTPUT_AND_PROMPT_TOKENS, UNIT_RL_ROLLOUTS, UNIT_PERIOD_SHARE)

# The RL ledger's archives (``rl_ledger.RL_PERIOD_ARCHIVE_SCHEMA``).
RL_ARCHIVE_SCHEMA = "reliquary/rl-async-period/v1"
RL_MECHANISM = "rl-async"

EARNED, HELD, DUE = "earned", "held", "due"
RESERVED, SUBMITTED, FINALIZED, RECONCILED = "reserved", "submitted", "finalized", "reconciled"
STATES = (EARNED, HELD, DUE, RESERVED, SUBMITTED, FINALIZED, RECONCILED)
# The states this ledger gives: past ``due``, an obligation is the payer's to move.
LEDGER_STATES = (EARNED, HELD, DUE)

# Where each converting rate came from (``Obligation.rate_source``).
RATE_FROZEN = "frozen"                # the ceiling the archive was paid under
RATE_FROZEN_LATEST = "frozen_latest"  # the order's latest ceiling: the archive's price is not among those known
RATE_CHAIN = "chain"                  # the chain's miner alpha per period
RATE_FIXED = "fixed"                  # given by the operator
RATE_MODES = (RATE_FROZEN, RATE_CHAIN, RATE_FIXED)
# What a frozen valuation may fall back to when no ceiling matches, each only when opted into.
FALLBACK_LATEST, FALLBACK_CHAIN = "latest", "chain"
FALLBACKS = (FALLBACK_LATEST, FALLBACK_CHAIN)

# Why an obligation was cut (``Obligation.capped``).
CAPPED_PERIOD_BUDGET = "period_budget"
CAPPED_ORDER_BUDGET = "order_budget"
# A period's rewards over its budget by at most this share of it are float rounding, as the
# weight setter reads them (``paid > cap * (1 + 1e-9)``): held to the budget, not reported.
FLOAT_SLACK = Fraction(1, 10**9)

# How an order's job is priced (``Ceiling.pricing``; the registry's ``pricing`` param).
PRICING_AUCTION = "auction"
PRICING_FIXED = "fixed"

DAY_SECONDS = 86_400


@dataclass(frozen=True, slots=True)
class PayTerms:
    """When obligations fall due. Every field is a setting, not a protocol rule.

    ``lag_days``: an obligation of work day D is due on the pass of D + lag.
    ``pay_at_seconds``: the pass's time of day, in seconds after 00:00 UTC.
    ``hold_seconds``: the least time from settlement to due; a settlement
    later than that before a pass waits for the next one.
    ``escalate_after_seconds``: work awaiting audit longer than this is
    flagged for a manual decision (never cancelled)."""

    lag_days: int = 2
    pay_at_seconds: int = 12 * 3600
    hold_seconds: float = 0.0
    escalate_after_seconds: float = 7 * DAY_SECONDS

    def __post_init__(self) -> None:
        if isinstance(self.lag_days, bool) or not isinstance(self.lag_days, int) or self.lag_days < 0:
            raise ValueError(f"lag_days must be a whole number of days >= 0, got {self.lag_days!r}")
        if (isinstance(self.pay_at_seconds, bool) or not isinstance(self.pay_at_seconds, int)
                or not 0 <= self.pay_at_seconds < DAY_SECONDS):
            raise ValueError(f"pay_at_seconds must be a whole second of the day, got {self.pay_at_seconds!r}")
        for name in ("hold_seconds", "escalate_after_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")

    def to_dict(self) -> dict:
        return {"lag_days": self.lag_days, "pay_at_utc": _clock_of(self.pay_at_seconds),
                "hold_seconds": float(self.hold_seconds),
                "escalate_after_seconds": float(self.escalate_after_seconds)}


@dataclass(frozen=True, slots=True)
class Ceiling:
    """A price ceiling an order was paid under, as ``pay.price_ceiling`` shows it
    (``cp.price_view``). ``pricing``: ``"auction"`` for a job priced by auction,
    ``"fixed"`` for one paid at the ceiling, None when unknown."""

    miner_alpha_per_period: float
    max_price_pool_share_per_token: float | None = None
    max_price_alpha_per_mtoken: float | None = None
    set_at: float | None = None
    max_price_input_pool_share_per_token: float | None = None
    max_price_input_alpha_per_mtoken: float | None = None
    pricing: str | None = None

    def __post_init__(self) -> None:
        rate = self.miner_alpha_per_period
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"a ceiling needs a positive miner alpha per period, got {rate!r}")
        if self.pricing not in (None, PRICING_AUCTION, PRICING_FIXED):
            raise ValueError(f"pricing must be {PRICING_AUCTION!r}, {PRICING_FIXED!r} or None, got {self.pricing!r}")

    @classmethod
    def from_view(cls, view: Mapping, *, pricing: str | None = None) -> Ceiling:
        """From ``pay.price_ceiling`` (``cp.price_view``) or the registry params' view."""
        return cls(miner_alpha_per_period=float(view["miner_alpha_per_period"]),
                   max_price_pool_share_per_token=_optional_float(view.get("max_price_pool_share_per_token")),
                   max_price_alpha_per_mtoken=_optional_float(view.get("max_price_alpha_per_mtoken")),
                   set_at=_optional_float(view.get("set_at")),
                   max_price_input_pool_share_per_token=_optional_float(
                       view.get("max_price_input_pool_share_per_token")),
                   max_price_input_alpha_per_mtoken=_optional_float(view.get("max_price_input_alpha_per_mtoken")),
                   pricing=pricing)

    def to_dict(self) -> dict:
        return {"miner_alpha_per_period": self.miner_alpha_per_period,
                "max_price_pool_share_per_token": self.max_price_pool_share_per_token,
                "max_price_alpha_per_mtoken": self.max_price_alpha_per_mtoken, "set_at": self.set_at,
                "max_price_input_pool_share_per_token": self.max_price_input_pool_share_per_token,
                "max_price_input_alpha_per_mtoken": self.max_price_input_alpha_per_mtoken,
                "pricing": self.pricing}


@dataclass(frozen=True, slots=True)
class RatePolicy:
    """How a share of the miners' emission becomes alpha (owner decision; a setting).

    ``frozen`` (default): the miner alpha per period of the ceiling the archive
    was paid under, and nothing else unless ``fallbacks`` allows it (each such
    obligation is labelled); ``chain``: ``alpha_per_period`` (the chain's miner
    alpha per period) for every archive; ``fixed``: ``alpha_per_period`` as
    given."""

    mode: str = RATE_FROZEN
    alpha_per_period: float | None = None
    # Frozen mode only: what an archive whose price matches no known ceiling may be valued at instead
    # (``latest``: the order's latest ceiling; ``chain``: ``alpha_per_period``). None by default: unvalued.
    fallbacks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in RATE_MODES:
            raise ValueError(f"rate mode must be one of {RATE_MODES}, got {self.mode!r}")
        if any(f not in FALLBACKS for f in self.fallbacks):
            raise ValueError(f"fallbacks must be among {FALLBACKS}, got {self.fallbacks!r}")
        if FALLBACK_CHAIN in self.fallbacks and self.alpha_per_period is None:
            raise ValueError("the chain fallback needs alpha_per_period")
        rate = self.alpha_per_period
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                                 or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"alpha_per_period must be positive, got {rate!r}")
        if self.mode != RATE_FROZEN and rate is None:
            raise ValueError(f"rate mode {self.mode!r} needs alpha_per_period")

    def to_dict(self) -> dict:
        return {"mode": self.mode, "alpha_per_period": self.alpha_per_period, "fallbacks": list(self.fallbacks)}


@dataclass(frozen=True, slots=True)
class ArchiveRef:
    """One period archive as read: its key and body, and when it was written
    (the object's time; None when unknown)."""

    task_id: str
    work_period: int
    entry_period: int
    document: Mapping
    settled_at: float | None = None


@dataclass(frozen=True, slots=True)
class Obligation:
    obligation_id: str
    task_id: str
    job_id: str | None
    work_period: int
    entry_period: int
    part: int
    # The entry the archive was first settled under: part of the id, stable whatever else is present.
    origin_entry: int
    hotkey: str
    unit_type: str
    # Verified output tokens when the archive can tell them per hotkey (a fixed
    # ceiling on output tokens alone), else None.
    units: int | None
    # The settled reward this obligation pays, in shares of the miners' emission
    # per period, after any cut to the period's budget.
    pool_share: float
    alpha_per_share: float
    rate_source: str
    amount_rao: int
    capped: str | None
    work_day: date
    period_end: float
    earned_at: float
    due_at: float
    # The end of the work period: the immutable instant a payer looks the hotkey's owner up at (D5).
    earn_at: float
    archived_rail: str | None = None

    @property
    def alpha(self) -> float:
        return self.amount_rao / RAO_PER_ALPHA

    def state_at(self, now: float) -> str | None:
        return state_at(self, now)

    def to_dict(self) -> dict:
        return {"obligation_id": self.obligation_id, "task_id": self.task_id, "job_id": self.job_id,
                "work_period": self.work_period, "entry_period": self.entry_period, "part": self.part,
                "origin_entry": self.origin_entry,
                "hotkey": self.hotkey, "unit_type": self.unit_type, "units": self.units,
                "pool_share": self.pool_share, "alpha_per_share": self.alpha_per_share,
                "rate_source": self.rate_source, "amount_rao": self.amount_rao, "capped": self.capped,
                "work_day": self.work_day.isoformat(), "period_end": self.period_end,
                "earned_at": self.earned_at, "due_at": self.due_at, "earn_at": self.earn_at, "archived_rail": self.archived_rail}


@dataclass(frozen=True, slots=True)
class Ledger:
    """Every obligation of the archives given, and what could not be turned into
    one (``anomalies``: each a dict with a ``kind``)."""

    obligations: tuple[Obligation, ...]
    anomalies: tuple[Mapping, ...] = ()
    # Every archive's settled shares after the period's budget, valued or not: (task id, entry period,
    # hotkey, pool share). What the weights pay, independent of any rate (``reconcile``).
    shares: tuple[tuple[str, int, str, float], ...] = ()

    def visible_at(self, now: float) -> list[Obligation]:
        """The obligations already earned at ``now``."""
        return [o for o in self.obligations if o.earned_at <= now]


@dataclass(frozen=True, slots=True)
class PendingWork:
    """Work that ended and is not settled: it owes nothing yet, is never due, and
    is never cancelled. ``work_period`` is None when only the job is known (work
    accepted and not audited); ``since`` is when it became pending (the period's
    end, or the oldest undecided arrival), None when unknown."""

    job_id: str
    reason: str
    task_id: str | None = None
    work_period: int | None = None
    since: float | None = None
    verified_tokens: int | None = None
    submissions_awaiting: int | None = None
    # At most this much alpha at the order's ceiling (an upper bound, never an obligation).
    ceiling_alpha: float | None = None
    held: str | None = None

    def age_at(self, now: float) -> float | None:
        return None if self.since is None else max(0.0, float(now) - float(self.since))

    def escalate_at(self, now: float, terms: PayTerms) -> bool | None:
        age = self.age_at(now)
        return None if age is None else age > terms.escalate_after_seconds

    def to_dict(self, now: float, terms: PayTerms) -> dict:
        return {"job_id": self.job_id, "task_id": self.task_id, "reason": self.reason,
                "work_period": self.work_period,
                "work_day": None if self.work_period is None else work_day(self.work_period).isoformat(),
                "since": self.since, "age_seconds": self.age_at(now),
                "escalate": self.escalate_at(now, terms), "verified_tokens": self.verified_tokens,
                "submissions_awaiting": self.submissions_awaiting, "ceiling_alpha": self.ceiling_alpha,
                "held": self.held}


# -- identity and amounts ---------------------------------------------------------------------------------------


def obligation_id(task_id: str, work_period: int, hotkey: str, unit_type: str, origin_entry: int) -> str:
    """The id of one obligation: a SHA-256 of its order, period, hotkey, unit
    type and its archive's origin entry. Intrinsic to the archive, so it is the
    same across runs, processes and machines, and whichever other archives of
    the period are present."""
    if unit_type not in UNIT_TYPES:
        raise ValueError(f"unknown unit type {unit_type!r}")
    if isinstance(origin_entry, bool) or not isinstance(origin_entry, int) or origin_entry < 0:
        raise ValueError(f"origin_entry must be a whole number >= 0, got {origin_entry!r}")
    return canonical_sha256([OBLIGATION_ID_SCHEMA, str(task_id), int(work_period), str(hotkey), unit_type,
                             origin_entry])


def amount_rao(pool_share, alpha_per_share) -> int:
    """``pool_share`` x ``alpha_per_share`` in whole rao, rounded down. Exact:
    the binary floats are multiplied as the rationals they are, so the same
    inputs give the same rao on every machine."""
    share, rate = _fraction(pool_share, "pool share"), _fraction(alpha_per_share, "alpha per share")
    return math.floor(share * rate * RAO_PER_ALPHA)


def unit_type_of(document: Mapping) -> str:
    """What an archive's obligations pay for, from how it was priced."""
    if document.get("schema") == RL_ARCHIVE_SCHEMA or document.get("mechanism") == RL_MECHANISM:
        return UNIT_RL_ROLLOUTS
    if document.get("max_price_input_per_token") is not None:
        return UNIT_OUTPUT_AND_PROMPT_TOKENS
    if document.get("max_price_per_token") is not None:
        return UNIT_OUTPUT_TOKENS
    return UNIT_PERIOD_SHARE


def resolve_rate(document: Mapping, ceilings: Sequence[Ceiling], policy: RatePolicy) -> tuple[float | None, str]:
    """The miner alpha per period an archive's shares convert at, and where it came from."""
    if policy.mode == RATE_FIXED:
        return float(policy.alpha_per_period), RATE_FIXED
    if policy.mode == RATE_CHAIN:
        return float(policy.alpha_per_period), RATE_CHAIN
    price = document.get("max_price_per_token")
    if price is not None and unit_type_of(document) != UNIT_RL_ROLLOUTS:
        same = [c for c in ceilings if c.max_price_pool_share_per_token == float(price)]
        if same:
            return float(_latest(same).miner_alpha_per_period), RATE_FROZEN
    if FALLBACK_LATEST in policy.fallbacks and ceilings and unit_type_of(document) != UNIT_RL_ROLLOUTS:
        return float(_latest(ceilings).miner_alpha_per_period), RATE_FROZEN_LATEST
    if FALLBACK_CHAIN in policy.fallbacks and policy.alpha_per_period is not None:
        return float(policy.alpha_per_period), RATE_CHAIN
    return None, "unvalued"


# -- the J+2 calendar -------------------------------------------------------------------------------------------


def utc_day(t: float) -> date:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).date()


def day_start(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def work_day(work_period: int, genesis: float = cp.PERIOD_EPOCH) -> date:
    """The UTC day a period of work ended in: the day holding its last instant."""
    end = cp.period_end(int(work_period), genesis)
    day = utc_day(end)
    return day - timedelta(days=1) if day_start(day) == end else day


def pay_instant(day: date, terms: PayTerms) -> float:
    """When the pass of ``day`` runs."""
    return day_start(day) + terms.pay_at_seconds


def pass_day(t: float, terms: PayTerms) -> date:
    """The day of the last pass at or before ``t``."""
    day = utc_day(t)
    return day if pay_instant(day, terms) <= t else day - timedelta(days=1)


def next_pass(t: float, terms: PayTerms) -> float:
    """The first pass at or after ``t``."""
    day = utc_day(t)
    at = pay_instant(day, terms)
    return at if at >= t else pay_instant(day + timedelta(days=1), terms)


def due_at(day: date, earned_at: float, terms: PayTerms) -> float:
    """The pass that pays an obligation of work ``day`` settled at ``earned_at``:
    the pass of day + lag, or the first pass after a later settlement."""
    return max(pay_instant(day + timedelta(days=terms.lag_days), terms),
               next_pass(float(earned_at) + float(terms.hold_seconds), terms))


def state_at(obligation: Obligation, now: float) -> str | None:
    """``earned`` while its work day is open, ``held`` from then until its pass,
    ``due`` from its pass on; None before it was earned. Never past ``due``."""
    if now < obligation.earned_at:
        return None
    if now >= obligation.due_at:
        return DUE
    if now >= max(obligation.earned_at, day_start(obligation.work_day + timedelta(days=1))):
        return HELD
    return EARNED


# -- building the ledger ----------------------------------------------------------------------------------------


def archive_parts(archives: Iterable[ArchiveRef]) -> list[tuple[ArchiveRef, int]]:
    """Each archive with its part in its work period, oldest first. An archive
    another one of the same work period replaces (that one names it in
    ``replaces_entry_period`` from an earlier entry) is left out, as the weight
    setter leaves it out; the one replacing it takes its place in the order."""
    by_work: dict[tuple[str, int], list[ArchiveRef]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for ref in archives:
        key = (str(ref.task_id), int(ref.work_period), int(ref.entry_period))
        if key in seen:
            continue  # one archive per key (the store refuses a second document under one)
        seen.add(key)
        by_work[key[:2]].append(ref)
    out: list[tuple[ArchiveRef, int]] = []
    for key in sorted(by_work):
        group = by_work[key]
        entries = {int(ref.entry_period) for ref in group}
        hidden = set()
        for ref in group:
            moved = _replaced_entry(ref)
            if moved is not None and moved in entries and int(ref.entry_period) < moved:
                hidden.add(moved)
        kept = [ref for ref in group if int(ref.entry_period) not in hidden]
        kept.sort(key=lambda ref: (origin_entry(ref), int(ref.entry_period)))
        out.extend((ref, part) for part, ref in enumerate(kept))
    return out


def build_ledger(archives: Iterable[ArchiveRef], *, ceilings: Mapping[str, Sequence[Ceiling]] | None = None,
                 rates: RatePolicy = RatePolicy(), terms: PayTerms = PayTerms(),
                 pay_ceilings: Mapping[str, float] | None = None,
                 order_budgets_rao: Mapping[str, int] | None = None,
                 genesis: float = cp.PERIOD_EPOCH) -> Ledger:
    """Every obligation of ``archives``.

    ``ceilings``: each order's price ceilings (its frozen rates), by task id.
    ``pay_ceilings``: each task's ``cp.pay_ceiling`` (the most one archive may
    pay, the weight setter's bound), when known. ``order_budgets_rao``: a budget
    for a whole order, in rao, spent in settlement order (the part that crosses
    it pro rata); it bounds the order only when ``archives`` hold its whole
    history. An archive whose rewards are not finite non-negative numbers, or
    that cannot be valued, gives no obligation and is reported in
    ``anomalies``; its shares still count in ``shares``."""
    ceilings = ceilings or {}
    pay_ceilings = pay_ceilings or {}
    budgets = {str(task): int(rao) for task, rao in (order_budgets_rao or {}).items()}
    for task, rao in budgets.items():
        if rao < 0:
            raise ValueError(f"the order budget of {task!r} is negative")
    anomalies: list[dict] = []
    shares: list[tuple[str, int, str, float]] = []
    valued: list[dict] = []
    spent_share: dict[tuple[str, int], Fraction] = defaultdict(Fraction)
    for ref, part in archive_parts(archives):
        task, doc = str(ref.task_id), ref.document
        where = {"task_id": task, "work_period": int(ref.work_period), "entry_period": int(ref.entry_period)}
        try:
            rewards = _rewards(doc)
            bound = _archive_bound(doc, pay_ceilings.get(task))
        except ValueError as exc:
            anomalies.append({"kind": "invalid_archive", **where, "detail": str(exc)})
            continue
        if not rewards:
            continue
        if doc.get("task_id") not in (None, task):
            anomalies.append({"kind": "task_id_mismatch", **where, "body_task_id": doc.get("task_id")})
        settled = sum(rewards.values(), Fraction(0))
        paid = rewards
        capped = None
        if bound is not None:
            room = max(Fraction(0), bound - spent_share[(task, ref.work_period)])
            if settled > room:
                # Held to the budget exactly, so no obligation ever owes past it. Float sums
                # of shares that make up the budget exceed it by an ulp (exact arithmetic sees
                # it): only an excess past that slack is a cut worth reporting.
                scale = room / settled if settled > 0 else Fraction(0)
                paid = {hotkey: share * scale for hotkey, share in rewards.items()}
                if settled > room * (1 + FLOAT_SLACK):
                    capped = CAPPED_PERIOD_BUDGET
                    anomalies.append({"kind": "over_period_budget", **where, "part": part,
                                      "settled_pool_share": float(settled), "room_pool_share": float(room)})
        spent_share[(task, ref.work_period)] += sum(paid.values(), Fraction(0))
        shares.extend((task, int(ref.entry_period), hotkey, float(share)) for hotkey, share in sorted(paid.items()))
        unit = unit_type_of(doc)
        rate, source = resolve_rate(doc, ceilings.get(task, ()), rates)
        if rate is None:
            anomalies.append({"kind": "unvalued", **where, "pool_share": float(sum(paid.values(), Fraction(0))),
                              "detail": "no ceiling matches its price, and no fallback was allowed"})
            continue
        if source not in (RATE_FROZEN, RATE_FIXED) and rates.mode == RATE_FROZEN:
            anomalies.append({"kind": "valued_by_fallback", **where, "rate_source": source})
        valued.append({"ref": ref, "part": part, "origin": origin_entry(ref), "unit": unit, "rate": rate,
                       "source": source, "paid": paid, "rewards": rewards, "capped": capped,
                       "amounts": {hotkey: amount_rao(share, rate) for hotkey, share in paid.items()},
                       "earned": (float(ref.settled_at) if ref.settled_at is not None
                                  # Unknown: the entry period's start, which a settlement always precedes.
                                  else cp.period_end(int(ref.entry_period) - 1, genesis))})
    # An order's budget, spent in settlement order: what settles later never takes budget from what came first.
    for task, budget in budgets.items():
        left = budget
        for row in sorted((r for r in valued if str(r["ref"].task_id) == task),
                          key=lambda r: (r["earned"], int(r["ref"].work_period), r["origin"])):
            total = sum(row["amounts"].values())
            if total > left:
                row["amounts"] = {hotkey: amount * left // total for hotkey, amount in row["amounts"].items()}
                row["capped"] = CAPPED_ORDER_BUDGET
            left = max(0, left - sum(row["amounts"].values()))
    obligations: list[Obligation] = []
    for row in valued:
        ref, task = row["ref"], str(row["ref"].task_id)
        where = {"task_id": task, "work_period": int(ref.work_period), "entry_period": int(ref.entry_period),
                 "part": row["part"]}
        # Old archives rely on the immutable task registration and its first collector period.
        # An explicit contradictory identity or malformed rail cannot attest a transfer.
        archived_rail = ref.document.get("payout_rail")
        if (ref.document.get("task_id") not in (None, task)
                or ref.document.get("work_period") not in (None, ref.work_period)
                or ("payout_rail" in ref.document and archived_rail not in ("native", "collector"))):
            archived_rail = "unknown"
        day = work_day(ref.work_period, genesis)
        due = due_at(day, row["earned"], terms)
        units = _units(ref.document, row["rewards"], ceilings.get(task, ()), row["unit"])
        unpaid: dict[str, int] = defaultdict(int)
        for hotkey in sorted(row["paid"]):
            amount = row["amounts"][hotkey]
            if amount <= 0:
                # Why nothing is owed: the period's budget was spent by an earlier part, an order's budget
                # ran out, or the share is under one rao.
                reason = ("period_budget_spent" if row["paid"][hotkey] == 0
                          else "order_budget_spent" if row["capped"] == CAPPED_ORDER_BUDGET else "under_one_rao")
                unpaid[reason] += 1
                continue
            obligations.append(Obligation(
                obligation_id=obligation_id(task, ref.work_period, hotkey, row["unit"], row["origin"]),
                task_id=task, job_id=_optional_str(ref.document.get("job_id")), work_period=int(ref.work_period),
                entry_period=int(ref.entry_period), part=row["part"], origin_entry=row["origin"], hotkey=hotkey,
                unit_type=row["unit"], units=None if units is None else units.get(hotkey),
                pool_share=float(row["paid"][hotkey]), alpha_per_share=float(row["rate"]),
                rate_source=row["source"], amount_rao=amount, capped=row["capped"], work_day=day,
                period_end=cp.period_end(int(ref.work_period), genesis), earned_at=row["earned"], due_at=due,
                earn_at=cp.period_end(int(ref.work_period), genesis), archived_rail=archived_rail))
        for reason, count in sorted(unpaid.items()):
            anomalies.append({"kind": reason, **where, "hotkeys": count})
    return Ledger(obligations=tuple(obligations), anomalies=tuple(anomalies), shares=tuple(shares))


def pending_from_archives(archives: Iterable[ArchiveRef], now: float,
                          genesis: float = cp.PERIOD_EPOCH) -> list[PendingWork]:
    """The periods that had ended at ``now`` but were settled only after it
    (known in hindsight from when their archives were written)."""
    out = []
    for ref, _ in archive_parts(archives):
        end = cp.period_end(int(ref.work_period), genesis)
        if ref.settled_at is not None and end <= now < float(ref.settled_at):
            out.append(PendingWork(job_id=str(ref.document.get("job_id") or ref.task_id), reason="settled_later",
                                   task_id=str(ref.task_id), work_period=int(ref.work_period), since=end))
    return out


def by_hotkey(obligations: Iterable[Obligation]) -> dict[str, int]:
    """Rao owed per hotkey."""
    totals: dict[str, int] = defaultdict(int)
    for obligation in obligations:
        totals[obligation.hotkey] += obligation.amount_rao
    return dict(sorted(totals.items()))


# -- helpers ----------------------------------------------------------------------------------------------------


def _clock_of(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_str(value) -> str | None:
    return None if value is None else str(value)


def _fraction(value, name: str) -> Fraction:
    if isinstance(value, Fraction):
        number = value
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    else:
        number = Fraction(value)
    if number < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return number


def _latest(ceilings: Sequence[Ceiling]) -> Ceiling:
    """The ceiling set last (one without a time counts as the oldest)."""
    return max(enumerate(ceilings), key=lambda pair: (pair[1].set_at is not None, pair[1].set_at or 0.0,
                                                       pair[0]))[1]


def _rewards(document: Mapping) -> dict[str, Fraction]:
    raw = document.get("rewards_by_hotkey") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("rewards_by_hotkey is not a map")
    rewards = {}
    for hotkey, value in raw.items():
        if not isinstance(hotkey, str) or not hotkey:
            raise ValueError(f"a hotkey must be a non-empty string, got {hotkey!r}")
        share = _fraction(value, f"the reward of {hotkey}")
        if share > 0:
            rewards[hotkey] = share
    return rewards


def _archive_bound(document: Mapping, pay_ceiling: float | None) -> Fraction | None:
    """The most an archive's work period may pay, in pool share: its budget (the
    settler's min(cap, price x units)) and the weight setter's bound
    (``cp.archive_bound``: its cap, never past the task's pay ceiling)."""
    bounds = []
    if document.get("budget") is not None:
        bounds.append(_fraction(document["budget"], "the archive's budget"))
    recorded = document.get("cap")
    if recorded is not None:
        _fraction(recorded, "the archive's cap")
    bound = cp.archive_bound(recorded, pay_ceiling)
    if bound is not None:
        bounds.append(_fraction(float(bound), "the archive's cap"))
    return min(bounds) if bounds else None


def _replaced_entry(ref: ArchiveRef) -> int | None:
    moved = ref.document.get("replaces_entry_period")
    return None if moved is None else int(moved)


def origin_entry(ref: ArchiveRef) -> int:
    """The entry an archive was first settled under: kept through requeues
    (``origin_entry_period``, else the ``replaces_entry_period`` of a single
    move), else its own entry."""
    origin = ref.document.get("origin_entry_period")
    if origin is not None:
        return int(origin)
    moved = _replaced_entry(ref)
    return int(ref.entry_period) if moved is None else moved


def _units(document: Mapping, rewards: Mapping[str, Fraction], ceilings: Sequence[Ceiling],
           unit: str) -> dict[str, int] | None:
    """Each hotkey's verified output tokens, where the archive tells them: on a
    fixed ceiling over output tokens alone, the settler splits the period by
    tokens, so a hotkey's tokens are the archive's tokens x its share of the
    rewards. Elsewhere (an auction, an input price, RL, or pricing unknown) the
    archive keeps no such split: None."""
    if unit != UNIT_OUTPUT_TOKENS or document.get("tokens") is None or not ceilings:
        return None
    price = float(document["max_price_per_token"])
    paid_under = [c for c in ceilings if c.max_price_pool_share_per_token == price]
    if not paid_under or any(c.pricing != PRICING_FIXED for c in paid_under):
        return None
    tokens = int(document["tokens"])
    total = sum(rewards.values(), Fraction(0))
    if tokens < 0 or total <= 0:
        return None
    return {hotkey: round(tokens * share / total) for hotkey, share in rewards.items()}


__all__ = [
    "ArchiveRef",
    "CAPPED_ORDER_BUDGET",
    "CAPPED_PERIOD_BUDGET",
    "Ceiling",
    "DUE",
    "EARNED",
    "HELD",
    "Ledger",
    "Obligation",
    "PRICING_AUCTION",
    "PRICING_FIXED",
    "PayTerms",
    "PendingWork",
    "RAO_PER_ALPHA",
    "RATE_CHAIN",
    "RATE_FIXED",
    "RATE_FROZEN",
    "RATE_FROZEN_LATEST",
    "FALLBACKS",
    "FALLBACK_CHAIN",
    "FALLBACK_LATEST",
    "RatePolicy",
    "LEDGER_STATES",
    "STATES",
    "UNIT_OUTPUT_AND_PROMPT_TOKENS",
    "UNIT_OUTPUT_TOKENS",
    "UNIT_PERIOD_SHARE",
    "UNIT_RL_ROLLOUTS",
    "UNIT_TYPES",
    "amount_rao",
    "archive_parts",
    "build_ledger",
    "by_hotkey",
    "day_start",
    "due_at",
    "next_pass",
    "obligation_id",
    "origin_entry",
    "pass_day",
    "pay_instant",
    "pending_from_archives",
    "resolve_rate",
    "state_at",
    "unit_type_of",
    "utc_day",
    "work_day",
]
