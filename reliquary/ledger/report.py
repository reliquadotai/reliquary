"""The payout ledger's daily report: one day's pass, every obligation's state,
the work awaiting audit, and the reconciliation with the chain.

Pure: it is handed the ledger, the pending work and the reconciliation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone

from reliquary.ledger.obligations import (
    RAO_PER_ALPHA,
    LEDGER_STATES,
    Ledger,
    Obligation,
    PayTerms,
    PendingWork,
    RatePolicy,
    by_hotkey,
    pay_instant,
    state_at,
)

REPORT_SCHEMA = "reliquary/obligation-report/v1"


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()


def daily_report(ledger: Ledger, *, now: float, pay_day: date, terms: PayTerms, rates: RatePolicy,
                 pending: Iterable[PendingWork] = (), reconciliation: Mapping | None = None,
                 sources: Mapping | None = None, floored_tasks: Sequence[str] = ()) -> dict:
    """The report of the pass of ``pay_day`` as seen at ``now``: before the pass
    it is a preview (more may settle in time to join it)."""
    visible = ledger.visible_at(now)
    at = pay_instant(pay_day, terms)
    in_pass = [o for o in visible if o.due_at == at]
    owed = by_hotkey(in_pass)
    by_order: dict[str, int] = defaultdict(int)
    for obligation in in_pass:
        by_order[obligation.task_id] += obligation.amount_rao
    states = {state: {"obligations": 0, "rao": 0} for state in LEDGER_STATES}
    for obligation in visible:
        row = states[state_at(obligation, now)]
        row["obligations"] += 1
        row["rao"] += obligation.amount_rao
    late = [o for o in in_pass if o.work_day + timedelta(days=terms.lag_days) < pay_day]
    ids = Counter(o.obligation_id for o in ledger.obligations)
    pending = list(pending)
    return {
        "schema": REPORT_SCHEMA,
        "mode": "ledger",
        "as_of": float(now),
        "as_of_utc": iso(now),
        "terms": terms.to_dict(),
        "rate": rates.to_dict(),
        "pass": {
            "day": pay_day.isoformat(), "at": at, "at_utc": iso(at), "preview": now < at,
            "obligations": len(in_pass), "rao": sum(owed.values()),
            "alpha": sum(owed.values()) / RAO_PER_ALPHA, "hotkeys": len(owed),
            "work_days": sorted({o.work_day.isoformat() for o in in_pass}),
            "late_obligations": len(late),
            "by_hotkey": [{"hotkey": hotkey, "rao": rao, "alpha": rao / RAO_PER_ALPHA}
                          for hotkey, rao in sorted(owed.items(), key=lambda kv: (-kv[1], kv[0]))],
            "by_order": [{"task_id": task, "rao": rao, "alpha": rao / RAO_PER_ALPHA}
                         for task, rao in sorted(by_order.items(), key=lambda kv: (-kv[1], kv[0]))],
        },
        "states": states,
        "pending_audit": {
            "items": [p.to_dict(now, terms) for p in pending],
            "escalate": sum(1 for p in pending if p.escalate_at(now, terms)),
        },
        "checks": {
            "obligation_ids_unique": all(count == 1 for count in ids.values()),
            "states_beyond_due": sum(1 for o in visible if state_at(o, now) not in LEDGER_STATES),
        },
        "anomalies": [dict(a) for a in ledger.anomalies],
        "incentive_floor_tasks": list(floored_tasks),
        "reconciliation": None if reconciliation is None else dict(reconciliation),
        "sources": dict(sources or {}),
    }


def obligation_rows(obligations: Iterable[Obligation], now: float) -> list[dict]:
    """Every obligation with its state at ``now``, for ``obligations-<day>.jsonl``."""
    return [{**o.to_dict(), "state": state_at(o, now)} for o in obligations if o.earned_at <= now]


def render_text(report: Mapping, *, top: int = 15) -> str:
    """A short human summary of a report."""
    lines = []
    pay = report["pass"]
    head = "PREVIEW of the pass" if pay["preview"] else "Pass"
    lines.append(f"{head} of {pay['day']} at {pay['at_utc']} (ledger only: nothing is paid here)")
    lines.append(f"  {pay['obligations']} obligations, {pay['hotkeys']} hotkeys, {pay['alpha']:.9f} alpha "
                 f"({pay['rao']} rao); work days {', '.join(pay['work_days']) or '-'}; "
                 f"{pay['late_obligations']} late")
    for row in pay["by_hotkey"][:top]:
        lines.append(f"    {row['hotkey']}  {row['alpha']:.9f}")
    if len(pay["by_hotkey"]) > top:
        lines.append(f"    ... {len(pay['by_hotkey']) - top} more")
    states = report["states"]
    lines.append("States as of " + report["as_of_utc"] + ": " + ", ".join(
        f"{state} {row['obligations']} ({row['rao'] / RAO_PER_ALPHA:.6f} alpha)" for state, row in states.items()))
    pending = report["pending_audit"]
    lines.append(f"Awaiting audit: {len(pending['items'])} item(s), {pending['escalate']} past the escalation age")
    for item in pending["items"][:top]:
        age = item["age_seconds"]
        lines.append(f"    {item['job_id']} {item['reason']} period {item['work_period']} "
                     f"age {'?' if age is None else f'{age / 3600:.1f} h'}"
                     + (f" submissions {item['submissions_awaiting']}" if item["submissions_awaiting"] else "")
                     + (f" held {item['held']}" if item["held"] else "")
                     + (" ESCALATE" if item["escalate"] else ""))
    if report["anomalies"]:
        kinds = Counter(a["kind"] for a in report["anomalies"])
        lines.append("Anomalies: " + ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())))
    recon = report.get("reconciliation")
    if recon:
        window, crit, totals = recon["window"], recon["criterion"], recon["totals"]
        lines.append(f"Reconciliation {iso(window['start'])} .. {iso(window['end'])}: {window['epochs']} epochs, "
                     f"lag {window['lag_epochs']} epoch(s)"
                     + (" (HISTORY TOO SHORT)" if window["history_too_short"] else ""))
        gap = totals["gap"]
        lines.append(f"  scheduled {totals['scheduled_alpha']:.6f} alpha, on-chain {totals['onchain_alpha']:.6f} "
                     f"alpha, gap {'-' if gap is None else f'{gap:+.3%}'}; on-chain unscheduled "
                     f"{totals['onchain_unscheduled_alpha']:.6f}; owner burned {totals['owner_burned_alpha']:.3f}")
        lines.append(f"  criterion |gap| <= {crit['tolerance']:.1%}: {crit['hotkeys_within']}/{crit['hotkeys_checked']} "
                     f"hotkeys within -> {'PASS' if crit['pass'] else 'FAIL'}")
        lines.append("  lag fit (summed |gap| / scheduled): " + ", ".join(
            f"{lag}: {'-' if value is None else f'{value:.3%}'}" for lag, value in recon["lag_fit"].items()))
        for row in recon["hotkeys"][:top]:
            gap = row["gap"]
            lines.append(f"    {row['hotkey']}  scheduled {row['scheduled_alpha']:.6f}  on-chain "
                         f"{row['onchain_alpha']:.6f}  gap {'-' if gap is None else f'{gap:+.3%}'}")
    return "\n".join(lines)


__all__ = ["REPORT_SCHEMA", "daily_report", "iso", "obligation_rows", "render_text"]
