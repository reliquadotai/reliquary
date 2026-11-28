"""``reliquary ledger``: the payout ledger, read-only.

``reliquary ledger report`` builds the obligations of the settled period
archives, the pass of one UTC day (J+2 at 12:00 by default), the work awaiting
audit, and the reconciliation of the last days with the miner incentive the
chain paid. It reads the bucket (list and get), the public order routes and
the chain; it writes only the local files under ``--out``.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import typer

ledger_app = typer.Typer(
    name="ledger",
    help="Payout ledger: miner obligations due at J+2, computed read-only (it pays nothing)",
    no_args_is_help=True,
)

FIRST_SEEN_FILE = "pending_first_seen.json"


@ledger_app.callback()
def _ledger() -> None:
    """Payout ledger: miner obligations due at J+2, computed read-only (it pays nothing)."""


def _parse_clock(value: str) -> int:
    hours, _, minutes = value.partition(":")
    try:
        seconds = int(hours) * 3600 + int(minutes or 0) * 60
    except ValueError as exc:
        raise typer.BadParameter(f"not a time of day (HH:MM): {value!r}") from exc
    if not 0 <= seconds < 86_400:
        raise typer.BadParameter(f"not a time of day: {value!r}")
    return seconds


def _parse_fallbacks(value: str) -> tuple[str, ...]:
    from reliquary.ledger.obligations import FALLBACKS

    found = tuple(part.strip() for part in (value or "").split(",") if part.strip())
    if any(f not in FALLBACKS for f in found):
        raise typer.BadParameter(f"--fallback takes a list of {', '.join(FALLBACKS)}, got {value!r}")
    return found


def _parse_budgets(values: list[str]) -> dict[str, int]:
    from reliquary.ledger.obligations import RAO_PER_ALPHA

    budgets = {}
    for value in values:
        task, sep, alpha = value.partition("=")
        if not sep or not task:
            raise typer.BadParameter(f"--order-budget takes TASK=ALPHA, got {value!r}")
        try:
            amount = float(alpha)
        except ValueError as exc:
            raise typer.BadParameter(f"--order-budget takes TASK=ALPHA, got {value!r}") from exc
        if not math.isfinite(amount) or amount < 0:
            raise typer.BadParameter(f"an order budget must be a finite alpha amount >= 0, got {alpha!r}")
        budgets[task] = math.floor(amount * RAO_PER_ALPHA)
    return budgets


def _parse_instant(value: str | None) -> float:
    if not value:
        return time.time()
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


@ledger_app.command("report")
def report_command(
    pay_day: str = typer.Option(None, "--pay-day", help="UTC day of the pass (YYYY-MM-DD); default: the last "
                                                        "pass at or before --as-of"),
    as_of: str = typer.Option(None, "--as-of", help="ISO time the ledger is seen at; default: now"),
    archives: str = typer.Option("r2", "--archives", help="'r2' (the bucket, read-only) or a local directory "
                                                          "laid out as reliquary/corpus-periods/"),
    prices: str = typer.Option("registry", "--prices", help="where the frozen ceilings come from: 'registry' "
                                                            "(the bucket), 'api' (pay.price_ceiling) or 'none'"),
    api: str = typer.Option(None, "--api", help="the public order routes (default https://api.reliqua.ai)"),
    pending: bool = typer.Option(True, "--pending/--no-pending", help="list work awaiting audit from the "
                                                                      "public routes"),
    rate: str = typer.Option("frozen", "--rate", help="'frozen' (each order's ceiling), 'chain' (the miners' "
                                                      "alpha per period now), or a number of alpha per period"),
    fallback: str = typer.Option("", "--fallback", help="with --rate frozen: what an archive whose price matches "
                                                        "no ceiling may be valued at (latest,chain), each labelled; "
                                                        "default none: it is unvalued"),
    collector_hotkey: str = typer.Option(None, "--collector-hotkey", help="the collector's hotkey: not a miner, "
                                                                          "left out of the reconciliation"),
    lag_days: int = typer.Option(2, "--lag-days", help="an obligation of work day D is due on the pass of D+lag"),
    pay_at: str = typer.Option("12:00", "--pay-at", help="time of the daily pass, UTC (HH:MM)"),
    hold_hours: float = typer.Option(0.0, "--hold-hours", help="least time from settlement to due"),
    escalate_days: float = typer.Option(7.0, "--escalate-days", help="work awaiting audit longer than this is "
                                                                     "flagged for a manual decision"),
    order_budget: list[str] = typer.Option([], "--order-budget", help="TASK=ALPHA: a budget for a whole order "
                                                                      "(repeatable)"),
    history_days: int = typer.Option(16, "--history-days", help="work days read back from the pass day"),
    chain: bool = typer.Option(True, "--chain/--no-chain", help="reconcile with the chain's miner incentive"),
    network: str = typer.Option("archive", "--network", help="the chain endpoint for historical reads"),
    netuid: int = typer.Option(81, "--netuid"),
    window_days: int = typer.Option(7, "--window-days", help="UTC days reconciled, ending at the pass day"),
    lag_epochs: int = typer.Option(None, "--lag-epochs", help="epochs between a weight-set and the epoch that "
                                                              "pays it; default: the commit-reveal period"),
    tolerance: float = typer.Option(0.01, "--tolerance", help="largest gap per hotkey over the window"),
    min_alpha: float = typer.Option(0.0, "--min-alpha", help="hotkeys scheduled less than this are not checked"),
    out: Path = typer.Option(None, "--out", help="directory for report-<day>.json and obligations-<day>.jsonl"),
    as_json: bool = typer.Option(False, "--json", help="print the whole report as JSON"),
) -> None:
    """Build the obligations and the report of one pass. Read-only."""
    from reliquary.ledger.obligations import PayTerms, pass_day
    from reliquary.ledger.report import obligation_rows, render_text
    from reliquary.ledger.sources import DEFAULT_API

    now = _parse_instant(as_of)
    terms = PayTerms(lag_days=lag_days, pay_at_seconds=_parse_clock(pay_at), hold_seconds=hold_hours * 3600,
                     escalate_after_seconds=escalate_days * 86_400)
    day = date.fromisoformat(pay_day) if pay_day else pass_day(now, terms)
    if rate not in ("frozen", "chain"):
        try:
            fixed = float(rate)
        except ValueError as exc:
            raise typer.BadParameter(f"--rate is 'frozen', 'chain' or a number, got {rate!r}") from exc
    else:
        fixed = None
    if prices not in ("registry", "api", "none"):
        raise typer.BadParameter(f"--prices is 'registry', 'api' or 'none', got {prices!r}")
    first_seen_path = None if out is None else out / FIRST_SEEN_FILE
    first_seen = _load_first_seen(first_seen_path)
    report, ledger = asyncio.run(collect_report(
        now=now, pay_day=day, terms=terms, archives=archives, prices=prices, api=api or DEFAULT_API,
        pending=pending, rate=rate, fixed_rate=fixed, budgets=_parse_budgets(order_budget),
        fallbacks=_parse_fallbacks(fallback), collector_hotkey=collector_hotkey,
        history_days=history_days, chain=chain, network=network, netuid=netuid, window_days=window_days,
        lag_epochs=lag_epochs, tolerance=tolerance, min_alpha=min_alpha, first_seen=first_seen))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report-{day.isoformat()}.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        with (out / f"obligations-{day.isoformat()}.jsonl").open("w") as handle:
            for row in obligation_rows(ledger.obligations, now):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        first_seen_path.write_text(json.dumps(first_seen, indent=1, sort_keys=True) + "\n")
    typer.echo(json.dumps(report, indent=1, sort_keys=True) if as_json else render_text(report))


async def collect_report(*, now: float, pay_day: date, terms, archives: str, prices: str, api: str,
                         pending: bool, rate: str, fixed_rate: float | None, budgets: dict[str, int],
                         history_days: int, chain: bool, network: str, netuid: int, window_days: int,
                         lag_epochs: int | None, tolerance: float, min_alpha: float,
                         first_seen: dict[str, float], fallbacks: tuple[str, ...] = (),
                         collector_hotkey: str | None = None):
    """Read every input (read-only) and assemble the report; returns (report, ledger)."""
    from reliquary.ledger import sources
    from reliquary.ledger.obligations import (
        RATE_CHAIN,
        RATE_FIXED,
        RATE_FROZEN,
        RatePolicy,
        build_ledger,
        day_start,
        pending_from_archives,
    )
    from reliquary.ledger.reconcile import reconcile
    from reliquary.ledger.report import daily_report
    from reliquary.validator import corpus_periods as cp

    first_work = cp.period_of(day_start(pay_day - timedelta(days=history_days)))
    source = sources.BucketArchives() if archives == "r2" else sources.DirectoryArchives(archives)
    # Recent work, work settled recently (late parts of older work), and the whole history of any order
    # with a budget: a budget bounds an order only over all of it.
    refs = [ref for ref in await source.refs(work_from=first_work, full_tasks=budgets)
            if ref.settled_at is None or ref.settled_at <= now]
    used = {"archives": archives if archives == "r2" else "directory", "archives_read": len(refs),
            "first_work_period": first_work, "history_days": history_days, "prices": prices,
            "full_history_tasks": sorted(budgets)}

    ceilings, pay_ceilings, floored = {}, {}, []
    pending_items, pricing = [], {}
    if pending:
        settled = {(ref.task_id, ref.work_period) for ref in refs}
        pending_items, pricing = await asyncio.to_thread(
            sources.api_pending, api, now=now, settled=settled, first_seen=first_seen)
        # The public status gives no time for unaudited work: its age counts from the first
        # report that saw it (kept under --out), and is forgotten once it is audited.
        awaiting = {item.job_id for item in pending_items if item.reason == "awaiting_audit"}
        for job_id in set(first_seen) - awaiting:
            del first_seen[job_id]
        for job_id in awaiting:
            first_seen.setdefault(job_id, now)
        pending_items = [_with_first_seen(item, first_seen) for item in pending_items]
        used["pending"] = api
    pending_items += pending_from_archives(refs, now)
    if prices == "registry":
        ceilings, pay_ceilings, floored = await sources.read_registry_ceilings()
    elif prices == "api":
        jobs = {str(ref.document.get("job_id") or ref.task_id): ref.task_id for ref in refs}
        by_job = await asyncio.to_thread(sources.api_ceilings, api, list(jobs), pricing=pricing)
        ceilings = {jobs[job]: found for job, found in by_job.items()}

    subtensor = None
    chain_rate = None
    try:
        if chain or rate == "chain":
            subtensor = await _subtensor(network)
            chain_rate = await sources.chain_alpha_per_period(subtensor, netuid)
            used["chain_alpha_per_period"] = chain_rate
        if fixed_rate is not None:
            policy = RatePolicy(mode=RATE_FIXED, alpha_per_period=fixed_rate)
        elif rate == "chain":
            policy = RatePolicy(mode=RATE_CHAIN, alpha_per_period=chain_rate)
        else:
            if "chain" in fallbacks and chain_rate is None:
                raise typer.BadParameter("--fallback chain needs the chain (drop --no-chain)")
            policy = RatePolicy(mode=RATE_FROZEN, alpha_per_period=chain_rate, fallbacks=fallbacks)
        ledger = build_ledger(refs, ceilings=ceilings, rates=policy, terms=terms, pay_ceilings=pay_ceilings,
                              order_budgets_rao=budgets)
        reconciliation = None
        if chain:
            end = day_start(pay_day)
            start = day_start(pay_day - timedelta(days=window_days))
            epochs, meta = await sources.read_epoch_pays(
                subtensor, netuid, start=start, end=end,
                progress=lambda count, block: typer.echo(f"chain: {count} epochs read, back to block {block}",
                                                         err=True))
            used["chain"] = {"network": network, "netuid": netuid, **meta}
            lag = lag_epochs if lag_epochs is not None else (
                meta["commit_reveal_period"] if meta["commit_reveal"] else 0)
            reconciliation = reconcile(ledger.shares, epochs, start=start, end=end, lag_epochs=lag,
                                       tolerance=tolerance, min_alpha=min_alpha, earliest_entry=first_work,
                                       exclude_hotkeys=[collector_hotkey] if collector_hotkey else [])
    finally:
        if subtensor is not None:
            from reliquary.infrastructure.chain import close_subtensor

            await close_subtensor(subtensor)
    report = daily_report(ledger, now=now, pay_day=pay_day, terms=terms, rates=policy, pending=pending_items,
                          reconciliation=reconciliation, sources=used, floored_tasks=floored)
    return report, ledger


async def _subtensor(network: str):
    import bittensor as bt

    subtensor = bt.AsyncSubtensor(network=network)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


def _with_first_seen(item, first_seen: dict[str, float]):
    from dataclasses import replace

    if item.reason == "awaiting_audit" and item.since is None and item.job_id in first_seen:
        return replace(item, since=first_seen[item.job_id])
    return item


def _load_first_seen(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}


__all__ = ["collect_report", "ledger_app"]
