"""``reliquary ledger``: the payout ledger and its payer.

``reliquary ledger report`` builds the obligations of the settled period
archives, the pass of one UTC day (J+2 at 12:00 by default), the work awaiting
audit, and the reconciliation of the last days with the miner incentive the
chain paid. It reads the bucket (list and get), the public order routes and
the chain; it writes only the local files under ``--out``.

``reliquary ledger pay`` runs one pass of the payer (``payout``) over the same
obligations: a dry run unless ``--execute`` is given with the collector's
wallet, and then only for tasks the registry puts on the collector rail.
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
# Where the payer keeps its one journal and its receipts on the machine it runs on (the owner's, for the pilot).
DEFAULT_JOURNAL = "~/.reliquary/payout/journal.sqlite"
DEFAULT_RECEIPTS = "~/.reliquary/payout/receipts"


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
        view = await sources.read_registry_ceilings()
        ceilings, pay_ceilings, floored = view.ceilings, view.pay_ceilings, view.floored
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


@ledger_app.command("pay")
def pay_command(
    collector_coldkey: str = typer.Option(..., "--collector-coldkey", help="the collector's coldkey (SS58), "
                                                                           "which holds the stake paid out"),
    collector_hotkey: str = typer.Option(..., "--collector-hotkey", help="the collector's hotkey (SS58): the "
                                                                         "stake moves on it"),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="the payer's one journal (SQLite); keep it, "
                                                                    "and back it up: it is what was paid"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS, "--receipts", help="directory each pass's receipt is written to"),
    pay_day: str = typer.Option(None, "--pay-day", help="UTC day of the pass; default: the last pass at or "
                                                        "before --as-of"),
    as_of: str = typer.Option(None, "--as-of", help="ISO time; default: now"),
    archives: str = typer.Option("r2", "--archives", help="'r2' (read-only) or a local directory"),
    prices: str = typer.Option("registry", "--prices", help="'registry' (needed to execute), 'api' or 'none'"),
    api: str = typer.Option(None, "--api", help="the public order routes, for --prices api"),
    rate: str = typer.Option("frozen", "--rate", help="'frozen', 'chain' or a number of alpha per period"),
    lag_days: int = typer.Option(2, "--lag-days"),
    pay_at: str = typer.Option("12:00", "--pay-at", help="time of the daily pass, UTC (HH:MM)"),
    hold_hours: float = typer.Option(0.0, "--hold-hours"),
    order_budget: list[str] = typer.Option([], "--order-budget", help="TASK=ALPHA (repeatable)"),
    history_days: int = typer.Option(16, "--history-days", help="work days read back from the pass day"),
    network: str = typer.Option("archive", "--network", help="the chain endpoint (an archive node: owners "
                                                             "are read at the blocks obligations were earned in)"),
    netuid: int = typer.Option(81, "--netuid"),
    min_transfer: float = typer.Option(None, "--min-transfer", help="alpha a destination must be owed in all to "
                                                                    "be paid; default: the chain's minimum stake"),
    batch_size: int = typer.Option(32, "--batch-size", help="transfers per batch_all"),
    reserve: float = typer.Option(0.0, "--reserve", help="alpha of the collector's stake never paid out"),
    era_period: int = typer.Option(64, "--era-period", help="blocks a signed batch stays valid (power of two)"),
    assume_collector: list[str] = typer.Option([], "--assume-collector", help="dry run only: plan TASK as if it "
                                                                             "were on the collector rail"),
    execute: bool = typer.Option(False, "--execute", help="sign and send (default: a dry run that signs "
                                                          "nothing and writes only its receipt)"),
    confirm_rao: int = typer.Option(None, "--confirm-rao", help="with --execute: the pass's total in rao, as the "
                                                                "dry run printed it; any other total sends nothing"),
    wallet_name: str = typer.Option(None, "--wallet-name", help="with --execute: the collector's wallet"),
    wallet_path: str = typer.Option(None, "--wallet-path", help="with --execute: where the wallet lives"),
    wait_seconds: float = typer.Option(600.0, "--wait-seconds", help="how long to wait for a batch to be final"),
) -> None:
    """Pay the obligations due at a pass from the collector's stake (transfer_stake), each once.

    A dry run by default: it plans, settles nothing, signs nothing, and writes only its receipt.

    Only tasks on the collector rail (payout_rail: collector in the registry) are ever paid."""
    from reliquary.ledger.obligations import RAO_PER_ALPHA, PayTerms, pass_day

    if execute and prices != "registry":
        raise typer.BadParameter("--execute pays only tasks the registry puts on the collector rail: "
                                 "use --prices registry")
    if execute and assume_collector:
        raise typer.BadParameter("--assume-collector is for a dry run only")
    if execute and not wallet_name:
        raise typer.BadParameter("--execute needs --wallet-name (the collector's coldkey signs)")
    if confirm_rao is not None and not execute:
        raise typer.BadParameter("--confirm-rao goes with --execute")
    if prices not in ("registry", "api", "none"):
        raise typer.BadParameter(f"--prices is 'registry', 'api' or 'none', got {prices!r}")
    now = _parse_instant(as_of)
    terms = PayTerms(lag_days=lag_days, pay_at_seconds=_parse_clock(pay_at), hold_seconds=hold_hours * 3600)
    day = date.fromisoformat(pay_day) if pay_day else pass_day(now, terms)
    fixed = None
    if rate not in ("frozen", "chain"):
        try:
            fixed = float(rate)
        except ValueError as exc:
            raise typer.BadParameter(f"--rate is 'frozen', 'chain' or a number, got {rate!r}") from exc
    for name, value in (("--min-transfer", min_transfer), ("--reserve", reserve)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise typer.BadParameter(f"{name} must be a finite alpha amount >= 0")
    receipt, path = asyncio.run(collect_payout(
        now=now, pay_day=day, terms=terms, archives=archives, prices=prices, api=api, rate=rate, fixed_rate=fixed,
        budgets=_parse_budgets(order_budget), history_days=history_days, network=network, netuid=netuid,
        collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey,
        journal_path=journal.expanduser(), receipts=receipts.expanduser(), batch_size=batch_size,
        era_period=era_period, confirm_rao=confirm_rao,
        reserve_rao=math.floor(reserve * RAO_PER_ALPHA),
        min_transfer_rao=None if min_transfer is None else math.floor(min_transfer * RAO_PER_ALPHA),
        assume_collector=assume_collector, execute=execute, wallet_name=wallet_name, wallet_path=wallet_path,
        wait_seconds=wait_seconds))
    typer.echo(render_payout(receipt))
    typer.echo(f"receipt: {path}")
    if receipt.get("stopped"):
        raise typer.Exit(code=3)


async def collect_payout(*, now: float, pay_day: date, terms, archives: str, prices: str, api: str | None,
                         rate: str, fixed_rate: float | None, budgets: dict[str, int], history_days: int,
                         network: str, netuid: int, collector_coldkey: str, collector_hotkey: str,
                         journal_path: Path, receipts: Path, batch_size: int, era_period: int, reserve_rao: int,
                         confirm_rao: int | None = None,
                         min_transfer_rao: int | None, assume_collector: list[str], execute: bool,
                         wallet_name: str | None, wallet_path: str | None, wait_seconds: float):
    """Build the ledger (read-only), then run one pass; returns (receipt, receipt path)."""
    from reliquary.ledger import sources
    from reliquary.ledger.obligations import (
        RATE_CHAIN,
        RATE_FIXED,
        RATE_FROZEN,
        RatePolicy,
        build_ledger,
        day_start,
    )
    from reliquary.ledger.payout import Journal, PayoutConfig, run_pass, write_receipt
    from reliquary.validator import corpus_periods as cp

    first_work = cp.period_of(day_start(pay_day - timedelta(days=history_days)))
    source = sources.BucketArchives() if archives == "r2" else sources.DirectoryArchives(archives)
    refs = [ref for ref in await source.refs(work_from=first_work)
            if ref.settled_at is None or ref.settled_at <= now]
    ceilings, pay_ceilings, collector_tasks = {}, {}, []
    if prices == "registry":
        view = await sources.read_registry_ceilings()
        ceilings, pay_ceilings, collector_tasks = view.ceilings, view.pay_ceilings, list(view.collector_tasks)
    elif prices == "api":
        jobs = {str(ref.document.get("job_id") or ref.task_id): ref.task_id for ref in refs}
        by_job = await asyncio.to_thread(sources.api_ceilings, api or sources.DEFAULT_API, list(jobs))
        ceilings = {jobs[job]: found for job, found in by_job.items()}
    collector_tasks = sorted(set(collector_tasks) | set(assume_collector))
    subtensor = await _subtensor(network)
    journal = None
    try:
        chain_rate = None
        if rate == "chain" or fixed_rate is None:
            chain_rate = await sources.chain_alpha_per_period(subtensor, netuid)
        if fixed_rate is not None:
            policy = RatePolicy(mode=RATE_FIXED, alpha_per_period=fixed_rate)
        elif rate == "chain":
            policy = RatePolicy(mode=RATE_CHAIN, alpha_per_period=chain_rate)
        else:
            policy = RatePolicy(mode=RATE_FROZEN, alpha_per_period=chain_rate)
        ledger = build_ledger(refs, ceilings=ceilings, rates=policy, terms=terms, pay_ceilings=pay_ceilings,
                              order_budgets_rao=budgets)
        config = PayoutConfig(
            netuid=netuid, collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey,
            min_transfer_rao=(min_transfer_rao if min_transfer_rao is not None
                              else await _min_transfer_rao(subtensor)),
            batch_size=batch_size, reserve_rao=reserve_rao, era_period=era_period)
        chain = _payout_chain(subtensor, netuid=netuid, collector_coldkey=collector_coldkey,
                              collector_hotkey=collector_hotkey)
        journal = Journal(journal_path, read_only=not execute)
        signer = (_wallet_signer(subtensor, wallet_name=wallet_name, wallet_path=wallet_path, netuid=netuid,
                                 collector_hotkey=collector_hotkey) if execute else None)
        receipt = await run_pass(ledger.obligations, journal=journal, chain=chain, config=config,
                                 collector_tasks=collector_tasks, pass_day=pay_day, terms=terms, now=now,
                                 execute=execute, signer=signer, confirm_rao=confirm_rao,
                                 wait_seconds=wait_seconds)
    finally:
        if journal is not None:
            journal.close()
        from reliquary.infrastructure.chain import close_subtensor

        await close_subtensor(subtensor)
    receipt["sources"] = {"archives": archives if archives == "r2" else "directory", "archives_read": len(refs),
                          "prices": prices, "rate": policy.to_dict(), "collector_tasks": collector_tasks,
                          "assumed_collector_tasks": sorted(set(assume_collector)),
                          "ledger_anomalies": len(ledger.anomalies)}
    return receipt, write_receipt(receipt, receipts)


@ledger_app.command("publish")
def publish_command(
    receipts: Path = typer.Option(DEFAULT_RECEIPTS, "--receipts", help="directory of the receipts to publish"),
    bucket: str = typer.Option(None, "--bucket", help="the bucket (default: R2_BUCKET_ID)"),
    execute: bool = typer.Option(False, "--execute", help="write them (default: list what would be written)"),
) -> None:
    """Publish executed passes' receipts as public JSON in the bucket, each once, with an index.

    A dry run by default. Dry-run receipts are never published."""
    from reliquary.ledger.receipts import publish

    paths = sorted(receipts.expanduser().glob("*.json"))
    results = asyncio.run(publish(paths, execute=execute, bucket=bucket))
    for result in results:
        typer.echo(f"{result['action']:<18} {result.get('key') or result['path']}")
    if not execute and any(r["action"] == "would_write" for r in results):
        typer.echo("dry run: nothing written (add --execute to publish)")


def render_payout(receipt) -> str:
    from reliquary.ledger.obligations import RAO_PER_ALPHA

    def alpha(rao) -> str:
        return f"{int(rao) / RAO_PER_ALPHA:.9f}"

    lines = [f"{receipt['mode'].upper()} pass of {receipt['pass_day']} (collector {receipt['collector_coldkey']})"]
    lines.append(f"  stake {alpha(receipt['collector_stake_rao'])} alpha at block {receipt['finalized_block']}, "
                 f"in flight {alpha(receipt['in_flight_rao'])}, available {alpha(receipt['available_rao'])}")
    lines.append(f"  due {receipt['due']['obligations']} ({alpha(receipt['due']['rao'])} alpha); native rail, "
                 f"never paid here: {receipt['native_rail']['obligations']} ({alpha(receipt['native_rail']['rao'])})")
    lines.append(f"  planned {receipt['planned']['transfers']} transfer(s), {alpha(receipt['planned']['rao'])} alpha; "
                 f"deferred under the minimum {receipt['deferred_below_minimum']['obligations']}, unfunded "
                 f"{receipt['unfunded']['obligations']}, ownerless {receipt['ownerless']['obligations']}, "
                 f"held out after failed attempts {receipt['stuck']['obligations']}")
    for batch in receipt["recovered"] + receipt["batches"]:
        lines.append(f"  batch {batch['batch_id']} {batch['state']} nonce {batch['nonce']} "
                     f"block {batch['block_number']}: {batch['detail']}")
    lines.append(f"  paid {receipt['paid']['transfers']} transfer(s), {alpha(receipt['paid']['rao'])} alpha")
    if receipt.get("stopped"):
        lines.append(f"  STOPPED: {receipt['stopped']}")
    return "\n".join(lines)


async def _subtensor(network: str):
    import bittensor as bt

    subtensor = bt.AsyncSubtensor(network=network)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


def _payout_chain(subtensor, **kwargs):
    from reliquary.ledger.payout_chain import SubtensorPayoutChain

    return SubtensorPayoutChain(subtensor, **kwargs)


def _wallet_signer(subtensor, *, wallet_name: str, wallet_path: str | None, netuid: int, collector_hotkey: str):
    import bittensor as bt

    from reliquary.ledger.payout_chain import WalletSigner

    wallet = bt.Wallet(name=wallet_name, path=wallet_path) if wallet_path else bt.Wallet(name=wallet_name)
    return WalletSigner(subtensor, wallet, netuid=netuid, collector_hotkey=collector_hotkey)


async def _min_transfer_rao(subtensor) -> int:
    return int((await asyncio.wait_for(subtensor.get_minimum_required_stake(), timeout=120.0)).rao)


def _with_first_seen(item, first_seen: dict[str, float]):
    from dataclasses import replace

    if item.reason == "awaiting_audit" and item.since is None and item.job_id in first_seen:
        return replace(item, since=first_seen[item.job_id])
    return item


def _load_first_seen(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}


__all__ = ["collect_payout", "collect_report", "ledger_app", "render_payout"]
