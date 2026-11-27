"""``reliquary ledger``: the payout ledger and its payer.

``reliquary ledger report`` builds the obligations of the settled period
archives, the pass of one UTC day (J+2 at 12:00 by default), the work awaiting
audit, and the reconciliation of the last days with the miner incentive the
chain paid. It reads the bucket (list and get), the public order routes and
the chain; it writes only the local files under ``--out``.

``reliquary ledger pay`` runs one pass of the payer (``payout``) over the same
obligations: a dry run unless ``--execute`` is given with the collector's
wallet, and then only for tasks the registry puts on the collector rail, at
the approved terms (J+2 at 12:00 UTC, frozen rates, the bucket's archives,
the clock's time). ``journal-init``, ``settle``, ``clear-held`` and
``find-nonce`` are the person's side of the payer: starting or re-aligning
its journal, deciding a batch that stopped it, clearing a refused coldkey,
and reading what took a nonce.
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
# The approved terms of a pass (J+2, 12:00 UTC, no extra hold): the only ones an execution pays at.
APPROVED_TERMS = {"lag_days": 2, "pay_at": "12:00", "hold_hours": 0.0}
MIN_TRANSFER_SOURCE_CHAIN = "the rao count of the chain's NominatorMinRequiredStake, read as alpha"


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

    ceilings, pay_ceilings, floored, collector_tasks = {}, {}, [], []
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
        # Paid by the collector, not by the weights: left out of the reconciliation with the chain.
        collector_tasks = list(view.collector_tasks)
        used["collector_tasks"] = collector_tasks
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
                                       exclude_tasks=collector_tasks,
                                       exclude_hotkeys=[collector_hotkey] if collector_hotkey else [])
    finally:
        if subtensor is not None:
            from reliquary.infrastructure.chain import close_subtensor

            await close_subtensor(subtensor)
    report = daily_report(ledger, now=now, pay_day=pay_day, terms=terms, rates=policy, pending=pending_items,
                          reconciliation=reconciliation, sources=used, floored_tasks=floored,
                          collector_tasks=collector_tasks)
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
    pay_day: str = typer.Option(None, "--pay-day", help="dry run only: UTC day of the pass; default: the last "
                                                        "pass at or before now"),
    as_of: str = typer.Option(None, "--as-of", help="dry run only: ISO time the pass is seen at; default: now"),
    archives: str = typer.Option("r2", "--archives", help="'r2' (read-only; the only source an execution reads) "
                                                          "or, for a dry run, a local directory"),
    prices: str = typer.Option("registry", "--prices", help="'registry' (needed to execute), 'api' or 'none'"),
    api: str = typer.Option(None, "--api", help="the public order routes, for --prices api"),
    lag_days: int = typer.Option(2, "--lag-days", help="dry run only, other than 2"),
    pay_at: str = typer.Option("12:00", "--pay-at", help="time of the daily pass, UTC (HH:MM); dry run only, other "
                                                         "than 12:00"),
    hold_hours: float = typer.Option(0.0, "--hold-hours", help="dry run only, other than 0"),
    history_days: int = typer.Option(16, "--history-days", help="work days of native tasks read back (counted, "
                                                                "never paid); collector-rail tasks are read whole"),
    network: str = typer.Option("archive", "--network", help="the chain endpoint (an archive node: owners "
                                                             "are read at the blocks obligations were earned in)"),
    netuid: int = typer.Option(81, "--netuid"),
    min_transfer: float = typer.Option(None, "--min-transfer", help="alpha a destination must be owed in all to "
                                                                    f"be paid; default: {MIN_TRANSFER_SOURCE_CHAIN}"),
    batch_size: int = typer.Option(32, "--batch-size", help="transfers per batch_all"),
    reserve: float = typer.Option(0.0, "--reserve", help="alpha of the collector's stake never paid out"),
    era_period: int = typer.Option(64, "--era-period", help="blocks a signed batch stays valid (power of two)"),
    assume_collector: list[str] = typer.Option([], "--assume-collector", help="dry run only: plan TASK as if it "
                                                                             "were on the collector rail"),
    execute: bool = typer.Option(False, "--execute", help="sign and send (default: a dry run that signs "
                                                          "nothing and writes only its receipt)"),
    confirm_rao: int = typer.Option(None, "--confirm-rao", help="with --execute: the pass's total in rao, as the "
                                                                "dry run printed it; any other total sends nothing"),
    confirm_plan: str = typer.Option(None, "--confirm-plan", help="with --execute: the plan digest the dry run "
                                                                  "printed; any other plan sends nothing"),
    wallet_name: str = typer.Option(None, "--wallet-name", help="with --execute: the collector's wallet"),
    wallet_path: str = typer.Option(None, "--wallet-path", help="with --execute: where the wallet lives"),
    wait_seconds: float = typer.Option(600.0, "--wait-seconds", help="how long to wait for a batch to be final"),
) -> None:
    """Pay the obligations due at a pass from the collector's stake (transfer_stake), each once.

    A dry run by default: it plans, settles nothing, signs nothing, and writes only its receipt.

    Only tasks on the collector rail (payout_rail: collector in the registry) are ever paid, from the rail's
    first work period on, at their frozen rate. An execution pays at the approved terms only, at the clock's
    time, from the bucket's archives, and only once the journal is initialised (journal-init)."""
    from reliquary.ledger.obligations import RAO_PER_ALPHA, PayTerms, pass_day

    if prices not in ("registry", "api", "none"):
        raise typer.BadParameter(f"--prices is 'registry', 'api' or 'none', got {prices!r}")
    if confirm_rao is not None and not execute:
        raise typer.BadParameter("--confirm-rao goes with --execute")
    if confirm_plan is not None and not execute:
        raise typer.BadParameter("--confirm-plan goes with --execute")
    if execute:
        # What an execution pays is pinned: what a dry run may vary to preview, it may not.
        if prices != "registry":
            raise typer.BadParameter("--execute pays only tasks the registry puts on the collector rail: "
                                     "use --prices registry")
        if archives != "r2":
            raise typer.BadParameter("--execute reads the bucket's archives (--archives r2), not a copy")
        if as_of or pay_day:
            raise typer.BadParameter("--execute pays as of now: --as-of and --pay-day are for a dry run")
        if (lag_days, pay_at, hold_hours) != (APPROVED_TERMS["lag_days"], APPROVED_TERMS["pay_at"],
                                               APPROVED_TERMS["hold_hours"]):
            raise typer.BadParameter("--execute pays at the approved terms (J+2 at 12:00 UTC, no hold): "
                                     "--lag-days, --pay-at and --hold-hours are for a dry run")
        if assume_collector:
            raise typer.BadParameter("--assume-collector is for a dry run only")
        if not wallet_name:
            raise typer.BadParameter("--execute needs --wallet-name (the collector's coldkey signs)")
    now = _parse_instant(as_of)
    terms = PayTerms(lag_days=lag_days, pay_at_seconds=_parse_clock(pay_at), hold_seconds=hold_hours * 3600)
    day = date.fromisoformat(pay_day) if pay_day else pass_day(now, terms)
    for name, value in (("--min-transfer", min_transfer), ("--reserve", reserve)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise typer.BadParameter(f"{name} must be a finite alpha amount >= 0")
    receipt, path = asyncio.run(collect_payout(
        now=now, as_of_supplied=bool(as_of), pay_day=day, terms=terms, archives=archives, prices=prices, api=api,
        history_days=history_days, network=network, netuid=netuid,
        collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey,
        journal_path=journal.expanduser(), receipts=receipts.expanduser(), batch_size=batch_size,
        era_period=era_period, confirm_rao=confirm_rao, confirm_plan=confirm_plan,
        reserve_rao=math.floor(reserve * RAO_PER_ALPHA),
        min_transfer_rao=None if min_transfer is None else math.floor(min_transfer * RAO_PER_ALPHA),
        assume_collector=assume_collector, execute=execute, wallet_name=wallet_name, wallet_path=wallet_path,
        wait_seconds=wait_seconds))
    typer.echo(render_payout(receipt))
    typer.echo(f"receipt: {path}")
    if receipt.get("stopped"):
        raise typer.Exit(code=3)


async def collect_payout(*, now: float, pay_day: date, terms, archives: str, prices: str, api: str | None,
                         history_days: int, network: str, netuid: int, collector_coldkey: str, collector_hotkey: str,
                         journal_path: Path, receipts: Path, batch_size: int, era_period: int, reserve_rao: int,
                         min_transfer_rao: int | None, assume_collector: list[str], execute: bool,
                         wallet_name: str | None, wallet_path: str | None, wait_seconds: float,
                         confirm_rao: int | None = None, confirm_plan: str | None = None,
                         as_of_supplied: bool = False):
    """Build the ledger (read-only), then run one pass; returns (receipt, receipt path).

    Collector-rail tasks are read whole (what is still owed never leaves view); other tasks over the last
    ``history_days`` only, to count what the weights pay. Obligations are valued at the frozen rate only."""
    from reliquary.ledger.payout import Journal, PayoutConfig, incomplete_receipt, run_pass, write_receipt

    ledger, rails, view, used = await _collector_ledger(
        now=now, pay_day=pay_day, terms=terms, archives=archives, prices=prices, api=api,
        history_days=history_days, assume_collector=assume_collector)
    subtensor = await _subtensor(network)
    journal = None
    try:
        if min_transfer_rao is None:
            min_transfer_rao, min_source = await _min_transfer_rao(subtensor), MIN_TRANSFER_SOURCE_CHAIN
        else:
            min_source = "--min-transfer"
        config = PayoutConfig(netuid=netuid, collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey,
                              min_transfer_rao=min_transfer_rao, batch_size=batch_size, reserve_rao=reserve_rao,
                              era_period=era_period)
        chain = _payout_chain(subtensor, netuid=netuid, collector_coldkey=collector_coldkey,
                              collector_hotkey=collector_hotkey)
        journal = Journal(journal_path, read_only=not execute)
        signer = (_wallet_signer(subtensor, wallet_name=wallet_name, wallet_path=wallet_path, netuid=netuid,
                                 collector_hotkey=collector_hotkey) if execute else None)
        try:
            receipt = await run_pass(ledger.obligations, journal=journal, chain=chain, config=config,
                                     collector_tasks=rails, pass_day=pay_day, terms=terms, now=now,
                                     execute=execute, signer=signer, confirm_rao=confirm_rao,
                                     confirm_plan=confirm_plan, wait_seconds=wait_seconds)
        except (Exception, asyncio.CancelledError) as exc:
            # Whatever the journal settled before the error goes in a receipt now, marked incomplete.
            if execute:
                partial = incomplete_receipt(journal, config=config, pass_day=pay_day, now=now, execute=True,
                                             error=exc)
                write_receipt(partial, receipts)
                journal.mark_receipted(partial["reported_batch_ids"])
            raise
        receipt["min_transfer_source"] = min_source
        receipt["sources"] = {**used, "as_of_supplied": as_of_supplied, "journal": str(journal_path)}
        path = write_receipt(receipt, receipts)
        if execute:
            journal.mark_receipted(receipt["reported_batch_ids"])
    finally:
        if journal is not None:
            journal.close()
        from reliquary.infrastructure.chain import close_subtensor

        await close_subtensor(subtensor)
    return receipt, path


# Ledger anomalies after which what a collector-rail task owes is not known: nothing is burned.
OWED_UNKNOWN = ("invalid_archive", "unvalued", "task_id_mismatch", "valued_by_fallback")


async def _collector_ledger(*, now: float, pay_day: date, terms, archives: str, prices: str, api: str | None,
                            history_days: int, assume_collector: list[str] = (), extra_tasks=(),
                            only_collector: bool = False):
    """(the ledger, each collector-rail task's first work period, the registry
    view or None, what was read), all read-only. Collector-rail tasks (and
    ``extra_tasks``) are read over their whole history: what is still owed
    never leaves view. Other tasks, unless ``only_collector``, over the last
    ``history_days``, to count what the weights pay. Valued at the rate frozen
    in each order's ceiling and no fallback: an archive priced under no known
    ceiling is an anomaly, and the payer holds anything not frozen."""
    from reliquary.ledger import sources
    from reliquary.ledger.obligations import RATE_FROZEN, RatePolicy, build_ledger, day_start
    from reliquary.validator import corpus_periods as cp

    first_work = cp.period_of(day_start(pay_day - timedelta(days=history_days)))
    ceilings, pay_ceilings, rails, view = {}, {}, {}, None
    if prices == "registry":
        view = await sources.read_registry_ceilings()
        ceilings, pay_ceilings, rails = view.ceilings, view.pay_ceilings, dict(view.collector_from)
    rails.update({str(task): 0 for task in assume_collector if str(task) not in rails})
    whole = set(rails) | {str(task) for task in extra_tasks}
    refs = []
    if whole or not only_collector:
        source = sources.BucketArchives() if archives == "r2" else sources.DirectoryArchives(archives)
        refs = [ref for ref in await source.refs(work_from=first_work, full_tasks=whole,
                                                 tasks=whole if only_collector else None)
                if ref.settled_at is None or ref.settled_at <= now]
    if prices == "api" and refs:
        jobs = {str(ref.document.get("job_id") or ref.task_id): ref.task_id for ref in refs}
        by_job = await asyncio.to_thread(sources.api_ceilings, api or sources.DEFAULT_API, list(jobs))
        ceilings = {jobs[job]: found for job, found in by_job.items()}
    policy = RatePolicy(mode=RATE_FROZEN)
    ledger = build_ledger(refs, ceilings=ceilings, rates=policy, terms=terms, pay_ceilings=pay_ceilings)
    used = {"archives": archives if archives == "r2" else f"directory {archives}", "archives_read": len(refs),
            "first_work_period": first_work, "history_days": history_days, "prices": prices,
            "registry_present": None if view is None else view.present, "rate": policy.to_dict(),
            "collector_rails": rails, "assumed_collector_tasks": sorted(set(assume_collector)),
            "ledger_anomalies": len(ledger.anomalies),
            "settled": sorted({(ref.task_id, int(ref.work_period)) for ref in refs if ref.task_id in whole})}
    return ledger, rails, view, used


@ledger_app.command("burn-surplus")
def burn_surplus_command(
    collector_coldkey: str = typer.Option(..., "--collector-coldkey", help="the collector's coldkey (SS58)"),
    collector_hotkey: str = typer.Option(..., "--collector-hotkey", help="the collector's hotkey (SS58)"),
    cap: float = typer.Option(None, "--cap", help="the most burned per UTC day, in alpha (needed to execute)"),
    reserve: float = typer.Option(0.0, "--reserve", help="alpha of the collector's stake never burned"),
    unsettled_days: float = typer.Option(7.0, "--unsettled-days", help="days back collector-rail work awaiting its "
                                                                        "audit is held back, at its pay ceiling"),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal", help="the payer's one journal (shared with pay)"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS, "--receipts", help="directory the receipt is written to"),
    as_of: str = typer.Option(None, "--as-of", help="dry run only: ISO time; default: now"),
    archives: str = typer.Option("r2", "--archives", help="'r2' (read-only; the only source an execution reads) "
                                                          "or, for a dry run, a local directory"),
    prices: str = typer.Option("registry", "--prices", help="'registry' (needed to execute) or 'none'"),
    assume_collector: list[str] = typer.Option([], "--assume-collector", help="dry run only: count TASK as on "
                                                                             "the collector rail"),
    network: str = typer.Option("archive", "--network", help="the chain endpoint"),
    netuid: int = typer.Option(81, "--netuid"),
    era_period: int = typer.Option(64, "--era-period", help="blocks the signed burn stays valid (power of two)"),
    execute: bool = typer.Option(False, "--execute", help="sign and send (default: a dry run)"),
    confirm_rao: int = typer.Option(None, "--confirm-rao", help="with --execute: the amount in rao, as the dry "
                                                                "run printed it; any other amount burns nothing"),
    wallet_name: str = typer.Option(None, "--wallet-name", help="with --execute: the collector's wallet"),
    wallet_path: str = typer.Option(None, "--wallet-path", help="with --execute: where the wallet lives"),
    wait_seconds: float = typer.Option(600.0, "--wait-seconds", help="how long to wait for the burn to be final"),
) -> None:
    """Burn the collector's surplus (burn_alpha): its stake beyond what it owes and its reserve, up to a cap.

    A dry run by default. Executing needs --cap, --confirm-rao naming the exact amount, and the collector's wallet;
    it burns as of now, from the bucket's archives and the registry, and only once the journal is initialised."""
    from reliquary.ledger.obligations import RAO_PER_ALPHA

    if confirm_rao is not None and not execute:
        raise typer.BadParameter("--confirm-rao goes with --execute")
    if prices not in ("registry", "none"):
        raise typer.BadParameter(f"--prices is 'registry' or 'none', got {prices!r}")
    if execute:
        if cap is None or not wallet_name:
            raise typer.BadParameter("--execute needs --cap and the collector's --wallet-name")
        if prices != "registry" or archives != "r2" or assume_collector:
            raise typer.BadParameter("--execute counts what is owed from the registry and the bucket's archives: "
                                     "--prices registry, --archives r2, no --assume-collector")
        if as_of:
            raise typer.BadParameter("--execute burns as of now: --as-of is for a dry run")
    for name, value in (("--cap", cap), ("--reserve", reserve), ("--unsettled-days", unsettled_days)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise typer.BadParameter(f"{name} must be a finite amount >= 0")
    now = _parse_instant(as_of)
    receipt, path = asyncio.run(collect_burn(
        now=now, as_of_supplied=bool(as_of), archives=archives, prices=prices, network=network, netuid=netuid,
        collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey, journal_path=journal.expanduser(),
        receipts=receipts.expanduser(), cap_rao=0 if cap is None else math.floor(cap * RAO_PER_ALPHA),
        reserve_rao=math.floor(reserve * RAO_PER_ALPHA), unsettled_days=unsettled_days, era_period=era_period,
        assume_collector=assume_collector, execute=execute, confirm_rao=confirm_rao, wallet_name=wallet_name,
        wallet_path=wallet_path, wait_seconds=wait_seconds))
    typer.echo(render_burn(receipt, capped=cap is not None))
    typer.echo(f"receipt: {path}")
    if receipt.get("stopped"):
        raise typer.Exit(code=3)


async def collect_burn(*, now: float, archives: str, prices: str, network: str, netuid: int, collector_coldkey: str,
                       collector_hotkey: str, journal_path: Path, receipts: Path, cap_rao: int, reserve_rao: int,
                       era_period: int, assume_collector: list[str], execute: bool, confirm_rao: int | None,
                       wallet_name: str | None, wallet_path: str | None, wait_seconds: float,
                       unsettled_days: float = 7.0, as_of_supplied: bool = False):
    """What the collector owes (read-only), then one surplus burn; returns (receipt, receipt path)."""
    from reliquary.ledger.obligations import PayTerms, utc_day
    from reliquary.ledger.payout import Journal, PayoutConfig, incomplete_receipt, write_receipt
    from reliquary.ledger.surplus import SURPLUS_RECEIPT_SCHEMA, run_burn, unsettled_rao

    journal = Journal(journal_path, read_only=not execute)
    subtensor = None
    try:
        paid_tasks = journal.paid_tasks()
        ledger, rails, view, used = await _collector_ledger(
            now=now, pay_day=utc_day(now), terms=PayTerms(), archives=archives, prices=prices, api=None,
            history_days=0, assume_collector=assume_collector, extra_tasks=paid_tasks, only_collector=True)
        cannot_know = []
        if view is not None and not view.present:
            cannot_know.append("the task registry is not in the bucket")
        cannot_know += [f"task {task} was paid by the collector and is not on the rail in the registry"
                        for task in sorted(paid_tasks - set(rails))]
        unknown = [a for a in ledger.anomalies if a.get("kind") in OWED_UNKNOWN and a.get("task_id") in rails]
        if unknown:
            cannot_know.append(f"{len(unknown)} archive(s) of collector-rail tasks could not be valued "
                               f"(first: {unknown[0]})")
        rates = {task: max(c.miner_alpha_per_period for c in found)
                 for task, found in (view.ceilings.items() if view is not None else ()) if found}
        unsettled, unbounded = unsettled_rao(rails, used["settled"], pay_ceilings={} if view is None
                                             else view.pay_ceilings, rates=rates, now=now, days=unsettled_days)
        cannot_know += [f"collector-rail task {task} has no frozen rate or cap to bound its unsettled work"
                        for task in unbounded]
        subtensor = await _subtensor(network)
        config = PayoutConfig(netuid=netuid, collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey,
                              min_transfer_rao=0, reserve_rao=reserve_rao, era_period=era_period)
        chain = _payout_chain(subtensor, netuid=netuid, collector_coldkey=collector_coldkey,
                              collector_hotkey=collector_hotkey)
        signer = (_wallet_signer(subtensor, wallet_name=wallet_name, wallet_path=wallet_path, netuid=netuid,
                                 collector_hotkey=collector_hotkey) if execute else None)
        try:
            receipt = await run_burn(ledger.obligations, journal=journal, chain=chain, config=config,
                                     collector_tasks=rails, now=now, cap_rao=cap_rao, unsettled=unsettled,
                                     cannot_know=cannot_know, execute=execute, signer=signer,
                                     confirm_rao=confirm_rao, wait_seconds=wait_seconds)
        except (Exception, asyncio.CancelledError) as exc:
            if execute:
                partial = incomplete_receipt(journal, config=config, pass_day=utc_day(now), now=now, execute=True,
                                             error=exc, schema=SURPLUS_RECEIPT_SCHEMA)
                write_receipt(partial, receipts, prefix="burn")
                journal.mark_receipted(partial["reported_batch_ids"])
            raise
        used.pop("settled", None)
        receipt["sources"] = {**used, "unsettled_days": unsettled_days, "as_of_supplied": as_of_supplied,
                              "journal": str(journal_path), "paid_tasks": sorted(paid_tasks)}
        path = write_receipt(receipt, receipts, prefix="burn")
        if execute:
            journal.mark_receipted(receipt["reported_batch_ids"])
    finally:
        journal.close()
        if subtensor is not None:
            from reliquary.infrastructure.chain import close_subtensor

            await close_subtensor(subtensor)
    return receipt, path


def render_burn(receipt, *, capped: bool) -> str:
    from reliquary.ledger.obligations import RAO_PER_ALPHA

    def alpha(rao) -> str:
        return "?" if rao is None else f"{int(rao) / RAO_PER_ALPHA:.9f}"

    lines = [f"{receipt['mode'].upper()} surplus burn (collector {receipt['collector_coldkey']})"
             + (" INCOMPLETE" if receipt.get("incomplete") else "")]
    if "collector_stake_rao" in receipt:
        nonce = receipt.get("nonce") or {}
        lines += [f"  stake {alpha(receipt['collector_stake_rao'])} alpha at block {receipt['finalized_block']}: "
                  f"owed {alpha(receipt['owed_rao'])}, awaiting audit {alpha(receipt['unsettled_rao'])}, in flight "
                  f"{alpha(receipt['in_flight_rao'])}, reserve {alpha(receipt['reserve_rao'])}, surplus "
                  f"{alpha(receipt['surplus_rao'])}; nonce {nonce.get('chain')} on chain, {nonce.get('journal')} "
                  "expected by the journal",
                  f"  already burned on {receipt['pass_day']} (UTC) {alpha(receipt['burned_today_rao'])}; this pass "
                  f"burns {alpha(receipt['planned_rao'])} alpha ({receipt['planned_rao']} rao)"
                  + ("" if capped else " -- no --cap given: nothing")]
    for batch in receipt.get("unreported", []) + receipt.get("recovered", []) + receipt.get("batches", []):
        lines.append(f"  {batch['kind']} {batch['batch_id']} {batch['state']} nonce {batch['nonce']} "
                     f"block {batch['block_number']}: {batch['detail']}")
    lines.append(f"  burned {alpha(receipt['burned']['rao'])} alpha; paid {receipt['paid']['transfers']} "
                 f"transfer(s) settled here")
    if receipt.get("stopped"):
        lines.append(f"  STOPPED: {receipt['stopped']}")
    elif receipt["mode"] != "execute" and receipt.get("planned_rao") and capped:
        lines.append(f"  to burn it: --execute --confirm-rao {receipt['planned_rao']}")
    return "\n".join(lines)


@ledger_app.command("collector-status")
def collector_status_command(
    collector_hotkey: str = typer.Option(..., "--collector-hotkey", help="the collector's hotkey (SS58)"),
    collector_coldkey: str = typer.Option(None, "--collector-coldkey", help="its coldkey, to read its stake"),
    before: float = typer.Option(None, "--before", help="MinerBurned (0 to 1) read just before a tier release"),
    fraction: float = typer.Option(None, "--fraction", help="the tier's fraction (0 to 1), with --before"),
    from_fraction: float = typer.Option(0.0, "--from-fraction", help="the fraction before the release (0 to 1)"),
    network: str = typer.Option("finney", "--network"),
    netuid: int = typer.Option(81, "--netuid"),
) -> None:
    """Read MinerBurned and the collector's UID, coldkey, incentive and stake from the chain. Read-only.

    With --before and --fraction (and --from-fraction for a later tier): whether the release landed."""
    from reliquary.ledger import sources

    if (before is None) != (fraction is None):
        raise typer.BadParameter("--before and --fraction go together")
    for name, value in (("--before", before), ("--fraction", fraction), ("--from-fraction", from_fraction)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise typer.BadParameter(f"{name} is a share from 0 to 1 (1 % is 0.01), got {value}")
    if fraction is not None and from_fraction >= 1.0:
        raise typer.BadParameter("--from-fraction 1 leaves nothing burning: MinerBurned cannot tell a tier")

    async def read():
        subtensor = await _subtensor(network)
        try:
            return await sources.read_collector_status(subtensor, netuid, collector_hotkey=collector_hotkey,
                                                       collector_coldkey=collector_coldkey)
        finally:
            from reliquary.infrastructure.chain import close_subtensor

            await close_subtensor(subtensor)

    status = asyncio.run(read())
    typer.echo(f"block {status['block']} (last epoch {status['last_step']}, tempo {status['tempo']}): MinerBurned "
               f"{status['miner_burned']:.6f} ({status['miner_burned']:.4%}), stake transfers "
               f"{'open' if status['transfer_toggle'] else 'CLOSED'}")
    lag = 1 + (status["commit_reveal_period"] if status["commit_reveal"] else 0)
    typer.echo(f"commit-reveal {'on, period ' + str(status['commit_reveal_period']) if status['commit_reveal'] else 'off'}"
               f": a release shows in MinerBurned from the {lag}th tempo after every validator runs it; "
               f"immunity period {status['immunity_period']} blocks")
    if status["uid"] is None:
        typer.echo(f"collector {collector_hotkey}: NOT REGISTERED on netuid {netuid}")
    else:
        immune_until = status["registered_at"] + status["immunity_period"]
        typer.echo(f"collector {collector_hotkey}: UID {status['uid']}, coldkey {status['coldkey']}"
                   + (f" (THE OWNER'S: {', '.join(status['held_by_owner'])}; the chain keeps its incentive)"
                      if status["held_by_owner"] else "")
                   + f", incentive {status['incentive']:.6f}, emission {status['emission_alpha']:.6f} alpha last epoch, "
                     f"registered at block {status['registered_at']} (immune until about block {immune_until})")
    if status["stake_rao"] is not None:
        typer.echo(f"collector stake: {status['stake_rao'] / 1e9:.9f} alpha")
    if before is not None:
        check = sources.tier_check(before, fraction, status["miner_burned"], from_fraction=from_fraction,
                                   collector_incentive=status["incentive"])
        share = ("" if check["collector_share"] is None else
                 f"; the collector's part of the unpaid share is {check['collector_share']:.4f} (expected "
                 f"{fraction:.4f}): {'ok' if check['collector_share_within'] else 'off'}")
        typer.echo(f"tier {from_fraction:.4f} -> {fraction:.4f}: MinerBurned expected {check['expected']:.6f} (from "
                   f"{before:.6f}), now {check['now']:.6f}" + share + " -> "
                   + ("AS EXPECTED" if check["landed"] else f"NOT YET: read it again after tempo {lag}"))


@ledger_app.command("journal-init")
def journal_init_command(
    collector_coldkey: str = typer.Option(..., "--collector-coldkey", help="the collector's coldkey (SS58)"),
    collector_hotkey: str = typer.Option(..., "--collector-hotkey", help="the collector's hotkey (SS58)"),
    reason: str = typer.Option(..., "--reason", help="why the journal starts, or is re-aligned, here: kept in it"),
    resync: bool = typer.Option(False, "--resync", help="re-align an initialised journal to the chain's nonce, "
                                                        "once what the collector's key signed outside it is "
                                                        "checked"),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal"),
    receipts: Path = typer.Option(DEFAULT_RECEIPTS, "--receipts", help="the local receipts, checked against the "
                                                                       "journal"),
    network: str = typer.Option("archive", "--network"),
    netuid: int = typer.Option(81, "--netuid"),
) -> None:
    """Start the payer's journal at the collector's nonce on chain, or re-align it (--resync).

    Refused while the bucket's published receipts, or the local ones, list a paid batch the journal does not
    hold: that journal is not the one that paid it (restore the one that did). It reads the chain and the
    bucket, and writes only the journal."""
    result = asyncio.run(collect_journal_init(
        collector_coldkey=collector_coldkey, collector_hotkey=collector_hotkey, reason=reason, resync=resync,
        journal_path=journal.expanduser(), receipts=receipts.expanduser(), network=network, netuid=netuid))
    typer.echo(result)


async def collect_journal_init(*, collector_coldkey: str, collector_hotkey: str, reason: str, resync: bool,
                               journal_path: Path, receipts: Path, network: str, netuid: int,
                               published=None) -> str:
    from reliquary.ledger.payout import Journal, PayoutError
    from reliquary.ledger.receipts import local_paid_hashes, published_batches

    paid = set(await published_batches()) if published is None else set(published)
    paid |= local_paid_hashes(receipts)
    subtensor = await _subtensor(network)
    try:
        chain = _payout_chain(subtensor, netuid=netuid, collector_coldkey=collector_coldkey,
                              collector_hotkey=collector_hotkey)
        head = await chain.finalized_block()
        nonce = await chain.collector_nonce(head)
    finally:
        from reliquary.infrastructure.chain import close_subtensor

        await close_subtensor(subtensor)
    journal = Journal(journal_path)
    try:
        unknown = sorted(paid - journal.known_hashes())
        if unknown:
            raise typer.BadParameter(
                f"{len(unknown)} paid batch(es) in the published or local receipts are not in {journal_path} "
                f"(first: {unknown[0]}): it is not the journal that paid them. Restore the one that did")
        try:
            if resync:
                before = journal.resync(nonce=nonce, reason=reason)
                return f"journal {journal_path} re-aligned: expects nonce {nonce} (was {before}) at block {head}"
            journal.initialise(collector_coldkey=collector_coldkey, netuid=netuid, nonce=nonce, reason=reason)
            return f"journal {journal_path} initialised at the collector's nonce {nonce} (block {head})"
        except PayoutError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        journal.close()


@ledger_app.command("settle")
def settle_command(
    batch_id: str = typer.Argument(..., help="the batch that stopped the payer"),
    landed: bool = typer.Option(None, "--landed/--not-landed", help="required: whether the batch's transfers "
                                                                    "were made, as read on the chain"),
    evidence: str = typer.Option(..., "--evidence", help="what was read: the extrinsic, its block, its events"),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal"),
) -> None:
    """Decide a batch that stopped the payer (foreign, unknown or mismatch), once the chain is read.

    --landed marks its obligations paid; --not-landed makes them due again. There is no default."""
    from reliquary.ledger.payout import Journal, PayoutError

    if landed is None:
        raise typer.BadParameter("say --landed or --not-landed: there is no default")
    handle = Journal(journal.expanduser())
    try:
        handle.settle_by_hand(batch_id, landed=landed, evidence=evidence)
    except PayoutError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        handle.close()
    typer.echo(f"batch {batch_id} settled as {'landed' if landed else 'paying nothing'}")


@ledger_app.command("clear-held")
def clear_held_command(
    coldkey: str = typer.Option(None, "--coldkey", help="a coldkey held after failed transfers"),
    obligation: list[str] = typer.Option([], "--obligation", help="an obligation id (repeatable)"),
    evidence: str = typer.Option(..., "--evidence", help="why the destination can be paid now"),
    journal: Path = typer.Option(DEFAULT_JOURNAL, "--journal"),
) -> None:
    """Plan again the obligations of a coldkey held after failed transfers, once its problem is fixed."""
    from reliquary.ledger.payout import Journal, PayoutError

    handle = Journal(journal.expanduser())
    try:
        cleared = handle.clear_held(evidence=evidence, coldkey=coldkey, obligation_ids=obligation)
    except PayoutError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        handle.close()
    typer.echo(f"{cleared} obligation(s) cleared")


@ledger_app.command("find-nonce")
def find_nonce_command(
    collector_coldkey: str = typer.Option(..., "--collector-coldkey"),
    nonce: int = typer.Option(..., "--nonce"),
    from_block: int = typer.Option(..., "--from-block"),
    to_block: int = typer.Option(..., "--to-block"),
    network: str = typer.Option("archive", "--network"),
    netuid: int = typer.Option(81, "--netuid"),
) -> None:
    """Which extrinsic the collector's coldkey signed at a nonce, in a range of blocks. Read-only."""
    async def find():
        subtensor = await _subtensor(network)
        try:
            chain = _payout_chain(subtensor, netuid=netuid, collector_coldkey=collector_coldkey,
                                  collector_hotkey=collector_coldkey)
            return await chain.find_nonce(nonce, from_block, to_block)
        finally:
            from reliquary.infrastructure.chain import close_subtensor

            await close_subtensor(subtensor)

    found = asyncio.run(find())
    typer.echo(json.dumps(found, indent=1, sort_keys=True) if found else
               f"no extrinsic of {collector_coldkey} at nonce {nonce} in blocks {from_block}-{to_block}")


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
        return "?" if rao is None else f"{int(rao) / RAO_PER_ALPHA:.9f}"

    def count(name) -> int:
        return int((receipt.get(name) or {}).get("obligations") or 0)

    lines = [f"{receipt['mode'].upper()} pass of {receipt['pass_day']} (collector {receipt['collector_coldkey']})"
             + (" INCOMPLETE" if receipt.get("incomplete") else "")]
    if "collector_stake_rao" in receipt:
        nonce = receipt.get("nonce") or {}
        lines.append(f"  stake {alpha(receipt['collector_stake_rao'])} alpha at block {receipt['finalized_block']}, "
                     f"in flight {alpha(receipt['in_flight_rao'])}, available {alpha(receipt['available_rao'])}; "
                     f"nonce {nonce.get('chain')} on chain, {nonce.get('journal')} expected by the journal")
        lines.append(f"  floor {alpha(receipt['min_transfer_rao'])} alpha per destination "
                     f"({receipt.get('min_transfer_source', 'given')}), batches of {receipt['batch_size']}")
        lines.append(f"  due {receipt['due']['obligations']} ({alpha(receipt['due']['rao'])} alpha); paid by the "
                     f"weights, never here: {receipt['native_rail']['obligations']} "
                     f"({alpha(receipt['native_rail']['rao'])})")
        held = receipt.get("held") or {}
        lines.append(f"  planned {receipt['planned']['transfers']} transfer(s), {alpha(receipt['planned']['rao'])} "
                     f"alpha, plan {receipt.get('plan_digest')}; deferred under the floor "
                     f"{count('deferred_below_minimum')}, unfunded {count('unfunded')}, ownerless {count('ownerless')}")
        lines.append("  held for a person: " + ", ".join(f"{name.replace('_', ' ')} {int(v.get('obligations') or 0)}"
                                                         for name, v in held.items())
                     + f"; due in the journal but in no archive {count('journal_only')}")
    for batch in receipt.get("unreported", []) + receipt.get("recovered", []) + receipt.get("batches", []):
        lines.append(f"  batch {batch['batch_id']} {batch['state']} nonce {batch['nonce']} "
                     f"block {batch['block_number']}: {batch['detail']}")
    lines.append(f"  paid {receipt['paid']['transfers']} transfer(s), {alpha(receipt['paid']['rao'])} alpha, "
                 f"fees {receipt['paid']['fee_tao']:.9f} TAO")
    if receipt.get("stopped"):
        lines.append(f"  STOPPED: {receipt['stopped']}")
    elif receipt["mode"] != "execute" and receipt.get("planned", {}).get("rao"):
        lines.append(f"  to pay this plan: --execute --confirm-rao {receipt['planned']['rao']} "
                     f"--confirm-plan {receipt['plan_digest']}")
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


__all__ = ["collect_burn", "collect_journal_init", "collect_payout", "collect_report", "ledger_app", "render_burn",
           "render_payout"]
