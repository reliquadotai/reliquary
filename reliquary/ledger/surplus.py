"""Burning the collector's surplus: ``burn_alpha`` of what it holds beyond what it owes and keeps.

The collector receives the share of the miners' emission the weights do not
pay. While every order is on the native rail it owes nothing by transfer, and
what it holds beyond its operating reserve is surplus. Once orders are on the
collector rail, what it owes them comes first:

    surplus = stake at the finalized head - owed - unsettled - in flight - reserve  (never below 0)
    burned  = min(surplus, cap - already burned that UTC day)

- ``owed``: every obligation of a collector-rail task, from the rail's first
  work period on (its archives read over the task's whole history), earned
  and not yet paid or in flight, at the larger of its ledger amount and the
  amount first reserved; and every obligation the journal holds as due whose
  archive the ledger no longer lists.
- ``unsettled``: what collector-rail work still awaiting its audit may owe.
  For each work period of the last ``unsettled_days`` with no archive yet,
  the most the period may pay: the task's pay ceiling at its frozen rate.
  The collector receives that work's share every tempo, long before it
  settles; burning it would leave the obligations unfunded when they come.
- ``in flight``: transfers and burns sent whose fate is not known.
- ``cap``: the most burned per UTC day, an owner setting, needed to execute.
  The day is the clock's when the burn is reserved, never a day given: a
  second run on the same day burns only what the cap has left.

``burn_alpha`` burns only the signer's own stake: the alpha miners received
on the collector's hotkey (their own positions) is never burned.

Nothing is burned when what is owed cannot be known: the registry could not
be read, a collector-rail archive could not be valued, a task the journal
paid is no longer on the rail, or a collector-rail task has no frozen rate.

The burn is one batch of the payer's journal (``payout.KIND_BURN``), with
the same care as a payout: the journal must be initialised and match the
collector's nonce on chain; every batch or burn of unknown fate is settled
first, and one still in its window holds every new send; the nonce, mortal
era and extrinsic hash are written before it is sent; the collector must
not be the subnet owner's; a dry run by default, and executing needs
``confirm_rao`` equal to the amount the dry run printed. A burn whose
``AlphaBurned`` event is missing or does not match is counted as burning
nothing and stops every send until a person settles it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import date

from reliquary.ledger.obligations import DUE, RAO_PER_ALPHA, Obligation, amount_rao, utc_day
from reliquary.ledger.payout import (
    MODE_DRY_RUN,
    MODE_EXECUTE,
    NEEDS_PERSON,
    Journal,
    PayoutChain,
    PayoutConfig,
    PayoutError,
    Signer,
    _row_receipt,
    check_collector,
    check_journal,
    close_receipt,
    collector_rails,
    recover,
    send_batch,
    stake_after,
)
from reliquary.validator import corpus_periods as cp

logger = logging.getLogger(__name__)

SURPLUS_RECEIPT_SCHEMA = "reliquary/surplus-burn-receipt/v1"
# How far back work awaiting its audit is held back from the burn (the ledger escalates older work to a person).
DEFAULT_UNSETTLED_DAYS = 7
# An execution burns as of the clock: an instant further from it is refused.
CLOCK_SLACK_SECONDS = 300.0


def owed_rao(obligations: Iterable[Obligation], journal: Journal, collector_tasks: Mapping[str, int] | Sequence[str],
             now: float) -> int:
    """What the collector still owes collector-rail work (in-flight transfers
    count apart, in ``in_flight_rao``)."""
    rails = collector_rails(collector_tasks)
    obligations = list(obligations)
    mine = [o for o in obligations if o.task_id in rails and o.work_period >= rails[o.task_id]
            and o.earned_at <= now]
    rows = journal.rows(o.obligation_id for o in mine)
    owed = 0
    for obligation in mine:
        row = rows.get(obligation.obligation_id)
        if row is None:
            owed += obligation.amount_rao
        elif row["state"] == DUE:
            owed += max(obligation.amount_rao, int(row["amount_rao"]))
    listed = {o.obligation_id for o in obligations}
    owed += sum(int(row["amount_rao"]) for oid, row in journal.due_rows().items() if oid not in listed)
    return owed


def unsettled_rao(collector_tasks: Mapping[str, int], settled: Iterable[tuple[str, int]], *,
                  pay_ceilings: Mapping[str, float], rates: Mapping[str, float], now: float,
                  days: float = DEFAULT_UNSETTLED_DAYS) -> tuple[int, list[str]]:
    """(What collector-rail work awaiting its audit may still owe, the tasks it
    cannot be bounded for): for each task, each work period from the rail's
    first, over the last ``days`` up to the current one, with no settled
    archive, at the task's pay ceiling and frozen rate."""
    have = {(str(task), int(work)) for task, work in settled}
    current = cp.period_of(now)
    first = cp.period_of(now - days * 86_400)
    total, unbounded = 0, []
    for task, start in sorted(collector_tasks.items()):
        ceiling, rate = pay_ceilings.get(task), rates.get(task)
        open_periods = [w for w in range(max(int(start), first), current + 1) if (task, w) not in have]
        if not open_periods:
            continue
        if ceiling is None or rate is None:
            unbounded.append(task)
            continue
        total += len(open_periods) * amount_rao(ceiling, rate)
    return total, unbounded


async def run_burn(obligations: Iterable[Obligation], *, journal: Journal, chain: PayoutChain, config: PayoutConfig,
                   collector_tasks: Mapping[str, int] | Sequence[str], now: float, cap_rao: int,
                   unsettled: int = 0, cannot_know: Sequence[str] = (), execute: bool = False,
                   signer: Signer | None = None, confirm_rao: int | None = None, wait_seconds: float = 600.0,
                   poll_seconds: float = 12.0, clock=time.time, sleep=asyncio.sleep) -> dict:
    """One surplus burn: settle what is unknown, check the journal and the
    collector, compute the surplus, and (``execute``, with ``confirm_rao``
    naming the amount) burn up to what ``cap_rao`` has left of the clock's UTC
    day. ``unsettled``: what work awaiting its audit may owe (``unsettled_rao``);
    ``cannot_know``: why what is owed is not known (each stops the burn).
    Returns the receipt. A dry run works on a snapshot of the journal: the
    file is never written and nothing is signed."""
    if isinstance(cap_rao, bool) or not isinstance(cap_rao, int) or cap_rao < 0:
        raise PayoutError(f"the cap must be a whole number of rao >= 0, got {cap_rao!r}")
    if execute:
        if signer is None or signer.coldkey != config.collector_coldkey:
            raise PayoutError("burning needs the collector's signer")
        if journal.read_only:
            raise PayoutError("burning needs the journal open for writing")
        if abs(float(now) - clock()) > CLOCK_SLACK_SECONDS:
            raise PayoutError("an execution burns as of the clock's time, not another")
        return await _burn(obligations, journal=journal, chain=chain, config=config, collector_tasks=collector_tasks,
                           now=now, cap_rao=cap_rao, unsettled=unsettled, cannot_know=cannot_know, execute=True,
                           signer=signer, confirm_rao=confirm_rao, wait_seconds=wait_seconds,
                           poll_seconds=poll_seconds, clock=clock, sleep=sleep)
    snapshot = journal if journal.read_only else journal.snapshot()
    try:
        return await _burn(obligations, journal=snapshot, chain=chain, config=config, collector_tasks=collector_tasks,
                           now=now, cap_rao=cap_rao, unsettled=unsettled, cannot_know=cannot_know, execute=False,
                           signer=None, confirm_rao=confirm_rao, wait_seconds=wait_seconds,
                           poll_seconds=poll_seconds, clock=clock, sleep=sleep)
    finally:
        if snapshot is not journal:
            snapshot.close()


async def _burn(obligations, *, journal: Journal, chain: PayoutChain, config: PayoutConfig, collector_tasks, now: float,
                cap_rao: int, unsettled: int, cannot_know: Sequence[str], execute: bool, signer: Signer | None,
                confirm_rao: int | None, wait_seconds: float, poll_seconds: float, clock, sleep) -> dict:
    rails = collector_rails(collector_tasks)
    # An execution's day is the clock's (``run_burn`` holds ``now`` to it); a dry run may preview another.
    day: date = utc_day(clock()) if execute else utc_day(now)
    receipt: dict = {"schema": SURPLUS_RECEIPT_SCHEMA, "mode": MODE_EXECUTE if execute else MODE_DRY_RUN,
                     "pass_day": day.isoformat(), "as_of": float(now), **config.to_dict(), "cap_rao": cap_rao,
                     "confirmed_rao": confirm_rao, "collector_rails": rails,
                     "unreported": [_row_receipt(b) for b in journal.unreported()], "recovered": [], "batches": [],
                     "stopped": None}
    pending, unreadable = await recover(journal, chain, config, receipt, execute=execute)
    head, expected = await check_journal(journal, chain, config, receipt, pending=pending, unreadable=unreadable)
    await check_collector(chain, config, receipt, head)
    for why in cannot_know:
        if receipt["stopped"] is None:
            receipt["stopped"] = f"what the collector owes cannot be known: {why}. Nothing is burned"
    stake = await chain.collector_stake_rao(head)
    in_flight = journal.in_flight_rao()
    owed = owed_rao(obligations, journal, rails, now)
    surplus = max(0, stake - owed - int(unsettled) - in_flight - config.reserve_rao)
    today = journal.burned_on(day)
    amount = min(surplus, max(0, cap_rao - today))
    receipt.update({"finalized_block": head, "collector_stake_rao": stake, "owed_rao": owed,
                    "unsettled_rao": int(unsettled), "in_flight_rao": in_flight, "surplus_rao": surplus,
                    "burned_today_rao": today, "planned_rao": amount, "planned_alpha": amount / RAO_PER_ALPHA})
    if receipt["stopped"] is not None or not execute or amount <= 0:
        return close_receipt(receipt)
    if confirm_rao != amount:
        receipt["stopped"] = (f"this burn would destroy {amount} rao ({amount / RAO_PER_ALPHA:.9f} alpha): confirm "
                              f"that exact amount with --confirm-rao {amount}")
        return close_receipt(receipt)
    tip = await chain.finalized_block()
    nonce = await chain.collector_nonce(tip)
    if nonce != expected:
        receipt["stopped"] = (f"the collector's nonce is {nonce} at block {tip}; this journal expects {expected}: "
                              "another extrinsic was signed with the collector's key. Nothing is burned")
        return close_receipt(receipt)
    batch_id = f"burn-{day.isoformat()}-{uuid.uuid4().hex[:12]}"
    journal.reserve_burn(batch_id, day, amount)
    signed = await signer.sign_burn(amount, nonce=nonce, era_block=tip, era_period=config.era_period)
    entry, known = await send_batch(journal, chain, config, batch_id, signed, wait_seconds=wait_seconds,
                                    poll_seconds=poll_seconds, clock=clock, sleep=sleep)
    receipt["batches"].append(entry)
    if not known:
        receipt["stopped"] = f"burn {batch_id} {entry['detail']}"
    elif entry["state"] in NEEDS_PERSON:
        receipt["stopped"] = entry["detail"]
    elif entry.get("refused_by_pool"):
        receipt["stopped"] = f"burn {batch_id} was {entry['detail']}: nothing was burned"
    return close_receipt(receipt, await stake_after(chain))


__all__ = ["DEFAULT_UNSETTLED_DAYS", "SURPLUS_RECEIPT_SCHEMA", "owed_rao", "run_burn", "unsettled_rao"]
