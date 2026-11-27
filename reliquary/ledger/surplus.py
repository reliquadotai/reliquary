"""Burning the collector's surplus: ``burn_alpha`` of what it holds beyond what it owes and keeps.

The collector receives the share of the miners' emission the weights do not
pay. While every order is on the native rail it owes nothing by transfer, and
what it holds beyond its operating reserve is surplus. Once orders are on the
collector rail, what it owes them comes first: every obligation earned and not
yet paid (earned, held or due), and every batch in flight.

    surplus = stake at the finalized head - owed - in flight - reserve  (never below 0)
    burned  = min(surplus, cap - already burned that UTC day)

``cap`` is the most burned in one UTC day, an owner setting, and must be
named to execute; a second run on the same day burns only what the cap has
left. The burn is one batch of the payer's journal (``payout.KIND_BURN``),
so it takes the same care as a payout:

- the nonce, mortal era and extrinsic hash are written before it is sent;
- every batch or burn of unknown fate is settled from the chain before
  anything new is sent, and one still in its window holds every new send;
- it is a dry run by default, and executing needs ``--confirm-rao`` equal to
  the exact amount the dry run printed;
- each pass writes a receipt, with the ``AlphaBurned`` event (coldkey,
  hotkey, amount, netuid) of the burn once final.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Collection, Iterable
from datetime import date

from reliquary.ledger.obligations import DUE, FINALIZED, RAO_PER_ALPHA, Obligation
from reliquary.ledger.payout import (
    BATCH_EXPIRED,
    BATCH_FAILED,
    BATCH_FINALIZED,
    BATCH_FOREIGN,
    BATCH_PENDING,
    KIND_BURN,
    MODE_DRY_RUN,
    MODE_EXECUTE,
    Journal,
    PayoutChain,
    PayoutConfig,
    PayoutError,
    Signer,
    _batch_receipt,
    _settle,
    recover,
    resolve_batch,
)

logger = logging.getLogger(__name__)

SURPLUS_RECEIPT_SCHEMA = "reliquary/surplus-burn-receipt/v1"


def owed_rao(obligations: Iterable[Obligation], journal: Journal, collector_tasks: Collection[str], now: float,
             would: dict[str, str] | None = None) -> int:
    """What the collector still owes collector-rail work: every obligation
    earned by ``now`` of a collector-rail task that the journal does not hold
    as paid or in flight (in-flight ones count apart, in ``in_flight_rao``).
    ``would``: a dry run's view of batches it would settle (released ones owe
    again, paid ones no more)."""
    tasks = {str(task) for task in collector_tasks}
    mine = [o for o in obligations if o.task_id in tasks and o.earned_at <= now]
    rows = journal.rows(o.obligation_id for o in mine)
    would = would or {}
    owed = 0
    for obligation in mine:
        row = rows.get(obligation.obligation_id)
        state = DUE if row is None else row["state"]
        settled = would.get(row["batch_id"]) if row is not None else None
        if settled == BATCH_FINALIZED:
            state = FINALIZED
        elif settled in (BATCH_FAILED, BATCH_EXPIRED):
            state = DUE
        if state == DUE:
            owed += obligation.amount_rao
    return owed


async def run_burn(obligations: Iterable[Obligation], *, journal: Journal, chain: PayoutChain, config: PayoutConfig,
                   collector_tasks: Collection[str], pass_day: date, now: float, cap_rao: int,
                   execute: bool = False, signer: Signer | None = None, confirm_rao: int | None = None,
                   wait_seconds: float = 600.0, poll_seconds: float = 12.0, clock=time.time,
                   sleep=asyncio.sleep) -> dict:
    """One surplus burn: settle what is unknown, compute the surplus, and
    (``execute``, with ``confirm_rao`` naming the amount) burn up to what
    ``cap_rao`` (the most per UTC day of ``pass_day``) has left.
    Returns the receipt. A dry run writes nothing and signs nothing."""
    if isinstance(cap_rao, bool) or not isinstance(cap_rao, int) or cap_rao < 0:
        raise PayoutError(f"the cap must be a whole number of rao >= 0, got {cap_rao!r}")
    if execute:
        if signer is None or signer.coldkey != config.collector_coldkey:
            raise PayoutError("burning needs the collector's signer")
        if journal.read_only:
            raise PayoutError("burning needs the journal open for writing")
    receipt: dict = {"schema": SURPLUS_RECEIPT_SCHEMA, "mode": MODE_EXECUTE if execute else MODE_DRY_RUN,
                     "pass_day": pass_day.isoformat(), "as_of": float(now), **config.to_dict(), "cap_rao": cap_rao,
                     "confirmed_rao": confirm_rao, "batches": []}
    known = await recover(journal, chain, config, execute=execute)
    receipt["recovered"], receipt["stopped"] = known["recovered"], known["stopped"]
    head = await chain.finalized_block()
    stake = await chain.collector_stake_rao(head)
    in_flight = journal.in_flight_rao() - known["would_rao"]
    owed = owed_rao(obligations, journal, collector_tasks, now, known["would"])
    surplus = max(0, stake - owed - in_flight - config.reserve_rao)
    released = {batch_id for batch_id, state in known["would"].items() if state != BATCH_FINALIZED}
    today = journal.burned_on(pass_day, excluding=released)
    amount = min(surplus, max(0, cap_rao - today))
    receipt.update({"finalized_block": head, "collector_stake_rao": stake, "owed_rao": owed,
                    "in_flight_rao": in_flight, "surplus_rao": surplus, "burned_today_rao": today,
                    "planned_rao": amount, "planned_alpha": amount / RAO_PER_ALPHA})
    if receipt["stopped"] is None and known["pending"]:
        receipt["stopped"] = (f"{known['pending']} batch(es) sent earlier may still land: nothing new is sent until "
                              "they are final or expired")
    if receipt["stopped"] is not None or not execute or amount <= 0:
        return _close(receipt)
    if confirm_rao != amount:
        receipt["stopped"] = (f"this burn would destroy {amount} rao ({amount / RAO_PER_ALPHA:.9f} alpha): confirm "
                              f"that exact amount with --confirm-rao {amount}")
        return _close(receipt)
    batch_id = f"burn-{pass_day.isoformat()}-{uuid.uuid4().hex[:12]}"
    journal.reserve_burn(batch_id, pass_day, amount)
    tip = await chain.finalized_block()
    nonce = await chain.collector_nonce(tip)
    signed = await signer.sign_burn(amount, nonce=nonce, era_block=tip, era_period=config.era_period)
    journal.mark_submitted(batch_id, signed)
    try:
        await chain.submit(signed)
    except Exception as exc:  # noqa: BLE001 - the send's fate is read from the chain, never assumed
        logger.warning("surplus burn %s: send raised %r; reading the chain for its fate", batch_id, exc)
    deadline = clock() + wait_seconds
    while True:
        batch = journal.batch(batch_id)
        state, inclusion, detail = await resolve_batch(batch, chain)
        if state != BATCH_PENDING:
            receipt["batches"].append(_settle(journal, batch, state, inclusion, detail, config))
            if state == BATCH_FOREIGN:
                receipt["stopped"] = detail
            break
        if clock() >= deadline:
            receipt["batches"].append(_batch_receipt(batch, state, None, detail))
            receipt["stopped"] = f"burn {batch_id} not final after {wait_seconds:.0f} s: the next pass settles it"
            break
        await sleep(poll_seconds)
    return _close(receipt, await chain.collector_stake_rao(await chain.finalized_block()))


def _close(receipt: dict, stake_after: int | None = None) -> dict:
    burned = [b for b in receipt["batches"] + receipt["recovered"]
              if b["state"] == BATCH_FINALIZED and b.get("kind") == KIND_BURN]
    receipt["burned"] = {"burns": len(burned), "rao": sum(_burned_rao(b) for b in burned)}
    receipt["burned"]["alpha"] = receipt["burned"]["rao"] / RAO_PER_ALPHA
    receipt["collector_stake_after_rao"] = stake_after
    return receipt


def _burned_rao(batch: dict) -> int:
    """The alpha the chain says it burned (its ``AlphaBurned`` event), else the amount asked."""
    for event in batch.get("events") or []:
        if event.get("module") == "SubtensorModule" and event.get("event") == "AlphaBurned":
            attributes = event.get("attributes")
            value = attributes.get("amount") if isinstance(attributes, dict) else (
                attributes[2] if isinstance(attributes, (list, tuple)) and len(attributes) >= 3 else None)
            try:
                return int(value)
            except (TypeError, ValueError):
                break
    return int(batch.get("burn_rao") or 0)


__all__ = ["SURPLUS_RECEIPT_SCHEMA", "owed_rao", "run_burn"]
