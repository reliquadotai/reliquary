"""Read-only simulation of replacement debits against attested collector balances.

A period archive may carry a list named `replacement_debits_v1`. Each entry
has event_id (stable across retries and requeues), hotkey, coldkey_at_earn,
reason, assessed_rao and recovered_at_settlement_rao. The last two fields are
nonnegative integer rao of alpha. Settlement records the amount actually
withheld after caps and rounding. The archive supplies task_id and work_period.
An absent list means the archive declares no debit.

The snapshot provides owners (hotkey, work_period, coldkey), balances
(obligation_id, rail, state, retained_rao) and all previous applications
(event_id, obligation_id, amount_rao) for the replay. Each owner is the owner
at the end of that work period. An omitted balance supplies no funds.
unrecovered_rao is the residual if the proposals were applied.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta

from .obligations import HELD, STATES, ArchiveRef, Obligation, archive_parts, origin_entry, work_day


def _text(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty, trimmed string")
    return value


def _whole(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _rows(document, field):
    rows = document.get(field)
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{field} must be a list of objects")
    return rows


@dataclass(frozen=True)
class _Event:
    event_id: str
    task_id: str
    period: int
    hotkey: str
    coldkey: str
    assessed: int
    settled: int
    reason: str


def _events(archives, owners):
    by_key, by_origin, events = {}, {}, {}
    for archive in archives:
        doc = archive.document
        period = _whole(archive.work_period, "work_period")
        key = (archive.task_id, period, archive.entry_period)
        if ("task_id" in doc and _text(doc, "task_id") != archive.task_id
                or "work_period" in doc and _whole(doc["work_period"], "work_period") != period
                or "entry_period" in doc and _whole(doc["entry_period"], "entry_period") != archive.entry_period):
            raise ValueError("archive document identity differs from its key")
        if key in by_key:
            if by_key[key].document != doc:
                raise ValueError("conflicting documents for one archive key")
            continue
        by_key[key] = archive
        event_ids = set()
        rows = _rows(doc, "replacement_debits_v1") if "replacement_debits_v1" in doc else ()
        for row in rows:
            if ("task_id" in row and row["task_id"] != archive.task_id
                    or "work_period" in row and _whole(row["work_period"], "work_period") != period):
                raise ValueError("debit origin differs from archive")
            event = _Event(_text(row, "event_id"), archive.task_id, period, _text(row, "hotkey"),
                           _text(row, "coldkey_at_earn"), _whole(row.get("assessed_rao"), "assessed_rao"),
                           _whole(row.get("recovered_at_settlement_rao"), "recovered_at_settlement_rao"),
                           _text(row, "reason"))
            if owners.get((event.hotkey, period)) != event.coldkey:
                raise ValueError(f"missing or conflicting earn-time owner for event {event.event_id}")
            if event.settled > event.assessed:
                raise ValueError(f"settlement recovery exceeds assessment for {event.event_id}")
            if events.setdefault(event.event_id, event) != event:
                raise ValueError(f"conflicting archive evidence for {event.event_id}")
            event_ids.add(event.event_id)
        origin = (archive.task_id, period, origin_entry(archive))
        if by_origin.setdefault(origin, event_ids) != event_ids:
            raise ValueError("replacement debits changed across archive requeue")

    active = {}
    for archive, _ in archive_parts(by_key.values()):
        for eid in by_origin[archive.task_id, archive.work_period, origin_entry(archive)]:
            active[eid] = events[eid]
    return active


def simulate_replacement_debits(archives: Iterable[ArchiveRef], obligations: Iterable[Obligation],
                                snapshot: Mapping) -> dict:
    """Allocate against a single coherent snapshot without changing any input.

    Owners attest to the end of the work period. Each held collector balance is
    the remaining envelope after external payments, before J+2 applications.
    Carry all prior applications for these events and balances, including on
    already-paid obligations. Each event/obligation pair is applied once.
    """
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot must be an object")
    owners = {}
    for row in _rows(snapshot, "owners"):
        key = (_text(row, "hotkey"), _whole(row.get("work_period"), "work_period"))
        coldkey = _text(row, "coldkey")
        if owners.setdefault(key, coldkey) != coldkey:
            raise ValueError("conflicting earn-time owners")
    events = _events(archives, owners)

    gross = {}
    for obligation in obligations:
        if gross.setdefault(obligation.obligation_id, obligation) != obligation:
            raise ValueError(f"conflicting obligation {obligation.obligation_id}")

    balances = {}
    available = {}
    candidates = defaultdict(list)
    for row in _rows(snapshot, "balances"):
        oid = _text(row, "obligation_id")
        if oid in balances:
            raise ValueError(f"duplicate balance {oid}")
        rail, state = _text(row, "rail"), _text(row, "state")
        retained = _whole(row.get("retained_rao"), "retained_rao")
        if rail not in ("collector", "native") or state not in (*STATES, "paid"):
            raise ValueError(f"unknown rail or balance state for {oid}")
        balances[oid] = rail, state
        if rail == "native":
            continue
        if oid not in gross:
            raise ValueError(f"unknown collector obligation {oid}")
        obligation = gross[oid]
        if retained > _whole(obligation.amount_rao, "amount_rao"):
            raise ValueError(f"retained amount exceeds gross amount for {oid}")
        if state == HELD:
            owner = owners.get((obligation.hotkey, obligation.work_period))
            if owner is None:
                raise ValueError(f"missing earn-time owner for {oid}")
            candidates[owner, obligation.work_day].append(oid)
            available[oid] = retained

    for ids in candidates.values():
        ids.sort(key=lambda oid: (gross[oid].work_period, oid))

    previous = defaultdict(int)
    used = defaultdict(int)
    applied_pairs = set()
    for row in _rows(snapshot, "applications"):
        eid, oid = _text(row, "event_id"), _text(row, "obligation_id")
        amount = _whole(row.get("amount_rao"), "amount_rao")
        if amount == 0 or (eid, oid) in applied_pairs:
            raise ValueError(f"empty or duplicate prior application for {eid}/{oid}")
        if eid not in events or oid not in balances:
            raise ValueError(f"unknown prior application for {eid}/{oid}")
        if balances[oid][0] != "collector":
            raise ValueError(f"prior application on native balance {oid}")
        event, obligation = events[eid], gross[oid]
        age_days = (obligation.work_day - work_day(event.period)).days
        if (age_days not in (0, 1)
                or owners.get((obligation.hotkey, obligation.work_period)) != event.coldkey):
            raise ValueError(f"ineligible prior application for {eid}/{oid}")
        previous[eid] += amount
        used[oid] += amount
        if previous[eid] > event.assessed - event.settled or used[oid] > obligation.amount_rao:
            raise ValueError(f"prior application exceeds assessment or obligation for {eid}/{oid}")
        if oid in available:
            if amount > available[oid]:
                raise ValueError(f"prior application exceeds retained balance for {oid}")
            available[oid] -= amount
        applied_pairs.add((eid, oid))

    results, proposals = [], []
    for event in sorted(events.values(), key=lambda event: (event.period, event.event_id)):
        eid = event.event_id
        residual = event.assessed - event.settled - previous[eid]
        remaining = residual
        day = work_day(event.period)
        for eligible_day in (day, day + timedelta(days=1)):
            for oid in candidates.get((event.coldkey, eligible_day), ()):
                if not remaining:
                    break
                if (eid, oid) in applied_pairs:
                    continue
                amount = min(remaining, available[oid])
                if amount:
                    proposals.append({"event_id": eid, "obligation_id": oid, "amount_rao": amount})
                    available[oid] -= amount
                    remaining -= amount
        results.append({"event_id": eid, "assessed_rao": event.assessed,
                        "recovered_at_settlement_rao": event.settled, "previously_applied_rao": previous[eid],
                        "outstanding_before_proposal_rao": residual, "proposed_rao": residual - remaining,
                        "unrecovered_rao": remaining})
    return {"events": results, "proposed_applications": proposals}
