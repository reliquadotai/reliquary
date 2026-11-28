#!/usr/bin/env python3
"""Replay private, normalized JSONL observations against the order contract book offline.

Every row has type, at (integer Unix seconds), and a stable event_id. Order events
also have order_id. Events are sorted by time; at one timestamp, source line order
is authoritative. An identical repeated event_id is ignored.

offer    market, hotkey, coldkey, price_micro, rate, slots, [sequence], [source]
         source may be observed, reconstructed or unspecified. Zero rate/slots withdraws.
open     market, duration_s, target_rate, max_tokens, budget_rao, ceiling_micro,
         new_rate_cap, [total_slots], [audit_cost_micro], [evaluation], [q_min], [trust_tokens]
reserve  hotkey, lease_id, max_tokens
cancel   lease_id
delivery hotkey, delivery_id, tokens, [audited_tokens], [observed_price_micro],
         [lease_id], [slot_id]. Supply final accepted work only. Use the same
         delivery_id for retries and distinct slot_id values for final results.
miss     hotkey (observed request without a delivery)
outage   end (verified service interruption starting at at)
exit     hotkey, reason (announced/disappeared/release), [expired_leases], [penalty_rao], [notice_s]
round    eligible_owners (attested hotkey-to-coldkey map for a supplemental round)
rate     hotkey, window_start, evidence_complete, [min_active_s]
debits   work_period, owners_at_period_end, recovered_rao (explicit final recovery by source event_id)
pace     card_tokens_per_second, queued_audit_tokens, audit_fraction, offered_rate, needed_rate
close    [delivered] (true only with observed successful client delivery at at)

Prices are integer micro-alpha per million tokens; tokens times price gives
thousandths of one rao. The report compares the intersection of observed work
and V1 contracts at the observed times. It cannot estimate alternative miners'
unobserved work, realized V1 delivery time or the effectiveness of a new audit draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliquary.corpus.contract_pacing import PaceState, paced_rate
from reliquary.corpus.order_contracts import ContractBook
from reliquary.corpus.replacement_debits import period_debit_rows


def _int(value, field_name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _key(value, field_name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).digest()


@dataclass
class Order:
    order_id: str
    opened_at: int
    deadline: int
    total_slots: int | None
    sources: dict
    counts: Counter = field(default_factory=Counter)
    unmatched: Counter = field(default_factory=Counter)
    slots: dict = field(default_factory=dict)
    deliveries: dict = field(default_factory=dict)
    reservations: dict = field(default_factory=dict)
    debit_rows: dict = field(default_factory=dict)
    outages: list = field(default_factory=list)
    pace: PaceState | None = None
    closed_at: int | None = None
    delivered_at: int | None = None

    def delivery(self, book, row):
        hotkey = _key(row["hotkey"], "hotkey")
        delivery_id = _key(row["delivery_id"], "delivery_id")
        tokens = _int(row["tokens"], "tokens", 1)
        audited = _int(row.get("audited_tokens", 0), "audited_tokens")
        if audited > tokens:
            raise ValueError("audited_tokens exceed accepted tokens")
        observed_price = row.get("observed_price_micro")
        if observed_price is not None:
            _int(observed_price, "observed_price_micro")
        slot = row.get("slot_id")
        if slot is not None:
            _key(slot, "slot_id")
        at = row["at"]
        lease = row.get("lease_id")
        payload = (hotkey, tokens, audited, observed_price, at, slot, lease)
        if delivery_id in self.deliveries:
            if self.deliveries[delivery_id] != payload:
                raise ValueError("conflicting delivery_id")
            return
        self.deliveries[delivery_id] = payload
        if slot is not None and slot in self.slots:
            raise ValueError("multiple final accepted deliveries for one slot_id")
        counts = self.counts
        counts["observed_tokens"] += tokens
        counts["observed_audited_tokens"] += audited
        if "audited_tokens" in row:
            counts["tokens_with_audit_evidence"] += tokens
        if observed_price is not None:
            counts["observed_priced_tokens"] += tokens
            counts["observed_cost_millirao"] += tokens * observed_price

        price, reason = None, None
        if lease is not None:
            grant = self.reservations.get(_key(lease, "lease_id"))
            if grant is None or grant[0] != hotkey or tokens > grant[2]:
                reason = "no_matching_reservation"
            else:
                price = grant[1]
        elif any(start <= at < end for start, end in self.outages):
            reason = "outage_without_prior_reservation"
        elif self.closed_at is not None or at > self.deadline:
            reason = "outside_contract_interval"
        else:
            contract = book.contract(self.order_id, hotkey)
            if contract is None:
                reason = "no_active_contract"
            elif contract["remaining_tokens"] < tokens:
                reason = "insufficient_contract_volume"
            else:
                price = contract["price_micro"]

        if reason is None:
            try:
                book.record_delivery(self.order_id, hotkey, delivery_id=delivery_id, tokens=tokens,
                                     at=at, audited_tokens=audited, lease_id=lease)
            except ValueError as exc:
                # Top-ups may split one hotkey's remaining volume across awards.
                # An observed record must fit in a single payable award.
                if str(exc) != "delivery exceeds remaining contract":
                    raise
                reason = "insufficient_contract_volume"
            if lease is not None:
                self.reservations.pop(lease, None)
        if reason:
            self.unmatched[reason] += tokens
        else:
            counts["matched_tokens"] += tokens
            counts["matched_audited_tokens"] += audited
            counts["matched_contract_cost_millirao"] += tokens * price
            if observed_price is not None:
                counts["paired_tokens"] += tokens
                counts["paired_observed_cost_millirao"] += tokens * observed_price
                counts["paired_contract_cost_millirao"] += tokens * price
        if slot is not None:
            self.slots[slot] = (at, reason is None)

    def report(self, book):
        c = self.counts
        tail = matched_tail = throughput = None
        total = self.total_slots
        if total and len(self.slots) == len(self.deliveries) == total:
            times = sorted(t for t, _ in self.slots.values())
            elapsed = times[-1] - self.opened_at
            if elapsed > 0:
                throughput = str(Fraction(c["observed_tokens"], elapsed))
            if total >= 50:
                tail = times[-1] - times[total - total // 50 - 1]
                if c["matched_tokens"] == c["observed_tokens"]:
                    matched_tail = tail
        paired = c["paired_tokens"]
        debts = book.debt_events(self.order_id)
        assessed = sum(e["assessed_rao"] for e in debts)
        recovered = sum(self.debit_rows.values())
        return {
            "order_id": self.order_id, "standing_offers_by_source": self.sources,
            "debits_with_settlement_evidence": len(self.debit_rows),
            "evidenced_recovery_rao": recovered if self.debit_rows else None,
            "observed_tokens": c["observed_tokens"], "observed_audited_tokens": c["observed_audited_tokens"],
            "tokens_with_audit_evidence": c["tokens_with_audit_evidence"],
            "observed_priced_tokens": c["observed_priced_tokens"],
            "observed_priced_cost_millirao": c["observed_cost_millirao"] if c["observed_priced_tokens"] else None,
            "matched_tokens": c["matched_tokens"], "matched_audited_tokens": c["matched_audited_tokens"],
            "matched_contract_cost_millirao": c["matched_contract_cost_millirao"] if c["matched_tokens"] else None,
            "unmatched_tokens_by_reason": dict(self.unmatched), "paired_priced_tokens": c["paired_tokens"],
            "paired_observed_cost_millirao": c["paired_observed_cost_millirao"] if c["paired_tokens"] else None,
            "paired_contract_cost_millirao": c["paired_contract_cost_millirao"] if c["paired_tokens"] else None,
            "paired_observed_micro_alpha_per_million": str(Fraction(c["paired_observed_cost_millirao"], paired)) if paired else None,
            "paired_contract_micro_alpha_per_million": str(Fraction(c["paired_contract_cost_millirao"], paired)) if paired else None,
            "observed_tokens_per_s_to_last_acceptance": throughput,
            "observed_last_2pct_accepted_slots_s": tail, "matched_trace_last_2pct_accepted_slots_s": matched_tail,
            "observed_delivered_at": self.delivered_at, "deadline_at": self.deadline,
            "observed_deadline_met": self.delivered_at <= self.deadline if self.delivered_at is not None else None,
            "reserve_accepted": c["reserve_accepted"], "reserve_refused": c["reserve_refused"],
            "observed_missed_requests": c["miss"], "missed_requests_with_active_contract": c["miss_contracted"],
            "unmatched_exits": c["unmatched_exits"],
            "assessed_rao": assessed,
            "unrecovered_after_settlement_rao": assessed - recovered if len(self.debit_rows) == len(debts) else None,
            "pace_samples": c["pace"], "max_sampled_queue_tokens": c["max_queue"] if c["pace"] else None,
            "max_sampled_queue_seconds_at_mean_rate": c["max_queue_s"] if c["finite_queue"] else None,
            "queue_samples_without_capacity": c["no_capacity"],
            "last_target_tokens_per_second": self.pace.rate if self.pace else None,
        }


def _dispatch(book, orders, offers, row):
    kind = row["type"]
    at = row["at"]
    if kind == "offer":
        source = row.get("source", "unspecified")
        if source not in ("observed", "reconstructed", "unspecified"):
            raise ValueError("unknown offer source")
        market, hotkey = row["market"], row["hotkey"]
        book.post_offer(market, hotkey, row["coldkey"], price_micro=row["price_micro"], rate=row["rate"],
                        slots=row["slots"], sequence=row.get("sequence"), at=at)
        offers.setdefault((market, hotkey), []).append((at, row["price_micro"], row["rate"], source))
        return
    job = _key(row["order_id"], "order_id")
    if kind == "open":
        if job in orders:
            raise ValueError("second opening for the same order; replay a retry with its original event_id")
        total = row.get("total_slots")
        if total is not None:
            _int(total, "total_slots", 1)
        market = row["market"]
        options = {k: row[k] for k in ("duration_s", "target_rate", "max_tokens", "budget_rao",
                   "ceiling_micro", "new_rate_cap", "audit_cost_micro", "evaluation", "q_min", "trust_tokens") if k in row}
        book.open_order(job, market, opened_at=at, **options)
        sources = Counter()
        for (m, _), versions in offers.items():
            if m == market:
                # The registry records whole seconds. The opening excludes that entire second.
                latest = next((version for version in reversed(versions) if version[0] < at), None)
                if latest is not None and latest[2] and latest[1] <= row["ceiling_micro"]:
                    sources[latest[3]] += 1
        orders[job] = Order(job, at, at + row["duration_s"], total, dict(sources))
        return
    if job not in orders:
        raise ValueError("order event before its opening")
    order = orders[job]
    counts = order.counts
    if kind == "delivery":
        order.delivery(book, row)
    elif kind == "reserve":
        hotkey, lease = row["hotkey"], row["lease_id"]
        old = order.reservations.get(lease)
        if old is not None:
            if (hotkey, row["max_tokens"]) != (old[0], old[2]):
                raise ValueError("conflicting lease reservation")
            return
        ok = book.grant(job, hotkey, lease_id=lease, max_tokens=row["max_tokens"], at=at)
        if ok:
            order.reservations[lease] = (hotkey, book.contract(job, hotkey)["price_micro"], row["max_tokens"])
        counts["reserve_accepted" if ok else "reserve_refused"] += 1
    elif kind == "cancel":
        lease = _key(row["lease_id"], "lease_id")
        book.cancel_grant(job, lease_id=lease, at=at)
        order.reservations.pop(lease, None)
    elif kind == "miss":
        hotkey = _key(row["hotkey"], "hotkey")
        counts["miss"] += 1
        if book.contract(job, hotkey) is not None:
            counts["miss_contracted"] += 1
    elif kind == "outage":
        book.service_interval(job, start=at, end=row["end"])
        order.outages.append((at, row["end"]))
    elif kind == "exit":
        if row["reason"] not in ("release", "announced", "disappeared"):
            raise ValueError("unknown exit reason")
        if book.contract(job, row["hotkey"]) is None:
            counts["unmatched_exits"] += 1
        else:
            fields = {k: row[k] for k in ("expired_leases", "penalty_rao", "notice_s") if k in row}
            book.exit(job, row["hotkey"], at=at, reason=row["reason"], event_id=row["event_id"], **fields)
    elif kind == "round":
        book.supplemental_round(job, at=at, event_id=row["event_id"], eligible_owners=row["eligible_owners"])
    elif kind == "rate":
        book.reconcile_rate(job, row["hotkey"], at=at, event_id=row["event_id"],
                            window_start=row["window_start"], evidence_complete=row["evidence_complete"],
                            min_active_s=row.get("min_active_s", 180))
    elif kind == "debits":
        values = period_debit_rows(book.debt_events(job), order_id=job, work_period=row["work_period"],
                                   owners_at_period_end=row["owners_at_period_end"], recovered_rao=row["recovered_rao"])
        for value in values:
            eid, recovery = value["event_id"], value["recovered_at_settlement_rao"]
            if order.debit_rows.setdefault(eid, recovery) != recovery:
                raise ValueError("conflicting settlement recovery")
    elif kind == "pace":
        queued = _int(row["queued_audit_tokens"], "queued_audit_tokens")
        order.pace = paced_rate(order.pace, now=at, **{k: row[k] for k in (
            "card_tokens_per_second", "queued_audit_tokens", "audit_fraction", "offered_rate", "needed_rate")})
        counts["pace"] += 1
        counts["max_queue"] = max(counts["max_queue"], queued)
        if order.pace.average_card_rate > 0:
            counts["finite_queue"] += 1
            counts["max_queue_s"] = max(counts["max_queue_s"], math.ceil(queued / order.pace.average_card_rate))
        elif queued:
            counts["no_capacity"] += 1
    elif kind == "close":
        if order.closed_at is not None or type(row.get("delivered", False)) is not bool:
            raise ValueError("invalid second close or delivered flag")
        book.close(job, at=at)
        order.closed_at = at
        order.delivered_at = at if row.get("delivered") else None
    else:
        raise ValueError(f"unknown event type {kind!r}")


def replay(lines):
    events, seen = [], {}
    duplicates = 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            event_id = _key(row["event_id"], "event_id")
            at = _int(row["at"], "at")
            digest = _digest(row)
            if event_id in seen:
                if seen[event_id] != digest:
                    raise ValueError("conflicting event_id")
                duplicates += 1
                continue
            seen[event_id] = digest
            events.append((at, n, row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"line {n}: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="order-replay-") as directory:
        book = ContractBook(Path(directory) / "book.sqlite")
        orders, offers = {}, {}
        for _, n, row in sorted(events, key=lambda e: (e[0], e[1])):
            try:
                _dispatch(book, orders, offers, row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {n} ({row.get('type')}): {exc}") from exc
        return {"cost_basis": "accepted tokens at observed lease prices and frozen offers, before settlement",
                "timing_basis": "observed slot acceptance times; alternative deliveries and pacing are not simulated",
                "unique_events": len(events), "duplicate_events": duplicates,
                "orders": [orders[k].report(book) for k in sorted(orders)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=argparse.FileType("r", encoding="utf-8"), help="JSONL trace file, or - for stdin")
    args = parser.parse_args()
    try:
        with args.events as stream:
            print(json.dumps(replay(stream), indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
