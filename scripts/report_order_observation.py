#!/usr/bin/env python3
"""Report aggregate Core order observations from one local, read-only SQLite snapshot.

Run with BOOK.sqlite, optionally --order ORDER_ID. JSON on stdout contains no
order or miner identifiers. The settled baseline values archived reward shares
at the opening emission rate when the archived price ceiling matches the
opening ceiling, rounding down to rao per hotkey and fragment. This is a
valuation of the shares at opening terms, without evidence of a chain transfer.

The paired comparison values the same verified deliveries at their observed
lease price and at the frozen offer. Only entire records fitting a live award's
time and funded volume are included. A conservative cumulative-rate heuristic
screens receipts without lease start times. Bursty miners can exceed it. If a
period lacks detail or final token coverage is unconfirmed, the order is left unpaired.
Work never observed, alternative completion times and actual V1 payouts are
not inferred. Coverage refers only to the period fragments present in this file.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


def _number(value, label):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite, nonnegative number")
    return Fraction(value)


def _tokens(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid token count in observation")
    return value


@dataclass
class Award:
    start: int
    end: int
    rate: int
    tokens: int
    price_micro: int
    consumed: int = 0

    def fits(self, at, tokens, pauses):
        if not self.start <= at < self.end or self.consumed + tokens > self.tokens:
            return False
        inactive = sum(max(0, min(at, end) - max(self.start, begin)) for begin, end in pauses)
        return self.consumed + tokens <= self.rate * (at - self.start - inactive)


def _pauses(db, job):
    merged = []
    for begin, end in db.execute("SELECT start,end FROM pauses WHERE order_id=? ORDER BY start,end", (job,)):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


@dataclass(frozen=True)
class Receipt:
    sid: str
    hotkey: str
    at: Fraction | None
    tokens: int
    audited: bool
    nominal_rao_per_token: Fraction | None


def _load_periods(db, job, counts):
    receipts = {}
    complete = True
    periods = db.execute("""SELECT document FROM observed_periods WHERE order_id=?
                            ORDER BY work_period,entry_period""", (job,))
    for (document,) in periods:
        row = json.loads(document)
        volume = _tokens(row["tokens"])
        counts["captured_period_fragments"] += 1
        counts["settled_tokens"] += volume
        rate = row["miner_alpha_per_period"] if row.get("valuation_source") == "opening_declaration" else None
        rate = _number(rate, "frozen emission") * 10**9 if rate is not None else None
        rewards = row["rewards_by_hotkey"]
        if not isinstance(rewards, dict):
            raise TypeError("invalid reward shares in observation")
        shares = [_number(share, "reward share") for share in rewards.values()]
        if rate is None:
            counts["fragments_missing_valuation"] += 1
        else:
            counts["valued_fragments"] += 1
            counts["settled_baseline_rao"] += sum(math.floor(share * rate) for share in shares)

        deliveries = row["deliveries"]
        if deliveries is None:
            counts["fragments_without_receipts"] += 1
            complete = complete and volume == 0
            continue
        if not isinstance(deliveries, list):
            raise TypeError("invalid delivery list")
        detailed = 0
        for delivery in deliveries:
            sid, miner = delivery["id"], delivery["hotkey"]
            if not isinstance(sid, str) or not sid or not isinstance(miner, str) or not miner:
                raise ValueError("missing receipt identity")
            if sid in receipts:
                raise ValueError("overlapping receipt ids in archived settlement evidence")
            audited = delivery["audited"]
            if type(audited) is not bool:
                raise ValueError("invalid audit outcome")
            volume_i = _tokens(delivery["tokens"])
            if not volume_i:
                raise ValueError("a verified receipt must contain at least one token")
            at = delivery["received_at"]
            if at is None:
                complete = False
                counts["tokens_missing_receipt_time"] += volume_i
            else:
                at = _number(at, "receipt time")
            price = delivery["lease_price"]
            price = _number(price, "observed lease price") if price is not None else None
            nominal = price * rate if price is not None and rate is not None else None
            receipts[sid] = Receipt(sid, miner, at, volume_i, audited, nominal)
            detailed += volume_i
            counts["receipt_tokens"] += volume_i
            counts["audited_receipt_tokens"] += volume_i if audited else 0
        if detailed != volume:
            counts["fragments_with_volume_mismatch"] += 1
            complete = False
    return receipts, complete


def _measure(db, order, counts, owner_known):
    job, spec_json, closed = order
    spec = json.loads(spec_json)
    deadline = spec["opened_at"] + spec["duration_s"]
    end = min(deadline, closed) if closed is not None else deadline
    awards = defaultdict(list)
    for miner, start, stop, rate, size, price in db.execute(
        """SELECT hotkey,started_at,ended_at,rate,tokens,price_micro FROM awards
           WHERE order_id=? ORDER BY id""", (job,)
    ):
        awards[miner].append(Award(start, min(stop, end) if stop is not None else end, rate, size, price))
    pauses = _pauses(db, job)
    before = counts["settled_tokens"]
    receipts, complete = _load_periods(db, job, counts)
    captured = counts["settled_tokens"] - before
    final = db.execute("SELECT verified_tokens,settled FROM observed_closures WHERE order_id=?", (job,)).fetchone()
    if final is None or final[0] is None:
        counts["orders_without_final_verified_total"] += 1
        reconciled = False
    else:
        verified = _tokens(final[0])
        counts["final_verified_tokens_on_known_orders"] += verified
        counts["captured_tokens_on_known_orders"] += captured
        if final[1] is not None and len(receipts) > _tokens(final[1]):
            raise ValueError("captured more unique verified receipts than final settled verdicts")
        reconciled = captured == verified
        counts["orders_with_matching_final_tokens" if reconciled else "orders_with_final_token_gap"] += 1
    if not complete:
        counts["orders_with_incomplete_detail"] += 1
    if not owner_known:
        counts["orders_without_owner_snapshot"] += 1
        counts["tokens_without_owner_snapshot"] += sum(r.tokens for r in receipts.values())
    if not complete or not reconciled:
        counts["unmatched_incomplete_history_tokens"] += sum(r.tokens for r in receipts.values())
        return

    if not owner_known:
        return
    capacity, funds = spec["max_tokens"], spec["budget_rao"] * 1000
    # Fragments can settle in a different order from when the work arrived.
    for r in sorted(receipts.values(), key=lambda r: (r.at, r.sid)):
        if any(begin <= r.at < stop for begin, stop in pauses):
            counts["unmatched_outage_tokens"] += r.tokens
            continue
        chosen = next((a for a in awards[r.hotkey] if a.fits(r.at, r.tokens, pauses)), None)
        if chosen is None or r.tokens > capacity or r.tokens * chosen.price_micro > funds:
            counts["unmatched_window_rate_or_volume_tokens"] += r.tokens
            continue
        chosen.consumed += r.tokens
        contract_millirao = r.tokens * chosen.price_micro
        capacity -= r.tokens
        funds -= contract_millirao
        counts["matched_contract_tokens"] += r.tokens
        counts["matched_contract_millirao"] += contract_millirao
        if r.nominal_rao_per_token is None:
            counts["matched_tokens_without_observed_price_or_rate"] += r.tokens
            continue
        counts["paired_tokens"] += r.tokens
        counts["paired_observed_millirao"] += 1000 * r.tokens * r.nominal_rao_per_token
        counts["paired_contract_millirao"] += contract_millirao


def report(path, *, order_id=None):
    counts = Counter()
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        sql = "SELECT order_id,spec,closed_at FROM orders"
        if order_id is not None:
            sql += " WHERE order_id=?"
        rows = db.execute(sql, (order_id,) if order_id is not None else ())
        has_owner_reads = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='opening_owner_reads'"
        ).fetchone()
        for row in rows:
            counts["orders"] += 1
            owner_known = has_owner_reads and db.execute(
                "SELECT 1 FROM opening_owner_reads WHERE order_id=? AND observed_at<=?",
                (row[0], json.loads(row[1])["opened_at"]),
            ).fetchone()
            _measure(db, row, counts, owner_known)
        if order_id is not None and not counts["orders"]:
            raise ValueError("the requested order is absent from this observation file")
    paired = counts["paired_tokens"]
    reasons = ("orders", "orders_without_owner_snapshot", "tokens_without_owner_snapshot", "captured_period_fragments", "valued_fragments", "fragments_missing_valuation",
               "settled_tokens", "fragments_without_receipts", "fragments_with_volume_mismatch",
               "receipt_tokens", "audited_receipt_tokens", "orders_with_incomplete_detail",
               "tokens_missing_receipt_time", "orders_without_final_verified_total", "orders_with_matching_final_tokens",
               "orders_with_final_token_gap", "final_verified_tokens_on_known_orders", "captured_tokens_on_known_orders",
               "unmatched_incomplete_history_tokens", "unmatched_outage_tokens",
               "unmatched_window_rate_or_volume_tokens", "matched_contract_tokens",
               "matched_tokens_without_observed_price_or_rate", "paired_tokens")
    return {
        **{name: counts[name] for name in reasons},
        "settled_baseline_rao_on_valued_fragments": counts["settled_baseline_rao"] if counts["valued_fragments"] else None,
        "matched_nominal_contract_millirao": counts["matched_contract_millirao"] if counts["matched_contract_tokens"] else None,
        "paired_nominal_observed_millirao_floor": math.floor(counts["paired_observed_millirao"]) if paired else None,
        "paired_nominal_contract_millirao": counts["paired_contract_millirao"] if paired else None,
        "ownership_basis": "last completed registry read known by the opening second within the registration cache TTL; rotations between reads are unobserved",
        "coverage_basis": "pairing requires complete receipt details and captured tokens equal to the authoritative final count",
        "settlement_basis": "archived net shares valued at the opening emission when the archived ceiling matches, floored to rao per miner per fragment",
        "comparison_basis": "gross prices at original receipt times; elapsed-rate envelope is a conservative heuristic for bursty arrivals",
    }


def standing_supply(path, market, before):
    """Count declared supply before an opening second, without revealing individual offers."""
    if before < 0 or before >= 2**63:
        raise ValueError("opening second is outside the supported range")
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        active, owners = db.execute(
            """WITH versions AS (
                 SELECT coldkey, rate, slots, ROW_NUMBER() OVER (
                   PARTITION BY hotkey ORDER BY effective_at DESC, id DESC
                 ) AS latest
                 FROM offer_versions WHERE market = ? AND effective_at < ?
               )
               SELECT COUNT(*), COUNT(DISTINCT coldkey) FROM versions
               WHERE latest = 1 AND rate > 0 AND slots > 0""",
            (market, before),
        ).fetchone()
    return {"before_unix_second": before, "active_offers": active, "offering_coldkeys": owners}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database", type=Path)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--order", help="restrict the aggregate to one local order")
    select.add_argument("--market", help="count posted standing offers for a market before opening; eligibility is checked at opening")
    parser.add_argument("--before", type=int, help="exclusive opening second for --market (default: current Unix time)")
    args = parser.parse_args()
    if args.before is not None and args.market is None:
        parser.error("--before requires --market")
    try:
        if args.market is None:
            result = report(args.database, order_id=args.order)
        else:
            cutoff = int(time.time()) if args.before is None else args.before
            result = standing_supply(args.database, args.market, cutoff)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, sqlite3.Error, TypeError) as exc:
        parser.exit(2, f"could not read order observations: {exc}\n")


if __name__ == "__main__":
    main()
