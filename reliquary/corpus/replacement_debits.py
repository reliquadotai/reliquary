"""Serialize assessed replacement costs with the settlement's attested deductions."""

from __future__ import annotations

import hashlib
import json

from reliquary.validator.corpus_periods import period_of


def period_debit_rows(
    events, *, order_id, work_period, owners_at_period_end, recovered_rao
):
    """Build replacement_debits_v1 for one order and work period.

    The owner map is historical. The recovered amounts are final integer rao
    after budget caps and rounding. Every event requires an explicit amount,
    including zero. An archive requeue keeps these rows unchanged.
    """
    rows = {}
    for event in events:
        if event["order_id"] != order_id:
            raise ValueError("replacement event belongs to another order")
        if period_of(event["at"]) != work_period:
            continue
        source = event["event_id"]
        owner = owners_at_period_end.get(event["hotkey"])
        if owner is None or owner != event["coldkey"]:
            raise ValueError("unconfirmed earn-time owner for a replacement debit")
        collected = recovered_rao.get(source)
        assessed = event["assessed_rao"]
        if (
            type(assessed) is not int
            or type(collected) is not int
            or not 0 <= collected <= assessed
        ):
            raise ValueError("missing or inconsistent settled recovery")
        identity = json.dumps(
            [order_id, source], ensure_ascii=True, separators=(",", ":")
        ).encode()
        row = {
            "event_id": hashlib.sha256(identity).hexdigest(),
            "hotkey": event["hotkey"],
            "coldkey_at_earn": owner,
            "reason": event["reason"],
            "assessed_rao": assessed,
            "recovered_at_settlement_rao": collected,
        }
        if source in rows and rows[source] != row:
            raise ValueError("conflicting replacement event")
        rows[source] = row
    return [rows[source] for source in sorted(rows)]
