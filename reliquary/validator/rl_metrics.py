"""The async controller's wandb row (one per ``METRICS_EVERY_S``): prices and the queue signal per env, the buffer,
leases, miners, audits and pay, read from the lease book's SQLite file. The trainer's row is ``AsyncTrainer``'s."""
from __future__ import annotations

import json
from typing import Any

METRICS_EVERY_S = 60.0
ACTIVE_MINER_S = 600.0          # a hotkey issued a lease this recently counts as an active miner


def controller_metrics(rl: Any, now: float, window_s: float = METRICS_EVERY_S) -> dict[str, float]:
    since = now - window_s
    out: dict[str, float] = {}
    for env, price in rl.prices.board().items():
        out[f"price/{env}/multiplier"] = price["multiplier"]
        out[f"price/{env}/per_million_tokens"] = price["per_million_tokens"]
        for key in ("mix", "supply_tps", "backlog", "starved_s"):
            if isinstance(price["signal"].get(key), (int, float)):
                out[f"price/{env}/{key}"] = price["signal"][key]
    db = rl.book.db
    with rl.book.lock:
        rows = {
            "buffer": db.execute("SELECT environment, state, COUNT(*) FROM rl_buffer GROUP BY 1, 2").fetchall(),
            "leases": db.execute("SELECT state, COUNT(*) FROM rl_leases WHERE state='open' OR closed_at>=? "
                                 "GROUP BY 1", (since,)).fetchall(),
            "issued": db.execute("SELECT environment, COUNT(*) FROM rl_leases WHERE issued_at>=? GROUP BY 1",
                                 (since,)).fetchall(),
            "groups": db.execute("SELECT environment, verdict, COUNT(*) FROM rl_groups WHERE arrived_at>=? "
                                 "GROUP BY 1, 2", (since,)).fetchall(),
            "events": db.execute("SELECT kind, COUNT(*) FROM rl_events WHERE at>=? GROUP BY 1", (since,)).fetchall(),
            "verdicts": db.execute("SELECT source, verdict, COUNT(*) FROM rl_verdicts WHERE at>=? GROUP BY 1, 2",
                                   (since,)).fetchall(),
            "accrued": db.execute("SELECT kind, SUM(amount), SUM(matched_tokens) FROM rl_accruals WHERE at>=? "
                                  "GROUP BY 1", (since,)).fetchall(),
        }
        audit_queue = db.execute("SELECT COUNT(*) FROM rl_audits").fetchone()[0]
        miners = db.execute("SELECT COUNT(DISTINCT hotkey) FROM rl_leases WHERE issued_at>=?",
                            (now - ACTIVE_MINER_S,)).fetchone()[0]
        banned = db.execute("SELECT COUNT(*) FROM rl_hotkey_bans WHERE until>?", (now,)).fetchone()[0]
        caps = [json.loads(state).get("cap", 0) for (state,) in db.execute("SELECT state FROM rl_lease_quota")]
    for env, state, n in rows["buffer"]:
        out[f"buffer/{env}/{state}"] = n
    for state, n in rows["leases"]:
        out[f"leases/{state}"] = n
    for env, n in rows["issued"]:
        out[f"leases_issued/{env}"] = n
    for env, verdict, n in rows["groups"]:
        out[f"groups/{env}/{verdict}"] = n
    for kind, n in rows["events"]:
        out[f"events/{kind}"] = n
    for source, verdict, n in rows["verdicts"]:
        out[f"verdicts/{source}/{verdict}"] = n
    for kind, amount, tokens in rows["accrued"]:
        out[f"pay/{kind}/amount"] = amount or 0.0
        out[f"pay/{kind}/matched_tokens"] = tokens or 0
    out.update({"audits/queued": audit_queue, "miners/active": miners, "miners/banned": banned,
                "miners/quota_cap_mean": sum(caps) / len(caps) if caps else 0.0})
    return out


__all__ = ["METRICS_EVERY_S", "controller_metrics"]
