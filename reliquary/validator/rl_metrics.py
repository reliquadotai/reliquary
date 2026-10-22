"""The async controller's wandb row (one per ``METRICS_EVERY_S``): prices, supply and demand per env, the buffer,
leases, miners, audits and pay, read from the lease book's SQLite file. The trainer's row is ``AsyncTrainer``'s."""
from __future__ import annotations

import json
import re
from typing import Any

METRICS_EVERY_S = 60.0
ACTIVE_MINER_S = 600.0          # a hotkey issued a lease this recently counts as an active miner
MAX_CARD_CLASSES = 12           # declared names are miner-chosen: the rest is folded into "other"


def card_class(name: str) -> str:
    """A declared GPU name as a wandb-safe key: ``NVIDIA GeForce RTX 4090`` -> ``rtx-4090``."""
    s = re.sub(r"\b(nvidia|geforce)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40] or "unknown"


def card_metrics(rl: Any, summary: list[dict], now: float, window_s: float = METRICS_EVERY_S) -> dict[str, float]:
    """Per declared card model (``miner_setup.summarize``): the active fleet and how the trainer judged its groups
    over the window. A hotkey whose cards differ is ``mixed``, one that declared nothing ``undeclared``."""
    cards = {}
    for line in summary:
        classes = {card_class(n) for n in line["gpus"]}
        cards[line["hotkey"]] = (classes.pop() if len(classes) == 1 else "mixed" if classes else "undeclared",
                                 int(line["gpu_count"]))
    db = rl.book.db
    with rl.book.lock:
        active = [h for (h,) in db.execute("SELECT DISTINCT hotkey FROM rl_leases WHERE issued_at>=?",
                                           (now - ACTIVE_MINER_S,))]
        events = db.execute("SELECT hotkey, kind, detail FROM rl_events WHERE at>=? AND kind IN (?, ?)",
                            (now - window_s, "trained", "untrained")).fetchall()
    fleet: dict[str, list[int]] = {}
    for hotkey in active:
        cls, gpus = cards.get(hotkey, ("undeclared", 0))
        miners_gpus = fleet.setdefault(cls, [0, 0])
        miners_gpus[0] += 1
        miners_gpus[1] += gpus
    kept = {c for c, _ in sorted(fleet.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))[:MAX_CARD_CLASSES]}

    def bucket(cls: str) -> str:
        return cls if cls in kept or cls in ("mixed", "undeclared") else "other"

    out: dict[str, float] = {}
    for cls, (miners, gpus) in fleet.items():
        key = bucket(cls)
        out[f"cards/{key}/miners"] = out.get(f"cards/{key}/miners", 0) + miners
        out[f"cards/{key}/gpus"] = out.get(f"cards/{key}/gpus", 0) + gpus
    judged: dict[str, dict[str, float]] = {}
    for hotkey, kind, raw in events:
        detail = json.loads(raw)
        match = detail.get("rollout_match") or []
        if not any(m > 0 for m in match):       # skipped step, stale or exhausted: the trainer never judged it
            continue
        t = judged.setdefault(bucket(cards.get(hotkey, ("undeclared", 0))[0]),
                              {"trained": 0, "rejected": 0, "match_sum": 0.0, "rollouts": 0, "match_min": 1.0,
                               "tokens": 0})
        t["trained" if kind == "trained" else "rejected"] += 1
        t["match_sum"] += sum(match)
        t["rollouts"] += len(match)
        t["match_min"] = min(t["match_min"], min(match))
        if kind == "trained":
            t["tokens"] += sum(detail.get("matched_tokens") or detail.get("completion_tokens") or [])
    total = sum(t["tokens"] for t in judged.values())
    for key, t in judged.items():
        out[f"cards/{key}/groups_trained"] = t["trained"]
        out[f"cards/{key}/groups_rejected"] = t["rejected"]
        out[f"cards/{key}/rejected_ratio"] = t["rejected"] / (t["trained"] + t["rejected"])
        out[f"cards/{key}/rollout_match_mean"] = t["match_sum"] / t["rollouts"]
        out[f"cards/{key}/rollout_match_min"] = t["match_min"]
        out[f"cards/{key}/trained_tokens"] = t["tokens"]
        out[f"cards/{key}/trained_token_share"] = t["tokens"] / total if total else 0.0
    return out


def controller_metrics(rl: Any, now: float, window_s: float = METRICS_EVERY_S,
                       setups: Any = None) -> dict[str, float]:
    since = now - window_s
    out: dict[str, float] = {}
    for env, price in rl.prices.board().items():
        out[f"price/{env}/multiplier"] = price["multiplier"]
        out[f"price/{env}/per_million_tokens"] = price["per_million_tokens"]
        for key in ("ratio", "supply_tps", "demand_tps", "trainer_tps", "starved_s"):
            if isinstance(price["signal"].get(key), (int, float)):
                out[f"price/{env}/{key}"] = price["signal"][key]
    db = rl.book.db
    with rl.book.lock:
        speed = db.execute("SELECT tokens, seconds FROM rl_trainer_speed WHERE id=0").fetchone()
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
    cooldown = getattr(getattr(rl.book, "config", None), "cooldown_of", None)
    snapshot = getattr(cooldown, "snapshot", None)
    if snapshot is not None:
        with rl.book.lock:
            out.update(snapshot(sorted(rl.prices.board())))
        with rl.book.lock:
            for env, state, n in db.execute("SELECT environment, state, COUNT(*) FROM rl_prompt_state GROUP BY 1, 2"):
                out[f"prompts/{env}/{state}"] = n
    for source, verdict, n in rows["verdicts"]:
        out[f"verdicts/{source}/{verdict}"] = n
    for kind, amount, tokens in rows["accrued"]:
        out[f"pay/{kind}/amount"] = amount or 0.0
        out[f"pay/{kind}/matched_tokens"] = tokens or 0
    if speed is not None and speed[1] > 0:
        out["trainer/completion_tokens_per_s"] = speed[0] / speed[1]
    out.update({"audits/queued": audit_queue, "miners/active": miners, "miners/banned": banned,
                "miners/quota_cap_mean": sum(caps) / len(caps) if caps else 0.0})
    if setups is not None:
        from reliquary.validator.miner_setup import summarize

        out.update(card_metrics(rl, summarize(setups.rows(), now=now), now, window_s))
    return out


__all__ = ["METRICS_EVERY_S", "card_class", "card_metrics", "controller_metrics"]
