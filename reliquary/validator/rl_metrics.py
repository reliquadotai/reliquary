"""The async controller's wandb row (one per ``METRICS_EVERY_S``): prices, supply and demand per env, the buffer,
leases, miners, audits and pay, read from the lease book's SQLite file. The trainer's row is ``AsyncTrainer``'s."""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METRICS_EVERY_S = 60.0
ACTIVE_MINER_S = 600.0          # a hotkey issued a lease this recently (or holding a live one) is an active miner
MAX_NAMED_CARDS = 9             # declared names are miner-chosen: past these, "other" (12 models with the 3 buckets)
BUCKETS = ("mixed", "undeclared", "other")
CARD_WINDOW_MAX_S = 240.0       # the dashboard drops a card window longer than 300 s
SHARE_WINDOW_S = 600.0          # trained-token shares: trailing, so a row between two trainer steps still has them
RUNNING_INSTANCE_S = 1800.0     # a miner process with a tagged submission this recently counts in the fleet
                                # (at least the longest lease lifetime: a slow card delivers that rarely)


def card_class(name: str) -> str:
    """A declared GPU name as a wandb-safe key: ``NVIDIA GeForce RTX 4090`` -> ``rtx-4090``."""
    s = re.sub(r"\b(nvidia|geforce)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40] or "unknown"


class CardFeed:
    """The controller row's ``cards/*`` (read by the platform run monitor, ``shared/rl-card-fleet.ts``).

    Every card model once published stays in every later row (zeros when idle), so the dashboard's model set (the
    run summary) always has a full same-row vector; it is persisted at ``path`` across restarts of a resumed run.
    Verdict counts cover the disjoint window ``[cards/window_start, _timestamp)``; the previous row's end is the
    next one's start. A new feed starts now, without replaying the interval before its process started."""

    def __init__(self, setups: Any, path: str | Path | None = None) -> None:
        self.setups = setups
        self.path = None if path is None else Path(path)
        self.named: list[str] = []
        if self.path is not None and self.path.exists():
            try:
                self.named = [str(c) for c in json.loads(self.path.read_text())][:MAX_NAMED_CARDS]
            except (OSError, ValueError, TypeError):
                logger.warning("card models at %s unreadable; starting over", self.path)
        self.since = time.time()

    def _admit(self, classes: list[str]) -> None:
        new = [c for c in classes if c not in self.named and c not in BUCKETS][:MAX_NAMED_CARDS - len(self.named)]
        if not new:
            return
        self.named.extend(new)
        if self.path is not None:
            try:
                self.path.write_text(json.dumps(self.named))
            except OSError:
                logger.warning("card models not saved to %s", self.path)

    def row(self, rl: Any, now: float, window_s: float = METRICS_EVERY_S) -> dict[str, float]:
        from reliquary.validator.miner_setup import summarize

        start = min(now, max(self.since, now - CARD_WINDOW_MAX_S))
        rows = self.setups.rows()
        out = card_metrics(rl, summarize(rows, now=now), now, start, self, instances=instance_classes(rows, now),
                           instance_gpus=instance_cards(rows, now))
        self.since = max(self.since, now)
        return out


def instance_cards(rows: list[dict], now: float) -> dict[tuple[str, str], dict[str, str]]:
    """``(hotkey, instance_id)`` -> ``{card key: card class}`` of that miner process's declared GPUs, over the setups
    seen in ``miner_setup.ACTIVE_WINDOW_S``; the latest report of an instance wins. A card's key is its ``id_hash``,
    so two processes sharing a card count it once."""
    from reliquary.validator.miner_setup import ACTIVE_WINDOW_S

    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: r["received_at"]):
        if now - row["received_at"] > ACTIVE_WINDOW_S:
            continue
        gpus = row["report"].get("gpus", [])
        cards = {gpu.get("id_hash") or f"{row['instance_id']}:{i}": card_class(gpu["name"]) for i, gpu in enumerate(gpus)}
        if cards:
            out[(row["hotkey"], row["instance_id"])] = cards
    return out


def instance_classes(rows: list[dict], now: float) -> dict[tuple[str, str], str]:
    """``(hotkey, instance_id)`` -> the card class of that miner process (``mixed`` if its GPUs differ)."""
    return {k: (v.pop() if len(v) == 1 else "mixed")
            for k, v in ((k, set(c.values())) for k, c in instance_cards(rows, now).items())}


def _events(db: Any, since: float, now: float) -> list[tuple]:
    """``(at, hotkey, kind, detail, instance_id or None)`` of the trained/untrained events in ``[since, now)``."""
    tagged = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rl_group_instances'").fetchone()
    if tagged is None:
        return [(*row, None) for row in db.execute(
            "SELECT at, hotkey, kind, detail FROM rl_events WHERE at>=? AND at<? AND kind IN (?, ?)",
            (since, now, "trained", "untrained"))]
    return db.execute("SELECT e.at, e.hotkey, e.kind, e.detail, i.instance_id FROM rl_events e "
                      "LEFT JOIN rl_group_instances i ON i.lease_id=e.lease_id AND i.hotkey=e.hotkey "
                      "WHERE e.at>=? AND e.at<? AND e.kind IN (?, ?)", (since, now, "trained", "untrained")).fetchall()


def active_hotkeys(db: Any, now: float) -> list[str]:
    """Hotkeys issued a lease in ``ACTIVE_MINER_S`` or holding an open unexpired one: a slow card at full quota
    asks for no new lease until it delivers, yet it is generating."""
    return [h for (h,) in db.execute("SELECT hotkey FROM rl_leases WHERE issued_at>=? UNION "
                                     "SELECT hotkey FROM rl_leases WHERE state='open' AND expires_at>?",
                                     (now - ACTIVE_MINER_S, now))]


def running_instance_s(rl: Any) -> float:
    """``RUNNING_INSTANCE_S``, raised to the longest lease lifetime."""
    expiry = getattr(getattr(rl.book, "config", None), "expiry_s", None) or {}
    return max([RUNNING_INSTANCE_S, *(float(v) for v in expiry.values())])


def _running_instances(db: Any, since: float) -> dict[str, set[str]]:
    """hotkey -> the instance ids its tagged submissions named since ``since``."""
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rl_group_instances'").fetchone() is None:
        return {}
    out: dict[str, set[str]] = {}
    for hotkey, instance in db.execute("SELECT DISTINCT hotkey, instance_id FROM rl_group_instances WHERE at>=?",
                                       (since,)):
        out.setdefault(hotkey, set()).add(instance)
    return out


def card_metrics(rl: Any, summary: list[dict], now: float, start: float, feed: CardFeed,
                 instances: dict[tuple[str, str], str] | None = None,
                 instance_gpus: dict[tuple[str, str], dict[str, str]] | None = None) -> dict[str, float]:
    """Per declared card model (``miner_setup.summarize``): the active fleet now, and how the trainer judged its
    groups in ``[start, now)``. An active hotkey's GPUs are those of its processes that submitted a tagged group in
    ``running_instance_s`` (``instance_gpus``, from ``instance_cards``), else all it declared; each card counts
    under its own model, and the hotkey as a miner of each. One that declared nothing is ``undeclared``.
    A group its miner tagged with a declared ``instance_id`` (``instances``, from ``instance_classes``) counts for
    that process's cards instead of its hotkey's (``mixed`` when the untagged hotkey's cards differ)."""
    instances, instance_gpus = instances or {}, instance_gpus or {}
    declared: dict[str, Counter] = {}
    cards = {}
    for line in summary:
        classes = {card_class(n) for n in line["gpus"]}
        cards[line["hotkey"]] = (classes.pop() if len(classes) == 1 else "mixed" if classes else "undeclared",
                                 int(line["gpu_count"]))
        declared[line["hotkey"]] = Counter()
        for name, n in line["gpus"].items():
            declared[line["hotkey"]][card_class(name)] += int(n)
    db = rl.book.db
    with rl.book.lock:
        active = active_hotkeys(db, now)
        events = _events(db, min(start, now - SHARE_WINDOW_S), now)
        running = _running_instances(db, now - running_instance_s(rl))
    fleet: dict[str, list[int]] = {}
    for hotkey in active:
        seen: dict[str, str] = {}
        for instance in running.get(hotkey, ()):
            seen.update(instance_gpus.get((hotkey, instance), {}))
        gpus = Counter(seen.values()) if seen else declared.get(hotkey) or Counter({"undeclared": 0})
        for cls, n in gpus.items():
            miners_gpus = fleet.setdefault(cls, [0, 0])
            miners_gpus[0] += 1
            miners_gpus[1] += n
    feed._admit([c for c, _ in sorted(fleet.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))])
    feed._admit(sorted({c for c in instances.values() if c != "mixed"}))   # a mixed process's own card kinds

    def bucket(cls: str) -> str:
        return cls if cls in feed.named or cls in BUCKETS else "other"

    models = [*feed.named, *BUCKETS]
    zero = {"miners": 0, "gpus": 0, "groups": 0, "groups_trained": 0, "groups_rejected": 0, "rollouts": 0,
            "match_sum": 0.0, "match_min": 1.0, "trained_tokens": 0, "share_tokens": 0}
    t = {m: dict(zero) for m in models}
    for cls, (miners, gpus) in fleet.items():
        t[bucket(cls)]["miners"] += miners
        t[bucket(cls)]["gpus"] += gpus
    for at, hotkey, kind, raw, instance in events:
        detail = json.loads(raw)
        match = detail.get("rollout_match") or []
        judged = detail.get("judged")
        if judged is False or (judged is None and not any(m > 0 for m in match)):
            continue                           # legacy all-zero reports cannot distinguish skipped from judged
        cls = instances.get((hotkey, instance)) if instance is not None else None
        c = t[bucket(cls or cards.get(hotkey, ("undeclared", 0))[0])]
        tokens = int(sum(detail.get("matched_tokens") or detail.get("completion_tokens") or [])) \
            if kind == "trained" else 0
        c["share_tokens"] += tokens
        if at < start:                          # only in the trailing share window
            continue
        c["groups"] += 1
        c["groups_trained" if kind == "trained" else "groups_rejected"] += 1
        c["rollouts"] += len(match)
        c["match_sum"] += sum(match)
        c["match_min"] = min(c["match_min"], min(match))
        c["trained_tokens"] += tokens
    share_total = sum(c["share_tokens"] for c in t.values())
    out: dict[str, float] = {"cards/window_start": start}
    for m, c in t.items():
        p = f"cards/{m}/"
        for key in ("miners", "gpus", "groups", "groups_trained", "groups_rejected", "trained_tokens"):
            out[p + key] = c[key]
        out[p + "match_weight"] = c["rollouts"]           # denominator of rollout_match_mean: judged rollouts
        out[p + "rejection_weight"] = c["groups"]         # denominator of rejected_ratio: judged groups
        if c["groups"]:
            out[p + "rejected_ratio"] = c["groups_rejected"] / c["groups"]
            mean = c["match_sum"] / c["rollouts"]
            out[p + "rollout_match_mean"] = mean
            out[p + "rollout_match_min"] = min(c["match_min"], mean)   # float sums: the reader drops min > mean
        if share_total:
            out[p + "trained_token_share"] = c["share_tokens"] / share_total
    return out


def controller_metrics(rl: Any, now: float, window_s: float = METRICS_EVERY_S,
                       cards: CardFeed | None = None) -> dict[str, float]:
    """``_timestamp`` is ``now``: the end of the card window, exact rather than stamped later by wandb."""
    since = now - window_s
    out: dict[str, float] = {}
    for env, price in rl.prices.board().items():
        out[f"price/{env}/multiplier"] = price["multiplier"]
        out[f"price/{env}/per_million_tokens"] = price["per_million_tokens"]
        for key in ("ratio", "supply_tps", "demand_tps", "trainer_tps", "starved_s", "weight", "relative"):
            if isinstance(price["signal"].get(key), (int, float)):
                out[f"price/{env}/{key}"] = price["signal"][key]
    level = getattr(rl.prices, "level", None)
    if callable(level) and level() is not None:
        out["price/level"] = level()
    db = rl.book.db
    with rl.book.lock:
        speed = db.execute("SELECT tokens, seconds FROM rl_trainer_work WHERE id=0").fetchone()
        rows = {
            "buffer": db.execute("SELECT environment, state, COUNT(*) FROM rl_buffer GROUP BY 1, 2").fetchall(),
            "leases": db.execute("SELECT state, COUNT(*) FROM rl_leases WHERE state='open' OR closed_at>=? "
                                 "GROUP BY 1", (since,)).fetchall(),
            "issued": db.execute("SELECT environment, COUNT(*) FROM rl_leases WHERE issued_at>=? GROUP BY 1",
                                 (since,)).fetchall(),
            # unsubmitted leases by cause: released ones (our sandbox failed them) and charged ones
            "unsubmitted": db.execute("SELECT state, environment, cause, COUNT(*) FROM rl_leases WHERE state IN "
                                      "('expired', 'released') AND closed_at>=? GROUP BY 1, 2, 3",
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
        miners = len(active_hotkeys(db, now))
        banned = db.execute("SELECT COUNT(*) FROM rl_hotkey_bans WHERE until>?", (now,)).fetchone()[0]
        caps = [json.loads(state).get("cap", 0) for (state,) in db.execute("SELECT state FROM rl_lease_quota")]
    for env, state, n in rows["buffer"]:
        out[f"buffer/{env}/{state}"] = n
    for state, n in rows["leases"]:
        out[f"leases/{state}"] = n
    for env, n in rows["issued"]:
        out[f"leases_issued/{env}"] = n
    out["leases/sandbox_failure_released"] = 0
    for state, env, cause, n in rows["unsubmitted"]:
        out[f"leases_{state}/{env}/{cause or 'unknown'}"] = n
        if state == "released":
            out["leases/sandbox_failure_released"] += n
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
    if cards is not None:
        out.update(cards.row(rl, now, window_s))
    out["_timestamp"] = now
    return out


__all__ = ["METRICS_EVERY_S", "CardFeed", "card_class", "card_metrics", "controller_metrics", "instance_cards",
           "instance_classes"]
