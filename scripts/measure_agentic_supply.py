"""Why an order job lacks miners: their throughput, their offers and their sandbox refusals.

Design: docs/design/2026-11-13-agentic-price-coefficient.md. Read-only; prints JSON, writes nothing. Three
optional sources, each answering one question:

* ``--job`` (repeatable; reads the job's manifest, records and verdicts from the bucket, credentials as the
  validator's): verified tokens per hour per miner, agentic (episode) jobs vs single-turn ones, and per model the
  ratio of the medians (what an agentic price factor would compensate).
* ``--auction-state`` (the control's ``RELIQUARY_AUCTION_STATE_DIR``, or one job's file) with ``--api`` (the
  public base URL: each job's ceiling from ``/corpus/pay``, its auction view): every live offer against the ceiling. An offer above it gets no lease
  (``ask_above_price``); a hotkey without an offer gets none either (``no_ask``).
* ``--log`` (repeatable; the order control's log, e.g. ``docker logs <control> > control.log``): sandbox session
  refusals by what was refused and why, and by hotkey prefix.

Lease refusals (``no_ask``, ``ask_above_price``, ``allowance_full``) are answered to the miner but not logged, so
the offers are read from the auction state instead.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import statistics
import sys
import time
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

# A miner silent longer than this between two verified records was idle: its rate counts only active sessions.
DEFAULT_IDLE_GAP_SECONDS = 1800.0
READS_AT_ONCE = 32
# sandbox/routes.py: "sandbox session %s refused for %s: %s"
SANDBOX_REFUSAL = re.compile(r"sandbox session (\S+) refused for (\S+): (\S+)")


def active_rate(points: Iterable[tuple[float, int]], idle_gap: float = DEFAULT_IDLE_GAP_SECONDS
                ) -> tuple[int, float]:
    """(tokens, seconds) of one miner's verified records ``(received_at, tokens)``, over its active sessions: a
    session ends at a gap over ``idle_gap``; its first record's tokens are left out (their work began before it)."""
    ordered = sorted(points)
    tokens, seconds = 0, 0.0
    previous = None
    for at, count in ordered:
        if previous is not None and at - previous <= idle_gap:
            tokens += int(count)
            seconds += at - previous
        previous = at
    return tokens, seconds


def miner_rates(records: Iterable[Mapping], idle_gap: float = DEFAULT_IDLE_GAP_SECONDS) -> dict[str, dict]:
    """Per hotkey: verified tokens, active hours and tokens per hour, from passed records' metadata."""
    by_hotkey: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for record in records:
        if record.get("received_at") is None:
            continue
        by_hotkey[str(record["hotkey"])].append((float(record["received_at"]), int(record["token_count"])))
    rates = {}
    for hotkey, points in by_hotkey.items():
        tokens, seconds = active_rate(points, idle_gap)
        rates[hotkey] = {"records": len(points), "tokens": sum(c for _, c in points),
                         "active_hours": round(seconds / 3600, 3),
                         "tokens_per_hour": round(tokens / seconds * 3600) if seconds > 0 else None}
    return rates


def summarize(jobs: list[dict]) -> dict:
    """Per model: the median miner rate of agentic and of single-turn jobs, and their ratio."""
    rates: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: {"agentic": [], "single_turn": []})
    for job in jobs:
        for miner in job["miners"].values():
            if miner["tokens_per_hour"]:
                rates[job["model"]][job["kind"]].append(miner["tokens_per_hour"])
    out = {}
    for model, kinds in rates.items():
        medians = {k: statistics.median(v) if v else None for k, v in kinds.items()}
        ratio = (medians["single_turn"] / medians["agentic"]
                 if medians["single_turn"] and medians["agentic"] else None)
        out[model] = {"miners": {k: len(v) for k, v in kinds.items()},
                      "median_tokens_per_hour": medians, "single_over_agentic": ratio}
    return out


def offers_against(state: Mapping, ceiling: Mapping | None, now: float, ttl: float) -> dict:
    """A job's live offers (auction state: pool share per token) in alpha per Mtoken, against its ceiling
    (``/corpus/pay``: ``max_price_alpha_per_mtoken`` and ``max_price_pool_share_per_token``)."""
    ceiling_alpha = (ceiling or {}).get("max_price_alpha_per_mtoken")
    ceiling_share = (ceiling or {}).get("max_price_pool_share_per_token")
    offers = []
    for hotkey, (share, capacity, posted_at) in (state.get("offers") or {}).items():
        if now - float(posted_at) > ttl:
            continue
        alpha = float(share) / float(ceiling_share) * float(ceiling_alpha) if ceiling_share and ceiling_alpha else None
        offers.append({"hotkey": hotkey[:12], "alpha_per_mtoken": alpha, "capacity": int(capacity),
                       "above_ceiling": (alpha is not None and ceiling_alpha is not None and alpha > ceiling_alpha)})
    offers.sort(key=lambda o: (o["alpha_per_mtoken"] is None, o["alpha_per_mtoken"] or 0.0))
    return {"ceiling_alpha_per_mtoken": ceiling_alpha, "budget": state.get("budget"),
            "live_offers": len(offers), "above_ceiling": sum(o["above_ceiling"] for o in offers),
            "capacity_offered": sum(o["capacity"] for o in offers), "offers": offers}


def sandbox_refusals(lines: Iterable[str]) -> dict:
    """Sandbox session refusals of a control log: by (what, reason) and by hotkey prefix."""
    by_reason: collections.Counter = collections.Counter()
    by_hotkey: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in lines:
        found = SANDBOX_REFUSAL.search(line)
        if found:
            what, hotkey, reason = found.groups()
            by_reason[f"{what}:{reason}"] += 1
            by_hotkey[hotkey][reason] += 1
    return {"by_reason": dict(by_reason.most_common()),
            "by_hotkey": {h: dict(c) for h, c in sorted(by_hotkey.items(), key=lambda kv: -sum(kv[1].values()))}}


async def job_records(store, job_id: str) -> list[dict]:
    """The metadata of a job's records whose verdict passed."""
    ids = await store.list_verdict_ids(job_id)
    gate = asyncio.Semaphore(READS_AT_ONCE)

    async def one(submission_id: str) -> dict | None:
        async with gate:
            verdict = await store.read_verdict(job_id, submission_id)
            if not verdict or not verdict.get("passed"):
                return None
            return await store.read_submission_meta(job_id, submission_id)

    return [m for m in await asyncio.gather(*(one(i) for i in ids)) if m]


async def measure_jobs(job_ids: list[str], idle_gap: float) -> list[dict]:
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    store = BucketRecordStore(max_pool_connections=READS_AT_ONCE)
    out = []
    for job_id in job_ids:
        spec, _ = await store.read_job(job_id)
        if spec is None:
            print(f"job {job_id}: no manifest", file=sys.stderr)
            continue
        records = await job_records(store, job_id)
        out.append({"job_id": job_id, "model": f"{spec.checkpoint_repo}@{spec.checkpoint_revision[:8]}",
                    "kind": "agentic" if spec.episode is not None else "single_turn",
                    "passed_records": len(records), "miners": miner_rates(records, idle_gap)})
    return out


def public_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=20) as answer:
            return json.load(answer)
    except Exception as exc:          # noqa: BLE001 -- a missing view only drops what it gives
        print(f"{url}: {exc!r}", file=sys.stderr)
        return None


def measure_offers(path: Path, api: str | None) -> dict:
    from reliquary.corpus.token_auction import ASK_TTL_SECONDS

    pay = public_json(f"{api.rstrip('/')}/corpus/pay") if api else None
    ceilings = {j.get("job_id"): j for j in (pay or {}).get("jobs") or []}
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out = {}
    for file in files:
        job_id = file.stem
        view = public_json(f"{api.rstrip('/')}/corpus/jobs/{job_id}/auction") if api else None
        out[job_id] = {**offers_against(json.loads(file.read_text()), ceilings.get(job_id), time.time(),
                                        ASK_TTL_SECONDS), "view": view}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", action="append", default=[], help="a job id (repeatable)")
    parser.add_argument("--idle-gap", type=float, default=DEFAULT_IDLE_GAP_SECONDS)
    parser.add_argument("--auction-state", type=Path, help="RELIQUARY_AUCTION_STATE_DIR or one job's file")
    parser.add_argument("--api", help="public base URL for the ceilings and auction views (https://api.reliqua.ai)")
    parser.add_argument("--log", action="append", default=[], type=Path, help="an order control log (repeatable)")
    args = parser.parse_args(argv)
    report: dict = {}
    if args.job:
        jobs = asyncio.run(measure_jobs(args.job, args.idle_gap))
        report["jobs"] = jobs
        report["by_model"] = summarize(jobs)
    if args.auction_state:
        report["offers"] = measure_offers(args.auction_state, args.api)
    if args.log:
        report["sandbox_refusals"] = sandbox_refusals(
            line for path in args.log for line in path.read_text(errors="replace").splitlines())
    json.dump(report, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
