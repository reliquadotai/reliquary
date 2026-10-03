"""Gate M2: replay recorded honest episodes and measure agreement.

    python scripts/agentic_replay_gate.py --traces /root/qual/27b-swesmith/runs/*/traces.jsonl \
        --out m2.json --concurrency 8
"""

import argparse
import asyncio
import json
import time

from reliquary.corpus.replay_compare import actions_from_trace, compare, normalize
from reliquary.validator.agentic_replay import replay_swe, swesmith_task


async def one(row, sem):
    trace = row["traces"][0]
    actions = actions_from_trace(trace)
    iid = row["task"]["data"]["instance_id"]
    async with sem:
        t0 = time.monotonic()
        try:
            task = swesmith_task(iid)
            observations, diff = await replay_swe(task, actions)
        except Exception as e:  # recorded, not fatal: the gate counts it
            return {"instance_id": iid, "error": repr(e)[:300]}
        seconds = time.monotonic() - t0
    report = compare(actions, observations, trace["info"].get("patch", ""), diff)
    return {"instance_id": iid, "actions": len(actions), "seconds": seconds,
            "diff_equal": report.diff_equal, "mismatched": report.mismatched,
            "samples": [{"index": i, "tool": actions[i].tool, "arguments": actions[i].arguments[:300],
                         "recorded": normalize(actions[i].observation)[:600],
                         "replayed": normalize(observations[i])[:600] if i < len(observations) else None}
                        for i in report.mismatched[:30]]}


async def main_async(args):
    rows = [json.loads(l) for path in args.traces for l in open(path)]
    rows = [r for r in rows if r.get("traces") and r["traces"][0].get("info", {}).get("patch") is not None]
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(one(r, sem) for r in rows))
    done = [r for r in results if "error" not in r]
    summary = {
        "gate": "M2", "episodes": len(rows), "replayed": len(done),
        "errors": len(results) - len(done),
        "diff_equal": sum(r["diff_equal"] for r in done),
        "observations": sum(r["actions"] for r in done),
        "observations_mismatched": sum(len(r["mismatched"]) for r in done),
        "replay_seconds_p50": sorted(r["seconds"] for r in done)[len(done) // 2] if done else None,
    }
    json.dump({"summary": summary, "episodes": results}, open(args.out, "w"))
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
