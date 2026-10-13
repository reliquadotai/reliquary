#!/usr/bin/env python3
"""Choose the scenarios ``envscaler_tools_v2`` serves, and pin the choice.

Writes ``reliquary/environment/agentic/envs/envscaler_tools_v2/selection.json``
from the fetched corpus (``scripts/fetch_envscaler_corpus.py``). The corpus is
not vendored; this list of positions is, with the sha256 of each corpus file
it was computed over, so the index space is reviewable in the repository and
reproducible from the pinned upstream revisions.

Every candidate (see ``corpus.candidates``: split by world, scenarios whose
task or initial state names OpenAI, ChatGPT or GPT already out) is run through the environment itself — the world process, the
runner, the grader — by a no-op agent whose first turn is a final answer.
A scenario is dropped when:

* ``init_error``: its world does not start (the process fails, or the
  constructor raises on its configuration);
* ``already_satisfied`` (graded ranges): the no-op agent wins it, i.e. no
  check is false in the initial state, so there is nothing to do;
* ``nondeterministic``: two independent processes disagree on the initial
  state's digest, so a validator could not replay it.

Usage::

    python scripts/filter_envscaler_scenarios.py --data ~/envscaler-corpus

Run from the repository root; the environment is imported from there.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.environment.agentic.envs.envscaler_tools_v2.corpus import (  # noqa: E402
    CORPUS_FILES,
    GRADED_RANGES,
    RANGE_ORDER,
    RANGE_SOURCE,
    SELECTION_PATH,
    SELECTION_SCHEMA,
    SFT_CORPUS_EXCLUDED_WORLDS,
    build_corpus,
    candidates,
    interleave,
    read_sources,
    scenario_mentions_vendor,
    sft_corpus_positions,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import (  # noqa: E402
    EnvScalerToolsV2Environment,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2.sandbox import (  # noqa: E402
    PROTOCOL,
)
from reliquary.environment.agentic.runner import (  # noqa: E402
    EpisodeRunner,
    ScriptedPolicy,
)
from reliquary.environment.agentic.types import AssistantAction  # noqa: E402


def _noop(environment: EnvScalerToolsV2Environment, index: int) -> tuple[str, str]:
    """(verdict, state digest) of one no-op episode."""
    task = environment.get_task(index)
    trace = EpisodeRunner().run(
        environment, task, seed=0,
        policy=ScriptedPolicy([AssistantAction.final("Done.")]),
    )
    report = trace.reward
    assert report is not None
    if report.environment_error is not None:
        return "init_error", report.state_digest
    scenario = environment.scenario(index)
    if scenario.range in GRADED_RANGES:
        required = [c for c in report.checks if c.name.startswith("check_") and c.weight > 0]
        if not required or report.reward == 1.0:
            return "already_satisfied", report.state_digest
    return "kept", report.state_digest


def _classify(environment: EnvScalerToolsV2Environment, index: int) -> str:
    verdict, first = _noop(environment, index)
    if verdict != "kept":
        return verdict
    _again, second = _noop(environment, index)
    return "kept" if first == second else "nondeterministic"


def render(selection: dict) -> str:
    """Indented JSON, each list of positions on one line: a reviewable diff."""
    text = json.dumps(selection, indent=1, sort_keys=True)
    return re.sub(
        r"\[(\s*-?\d+(?:,\s*-?\d+)*\s*)\]",
        lambda match: "[" + ",".join(match.group(1).split()).replace(",,", ",") + "]",
        text,
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SELECTION_PATH)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    sources = read_sources(args.data)
    split, pools = candidates(sources)
    # The filtered ranges; `sft_corpus` is derived from `sft_graded` after.
    filtered_ranges = [name for name in RANGE_ORDER if name in pools]
    unfiltered = build_corpus(
        sources, {name: [p for _w, p in pools[name]] for name in filtered_ranges}
    )
    environment = EnvScalerToolsV2Environment(unfiltered)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        verdicts = list(pool.map(
            lambda index: _classify(environment, index), range(len(unfiltered))
        ))
    elapsed = time.monotonic() - started

    vendor: dict[str, list[int]] = {}
    for name in filtered_ranges:
        records = sources[RANGE_SOURCE[name]]
        world_ids = set(split["rl"] if name == "rl" else split["sft"]) if name != "sft_ungraded" else None
        vendor[name] = [
            position for position, record in enumerate(records)
            if scenario_mentions_vendor(record)
            and (world_ids is None or record["env_id"] in world_ids)
        ]

    ranges: dict[str, dict] = {}
    dropped: dict[str, dict[str, list[int]]] = {}
    report: dict[str, Counter] = {}
    for name in filtered_ranges:
        start, end = unfiltered.ranges[name]
        kept: list[tuple[str, int]] = []
        drops: dict[str, list[int]] = {"vendor_mention": vendor[name]}
        counts: Counter = Counter({"candidates": end - start + len(vendor[name])})
        counts["vendor_mention"] = len(vendor[name])
        for index in range(start, end):
            scenario = unfiltered.scenarios[index]
            verdict = verdicts[index]
            counts[verdict] += 1
            if verdict == "kept":
                kept.append((scenario.env_id, scenario.position))
            else:
                drops.setdefault(verdict, []).append(scenario.position)
        positions = interleave(kept)
        ranges[name] = {
            "file": RANGE_SOURCE[name],
            "worlds": len({world for world, _p in kept}),
            "positions": positions,
        }
        dropped[name] = {reason: sorted(p) for reason, p in sorted(drops.items())}
        report[name] = counts

    # The corpus range: sft_graded's kept positions, in order, minus the
    # excluded worlds' (corpus.SFT_CORPUS_EXCLUDED_WORLDS).
    kept_corpus = sft_corpus_positions(
        sources[RANGE_SOURCE["sft_corpus"]], ranges["sft_graded"]["positions"]
    )
    ranges["sft_corpus"] = {
        "file": RANGE_SOURCE["sft_corpus"],
        "worlds": ranges["sft_graded"]["worlds"] - len(SFT_CORPUS_EXCLUDED_WORLDS),
        "positions": kept_corpus,
        "excluded_worlds": sorted(SFT_CORPUS_EXCLUDED_WORLDS),
    }
    dropped["sft_corpus"] = {}
    report["sft_corpus"] = Counter({
        "from_sft_graded": len(ranges["sft_graded"]["positions"]),
        "excluded_world": len(ranges["sft_graded"]["positions"]) - len(kept_corpus),
        "kept": len(kept_corpus),
    })

    selection = {
        "schema": SELECTION_SCHEMA,
        "sources": dict(CORPUS_FILES),
        "world_split": split,
        "range_order": list(RANGE_ORDER),
        "ranges": ranges,
        "dropped": dropped,
        "filter": {
            "script": "scripts/filter_envscaler_scenarios.py",
            "agent": "no-op: the first turn is a final answer",
            "worker_protocol": PROTOCOL,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    }
    args.out.write_text(render(selection), encoding="utf-8")

    offset = 0
    for name in RANGE_ORDER:
        count = len(ranges[name]["positions"])
        print(
            f"{name:<13} [{offset}, {offset + count})  kept {count} "
            f"over {ranges[name]['worlds']} worlds  "
            + "  ".join(f"{k}={v}" for k, v in sorted(report[name].items()))
        )
        offset += count
    print(f"{offset} scenarios, {elapsed:.0f}s, wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
