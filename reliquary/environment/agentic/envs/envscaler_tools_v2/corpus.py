"""The ``envscaler_tools_v2`` index space: which scenario every index names.

The corpus is fetched, not vendored (``scripts/fetch_envscaler_corpus.py``),
so this module carries only the decisions taken over it, and pins the inputs
they were taken against:

**Split by world.** The 51 worlds that have RL scenarios are sorted by their
number (``env_<n>_rl``) and dealt alternately: even rank to RL (26 worlds),
odd rank to SFT (25 worlds). No world contributes to both, so a policy trained
on the SFT corpus has never seen an RL world's tools or rules. Alternating
rather than hashing gives exact, reviewable counts; the world numbers carry no
meaning upstream, so alternation is as unbiased as a salt.

**Three ranges, contiguous, in this order** (boundaries in ``environment``):

* ``rl`` — scenarios of the 26 RL worlds, graded;
* ``sft_graded`` — scenarios of the 25 SFT worlds, graded the same way;
* ``sft_ungraded`` — EnvScaler-SFT-Scenario over the 140 ``_sft`` worlds,
  which ships tasks and initial states but no checks. Reward is always 0 here;
  a corpus job over this range must declare no reward filter.

Within a range the order is round-robin over worlds (each world's k-th
scenario before any world's k+1-th), so any contiguous slice — a corpus job's
``prompt_start``/``prompt_count``, an RL window's prompt range — draws evenly
from every world instead of from one or two.

**Which scenarios** is ``selection.json``, written by
``scripts/filter_envscaler_scenarios.py``: candidates minus those whose task or
initial state names OpenAI, ChatGPT or GPT, minus those whose world fails to start, minus
(graded) those a no-op agent already wins, minus those whose initial state is
not reproducible across processes. It records the sha256 of each corpus file
it was computed over; the loader refuses any other bytes, because a different
file under the same positions would silently renumber the space.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

DATA_ENV_VAR = "RELIQUARY_ENVSCALER_DATA"
SELECTION_PATH = Path(__file__).with_name("selection.json")
SELECTION_SCHEMA = "reliquary/envscaler-selection/v1"

# Normalized bytes `scripts/fetch_envscaler_corpus.py` writes, by file name.
CORPUS_FILES: Mapping[str, str] = MappingProxyType({
    "env_meta.json": (
        "f0e874c28b3592c5820afcdbae28d3c1"
        "0d7687632d5e1f922a6a68a2d904f81d"
    ),
    "rl_scen.json": (
        "a953b969366d0ad0dc6e5262084fc7d4"
        "6b6cefb992bd02ac5083fec537622e06"
    ),
    "sft_scen.json": (
        "f3af68f6a1f50c6ff7be6fba9d605dcd"
        "9dfdf1a80b4d14d61f5a08d4bd484e0e"
    ),
})

RANGE_ORDER = ("rl", "sft_graded", "sft_ungraded")
GRADED_RANGES = frozenset({"rl", "sft_graded"})
# Which file each range's positions index into.
RANGE_SOURCE: Mapping[str, str] = MappingProxyType({
    "rl": "rl_scen.json",
    "sft_graded": "rl_scen.json",
    "sft_ungraded": "sft_scen.json",
})

# Scenarios that name a vendor's model — in the task, or in the initial state
# the agent can read back through its tools — are dropped: a policy trained on
# them learns to talk about a competitor's assistant. Word-bounded, so "gpt"
# inside another token does not count. Measured: no task text matches; eight
# initial states do (bookmarks of an OpenAI blog, forum posts about GPT-4).
_VENDOR = re.compile(r"open\s*ai|chat\s*gpt|\bgpt\b|\bgpt-?\d", re.IGNORECASE)
_WORLD_NUMBER = re.compile(r"^env_(\d+)_rl$")


def mentions_vendor(text: str) -> bool:
    return bool(_VENDOR.search(text or ""))


def scenario_mentions_vendor(scenario: Mapping[str, Any]) -> bool:
    """The task, or anything in the initial state the agent could read back."""
    return mentions_vendor(str(scenario.get("task", ""))) or mentions_vendor(
        json.dumps(scenario.get("init_config"), ensure_ascii=False, sort_keys=True)
    )


def split_worlds(world_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(RL worlds, SFT worlds): sorted by number, even rank RL, odd rank SFT."""
    numbered = []
    for world_id in set(world_ids):
        match = _WORLD_NUMBER.match(world_id)
        if match is None:
            raise ValueError(f"not an RL world id: {world_id!r}")
        numbered.append((int(match.group(1)), world_id))
    ordered = [world_id for _number, world_id in sorted(numbered)]
    return tuple(ordered[0::2]), tuple(ordered[1::2])


def interleave(entries: Sequence[tuple[str, int]]) -> list[int]:
    """Positions round-robin over worlds, first appearance deciding world order.

    ``entries`` is (world id, position) in upstream file order; within one
    world, upstream order is kept.
    """
    queues: dict[str, list[int]] = {}
    for world_id, position in entries:
        queues.setdefault(world_id, []).append(position)
    ordered: list[int] = []
    depth = 0
    while True:
        row = [queue[depth] for queue in queues.values() if depth < len(queue)]
        if not row:
            return ordered
        ordered.extend(row)
        depth += 1


@dataclass(frozen=True)
class Scenario:
    range: str
    source: str
    position: int
    task_id: str
    env_id: str
    task: str
    init_config: Mapping[str, Any]
    # None for the ungraded range, which ships no checks.
    checks: tuple[Mapping[str, Any], ...] | None


@dataclass(frozen=True)
class Corpus:
    worlds: Mapping[str, Mapping[str, Any]]
    scenarios: tuple[Scenario, ...]
    ranges: Mapping[str, tuple[int, int]]

    def __len__(self) -> int:
        return len(self.scenarios)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sources(
    root: Path, *, verify: bool = True
) -> dict[str, list[dict[str, Any]]]:
    """The three normalized corpus files, refused unless they are the pinned bytes."""
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, expected in CORPUS_FILES.items():
        path = Path(root) / name
        if not path.is_file():
            raise RuntimeError(
                f"{path} is missing; rebuild the corpus with "
                "scripts/fetch_envscaler_corpus.py"
            )
        if verify:
            actual = _sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"{path} is not the pinned corpus: sha256 {actual}, "
                    f"expected {expected}"
                )
        records = json.loads(path.read_text(encoding="utf-8"))
        loaded[name] = list(records.values()) if isinstance(records, dict) else records
    return loaded


def candidates(
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, list[tuple[str, int]]]]:
    """Every scenario a range could hold, before the quality filter.

    Returns the world split and, per range, (world id, position) in upstream
    order. Scenarios naming a vendor are already out; the filter script
    reports how many.
    """
    worlds = {world["env_id"]: world for world in sources["env_meta.json"]}
    rl_scenarios = sources["rl_scen.json"]
    rl_world_ids = sorted({s["env_id"] for s in rl_scenarios if s["env_id"] in worlds})
    rl_worlds, sft_worlds = split_worlds(rl_world_ids)
    rl_set, sft_set = set(rl_worlds), set(sft_worlds)
    ranges: dict[str, list[tuple[str, int]]] = {name: [] for name in RANGE_ORDER}
    for position, scenario in enumerate(rl_scenarios):
        if scenario_mentions_vendor(scenario):
            continue
        if scenario["env_id"] in rl_set:
            ranges["rl"].append((scenario["env_id"], position))
        elif scenario["env_id"] in sft_set:
            ranges["sft_graded"].append((scenario["env_id"], position))
    for position, scenario in enumerate(sources["sft_scen.json"]):
        if scenario["env_id"] in worlds and not scenario_mentions_vendor(scenario):
            ranges["sft_ungraded"].append((scenario["env_id"], position))
    split = {
        "rule": (
            "RL-scenario worlds sorted by the number in env_<n>_rl; "
            "even rank -> rl, odd rank -> sft"
        ),
        "rl": list(rl_worlds),
        "sft": list(sft_worlds),
    }
    return split, ranges


def build_corpus(
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    positions: Mapping[str, Sequence[int]],
) -> Corpus:
    """The index space: each range's positions, in ``RANGE_ORDER``."""
    worlds = {world["env_id"]: world for world in sources["env_meta.json"]}
    scenarios: list[Scenario] = []
    bounds: dict[str, tuple[int, int]] = {}
    for name in RANGE_ORDER:
        start = len(scenarios)
        records = sources[RANGE_SOURCE[name]]
        for position in positions.get(name, ()):
            record = records[int(position)]
            if record["env_id"] not in worlds:
                raise ValueError(f"{name}[{position}] names an unknown world")
            checks = None
            if name in GRADED_RANGES:
                checks = tuple(dict(entry) for entry in record["checklist_with_func"])
            scenarios.append(Scenario(
                range=name,
                source=RANGE_SOURCE[name],
                position=int(position),
                task_id=str(record["task_id"]),
                env_id=str(record["env_id"]),
                task=str(record["task"]),
                init_config=dict(record["init_config"]),
                checks=checks,
            ))
        bounds[name] = (start, len(scenarios))
    return Corpus(
        worlds=MappingProxyType(worlds),
        scenarios=tuple(scenarios),
        ranges=MappingProxyType(bounds),
    )


def read_selection(path: Path = SELECTION_PATH) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"{path} is not a {SELECTION_SCHEMA} document")
    if dict(selection.get("sources") or {}) != dict(CORPUS_FILES):
        raise ValueError(
            f"{path} was computed over other corpus bytes than the ones pinned here"
        )
    return selection


def selected_positions(selection: Mapping[str, Any]) -> dict[str, list[int]]:
    return {
        name: [int(p) for p in selection["ranges"][name]["positions"]]
        for name in RANGE_ORDER
    }


@lru_cache(maxsize=1)
def default_corpus() -> Corpus:
    configured = os.environ.get(DATA_ENV_VAR)
    if not configured:
        raise RuntimeError(f"{DATA_ENV_VAR} must point at the pinned EnvScaler release")
    selection = read_selection()
    return build_corpus(
        read_sources(Path(configured)), selected_positions(selection)
    )


__all__ = [
    "CORPUS_FILES",
    "Corpus",
    "DATA_ENV_VAR",
    "GRADED_RANGES",
    "RANGE_ORDER",
    "RANGE_SOURCE",
    "SELECTION_PATH",
    "Scenario",
    "build_corpus",
    "candidates",
    "default_corpus",
    "interleave",
    "mentions_vendor",
    "read_selection",
    "read_sources",
    "scenario_mentions_vendor",
    "selected_positions",
    "split_worlds",
]
