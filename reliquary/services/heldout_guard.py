"""Never curate prompts from held-out evaluation sets (decision K, ruling R11).

Three independent signals, any of which refuses:

* a name (dataset id, set id, source, env, taskset) that is one of the held-out
  benchmarks, matched on word boundaries so ``aime`` is not found in "claimed";
* a card that is not a catalog training slice, or that declares itself an external
  benchmark or reports held-out overlaps;
* a (source, split, index range) that overlaps a region ``reliquary.eval.sets.HELD_OUT``
  declares (the single place the repository declares its own held-out regions).

Offline/admin only: nothing here touches admission or payment.
"""
from __future__ import annotations

import re

from reliquary.protocol.service_contract import ServiceContract

# The seven Prime Tasksets of reliquary-environments ``benchmarks/heldout/configs/*.toml``
# (aime25, aime26, bfcl-v3, gpqa, ifbench, livecodebench, mmlu-pro) plus the rest of decision K's list.
HELD_OUT_BENCHMARK_NAMES: tuple[str, ...] = (
    "aime", "bfcl", "gpqa", "ifbench", "livecodebench", "lcb", "mmlu-pro", "mmlu_pro", "mmlupro",
    "tau2", "tau-bench", "tau_bench", "terminal-bench", "terminal_bench", "swe-bench-verified",
    "swe_bench_verified", "swebench-verified",
)

HELD_OUT_BENCHMARKS = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in HELD_OUT_BENCHMARK_NAMES) + r")(?![a-z])",
    re.IGNORECASE)

_NAME_FIELDS = ("set_id", "source", "env", "taskset", "name")


class HeldOutEvalSet(ValueError):
    pass


def _names(contract: ServiceContract, set_card: dict) -> list[str]:
    names = [str(contract.to_dict()["dataset"]["id"])]
    for key in _NAME_FIELDS:
        value = set_card.get(key)
        if isinstance(value, dict):
            value = " ".join(str(v) for v in value.values())
        if value is not None:
            names.append(str(value))
    return names


def refuse_held_out(contract: ServiceContract, set_card: dict) -> None:
    if not isinstance(set_card, dict):
        raise HeldOutEvalSet("curation needs the source set card")
    for name in _names(contract, set_card):
        if HELD_OUT_BENCHMARKS.search(name):
            raise HeldOutEvalSet(f"{name!r} names a held-out benchmark: such sets are never curated")
    if set_card.get("source_kind") != "catalog":
        raise HeldOutEvalSet("only catalog training-source sets can be curated")
    disjointness = set_card.get("disjointness")
    if isinstance(disjointness, dict) and (disjointness.get("external_benchmark") or disjointness.get("held_out")):
        raise HeldOutEvalSet("held-out evaluation regions are never curated")
    from reliquary.eval.sets import HELD_OUT, RL_SPLIT, range_overlaps
    source, split = set_card.get("source"), set_card.get("split")
    for held in HELD_OUT.values():
        if source == held.source and split == held.split and split != RL_SPLIT:
            raise HeldOutEvalSet(f"{source!r} split {split!r} is the {held.env!r} held-out evaluation set")
    window = set_card.get("index_range")
    if isinstance(window, (list, tuple)) and len(window) >= 2 and isinstance(source, str):
        if range_overlaps(source, split, int(window[0]), int(window[1]))["held_out"]:
            raise HeldOutEvalSet("rows overlap a held-out evaluation region")
