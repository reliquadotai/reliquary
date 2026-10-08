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

# Matching runs on ONE normal form (``normalise``): lowercase, every run of non [a-z0-9] -> "-".
# Distinctive long names match with optional separators and no trailing boundary (``swebench``,
# ``livecodebenchv6``, ``gpqadiamond``); short or common-word names (``aime``, ``lcb``, ``gpqa``,
# ``tau2``) also need a trailing boundary: end, "-", a digit, "v<digit>" or a known suffix.
# Every pattern needs a leading boundary, so ``claimed`` / ``xaime`` are not found.
_LONG = ("swe-?bench", "mmlu-?pro", "terminal-?bench", "live-?code-?bench", "bfcl", "ifbench",
         "if-?eval-?bench", "tau-?bench", "tau2-?bench")
_SHORT = {"aime": r"(?=$|-|\d|v\d)", "lcb": r"(?=$|-|\d|v\d)", "tau-?2": r"(?=$|-|\d|bench)",
          "gpqa": r"(?=$|-|\d|v\d|diamond|main|extended)"}

HELD_OUT_BENCHMARKS = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(_LONG) + "|"
    + "|".join(p + t for p, t in _SHORT.items()) + ")")


def normalise(name: str) -> str:
    """The single normal form every guarded name (dataset id, card fields, ``org/repo`` ids) goes through."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def names_held_out_benchmark(name: str) -> bool:
    return HELD_OUT_BENCHMARKS.search(normalise(name)) is not None

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
    """``set_card`` is caller-supplied: it is NOT cryptographically tied to ``source_bytes`` (the contract
    only pins the dataset id and sha256), so this is a guard against honest mistakes by the offline admin,
    not proof of provenance. Where the contract's dataset carries ``source``/``split``, a card that
    disagrees is refused."""
    if not isinstance(set_card, dict):
        raise HeldOutEvalSet("curation needs the source set card")
    for name in _names(contract, set_card):
        if names_held_out_benchmark(name):
            raise HeldOutEvalSet(f"{name!r} names a held-out benchmark: such sets are never curated")
    if set_card.get("source_kind") != "catalog":
        raise HeldOutEvalSet("only catalog training-source sets can be curated")
    # Fail closed on an incomplete card: a split, a two-element index range and a disjointness
    # statement are all required (an absent field is not evidence of a training slice).
    window = set_card.get("index_range")
    if not isinstance(set_card.get("split"), str) or not set_card["split"]:
        raise HeldOutEvalSet("the set card needs a split")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise HeldOutEvalSet("the set card needs a two-element index_range")
    disjointness = set_card.get("disjointness")
    if not isinstance(disjointness, dict):
        raise HeldOutEvalSet("the set card needs a disjointness statement")
    if disjointness.get("external_benchmark") or disjointness.get("held_out"):
        raise HeldOutEvalSet("held-out evaluation regions are never curated")
    dataset = contract.to_dict()["dataset"]
    for key in ("source", "split"):   # the contract may carry the dataset's own provenance
        if key in dataset and key in set_card and dataset[key] != set_card[key]:
            raise HeldOutEvalSet(f"the set card {key} disagrees with the contract dataset")
    from reliquary.eval.sets import HELD_OUT, RL_SPLIT, range_overlaps
    source, split = set_card.get("source"), set_card.get("split")
    for held in HELD_OUT.values():
        if source == held.source and split == held.split and split != RL_SPLIT:
            raise HeldOutEvalSet(f"{source!r} split {split!r} is the {held.env!r} held-out evaluation set")
    if isinstance(source, str):
        if range_overlaps(source, split, int(window[0]), int(window[1]))["held_out"]:
            raise HeldOutEvalSet("rows overlap a held-out evaluation region")
