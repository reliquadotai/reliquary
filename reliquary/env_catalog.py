"""The environment catalog: what each environment is for, row by row.

One entry per environment Reliquary serves, keyed by the name a task or a job uses: a registry environment
(``reliquary_code_v1``) or an episode package (``reliquary-swe``). Each entry lists ``Rows``: a range of one
split and who may draw it.

* ``rl``: an RL task draws its prompts or tasks there.
* ``batch``: a batch job (a corpus job or a generation order, ours or a customer's) generates there; agentic
  traces come from an episode env's batch rows.
* ``eval``: an evaluation set is built there. Nothing else ever draws an eval row.

An environment's flags are the uses of its rows. Ranges of different uses never overlap, except where an entry
records a ruling (``shared``) that its RL and batch rows are knowingly the same rows.

The rest of the repository reads its lists from here: the RL prompt ranges (``protocol.rl_prompt_ranges``), the
rows a batch job may take (``eval.sets.refuse_held_out_overlap``), the held-out eval regions
(``eval.sets.HELD_OUT``), the generation-order sources (``eval.qualification.ORDER_ENVIRONMENTS``), the bridged
eval packages (``corpus.job.BRIDGED_EVAL_PACKAGES``) and the splits an RL episode env may draw
(``validator.rl_async_episodes``). ``catalog_document()`` is the copy the platform reads
(``reliquary envs catalog``).

This module imports nothing from the package: every other layer may read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

CATALOG_SCHEMA = "reliquary/env-catalog/v1"
USES = ("rl", "eval", "batch")
Use = Literal["rl", "eval", "batch"]
Kind = Literal["single_turn", "suite", "episode"]
ANY_SPLIT = "*"


@dataclass(frozen=True)
class Rows:
    """``[start, start + count)`` of ``split`` (``count`` None: to the end). ``live`` False: declared, but the
    path that would draw it does not serve it yet. ``eval_set``: the eval env name of a held-out region."""

    use: Use
    split: str
    start: int = 0
    count: int | None = None
    what: str = ""
    live: bool = True
    eval_set: str | None = None

    def __post_init__(self) -> None:
        if self.use not in USES:
            raise ValueError(f"unknown use {self.use!r}")
        if not self.split:
            raise ValueError("a split name is required")
        if self.start < 0 or (self.count is not None and self.count <= 0):
            raise ValueError("rows are a non-empty [start, start + count)")
        if self.eval_set is not None and self.use != "eval":
            raise ValueError("only eval rows name an eval set")

    @property
    def end(self) -> int | None:
        return None if self.count is None else self.start + self.count

    def bounds(self, split_length: int | None = None) -> tuple[int, float]:
        end = self.end if self.end is not None else (split_length if split_length is not None else float("inf"))
        return self.start, end

    def covers_split(self) -> bool:
        return self.start == 0 and self.count is None


@dataclass(frozen=True)
class CatalogEnv:
    name: str
    kind: Kind
    description: str
    rows: tuple[Rows, ...]
    # The pinned length of each split some range was cut against: a reinstalled source of another length
    # fails closed instead of offering shifted rows.
    lengths: Mapping[str, int] = field(default_factory=dict)
    # Why RL and batch knowingly draw the same rows, when they do.
    shared: str | None = None
    # A generation order may draw it (the order control renders and grades it; a decision per source).
    orders: bool = False
    # Served to evals through the sandbox's verifiers bridge (an episode package).
    bridged: bool = False

    @property
    def uses(self) -> frozenset[str]:
        return frozenset(r.use for r in self.rows)

    def rows_of(self, use: str, split: str | None = None) -> tuple[Rows, ...]:
        return tuple(r for r in self.rows if r.use == use and (split is None or r.split in (split, ANY_SPLIT)))


def _env(name: str, kind: Kind, description: str, *rows: Rows, **extra: Any) -> CatalogEnv:
    return CatalogEnv(name=name, kind=kind, description=description, rows=tuple(rows), **extra)


_CODE_LENGTH = 2_481_806
_CODE_HELD_OUT = 100_000

_ENTRIES = (
    _env(
        "reliquary_dapo_math_v1", "single_turn", "Competition math (DAPO-Math-17k), boxed answers.",
        Rows("rl", "train", what="RL prompts"),
        Rows("batch", "train", what="generation orders"),
        Rows("eval", "eval", eval_set="math", what=(
            "The package's eval split: problems are assigned to train/eval/qualification by a salted hash of "
            "the problem key, so no eval problem is a train problem. RL and corpus jobs build the train split "
            "only. OpenMathInstruct was not used: RL samples its whole index space, offset 10000 measured 43.6% "
            "trained, and its rows repeat each question ~20 times, so no index range is held out.")),
        shared="No split of train between RL and batch is decided: both draw the whole train split.",
        orders=True,
    ),
    _env(
        "reliquary_hard_math_v1", "single_turn", "Hard math in three difficulty tiers.",
        Rows("batch", "train", 0, 15_749, "tiers 1 and 2: the hard-math corpus jobs' prompts"),
        Rows("rl", "train", 15_749, None,
             "Rows [0, 15749) (tiers 1 and 2) are the hard-math corpus jobs' prompts; the third tier is RL's."),
        lengths={"train": 23_227},
    ),
    _env(
        "reliquary_science_v1", "single_turn", "Science questions.",
        Rows("batch", "train", what="both science corpus jobs"),
        Rows("rl", "train", what="RL prompts"),
        lengths={"train": 10_871},
        shared="Both science corpus jobs took the whole train split [0, 10871).",
    ),
    _env(
        "reliquary_instruction_following_v1", "single_turn", "Instruction following with checkable constraints.",
        Rows("batch", "train", 0, 20_000, "the instruction-following corpus job's prompts"),
        Rows("rl", "train", 20_000, None, "Rows [0, 20000) are the instruction-following corpus job's prompts."),
        Rows("eval", "eval", eval_set="instruction_following", what=(
            "The package's eval split (10%): rows are assigned by a salted hash of the row key, so no eval row "
            "is a train row. RL and corpus jobs (IF 0..20k) build the train split only.")),
        lengths={"train": 29_435},
        orders=True,
    ),
    _env(
        "reliquary_code_v1", "single_turn", "Python programming (OpenCodeInstruct curation), hidden tests.",
        Rows("batch", "train", 0, 200_000, "the two code corpus jobs' prompts"),
        Rows("rl", "train", 200_000, _CODE_LENGTH - _CODE_HELD_OUT - 200_000, (
            "Rows [0, 200000) are the two code corpus jobs' prompts, and the last 100,000 rows are the code "
            "evaluation set (reliquary.eval.sets.HELD_OUT).")),
        Rows("eval", "train", _CODE_LENGTH - _CODE_HELD_OUT, _CODE_HELD_OUT, eval_set="code", what=(
            "The last 100,000 rows of the pinned OpenCodeInstruct curation (d3caaefc, 2,481,806 rows), the "
            "corpus export's grader. The package has no other split. Disjoint from every corpus job (code "
            "0..100k), and `jobs create` refuses a job reaching it. NOT disjoint from past RL: RL samples the "
            "whole train split, so these rows were eligible (see the rulings).")),
        lengths={"train": _CODE_LENGTH},
        orders=True,
    ),
    _env(
        "reliquary_competitive_code_v1", "single_turn", "Competitive programming, judged on hidden tests.",
        Rows("batch", "train", 0, 4_181, "SFT: the competitive-code corpus jobs' prompts"),
        Rows("rl", "train", 4_181, 2_718, (
            "Train is SFT [0, 4181), RL [4181, 6899), then the v1.1 problems [6899, 8792), all SFT (the "
            "package's layout.use_ranges); the SFT rows are the competitive-code corpus jobs' prompts.")),
        Rows("batch", "train", 6_899, 1_893, "SFT: the v1.1 problems"),
        lengths={"train": 8_792},
    ),
    _env(
        "reliquary_logic_v2", "single_turn", "Generated logic puzzles answered in JSON.",
        Rows("rl", "train", what="RL prompts"),
        Rows("batch", "train", what="corpus jobs (rows 100M..100.05M) and generation orders"),
        Rows("eval", "eval", eval_set="logic", what=(
            "The generator's eval split: splits interleave (position = index*3 + split), so no eval task is a "
            "train task whatever the index. RL and corpus jobs (logic rows 100M..100.05M) build the train "
            "split only.")),
        shared="A generator: no split of train between RL and batch is decided, both draw any index.",
        orders=True,
    ),
    _env(
        "reliquary_general_v1", "single_turn", "General chat, safety, instruction and structured-output prompts.",
        Rows("rl", "train", what="RL prompts"),
        *(Rows("batch", "train", start, stop - start, f"{segment} single-turn (stage A)")
          for segment, start, stop in (
              ("chat/direct", 0, 30_141),
              ("chat/thinking", 30_141, 51_965),
              ("ifeval/direct", 66_291, 70_971),
              ("ifeval/thinking", 71_720, 82_808),
              ("structured/direct", 84_587, 86_967),
              ("structured/thinking", 87_540, 93_067),
              ("safety/direct", 94_476, 101_778),
              ("safety/thinking", 101_778, 107_165),
          )),
        shared=("RL draws the whole train split; corpus jobs take one stage-A single-turn segment each "
                "(counts from the package's `segments()` at ReliquaryForge/general-prompts-curated@02b3ef7a)."),
    ),
    _env(
        "envscaler_tools_v2", "suite", "Stateful tool-use worlds (EnvScaler), split by world.",
        Rows("rl", "train", 0, 1_296, (
            "The scenarios are split by world (the package's corpus layout): RL [0, 1296), 26 worlds no SFT row "
            "comes from; the rest are the SFT ranges and the corpus jobs' prompts.")),
        Rows("batch", "train", 7_225, 600, "SFT worlds kept for the corpus, graded"),
        lengths={"train": 7_825},
    ),
    _env(
        "reliquary_telecom_solo_v1", "suite", "Telecom support tickets, single agent (v1 layout).",
        Rows("rl", "train", what="RL prompts"),
    ),
    _env(
        "reliquary_telecom_solo_v2", "suite", "Telecom support tickets, single agent; SFT first then RL.",
        Rows("batch", "train", 0, 1_220, "telecom v2 SFT share"),
        Rows("rl", "train", 1_220, 493, "telecom v2 RL share (environment.toml [data] train_rl_start/count)"),
        lengths={"train": 1_713},
    ),
    _env(
        "reliquary-swe", "episode", "Software engineering in a sandboxed repository (bash harness).",
        Rows("batch", "train", 0, 9_276, "SWE-smith rows the SFT corpus took"),
        Rows("rl", "train", 9_276, 1_972, "SWE-smith RL share"),
        Rows("batch", "train", 11_248, 200, "pypika: the signed SWE corpus job (2026-10-14)"),
        Rows("rl", "train", 11_448, 7_098, "SWE-smith RL share"),
        Rows("batch", "r2e", what="R2E-Gym subset (10 Python repositories, none in Verified)", live=False),
        Rows("batch", "polyglot", what="multi-language repositories", live=False),
        Rows("eval", "eval", what="SWE-bench Verified"),
        lengths={"train": 18_546},
    ),
    _env(
        "reliquary-terminal", "episode", "Terminal tasks (TMax-15K) in a sandbox (bash harness).",
        Rows("rl", "tmax_rl", what="TMax RL share"),
        Rows("eval", "tmax_sft", what="never trained on: SFT v3 had no TMax and RL draws tmax_rl"),
        bridged=True,
    ),
    _env("reliquary-deepswe", "episode", "DeepSWE v1.1 (Datacurve's 113 tasks).",
         Rows("eval", ANY_SPLIT, what="eval only"), bridged=True),
    _env("reliquary-swebench-pro", "episode", "SWE-bench Pro V2 (Scale AI, public set).",
         Rows("eval", ANY_SPLIT, what="eval only"), bridged=True),
    _env("reliquary-nl2repo", "episode", "NL2Repo-Bench.",
         Rows("eval", ANY_SPLIT, what="eval only"), bridged=True),
    _env("reliquary-terminal-bench", "episode", "Terminal-Bench 2.1.",
         Rows("eval", ANY_SPLIT, what="eval only"), bridged=True),
)

CATALOG: Mapping[str, CatalogEnv] = MappingProxyType({e.name: e for e in _ENTRIES})

# Registry environments left out on purpose: legacy profiles' environments, which keep the rules they ran under.
LEGACY_ENVIRONMENTS = frozenset({
    "openmathinstruct", "opencodeinstruct", "reliquaryverifiable_v1", "reliquarylogic_v1",
    "reliquary_stateful_tools_v1", "reliquary_stateful_tools_v2", "reliquary_retrieval_tools_v1",
    "reliquary_workspace_tools_v1", "envscaler_tools_v1", "reliquary_automationbench_v1",
})


def _intervals(rows: tuple[Rows, ...], length: int | None) -> list[tuple[int, float]]:
    return sorted(r.bounds(length) for r in rows)


def check_catalog(catalog: Mapping[str, CatalogEnv] = CATALOG) -> None:
    """ValueError unless, in every split, eval rows overlap nothing and RL rows overlap batch rows only under a
    ``shared`` ruling; a bounded range must fit the split's pinned length."""
    for env in catalog.values():
        splits = {r.split for r in env.rows}
        for split in splits:
            mine = [r for r in env.rows if r.split in (split, ANY_SPLIT)]
            length = env.lengths.get(split)
            for r in mine:
                if length is not None and r.end is not None and r.end > length:
                    raise ValueError(f"{env.name}/{split}: [{r.start}, {r.end}) runs past {length} rows")
            for i, a in enumerate(mine):
                for b in mine[i + 1:]:
                    (a0, a1), (b0, b1) = a.bounds(length), b.bounds(length)
                    if not (a0 < b1 and b0 < a1):
                        continue
                    if a.use == b.use and a.use != "eval":
                        raise ValueError(f"{env.name}/{split}: two {a.use} ranges overlap")
                    if "eval" in (a.use, b.use):
                        raise ValueError(f"{env.name}/{split}: an eval range overlaps a {a.use if b.use == 'eval' else b.use} range")
                    if env.shared is None:
                        raise ValueError(f"{env.name}/{split}: RL and batch rows overlap without a ruling")


def entry(name: str) -> CatalogEnv | None:
    return CATALOG.get(name)


def package_name(env_package: str) -> str:
    """``reliquary-swe==0.3.0+g...`` -> ``reliquary-swe``."""
    return str(env_package).split("==", 1)[0]


def uses_split(name: str, split: str, use: str) -> bool:
    env = CATALOG.get(name)
    return env is not None and bool(env.rows_of(use, split))


def inside(name: str, use: str, split: str, start: int, end: int, *, join: bool = True) -> bool:
    """Whether every row of ``[start, end)`` of ``split`` lies in ``name``'s ``use`` ranges: adjacent ranges join,
    or, with ``join`` False, inside ONE of them."""
    env = CATALOG[name]
    if not join:
        return any(lo <= start and end <= hi for lo, hi in _intervals(env.rows_of(use, split), env.lengths.get(split)))
    covered = start
    for lo, hi in _intervals(env.rows_of(use, split), env.lengths.get(split)):
        if lo <= covered < hi:
            covered = hi
        if covered >= end:
            return True
    return covered >= end


def refuse_batch_rows(name: str, split: str, start: int, count: int) -> None:
    """A batch job on a catalog env takes rows of ONE of its batch ranges (a range is one job's renderer and
    layout segment). ValueError otherwise; an env outside the catalog is not judged here."""
    env = CATALOG.get(name)
    if env is None:
        return
    end = int(start) + int(count)
    if not env.rows_of("batch", split):
        raise ValueError(f"{name!r} split {split!r} serves no corpus job or generation order "
                         "(the environment catalog gives it no batch rows)")
    if not inside(name, "batch", split, int(start), end, join=False):
        ranges = ", ".join(f"[{r.start}, {'end' if r.end is None else r.end}) {r.what}"
                           for r in env.rows_of("batch", split))
        rl = [r for r in env.rows_of("rl", split) if r.start < end and int(start) < r.bounds(env.lengths.get(split))[1]]
        reach = f"; it reaches the RL share ({rl[0].what})" if rl else ""
        raise ValueError(f"{name!r} serves corpus jobs and generation orders only inside one of its batch ranges "
                         f"(the catalog's corpus ranges: {ranges}); [{start}, {end}) is not inside one{reach}")


def refuse_rl_rows(name: str, split: str, intervals: list[tuple[int, int]], *, whole_split: bool) -> None:
    """An RL task on a catalog episode package draws only its RL rows of ``split``. ``intervals`` are the rows it
    declares; ``whole_split``: it declares none, so it offers the whole split. ValueError otherwise, and for a
    package outside the catalog (an RL source is a decision)."""
    env = CATALOG.get(name)
    if env is None:
        raise ValueError(f"{name!r} is not in the environment catalog: no RL rows are declared for it")
    rl = env.rows_of("rl", split)
    if not rl:
        uses = sorted({r.use for r in env.rows if r.split in (split, ANY_SPLIT)}) or ["nothing"]
        raise ValueError(f"{name!r} split {split!r} is not an RL split (the catalog gives it to {', '.join(uses)})")
    if whole_split:
        if any(r.covers_split() for r in rl):
            return
        raise ValueError(f"{name!r} split {split!r}: RL draws only part of it; declare rows inside its RL rows")
    for start, end in intervals:
        if not inside(name, "rl", split, start, end):
            raise ValueError(f"{name!r} split {split!r}: rows [{start}, {end}) are not RL rows")


def catalog_document() -> dict:
    """The catalog as the platform reads it: one entry per environment, its flags and its ranges."""
    return {
        "schema": CATALOG_SCHEMA,
        "environments": [
            {
                "name": env.name, "kind": env.kind, "description": env.description,
                "uses": sorted(env.uses),
                "live_uses": sorted({r.use for r in env.rows if r.live}),
                "orders": env.orders, "bridged": env.bridged, "shared": env.shared,
                "lengths": dict(sorted(env.lengths.items())),
                "rows": [
                    {"use": r.use, "split": r.split, "start": r.start, "end": r.end, "what": r.what,
                     "live": r.live, **({"eval_set": r.eval_set} if r.eval_set else {})}
                    for r in env.rows
                ],
            }
            for env in sorted(CATALOG.values(), key=lambda e: e.name)
        ],
    }


check_catalog()

__all__ = ["ANY_SPLIT", "CATALOG", "CATALOG_SCHEMA", "LEGACY_ENVIRONMENTS", "USES", "CatalogEnv", "Rows",
           "catalog_document", "check_catalog", "entry", "inside", "package_name", "refuse_batch_rows",
           "refuse_rl_rows", "uses_split"]
