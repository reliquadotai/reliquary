"""Frozen evaluation sets: where each environment's held-out problems come from,
and `build-set`, which freezes them.

A set is three files. ``prompts.jsonl`` is all the pod ever sees;
``grading.jsonl`` stays on the admin host and names the source row each problem
was taken from; ``set.json`` is the card both sides read. Problems are written
in a seeded permutation of the held-out region, so an order's first N lines are
its sample and the same order on the same model gets the same problems.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SET_SCHEMA = "reliquary/eval-set/v1"
RL_SPLIT = "train"
_SET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class UsedRange:
    """Rows of a source someone else consumes. ``count`` None runs to the end."""

    kind: str  # "corpus", "rl" or "eval"
    source: str
    split: str
    start: int
    count: int | None
    what: str

    @property
    def end(self) -> int | None:
        return None if self.count is None else self.start + self.count


@dataclass(frozen=True)
class HeldOut:
    env: str
    source: str
    split: str
    justification: str
    start: int = 0
    # None: the whole split.
    count: int | None = None
    # The source length the range was chosen against; build-set refuses another.
    source_length: int | None = None

    @property
    def region(self) -> UsedRange:
        return UsedRange("eval", self.source, self.split, self.start, self.count, self.env)


# Every corpus job running in prod (2026-09-30), on the train split like every
# corpus job. `prompt_start`/`prompt_count` as their manifests declare them.
CORPUS_RANGES: tuple[UsedRange, ...] = (
    UsedRange("corpus", "reliquary_code_v1", "train", 0, 100_000, "code-qwen38-27b-v1"),
    UsedRange("corpus", "openmathinstruct", "train", 10_000_000, 400_000,
              "math-omi-qwen38-27b-v1"),
    UsedRange("corpus", "reliquary_instruction_following_v1", "train", 0, 20_000,
              "if-qwen38-27b-v1"),
    UsedRange("corpus", "reliquary_logic_v2", "train", 100_000_000, 50_000,
              "logic-qwen38-27b-v1"),
)

_CODE_LENGTH = 2_481_806
_CODE_HELD_OUT = 100_000

HELD_OUT: dict[str, HeldOut] = {
    "math": HeldOut(
        env="math", source="reliquary_dapo_math_v1", split="eval",
        justification=(
            "The package's eval split: problems are assigned to train/eval/qualification by "
            "a salted hash of the problem key, so no eval problem is a train problem. RL "
            "and corpus jobs build the train split only. OpenMathInstruct was not used: RL "
            "samples its whole index space, offset 10000 measured 43.6% trained, and its "
            "rows repeat each question ~20 times, so no index range is held out."),
    ),
    "code": HeldOut(
        env="code", source="reliquary_code_v1", split="train",
        start=_CODE_LENGTH - _CODE_HELD_OUT, count=_CODE_HELD_OUT, source_length=_CODE_LENGTH,
        justification=(
            "The last 100,000 rows of the pinned OpenCodeInstruct curation (d3caaefc, "
            "2,481,806 rows), the corpus export's grader. The package has no other split. "
            "Disjoint from every corpus job (code 0..100k), and `jobs create` refuses a job "
            "reaching it. NOT disjoint from past RL: RL samples the whole train split, so "
            "these rows were eligible (see the rulings)."),
    ),
    "logic": HeldOut(
        env="logic", source="reliquary_logic_v2", split="eval",
        justification=(
            "The generator's eval split: splits interleave (position = index*3 + split), so "
            "no eval task is a train task whatever the index. RL and corpus jobs (logic "
            "rows 100M..100.05M) build the train split only."),
    ),
    "instruction_following": HeldOut(
        env="instruction_following", source="reliquary_instruction_following_v1",
        split="eval",
        justification=(
            "The package's eval split (10%): rows are assigned by a salted hash of the row "
            "key, so no eval row is a train row. RL and corpus jobs (IF 0..20k) build the "
            "train split only."),
    ),
}

# Environments whose held-out region RL could sample, each by a ruling.
RL_OVERLAP_RULED: dict[str, str] = {
    "code": ("ruled: RL samples the whole train split of reliquary_code_v1 and of "
             "opencodeinstruct (the same curation, the same row indices), so these rows "
             "were eligible for past RL; no split of the source exists. Excluding them from "
             "RL sampling is a follow-up."),
}
# What a customer reads beside an environment's score when RL may have seen it.
CONTAMINATION_NOTES: dict[str, str] = {
    "code": ("These problems were eligible for Reliquary's RL training (OpenCodeInstruct "
             "rows sampled by reliquary_code_v1 and opencodeinstruct). Scores of models "
             "trained by Reliquary may be inflated; other models are unaffected by this."),
}


def rl_ranges() -> list[UsedRange]:
    """RL's prompt universe: the whole train split of every catalog source
    (`window_prompt_range` slices [0, len(env)), and runtime factories build
    the train split)."""
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

    return [UsedRange("rl", source, RL_SPLIT, 0, None, "RL prompt sampling")
            for source in sorted(ENVIRONMENT_CATALOG)]


# Source -> (the corpus its rows come from, its index space). Two sources of
# one corpus may share content; when they also share an index space, row i is
# the same problem in both (reliquary_code_v1 says so of opencodeinstruct).
SOURCE_LINEAGE: dict[str, tuple[str, str]] = {
    "opencodeinstruct": ("nvidia/OpenCodeInstruct", "R0mAI/opencodeinstruct-curated@d3caaefc"),
    "reliquary_code_v1": ("nvidia/OpenCodeInstruct", "R0mAI/opencodeinstruct-curated@d3caaefc"),
    "openmathinstruct": ("nvidia/OpenMathInstruct-2", "openmathinstruct"),
    "reliquary_dapo_math_v1": ("BytedTsinghua-SIA/DAPO-Math-17k", "reliquary_dapo_math_v1"),
    "reliquarylogic_v1": ("reliquary-logic-generator-v1", "reliquarylogic_v1"),
    "reliquary_logic_v2": ("reliquary-logic-generator-v2", "reliquary_logic_v2"),
    "reliquary_instruction_following_v1": ("nvidia/Nemotron-Cascade-2-RL-data:IF-RL",
                                           "reliquary_instruction_following_v1"),
}


def lineage(source: str) -> tuple[str, str]:
    return SOURCE_LINEAGE.get(source, (source, source))


def overlaps(a: UsedRange, b: UsedRange) -> bool:
    """Whether two ranges can hold one problem. Sources of one corpus with
    different index spaces, or splits not comparable, are taken to overlap."""
    (corpus_a, space_a), (corpus_b, space_b) = lineage(a.source), lineage(b.source)
    if corpus_a != corpus_b:
        return False
    if space_a != space_b:
        return True
    if a.split != b.split:
        # Splits of one source partition it; across sources they say nothing.
        return a.source != b.source
    a_end = float("inf") if a.end is None else a.end
    b_end = float("inf") if b.end is None else b.end
    return a.start < b_end and b.start < a_end


def refuse_held_out_overlap(source: str, prompt_start: int, prompt_count: int) -> None:
    """A corpus job (train split) may not take rows an eval set holds out."""
    job = UsedRange("corpus", source, RL_SPLIT, int(prompt_start), int(prompt_count), "job")
    for held in HELD_OUT.values():
        # Only an index range can be refused; a source of the same corpus with
        # another index space is a lineage question, not a range (rulings).
        if lineage(source)[1] == lineage(held.source)[1] and overlaps(job, held.region):
            raise ValueError(
                f"rows [{held.start}, {held.region.end}) of {source!r} are held out for the "
                f"{held.env!r} evaluation set; this job's [{job.start}, {job.end}) reaches them"
            )


def _disjointness(held: HeldOut) -> dict:
    corpus = [r for r in CORPUS_RANGES if lineage(r.source)[0] == lineage(held.source)[0]]
    rl = [r for r in rl_ranges() if lineage(r.source)[0] == lineage(held.source)[0]]
    hit = [r for r in corpus if overlaps(held.region, r)]
    if hit:
        raise ValueError(f"{held.env!r} held-out region overlaps corpus job {hit[0].what!r}")
    if any(overlaps(held.region, r) for r in rl):
        if held.env not in RL_OVERLAP_RULED:
            raise ValueError(f"{held.env!r} held-out region overlaps RL sampling, unruled")
        rl_verdict = RL_OVERLAP_RULED[held.env]
    else:
        rl_verdict = f"disjoint: RL builds the {RL_SPLIT!r} split only"
    return {
        "justification": held.justification,
        "corpus": "disjoint",
        "rl": rl_verdict,
        "contamination_note": CONTAMINATION_NOTES.get(held.env),
        "checked_against": [
            {"kind": r.kind, "what": r.what, "source": r.source, "split": r.split,
             "start": r.start, "end": r.end} for r in corpus + rl
        ],
    }


def open_source(source: str, split: str):
    """The catalog environment at ``split``: the runtime factory for train, the
    package's own split otherwise (an external answer environment)."""
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    spec = ENVIRONMENT_SPECS[source]
    if split == RL_SPLIT:
        return spec.create()
    if spec.external_distribution is None or spec.interaction_mode != "single_turn":
        raise ValueError(f"{source!r} has no {split!r} split")
    from reliquary.environment.agentic.external import (
        ExternalAnswerEnvironment,
        load_external_backend,
    )

    return ExternalAnswerEnvironment(load_external_backend(spec, split=split), spec)


def _jsonl(rows: Iterable[dict]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                    for row in rows)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def validated_set_id(set_id: Any) -> str:
    if not isinstance(set_id, str) or not _SET_ID_RE.fullmatch(set_id):
        raise ValueError(f"set id {set_id!r} is not a name")
    return set_id


def build_set(env: str, *, count: int, seed: int, out: str | Path,
              set_id: str | None = None,
              open_environment: Callable[[str, str], Any] = open_source,
              clock: Callable[[], float] = time.time) -> dict:
    """Freeze ``count`` problems of ``env``'s held-out region into ``out``."""
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

    if env not in HELD_OUT:
        raise ValueError(f"no held-out region for {env!r}; known: {sorted(HELD_OUT)}")
    if count <= 0:
        raise ValueError("count must be positive")
    held = HELD_OUT[env]
    set_id = validated_set_id(set_id or f"{env.replace('_', '-')}-{held.split}-s{seed}-n{count}")
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty: a set is frozen once")
    disjointness = _disjointness(held)
    environment = open_environment(held.source, held.split)
    length = len(environment)
    if held.source_length is not None and length != held.source_length:
        raise ValueError(f"{held.source!r} has length {length}, but its held-out range was "
                         f"chosen against length {held.source_length}")
    low = held.start
    high = length if held.count is None else held.start + held.count
    if high > length:
        raise ValueError(f"held-out range [{low}, {high}) runs past {held.source!r} ({length})")
    if count > high - low:
        raise ValueError(f"{env!r} holds {high - low} problems, fewer than {count}")
    indices = random.Random(seed).sample(range(low, high), count)
    prompts, grading = [], []
    for ordinal, index in enumerate(indices):
        problem = environment.get_problem(index)
        prompt = problem.get("prompt") if isinstance(problem, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{held.source!r} row {index} has no prompt")
        problem_id = f"{set_id}-{ordinal:06d}"
        prompts.append({"problem_id": problem_id, "env": env, "set_id": set_id,
                        "messages": [{"role": "user", "content": prompt}]})
        grading.append({"problem_id": problem_id, "source": held.source, "split": held.split,
                        "source_index": index, "prompt_sha256": prompt_sha256(prompt)})
    prompts_body, grading_body = _jsonl(prompts), _jsonl(grading)
    profile = ENVIRONMENT_CATALOG[held.source]
    card = {
        "schema": SET_SCHEMA, "set_id": set_id, "env": env, "source": held.source,
        "split": held.split, "index_range": [low, high], "source_length": length,
        "count": count, "seed": seed,
        "order": ("problems are a seeded permutation of index_range "
                  "(random.Random(seed).sample); an order of N problems takes the first N"),
        "prompts_sha256": hashlib.sha256(prompts_body).hexdigest(),
        "grading_sha256": hashlib.sha256(grading_body).hexdigest(),
        "created_at": clock(),
        "prompt_template_id": getattr(profile.prompt_template, "template_id", None),
        "default_max_new_tokens": profile.max_new_tokens,
        "environment_manifest_sha256": getattr(profile, "environment_manifest_sha256", None),
        "disjointness": disjointness,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompts.jsonl").write_bytes(prompts_body)
    (directory / "grading.jsonl").write_bytes(grading_body)
    (directory / "set.json").write_text(json.dumps(card, sort_keys=True, indent=1))
    return card


__all__ = [
    "CORPUS_RANGES",
    "HELD_OUT",
    "HeldOut",
    "CONTAMINATION_NOTES",
    "RL_OVERLAP_RULED",
    "SOURCE_LINEAGE",
    "lineage",
    "SET_SCHEMA",
    "UsedRange",
    "build_set",
    "open_source",
    "overlaps",
    "prompt_sha256",
    "refuse_held_out_overlap",
    "rl_ranges",
    "validated_set_id",
]
