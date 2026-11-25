"""Chained evaluations: a benchmark whose set at position k is built from the model's graded answers at k - 1.

SciCode (``verifiers:scicode-steps``, ``tasksets/scicode_steps``) asks each sub-step in a fresh single-turn prompt
that carries the model's own code for the earlier steps, so one evaluation is a chain of sets, one per step
position (design: docs/design/2026-11-14-scicode-chained-orders.md). An order on it runs one eval job per
position, one after the other; the platform asks here for each next set:

* ``next_set``: from position k's grading, the set of position k + 1 (its taskset args are position k's with the
  answers recorded), published to the subnet bucket only (create-only: asked again, it answers the set already
  there), with its prompt lengths in the model's chat template (the job's ``max_prompt_tokens``);
* ``chain_report``: the chain's own rates over every position graded, checked to be one chain (each position's
  set built from the answers graded before it), for the order's combined report.

Each position's set is an ordinary single-turn set: its job, forced draw, audit and grading are the ones any set
has. Nothing here re-grades: answers and verdicts are read from each position's graded rows.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "reliquary/eval-chain/v1"
# A position goes on when at least this share of its steps were graded (a missing or ungraded answer fails its
# step, as upstream counts a missing answer); at least one step may always be missing. Under it something
# systemic happened (the grader down, the job cut early): the chain stops instead of carrying blanks forward.
MIN_GRADED_ENV = "RELIQUARY_CHAIN_MIN_GRADED"
DEFAULT_MIN_GRADED = 0.9


def min_graded() -> float:
    import os

    raw = os.environ.get(MIN_GRADED_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_MIN_GRADED
    except ValueError:
        value = DEFAULT_MIN_GRADED
    return min(1.0, max(0.0, value))


def missing_allowed(asked: int, share: float | None = None) -> int:
    """How many of a position's ``asked`` steps may be missing or ungraded: ``(1 - share) * asked``, at least one."""
    share = min_graded() if share is None else share
    return max(1, int((1.0 - share) * asked + 1e-9))


@dataclass(frozen=True)
class Chain:
    source: str
    """The Verifiers source of every position's set."""
    module: str
    """The taskset's chain bookkeeping: ``asked_at``, ``next_args``, ``max_position``, ``summary``."""


CHAINS = {"scicode-steps": Chain(source="verifiers:scicode-steps", module="scicode_steps.chain")}


class ChainRefused(ValueError):
    """The request names no chain step that can be served (another set, a finished chain, a broken link)."""


class PartNotGraded(LookupError):
    """A position has no grading yet."""


@dataclass
class _Position:
    eval_id: str
    report: dict
    set_id: str
    card: dict
    grading: list[dict]
    args: dict


def chain_of(chain_id: str) -> Chain:
    try:
        return CHAINS[chain_id]
    except KeyError:
        raise ChainRefused(f"chain_unknown: {chain_id!r}; chains: {sorted(CHAINS)}") from None


def _module(chain: Chain):
    return importlib.import_module(chain.module)


async def _position(platform, subnet, chain: Chain, eval_id: str) -> _Position:
    from reliquary.eval.grading import evaluation_prefix
    from reliquary.eval.storage import subnet_key

    prefix = evaluation_prefix(eval_id)
    report = await platform.get_json(f"{prefix}/report.json")
    if report is None or await platform.get_json(f"{prefix}/manifest.json") is None:
        raise PartNotGraded(eval_id)
    sets = (report.get("provenance") or {}).get("sets") or []
    if len(sets) != 1:
        raise ChainRefused(f"{eval_id} graded {len(sets)} sets: a chain position is one set")
    set_id = sets[0]["set_id"]
    card_body = await subnet.get_bytes(subnet_key(set_id, "set.json"))
    grading_body = await subnet.get_bytes(subnet_key(set_id, "grading.jsonl"))
    if card_body is None or grading_body is None:
        raise ChainRefused(f"set {set_id} is not published")
    card = json.loads(card_body)
    if hashlib.sha256(grading_body).hexdigest() != card.get("grading_sha256"):
        raise ChainRefused(f"set {set_id}: grading.jsonl does not match its card")
    taskset = card.get("taskset") or {}
    if card.get("source") != chain.source or f"verifiers:{taskset.get('id')}" != chain.source:
        raise ChainRefused(f"set {set_id} is a {card.get('source')!r} set, not {chain.source!r}")
    grading = [json.loads(line) for line in grading_body.decode().splitlines() if line.strip()]
    return _Position(eval_id, report, set_id, card, grading, dict(taskset.get("args") or {}))


async def _graded_rows(platform, eval_id: str) -> list[dict]:
    import pyarrow.parquet as pq

    from reliquary.eval.grading import evaluation_prefix

    with tempfile.TemporaryDirectory() as scratch:
        local = Path(scratch) / "graded.parquet"
        if not await platform.get_file(f"{evaluation_prefix(eval_id)}/graded.parquet", local):
            raise PartNotGraded(eval_id)
        table = await asyncio.to_thread(pq.read_table, local,
                                        columns=["problem_id", "sample_index", "completion", "correct"])
    return table.to_pylist()


@dataclass
class _Answers:
    answers: dict[str, str]
    """Each asked step's answer text (what its grader read); "" for a step with no row."""
    passed: dict[str, bool]
    """Each asked step's verdict: False for a step with no row or no grade."""
    answered: int
    graded: int

    def counts(self) -> dict:
        asked = len(self.answers)
        return {"asked": asked, "answered": self.answered, "graded": self.graded,
                "missing": asked - self.answered, "ungraded": self.answered - self.graded}


def _final(position: _Position) -> bool:
    """The position's job can get no more work: its grading is complete, or the grading says every prompt holds
    its sample or used every attempt (``job_complete``), or it was graded with what the job held at its end
    (``allow_incomplete``: the run deadline). A grading taken while prompts were still open is none of these."""
    provenance = position.report.get("provenance") or {}
    return (position.report.get("complete") is True or provenance.get("job_complete") is True
            or provenance.get("allow_incomplete") is True)


async def _answers(platform, module, position: _Position) -> _Answers:
    from reliquary.eval.grading import answer_text

    args = position.args
    asked = await asyncio.to_thread(module.asked_at, args.get("split", "test"), int(args.get("step", 1)),
                                    *([args["dataset_revision"]] if "dataset_revision" in args else []))
    if len(position.grading) != len(asked) or int(position.card.get("count", -1)) != len(asked):
        raise ChainRefused(f"set {position.set_id} holds {len(position.grading)} rows; "
                           f"its position asks {len(asked)} steps")
    try:
        indexes = {row["problem_id"]: row["source_index"] for row in position.grading}
    except (KeyError, TypeError) as exc:
        raise ChainRefused(f"set {position.set_id}: a grading row has no problem_id or source_index") from exc
    if (len(indexes) != len(asked) or any(type(i) is not int for i in indexes.values())
            or sorted(indexes.values()) != list(range(len(asked)))):
        raise ChainRefused(f"set {position.set_id}: its grading rows are not its position's {len(asked)} steps")
    step_of = {problem_id: asked[index] for problem_id, index in indexes.items()}
    thinking = bool((position.report.get("provenance") or {}).get("thinking"))
    answers = {step: "" for step in asked}
    passed = {step: False for step in asked}
    seen: set[str] = set()
    graded = 0
    for row in await _graded_rows(platform, position.eval_id):
        step = step_of.get(row["problem_id"])
        # One sample per problem: a chain carries one answer forward.
        if step is None or int(row["sample_index"]) != 0 or step in seen:
            continue
        seen.add(step)
        answers[step] = answer_text("text", row["completion"] or "", thinking=thinking)
        passed[step] = row["correct"] is True
        graded += row["correct"] is not None
    return _Answers(answers, passed, len(seen), graded)


def _build(source: str, args: dict, directory: Path, open_taskset, clock) -> dict:
    from reliquary.eval import verifiers_source

    return verifiers_source.build_set(source, out=directory, start=0, count=None, sample=None, seed=None,
                                      set_id=None, taskset_args=args, open_taskset=open_taskset, clock=clock)


async def next_set(*, chain_id: str, eval_id: str, platform, subnet,
                   open_taskset: Callable[[str, dict], Any] | None = None,
                   load_tokenizer: Callable[[str, str], tuple[Any, int]] | None = None,
                   clock: Callable[[], float] | None = None, work_dir: str | Path | None = None) -> dict:
    """The set of the position after ``eval_id``'s (a graded chain position), published to the subnet bucket.
    ``load_tokenizer(model, revision) -> (tokenizer, vocab)`` reads the graded model's tokenizer."""
    import time

    from reliquary.eval import verifiers_source
    from reliquary.eval.storage import SetConflict, publish_set, subnet_key

    chain = chain_of(chain_id)
    module = _module(chain)
    position = await _position(platform, subnet, chain, eval_id)
    # Only a final grading seeds the next position: one taken while its job ran would carry answers it lacks.
    if not _final(position):
        raise ChainRefused(f"position_incomplete: {eval_id} was graded while its job could still take work")
    got = await _answers(platform, module, position)
    counts = got.counts()
    if counts["missing"] + counts["ungraded"] > missing_allowed(counts["asked"]):
        raise ChainRefused(f"position_incomplete: {eval_id} graded {counts['graded']} of {counts['asked']} steps, "
                           f"under {min_graded():.0%}")
    answers = got.answers
    args = position.args
    last = await asyncio.to_thread(module.max_position, args.get("split", "test"),
                                   *([args["dataset_revision"]] if "dataset_revision" in args else []))
    if int(args.get("step", 1)) >= last:
        raise ChainRefused(f"chain_complete: position {args.get('step')} is the last of {last}")
    following = module.next_args(args, answers)
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as scratch:
        directory = Path(scratch) / "set"
        card = await asyncio.to_thread(_build, chain.source, following, directory,
                                       open_taskset or verifiers_source.open_taskset, clock or time.time)
        set_id = card["set_id"]
        try:
            await publish_set(directory, platform=None, subnet=subnet)
        except SetConflict:
            # Built before (a lost answer, a retry): the set there is this step's when its args are these.
            stored = json.loads(await subnet.get_bytes(subnet_key(set_id, "set.json")))
            if (stored.get("taskset") or {}).get("args") != card["taskset"]["args"]:
                raise ChainRefused(f"set {set_id} holds another chain step") from None
            card = stored
        prompts = await subnet.get_bytes(subnet_key(set_id, "prompts.jsonl"))
    if hashlib.sha256(prompts).hexdigest() != card["prompts_sha256"]:
        raise ChainRefused(f"set {set_id}: prompts.jsonl does not match its card")
    provenance = position.report.get("provenance") or {}
    model, revision = provenance.get("model"), provenance.get("revision")
    if not model or not revision:
        raise ChainRefused(f"{eval_id} names no model revision")
    if load_tokenizer is None:
        from reliquary.validator.eval_control import load_cpu_tokenizer as load_tokenizer
    tokenizer, _ = await asyncio.to_thread(load_tokenizer, model, revision)
    rows = [json.loads(line) for line in prompts.decode().splitlines() if line.strip()]
    lengths = await asyncio.to_thread(verifiers_source.count_prompt_tokens, rows, f"{model}@{revision}",
                                      lambda _name: tokenizer)
    return {"schema": SCHEMA, "chain": chain_id, "after": eval_id, "set_id": set_id, "env": card["env"],
            "position": int(following["step"]), "positions": last, "count": int(card["count"]),
            "prompts_sha256": card["prompts_sha256"],
            "prompt_tokens": {"max": lengths["max"], "total": sum(lengths["per_problem"])},
            # The position it was built from: its steps without an answer carried forward as no code.
            "from": {"position": int(args.get("step", 1)), **counts}}


async def chain_report(*, chain_id: str, part_ids: Sequence[str], platform, subnet) -> dict:
    """The chain's rates over ``part_ids`` (its positions from the first, in order): every position not graded
    counts its steps as failed. Each position's set must be the one built from the answers graded before it."""
    chain = chain_of(chain_id)
    module = _module(chain)
    passed: dict[str, bool] = {}
    expected: dict | None = None
    split = None
    graded = None
    per_position = []
    for index, part_id in enumerate(part_ids):
        position = await _position(platform, subnet, chain, part_id)
        args = position.args
        provenance = position.report.get("provenance") or {}
        identity = (provenance.get("model"), provenance.get("revision"), bool(provenance.get("thinking")))
        if graded is not None and identity != graded:
            raise ChainRefused(f"{part_id} graded another model, revision or thinking mode than the chain's first")
        graded = identity
        if index == 0:
            split = args.get("split", "test")
            if int(args.get("step", 1)) != 1 or args.get("previous"):
                raise ChainRefused(f"{part_id} is not a chain's first position")
        elif args != expected:
            raise ChainRefused(f"{part_id}'s set was not built from the answers graded before it")
        got = await _answers(platform, module, position)
        passed.update(got.passed)
        per_position.append({"position": int(args.get("step", 1)), "eval_id": part_id,
                             "set_id": position.set_id, **got.counts()})
        expected = module.next_args(args, got.answers)
    revision = (expected or {}).get("dataset_revision")
    summary = await asyncio.to_thread(module.summary, split, passed, *([revision] if revision else []))
    positions = await asyncio.to_thread(module.max_position, split, *([revision] if revision else []))
    # Steps with no answer or no grade count as failed in the rates, as upstream counts a missing answer.
    return {"schema": SCHEMA, "chain": chain_id, "positions_graded": len(part_ids), "positions": positions,
            "complete": len(part_ids) == positions, **summary,
            "steps_missing": sum(p["missing"] for p in per_position),
            "steps_ungraded": sum(p["ungraded"] for p in per_position),
            "per_position": per_position}


__all__ = ["CHAINS", "DEFAULT_MIN_GRADED", "MIN_GRADED_ENV", "Chain", "ChainRefused", "PartNotGraded", "SCHEMA",
           "chain_of", "chain_report", "min_graded", "missing_allowed", "next_set"]
