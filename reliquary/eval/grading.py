"""Grading an evaluation order on the admin host, and its report.

The pod's completions are read from the platform bucket; the problems' source
rows come from the private ``grading.jsonl`` in the subnet bucket. Every row is
graded with its environment's own grader (``compute_reward``, the path
``jobs export --apply-filter`` uses) or flagged: a grader crash is
``score=None`` with its ``grader_detail``, never a silent 0.

Written under ``deliveries/{eval_id}/``: ``graded.parquet``, ``report.json``,
then ``manifest.json``, whose presence makes the grading final.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from reliquary.eval.metrics import bootstrap_mean_ci, pass_at_k, report_ks
from reliquary.eval.sets import open_source, prompt_sha256, validated_set_id
from reliquary.eval.storage import subnet_key

logger = logging.getLogger(__name__)

REPORT_SCHEMA = "reliquary/eval-report/v1"
DELIVERY_PREFIX = "deliveries"
BOOTSTRAP_SEED = 0
BATCH_ROWS = 512
MAX_COMPLETION_KEYS = 10_000
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=-]{0,511}$")
_FENCE_RE = re.compile(r"(?s)(```|~~~)[^\n]*\n.*?\1")
GRADED_COLUMNS = ("env", "set_id", "problem_id", "sample_index", "completion", "tokens",
                  "finish_reason", "correct", "score", "grader_detail")


class SetUnknown(LookupError):
    """A set id the subnet bucket does not hold."""


class GradeRequestError(ValueError):
    """A grade request that can never succeed as sent."""


def request_digest(set_ids: Sequence[str], completion_keys: Sequence[str],
                   problems_per_set: Mapping[str, int]) -> str:
    """What makes two grade calls the same grading."""
    body = {"set_ids": list(set_ids), "completion_keys": list(completion_keys),
            "problems_per_set": {k: int(v) for k, v in sorted(problems_per_set.items())}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def validated_completion_keys(keys: Sequence[str]) -> list[str]:
    if not keys or len(keys) > MAX_COMPLETION_KEYS:
        raise GradeRequestError(f"between 1 and {MAX_COMPLETION_KEYS} completion keys")
    for key in keys:
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key) or ".." in key:
            raise GradeRequestError(f"completion key {key!r} is not a bucket key")
    if len(set(keys)) != len(keys):
        raise GradeRequestError("a completion key is listed twice")
    return list(keys)


async def load_sets(set_ids: Sequence[str], problems_per_set: Mapping[str, int], *,
                    subnet) -> dict[str, tuple[dict, list[dict]]]:
    """Each set's card and its first ``problems_per_set[set_id]`` grading rows,
    from the subnet bucket; ``SetUnknown`` for a set it does not hold."""
    if not set_ids:
        raise GradeRequestError("no set ids")
    if len(set(set_ids)) != len(set_ids) or set(problems_per_set) != set(set_ids):
        raise GradeRequestError("problems_per_set must name each set id once")
    loaded = {}
    for set_id in set_ids:
        try:
            validated_set_id(set_id)
        except ValueError as exc:
            raise SetUnknown(set_id) from exc
        card_body = await subnet.get_bytes(subnet_key(set_id, "set.json"))
        grading_body = await subnet.get_bytes(subnet_key(set_id, "grading.jsonl"))
        if card_body is None or grading_body is None:
            raise SetUnknown(set_id)
        card = json.loads(card_body)
        if hashlib.sha256(grading_body).hexdigest() != card.get("grading_sha256"):
            raise RuntimeError(f"set {set_id}: grading.jsonl does not match its card")
        wanted = problems_per_set[set_id]
        if not isinstance(wanted, int) or isinstance(wanted, bool) or not \
                1 <= wanted <= int(card["count"]):
            raise GradeRequestError(
                f"set {set_id} holds {card['count']} problems; {wanted!r} asked")
        rows = [json.loads(line) for line in grading_body.decode().splitlines()[:wanted]]
        loaded[set_id] = (card, rows)
    return loaded


def answer_text(policy: str, completion: str) -> str:
    """What a free-text grader reads: the part after the reasoning block. An
    unterminated block is all reasoning, so nothing is left."""
    if policy != "text":
        return completion
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        return ""
    return completion


def format_failed(policy: str, answer: str) -> bool:
    """The grader would find no answer to read."""
    if policy == "boxed":
        from reliquary.environment.openmathinstruct import _last_boxed_only_string

        return _last_boxed_only_string(answer) is None
    if policy == "fenced_python":
        return _FENCE_RE.search(answer) is None
    if policy == "json":
        from reliquary.environment.structured_output import (
            StructuredOutputError,
            extract_json_answer,
        )

        try:
            extract_json_answer(answer)
        except StructuredOutputError:
            return True
        return False
    return not answer.strip()


class _Graders:
    """One environment per (source, split), opened on first use."""

    def __init__(self, open_environment: Callable[[str, str], Any]) -> None:
        self._open = open_environment
        self._environments: dict[tuple[str, str], Any] = {}

    def policy(self, source: str) -> str:
        from reliquary.environment.registry import ENVIRONMENT_SPECS

        return ENVIRONMENT_SPECS[source].final_answer_policy

    def grade(self, grading: dict, completion: str) -> tuple[float | None, str, bool]:
        """``(score, grader_detail, format_failure)``."""
        source, split = grading["source"], grading["split"]
        policy = self.policy(source)
        answer = answer_text(policy, completion)
        failed_format = format_failed(policy, answer)
        try:
            key = (source, split)
            if key not in self._environments:
                self._environments[key] = self._open(source, split)
            environment = self._environments[key]
            problem = environment.get_problem(int(grading["source_index"]))
            if prompt_sha256(problem["prompt"]) != grading["prompt_sha256"]:
                return None, "source_drift: the source row is not the frozen prompt", \
                    failed_format
            score = float(environment.compute_reward(problem, answer))
        except Exception as exc:
            return None, f"grader_error: {type(exc).__name__}: {exc}"[:500], failed_format
        return score, "format_failure" if failed_format else "", failed_format


def _graded_schema():
    import pyarrow as pa

    return pa.schema([
        ("env", pa.string()), ("set_id", pa.string()), ("problem_id", pa.string()),
        ("sample_index", pa.int32()), ("completion", pa.string()), ("tokens", pa.int32()),
        ("finish_reason", pa.string()), ("correct", pa.bool_()), ("score", pa.float64()),
        ("grader_detail", pa.string()),
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _env_report(problems: list[str], per_problem: Mapping[str, list[dict]],
                counts: Mapping[str, int], *, seed: int) -> dict:
    graded = []
    for problem_id in problems:
        rows = [r for r in per_problem.get(problem_id, ()) if r["score"] is not None]
        if rows:
            graded.append((sum(1 for r in rows if r["correct"]), len(rows),
                           sum(r["score"] for r in rows) / len(rows)))
    all_rows = [r for p in problems for r in per_problem.get(p, ())]
    samples = max((len(per_problem.get(p, ())) for p in problems), default=0)
    report: dict[str, Any] = {
        "n_problems": len(problems),
        "n_problems_graded": len(graded),
        "missing_problems": sum(1 for p in problems if not per_problem.get(p)),
        "samples": samples,
        "rows": len(all_rows),
        "rows_graded": sum(1 for r in all_rows if r["score"] is not None),
        "grader_errors": sum(1 for r in all_rows if r["score"] is None),
        "duplicate_rows": counts.get("duplicate_rows", 0),
        "truncation_rate": (sum(1 for r in all_rows if r["finish_reason"] == "length")
                            / len(all_rows)) if all_rows else None,
        "format_failure_rate": (sum(1 for r in all_rows if r["format_failure"])
                                / len(all_rows)) if all_rows else None,
        "mean_completion_tokens": (sum(r["tokens"] for r in all_rows) / len(all_rows))
        if all_rows else None,
    }
    if not graded:
        report.update({"pass@1": None, "pass@k": {}, "mean_score": None})
        return report
    per_problem_pass1 = [c / n for c, n, _ in graded]
    low, high = bootstrap_mean_ci(per_problem_pass1, seed=seed)
    report["pass@1"] = {"value": sum(per_problem_pass1) / len(graded), "ci95": [low, high]}
    report["pass@k"] = {}
    for k in report_ks(samples):
        eligible = [(c, n) for c, n, _ in graded if n >= k]
        if eligible:
            report["pass@k"][str(k)] = {"value": pass_at_k(eligible, k),
                                        "n_problems": len(eligible)}
    report["mean_score"] = sum(s for _, _, s in graded) / len(graded)
    return report


def _macro(envs: Mapping[str, dict]) -> dict:
    scored = [r for r in envs.values() if r["pass@1"] is not None]
    if not scored:
        return {"pass@1": None, "pass@k": {}, "envs": 0}
    common = set.intersection(*(set(r["pass@k"]) for r in scored))
    return {
        "envs": len(scored),
        "pass@1": sum(r["pass@1"]["value"] for r in scored) / len(scored),
        "pass@k": {k: sum(r["pass@k"][k]["value"] for r in scored) / len(scored)
                   for k in sorted(common, key=int)},
    }


def _reliquary_version() -> str:
    try:
        from importlib.metadata import version

        return version("reliquary")
    except Exception:
        return "unknown"


async def grade_evaluation(*, eval_id: str, set_ids: Sequence[str],
                           completion_keys: Sequence[str],
                           problems_per_set: Mapping[str, int], platform, subnet,
                           provenance: Mapping[str, Any] | None = None,
                           open_environment: Callable[[str, str], Any] = open_source,
                           work_dir: str | Path | None = None,
                           clock: Callable[[], float] = time.time,
                           bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Grade every completion and write the delivery; the manifest. A grading
    whose manifest exists is returned as stored."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    prefix = f"{DELIVERY_PREFIX}/{eval_id}"
    manifest_key = f"{prefix}/manifest.json"
    stored = await platform.get_json(manifest_key)
    if stored is not None:
        return stored
    keys = validated_completion_keys(completion_keys)
    sets = await load_sets(set_ids, problems_per_set, subnet=subnet)
    selected: dict[str, tuple[str, dict, dict]] = {}
    for set_id, (card, rows) in sets.items():
        for row in rows:
            selected[row["problem_id"]] = (set_id, card, row)
    graders = _Graders(open_environment)
    per_problem: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    counts: Counter = Counter()
    duplicates_by_env: Counter = Counter()
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"grade-{eval_id}-", dir=root))
    graded_path = directory / "graded.parquet"
    try:
        writer = pq.ParquetWriter(str(graded_path), _graded_schema(), compression="zstd")
        try:
            for key in keys:
                local = directory / "completions.jsonl"
                if not await platform.get_file(key, local):
                    raise GradeRequestError(f"completion key {key!r} does not exist")
                batch: list[tuple[dict, dict]] = []
                with open(local, encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            problem_id = str(row["problem_id"])
                            sample_index = int(row["sample_index"])
                        except (ValueError, KeyError, TypeError):
                            counts["malformed_rows"] += 1
                            continue
                        if problem_id not in selected:
                            counts["unexpected_rows"] += 1
                            continue
                        if (problem_id, sample_index) in seen:
                            duplicates_by_env[selected[problem_id][1]["env"]] += 1
                            continue
                        seen.add((problem_id, sample_index))
                        batch.append((row, selected[problem_id]))
                        if len(batch) >= BATCH_ROWS:
                            await _grade_batch(batch, graders, per_problem, writer, pa)
                            batch = []
                if batch:
                    await _grade_batch(batch, graders, per_problem, writer, pa)
                local.unlink()
        finally:
            writer.close()
        by_env: dict[str, list[str]] = defaultdict(list)
        for problem_id, (set_id, card, _) in selected.items():
            by_env[card["env"]].append(problem_id)
        envs = {env: _env_report(problems, per_problem,
                                 {"duplicate_rows": duplicates_by_env[env]},
                                 seed=bootstrap_seed)
                for env, problems in sorted(by_env.items())}
        report = {
            "schema": REPORT_SCHEMA, "eval_id": eval_id, "created_at": clock(),
            "envs": envs, "macro": _macro(envs),
            "counts": {"unexpected_rows": counts["unexpected_rows"],
                       "malformed_rows": counts["malformed_rows"]},
            "bootstrap": {"seed": bootstrap_seed, "level": 0.95, "unit": "problem"},
            "provenance": {
                **dict(provenance or {}),
                "sets": [{"set_id": set_id, "env": card["env"], "source": card["source"],
                          "split": card["split"], "problems": problems_per_set[set_id],
                          "prompts_sha256": card["prompts_sha256"],
                          "grading_sha256": card["grading_sha256"],
                          "environment_manifest_sha256":
                              card.get("environment_manifest_sha256")}
                         for set_id, (card, _) in sets.items()],
                "completion_keys": keys,
                "reliquary_version": _reliquary_version(),
            },
        }
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report, sort_keys=True, indent=1))
        files = []
        for path in (graded_path, report_path):
            key = f"{prefix}/{path.name}"
            await platform.put_file(key, path)
            files.append({"name": path.name, "key": key, "bytes": path.stat().st_size,
                          "sha256": await asyncio.to_thread(_sha256, path)})
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    manifest = {
        "schema": REPORT_SCHEMA, "eval_id": eval_id, "created_at": report["created_at"],
        "request_sha256": request_digest(set_ids, keys, problems_per_set),
        "rows": sum(len(v) for v in per_problem.values()), "files": files,
        "keys": [f["key"] for f in files] + [manifest_key],
    }
    await platform.put_json(manifest_key, manifest)
    logger.info("eval %s graded: %d rows over %d sets", eval_id, manifest["rows"], len(sets))
    return manifest


async def _grade_batch(batch, graders: _Graders, per_problem, writer, pa) -> None:
    def run() -> list[dict]:
        out = []
        for row, (set_id, card, grading) in batch:
            completion = row.get("completion")
            completion = completion if isinstance(completion, str) else ""
            score, detail, failed_format = graders.grade(grading, completion)
            out.append({
                "env": card["env"], "set_id": set_id, "problem_id": grading["problem_id"],
                "sample_index": int(row["sample_index"]), "completion": completion,
                "tokens": int(row.get("completion_tokens") or 0),
                "finish_reason": str(row.get("finish_reason") or ""),
                "correct": None if score is None else score >= 1.0, "score": score,
                "grader_detail": detail, "format_failure": failed_format,
            })
        return out

    # Graders are synchronous and may run code: off the event loop.
    rows = await asyncio.to_thread(run)
    for row in rows:
        per_problem[row["problem_id"]].append(
            {k: row[k] for k in ("score", "correct", "finish_reason", "tokens",
                                 "format_failure")})
    table = pa.Table.from_pylist([{k: r[k] for k in GRADED_COLUMNS} for r in rows],
                                 schema=_graded_schema())
    await asyncio.to_thread(writer.write_table, table)


__all__ = [
    "GRADED_COLUMNS",
    "GradeRequestError",
    "REPORT_SCHEMA",
    "SetUnknown",
    "answer_text",
    "format_failed",
    "grade_evaluation",
    "load_sets",
    "request_digest",
    "validated_completion_keys",
]
