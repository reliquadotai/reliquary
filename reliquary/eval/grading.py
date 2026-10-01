"""Grading an evaluation order on the admin host, and its report.

The pod's completions are read from the platform bucket; the problems' source
rows come from the private ``grading.jsonl`` in the subnet bucket. Each row is
scored as admission scores it (``reward_scorer``: code goes to the sandboxed
grading service, never to the package's own runner) or flagged: a grader crash
is ``score=None`` with its ``grader_detail``, never a silent 0.

Each problem is expected to have exactly ``samples_per_set[set_id]`` samples,
indexed ``0..samples-1``. The headline pass@k counts a missing or ungraded
sample as a failure; the variant that leaves them out is shown beside it.

Written under ``evaluations/{eval_id}/`` (never ``deliveries/``, the corpus
exports' namespace): ``graded.parquet``, ``report.json``, then
``manifest.json``, whose presence makes the grading final.
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

REPORT_SCHEMA = "reliquary/eval-report/v2"
EVALUATION_PREFIX = "evaluations"
BOOTSTRAP_SEED = 0
BATCH_ROWS = 512
MAX_COMPLETION_KEYS = 10_000
MAX_SAMPLES = 1024
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/=-]{0,511}$")
_FENCE_RE = re.compile(r"(?s)(```|~~~)[^\n]*\n.*?\1")
GRADED_COLUMNS = ("env", "set_id", "problem_id", "sample_index", "completion", "tokens",
                  "finish_reason", "correct", "score", "grader_detail")


class SetUnknown(LookupError):
    """A set id the subnet bucket does not hold."""


class GradeRequestError(ValueError):
    """A grade request that can never succeed as sent."""


class SandboxUnavailable(RuntimeError):
    """A code set cannot be graded on this host: the grading service is absent."""


def evaluation_prefix(eval_id: str) -> str:
    return f"{EVALUATION_PREFIX}/{eval_id}"


def request_digest(set_ids: Sequence[str], completion_keys: Sequence[str],
                   problems_per_set: Mapping[str, int],
                   samples_per_set: Mapping[str, int], job_id: str | None = None) -> str:
    """What makes two grade calls the same grading."""
    body = {"set_ids": list(set_ids), "completion_keys": list(completion_keys),
            "problems_per_set": {k: int(v) for k, v in sorted(problems_per_set.items())},
            "samples_per_set": {k: int(v) for k, v in sorted(samples_per_set.items())}}
    if job_id is not None:
        # Only when set, so an uploads grading hashes as it always did.
        body["job_id"] = job_id
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


def _count(value: Any, *, low: int, high: int, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise GradeRequestError(f"{what} must be in [{low}, {high}], got {value!r}")
    return value


async def load_sets(set_ids: Sequence[str], problems_per_set: Mapping[str, int],
                    samples_per_set: Mapping[str, int], *,
                    subnet) -> dict[str, tuple[dict, list[dict]]]:
    """Each set's card and its first ``problems_per_set[set_id]`` grading rows,
    from the subnet bucket; ``SetUnknown`` for a set it does not hold."""
    if not set_ids:
        raise GradeRequestError("no set ids")
    if (len(set(set_ids)) != len(set_ids) or set(problems_per_set) != set(set_ids)
            or set(samples_per_set) != set(set_ids)):
        raise GradeRequestError("problems_per_set and samples_per_set must name each set once")
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
        wanted = _count(problems_per_set[set_id], low=1, high=int(card["count"]),
                        what=f"problems_per_set[{set_id}]")
        _count(samples_per_set[set_id], low=1, high=MAX_SAMPLES,
               what=f"samples_per_set[{set_id}]")
        rows = [json.loads(line) for line in grading_body.decode().splitlines()[:wanted]]
        loaded[set_id] = (card, rows)
    return loaded


def require_sandboxes(sets: Mapping[str, tuple[dict, list[dict]]], require=None) -> None:
    """Refuse an order with a code set where the grading service is not running."""
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    if require is None:
        from reliquary.corpus.export import require_code_sandbox as require
    for card, _ in sets.values():
        try:
            require(ENVIRONMENT_SPECS[card["source"]])
        except ValueError as exc:
            raise SandboxUnavailable(str(exc)) from exc


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
    """The grader would find no answer to read (an approximation of each
    package's own extractor; see the rulings)."""
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
    """One environment and its scorer per (source, split), opened on first use."""

    def __init__(self, open_environment: Callable[[str, str], Any],
                 scorer_for: Callable[[Any, Any], Callable[[dict, str], float]]) -> None:
        self._open = open_environment
        self._scorer_for = scorer_for
        self._opened: dict[tuple[str, str], tuple[Any, Callable]] = {}

    @staticmethod
    def spec(source: str):
        from reliquary.environment.registry import ENVIRONMENT_SPECS

        return ENVIRONMENT_SPECS[source]

    def grade(self, grading: dict, completion: str) -> tuple[float | None, str, bool]:
        """``(score, grader_detail, format_failure)``."""
        source, split = grading["source"], grading["split"]
        spec = self.spec(source)
        answer = answer_text(spec.final_answer_policy, completion)
        failed_format = format_failed(spec.final_answer_policy, answer)
        try:
            key = (source, split)
            if key not in self._opened:
                environment = self._open(source, split)
                self._opened[key] = (environment, self._scorer_for(spec, environment))
            environment, score = self._opened[key]
            problem = environment.get_problem(int(grading["source_index"]))
            if prompt_sha256(problem["prompt"]) != grading["prompt_sha256"]:
                return None, "source_drift: the source row is not the frozen prompt", \
                    failed_format
            value = float(score(problem, answer))
        except Exception as exc:
            return None, f"grader_error: {type(exc).__name__}: {exc}"[:500], failed_format
        return value, "format_failure" if failed_format else "", failed_format


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


def _pass_block(problems: list[tuple[int, int]], ks: list[int], *, seed: int) -> dict:
    """pass@1 with its CI and pass@k, over ``(c, n)`` problems."""
    if not problems:
        return {"pass@1": None, "pass@k": {}}
    per_problem = [c / n for c, n in problems]
    low, high = bootstrap_mean_ci(per_problem, seed=seed)
    block: dict[str, Any] = {
        "pass@1": {"value": sum(per_problem) / len(problems), "ci95": [low, high]},
        "pass@k": {},
    }
    for k in ks:
        eligible = [(c, n) for c, n in problems if n >= k]
        if eligible:
            block["pass@k"][str(k)] = {"value": pass_at_k(eligible, k),
                                       "n_problems": len(eligible)}
    return block


def _env_report(problems: list[tuple[str, int]], per_problem: Mapping[str, list[dict]],
                counts: Mapping[str, int], *, seed: int) -> dict:
    """``problems`` is ``(problem_id, samples ordered)``."""
    conservative, graded_only, scores = [], [], []
    for problem_id, samples in problems:
        rows = per_problem.get(problem_id, ())
        graded = [r for r in rows if r["score"] is not None]
        correct = sum(1 for r in graded if r["correct"])
        # Missing and ungraded samples count as failures in the headline.
        conservative.append((correct, samples))
        if graded:
            graded_only.append((correct, len(graded)))
            scores.append(sum(r["score"] for r in graded) / len(graded))
    all_rows = [r for p, _ in problems for r in per_problem.get(p, ())]
    expected = sum(s for _, s in problems)
    graded_rows = sum(1 for r in all_rows if r["score"] is not None)
    samples = max(s for _, s in problems)
    ks = report_ks(samples)
    headline = _pass_block(conservative, ks, seed=seed)
    excluded = _pass_block(graded_only, ks, seed=seed)
    report: dict[str, Any] = {
        "n_problems": len(problems),
        "n_problems_graded": len(graded_only),
        "missing_problems": sum(1 for p, _ in problems if not per_problem.get(p)),
        "samples": samples,
        "expected_rows": expected,
        "graded_rows": graded_rows,
        "ungraded_rows": len(all_rows) - graded_rows,
        "missing_rows": expected - len(all_rows),
        "duplicate_rows": counts.get("duplicate_rows", 0),
        "out_of_range_rows": counts.get("out_of_range_rows", 0),
        "no_graded_rows": graded_rows == 0,
        **headline,
        "excluding_ungraded_and_missing": excluded,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "truncation_rate": (sum(1 for r in all_rows if r["finish_reason"] == "length")
                            / len(all_rows)) if all_rows else None,
        "format_failure_rate": (sum(1 for r in all_rows if r["format_failure"])
                                / len(all_rows)) if all_rows else None,
        "mean_completion_tokens": (sum(r["tokens"] for r in all_rows) / len(all_rows))
        if all_rows else None,
    }
    report["complete"] = report["missing_rows"] == 0 and report["ungraded_rows"] == 0
    return report


def _macro(envs: Mapping[str, dict]) -> dict:
    """The mean over every ordered environment; one with no graded row counts
    as 0 (its headline is conservative) and is named."""
    common = set.intersection(*(set(r["pass@k"]) for r in envs.values()))
    return {
        "envs": len(envs),
        "envs_without_graded_rows": sorted(e for e, r in envs.items() if r["no_graded_rows"]),
        "pass@1": sum(r["pass@1"]["value"] for r in envs.values()) / len(envs),
        "pass@k": {k: sum(r["pass@k"][k]["value"] for r in envs.values()) / len(envs)
                   for k in sorted(common, key=int)},
    }


def _reliquary_version() -> str:
    try:
        from importlib.metadata import version

        return version("reliquary")
    except Exception:
        return "unknown"


def _default_scorer(spec, environment):
    from reliquary.corpus.export import reward_scorer

    return reward_scorer(spec, environment)


MALFORMED = object()


async def upload_rows(platform, keys: Sequence[str], directory: Path):
    """The pod's completion lines, file by file; ``MALFORMED`` for a bad line."""
    for key in keys:
        local = directory / "completions.jsonl"
        if not await platform.get_file(key, local):
            raise GradeRequestError(f"completion key {key!r} does not exist")
        with open(local, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    yield MALFORMED
        local.unlink()


READ_CONCURRENCY = 16


async def collect_job_records(job, records, *, concurrency: int = READ_CONCURRENCY) -> dict:
    """An eval job's verdicts and passing (not voided) records, read
    ``concurrency`` at a time: ``{"passing": [(sid, record)], "verdicts",
    "audited", "samples_by_prompt"}``."""
    ids = list(await records.list_verdict_ids(job.job_id))
    lister = getattr(records, "list_voided_ids", None)
    voided = set(await lister(job.job_id)) if lister is not None else set()
    gate = asyncio.Semaphore(concurrency)

    async def one(submission_id):
        async with gate:
            verdict = await records.read_verdict(job.job_id, submission_id)
            if not verdict or not verdict.get("passed") or submission_id in voided:
                return submission_id, verdict, None
            return submission_id, verdict, await records.read_submission(job.job_id, submission_id)

    read = await asyncio.gather(*(one(sid) for sid in ids))
    passing = [(sid, record) for sid, _, record in read if record is not None]
    samples: Counter = Counter()
    for _, record in passing:
        samples[int(record["prompt_index"])] += len(record["completions"])
    verdicts = [v for _, v, _ in read if v]
    return {"passing": passing, "verdicts": len(verdicts),
            "audited": sum(1 for v in verdicts if v.get("audited")),
            "samples_by_prompt": dict(samples)}


def job_complete(job, collected: dict, samples: int, slots=None) -> bool:
    """Every prompt of the job holds its passing samples, or (with the job's
    ledger ``slots``) has used every attempt and is exhausted, its missing
    samples counted as failures. Never true with nothing passing."""
    held = collected["samples_by_prompt"]
    if not held:
        return False
    return all(held.get(index, 0) >= samples
               or (slots is not None and slots.prompt_state(index) == "exhausted")
               for index in range(job.prompt_start, job.prompt_end))


def exhausted_prompts(job, collected: dict, samples: int, slots) -> list[int]:
    """The prompts short of their samples whose every attempt is used."""
    held = collected["samples_by_prompt"]
    return [index for index in range(job.prompt_start, job.prompt_end)
            if held.get(index, 0) < samples and slots.prompt_state(index) == "exhausted"]


class JobRows:
    """The completions of an eval job's passing submissions, as grade rows.

    ``prompt_index`` maps to ``problem_id`` through the set's order (row i of
    the job is line i of the set); within a problem, passing submissions in
    submission-id order and their completions in order are samples 0, 1, ...
    A voided pass (its executor quarantined) is left out like a failed one.
    """

    def __init__(self, *, job, collected: dict) -> None:
        self.job = job
        self.collected = collected
        self.hotkeys: set[str] = set()
        self.verdicts = collected["verdicts"]
        self.audited = collected["audited"]

    async def __aiter__(self):
        from reliquary.eval.prompt_source import load_eval_rows, parse_eval_source

        rows = await asyncio.to_thread(load_eval_rows, parse_eval_source(self.job.prompt_source))
        by_prompt: dict[int, list[tuple[str, dict]]] = defaultdict(list)
        for submission_id, record in self.collected["passing"]:
            by_prompt[int(record["prompt_index"])].append((submission_id, record))
        for prompt_index in sorted(by_prompt):
            if not 0 <= prompt_index < len(rows):
                yield MALFORMED
                continue
            sample = 0
            for _, record in sorted(by_prompt[prompt_index], key=lambda pair: pair[0]):
                self.hotkeys.add(record["hotkey"])
                for completion in record["completions"]:
                    tokens = completion.get("tokens") or ()
                    yield {"problem_id": rows[prompt_index]["problem_id"],
                           "sample_index": sample, "completion": completion["text"],
                           "completion_tokens": len(tokens),
                           "finish_reason": ("stop" if tokens and tokens[-1] == self.job.eos_token_id
                                             else "length")}
                    sample += 1


async def grade_evaluation(*, eval_id: str, set_ids: Sequence[str],
                           completion_keys: Sequence[str],
                           problems_per_set: Mapping[str, int],
                           samples_per_set: Mapping[str, int], platform, subnet,
                           provenance: Mapping[str, Any],
                           job_rows: "JobRows | None" = None,
                           open_environment: Callable[[str, str], Any] = open_source,
                           scorer_for: Callable = _default_scorer,
                           require_sandbox=None,
                           work_dir: str | Path | None = None,
                           clock: Callable[[], float] = time.time,
                           bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Grade every completion and write the evaluation; the manifest. A grading
    whose manifest exists is returned as stored. The completions are the pod's
    uploads (``completion_keys``) or an eval job's passing records (``job_rows``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    prefix = evaluation_prefix(eval_id)
    manifest_key = f"{prefix}/manifest.json"
    stored = await platform.get_json(manifest_key)
    if stored is not None:
        return stored
    keys = validated_completion_keys(completion_keys) if job_rows is None else []
    sets = await load_sets(set_ids, problems_per_set, samples_per_set, subnet=subnet)
    require_sandboxes(sets, require_sandbox)
    selected: dict[str, tuple[str, dict, dict, int]] = {}
    for set_id, (card, rows) in sets.items():
        for row in rows:
            selected[row["problem_id"]] = (set_id, card, row, int(samples_per_set[set_id]))
    graders = _Graders(open_environment, scorer_for)
    per_problem: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    counts: Counter = Counter()
    by_env_counts: dict[str, Counter] = defaultdict(Counter)
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"grade-{eval_id}-", dir=root))
    graded_path = directory / "graded.parquet"
    try:
        writer = pq.ParquetWriter(str(graded_path), _graded_schema(), compression="zstd")
        try:
            batch: list[tuple[dict, tuple]] = []
            source = upload_rows(platform, keys, directory) if job_rows is None else job_rows
            async for row in source:
                try:
                    if row is MALFORMED:
                        raise TypeError("line")
                    problem_id = str(row["problem_id"])
                    sample_index = row["sample_index"]
                    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
                        raise TypeError("sample_index")
                except (ValueError, KeyError, TypeError):
                    counts["malformed_rows"] += 1
                    continue
                if problem_id not in selected:
                    counts["unexpected_rows"] += 1
                    continue
                entry = selected[problem_id]
                env = entry[1]["env"]
                if not 0 <= sample_index < entry[3]:
                    by_env_counts[env]["out_of_range_rows"] += 1
                    continue
                if (problem_id, sample_index) in seen:
                    by_env_counts[env]["duplicate_rows"] += 1
                    continue
                seen.add((problem_id, sample_index))
                batch.append((row, entry))
                if len(batch) >= BATCH_ROWS:
                    await _grade_batch(batch, graders, per_problem, writer, pa)
                    batch = []
            if batch:
                await _grade_batch(batch, graders, per_problem, writer, pa)
        finally:
            writer.close()
        if job_rows is not None:
            provenance = {
                **dict(provenance), "generation": "sn81-miners", "job_id": job_rows.job.job_id,
                "miner_hotkeys": len(job_rows.hotkeys),
                "audited_fraction": (job_rows.audited / job_rows.verdicts
                                     if job_rows.verdicts else None),
                "sampling_verified": False,
                "note": "sampling not verified: TOPLOC proves the completions were computed "
                        "by the model, not that they were sampled as the order asked",
            }
        by_env: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for problem_id, (_, card, _, samples) in selected.items():
            by_env[card["env"]].append((problem_id, samples))
        envs = {env: _env_report(problems, per_problem, by_env_counts[env],
                                 seed=bootstrap_seed)
                for env, problems in sorted(by_env.items())}
        report = {
            "schema": REPORT_SCHEMA, "eval_id": eval_id, "created_at": clock(),
            "complete": all(r["complete"] for r in envs.values())
            and not counts["malformed_rows"],
            "envs": envs, "macro": _macro(envs),
            "counts": {"unexpected_rows": counts["unexpected_rows"],
                       "malformed_rows": counts["malformed_rows"]},
            "headline": "missing and ungraded samples count as failures; "
                        "excluding_ungraded_and_missing leaves them out",
            "bootstrap": {"seed": bootstrap_seed, "level": 0.95, "unit": "problem"},
            "provenance": {
                **dict(provenance),
                "sets": [{"set_id": set_id, "env": card["env"], "source": card["source"],
                          "split": card["split"], "problems": problems_per_set[set_id],
                          "samples": samples_per_set[set_id],
                          "prompts_sha256": card["prompts_sha256"],
                          "grading_sha256": card["grading_sha256"],
                          "environment_manifest_sha256":
                              card.get("environment_manifest_sha256"),
                          "rl_disjointness": card.get("disjointness", {}).get("rl"),
                          "contamination_note":
                              card.get("disjointness", {}).get("contamination_note")}
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
            # Hashed before the upload, from the bytes uploaded.
            digest = await asyncio.to_thread(_sha256, path)
            await platform.put_file(key, path)
            files.append({"name": path.name, "key": key, "bytes": path.stat().st_size,
                          "sha256": digest})
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    manifest = {
        "schema": REPORT_SCHEMA, "eval_id": eval_id, "created_at": report["created_at"],
        "request_sha256": request_digest(
            set_ids, keys, problems_per_set, samples_per_set,
            job_id=None if job_rows is None else job_rows.job.job_id),
        "complete": report["complete"],
        "rows": sum(len(v) for v in per_problem.values()), "files": files,
        "keys": [f["key"] for f in files] + [manifest_key],
    }
    await platform.put_json(manifest_key, manifest)
    logger.info("eval %s graded: %d rows over %d sets", eval_id, manifest["rows"], len(sets))
    return manifest


async def _grade_batch(batch, graders: _Graders, per_problem, writer, pa) -> None:
    def run() -> list[dict]:
        out = []
        for row, (set_id, card, grading, _) in batch:
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

    # Graders are synchronous and may be slow: off the event loop.
    rows = await asyncio.to_thread(run)
    for row in rows:
        per_problem[row["problem_id"]].append(
            {k: row[k] for k in ("score", "correct", "finish_reason", "tokens",
                                 "format_failure")})
    table = pa.Table.from_pylist([{k: r[k] for k in GRADED_COLUMNS} for r in rows],
                                 schema=_graded_schema())
    await asyncio.to_thread(writer.write_table, table)


__all__ = [
    "EVALUATION_PREFIX",
    "GRADED_COLUMNS",
    "JobRows",
    "collect_job_records",
    "exhausted_prompts",
    "job_complete",
    "GradeRequestError",
    "REPORT_SCHEMA",
    "SandboxUnavailable",
    "SetUnknown",
    "answer_text",
    "evaluation_prefix",
    "format_failed",
    "grade_evaluation",
    "load_sets",
    "request_digest",
    "require_sandboxes",
    "validated_completion_keys",
]
