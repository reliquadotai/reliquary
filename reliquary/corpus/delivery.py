"""Export v2: a job's passing rows delivered as Parquet shards.

Streamed end to end: verdicts and records are read a window at a time, a
bounded number in flight, and each shard is uploaded and deleted locally as
soon as it closes, so neither memory nor disk ever holds a whole job. The
manifest is written last; its presence is what makes a delivery complete.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DELIVERY_SCHEMA = "reliquary/corpus-delivery/v2"
DELIVERY_PREFIX = "deliveries"
SHARD_MAX_BYTES = 500 * 1024 * 1024
ROW_GROUP_ROWS = 2048
# A row group is also flushed at this many raw bytes: long completions are ~128 KB each.
ROW_GROUP_MAX_BYTES = 64 * 1024 * 1024
# Reads in flight at once, as the auditor reads records.
READ_CONCURRENCY = 16
# Verdict ids handled per window: what bounds the records held in memory
# (a record of 8 completions near 32k tokens is ~1 MB).
READ_WINDOW = 64
# Room left in a shard for the Parquet footer and page headers.
SHARD_OVERHEAD_BYTES = 1024 * 1024
# The private dataset library's JSONL limits, including JavaScript string length.
INSTRUCTION_MAX_BYTES = 5 * 1024 * 1024
INSTRUCTION_MAX_ROWS = 10_000
INSTRUCTION_MAX_UTF16 = 32_000

_DELIVERY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

ROW_FIELDS = ("job_id", "submission_id", "prompt_index", "completion_index", "prompt",
              "completion", "completion_tokens", "accepted", "score")
# Episode jobs (spec §5 N6): one row per certified trajectory.
# ``tokens`` + ``assistant_mask`` are the authoritative training form: what the
# miner proved and the replay certified. ``messages`` is the same conversation
# parsed with the pinned renderer, proven per row to render back to ``tokens``
# with whitespace erased, not byte for byte: the pinned parser strips the
# whitespace around a tool-call parameter value (an ``edit``'s indented
# ``old_str``/``new_str`` loses its leading indentation, and the template ends
# on a newline). The stripped call is what the harness executed. Turn-by-turn,
# ``parse_trajectory`` proves each span canonical and each tool segment the
# exact bridge rendering.
EPISODE_ROW_FIELDS = ("job_id", "submission_id", "prompt_index", "task_id", "messages", "tokens",
                      "assistant_mask", "final_diff", "graded_success", "replay_certified", "turns",
                      "stop")


def validated_delivery_id(delivery_id: Any) -> str:
    if not isinstance(delivery_id, str) or not _DELIVERY_ID_RE.fullmatch(delivery_id):
        raise ValueError(f"delivery id {delivery_id!r} is not a name")
    return delivery_id


def _row_schema():
    import pyarrow as pa

    return pa.schema([
        ("job_id", pa.string()), ("submission_id", pa.string()),
        ("prompt_index", pa.int64()), ("completion_index", pa.int32()),
        ("prompt", pa.string()), ("completion", pa.string()),
        ("completion_tokens", pa.int32()), ("accepted", pa.bool_()), ("score", pa.float64()),
    ])


def _episode_row_schema():
    import pyarrow as pa

    return pa.schema([
        ("job_id", pa.string()), ("submission_id", pa.string()), ("prompt_index", pa.int64()),
        ("task_id", pa.string()), ("messages", pa.string()), ("tokens", pa.list_(pa.int32())),
        ("assistant_mask", pa.list_(pa.int8())), ("final_diff", pa.string()),
        ("graded_success", pa.bool_()), ("replay_certified", pa.bool_()), ("turns", pa.string()),
        ("stop", pa.string()),
    ])


class EpisodePromptMismatch(ValueError):
    """The record's prompt tokens are not the pinned render of its task's prompt."""


class EpisodeUnrendered(ValueError):
    """The rebuilt messages do not render back to the row's tokens."""


def episode_row(*, job, submission_id: str, record: Mapping, grade: Mapping, renderer,
                user_prompt: str) -> dict:
    """The row of one graded trajectory. Messages are rebuilt from the proven
    tokens with the pinned renderer (never from miner text), and the system and
    user messages must render to the record's prompt tokens, so the messages
    and the tokens are one and the same trajectory.

    Raises ``TrajectoryRefused``, ``EpisodePromptMismatch`` or
    ``EpisodeUnrendered``."""
    from reliquary.corpus.trajectory_parse import parse_trajectory
    from reliquary.environment.agentic_swe import BASH_SYSTEM_PROMPT

    trajectory = record["completions"][0]
    prompt = [int(t) for t in trajectory["prompt_tokens"]]
    tokens = [int(t) for t in trajectory["tokens"]]
    spans = [(int(turn["start"]), int(turn["end"])) for turn in trajectory["turns"]]
    if [int(t) for t in renderer.initial_ids(user_prompt)] != prompt:
        raise EpisodePromptMismatch(submission_id)
    parsed = parse_trajectory(renderer, prompt_ids=prompt, tokens=tokens, spans=spans,
                              stop=trajectory["stop"], max_turns=job.episode.max_turns)
    messages = [{"role": "system", "content": BASH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    for t, ((start, end), turn) in enumerate(zip(spans, parsed.turns)):
        message = renderer.assistant_message(tokens[start:end])
        for k, call in enumerate(message.get("tool_calls") or ()):
            call["id"] = f"call_{t}_{k}"         # unique in the conversation, not per turn
        messages.append(message)
        messages += [{"role": "tool", "tool_call_id": f"call_{t}_{k}", "content": text}
                     for k, text in enumerate(turn.observations)]
    # The proof the messages are the tokens (see EPISODE_ROW_FIELDS for why
    # whitespace is erased first).
    if renderer.whitespace_free(renderer.render_messages(messages)) \
            != renderer.whitespace_free(prompt + tokens):
        raise EpisodeUnrendered(submission_id)
    mask = [0] * (len(prompt) + len(tokens))
    for start, end in spans:
        mask[len(prompt) + start:len(prompt) + end] = [1] * (end - start)
    return {"job_id": job.job_id, "submission_id": submission_id,
            "prompt_index": int(record["prompt_index"]),
            "task_id": str(grade.get("instance_id") or ""),
            "messages": json.dumps(messages, ensure_ascii=False), "tokens": prompt + tokens,
            "assistant_mask": mask, "final_diff": trajectory["final_diff"],
            "graded_success": bool(grade.get("graded_success")),
            "replay_certified": bool(grade.get("replay_certified")),
            "turns": json.dumps([[s, e] for s, e in spans]), "stop": trajectory["stop"]}


def _lone(deciders) -> str | None:
    deciders = set(deciders or ())
    return next(iter(deciders)) if len(deciders) == 1 else None


def held_by_quarantine(document: Mapping, quarantined: Collection[str]) -> list[str]:
    """The quarantined executors that decided part of ``document`` alone: the
    validator holds such a grade from payment until a regrade replaces it
    (``CorpusGrader._caught``), and an export refuses it the same way."""
    lone = {_lone(document.get("graded_by")),
            _lone((document.get("replay") or {}).get("graded_by"))} - {None}
    return sorted(e for e in lone if e in quarantined)


def certified(document: Mapping) -> bool:
    """An ``ok`` grade whose replay certified the episode (within tolerance).
    Every other outcome (ungradeable, disputed, timeout, error, unparseable,
    audit_failed, a failed or unconfirmed replay) certifies nothing."""
    replay = document.get("replay") or {}
    return (document.get("status") == "ok" and document.get("replay_certified") is True
            and replay.get("status") == "ok" and replay.get("certified") is True)


async def _bounded(calls, gate: asyncio.Semaphore) -> list:
    async def one(call):
        async with gate:
            return await call

    return await asyncio.gather(*(one(c) for c in calls))


async def delivery_rows(*, job, records, counts: dict, grade=None,
                        concurrency: int = READ_CONCURRENCY,
                        window: int = READ_WINDOW) -> AsyncIterator[dict]:
    """One row per completion of every passing submission, in verdict-id order.

    No hotkey: the delivery is the work, not who did it. ``grade`` annotates
    (``accepted``, ``score``) and never drops a row. ``counts`` is filled in as
    the rows go.
    """
    ids = list(await records.list_verdict_ids(job.job_id))
    counts.update(verdicts=len(ids), passing_submissions=0, missing_records=0, rows=0,
                  rows_accepted=0 if grade is not None else None)
    gate = asyncio.Semaphore(concurrency)
    for start in range(0, len(ids), window):
        chunk = ids[start:start + window]
        verdicts = await _bounded((records.read_verdict(job.job_id, s) for s in chunk), gate)
        passing = [sid for sid, v in zip(chunk, verdicts) if v and v.get("passed")]
        found = await _bounded((records.read_submission(job.job_id, s) for s in passing), gate)
        rows = []
        for sid, record in zip(passing, found):
            if record is None:
                # A verdict can be visible before its record (R2 is not transactional).
                logger.warning("corpus delivery: submission %s passed but has no record", sid[:12])
                counts["missing_records"] += 1
                continue
            counts["passing_submissions"] += 1
            for index, completion in enumerate(record["completions"]):
                rows.append({
                    "job_id": job.job_id, "submission_id": sid,
                    "prompt_index": int(record["prompt_index"]), "completion_index": index,
                    "prompt": record["rendered_prompt"], "completion": completion["text"],
                    "completion_tokens": len(completion.get("tokens") or ()),
                    "accepted": None, "score": None,
                })
        if grade is not None and rows:
            def annotate(batch=rows):
                for row in batch:
                    accepted, score = grade(row["prompt_index"], row["completion"])
                    row["accepted"], row["score"] = bool(accepted), float(score)

            # Graders are synchronous and may be slow: off the event loop.
            await asyncio.to_thread(annotate)
        for row in rows:
            counts["rows"] += 1
            if row["accepted"]:
                counts["rows_accepted"] += 1
            yield row


async def episode_rows(*, job, records, renderer, source, counts: dict, sft_only: bool = False,
                       quarantined: Collection[str] | None,
                       concurrency: int = READ_CONCURRENCY,
                       window: int = READ_WINDOW) -> AsyncIterator[dict]:
    """One row per passing, not voided, certified trajectory, in verdict-id
    order; ``sft_only`` keeps the certified successes (spec N6's SFT set).

    The effective grade is the latest regrade, else the grade. A grade that an
    executor of ``quarantined`` decided alone is held, as the settler holds it;
    the caller names the registry's quarantined grade executors (None refuses:
    an export blind to quarantines must not run)."""
    if quarantined is None:
        raise ValueError("an episode export needs the quarantined grade executors "
                         "(the registry's), to hold what they decided alone")
    quarantined = frozenset(quarantined)
    ids = list(await records.list_verdict_ids(job.job_id))
    voided = set(await records.list_voided_ids(job.job_id))
    read_voided = getattr(records, "read_voided", None)
    counts.update(verdicts=len(ids), passing_submissions=0, voided=0, ungraded=0, held=0,
                  uncertified=0, not_successful=0, missing_records=0, task_mismatch=0,
                  prompt_mismatch=0, unparseable=0, unrendered=0, sft_rows=0, rows=0)
    gate = asyncio.Semaphore(concurrency)
    for start in range(0, len(ids), window):
        chunk = ids[start:start + window]
        verdicts = await _bounded((records.read_verdict(job.job_id, s) for s in chunk), gate)
        passing = [sid for sid, v in zip(chunk, verdicts) if v and v.get("passed")]
        counts["passing_submissions"] += len(passing)
        kept = [sid for sid in passing if sid not in voided]
        counts["voided"] += len(passing) - len(kept)
        regrades = await _bounded((records.read_regrade(job.job_id, s) for s in kept), gate)
        grades = await _bounded((records.read_grade(job.job_id, s) for s in kept), gate)
        candidates = []
        for sid, regrade, grade in zip(kept, regrades, grades):
            document = regrade if regrade is not None else grade
            if document is None:
                counts["ungraded"] += 1
            elif held_by_quarantine(document, quarantined):
                counts["held"] += 1
            elif not certified(document):
                counts["uncertified"] += 1
            elif sft_only and document.get("graded_success") is not True:
                counts["not_successful"] += 1
            else:
                candidates.append((sid, document))
        if read_voided is not None and candidates:
            # A void written after the listing: read before the row is built.
            late = await _bounded((read_voided(job.job_id, s) for s, _ in candidates), gate)
            counts["voided"] += sum(1 for v in late if v is not None)
            candidates = [c for c, v in zip(candidates, late) if v is None]
        found = await _bounded((records.read_submission(job.job_id, s) for s, _ in candidates),
                               gate)
        for (sid, document), record in zip(candidates, found):
            if record is None:
                counts["missing_records"] += 1
                continue
            index = int(record["prompt_index"])
            if document.get("instance_id") != source.instance_id(index):
                logger.error("corpus export: %s's grade names task %r, its index %d is %r",
                             sid[:12], document.get("instance_id"), index,
                             source.instance_id(index))
                counts["task_mismatch"] += 1
                continue
            try:
                row = await asyncio.to_thread(
                    episode_row, job=job, submission_id=sid, record=record, grade=document,
                    renderer=renderer, user_prompt=source.prompt(index))
            except EpisodeUnrendered:
                logger.error("corpus export: %s's messages do not render to its tokens", sid[:12])
                counts["unrendered"] += 1
                continue
            except EpisodePromptMismatch:
                logger.error("corpus export: %s's prompt tokens are not its task's", sid[:12])
                counts["prompt_mismatch"] += 1
                continue
            except ValueError as refused:            # TrajectoryRefused: a renderer change
                logger.error("corpus export: %s no longer parses: %s", sid[:12], refused)
                counts["unparseable"] += 1
                continue
            counts["sft_rows"] += int(row["graded_success"] and row["replay_certified"])
            counts["rows"] += 1                      # the rows delivered
            yield row


def _raw_size(row: Mapping) -> int:
    """An upper bound on a row's encoded bytes before compression."""
    size = 0
    for value in row.values():
        if isinstance(value, str):
            size += 8 + len(value.encode())
        elif isinstance(value, list):
            size += 8 + 4 * len(value)
        else:
            size += 8
    return size


class _ShardWriter:
    """Parquet shards no larger than ``max_bytes``, handed to ``on_close`` as
    each one closes."""

    def __init__(self, directory: Path, *, max_bytes: int, row_group_rows: int,
                 on_close: Callable[[Path, int], Any], schema=None) -> None:
        self._directory = directory
        self._max = max_bytes
        self._group_rows = row_group_rows
        self._on_close = on_close
        self._schema = schema if schema is not None else _row_schema()
        self._buffer: list[dict] = []
        self._buffer_bytes = 0
        self._writer = None
        self._path: Path | None = None
        self._shard_bytes = 0
        self._shard_rows = 0
        self.index = 0

    async def add(self, row: dict) -> None:
        self._buffer.append(row)
        self._buffer_bytes += _raw_size(row)
        if len(self._buffer) >= self._group_rows or self._buffer_bytes >= ROW_GROUP_MAX_BYTES:
            await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        budget = self._max - SHARD_OVERHEAD_BYTES if self._max > 2 * SHARD_OVERHEAD_BYTES \
            else self._max // 2
        if self._writer is not None and self._shard_bytes + self._buffer_bytes > budget:
            await self._close_shard()
        if self._writer is None:
            self._path = self._directory / f"part-{self.index:05d}.parquet"
            self._writer = pq.ParquetWriter(str(self._path), self._schema, compression="zstd")
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        await asyncio.to_thread(self._writer.write_table, table)
        self._shard_bytes += self._buffer_bytes
        self._shard_rows += len(self._buffer)
        self._buffer, self._buffer_bytes = [], 0

    async def _close_shard(self) -> None:
        await asyncio.to_thread(self._writer.close)
        path, rows = self._path, self._shard_rows
        self._writer, self._path, self._shard_bytes, self._shard_rows = None, None, 0, 0
        self.index += 1
        size = path.stat().st_size
        if size > self._max:
            raise RuntimeError(f"shard {path.name} is {size} bytes, over {self._max}")
        await self._on_close(path, rows)

    async def close(self) -> None:
        await self._flush()
        if self._writer is not None:
            await self._close_shard()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instruction_source_for_job(job):
    """Only chat-template single-turn jobs have an authoritative raw user prompt.

    Legacy sources may already contain a rendered prompt; episode and system
    messages cannot be represented by the library's prompt/response schema.
    A missing source leaves the original Parquet delivery available.
    """
    from reliquary.validator.corpus_service import (
        CHAT_TEMPLATE_RENDERERS, CorpusPromptSourceError, SingleTurnPromptJob,
        prompt_job_for_spec,
    )

    if getattr(job, "episode", None) is not None:
        return None, "episode_schema"
    if getattr(job, "renderer_id", None) not in CHAT_TEMPLATE_RENDERERS:
        return None, "unsupported_renderer"
    try:
        source = prompt_job_for_spec(job)
    except CorpusPromptSourceError:
        return None, "prompt_source_unavailable"
    if not isinstance(source, SingleTurnPromptJob):
        return None, "unsupported_prompt_source"
    return source, None


class _InstructionWriter:
    """Bounded JSONL shards alongside (not replacing) the Parquet shards."""

    def __init__(self, directory: Path, on_close) -> None:
        self.directory, self.on_close = directory, on_close
        self.path, self.rows, self.size, self.index = None, 0, 0, 0

    async def add(self, prompt: str, response: str) -> None:
        body = (json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False,
                           separators=(",", ":")) + "\n").encode()
        if len(body) > INSTRUCTION_MAX_BYTES:
            raise ValueError("an instruction example exceeds its shard byte limit")
        if self.path is not None and (self.rows >= INSTRUCTION_MAX_ROWS
                                      or self.size + len(body) > INSTRUCTION_MAX_BYTES):
            await self.close()
        if self.path is None:
            self.path = self.directory / f"instruction-{self.index:05d}.jsonl"
        with self.path.open("ab") as handle:
            handle.write(body)
        self.rows += 1
        self.size += len(body)

    async def close(self) -> None:
        if self.path is not None:
            await self.on_close(self.path, self.rows)
            self.path, self.rows, self.size = None, 0, 0
            self.index += 1


def _instruction_text(value) -> bool:
    # JS String.trim() also removes FEFF, which Python str.strip() retains.
    if not isinstance(value, str) or re.search(r"[^\s\ufeff]", value) is None:
        return False
    try:
        return len(value.encode("utf-16-le")) // 2 <= INSTRUCTION_MAX_UTF16
    except UnicodeError:
        return False


async def export_delivery(*, job, records, sink, delivery_id: str, grade=None,
                          filter_note: str | None = None, work_dir: str | Path | None = None,
                          shard_max_bytes: int = SHARD_MAX_BYTES,
                          row_group_rows: int = ROW_GROUP_ROWS,
                          concurrency: int = READ_CONCURRENCY, window: int = READ_WINDOW,
                          clock: Callable[[], float] = time.time, renderer=None, source=None,
                          sft_only: bool = False,
                          quarantined: Collection[str] | None = None,
                          instruction_source=None, instruction_note: str | None = None) -> dict:
    """Write ``deliveries/{delivery_id}/``: the shards, ``report.json``, then
    ``manifest.json``. A delivery whose manifest exists is returned as stored.

    An episode job is delivered as episode rows (``episode_rows``): it needs
    its pinned ``renderer``, its task ``source`` and the registry's
    ``quarantined`` grade executors."""
    episode = getattr(job, "episode", None) is not None
    if episode and (renderer is None or source is None or quarantined is None):
        raise ValueError("an episode job is exported with its pinned renderer, its task source "
                         "and the quarantined grade executors")
    delivery_id = validated_delivery_id(delivery_id)
    prefix = f"{DELIVERY_PREFIX}/{delivery_id}"
    manifest_key = f"{prefix}/manifest.json"
    stored = await sink.get_json(manifest_key)
    if stored is not None:
        return stored
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"delivery-{delivery_id}-", dir=root))
    shards: list[dict] = []
    instruction_shards: list[dict] = []
    instruction_reason = ("episode_schema" if episode else instruction_note
                          or ("raw_prompt_source_not_provided" if instruction_source is None
                              else None))
    if instruction_reason is None:
        from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

        if getattr(job, "renderer_id", None) not in CHAT_TEMPLATE_RENDERERS:
            instruction_reason = "unsupported_renderer"
    instruction = {"format": "instruction", "rows": 0, "omitted": {},
                   "source": {"supported": instruction_reason is None,
                              "reason": instruction_reason},
                   "filter_applied": grade is not None,
                   "policy": ("accepted_completions" if grade is not None
                              else "audited_passing_completions"),
                   "limits": {"bytes_per_shard": INSTRUCTION_MAX_BYTES,
                              "rows_per_shard": INSTRUCTION_MAX_ROWS,
                              "utf16_units_per_field": INSTRUCTION_MAX_UTF16}}

    async def uploaded(path: Path, rows: int) -> None:
        digest = await asyncio.to_thread(_sha256, path)
        size = path.stat().st_size
        key = f"{prefix}/{path.name}"
        await sink.put_file(key, path)
        target = instruction_shards if path.suffix == ".jsonl" else shards
        target.append({"name": path.name, "key": key, "rows": rows, "bytes": size,
                       "sha256": digest})
        path.unlink()

    def omitted(reason: str) -> None:
        instruction["omitted"][reason] = instruction["omitted"].get(reason, 0) + 1

    counts: dict = {}
    try:
        writer = _ShardWriter(directory, max_bytes=shard_max_bytes,
                              row_group_rows=row_group_rows, on_close=uploaded,
                              schema=_episode_row_schema() if episode else None)
        instructions = _InstructionWriter(directory, uploaded)
        rows = (episode_rows(job=job, records=records, renderer=renderer, source=source,
                             counts=counts, sft_only=sft_only, quarantined=quarantined,
                             concurrency=concurrency, window=window)
                if episode else
                delivery_rows(job=job, records=records, counts=counts, grade=grade,
                              concurrency=concurrency, window=window))
        async for row in rows:
            await writer.add(row)
            if instruction_reason is not None:
                omitted(instruction_reason)
            elif grade is not None and row["accepted"] is not True:
                omitted("filter_rejected")
            else:
                from reliquary.validator.corpus_service import CorpusPromptSourceError

                try:
                    task = await asyncio.to_thread(instruction_source.task_for,
                                                   row["prompt_index"])
                except CorpusPromptSourceError:
                    omitted("prompt_source_unavailable")
                    continue
                if (getattr(task, "metadata", None) or {}).get("system"):
                    omitted("system_message")
                elif not (_instruction_text(task.prompt)
                          and _instruction_text(row["completion"])):
                    omitted("invalid_or_oversized_text")
                else:
                    await instructions.add(task.prompt, row["completion"])
                    instruction["rows"] += 1
        await writer.close()
        await instructions.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    report = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "created_at": clock(), "job": job.to_contract(), "counts": counts,
        "filter": (
            {"applied": sft_only,
             "note": ("graded_success and replay_certified" if sft_only
                      else "every replay-certified trajectory")} if episode
            else {"applied": True, "grader_id": job.filter.grader_id,
                  "threshold": job.filter.threshold} if grade is not None
            else {"applied": False, "note": filter_note or (
                "the declared filter was not applied" if job.filter is not None
                else "the job declares no filter")}),
        "shards": len(shards),
        "instruction": instruction,
    }
    report_key = f"{prefix}/report.json"
    await sink.put_json(report_key, report)
    manifest = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "created_at": report["created_at"], "rows": counts.get("rows", 0), "shards": shards,
        "instruction_shards": instruction_shards, "instruction": instruction,
        "columns": list(EPISODE_ROW_FIELDS if episode else ROW_FIELDS), "report": report_key,
        "keys": [s["key"] for s in shards + instruction_shards] + [report_key, manifest_key],
    }
    await sink.put_json(manifest_key, manifest)
    logger.info("corpus delivery %s of %s: %d rows in %d shards", delivery_id, job.job_id,
                manifest["rows"], len(shards))
    return manifest


class LocalDirectorySink:
    """A directory standing in for the platform bucket (tests, dry runs)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def put_file(self, key: str, path: Path) -> None:
        await asyncio.to_thread(shutil.copyfile, path, self._path(key))

    async def put_json(self, key: str, document: Mapping) -> None:
        self._path(key).write_text(json.dumps(document, sort_keys=True))

    async def get_json(self, key: str) -> dict | None:
        path = self._root / key
        return json.loads(path.read_text()) if path.exists() else None

    async def put_bytes(self, key: str, body: bytes) -> None:
        self._path(key).write_bytes(body)

    async def get_bytes(self, key: str) -> bytes | None:
        path = self._root / key
        return path.read_bytes() if path.exists() else None

    async def get_file(self, key: str, path: Path) -> bool:
        source = self._root / key
        if not source.exists():
            return False
        await asyncio.to_thread(shutil.copyfile, source, path)
        return True


class R2DeliverySink:
    """The platform bucket, through credentials scoped to it and held by the
    admin host only (``RELIQUARY_PLATFORM_R2_*``), never the subnet's."""

    def __init__(self, *, bucket: str, client=None) -> None:
        self._bucket = bucket
        self._client = client

    @classmethod
    def from_environment(cls) -> "R2DeliverySink":
        import boto3
        from botocore.config import Config

        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise RuntimeError(f"{name} is not set; deliveries need the platform bucket")
            return value

        account = required("RELIQUARY_PLATFORM_R2_ACCOUNT_ID")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("RELIQUARY_PLATFORM_R2_ENDPOINT_URL")
            or f"https://{account}.r2.cloudflarestorage.com",
            region_name=os.getenv("R2_REGION", "us-east-1"),
            aws_access_key_id=required("RELIQUARY_PLATFORM_R2_ACCESS_KEY_ID"),
            aws_secret_access_key=required("RELIQUARY_PLATFORM_R2_SECRET_ACCESS_KEY"),
            config=Config(connect_timeout=15, read_timeout=60,
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
        return cls(bucket=required("RELIQUARY_PLATFORM_BUCKET"), client=client)

    async def put_file(self, key: str, path: Path) -> None:
        from boto3.s3.transfer import TransferConfig

        config = TransferConfig(multipart_threshold=32 * 1024 * 1024,
                                multipart_chunksize=32 * 1024 * 1024, max_concurrency=8)
        await asyncio.to_thread(self._client.upload_file, str(path), self._bucket, key,
                                Config=config)

    async def put_json(self, key: str, document: Mapping) -> None:
        body = json.dumps(document, sort_keys=True).encode()
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key,
                                Body=body, ContentType="application/json")

    async def get_json(self, key: str) -> dict | None:
        from botocore.exceptions import ClientError

        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket,
                                               Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return json.loads(await asyncio.to_thread(response["Body"].read))

    async def put_bytes(self, key: str, body: bytes) -> None:
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key, Body=body)

    async def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket,
                                               Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return await asyncio.to_thread(response["Body"].read)

    async def get_file(self, key: str, path: Path) -> bool:
        """Stream an object to ``path``; False when it does not exist."""
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self._client.download_file, self._bucket, key, str(path))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise
        return True


__all__ = [
    "DELIVERY_SCHEMA",
    "EPISODE_ROW_FIELDS",
    "LocalDirectorySink",
    "R2DeliverySink",
    "ROW_FIELDS",
    "SHARD_MAX_BYTES",
    "delivery_rows",
    "episode_row",
    "episode_rows",
    "export_delivery",
    "instruction_source_for_job",
    "validated_delivery_id",
]
