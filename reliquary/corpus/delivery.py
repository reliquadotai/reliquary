"""Export v2: a job's passing rows delivered as Parquet shards; v3 (opt-in): the one delivery of every order.

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
DELIVERY_SCHEMA_V3 = "reliquary/corpus-delivery/v3"
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


class EpisodeKindMismatch(ValueError):
    """A record whose trajectory is not its job's kind: a transcript on a replay job's
    record, or none on a signed-sandbox job's. The job chooses the parser, never the
    record."""


class EpisodeSandboxUnavailable(ValueError):
    """A signed-sandbox job's row needs reliquary-sandbox, which this host cannot import."""


def episode_row(*, job, submission_id: str, record: Mapping, grade: Mapping, renderer,
                user_prompt: str) -> dict:
    """The row of one graded trajectory. Messages are rebuilt from the proven
    tokens with the pinned renderer (never from miner text), and the system and
    user messages must render to the record's prompt tokens, so the messages
    and the tokens are one and the same trajectory.

    The job chooses the parser (``is_signed_sandbox``): a replay job's record never
    carries a transcript and a signed job's always does (``EpisodeKindMismatch``).

    Raises ``TrajectoryRefused``, ``EpisodePromptMismatch``, ``EpisodeUnrendered``,
    ``EpisodeKindMismatch`` or ``EpisodeSandboxUnavailable``."""
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
    from reliquary.environment.agentic_swe import BASH_SYSTEM_PROMPT

    trajectory = record["completions"][0]
    prompt = [int(t) for t in trajectory["prompt_tokens"]]
    tokens = [int(t) for t in trajectory["tokens"]]
    spans = [(int(turn["start"]), int(turn["end"])) for turn in trajectory["turns"]]
    if [int(t) for t in renderer.initial_ids(user_prompt)] != prompt:
        raise EpisodePromptMismatch(submission_id)
    signed = trajectory.get("transcript")
    signed_job = is_signed_sandbox(job)
    if signed_job != (signed is not None):
        raise EpisodeKindMismatch(submission_id)
    if not signed_job:
        parsed = parse_trajectory(renderer, prompt_ids=prompt, tokens=tokens, spans=spans,
                                  stop=trajectory["stop"], max_turns=job.episode.max_turns)
        system_prompt = BASH_SYSTEM_PROMPT
        final_diff = trajectory["final_diff"]
    else:
        # A signed episode: its observations are the signed records' renderings, compared
        # forward against the tokens, never decoded from them (plan 3 ruling 6).
        try:
            from reliquary.corpus.signed_parse import parse_signed_trajectory, signed_records
            from reliquary.corpus.signed_state import swe_patch
            import reliquary_sandbox_service.episodes.state_archive  # noqa: F401
        except ImportError as exc:
            raise EpisodeSandboxUnavailable(submission_id) from exc
        from reliquary.corpus.signed_reasons import REASON_SANDBOX_TRANSCRIPT

        try:
            found = signed_records(signed)
        except (KeyError, TypeError, ValueError) as exc:
            raise TrajectoryRefused(REASON_SANDBOX_TRANSCRIPT, {"why": str(exc)}) from exc
        parsed = parse_signed_trajectory(renderer, prompt_ids=prompt, tokens=tokens, spans=spans,
                                         stop=trajectory["stop"], max_turns=job.episode.max_turns,
                                         calls=found.calls, offered=found.tools,
                                         final=found.final)
        # The job's harness, as the renderer shows it (the compatibility harness: verifiers' prompt).
        system_prompt = renderer.spec.system_prompt_for(found.tools)
        user_prompt = renderer.spec.user_prompt(user_prompt)
        # The graded state is the bridge's (header + artifact archive): the row carries
        # the git patch inside it, the shape of a replay job's final_diff. An unreadable
        # state is a ValueError (counted unparseable).
        final_diff = swe_patch(trajectory["final_diff"])
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    for t, ((start, end), turn) in enumerate(zip(spans, parsed.turns)):
        message = renderer.assistant_message(tokens[start:end])
        for k, call in enumerate(message.get("tool_calls") or ()):
            call["id"] = f"call_{t}_{k}"         # unique in the conversation, not per turn
        messages.append(message)
        for k, observation in enumerate(renderer.tool_messages(turn.observations)
                                        if signed_job else _tool_messages(turn.observations)):
            if "tool_call_id" in observation:
                observation["tool_call_id"] = f"call_{t}_{k}"
            messages.append(observation)
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
            "assistant_mask": mask, "final_diff": final_diff,
            "graded_success": bool(grade.get("graded_success")),
            "replay_certified": bool(grade.get("replay_certified")),
            "turns": json.dumps([[s, e] for s, e in spans]), "stop": trajectory["stop"]}


def _tool_messages(observations) -> list[dict]:
    """A replay job's observations, as verifiers' bash harness bridges them."""
    return [{"role": "tool", "tool_call_id": f"call_{k}", "content": text} for k, text in enumerate(observations)]


# Suite episode jobs (design 2026-10-13): one row per certified trajectory,
# in the form the post-training `kind: episode` reader takes (it trains on
# `tokens` and `assistant_mask` as they are, and drops a row whose
# `graded_success` or `replay_certified` is false). `tokens` are the proven
# ones rewritten into the chat template (`suite_episode.export_tokens`, named
# in `tokens_transform`); `messages` + `tools` are the same conversation for
# reading, rebuilt by replaying the trajectory, never from miner text.
SUITE_EPISODE_ROW_FIELDS = ("job_id", "submission_id", "prompt_index", "task_id", "messages",
                            "tools", "tokens", "assistant_mask", "tokens_transform",
                            "graded_success", "replay_certified", "reward", "turns", "stop")


class EpisodeUnreplayed(ValueError):
    """The stored trajectory no longer replays to itself (another environment
    or tokenizer than the one that admitted it)."""


def suite_episode_row(*, job, submission_id: str, record: Mapping, grade: Mapping, source,
                      encode, decode) -> dict:
    """The row of one graded suite trajectory. Raises ``EpisodePromptMismatch``
    or ``EpisodeUnreplayed``."""
    from reliquary.corpus.suite_episode import (
        END_OF_TEXT, IM_END, compare_replay, episode_seed, export_tokens, replay_suite, single_id,
        suite_messages, tools_of,
    )

    trajectory = record["completions"][0]
    index = int(record["prompt_index"])
    prompt = [int(t) for t in trajectory["prompt_tokens"]]
    tokens = [int(t) for t in trajectory["tokens"]]
    spans = [(int(turn["start"]), int(turn["end"])) for turn in trajectory["turns"]]
    task = source.task(index)
    if [int(t) for t in source.renderer(encode).encode_initial(task)] != prompt:
        raise EpisodePromptMismatch(submission_id)
    # The task through the source's row map (its environment is built at the
    # job's split); the seed is the row's.
    trace = replay_suite(source.environment, renderer_id=job.episode.renderer,
                         task_index=source.rows.index(index),
                         seed=episode_seed(job.job_id, index), prompt_ids=prompt, tokens=tokens,
                         spans=spans, encode=encode, decode=decode,
                         max_turns=job.episode.max_turns)
    refusal = compare_replay(trace, prompt_ids=prompt, tokens=tokens, spans=spans,
                             stop=trajectory["stop"])
    if refusal is not None:
        raise EpisodeUnreplayed(f"{submission_id}: {refusal}")
    full_spans = [(len(prompt) + s, len(prompt) + e) for s, e in spans]
    try:
        end_of_text = single_id(encode, END_OF_TEXT)
    except ValueError:
        end_of_text = None
    exported, mask, transforms = export_tokens(
        prompt + tokens, full_spans, im_end_id=single_id(encode, IM_END),
        newline_ids=encode("\n"), end_of_text_id=end_of_text)
    return {"job_id": job.job_id, "submission_id": submission_id, "prompt_index": index,
            "task_id": str(grade.get("instance_id") or trace.task_id),
            "messages": json.dumps(suite_messages(trace, task=task, decode=decode,
                                                  spans=full_spans), ensure_ascii=False),
            "tools": json.dumps(tools_of(task), ensure_ascii=False),
            "tokens": exported, "assistant_mask": mask,
            "tokens_transform": "+".join(transforms),
            "graded_success": bool(grade.get("graded_success")),
            "replay_certified": bool(grade.get("replay_certified")),
            "reward": float(trace.reward.reward),
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


async def passing_batches(*, job, records, counts: dict, concurrency: int = READ_CONCURRENCY,
                          window: int = READ_WINDOW,
                          on_verdict: Callable[[str, Mapping], Any] | None = None,
                          ) -> AsyncIterator[list[tuple[str, dict, dict]]]:
    """The one reader of a job's delivered work: ``(submission_id, verdict, record)`` of every passing, not
    voided submission whose record is readable, in verdict-id order, a window at a time (at most ``concurrency``
    reads in flight). The delivery and the eval grading both read through it.

    ``counts`` gets ``verdicts``, ``passing_submissions``, ``voided`` and ``missing_records``; ``on_verdict``
    sees every verdict read, passing or not."""
    from reliquary.corpus.export import submission_voided, voided_submission_ids

    ids = list(await records.list_verdict_ids(job.job_id))
    voided = await voided_submission_ids(records, job.job_id)
    counts.update(verdicts=len(ids), passing_submissions=0, voided=0, missing_records=0)
    gate = asyncio.Semaphore(concurrency)
    for start in range(0, len(ids), window):
        chunk = ids[start:start + window]
        verdicts = await _bounded((records.read_verdict(job.job_id, s) for s in chunk), gate)
        if on_verdict is not None:
            for sid, verdict in zip(chunk, verdicts):
                if verdict:
                    on_verdict(sid, verdict)
        passing = [(sid, v) for sid, v in zip(chunk, verdicts) if v and v.get("passed")]
        kept = [(sid, v) for sid, v in passing if sid not in voided]
        counts["voided"] += len(passing) - len(kept)
        found = await _bounded((records.read_submission(job.job_id, s) for s, _ in kept), gate)
        invalidated = await _bounded((submission_voided(records, job.job_id, s, voided)
                                      for s, _ in kept), gate)
        batch = []
        for (sid, verdict), record, is_voided in zip(kept, found, invalidated):
            if is_voided:
                counts["voided"] += 1
                continue
            if record is None:
                # A verdict can be visible before its record (R2 is not transactional).
                logger.warning("corpus delivery: submission %s passed but has no record", sid[:12])
                counts["missing_records"] += 1
                continue
            counts["passing_submissions"] += 1
            batch.append((sid, verdict, record))
        yield batch


def finish_reason(completion: Mapping, eos_token_id: int) -> str:
    """``"stop"`` when the completion ends on the model's eos, else ``"length"`` (cut at the budget)."""
    tokens = completion.get("tokens") or ()
    return "stop" if tokens and int(tokens[-1]) == int(eos_token_id) else "length"


async def delivery_rows(*, job, records, counts: dict, grade=None,
                        seen: set[bytes] | None = None,
                        concurrency: int = READ_CONCURRENCY,
                        window: int = READ_WINDOW) -> AsyncIterator[dict]:
    """One row per finished completion of every passing, not voided submission, in
    verdict-id order.

    No hotkey: the delivery is the work, not who did it. An unfinished
    completion (``unfinished_reason``: no EOS, or a thinking block not closed
    exactly once) is left out before grading and counted under ``unfinished``
    by reason. With ``seen`` (``is_duplicate``), a completion already seen, in
    this job or in the ones ``seen`` was filled from, is left out too, before
    grading, and counted as ``duplicate_completion``. ``grade`` annotates
    (``accepted``, ``score``) and never drops a row. ``counts`` is filled in as
    the rows go.
    """
    from reliquary.corpus.export import (
        is_duplicate, new_unfinished_counts, thinking_render, unfinished_reason,
    )

    thinking = thinking_render(job)
    counts.update(completions=0, unfinished=new_unfinished_counts(),
                  duplicate_completion=0 if seen is not None else None, rows=0,
                  rows_accepted=0 if grade is not None else None)
    async for batch in passing_batches(job=job, records=records, counts=counts,
                                       concurrency=concurrency, window=window):
        rows = []
        for sid, _, record in batch:
            for index, completion in enumerate(record["completions"]):
                counts["completions"] += 1
                reason = unfinished_reason(completion, eos_token_id=job.eos_token_id,
                                           thinking=thinking)
                if reason is not None:
                    counts["unfinished"][reason] += 1
                    continue
                if is_duplicate(completion, seen):
                    counts["duplicate_completion"] += 1
                    continue
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


# Delivery v3 (design 2026-10-31 §3.5): one schema for every order (a batch on a file, a batch on an env range,
# an eval), one row per delivered completion, with what a customer joins and checks on.
V3_ROW_FIELDS = ("job_id", "submission_id", "prompt_index", "problem_id", "custom_id", "completion_index",
                 "sample_index", "draw_slot", "seed", "prompt_sha256", "prompt_tokens", "completion_token_ids",
                 "completion", "completion_tokens", "finish_reason", "audited", "draw_match", "accepted", "score")
V3_FILTERS = ("none", "annotate", "drop")
V3_FORMATS = ("parquet", "openai_jsonl")


def _v3_row_schema():
    import pyarrow as pa

    return pa.schema([
        ("job_id", pa.string()), ("submission_id", pa.string()), ("prompt_index", pa.int64()),
        ("problem_id", pa.string()), ("custom_id", pa.string()), ("completion_index", pa.int32()),
        ("sample_index", pa.int64()), ("draw_slot", pa.int64()), ("seed", pa.int64()),
        ("prompt_sha256", pa.string()), ("prompt_tokens", pa.int64()),
        ("completion_token_ids", pa.list_(pa.int32())), ("completion", pa.string()),
        ("completion_tokens", pa.int32()), ("finish_reason", pa.string()), ("audited", pa.bool_()),
        ("draw_match", pa.float64()), ("accepted", pa.bool_()), ("score", pa.float64()),
    ])


def set_rows_for_job(job) -> tuple[dict, ...] | None:
    """The frozen set rows a job reads (an eval-set source: ours, or an upload compiled to one), None for a
    catalog source. Row ``i`` is the job's prompt ``i``."""
    from reliquary.eval.prompt_source import is_eval_source, load_eval_rows, parse_eval_source

    if not is_eval_source(getattr(job, "prompt_source", "")):
        return None
    return load_eval_rows(parse_eval_source(job.prompt_source))


def _row_ids(set_rows, prompt_index: int) -> tuple[str | None, str | None]:
    if set_rows is None or not 0 <= prompt_index < len(set_rows):
        return None, None
    row = set_rows[prompt_index]
    custom = row.get("custom_id")
    return row.get("problem_id"), custom if isinstance(custom, str) else None


def _prompt_token_counts(
    rendered: str, count_tokens, frozen_ids: list[int] | None,
) -> tuple[int | None, int | None]:
    """Return the reporting count and the billable count backed by matching frozen token IDs."""
    from reliquary.corpus.encoding import is_token_ids_text, prompt_token_ids

    if is_token_ids_text(rendered):
        try:
            ids = prompt_token_ids(None, rendered)
        except ValueError:
            return None, None
        length = len(ids)
        return length, length if ids == frozen_ids else None
    reported = None if count_tokens is None else int(count_tokens(rendered))
    return reported, None


async def delivery_rows_v3(*, job, records, counts: dict, grade=None, drop_rejected: bool = False,
                           include_truncated: bool = True, seen: set[bytes] | None = None,
                           set_rows=None, count_tokens=None,
                           on_verdict: Callable[[str, Mapping], Any] | None = None,
                           concurrency: int = READ_CONCURRENCY,
                           window: int = READ_WINDOW) -> AsyncIterator[dict]:
    """One v3 row per delivered completion of every passing, not voided submission, in verdict-id order.

    Every completion is a sample: one cut at the token budget (``finish_reason`` ``length``) is kept unless
    ``include_truncated`` is off; nothing is dropped for its thinking block. A forced-draw job's row names its
    draw slot, its sample index (``draw_slot * n + completion``, the index the draw keys) and the job's seed, so
    the customer can recompute it. ``audited``/``draw_match`` are the submission's verdict's (the match is the
    lowest of the submission's samples). ``grade`` annotates (``accepted``, ``score``); ``drop_rejected`` then
    leaves out the rejected rows."""
    from reliquary.corpus.eval_draw import has_draw, rollout_index
    from reliquary.corpus.export import is_duplicate

    drawn = has_draw(job)
    counts.update(completions=0, finish_reason={"stop": 0, "length": 0}, truncated_dropped=0,
                  duplicate_completion=0 if seen is not None else None, rows=0,
                  rows_accepted=0 if grade is not None else None,
                  rejected_dropped=0 if drop_rejected else None,
                  completion_tokens=0, audited_completion_tokens=0, prompt_tokens=0,
                  prompt_tokens_unknown=0, billable_prompt_tokens=0,
                  samples_by_prompt={}, delivered_by_prompt={})
    prompted: set[str] = set()
    async for batch in passing_batches(job=job, records=records, counts=counts, concurrency=concurrency,
                                       window=window, on_verdict=on_verdict):
        rows = []
        billable_prompts: dict[str, int] = {}
        for sid, verdict, record in batch:
            prompt_index = int(record["prompt_index"])
            problem_id, custom_id = _row_ids(set_rows, prompt_index)
            rendered = record.get("rendered_prompt") or ""
            prompt_sha = hashlib.sha256(rendered.encode()).hexdigest()
            frozen_ids = (set_rows[prompt_index].get("prompt_token_ids")
                          if set_rows is not None and 0 <= prompt_index < len(set_rows) else None)
            prompt_tokens, billable = _prompt_token_counts(rendered, count_tokens, frozen_ids)
            if billable is not None:
                billable_prompts[sid] = billable
            cursor = record.get("cursor")
            slot = int(cursor) if drawn and isinstance(cursor, int) and not isinstance(cursor, bool) else None
            audited = verdict.get("audited", True) is True
            match = (verdict.get("forced_draw") or {}).get("match") if audited else None
            for index, completion in enumerate(record["completions"]):
                counts["completions"] += 1
                # Generated and verified: a sample this delivery leaves out on purpose is not a missing one.
                generated = counts["samples_by_prompt"]
                generated[prompt_index] = generated.get(prompt_index, 0) + 1
                reason = finish_reason(completion, job.eos_token_id)
                if reason == "length" and not include_truncated:
                    counts["truncated_dropped"] += 1
                    continue
                if is_duplicate(completion, seen):
                    counts["duplicate_completion"] += 1
                    continue
                tokens = [int(t) for t in completion.get("tokens") or ()]
                rows.append({
                    "job_id": job.job_id, "submission_id": sid, "prompt_index": prompt_index,
                    "problem_id": problem_id, "custom_id": custom_id, "completion_index": index,
                    "sample_index": rollout_index(job, slot, index) if slot is not None else None,
                    "draw_slot": slot, "seed": int(job.seed) if drawn and job.seed is not None else None,
                    "prompt_sha256": prompt_sha, "prompt_tokens": prompt_tokens,
                    "completion_token_ids": tokens, "completion": completion["text"],
                    "completion_tokens": len(tokens), "finish_reason": reason, "audited": audited,
                    "draw_match": float(match) if isinstance(match, (int, float)) else None,
                    "accepted": None, "score": None,
                })
        if grade is not None and rows:
            def annotate(batch=rows):
                for row in batch:
                    accepted, score = grade(row["prompt_index"], row["completion"])
                    row["accepted"], row["score"] = bool(accepted), float(score)

            await asyncio.to_thread(annotate)
        for row in rows:
            if grade is not None and row["accepted"]:
                counts["rows_accepted"] += 1
            elif drop_rejected:
                counts["rejected_dropped"] += 1
                continue
            counts["rows"] += 1
            counts["finish_reason"][row["finish_reason"]] += 1
            counts["completion_tokens"] += row["completion_tokens"]
            if counts["billable_prompt_tokens"] is not None:
                # A missing count makes the delivery total unknown, even if other rows are exact.
                billable = billable_prompts.get(row["submission_id"])
                counts["billable_prompt_tokens"] = (
                    None if billable is None else counts["billable_prompt_tokens"] + billable)
            if row["audited"]:
                counts["audited_completion_tokens"] += row["completion_tokens"]
            if row["submission_id"] not in prompted:
                # A prompt counts once per submission: its completions share it.
                prompted.add(row["submission_id"])
                if row["prompt_tokens"] is None:
                    counts["prompt_tokens_unknown"] += 1
                else:
                    counts["prompt_tokens"] += row["prompt_tokens"]
            held = counts["delivered_by_prompt"]
            held[row["prompt_index"]] = held.get(row["prompt_index"], 0) + 1
            yield row


def _openai_choice(row: Mapping, *, chat: bool, index: int) -> dict:
    choice = {"index": index, "finish_reason": row["finish_reason"]}
    if chat:
        choice["message"] = {"role": "assistant", "content": row["completion"]}
    else:
        choice["text"] = row["completion"]
    choice["logprobs"] = None
    return choice


def openai_line(rows: list[Mapping], *, job, chat: bool, created_at: float) -> dict:
    """One OpenAI Batch output line for one prompt: its delivered samples as choices, in sample order."""
    first = rows[0]
    custom = first["custom_id"] if first["custom_id"] is not None else f"prompt-{first['prompt_index']}"
    request_id = f"{job.job_id}:{first['prompt_index']}"
    prompt_tokens = first["prompt_tokens"]
    completion_tokens = sum(int(r["completion_tokens"]) for r in rows)
    body = {"id": f"{'chatcmpl' if chat else 'cmpl'}-{hashlib.sha256(request_id.encode()).hexdigest()[:24]}",
            "object": "chat.completion" if chat else "text_completion",
            "model": job.checkpoint_repo, "created": int(created_at),
            "choices": [_openai_choice(r, chat=chat, index=i) for i, r in enumerate(rows)],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": None if prompt_tokens is None else prompt_tokens + completion_tokens}}
    return {"id": f"batch_req_{hashlib.sha256(request_id.encode()).hexdigest()[:24]}", "custom_id": custom,
            "response": {"status_code": 200, "request_id": request_id, "body": body}, "error": None}


def _chat_output(job, set_rows) -> bool:
    """A chat completion object unless the job's prompts were given as text-completion token ids: a set row
    names its endpoint (``url``) when it came from an OpenAI batch file."""
    if set_rows:
        url = set_rows[0].get("url")
        if isinstance(url, str):
            return url.endswith("/chat/completions")
        return "prompt_token_ids" not in set_rows[0]
    return True


class _JsonlShards:
    """JSONL shards (``{stem}-NNNNN.jsonl``) no larger than ``max_bytes``, handed to ``on_close``."""

    def __init__(self, directory: Path, stem: str, *, max_bytes: int, on_close) -> None:
        self.directory, self.stem, self.max, self.on_close = directory, stem, max_bytes, on_close
        self.path, self.rows, self.size, self.index = None, 0, 0, 0

    async def add(self, document: Mapping) -> None:
        body = (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if len(body) > self.max:
            raise ValueError(f"a {self.stem} line of {len(body)} bytes exceeds its shard limit")
        if self.path is not None and self.size + len(body) > self.max:
            await self.close()
        if self.path is None:
            self.path = self.directory / f"{self.stem}-{self.index:05d}.jsonl"
        with self.path.open("ab") as handle:
            handle.write(body)
        self.rows += 1
        self.size += len(body)

    async def close(self) -> None:
        if self.path is not None:
            await self.on_close(self.path, self.rows)
            self.path, self.rows, self.size = None, 0, 0
            self.index += 1


class _ByPrompt:
    """Rows parked on local disk (sqlite) and read back grouped by prompt, in prompt then sample order: an
    OpenAI line holds all of a prompt's samples, which the verdict order scatters."""

    def __init__(self, directory: Path) -> None:
        import sqlite3

        self._db = sqlite3.connect(str(directory / "by-prompt.sqlite"))
        self._db.execute("CREATE TABLE rows (prompt INTEGER, sample INTEGER, seq INTEGER, body TEXT)")
        self._seq = 0

    def add(self, row: Mapping) -> None:
        sample = row["sample_index"] if row["sample_index"] is not None else self._seq
        self._db.execute("INSERT INTO rows VALUES (?, ?, ?, ?)",
                         (int(row["prompt_index"]), int(sample), self._seq, json.dumps(row)))
        self._seq += 1

    def groups(self):
        self._db.execute("CREATE INDEX by_prompt ON rows (prompt, sample, seq)")
        current, rows = None, []
        for prompt, body in self._db.execute("SELECT prompt, body FROM rows ORDER BY prompt, sample, seq"):
            if prompt != current and rows:
                yield rows
                rows = []
            current = prompt
            rows.append(json.loads(body))
        if rows:
            yield rows

    def close(self) -> None:
        self._db.close()


def missing_prompts(job, samples_by_prompt: Mapping[int, int], slots=None) -> list[tuple[int, int, str]]:
    """``(prompt_index, held, code)`` of every prompt of the job short of its samples: ``prompt_exhausted`` when
    the job's ledger ``slots`` says every attempt was used (``draw_unfillable`` for a forced-draw job), else
    ``missing_samples``."""
    from reliquary.corpus.eval_draw import has_draw

    wanted = int(job.slots_per_prompt) * int(job.sampling.n)
    exhausted = "draw_unfillable" if has_draw(job) else "prompt_exhausted"
    short = []
    for index in range(job.prompt_start, job.prompt_start + job.prompt_count):
        held = int(samples_by_prompt.get(index, 0))
        if held < wanted:
            code = exhausted if slots is not None and slots.prompt_state(index) == "exhausted" \
                else "missing_samples"
            short.append((index, held, code))
    return short


def verification_block(job, *, verdicts: int, audited: int, refusals: Mapping[str, int]) -> dict:
    """What the delivery's samples were checked against: the forced draw (key recipe, warp, judge) or the
    proof, and how much was audited."""
    from reliquary.corpus.eval_draw import EVAL_DRAW_DOMAIN, draw_randomness, has_draw, job_warp

    block = {"verdicts": verdicts, "audited": audited,
             "audited_share": (audited / verdicts) if verdicts else None,
             "refusals": dict(sorted(refusals.items()))}
    if has_draw(job):
        from reliquary.corpus.job import DEFAULT_DRAW_JUDGE

        judge = getattr(job, "judge", None) or DEFAULT_DRAW_JUDGE
        block.update(scheme="forced-draw", draw={
            "kind": job.sampling.draw, "domain": EVAL_DRAW_DOMAIN, "seed": job.seed,
            "randomness": draw_randomness(job), "checkpoint_hash": job.checkpoint_revision,
            "warp": job_warp(job).to_dict(),
            "recipe": "randomness = sha256 of the length-prefixed (domain, job_id, checkpoint_sha256, seed); a "
                      "sample's uniforms are u_range(randomness, prompt_index, checkpoint_hash, "
                      "draw_slot * n + completion_index, positions, domain)"},
            judge={k: getattr(judge, k) for k in (
                "floor_rate", "grace", "gap_tolerance", "terminal_gap", "edge_margin")})
    else:
        block.update(scheme="toploc-v1", draw=None, judge=None)
    return block


async def episode_rows(*, job, records, renderer, source, counts: dict, sft_only: bool = False,
                       quarantined: Collection[str] | None,
                       concurrency: int = READ_CONCURRENCY,
                       window: int = READ_WINDOW, codec=None) -> AsyncIterator[dict]:
    """One row per passing, not voided, certified trajectory, in verdict-id
    order; ``sft_only`` keeps the certified successes (spec N6's SFT set).

    The effective grade is the latest regrade, else the grade. A grade that an
    executor of ``quarantined`` decided alone is held, as the settler holds it;
    the caller names the registry's quarantined grade executors (None refuses:
    an export blind to quarantines must not run).

    A suite job's rows are ``suite_episode_row``s: ``source`` is its
    ``SuiteSource``, ``codec`` the checkpoint tokenizer's ``encode``/``decode``
    (an object with both), and ``renderer`` is unused."""
    suite = job.episode.is_suite
    if suite and codec is None:
        raise ValueError("a suite episode export replays its trajectories: it needs a codec")
    if quarantined is None:
        raise ValueError("an episode export needs the quarantined grade executors "
                         "(the registry's), to hold what they decided alone")
    quarantined = frozenset(quarantined)
    ids = list(await records.list_verdict_ids(job.job_id))
    voided = set(await records.list_voided_ids(job.job_id))
    read_voided = getattr(records, "read_voided", None)
    counts.update(verdicts=len(ids), passing_submissions=0, voided=0, ungraded=0, held=0,
                  uncertified=0, not_successful=0, missing_records=0, task_mismatch=0,
                  prompt_mismatch=0, unparseable=0, unrendered=0, kind_mismatch=0,
                  sandbox_unavailable=0, sft_rows=0, rows=0)
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
                if suite:
                    row = await asyncio.to_thread(
                        suite_episode_row, job=job, submission_id=sid, record=record,
                        grade=document, source=source, encode=codec.encode, decode=codec.decode)
                else:
                    row = await asyncio.to_thread(
                        episode_row, job=job, submission_id=sid, record=record, grade=document,
                        renderer=renderer, user_prompt=source.prompt(index))
            except EpisodeUnrendered:
                logger.error("corpus export: %s's messages do not render to its tokens", sid[:12])
                counts["unrendered"] += 1
                continue
            except EpisodeKindMismatch:
                logger.error("corpus export: %s's record is not its job's kind (replay or "
                             "signed)", sid[:12])
                counts["kind_mismatch"] += 1
                continue
            except EpisodeSandboxUnavailable:
                logger.error("corpus export: %s is a signed episode and reliquary-sandbox is "
                             "not installed here", sid[:12])
                counts["sandbox_unavailable"] += 1
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
        raw_bytes = _raw_size(row)
        group_budget = min(ROW_GROUP_MAX_BYTES, max(1, self._max // 2))
        if self._buffer and self._buffer_bytes + raw_bytes > group_budget:
            await self._flush()
        self._buffer.append(row)
        self._buffer_bytes += raw_bytes
        if len(self._buffer) >= self._group_rows or self._buffer_bytes >= group_budget:
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


def validate_cached_delivery(manifest: Mapping, *, job, sink) -> None:
    """Bind a completed delivery to its immutable job, retaining legacy storage reads."""
    from reliquary.protocol.release_contract import canonical_sha256

    if manifest.get("job_id") != job.job_id:
        raise ValueError("delivery belongs to another job")
    if ("job_manifest_sha256" in manifest
            or getattr(sink, "requires_job_contract_pin", False)):
        if manifest.get("job_manifest_sha256") != canonical_sha256(job.to_contract()):
            raise ValueError("delivery belongs to another job contract")


async def export_delivery(*, job, records, sink, delivery_id: str, grade=None,
                          filter_note: str | None = None, work_dir: str | Path | None = None,
                          shard_max_bytes: int = SHARD_MAX_BYTES,
                          row_group_rows: int = ROW_GROUP_ROWS,
                          concurrency: int = READ_CONCURRENCY, window: int = READ_WINDOW,
                          clock: Callable[[], float] = time.time, renderer=None, source=None,
                          sft_only: bool = False,
                          quarantined: Collection[str] | None = None,
                          instruction_source=None, instruction_note: str | None = None,
                          keep_rejected: bool = False, dedupe: bool = False,
                          dedupe_against: Collection[str] = (), schema_version: int = 2,
                          include_truncated: bool = True, filter_mode: str = "none",
                          output_format: str = "parquet", slots=None, set_rows=None,
                          count_tokens=None) -> dict:
    """Write ``deliveries/{delivery_id}/``: the shards, ``report.json``, then
    ``manifest.json``. A delivery whose manifest exists is returned as stored.

    ``schema_version`` 3 is the one delivery of every order (``_export_v3``): ``include_truncated``,
    ``filter_mode`` (``none``, ``annotate`` or ``drop``, with ``grade``) and ``output_format`` (``parquet`` or
    ``openai_jsonl``) apply to it only; ``slots`` (the job's ledger) names exhausted prompts in its errors,
    ``set_rows`` maps a frozen set's rows to their ids and ``count_tokens`` counts text prompts.

    ``keep_rejected`` keeps the finished completions ``grade`` rejects in the
    instruction shards instead of omitting them (an SFT without the
    correctness filter); the Parquet rows carry ``accepted``/``score`` either
    way. Unfinished completions are never delivered.

    ``dedupe`` delivers a completion once (by ``completion_fingerprint``):
    the first copy in verdict-id order, and none already in a passing
    submission of a job in ``dedupe_against`` (which implies ``dedupe``), e.g.
    a v2 job on the prompts of its v1.

    An episode job is delivered as episode rows (``episode_rows``): it needs
    its pinned ``renderer``, its task ``source`` and the registry's
    ``quarantined`` grade executors."""
    episode = getattr(job, "episode", None) is not None
    if schema_version == 3:
        if episode:
            raise ValueError("an episode job is exported with `reliquary jobs export --sft` (delivery v3 holds "
                             "single-turn completions)")
        if keep_rejected:
            raise ValueError("delivery v3 keeps rejected completions with filter_mode 'annotate'")
        return await _export_v3(
            job=job, records=records, sink=sink, delivery_id=delivery_id, grade=grade, filter_note=filter_note,
            work_dir=work_dir, shard_max_bytes=shard_max_bytes, row_group_rows=row_group_rows,
            concurrency=concurrency, window=window, clock=clock, dedupe=dedupe, dedupe_against=dedupe_against,
            include_truncated=include_truncated, filter_mode=filter_mode, output_format=output_format,
            slots=slots, set_rows=set_rows, count_tokens=count_tokens)
    if schema_version != 2:
        raise ValueError(f"delivery schema version {schema_version!r} is not 2 or 3")
    if keep_rejected and (episode or grade is None):
        raise ValueError("keep_rejected needs a grader: it keeps what the filter rejects")
    dedupe_against = sorted(set(dedupe_against))
    if job.job_id in dedupe_against:
        raise ValueError("a delivery cannot be deduplicated against itself")
    dedupe = dedupe or bool(dedupe_against)
    if episode and dedupe:
        raise ValueError("an episode delivery is not deduplicated by completion")
    if episode and job.episode.is_suite:
        # Its rows have their own fields (SUITE_EPISODE_ROW_FIELDS), which the
        # Parquet schema here does not carry.
        raise ValueError("a suite episode job is exported with `reliquary jobs export --sft`")
    if episode and (renderer is None or source is None or quarantined is None):
        raise ValueError("an episode job is exported with its pinned renderer, its task source "
                         "and the quarantined grade executors")
    delivery_id = validated_delivery_id(delivery_id)
    prefix = f"{DELIVERY_PREFIX}/{delivery_id}"
    manifest_key = f"{prefix}/manifest.json"
    stored = await sink.get_json(manifest_key)
    if stored is not None:
        validate_cached_delivery(stored, job=job, sink=sink)
        return stored
    report_key = f"{prefix}/report.json"
    stored_report = await sink.get_json(report_key)
    if stored_report is not None:
        if (stored_report.get("job_id") != job.job_id
                or stored_report.get("delivery_id") != delivery_id
                or stored_report.get("job") != job.to_contract()):
            raise ValueError("delivery report belongs to another job contract")
        stamp = stored_report.get("created_at")
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
            raise ValueError("delivery report has no creation timestamp")
    else:
        stamp = clock()
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
                   "policy": ("audited_passing_completions" if grade is None
                              else "finished_completions_annotated" if keep_rejected
                              else "accepted_completions"),
                   "limits": {"bytes_per_shard": INSTRUCTION_MAX_BYTES,
                              "rows_per_shard": INSTRUCTION_MAX_ROWS,
                              "utf16_units_per_field": INSTRUCTION_MAX_UTF16}}
    if keep_rejected:
        instruction["rejected_kept"] = 0

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
    completion_tokens = 0
    seen = None
    if dedupe:
        from reliquary.corpus.export import job_fingerprints

        seen = await job_fingerprints(records, dedupe_against, concurrency=concurrency)
    fingerprints_against = len(seen) if seen is not None else 0
    try:
        shard_max_bytes = min(shard_max_bytes, getattr(sink, "max_file_bytes", shard_max_bytes))
        writer = _ShardWriter(directory, max_bytes=shard_max_bytes,
                              row_group_rows=row_group_rows, on_close=uploaded,
                              schema=_episode_row_schema() if episode else None)
        instructions = _InstructionWriter(directory, uploaded)
        rows = (episode_rows(job=job, records=records, renderer=renderer, source=source,
                             counts=counts, sft_only=sft_only, quarantined=quarantined,
                             concurrency=concurrency, window=window)
                if episode else
                delivery_rows(job=job, records=records, counts=counts, grade=grade,
                              seen=seen, concurrency=concurrency, window=window))
        async for row in rows:
            await writer.add(row)
            if not episode:
                completion_tokens += row["completion_tokens"]
            if instruction_reason is not None:
                omitted(instruction_reason)
            elif grade is not None and row["accepted"] is not True and not keep_rejected:
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
                    if keep_rejected and row["accepted"] is not True:
                        instruction["rejected_kept"] += 1
        await writer.close()
        await instructions.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    report = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "created_at": stamp, "job": job.to_contract(), "counts": counts,
        "filter": (
            {"applied": sft_only,
             "note": ("graded_success and replay_certified" if sft_only
                      else "every replay-certified trajectory")} if episode
            else {"applied": True, "grader_id": job.filter.grader_id,
                  "threshold": job.filter.threshold, "keep_rejected": keep_rejected}
            if grade is not None
            else {"applied": False, "note": filter_note or (
                "the declared filter was not applied" if job.filter is not None
                else "the job declares no filter")}),
        "shards": len(shards),
        "instruction": instruction,
    }
    if not episode:
        report["billing"] = {"basis": "completion_tokens_of_delivered_rows",
                             "completion_tokens": completion_tokens}
    if dedupe:
        from reliquary.corpus.export import DEDUPE_KEY

        report["dedupe"] = {"within_job": True, "against": dedupe_against,
                            "fingerprints_against": fingerprints_against, "key": DEDUPE_KEY}
    await sink.put_json(report_key, report)
    from reliquary.protocol.release_contract import canonical_sha256

    manifest = {
        "schema": DELIVERY_SCHEMA, "delivery_id": delivery_id, "job_id": job.job_id,
        "job_manifest_sha256": canonical_sha256(job.to_contract()),
        "created_at": report["created_at"], "rows": counts.get("rows", 0), "shards": shards,
        "instruction_shards": instruction_shards, "instruction": instruction,
        "columns": list(EPISODE_ROW_FIELDS if episode else ROW_FIELDS), "report": report_key,
        **({"billing": report["billing"]} if "billing" in report else {}),
        "keys": [s["key"] for s in shards + instruction_shards] + [report_key, manifest_key],
    }
    await sink.put_json(manifest_key, manifest)
    logger.info("corpus delivery %s of %s: %d rows in %d shards", delivery_id, job.job_id,
                manifest["rows"], len(shards))
    return manifest


async def _stored_stamp(sink, report_key: str, *, job, delivery_id: str, clock) -> float:
    """The creation time of a delivery whose report was written before a crash, else now."""
    stored_report = await sink.get_json(report_key)
    if stored_report is None:
        return clock()
    if (stored_report.get("job_id") != job.job_id or stored_report.get("delivery_id") != delivery_id
            or stored_report.get("job") != job.to_contract()):
        raise ValueError("delivery report belongs to another job contract")
    stamp = stored_report.get("created_at")
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
        raise ValueError("delivery report has no creation timestamp")
    return stamp


async def _export_v3(*, job, records, sink, delivery_id: str, grade, filter_note, work_dir, shard_max_bytes,
                     row_group_rows, concurrency, window, clock, dedupe, dedupe_against, include_truncated,
                     filter_mode, output_format, slots, set_rows, count_tokens) -> dict:
    """Delivery v3: the job's delivered completions as v3 Parquet rows or OpenAI Batch output lines (one per
    prompt, its samples as choices) plus an errors file naming the prompts short of their samples, then
    ``report.json`` (counts, verification, billing basis) and ``manifest.json``."""
    from reliquary.protocol.release_contract import canonical_sha256

    if filter_mode not in V3_FILTERS:
        raise ValueError(f"filter_mode must be one of {V3_FILTERS}")
    if output_format not in V3_FORMATS:
        raise ValueError(f"output_format must be one of {V3_FORMATS}")
    if (filter_mode != "none") != (grade is not None):
        raise ValueError("filter_mode 'annotate' and 'drop' need the job's grader, 'none' takes none")
    dedupe_against = sorted(set(dedupe_against))
    if job.job_id in dedupe_against:
        raise ValueError("a delivery cannot be deduplicated against itself")
    dedupe = dedupe or bool(dedupe_against)
    delivery_id = validated_delivery_id(delivery_id)
    prefix = f"{DELIVERY_PREFIX}/{delivery_id}"
    manifest_key, report_key = f"{prefix}/manifest.json", f"{prefix}/report.json"
    stored = await sink.get_json(manifest_key)
    if stored is not None:
        validate_cached_delivery(stored, job=job, sink=sink)
        return stored
    stamp = await _stored_stamp(sink, report_key, job=job, delivery_id=delivery_id, clock=clock)
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"delivery-{delivery_id}-", dir=root))
    shards: list[dict] = []
    error_shards: list[dict] = []

    async def uploaded(path: Path, rows: int) -> None:
        digest = await asyncio.to_thread(_sha256, path)
        key = f"{prefix}/{path.name}"
        await sink.put_file(key, path)
        (error_shards if path.name.startswith("errors-") else shards).append(
            {"name": path.name, "key": key, "rows": rows, "bytes": path.stat().st_size, "sha256": digest})
        path.unlink()

    counts: dict = {}
    refusals: dict[str, int] = {}
    audited = 0

    def on_verdict(_sid: str, verdict: Mapping) -> None:
        nonlocal audited
        if verdict.get("audited", True) is True:
            audited += 1
        if not verdict.get("passed"):
            reason = str(verdict.get("reason") or "unknown")
            refusals[reason] = refusals.get(reason, 0) + 1

    seen = None
    if dedupe:
        from reliquary.corpus.export import job_fingerprints

        seen = await job_fingerprints(records, dedupe_against, concurrency=concurrency)
    errors_written = 0
    try:
        shard_max_bytes = min(shard_max_bytes, getattr(sink, "max_file_bytes", shard_max_bytes))
        rows = delivery_rows_v3(job=job, records=records, counts=counts, grade=grade,
                                drop_rejected=filter_mode == "drop", include_truncated=include_truncated,
                                seen=seen, set_rows=set_rows, count_tokens=count_tokens, on_verdict=on_verdict,
                                concurrency=concurrency, window=window)
        errors = _JsonlShards(directory, "errors", max_bytes=shard_max_bytes, on_close=uploaded)
        if output_format == "parquet":
            writer = _ShardWriter(directory, max_bytes=shard_max_bytes, row_group_rows=row_group_rows,
                                  on_close=uploaded, schema=_v3_row_schema())
            async for row in rows:
                await writer.add(row)
            await writer.close()
        else:
            parked = _ByPrompt(directory)
            try:
                async for row in rows:
                    parked.add(row)
                output = _JsonlShards(directory, "output", max_bytes=shard_max_bytes, on_close=uploaded)
                chat = _chat_output(job, set_rows)
                for group in parked.groups():
                    await output.add(openai_line(group, job=job, chat=chat, created_at=stamp))
                await output.close()
            finally:
                parked.close()
        wanted = int(job.slots_per_prompt) * int(job.sampling.n)
        short = missing_prompts(job, counts["samples_by_prompt"], slots)
        for index, held, code in short:
            _, custom_id = _row_ids(set_rows, index)
            await errors.add({"custom_id": custom_id if custom_id is not None else f"prompt-{index}",
                              "prompt_index": index,
                              "error": {"code": code, "message": f"{held} of {wanted} samples delivered"}})
            errors_written += 1
        await errors.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    counts.pop("samples_by_prompt")
    held = counts.pop("delivered_by_prompt")
    counts["prompts_delivered"] = len(held)
    counts["prompts_short"] = errors_written
    report = {
        "schema": DELIVERY_SCHEMA_V3, "delivery_id": delivery_id, "job_id": job.job_id, "created_at": stamp,
        "job": job.to_contract(), "counts": counts,
        "options": {"format": output_format, "include_truncated": include_truncated, "filter": filter_mode,
                    "dedupe": dedupe, "dedupe_against": dedupe_against},
        "filter": ({"applied": True, "mode": filter_mode, "grader_id": job.filter.grader_id,
                    "threshold": job.filter.threshold} if grade is not None
                   else {"applied": False, "mode": "none", "note": filter_note}),
        "verification": verification_block(job, verdicts=counts["verdicts"], audited=audited, refusals=refusals),
        "billing": {"basis": "completion_tokens_of_delivered_rows",
                    "completion_tokens": counts["completion_tokens"],
                    "audited_completion_tokens": counts["audited_completion_tokens"],
                    "prompt_tokens": counts["prompt_tokens"],
                    "prompt_tokens_unknown": counts["prompt_tokens_unknown"],
                    "billable_prompt_tokens": counts["billable_prompt_tokens"]},
        "shards": len(shards), "error_shards": len(error_shards),
    }
    await sink.put_json(report_key, report)
    manifest = {
        "schema": DELIVERY_SCHEMA_V3, "delivery_id": delivery_id, "job_id": job.job_id,
        "job_manifest_sha256": canonical_sha256(job.to_contract()), "created_at": stamp,
        "format": output_format, "options": report["options"],
        "rows": counts.get("rows", 0), "shards": shards, "error_shards": error_shards,
        "instruction_shards": [],
        "columns": list(V3_ROW_FIELDS) if output_format == "parquet" else None, "report": report_key,
        "billing": report["billing"],
        "keys": [s["key"] for s in shards + error_shards] + [report_key, manifest_key],
    }
    await sink.put_json(manifest_key, manifest)
    logger.info("corpus delivery v3 %s of %s: %d rows in %d shards (%s)", delivery_id, job.job_id,
                manifest["rows"], len(shards), output_format)
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

        digest = await asyncio.to_thread(_sha256, path)
        config = TransferConfig(multipart_threshold=32 * 1024 * 1024,
                                multipart_chunksize=32 * 1024 * 1024, max_concurrency=8)
        await asyncio.to_thread(self._client.upload_file, str(path), self._bucket, key,
                                Config=config, ExtraArgs={"Metadata": {"sha256": digest}})

    async def put_json(self, key: str, document: Mapping) -> None:
        body = json.dumps(document, sort_keys=True).encode()
        await asyncio.to_thread(self._client.put_object, Bucket=self._bucket, Key=key,
                                Body=body, ContentType="application/json",
                                Metadata={"sha256": hashlib.sha256(body).hexdigest()})

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


class HTTPDeliverySink:
    """Immutable bounded deliveries and batch inputs on one HTTPS origin."""

    max_file_bytes = 64 * 1024 * 1024
    max_json_bytes = 1024 * 1024
    requires_job_contract_pin = True
    evaluation_supported = False
    _key = re.compile(r"^deliveries/subnet-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
                      r"[89ab][0-9a-f]{3}-[0-9a-f]{12})/([A-Za-z0-9_.()-]{1,100})$")
    _uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    _input = re.compile(rf"^batch-inputs/{_uuid}/(file-{_uuid})\.jsonl$")
    _errors = re.compile(r"^batch-uploads/([0-9a-f]{32})/errors\.jsonl$")
    _batch = re.compile(rf"^deliveries/dl-({_uuid})/([A-Za-z0-9_.()-]{{1,100}})$")
    _batch_root = "/api/internal/subnet/batch"
    max_input_bytes = 200 * 1024 * 1024

    def __init__(self, *, url: str, secret: str) -> None:
        from urllib.parse import urlsplit

        origin = urlsplit(url)
        if (origin.scheme != "https" or not origin.hostname or origin.username is not None
                or origin.password is not None or origin.path not in ("", "/")
                or origin.query or origin.fragment):
            raise ValueError("platform delivery URL must be an HTTPS origin")
        if len(secret) < 32 or not secret.isascii() or any(ord(c) < 33 or ord(c) == 127
                                                        for c in secret):
            raise ValueError("platform delivery secret must be at least 32 header-safe characters")
        self._host, self._port, self._secret = origin.hostname, origin.port or 443, secret

    @classmethod
    def from_environment(cls) -> "HTTPDeliverySink":
        return cls(url=os.getenv("RELIQUARY_PLATFORM_DELIVERY_URL", "").strip(),
                   secret=os.getenv("RELIQUARY_PLATFORM_DELIVERY_SECRET", ""))

    def _route(self, key: str) -> str:
        match = self._key.fullmatch(key)
        if match is not None:
            return f"/api/internal/subnet/tasks/{match[1]}/deliveries/{match[2]}"
        match = self._input.fullmatch(key)
        if match is not None:
            return f"{self._batch_root}/files/{match[1]}/content"
        match = self._errors.fullmatch(key)
        if match is not None:
            return f"{self._batch_root}/uploads/{match[1]}/errors.jsonl"
        match = self._batch.fullmatch(key)
        if match is not None and match[2] not in (".", ".."):
            return f"{self._batch_root}/orders/{match[1]}/deliveries/{match[2]}"
        raise ValueError("HTTP delivery sink only accepts subnet-run and batch keys")

    def accepts_delivery_id(self, delivery_id: str) -> bool:
        key = f"deliveries/{delivery_id}/manifest.json"
        return self._key.fullmatch(key) is not None or self._batch.fullmatch(key) is not None

    def _connection(self):
        from http.client import HTTPSConnection

        return HTTPSConnection(self._host, self._port, timeout=60)

    def _put(self, key: str, *, size: int, digest: str, body=None, path=None,
             route: str | None = None) -> None:
        route = route or self._route(key)
        limit = self.max_json_bytes if key.endswith(".json") else self.max_file_bytes
        minimum = 0 if route.startswith(self._batch_root + "/") and not key.endswith(".json") else 1
        if size < minimum or size > limit:
            raise ValueError("HTTP delivery object exceeds its size limit")
        conn = self._connection()
        try:
            conn.putrequest("PUT", route)
            conn.putheader("Authorization", "Bearer " + self._secret)
            conn.putheader("Content-Length", str(size))
            conn.putheader("X-Content-SHA256", digest)
            conn.putheader("Content-Type", "application/json" if key.endswith(".json")
                           else "application/octet-stream")
            conn.endheaders()
            if path is not None:
                with Path(path).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        conn.send(block)
            else:
                conn.send(body)
            response = conn.getresponse()
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"platform delivery upload refused ({response.status})")
            response.read(self.max_json_bytes + 1)
        finally:
            conn.close()

    async def put_file(self, key: str, path: Path) -> None:
        digest = await asyncio.to_thread(_sha256, path)
        await asyncio.to_thread(self._put, key, size=path.stat().st_size, digest=digest, path=path)

    async def put_json(self, key: str, document: Mapping) -> None:
        body = json.dumps(document, sort_keys=True).encode()
        await asyncio.to_thread(self._put, key, size=len(body),
                                digest=hashlib.sha256(body).hexdigest(), body=body)

    async def put_upload_errors(self, key: str, body: bytes, *, object_key: str) -> None:
        source, errors = self._input.fullmatch(object_key), self._errors.fullmatch(key)
        if source is None or errors is None:
            raise ValueError("batch errors require their processed input file")
        route = f"{self._batch_root}/files/{source[1]}/uploads/{errors[1]}/errors.jsonl"
        await asyncio.to_thread(self._put, key, size=len(body),
                                digest=hashlib.sha256(body).hexdigest(), body=body, route=route)

    def _get_bytes(self, key: str) -> bytes | None:
        if self._input.fullmatch(key):
            limit = self.max_input_bytes
        elif self._errors.fullmatch(key):
            limit = self.max_file_bytes
        else:
            raise ValueError("HTTP batch reads require an input or its errors")
        conn = self._connection()
        try:
            conn.request("GET", self._route(key), headers={"Authorization": "Bearer " + self._secret})
            response = conn.getresponse()
            if response.status == 404:
                return None
            if response.status != 200:
                raise RuntimeError(f"platform batch read refused ({response.status})")
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError("HTTP batch object exceeds its size limit")
            return body
        finally:
            conn.close()

    async def get_bytes(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._get_bytes, key)

    def _get_json(self, key: str) -> dict | None:
        route = self._route(key)
        if key.rsplit("/", 1)[1] not in ("manifest.json", "report.json"):
            raise ValueError("HTTP delivery reads only allow its manifest and report")
        conn = self._connection()
        try:
            conn.request("GET", route, headers={"Authorization": "Bearer " + self._secret})
            response = conn.getresponse()
            if response.status == 404:
                return None
            if response.status != 200:
                raise RuntimeError(f"platform delivery read refused ({response.status})")
            body = response.read(self.max_json_bytes + 1)
            if len(body) > self.max_json_bytes:
                raise ValueError("HTTP delivery JSON exceeds its size limit")
            document = json.loads(body)
            if not isinstance(document, dict):
                raise ValueError("HTTP delivery JSON must be an object")
            return document
        finally:
            conn.close()

    async def get_json(self, key: str) -> dict | None:
        return await asyncio.to_thread(self._get_json, key)


__all__ = [
    "DELIVERY_SCHEMA",
    "DELIVERY_SCHEMA_V3",
    "V3_FILTERS",
    "V3_FORMATS",
    "V3_ROW_FIELDS",
    "delivery_rows_v3",
    "finish_reason",
    "missing_prompts",
    "openai_line",
    "passing_batches",
    "set_rows_for_job",
    "verification_block",
    "EPISODE_ROW_FIELDS",
    "EpisodeKindMismatch",
    "EpisodeSandboxUnavailable",
    "SUITE_EPISODE_ROW_FIELDS",
    "suite_episode_row",
    "LocalDirectorySink",
    "HTTPDeliverySink",
    "R2DeliverySink",
    "ROW_FIELDS",
    "SHARD_MAX_BYTES",
    "delivery_rows",
    "episode_row",
    "episode_rows",
    "export_delivery",
    "instruction_source_for_job",
    "validate_cached_delivery",
    "validated_delivery_id",
]


class SplitDeliverySink:
    """The HTTP sink for every key it routes (subnet runs, batch files), the
    platform bucket for the rest: evaluation files, which the HTTP routes do
    not cover. Lets one admin deliver both through ``RELIQUARY_PLATFORM_DELIVERY_*``
    and ``RELIQUARY_PLATFORM_R2_*`` together."""

    evaluation_supported = True

    def __init__(self, *, http: "HTTPDeliverySink", bucket: "R2DeliverySink") -> None:
        self._http, self._bucket = http, bucket

    def __getattr__(self, name: str):
        # Limits, contract pins and batch error uploads stay the HTTP sink's.
        return getattr(self._http, name)

    def _for(self, key: str):
        try:
            self._http._route(key)
        except ValueError:
            return self._bucket
        return self._http

    async def put_file(self, key: str, path: Path) -> None:
        await self._for(key).put_file(key, path)

    async def put_json(self, key: str, document: Mapping) -> None:
        await self._for(key).put_json(key, document)

    async def put_bytes(self, key: str, body: bytes) -> None:
        sink = self._for(key)
        if sink is self._http:
            raise ValueError("bytes go to the platform bucket only")
        await sink.put_bytes(key, body)

    async def get_json(self, key: str) -> dict | None:
        return await self._for(key).get_json(key)

    async def get_bytes(self, key: str) -> bytes | None:
        return await self._for(key).get_bytes(key)

    async def get_file(self, key: str, path: Path):
        sink = self._for(key)
        if sink is self._http:
            raise ValueError("files are read from the platform bucket only")
        return await sink.get_file(key, path)
