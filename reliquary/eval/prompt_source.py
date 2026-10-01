"""An eval set as a corpus job's prompt source.

A job generating an evaluation names its prompts ``eval-set:<set_id>:<n>:<sha256>``:
the first ``n`` problems of the frozen set, whose ``prompts.jsonl`` lines hash to
``sha256``. The prompt text is the set's rendered user turn, wrapped by the
model's chat template like any chat-template job. No grading data is involved:
the set's ``grading.jsonl`` never leaves the admin host.

Row ``i`` of the job is line ``i`` of the set, so ``prompt_index`` maps to the
set's ``problem_id`` through the set's recorded order.

The lines come from wherever this process can read them, checked against the
sha256 the source names: a directory (``RELIQUARY_EVAL_SETS_DIR``), the subnet
bucket (validators, the admin host), or bytes handed over by the control that
serves the job (miners, ``GET /corpus/jobs/{job_id}/eval-prompts``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

EVAL_SOURCE_PREFIX = "eval-set:"
# Job ids of evaluation jobs are `<admin task prefix>eval-`: the eval control
# serves them, the corpus control never.
TASK_PREFIX_ENV = "RELIQUARY_ADMIN_TASK_PREFIX"
DEFAULT_TASK_PREFIX = "order-"


def eval_job_prefix(task_prefix: str | None = None) -> str:
    """``${RELIQUARY_ADMIN_TASK_PREFIX}eval-`` (``order-eval-`` by default)."""
    base = task_prefix if task_prefix is not None else (
        os.environ.get(TASK_PREFIX_ENV, "").strip() or DEFAULT_TASK_PREFIX)
    return f"{base}eval-"


def is_eval_job_id(job_id, task_prefix: str | None = None) -> bool:
    return str(job_id or "").startswith(eval_job_prefix(task_prefix))
SETS_DIR_ENV = "RELIQUARY_EVAL_SETS_DIR"
_SOURCE_RE = re.compile(
    r"\Aeval-set:([a-z0-9][a-z0-9_-]{0,127}):([1-9][0-9]{0,8}):([0-9a-f]{64})\Z")

logger = logging.getLogger(__name__)
_lock = threading.Lock()
# (set_id, n, sha256) -> (the rows, their bytes), once checked.
_loaded: dict[tuple[str, int, str], tuple[tuple[dict, ...], bytes]] = {}


@dataclass(frozen=True)
class EvalSource:
    set_id: str
    count: int
    sha256: str

    @property
    def name(self) -> str:
        return f"{EVAL_SOURCE_PREFIX}{self.set_id}:{self.count}:{self.sha256}"


def is_eval_source(prompt_source: str) -> bool:
    return isinstance(prompt_source, str) and prompt_source.startswith(EVAL_SOURCE_PREFIX)


def parse_eval_source(prompt_source: str) -> EvalSource:
    match = _SOURCE_RE.match(prompt_source or "")
    if match is None:
        raise ValueError(f"{prompt_source!r} is not an eval-set prompt source")
    return EvalSource(match.group(1), int(match.group(2)), match.group(3))


def head_lines(body: bytes, count: int) -> bytes:
    """The first ``count`` lines of a JSONL body, each with its newline."""
    lines = body.splitlines(keepends=True)
    if len(lines) < count:
        raise ValueError(f"the set holds {len(lines)} problems, fewer than {count}")
    return b"".join(lines[:count])


def eval_source_for(set_id: str, prompts_body: bytes, count: int) -> EvalSource:
    """The source naming the first ``count`` problems of a set's prompts."""
    head = head_lines(prompts_body, count)
    return EvalSource(set_id, count, hashlib.sha256(head).hexdigest())


def register_eval_prompts(source: EvalSource, body: bytes) -> tuple[dict, ...]:
    """Check ``body`` (the job's lines, or the whole set's) against the source
    and keep the rows."""
    head = head_lines(body, source.count)
    if hashlib.sha256(head).hexdigest() != source.sha256:
        raise ValueError(f"the prompts of {source.set_id!r} do not hash to the job's sha256")
    rows = tuple(json.loads(line) for line in head.splitlines())
    for row in rows:
        messages = row.get("messages")
        # Exactly one user turn: anything else would be dropped silently.
        if not (isinstance(messages, list) and len(messages) == 1
                and messages[0].get("role") == "user"
                and isinstance(messages[0].get("content"), str)):
            raise ValueError(f"problem {row.get('problem_id')!r} is not one user turn")
    with _lock:
        _loaded[(source.set_id, source.count, source.sha256)] = (rows, head)
    return rows


def forget_eval_prompts(source: EvalSource) -> None:
    """A job unwired: its rows are not kept."""
    with _lock:
        _loaded.pop((source.set_id, source.count, source.sha256), None)


def job_prompt_lines(source: EvalSource) -> bytes:
    """The job's own lines, byte for byte, as a miner fetches them."""
    load_eval_rows(source)
    with _lock:
        return _loaded[(source.set_id, source.count, source.sha256)][1]


def _from_directory(source: EvalSource) -> bytes | None:
    root = os.environ.get(SETS_DIR_ENV, "").strip()
    if not root:
        return None
    path = Path(root) / source.set_id / "prompts.jsonl"
    return path.read_bytes() if path.exists() else None


def _from_subnet_bucket(source: EvalSource) -> bytes | None:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from reliquary.eval.storage import SubnetEvalStore, subnet_key

    key = subnet_key(source.set_id, "prompts.jsonl")

    async def read():
        return await SubnetEvalStore().get_bytes(key)

    # Called from sync code that may itself run inside an event loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        body = pool.submit(asyncio.run, read()).result()
    if body is None:
        logger.warning("eval set prompts not in the subnet bucket: %s", key)
    return body


# Tried in order; tests replace this list.
FETCHERS: list[Callable[[EvalSource], bytes | None]] = [_from_directory, _from_subnet_bucket]


def load_eval_rows(source: EvalSource) -> tuple[dict, ...]:
    with _lock:
        loaded = _loaded.get((source.set_id, source.count, source.sha256))
    if loaded is not None:
        return loaded[0]
    for fetch in FETCHERS:
        body = fetch(source)
        if body is not None:
            return register_eval_prompts(source, body)
    raise ValueError(f"the prompts of eval set {source.set_id!r} are not readable here")


class EvalSetEnvironment:
    """The job's rows, in the single-turn shape a corpus prompt job reads."""

    def __init__(self, source: EvalSource, rows: tuple[dict, ...]) -> None:
        self.name = source.name
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def get_problem(self, index: int) -> dict:
        row = self._rows[int(index)]
        return {"prompt": row["messages"][-1]["content"], "id": row["problem_id"],
                "problem_id": row["problem_id"]}

    def problem_id(self, index: int) -> str:
        return self._rows[int(index)]["problem_id"]


class EvalSetSpec:
    """What ``resolve_prompt_source`` hands back for an eval-set source."""

    interaction_mode = "single_turn"

    def __init__(self, prompt_source: str) -> None:
        self.source = parse_eval_source(prompt_source)
        self.name = prompt_source

    def create(self) -> EvalSetEnvironment:
        return EvalSetEnvironment(self.source, load_eval_rows(self.source))


__all__ = [
    "eval_job_prefix",
    "is_eval_job_id",
    "EVAL_SOURCE_PREFIX",
    "EvalSetEnvironment",
    "EvalSetSpec",
    "EvalSource",
    "FETCHERS",
    "eval_source_for",
    "forget_eval_prompts",
    "head_lines",
    "is_eval_source",
    "job_prompt_lines",
    "load_eval_rows",
    "parse_eval_source",
    "register_eval_prompts",
]
