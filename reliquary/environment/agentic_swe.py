"""What an agentic SWE corpus job pins, and the prompt source behind it.

`renderers`, `verifiers` and `reliquary_swe` are optional: imported inside
functions only. The constants below are copies of verifiers b2e4e81's bash
harness (edit on, search off); `tests/unit/test_agentic_swe_renderer.py`
holds them equal to the pinned package when it is installed.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import subprocess
from collections.abc import Sequence

from reliquary.environment.agentic.types import EpisodeTask

SUPPORTED_RENDERER = "renderers:qwen38@0.1.11"
RENDERERS_VERSION = "0.1.11"
SUPPORTED_VERIFIERS = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"

# harnesses/bash/harness.py: BASH_SYSTEM_PROMPT + " " + EDIT_SYSTEM_PROMPT.
BASH_SYSTEM_PROMPT = (
    "You are a coding agent. You have access to a bash tool for running shell commands. "
    "You also have an edit tool for single-occurrence string replacement in a file."
)
# harnesses/bash/program.py: BASH_TOOL, EDIT_TOOL, as the train client renders them.
BASH_HARNESS_TOOLS: tuple[dict, ...] = (
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string",
                                                  "description": "The bash command to run."}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace a unique string in a file. old_str must appear exactly once in the file.",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string",
                                    "description": "File path (relative to cwd or absolute)."},
                           "old_str": {"type": "string",
                                       "description": "Exact string to find (must appear exactly once)."},
                           "new_str": {"type": "string", "description": "Replacement string."}},
                       "required": ["path", "old_str", "new_str"]}}},
)


def _dist_commit(name: str) -> str | None:
    """The commit a distribution was installed from: its VCS pin, or the HEAD
    of the checkout an editable install points at. None when unknowable."""
    try:
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    info = json.loads(raw)
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url") or ""
    if url.startswith("file://"):
        checkout = url[len("file://"):]
        git = ["git", "-c", "safe.directory=*", "-C", checkout]
        try:
            head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10)
            dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True,
                                   timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return None
        # An editable checkout with local changes is not the pinned code,
        # whatever its HEAD says: report nothing rather than the commit.
        if head.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip():
            return head.stdout.strip()
    return None


def installed_env_commit() -> str | None:
    return _dist_commit("reliquary-swe")


def installed_verifiers_commit() -> str | None:
    return _dist_commit("verifiers")


def _renderers_version() -> str | None:
    try:
        return importlib.metadata.version("renderers")
    except importlib.metadata.PackageNotFoundError:
        return None


def episode_support_refusal(episode, *, need_verifiers: bool) -> str | None:
    """Why this binary cannot serve `episode`, or None. ``need_verifiers`` for
    the processes that run episodes (miner, grade executor)."""
    if episode.renderer != SUPPORTED_RENDERER:
        return f"episode.renderer {episode.renderer!r} is not {SUPPORTED_RENDERER!r}"
    if episode.verifiers != SUPPORTED_VERIFIERS:
        return f"episode.verifiers {episode.verifiers} is not the supported {SUPPORTED_VERIFIERS}"
    if _renderers_version() != RENDERERS_VERSION:
        return f"renderers {RENDERERS_VERSION} is not installed (found {_renderers_version()})"
    installed = installed_env_commit()
    if installed != episode.env.version:
        return (f"reliquary-swe is installed at {installed}, the job pins "
                f"{episode.env.version}")
    if need_verifiers and installed_verifiers_commit() != SUPPORTED_VERIFIERS:
        return (f"verifiers is installed at {installed_verifiers_commit()}, "
                f"the job pins {SUPPORTED_VERIFIERS}")
    return None


class SweSource:
    """SWE-smith tasks by source index: the instance id and the user prompt,
    nothing else (a full task set costs over 5 GB of memory as tasks)."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[tuple[str, str]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def instance_id(self, index: int) -> str:
        return self._rows[index][0]

    def prompt(self, index: int) -> str:
        return self._rows[index][1]

    def task_for(self, index: int) -> EpisodeTask:
        instance_id, prompt = self._rows[index]
        return EpisodeTask(id=instance_id, prompt=prompt, tools=())


@functools.lru_cache(maxsize=4)
def load_swe_source(num_images: int) -> SweSource:
    """The task set `reliquary-swe` builds for `num_images` (split train), in
    its own deterministic order: index i here is task i on every machine."""
    from reliquary_swe import corpus
    from reliquary_swe.taskset import PROMPT

    rows = corpus.load_swesmith_rows(num_images)
    return SweSource([(row.instance_id,
                       PROMPT.format(workdir=row.workdir, problem_statement=row.problem_statement))
                      for row in rows])
