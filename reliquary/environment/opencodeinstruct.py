"""OpenCodeInstruct code-execution environment.

Loads the reproducible curated subset of nvidia/OpenCodeInstruct (per-test
filtered + structured — see scripts/build_opencode_curated.py), lazily via a
VirtualParquetDataset so only the row-groups a window touches are fetched, and
scores miner completions by calling structured cases inside a gVisor sandbox
managed by the grader subprocess.

The class itself is a thin wrapper: it knows nothing about sandboxes.
All execution happens via reliquary.environment.grader_client, which
talks to the grader server over a Unix socket. This keeps the class
testable without the sandbox infrastructure (see tests/unit/).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import ClassVar

from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Code extraction from model completions
# ---------------------------------------------------------------------------

# Match fenced code blocks: ``` or ~~~ optionally followed by a language tag.
# Greedy match on the closing fence so the last block wins (model's final
# answer wins over earlier drafts).
_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)


def _extract_python(completion: str) -> str:
    """Extract Python code from a model completion.

    Strategy: find all fenced code blocks (``` or ~~~ with optional
    'python' tag), return the last one's contents. Falls back to the
    raw completion string if no fence is present — exec will reject
    obviously-non-code, scoring zero.
    """
    if not completion:
        return ""
    matches = _FENCE_RE.findall(completion)
    if matches:
        return matches[-1][1]
    return completion


def _load_dataset(repo: str, revision: str):
    """Lazy virtual-parquet view of the curated dataset.

    A ``save_to_disk`` directory path is loaded eagerly (offline / fixtures);
    a ``owner/name`` repo id is wrapped in a ``VirtualParquetDataset`` so only
    the row-groups a window touches are fetched — no multi-GB bulk download.
    """
    path = Path(repo).expanduser()
    if path.exists() and (path / "dataset_info.json").exists():
        import datasets as hf
        return hf.load_from_disk(str(path))
    from reliquary.environment.virtual_parquet import VirtualParquetDataset
    return VirtualParquetDataset(repo, revision, columns=["input", "structured_cases"])


def _contract_instruction(cases: list[dict]) -> str:
    """The grader calls a named function and checks its RETURN value, but the raw
    prompts are stdin/stdout-framed and rarely name the function. Append the exact
    contract (name + "return, don't print") derived from the cases so the model
    writes a callable returning function instead of guessing. Empty for non-
    function entries (nothing to pin)."""
    for case in cases:
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            nargs = len(case.get("args") or [])
            args = "argument" if nargs == 1 else "arguments"
            return (
                f"\n\nWrite your solution as a Python function named `{name}` that "
                f"takes {nargs} {args} and returns the result; do not read from "
                f"stdin or print."
            )
    return ""


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class OpenCodeInstructEnvironment:
    """nvidia/OpenCodeInstruct (deterministic subset) — Python codegen.

    Each problem is a coding instruction; the public ground truth is an
    opaque case-set id. The actual structured cases stay in this
    environment instance and are scored by the trusted grader server.

    The dataset is the reproducible curated subset built by
    scripts/build_opencode_curated.py (per-test filtered + structured) and
    published to R0mAI/opencodeinstruct-curated. Both validator and miner load
    the same pinned revision lazily (only the touched row-groups), so tests are
    no longer hidden — the reward grades honest model output by value, not
    secrecy. Override with RELIQUARY_OCI_REPO / RELIQUARY_OCI_REVISION.
    """

    name: str = "opencodeinstruct"
    validator_authoritative_reward: ClassVar[bool] = True

    _dataset_cache: ClassVar = {}
    _CURATED_REPO: ClassVar[str] = "R0mAI/opencodeinstruct-curated"
    _CURATED_REVISION: ClassVar[str] = "d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7"

    def __init__(self) -> None:
        repo = os.environ.get("RELIQUARY_OCI_REPO", self._CURATED_REPO)
        revision = os.environ.get("RELIQUARY_OCI_REVISION", self._CURATED_REVISION)
        cache = OpenCodeInstructEnvironment._dataset_cache
        if isinstance(cache, dict):
            key = (repo, revision)
            if key not in cache:
                cache[key] = _load_dataset(repo, revision)
            self._dataset = cache[key]
        else:
            # Tests may monkeypatch _dataset_cache directly with a fake dataset.
            self._dataset = cache

        from reliquary.environment.grader_client import GraderClient
        self._grader = GraderClient()
        self._cases_by_id: dict[str, list[dict]] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def source_health(self) -> dict:
        snapshot = getattr(self._dataset, "source_health", None)
        if callable(snapshot):
            return dict(snapshot())
        return {"status": "unreported"}

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        cases = self._row_cases(row)
        # Pin the grader's function-call contract onto the prompt. Changes prompt
        # tokens (GRAIL-bound), so a release shipping this must reach miners too.
        prompt = prompt + _contract_instruction(cases)
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        case_id = hashlib.sha256(
            (problem_id + json.dumps(cases, sort_keys=True, separators=(",", ":"))).encode()
        ).hexdigest()[:16]
        self._cases_by_id[case_id] = cases
        return {
            "prompt": prompt,
            "ground_truth": case_id,
            "id": problem_id,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        case_id = problem.get("ground_truth", "")
        if not isinstance(case_id, str):
            return 0.0
        cases = self._cases_by_id.get(case_id)
        if not cases:
            return 0.0
        code = _extract_python(completion or "")
        return self._grader.evaluate_cases(code, cases, timeout_s=GRADER_EVAL_TIMEOUT_SECONDS)

    def admission_reward_cases(self, problem: dict) -> list[dict]:
        """Return an isolated copy of the cases for a materialized problem."""
        case_id = problem.get("ground_truth", "")
        if not isinstance(case_id, str):
            return []
        return [dict(case) for case in self._cases_by_id.get(case_id, ())]

    @staticmethod
    def _row_cases(row) -> list[dict]:
        raw = row.get("structured_cases", [])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        return [dict(c) for c in raw if isinstance(c, dict)]
