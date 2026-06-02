"""OpenCodeInstruct code-execution environment.

Loads a deterministic subset of nvidia/OpenCodeInstruct (filtered for
test stability — see scripts/build_opencodeinstruct_subset.py) and
scores miner completions by calling hidden structured cases inside a
gVisor sandbox managed by the grader subprocess.

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


def _load_subset_dataset(repo: str, hf_module=None):
    """Load either a HF dataset repo or a local ``save_to_disk`` dataset."""
    if hf_module is None:
        import datasets as hf_module
    path = Path(repo).expanduser()
    if path.exists() and (path / "dataset_info.json").exists() and (path / "state.json").exists():
        return hf_module.load_from_disk(str(path))
    return hf_module.load_dataset(repo, split="train")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _has_structured_cases_column(dataset) -> bool:
    columns = getattr(dataset, "column_names", None)
    if columns is not None:
        return "structured_cases" in columns
    if len(dataset) == 0:
        return False
    row = dataset[0]
    return hasattr(row, "get") and row.get("structured_cases") is not None


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class OpenCodeInstructEnvironment:
    """nvidia/OpenCodeInstruct (deterministic subset) — Python codegen.

    Each problem is a coding instruction; the public ground truth is an
    opaque case-set id. The actual structured cases stay in this
    environment instance and are scored by the trusted grader server.

    The dataset is the filtered subset built by
    scripts/build_opencodeinstruct_subset.py and published to
    reliquadotai/opencodeinstruct-structured-subset on HF Hub.
    Override the source repo with RELIQUARY_OCI_SUBSET_REPO.
    """

    name: str = "opencodeinstruct"
    validator_authoritative_reward: ClassVar[bool] = True

    _dataset_cache: ClassVar = None
    _DEFAULT_SUBSET_REPO: ClassVar[str] = "reliquadotai/opencodeinstruct-structured-subset"

    def __init__(self) -> None:
        if OpenCodeInstructEnvironment._dataset_cache is None:
            repo = os.environ.get("RELIQUARY_OCI_SUBSET_REPO", self._DEFAULT_SUBSET_REPO)
            OpenCodeInstructEnvironment._dataset_cache = _load_subset_dataset(repo)
        self._dataset = OpenCodeInstructEnvironment._dataset_cache
        self._prompt_only = _env_flag("RELIQUARY_OCI_PROMPT_ONLY")
        if not self._prompt_only and not _has_structured_cases_column(self._dataset):
            raise RuntimeError(
                "OpenCodeInstruct validator dataset must include structured_cases. "
                "Use RELIQUARY_OCI_PROMPT_ONLY=1 only for miner prompt-only mirrors."
            )

        from reliquary.environment.grader_client import GraderClient
        self._grader = GraderClient()
        self._cases_by_id: dict[str, list[dict]] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        cases = self._row_cases(row)
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
