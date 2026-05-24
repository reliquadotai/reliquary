"""OpenCodeInstruct code-execution environment.

Loads a deterministic subset of nvidia/OpenCodeInstruct (filtered for
test stability — see scripts/build_opencodeinstruct_subset.py) and
scores miner completions by executing them against the dataset's unit
tests inside a gVisor sandbox managed by the grader subprocess.

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


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class OpenCodeInstructEnvironment:
    """nvidia/OpenCodeInstruct (deterministic subset) — Python codegen.

    Each problem is a coding instruction; the ground truth is the
    JSON-serialized list of assertion strings (unit tests). Reward
    is the fraction of assertions that pass when the miner's code is
    executed in the grader sandbox.

    The dataset is the filtered subset built by
    scripts/build_opencodeinstruct_subset.py and published to
    reliquadotai/opencodeinstruct-deterministic-subset on HF Hub.
    Override the source repo with RELIQUARY_OCI_SUBSET_REPO.
    """

    name: str = "opencodeinstruct"

    _dataset_cache: ClassVar = None
    _DEFAULT_SUBSET_REPO: ClassVar[str] = "reliquadotai/opencodeinstruct-deterministic-subset"

    def __init__(self) -> None:
        if OpenCodeInstructEnvironment._dataset_cache is None:
            import datasets as hf
            repo = os.environ.get("RELIQUARY_OCI_SUBSET_REPO", self._DEFAULT_SUBSET_REPO)
            OpenCodeInstructEnvironment._dataset_cache = hf.load_dataset(
                repo, split="train",
            )
        self._dataset = OpenCodeInstructEnvironment._dataset_cache

        from reliquary.environment.grader_client import GraderClient
        self._grader = GraderClient()

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        tests: list[str] = list(row["unit_tests_parsed"])
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return {
            "prompt": prompt,
            "ground_truth": json.dumps(tests),
            "id": problem_id,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        try:
            tests = json.loads(problem.get("ground_truth", "[]"))
            if not isinstance(tests, list):
                return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0
        code = _extract_python(completion or "")
        return self._grader.evaluate(code, tests, timeout_s=GRADER_EVAL_TIMEOUT_SECONDS)
