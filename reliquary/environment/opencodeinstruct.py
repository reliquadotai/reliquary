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


def _entry_function_name(cases: list[dict]) -> str | None:
    """The contract's graded entry function, or None when it isn't a function.

    Same source as ``_contract_instruction``: the cases carry the exact name the
    grader will call, so the extractor can pin the graded block to a definition
    rather than to a position. Method entries define no top-level ``def``, so
    they pin nothing.
    """
    for case in cases or ():
        entry = case.get("entry") or {}
        name = entry.get("name")
        if entry.get("kind") == "function" and name:
            return str(name)
    return None


def _extract_python(completion: str, entry_name: str | None = None) -> str:
    """Extract Python code from a model completion.

    Strategy: find all fenced code blocks (``` or ~~~ with optional
    'python' tag). From protocol v6 on, return the last block that *defines*
    ``entry_name``; otherwise return the last block.

    With no fence at all, v4/v5 return the raw completion and let exec reject
    obviously-non-code; v6 returns nothing, because the fenced block is the only
    answer channel. That fallback fired 762 times across 30 768 production
    rollouts without ever producing a positive reward — a rollout holding code
    always fences it — so it only ever ran ``exec`` on reasoning prose.

    Why the definition beats the position (v6): "last block wins" assumed the
    model closes with its final implementation, which held for the v2-v4 chat
    model. Under the v5 reasoning prompt the model routinely closes with a usage
    demo, an expected-output listing, or a test block — 13.1% of code rollouts
    at the v5 cutover — and grading that span scores a correct answer zero.
    Because the group-relative advantage is what trains the policy, those zeros
    read as "never open a second block", which the model generalised into "never
    reason".

    The graded span is wire-affecting: miners declare the reward they computed
    and the validator re-runs this function, rejecting a mismatch beyond 1e-6.
    Changing it before a coordinated cutover would reject honest miners, hence
    the PROTOCOL_VERSION gate — the new rule is inert on v4/v5 profiles.
    """
    if not completion:
        return ""
    from reliquary.constants import PROTOCOL_VERSION

    v6 = PROTOCOL_VERSION >= 6
    matches = _FENCE_RE.findall(completion)
    if not matches:
        # v6 has a single answer channel: what is between the fences. The raw
        # fallback fired 762 times across 30 768 production rollouts and never
        # produced a positive reward — a rollout holding code always fences it,
        # so the fallback only ever ran `exec` on reasoning prose. Those zeros
        # were deserved and stay zeros; executing prose as Python does not.
        return "" if v6 else completion
    if entry_name and v6:
        needle = f"def {entry_name}"
        for _fence, body in reversed(matches):
            if needle in body:
                return body
    return matches[-1][1]


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
        contract = _contract_instruction(cases)
        # Pin the grader's function-call contract onto the prompt. Changes prompt
        # tokens (GRAIL-bound), so a release shipping this must reach miners too.
        from reliquary.protocol.profiles import render_active_prompt

        rendered_prompt = render_active_prompt(
            self.name,
            problem=prompt,
            contract=contract,
        )
        prompt = prompt + contract if rendered_prompt is None else rendered_prompt
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
        code = _extract_python(completion or "", entry_name=_entry_function_name(cases))
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
