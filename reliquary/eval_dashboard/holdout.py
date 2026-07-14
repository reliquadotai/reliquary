"""Load and verify sealed math/code holdout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from reliquary.eval_dashboard.config import canonical_json_bytes, sha256_file
from reliquary.eval_dashboard.models import (
    CodeTask,
    ContaminationReview,
    HoldoutSpec,
    MathTask,
)


TaskT = TypeVar("TaskT", MathTask, CodeTask)


def task_ids_sha256(task_ids: Iterable[str]) -> str:
    """Hash the ordered task-id list without ambiguous concatenation."""
    return hashlib.sha256(canonical_json_bytes(list(task_ids))).hexdigest()


def _load_jsonl(path: str | Path, model: type[TaskT]) -> list[TaskT]:
    tasks: list[TaskT] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_n, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                tasks.append(model.model_validate(raw))
            except Exception as exc:
                raise ValueError(f"invalid holdout row {path}:{line_n}: {exc}") from exc
    return tasks


def _validate_review(
    review_path: str | Path,
    *,
    spec: HoldoutSpec,
    actual_holdout_sha256: str,
    actual_task_ids_sha256: str,
) -> ContaminationReview:
    review_sha = sha256_file(review_path)
    if review_sha != spec.contamination_review_sha256:
        raise ValueError(
            "contamination review hash mismatch: "
            f"expected {spec.contamination_review_sha256}, got {review_sha}"
        )
    with Path(review_path).open("rb") as handle:
        review = ContaminationReview.model_validate_json(handle.read())
    if review.domain != spec.domain:
        raise ValueError("contamination review domain does not match holdout")
    if review.holdout_sha256 != actual_holdout_sha256:
        raise ValueError("contamination review refers to a different holdout artifact")
    if review.task_ids_sha256 != actual_task_ids_sha256:
        raise ValueError("contamination review refers to a different task-id list")
    if review.decision != "approved":
        raise ValueError("contamination review is not approved")
    if review.exact_overlap_count != 0 or review.near_duplicate_count != 0:
        raise ValueError("approved holdout still contains training overlap")
    return review


def load_locked_holdout(
    holdout_path: str | Path,
    review_path: str | Path,
    spec: HoldoutSpec,
) -> tuple[list[MathTask] | list[CodeTask], ContaminationReview]:
    """Load a holdout only when every immutable lock matches.

    The task file itself is never published by the dashboard producer.  It may
    therefore contain private code cases, while the public artifacts expose
    only task ids, prompt hashes, rewards, and completion hashes.
    """
    actual_sha = sha256_file(holdout_path)
    if actual_sha != spec.artifact_sha256:
        raise ValueError(
            f"{spec.domain} holdout hash mismatch: expected "
            f"{spec.artifact_sha256}, got {actual_sha}"
        )

    model: type[BaseModel]
    if spec.domain == "math":
        model = MathTask
    else:
        model = CodeTask
    tasks = _load_jsonl(holdout_path, model)  # type: ignore[arg-type]
    if len(tasks) != spec.n_prompts:
        raise ValueError(
            f"{spec.domain} holdout expected {spec.n_prompts} prompts, got {len(tasks)}"
        )

    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{spec.domain} holdout contains duplicate task ids")
    prompt_hashes = [
        hashlib.sha256(
            " ".join(task.prompt.casefold().split()).encode("utf-8")
        ).digest()
        for task in tasks
    ]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError(f"{spec.domain} holdout contains duplicate prompts")
    ids_sha = task_ids_sha256(ids)
    if ids_sha != spec.task_ids_sha256:
        raise ValueError(
            f"{spec.domain} task-id hash mismatch: expected "
            f"{spec.task_ids_sha256}, got {ids_sha}"
        )

    review = _validate_review(
        review_path,
        spec=spec,
        actual_holdout_sha256=actual_sha,
        actual_task_ids_sha256=ids_sha,
    )
    return tasks, review


def write_canonical_jsonl(path: str | Path, tasks: Iterable[BaseModel]) -> None:
    """Write a deterministic holdout artifact for an operator lock step."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for task in tasks:
            handle.write(canonical_json_bytes(task.model_dump(mode="json")))
            handle.write(b"\n")
    temporary.replace(target)
