#!/usr/bin/env python3
"""Review an eval holdout for exact and near-duplicate training overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from reliquary.eval_dashboard.config import canonical_json_bytes, sha256_file
from reliquary.eval_dashboard.holdout import task_ids_sha256
from reliquary.eval_dashboard.models import (
    CodeTask,
    ContaminationReview,
    MathTask,
    TrainingArtifact,
    TrainingSource,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def _normalize_prompt(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _ngrams(value: str, width: int = 5) -> set[str]:
    tokens = _normalize_prompt(value).split()
    if len(tokens) < width:
        return set(tokens)
    return {
        " ".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _iter_jsonl(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_n, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if not isinstance(value, str):
                raise ValueError(f"{path}:{line_n} has no string field {field!r}")
            yield value


def _iter_parquet(path: Path, field: str) -> Iterator[str]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=[field], batch_size=4096):
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str):
                raise ValueError(f"{path} has a non-string value in field {field!r}")
            yield value


def _iter_training(path: Path, field: str) -> Iterator[str]:
    if path.suffix == ".parquet":
        yield from _iter_parquet(path, field)
    else:
        yield from _iter_jsonl(path, field)


def _load_holdout(path: Path, domain: str):
    model = MathTask if domain == "math" else CodeTask
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_n, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid holdout row {path}:{line_n}: {exc}") from exc
    if not tasks:
        raise ValueError("holdout is empty")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("math", "code"), required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument(
        "--training-source",
        action="append",
        nargs=4,
        required=True,
        metavar=("REPO_ID", "REVISION", "PROMPT_FIELD", "FILE"),
        help=(
            "pinned source shard; repeat for every shard and every math/code "
            "training source"
        ),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.near_threshold <= 1.0:
        raise ValueError("near-threshold must be in (0, 1]")
    sources = [
        (repo_id, revision, prompt_field, Path(path))
        for repo_id, revision, prompt_field, path in args.training_source
    ]
    source_paths = [path.resolve() for _, _, _, path in sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("the same training shard was supplied more than once")
    for _, _, _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = _load_holdout(args.holdout, args.domain)
    holdout_ngrams = [_ngrams(task.prompt) for task in tasks]
    holdout_exact: dict[str, set[int]] = defaultdict(set)
    for index, task in enumerate(tasks):
        digest = hashlib.sha256(
            _normalize_prompt(task.prompt).encode("utf-8")
        ).hexdigest()
        holdout_exact[digest].add(index)
    feature_index: dict[str, set[int]] = defaultdict(set)
    for index, features in enumerate(holdout_ngrams):
        for feature in features:
            feature_index[feature].add(index)

    exact_matches: set[int] = set()
    near_matches: set[int] = set()
    scanned = 0
    source_artifacts: dict[tuple[str, str, str], list[TrainingArtifact]] = defaultdict(
        list
    )
    for repo_id, revision, prompt_field, path in sources:
        shard_count = 0
        for prompt in _iter_training(path, prompt_field):
            shard_count += 1
            scanned += 1
            normalized = _normalize_prompt(prompt)
            exact_indices = holdout_exact.get(
                hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            )
            if exact_indices:
                exact_matches.update(exact_indices)
                continue
            features = _ngrams(prompt)
            candidates: set[int] = set()
            for feature in features:
                candidates.update(feature_index.get(feature, set()))
            for index in candidates:
                if _jaccard(features, holdout_ngrams[index]) >= args.near_threshold:
                    near_matches.add(index)
        if shard_count == 0:
            raise ValueError(f"training source shard is empty: {path}")
        source_artifacts[(repo_id, revision, prompt_field)].append(
            TrainingArtifact(sha256=sha256_file(path), n_prompts=shard_count)
        )

    decision = "approved" if not exact_matches and not near_matches else "rejected"
    review = ContaminationReview(
        domain=args.domain,
        holdout_sha256=sha256_file(args.holdout),
        task_ids_sha256=task_ids_sha256(task.task_id for task in tasks),
        reviewed_at=args.reviewed_at,
        reviewer=args.reviewer,
        method=(
            "normalized exact SHA-256 plus inverted token-shingle candidate "
            f"search and token 5-gram Jaccard >= {args.near_threshold:.3f}; "
            f"scanned {scanned} training prompts"
        ),
        training_sources=[
            TrainingSource(
                repo_id=repo_id,
                revision=revision,
                prompt_field=prompt_field,
                artifacts=sorted(
                    artifacts,
                    key=lambda artifact: (artifact.sha256, artifact.n_prompts),
                ),
            )
            for (repo_id, revision, prompt_field), artifacts in sorted(
                source_artifacts.items()
            )
        ],
        exact_overlap_count=len(exact_matches),
        near_duplicate_count=len(near_matches),
        decision=decision,
        notes=(
            f"exact_task_ids={[tasks[index].task_id for index in sorted(exact_matches)]}; "
            f"near_task_ids={[tasks[index].task_id for index in sorted(near_matches)]}"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(review.model_dump(mode="json")))
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "decision": decision,
                "holdout_sha256": review.holdout_sha256,
                "task_ids_sha256": review.task_ids_sha256,
                "review_sha256": sha256_file(args.output),
                "n_tasks": len(tasks),
                "n_training_prompts_scanned": scanned,
                "exact_overlap_count": review.exact_overlap_count,
                "near_duplicate_count": review.near_duplicate_count,
            },
            sort_keys=True,
        )
    )
    return 0 if decision == "approved" else 2


if __name__ == "__main__":
    sys.exit(main())
