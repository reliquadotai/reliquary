#!/usr/bin/env python3
"""Replay one guarded production training step on archived balanced data.

The script reconstructs the exact cross-window accumulator from archive
metadata, recomputes PPO's behavior log-probabilities with the published
checkpoint, and keeps the immutable KL reference separate. Run each grid point
in a fresh process so model and optimizer state cannot leak between candidates.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any


_SYNTHETIC_CLAIM_METRIC_PREFIX = "train/pi_old_claim_"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _read_archive(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        archive = json.load(handle)
    if not isinstance(archive, dict):
        raise ValueError(f"archive must contain an object: {path}")
    return archive


def reconstruct_balanced_batch(
    archives: list[dict[str, Any]],
    *,
    legacy_checkpoint_revision: str | None = None,
    legacy_targets: dict[str, int] | None = None,
) -> tuple[list[str], list[list[dict[str, Any]]], dict[str, Any]]:
    """Rebuild the accumulator using each archive's accepted `added` counts."""
    ordered = sorted(archives, key=lambda row: int(row["window_start"]))
    legacy_rows = [
        archive
        for archive in ordered
        if not (archive.get("training_accumulator") or {})
    ]
    if legacy_rows:
        if len(legacy_rows) != len(ordered) or len(ordered) != 1:
            raise ValueError(
                "legacy full-window replay requires exactly one legacy archive "
                "per balanced batch"
            )
        if (
            legacy_checkpoint_revision is None
            or _IMMUTABLE_REVISION.fullmatch(legacy_checkpoint_revision) is None
        ):
            raise ValueError(
                "legacy full-window replay requires a full immutable checkpoint "
                "revision"
            )
        if not legacy_targets or any(count <= 0 for count in legacy_targets.values()):
            raise ValueError(
                "legacy full-window replay requires positive explicit env targets"
            )
        archive = ordered[0]
        grouped: dict[str, list[dict[str, Any]]] = {
            name: [] for name in legacy_targets
        }
        for group in archive.get("batch") or []:
            env_name = str(group.get("env_name") or "")
            if env_name not in grouped:
                raise ValueError(
                    f"legacy full window contains unexpected environment {env_name!r}"
                )
            grouped[env_name].append(group)
        counts = {name: len(groups) for name, groups in grouped.items()}
        if counts != legacy_targets:
            raise ValueError(
                "legacy full-window counts do not match explicit targets: "
                f"counts={counts} targets={legacy_targets}"
            )
        env_order = list(legacy_targets)
        return env_order, [grouped[name] for name in env_order], {
            "checkpoint_revision": legacy_checkpoint_revision,
            "targets": dict(legacy_targets),
            "counts": counts,
            "source_windows": [int(archive["window_start"])],
            "legacy_full_window": True,
        }

    retained: dict[str, list[dict[str, Any]]] = {}
    targets: dict[str, int] = {}
    checkpoint_revision: str | None = None
    source_windows: list[int] = []

    for archive in ordered:
        metadata = archive.get("training_accumulator") or {}
        snapshot = metadata.get("snapshot") or {}
        if not metadata or not snapshot:
            raise ValueError(
                f"window {archive.get('window_start')} lacks accumulator metadata"
            )
        revision = str(snapshot.get("checkpoint_revision") or "")
        if not revision:
            raise ValueError("accumulator checkpoint revision is missing")
        if metadata.get("checkpoint_reset") is not None or (
            checkpoint_revision is not None
            and checkpoint_revision != revision
        ):
            retained.clear()
        checkpoint_revision = revision

        targets = {
            str(name): int(count)
            for name, count in (snapshot.get("targets") or {}).items()
        }
        for name in targets:
            retained.setdefault(name, [])

        counts_before = metadata.get("counts_before") or {}
        for name, expected in counts_before.items():
            actual = len(retained.get(str(name), []))
            if actual != int(expected):
                raise ValueError(
                    f"missing source archive before window "
                    f"{archive.get('window_start')}: {name} expected "
                    f"{expected} retained groups, found {actual}"
                )

        batch = list(archive.get("batch") or [])
        for name, raw_count in (metadata.get("added") or {}).items():
            count = int(raw_count)
            candidates = [
                group for group in batch
                if str(group.get("env_name") or "") == str(name)
            ]
            if len(candidates) < count:
                raise ValueError(
                    f"window {archive.get('window_start')} contains "
                    f"{len(candidates)} {name} groups but metadata added {count}"
                )
            retained[str(name)].extend(candidates[:count])

        expected_counts = snapshot.get("counts") or {}
        for name, expected in expected_counts.items():
            actual = len(retained.get(str(name), []))
            if actual != int(expected):
                raise ValueError(
                    f"window {archive.get('window_start')} accumulator mismatch "
                    f"for {name}: expected {expected}, reconstructed {actual}"
                )
        source_windows.append(int(archive["window_start"]))

    if not targets or any(len(retained[name]) < target for name, target in targets.items()):
        raise ValueError("provided archives do not reconstruct a ready balanced batch")
    env_order = list(targets)
    batches = [retained[name][: targets[name]] for name in env_order]
    return env_order, batches, {
        "checkpoint_revision": checkpoint_revision,
        "targets": targets,
        "counts": {name: len(retained[name]) for name in env_order},
        "source_windows": source_windows,
        "legacy_full_window": False,
    }


def parse_legacy_targets(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    targets: dict[str, int] = {}
    for value in values:
        name, separator, raw_count = value.partition("=")
        name = name.strip()
        if not separator or not name or name in targets:
            raise ValueError(
                "--legacy-target must be unique NAME=COUNT entries"
            )
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                "--legacy-target count must be an integer"
            ) from exc
        if count <= 0:
            raise ValueError("--legacy-target count must be positive")
        targets[name] = count
    return targets


def materialize_training_batch(
    env_order: list[str],
    archived_batches: list[list[dict[str, Any]]],
    tokenizer: Any,
) -> tuple[list[list[Any]], dict[str, int]]:
    """Convert public archive rows into the minimal trusted training shape."""
    from reliquary.constants import BFT_THINKING_BUDGET
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import force_close_token_ids

    force_ids = force_close_token_ids(tokenizer)
    forced_rollouts = 0
    rollouts_total = 0
    batches: list[list[Any]] = []

    for env_name, groups in zip(env_order, archived_batches):
        materialized_groups = []
        for group in groups:
            prompt_tokens = encode_prompt(tokenizer, str(group["prompt"]))
            prompt_length = len(prompt_tokens)
            materialized_rollouts = []
            for row in group.get("rollouts") or []:
                tokens = [int(token) for token in row["tokens"]]
                if tokens[:prompt_length] != prompt_tokens:
                    raise ValueError(
                        f"prompt-token mismatch for window group "
                        f"{group.get('prompt_idx')} ({env_name})"
                    )
                completion_length = len(tokens) - prompt_length
                if completion_length <= 0:
                    raise ValueError("archived rollout has no completion tokens")

                force_start = prompt_length + BFT_THINKING_BUDGET
                force_end = force_start + len(force_ids)
                force_span = None
                if (
                    env_name == "openmathinstruct"
                    and tokens[force_start:force_end] == force_ids
                ):
                    force_span = (force_start, force_end)
                    forced_rollouts += 1

                commit = {
                    "tokens": tokens,
                    "rollout": {
                        "prompt_length": prompt_length,
                        "completion_length": completion_length,
                        # Placeholder only. The replay always passes a trusted
                        # behavior model, so these claims never enter PPO.
                        "token_logprobs": [0.0] * completion_length,
                        "forced": force_span is not None,
                        "force_span": list(force_span) if force_span else None,
                        "truncated": not bool(row.get("eos_terminated", False)),
                    },
                }
                rollout = SimpleNamespace(
                    tokens=tokens,
                    reward=float(row["reward"]),
                    commit=commit,
                    env_name=env_name,
                )
                rollout._validated_force_span = force_span
                rollout._validated_termination_path = (
                    "forced_phase2_eos"
                    if force_span is not None
                    else (
                        "phase1_eos"
                        if row.get("eos_terminated", False)
                        else "cap_truncated"
                    )
                )
                materialized_rollouts.append(rollout)
                rollouts_total += 1

            materialized_groups.append(SimpleNamespace(
                rollouts=materialized_rollouts,
                prompt_idx=int(group.get("prompt_idx", 0)),
                env_name=env_name,
                hotkey=str(group.get("hotkey") or ""),
            ))
        batches.append(materialized_groups)

    return batches, {
        "groups": sum(len(batch) for batch in batches),
        "rollouts": rollouts_total,
        "forced_rollouts": forced_rollouts,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_synthetic_claim_metrics(
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Remove claim-error metrics that archives cannot faithfully replay.

    Public window archives intentionally omit miner-claimed token
    log-probabilities. ``materialize_training_batch`` inserts zero placeholders
    because the trusted behavior model supplies PPO's actual pi_old. Comparing
    those placeholders with recomputed values would produce plausible-looking
    but meaningless claim-error telemetry.
    """
    cleaned = dict(metrics)
    ignored = sorted(
        key
        for key in cleaned
        if key.startswith(_SYNTHETIC_CLAIM_METRIC_PREFIX)
    )
    for key in ignored:
        cleaned.pop(key)
    return cleaned, ignored


def _snapshot(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        help="Source archive for one replay batch; repeat for cross-window data.",
    )
    parser.add_argument(
        "--batch-archives",
        action="append",
        help=(
            "Comma-separated source archives for one balanced batch. Repeat "
            "the option to replay sequential production batches."
        ),
    )
    parser.add_argument(
        "--checkpoint-repo",
        default="ReliquaryForge/qwen3.5-2b-reliquary-v2",
    )
    parser.add_argument("--behavior-revision", required=True)
    parser.add_argument(
        "--legacy-checkpoint-revision",
        help=(
            "Immutable behavior SHA for archives created before accumulator "
            "metadata existed. Requires one or more --legacy-target entries."
        ),
    )
    parser.add_argument(
        "--legacy-target",
        action="append",
        help="Exact NAME=COUNT target for a legacy full-window archive.",
    )
    parser.add_argument("--base-repo", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--base-revision",
        default="15852e8c16360a2fea060d615a32b45270f8a8fc",
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--kl-beta", type=float, required=True)
    parser.add_argument("--shape-penalty", type=float, default=0.5)
    parser.add_argument("--shape-len-frac", type=float, default=0.5)
    parser.add_argument("--grad-threshold", type=float, default=100.0)
    parser.add_argument(
        "--attention-implementation", default="flash_attention_2"
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-model", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    os.environ["RELIQUARY_LEARNING_RATE"] = str(args.learning_rate)
    os.environ["RELIQUARY_KL_BETA"] = str(args.kl_beta)
    os.environ["RELIQUARY_SHAPE_PENALTY"] = str(args.shape_penalty)
    os.environ["RELIQUARY_SHAPE_LEN_FRAC"] = str(args.shape_len_frac)
    os.environ["RELIQUARY_GRAD_NORM_SKIP_THRESHOLD"] = str(
        args.grad_threshold
    )
    os.environ["RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY"] = "true"
    os.environ["GRAIL_ATTN_IMPL"] = args.attention_implementation

    import torch

    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
    from reliquary.validator import telemetry
    from reliquary.validator.training import (
        TrainingStepSkipped,
        reset_training_state,
        train_step,
    )

    if args.archive and args.batch_archives:
        raise ValueError("use --archive or --batch-archives, not both")
    if args.batch_archives:
        archive_sets = [
            [Path(value).resolve() for value in raw.split(",") if value]
            for raw in args.batch_archives
        ]
    elif args.archive:
        archive_sets = [[path.resolve() for path in args.archive]]
    else:
        raise ValueError("at least one archive batch is required")
    legacy_targets = parse_legacy_targets(args.legacy_target)
    if bool(args.legacy_checkpoint_revision) != bool(legacy_targets):
        raise ValueError(
            "--legacy-checkpoint-revision and --legacy-target must be used together"
        )

    reconstructed = []
    for archive_paths in archive_sets:
        archives = [_read_archive(path) for path in archive_paths]
        env_order, archived_batches, accumulator = reconstruct_balanced_batch(
            archives,
            legacy_checkpoint_revision=args.legacy_checkpoint_revision,
            legacy_targets=legacy_targets,
        )
        if accumulator["checkpoint_revision"] != args.behavior_revision:
            raise ValueError(
                "archive behavior revision does not match --behavior-revision: "
                f"{accumulator['checkpoint_revision']} != "
                f"{args.behavior_revision}"
            )
        reconstructed.append(
            (archive_paths, env_order, archived_batches, accumulator)
        )

    base_path = _snapshot(args.base_repo, args.base_revision)
    behavior_path = _snapshot(
        args.checkpoint_repo, args.behavior_revision
    )
    tokenizer = load_tokenizer(base_path)
    replay_batches = []
    for archive_paths, env_order, archived_batches, accumulator in reconstructed:
        batches, batch_summary = materialize_training_batch(
            env_order, archived_batches, tokenizer
        )
        replay_batches.append({
            "archive_paths": archive_paths,
            "accumulator": accumulator,
            "batches": batches,
            "batch_summary": batch_summary,
        })

    def load_model(path: str):
        return load_text_generation_model(
            path,
            dtype=torch.bfloat16,
            attn_implementation=args.attention_implementation,
        ).to("cuda:0").eval()

    behavior_model = load_model(behavior_path)
    for parameter in behavior_model.parameters():
        parameter.requires_grad = False
    train_model = copy.deepcopy(behavior_model)
    train_model.train()
    for parameter in train_model.parameters():
        parameter.requires_grad = True
    try:
        train_model.gradient_checkpointing_enable()
    except (AttributeError, NotImplementedError):
        pass
    ref_model = load_model(base_path)
    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    original_log = telemetry.log_training_step
    reset_training_state()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    status = "stepped"
    skip_reason = None
    step_results = []
    for step_offset, replay_batch in enumerate(replay_batches):
        captured: dict[str, Any] = {}

        def capture(metrics: dict, step: int | None) -> None:
            captured.update(metrics)
            captured["_step"] = step

        telemetry.log_training_step = capture
        step_started = time.perf_counter()
        telemetry_step = (
            args.window_index + step_offset
            if args.window_index
            else max(replay_batch["accumulator"]["source_windows"])
        )
        try:
            train_step(
                train_model,
                replay_batch["batches"],
                ref_model=ref_model,
                behavior_model=behavior_model,
                window_index=telemetry_step,
            )
        except TrainingStepSkipped as exc:
            status = "skipped"
            skip_reason = exc.reason
        finally:
            telemetry.log_training_step = original_log
        captured, ignored_synthetic_metrics = strip_synthetic_claim_metrics(
            captured
        )
        step_results.append({
            "step_index": step_offset,
            "window_index": telemetry_step,
            "status": status,
            "skip_reason": skip_reason,
            "elapsed_seconds": time.perf_counter() - step_started,
            "archives": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in replay_batch["archive_paths"]
            ],
            "accumulator": replay_batch["accumulator"],
            "batch": replay_batch["batch_summary"],
            "metrics": captured,
            "ignored_synthetic_metrics": ignored_synthetic_metrics,
        })
        if status != "stepped":
            break
    elapsed = time.perf_counter() - started

    if status == "stepped" and args.save_model is not None:
        args.save_model.mkdir(parents=True, exist_ok=True)
        train_model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    result = {
        "schema_version": 1,
        "status": status,
        "skip_reason": skip_reason,
        "checkpoint_repo": args.checkpoint_repo,
        "behavior_revision": args.behavior_revision,
        "base_repo": args.base_repo,
        "base_revision": args.base_revision,
        "learning_rate": args.learning_rate,
        "kl_beta": args.kl_beta,
        "shape_penalty": args.shape_penalty,
        "shape_len_frac": args.shape_len_frac,
        "grad_threshold": args.grad_threshold,
        "attention_implementation": args.attention_implementation,
        "requested_steps": len(replay_batches),
        "completed_steps": sum(
            step["status"] == "stepped" for step in step_results
        ),
        "steps": step_results,
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "archive_limitations": {
            "miner_claimed_token_logprobs_available": False,
            "pi_old_claim_error_metrics_valid": False,
            "reason": (
                "public archives omit miner-claimed token log-probabilities; "
                "PPO uses behavior-model recomputation during replay"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "stepped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
