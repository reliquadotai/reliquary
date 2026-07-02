#!/usr/bin/env python3
"""Publish a base-model reset as the next Reliquary HF checkpoint.

Use this after an incident when operators want miners to move forward to a
fresh base-model checkpoint instead of resuming the latest trained checkpoint.

The script is intentionally append-only: it creates ``checkpoint N+1`` in the
configured HF repo and prints the ``RELIQUARY_RESUME_FROM=sha:<commit>`` line
to pin the validator restart.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
CHECKPOINT_TITLE = re.compile(r"^checkpoint\s+(\d+)\s*$", re.IGNORECASE)


def _env_or_arg(value: str | None, env_name: str, label: str) -> str:
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise SystemExit(f"{label} is required: pass the flag or set {env_name}")
    return resolved


def _latest_checkpoint_n(api, repo_id: str) -> int:
    latest = 0
    for commit in api.list_repo_commits(repo_id=repo_id):
        match = CHECKPOINT_TITLE.match(commit.title or "")
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload RELIQUARY_CHECKPOINT/base model as the next HF "
            "Reliquary checkpoint and print the resume env var."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HF repo to publish into; defaults to RELIQUARY_HF_REPO_ID.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "HF repo/path to reset to; defaults to RELIQUARY_CHECKPOINT or "
            f"{DEFAULT_BASE_MODEL}."
        ),
    )
    parser.add_argument(
        "--checkpoint-n",
        type=int,
        default=None,
        help="Explicit checkpoint number. Default is latest checkpoint + 1.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="20GB",
        help="Model save shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional directory for the temporary HF snapshot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve latest checkpoint and print the planned action.",
    )
    args = parser.parse_args()

    repo_id = _env_or_arg(args.repo_id, "RELIQUARY_HF_REPO_ID", "HF repo id")
    base_model = (
        args.base_model
        or os.environ.get("RELIQUARY_CHECKPOINT")
        or DEFAULT_BASE_MODEL
    )
    token = _env_or_arg(None, "HF_TOKEN", "HF token")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    latest_n = _latest_checkpoint_n(api, repo_id)
    next_n = args.checkpoint_n if args.checkpoint_n is not None else latest_n + 1
    if next_n <= latest_n:
        raise SystemExit(
            f"checkpoint {next_n} would not advance repo latest checkpoint "
            f"{latest_n}; choose a number > {latest_n}"
        )

    print(f"HF repo: {repo_id}")
    print(f"Base model: {base_model}")
    print(f"Latest checkpoint: {latest_n}")
    print(f"Publishing reset as: checkpoint {next_n}")
    if args.dry_run:
        return

    import torch
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    parent = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir())
    snapshot_dir = parent / f"reliquary_base_reset_ckpt_{next_n}"
    shutil.rmtree(snapshot_dir, ignore_errors=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(base_model, token=token)
        print("Loading base model...")
        model = load_text_generation_model(
            base_model,
            torch_dtype=torch.bfloat16,
            token=token,
        )

        print(f"Saving snapshot to {snapshot_dir} ...")
        model.save_pretrained(
            snapshot_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(snapshot_dir)

        print(f"Uploading {snapshot_dir} to {repo_id} as checkpoint {next_n} ...")
        commit = api.upload_folder(
            folder_path=str(snapshot_dir),
            repo_id=repo_id,
            commit_message=f"checkpoint {next_n}",
            delete_patterns="*",
        )
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    print()
    print("Add these to docker/.env before restarting the trainer:")
    print(f"RELIQUARY_RESUME_FROM=sha:{commit.oid}")
    print("RELIQUARY_WANDB_VERSION=base-reset-qwen35")
    print(f"# checkpoint_n={next_n}")


if __name__ == "__main__":
    main()
