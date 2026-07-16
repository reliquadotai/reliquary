#!/usr/bin/env python3
"""Publish recovery weights as the next Reliquary HF checkpoint.

Use this after an incident when operators want miners to move forward to a
fresh base model, an older immutable checkpoint, or a locally calibrated
candidate instead of resuming the latest trained checkpoint.

The script is intentionally append-only: it creates ``checkpoint N+1`` in the
configured HF repo and prints the ``RELIQUARY_RESUME_FROM=sha:<commit>`` line
to pin the validator restart. Never expose an older checkpoint number directly:
miners only download monotonically newer manifests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
CHECKPOINT_TITLE = re.compile(r"^checkpoint\s+(\d+)\s*$", re.IGNORECASE)
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
RECOVERY_MANIFEST_NAME = "reliquary_recovery_manifest.json"


def _env_or_arg(value: str | None, env_name: str, label: str) -> str:
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise SystemExit(f"{label} is required: pass the flag or set {env_name}")
    return resolved


def _repo_checkpoint_state(api, repo_id: str) -> tuple[int, str | None]:
    commits = list(api.list_repo_commits(repo_id=repo_id))
    latest = 0
    for commit in commits:
        match = CHECKPOINT_TITLE.match(commit.title or "")
        if match:
            latest = max(latest, int(match.group(1)))
    head = None
    if commits:
        head = getattr(commits[0], "commit_id", None) or getattr(
            commits[0], "id", None
        )
    return latest, head


def _latest_checkpoint_n(api, repo_id: str) -> int:
    return _repo_checkpoint_state(api, repo_id)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_recovery_manifest(
    snapshot_dir: Path,
    *,
    checkpoint_n: int,
    repo_id: str,
    source_model: str,
    source_revision: str | None,
    parent_commit: str | None,
    created_at: str,
) -> dict:
    artifacts = []
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file() or path.name == RECOVERY_MANIFEST_NAME:
            continue
        artifacts.append({
            "path": path.relative_to(snapshot_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    if not any(row["path"].endswith(".safetensors") for row in artifacts):
        raise SystemExit("recovery snapshot contains no safetensors weights")
    local_source = Path(source_model).expanduser().exists()
    return {
        "schema_version": 1,
        "kind": "reliquary_checkpoint_recovery",
        "created_at": created_at,
        "checkpoint_n": checkpoint_n,
        "target_repo": repo_id,
        "parent_commit": parent_commit,
        "source": {
            "kind": "local" if local_source else "hub",
            # Do not publish an operator's absolute local filesystem path.
            "repo": None if local_source else source_model,
            "revision": source_revision,
        },
        "artifacts": artifacts,
    }


def _source_load_kwargs(
    source_model: str,
    source_revision: str | None,
    token: str,
) -> dict[str, str]:
    """Build fail-closed loader kwargs for a remote or local source."""
    if source_revision is None:
        return {"token": token}
    if Path(source_model).expanduser().exists():
        raise SystemExit(
            "--source-revision cannot be combined with a local source path"
        )
    if IMMUTABLE_REVISION.fullmatch(source_revision) is None:
        raise SystemExit(
            "--source-revision must be a full 40-character commit SHA"
        )
    return {"token": token, "revision": source_revision}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Republish source weights as the next append-only Reliquary HF "
            "checkpoint and print the resume env var."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HF repo to publish into; defaults to RELIQUARY_HF_REPO_ID.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-model",
        default=None,
        help=(
            "HF repo or local model path to republish; defaults to "
            f"RELIQUARY_CHECKPOINT or {DEFAULT_BASE_MODEL}."
        ),
    )
    source.add_argument(
        "--base-model",
        default=None,
        help="Deprecated alias for --source-model.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help=(
            "Required full commit SHA when republishing an immutable older "
            "checkpoint from a remote repository. Omit for a local path."
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
    source_model = (
        args.source_model
        or args.base_model
        or os.environ.get("RELIQUARY_CHECKPOINT")
        or DEFAULT_BASE_MODEL
    )
    token = _env_or_arg(None, "HF_TOKEN", "HF token")
    source_kwargs = _source_load_kwargs(
        source_model,
        args.source_revision,
        token,
    )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    latest_n, parent_commit = _repo_checkpoint_state(api, repo_id)
    next_n = args.checkpoint_n if args.checkpoint_n is not None else latest_n + 1
    if next_n <= latest_n:
        raise SystemExit(
            f"checkpoint {next_n} would not advance repo latest checkpoint "
            f"{latest_n}; choose a number > {latest_n}"
        )

    print(f"HF repo: {repo_id}")
    print(f"Source model: {source_model}")
    print(f"Source revision: {args.source_revision or 'local/default'}")
    print(f"Latest checkpoint: {latest_n}")
    print(f"Parent commit: {parent_commit or 'empty-repository'}")
    print(f"Publishing reset as: checkpoint {next_n}")
    if args.dry_run:
        return

    import torch
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    parent = Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir())
    snapshot_dir = parent / f"reliquary_recovery_ckpt_{next_n}"
    shutil.rmtree(snapshot_dir, ignore_errors=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(source_model, **source_kwargs)
        print("Loading source model...")
        model = load_text_generation_model(
            source_model,
            torch_dtype=torch.bfloat16,
            **source_kwargs,
        )

        print(f"Saving snapshot to {snapshot_dir} ...")
        model.save_pretrained(
            snapshot_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(snapshot_dir)

        manifest = _build_recovery_manifest(
            snapshot_dir,
            checkpoint_n=next_n,
            repo_id=repo_id,
            source_model=source_model,
            source_revision=args.source_revision,
            parent_commit=parent_commit,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        (snapshot_dir / RECOVERY_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"Uploading {snapshot_dir} to {repo_id} as checkpoint {next_n} ...")
        commit = api.upload_folder(
            folder_path=str(snapshot_dir),
            repo_id=repo_id,
            commit_message=f"checkpoint {next_n}",
            parent_commit=parent_commit,
            delete_patterns="*",
        )
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    print()
    print("Add these to docker/.env before restarting the trainer:")
    print(f"RELIQUARY_RESUME_FROM=sha:{commit.oid}")
    print(f"RELIQUARY_WANDB_VERSION=recovery-checkpoint-{next_n}")
    print(f"# checkpoint_n={next_n}")


if __name__ == "__main__":
    main()
