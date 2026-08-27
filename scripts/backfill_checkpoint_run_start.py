#!/usr/bin/env python3
"""Backfill a legacy run's first trainer publications from HF to R2.

The command is a resumable dry-run unless ``--apply`` is supplied. It writes
each snapshot manifest last, so a completed manifest is the resume marker. The
optional protected-branch squash is separate and only becomes available after
all run-start snapshots have been verified in R2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from botocore.config import Config  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
import boto3  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from reliquary.trainer.publisher import (  # noqa: E402
    _multipart_transfer_config,
    _snapshot_inventory,
)
from reliquary.trainer.retention import (  # noqa: E402
    CheckpointRetentionPolicy,
    HfHistoryManager,
)
from scripts.prepare_hf_retention import (  # noqa: E402
    CHECKPOINT_PROFILE_NAME,
    _checkpoint_rows,
    _infer_run_checkpoints,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keep-initial", type=int, default=50)
    parser.add_argument(
        "--r2-bucket",
        default=os.environ.get("R2_BUCKET_ID", "reliquary"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Download missing HF snapshots and upload them to R2.",
    )
    parser.add_argument(
        "--root-protected-branch",
        action="store_true",
        help=(
            "After a complete R2 backfill, irreversibly super-squash the "
            "protected first-history branch to its publication-50 tip."
        ),
    )
    parser.add_argument(
        "--expected-branch-target",
        help="Required with --root-protected-branch (immutable safety check).",
    )
    return parser


def _r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not account_id or not access_key or not secret_key:
        raise SystemExit(
            "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY "
            "are required with --apply"
        )
    endpoint = (
        os.environ.get("R2_ENDPOINT_URL", "").strip()
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get("R2_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=120,
            max_pool_connections=32,
        ),
    )


def _existing_manifest(client, bucket: str, key: str) -> dict | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    value = json.loads(response["Body"].read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"R2 manifest {key!r} is not a JSON object")
    return value


def _verify_manifest_objects(
    client, bucket: str, prefix: str, manifest: dict,
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"R2 manifest under {prefix!r} has no file inventory")
    for item in files:
        if not isinstance(item, dict) or not item.get("path"):
            raise RuntimeError(
                f"R2 manifest under {prefix!r} has an invalid file entry"
            )
        key = f"{prefix}/{item['path']}"
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise RuntimeError(f"R2 snapshot object is missing: {key}") from exc
        if int(head.get("ContentLength", -1)) != int(item.get("size", -2)):
            raise RuntimeError(f"R2 snapshot object has the wrong size: {key}")


def _upload_snapshot(client, bucket: str, directory: Path, prefix: str) -> None:
    transfer = _multipart_transfer_config()
    for path in sorted(directory.iterdir()):
        if path.is_file():
            client.upload_file(
                str(path),
                bucket,
                f"{prefix}/{path.name}",
                Config=transfer,
            )


def main() -> int:
    args = _parser().parse_args()
    if args.keep_initial <= 0:
        raise SystemExit("--keep-initial must be positive")
    if args.root_protected_branch and not args.apply:
        raise SystemExit("--root-protected-branch requires --apply")
    if args.root_protected_branch and not args.expected_branch_target:
        raise SystemExit(
            "--root-protected-branch requires --expected-branch-target"
        )

    api = HfApi()
    commits = list(reversed(list(api.list_repo_commits(args.repo_id))))
    checkpoints = _infer_run_checkpoints(
        api,
        args.repo_id,
        args.run_id,
        _checkpoint_rows(commits),
    )
    if len(checkpoints) < args.keep_initial:
        raise SystemExit(
            f"run has {len(checkpoints)} trainer publications; expected at "
            f"least {args.keep_initial}"
        )
    selected = checkpoints[:args.keep_initial]
    policy = CheckpointRetentionPolicy(
        enabled=True,
        keep_initial=args.keep_initial,
    )
    branch = policy.first_history_branch(args.run_id)
    plan = {
        "repo_id": args.repo_id,
        "run_id": args.run_id,
        "publication_count": len(checkpoints),
        "backfill_count": len(selected),
        "first_revision": selected[0]["revision"],
        "last_revision": selected[-1]["revision"],
        "protected_branch": branch,
        "r2_prefix": policy.run_start_prefix(args.run_id),
        "apply": bool(args.apply),
        "root_protected_branch": bool(args.root_protected_branch),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.apply:
        return 0

    client = _r2_client()
    for seq, checkpoint in enumerate(selected, start=1):
        revision = str(checkpoint["revision"])
        prefix = policy.snapshot_prefix(args.run_id, seq)
        assert prefix is not None
        manifest_key = f"{prefix}/manifest.json"
        existing = _existing_manifest(client, args.r2_bucket, manifest_key)
        if existing is not None:
            if (
                existing.get("revision") != revision
                or int(existing.get("publication_seq", -1)) != seq
                or existing.get("training_run_id") != args.run_id
            ):
                raise RuntimeError(
                    f"existing R2 manifest conflicts at publication {seq}"
                )
            _verify_manifest_objects(
                client,
                args.r2_bucket,
                prefix,
                existing,
            )
            print(f"skip publication {seq}: verified existing manifest")
            continue

        with tempfile.TemporaryDirectory(
            prefix=f"reliquary-run-start-{seq:06d}-",
        ) as tmp:
            directory = Path(tmp)
            snapshot_download(
                repo_id=args.repo_id,
                revision=revision,
                local_dir=directory,
                token=os.environ.get("HF_TOKEN") or None,
            )
            profile_path = directory / CHECKPOINT_PROFILE_NAME
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if (
                profile.get("training_run_id") != args.run_id
                or profile.get("trained_window_cursor") is None
            ):
                raise RuntimeError(
                    f"checkpoint {revision} is not a trainer publication "
                    f"for run {args.run_id!r}"
                )
            inventory = _snapshot_inventory(
                directory,
                include_sha256=True,
            )
            manifest = {
                "schema_version": 1,
                "protocol_profile_id": profile.get("profile_id"),
                "protocol_version": profile.get("protocol_version"),
                "training_run_id": args.run_id,
                "generation_contract_sha256": profile.get(
                    "generation_contract_sha256"
                ),
                "checkpoint_n": int(checkpoint["checkpoint_n"]),
                "publication_seq": seq,
                "repo_id": args.repo_id,
                "revision": revision,
                "trained_window_cursor": int(
                    profile["trained_window_cursor"]
                ),
                "lr_schedule_step": profile.get("lr_schedule_step"),
                "reason": "legacy_run_start_backfill",
                "retention_class": "run_start_history",
                "snapshot_prefix": prefix,
                "files": inventory,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _upload_snapshot(client, args.r2_bucket, directory, prefix)
            body = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            client.put_object(
                Bucket=args.r2_bucket,
                Key=manifest_key,
                Body=body,
                ContentType="application/json",
            )
            client.put_object(
                Bucket=args.r2_bucket,
                Key=policy.ledger_key(args.run_id, seq),
                Body=body,
                ContentType="application/json",
            )
        print(f"backfilled publication {seq}/{args.keep_initial}: {revision}")

    if args.root_protected_branch:
        expected = str(args.expected_branch_target)
        if expected != str(selected[-1]["revision"]):
            raise SystemExit(
                "--expected-branch-target is not the inferred publication-"
                f"{args.keep_initial} revision"
            )
        rooted = HfHistoryManager(api).compact_protected_branch(
            repo_id=args.repo_id,
            branch=branch,
            expected_revision=expected,
            publication_seq=args.keep_initial,
        )
        print(f"rooted protected branch {branch}@{rooted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
