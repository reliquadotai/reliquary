#!/usr/bin/env python3
"""Archive selected checkpoints from a finished HF run, then compact it.

The command is deliberately two phase. ``--apply`` copies and verifies the
selected immutable revisions in R2. Adding ``--squash`` performs the separate,
irreversible Hugging Face super-squash only after every archive manifest and
object has been verified. The active trainer repository is always rejected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from huggingface_hub import HfApi, snapshot_download


CHECKPOINT_TITLE = re.compile(r"^checkpoint\s+(\d+)(?:\s|$)", re.IGNORECASE)
ARCHIVE_ROOT = "reliquary/checkpoint-milestones"
LEDGER_ROOT = "reliquary/checkpoint-ledger"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated checkpoint numbers to preserve.",
    )
    parser.add_argument(
        "--expected-head",
        help="Required with --apply; immutable precondition for all writes.",
    )
    parser.add_argument(
        "--r2-bucket",
        default=(
            os.environ.get("R2_BUCKET_ID")
            or os.environ.get("R2_BUCKET")
            or "reliquary"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy and verify the selected snapshots in R2.",
    )
    parser.add_argument(
        "--squash",
        action="store_true",
        help="After archive verification, super-squash the HF main branch.",
    )
    return parser


def _checkpoint_numbers(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--checkpoints must contain only integers") from exc
    if not values or any(value < 0 for value in values):
        raise SystemExit("--checkpoints requires positive checkpoint numbers")
    if len(values) != len(set(values)):
        raise SystemExit("--checkpoints contains a duplicate")
    return sorted(values)


def _run_key(repo_id: str, source_head: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo_id).strip("-._")
    return f"finished-{name[:60]}-{source_head[:12]}"


def _selected_commits(api: HfApi, repo_id: str, numbers: list[int]) -> list[dict]:
    # The Hub returns newest first. setdefault intentionally selects the newest
    # commit if a checkpoint title was used more than once during publication.
    by_number: dict[int, Any] = {}
    for commit in api.list_repo_commits(repo_id=repo_id, repo_type="model"):
        match = CHECKPOINT_TITLE.match(str(commit.title or ""))
        if match:
            by_number.setdefault(int(match.group(1)), commit)
    missing = sorted(set(numbers) - set(by_number))
    if missing:
        raise SystemExit(f"checkpoint(s) are absent from HF history: {missing}")
    return [
        {
            "checkpoint_n": number,
            "revision": str(by_number[number].commit_id),
            "created_at": by_number[number].created_at.isoformat(),
            "title": str(by_number[number].title or ""),
        }
        for number in numbers
    ]


def _r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
    access_key = (
        os.environ.get("R2_ACCESS_KEY_ID")
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or ""
    ).strip()
    secret_key = (
        os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
        or ""
    ).strip()
    if not account_id or not access_key or not secret_key:
        raise SystemExit(
            "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY "
            "are required with --apply"
        )
    endpoint = (
        os.environ.get("R2_ENDPOINT_URL", "").strip()
        or os.environ.get("R2_ENDPOINT", "").strip()
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=(
            os.environ.get("R2_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "auto"
        ),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=120,
            max_pool_connections=4,
        ),
    )


def _snapshot_files(directory: Path) -> list[Path]:
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(directory: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _snapshot_files(directory)
    ]


def _manifest(client, bucket: str, key: str) -> dict | None:
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


def _assert_manifest_identity(
    manifest: dict,
    *,
    repo_id: str,
    source_head: str,
    checkpoint_n: int,
    revision: str,
) -> None:
    expected = {
        "repo_id": repo_id,
        "source_head": source_head,
        "checkpoint_n": checkpoint_n,
        "revision": revision,
        "retention_class": "finished_run_milestone",
    }
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "existing R2 manifest conflicts with requested archive: "
            f"expected={expected!r} observed={observed!r}"
        )


def _verify_objects(client, bucket: str, prefix: str, manifest: dict) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"R2 manifest under {prefix!r} has no files")
    for item in files:
        if not isinstance(item, dict) or not item.get("path"):
            raise RuntimeError(f"invalid R2 inventory under {prefix!r}")
        key = f"{prefix}/{item['path']}"
        try:
            result = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise RuntimeError(f"missing R2 snapshot object: {key}") from exc
        if int(result.get("ContentLength", -1)) != int(item.get("size", -2)):
            raise RuntimeError(f"R2 object has wrong size: {key}")
        metadata = result.get("Metadata") or {}
        if metadata.get("sha256") != item.get("sha256"):
            raise RuntimeError(f"R2 object has wrong SHA-256 metadata: {key}")


def _upload_snapshot(
    client,
    *,
    bucket: str,
    prefix: str,
    directory: Path,
    inventory: list[dict],
) -> None:
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=2,
        use_threads=True,
    )
    for item in inventory:
        source = directory / str(item["path"])
        client.upload_file(
            str(source),
            bucket,
            f"{prefix}/{item['path']}",
            ExtraArgs={"Metadata": {"sha256": str(item["sha256"])}},
            Config=transfer,
        )


def _put_manifest(client, bucket: str, key: str, manifest: dict) -> None:
    body = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def _hf_tree(api: HfApi, repo_id: str, revision: str) -> dict[str, int]:
    info = api.model_info(
        repo_id=repo_id,
        revision=revision,
        files_metadata=True,
    )
    return {
        str(item.rfilename): int(item.size or 0)
        for item in info.siblings
    }


def _verify_completed_squash(
    *,
    args: argparse.Namespace,
    api: HfApi,
    token: str | None,
    numbers: list[int],
    expected_source_head: str,
    final_head: str,
) -> int:
    """Resume safely when the Hub completed squash before its cache updated."""

    commits = api.list_repo_commits(args.repo_id, repo_type="model")
    expected_message = (
        f"checkpoint {numbers[-1]} "
        "(finished run; R2 milestones verified; "
        f"source {expected_source_head[:12]})"
    )
    if (
        len(commits) != 1
        or str(commits[0].commit_id) != final_head
        or str(commits[0].title or "") != expected_message
    ):
        raise SystemExit(
            "HF HEAD moved away from --expected-head and is not the exact "
            "single-commit root created by this command; refusing to continue"
        )
    refs = api.list_repo_refs(args.repo_id, repo_type="model")
    if [branch.name for branch in refs.branches] != ["main"]:
        raise RuntimeError("unexpected HF branches after super-squash")

    client = _r2_client()
    run_key = _run_key(args.repo_id, expected_source_head)
    archived: list[dict] = []
    final_manifest: dict | None = None
    for checkpoint_n in numbers:
        prefix = f"{ARCHIVE_ROOT}/{run_key}/checkpoint-{checkpoint_n:06d}"
        manifest_key = f"{prefix}/manifest.json"
        manifest = _manifest(client, args.r2_bucket, manifest_key)
        if manifest is None:
            raise RuntimeError(f"missing R2 manifest after squash: {manifest_key}")
        revision = str(manifest.get("revision") or "")
        _assert_manifest_identity(
            manifest,
            repo_id=args.repo_id,
            source_head=expected_source_head,
            checkpoint_n=checkpoint_n,
            revision=revision,
        )
        _verify_objects(client, args.r2_bucket, prefix, manifest)
        archived.append(
            {
                "checkpoint_n": checkpoint_n,
                "revision": revision,
                "manifest_key": manifest_key,
                "bytes": sum(int(item["size"]) for item in manifest["files"]),
            }
        )
        if checkpoint_n == numbers[-1]:
            final_manifest = manifest
    if final_manifest is None or final_manifest.get("revision") != expected_source_head:
        raise RuntimeError("the final R2 manifest does not bind the source HEAD")

    archived_tree = {
        str(item["path"]): int(item["size"])
        for item in final_manifest["files"]
    }
    if _hf_tree(api, args.repo_id, final_head) != archived_tree:
        raise RuntimeError("rooted HF file tree sizes differ from R2 final snapshot")
    with tempfile.TemporaryDirectory(prefix="reliquary-root-resume-verify-") as temporary:
        directory = Path(temporary)
        snapshot_download(
            repo_id=args.repo_id,
            revision=final_head,
            local_dir=directory,
            token=token,
            max_workers=2,
        )
        rooted_inventory = _inventory(directory)
    archived_inventory = sorted(
        final_manifest["files"], key=lambda item: str(item["path"])
    )
    if rooted_inventory != archived_inventory:
        raise RuntimeError("rooted HF HEAD differs from verified R2 final snapshot")
    info = api.model_info(args.repo_id, files_metadata=True)
    print(json.dumps({
        "repo_id": args.repo_id,
        "source_head": expected_source_head,
        "final_head": final_head,
        "final_commit_count": 1,
        "r2_run_key": run_key,
        "archived": archived,
        "archive_verified": True,
        "squash_complete": True,
        "rooted_head_matches_r2": True,
        "reported_used_storage_bytes": int(info.used_storage or 0),
        "storage_reclamation_pending": True,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    numbers = _checkpoint_numbers(args.checkpoints)
    if args.squash and not args.apply:
        raise SystemExit("--squash requires --apply")
    if args.apply and not args.expected_head:
        raise SystemExit("--apply requires --expected-head")
    active_repo = os.environ.get("RELIQUARY_HF_REPO_ID", "").strip()
    if active_repo and active_repo == args.repo_id:
        raise SystemExit(
            "refusing to archive or compact RELIQUARY_HF_REPO_ID: this is "
            "the active trainer repository"
        )

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    info = api.model_info(args.repo_id, files_metadata=True)
    source_head = str(info.sha)
    if (
        args.apply
        and args.expected_head
        and source_head != args.expected_head
    ):
        if not args.squash:
            raise SystemExit(
                f"HF HEAD moved: expected {args.expected_head}, "
                f"observed {source_head}"
            )
        return _verify_completed_squash(
            args=args,
            api=api,
            token=token,
            numbers=numbers,
            expected_source_head=str(args.expected_head),
            final_head=source_head,
        )
    selected = _selected_commits(api, args.repo_id, numbers)
    if selected[-1]["revision"] != source_head:
        raise SystemExit(
            "the newest selected checkpoint must be repository HEAD; retain "
            "the final model before compacting"
        )
    refs = api.list_repo_refs(args.repo_id, repo_type="model")
    extra_branches = [
        branch.name for branch in refs.branches if branch.name != "main"
    ]
    if extra_branches:
        raise SystemExit(
            "refusing to compact while non-main branches retain history: "
            f"{extra_branches}"
        )
    head_tree = {
        str(item.rfilename): int(item.size or 0) for item in info.siblings
    }
    result: dict[str, Any] = {
        "repo_id": args.repo_id,
        "source_head": source_head,
        "used_storage_bytes": int(info.used_storage or 0),
        "head_tree_bytes": sum(head_tree.values()),
        "estimated_reclaim_bytes": max(
            0, int(info.used_storage or 0) - sum(head_tree.values())
        ),
        "selected": selected,
        "apply": bool(args.apply),
        "squash": bool(args.squash),
    }
    if not args.apply:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if source_head != args.expected_head:
        raise SystemExit(
            f"HF HEAD moved: expected {args.expected_head}, observed {source_head}"
        )

    client = _r2_client()
    run_key = _run_key(args.repo_id, source_head)
    archived: list[dict] = []
    head_manifest: dict | None = None
    for checkpoint in selected:
        checkpoint_n = int(checkpoint["checkpoint_n"])
        revision = str(checkpoint["revision"])
        prefix = f"{ARCHIVE_ROOT}/{run_key}/checkpoint-{checkpoint_n:06d}"
        manifest_key = f"{prefix}/manifest.json"
        existing = _manifest(client, args.r2_bucket, manifest_key)
        if existing is not None:
            _assert_manifest_identity(
                existing,
                repo_id=args.repo_id,
                source_head=source_head,
                checkpoint_n=checkpoint_n,
                revision=revision,
            )
            _verify_objects(client, args.r2_bucket, prefix, existing)
            manifest = existing
            print(
                f"verified existing R2 checkpoint {checkpoint_n} "
                f"({revision[:12]})",
                flush=True,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"reliquary-finished-{checkpoint_n:06d}-",
            ) as temporary:
                directory = Path(temporary)
                snapshot_download(
                    repo_id=args.repo_id,
                    revision=revision,
                    local_dir=directory,
                    token=token,
                    max_workers=2,
                )
                inventory = _inventory(directory)
                if not inventory:
                    raise RuntimeError(f"HF snapshot {revision} is empty")
                manifest = {
                    "schema_version": 1,
                    "repo_id": args.repo_id,
                    "source_head": source_head,
                    "checkpoint_n": checkpoint_n,
                    "revision": revision,
                    "checkpoint_created_at": checkpoint["created_at"],
                    "retention_class": "finished_run_milestone",
                    "reason": "finished_run_compaction",
                    "snapshot_prefix": prefix,
                    "files": inventory,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                _upload_snapshot(
                    client,
                    bucket=args.r2_bucket,
                    prefix=prefix,
                    directory=directory,
                    inventory=inventory,
                )
                # The manifest is the atomic completion marker and is always
                # written after every data object has finished uploading.
                _put_manifest(client, args.r2_bucket, manifest_key, manifest)
                _put_manifest(
                    client,
                    args.r2_bucket,
                    f"{LEDGER_ROOT}/{run_key}/checkpoint-{checkpoint_n:06d}.json",
                    manifest,
                )
            persisted = _manifest(client, args.r2_bucket, manifest_key)
            if persisted is None:
                raise RuntimeError(f"R2 manifest disappeared: {manifest_key}")
            _assert_manifest_identity(
                persisted,
                repo_id=args.repo_id,
                source_head=source_head,
                checkpoint_n=checkpoint_n,
                revision=revision,
            )
            _verify_objects(client, args.r2_bucket, prefix, persisted)
            manifest = persisted
            print(
                f"archived and verified checkpoint {checkpoint_n} "
                f"({revision[:12]})",
                flush=True,
            )
        archived.append(
            {
                "checkpoint_n": checkpoint_n,
                "revision": revision,
                "manifest_key": manifest_key,
                "bytes": sum(int(item["size"]) for item in manifest["files"]),
            }
        )
        if revision == source_head:
            head_manifest = manifest

    if head_manifest is None:
        raise RuntimeError("final HF HEAD snapshot was not archived")
    result["r2_run_key"] = run_key
    result["archived"] = archived
    if not args.squash:
        result["archive_verified"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Re-check all mutable preconditions immediately before the irreversible
    # call. No branch is allowed to keep the discarded ancestry billable.
    observed_head = str(api.model_info(args.repo_id).sha)
    if observed_head != source_head:
        raise RuntimeError(
            f"HF HEAD moved before squash: expected={source_head} "
            f"observed={observed_head}"
        )
    observed_refs = api.list_repo_refs(args.repo_id, repo_type="model")
    if [branch.name for branch in observed_refs.branches] != ["main"]:
        raise RuntimeError("HF branches changed before squash")
    message = (
        f"checkpoint {selected[-1]['checkpoint_n']} "
        f"(finished run; R2 milestones verified; source {source_head[:12]})"
    )
    api.super_squash_history(
        repo_id=args.repo_id,
        branch="main",
        repo_type="model",
        commit_message=message,
    )
    final_head = str(api.model_info(args.repo_id).sha)
    if not final_head or final_head == source_head:
        raise RuntimeError("HF super-squash did not create a new root SHA")
    commits = api.list_repo_commits(args.repo_id, repo_type="model")
    if len(commits) != 1 or str(commits[0].commit_id) != final_head:
        raise RuntimeError("HF main is not a single rooted commit after squash")
    if _hf_tree(api, args.repo_id, final_head) != head_tree:
        raise RuntimeError("HF file tree sizes changed during super-squash")

    # Verify the new rooted HEAD byte-for-byte against its R2 recovery copy.
    with tempfile.TemporaryDirectory(prefix="reliquary-root-verify-") as temporary:
        directory = Path(temporary)
        snapshot_download(
            repo_id=args.repo_id,
            revision=final_head,
            local_dir=directory,
            token=token,
            max_workers=2,
        )
        rooted_inventory = _inventory(directory)
    archived_inventory = sorted(
        head_manifest["files"], key=lambda item: str(item["path"])
    )
    if rooted_inventory != archived_inventory:
        raise RuntimeError("rooted HF HEAD differs from verified R2 final snapshot")
    result.update(
        {
            "archive_verified": True,
            "squash_complete": True,
            "final_head": final_head,
            "final_commit_count": 1,
            "rooted_head_matches_r2": True,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
