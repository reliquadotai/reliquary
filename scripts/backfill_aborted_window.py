#!/usr/bin/env python3
"""Backfill an explicit, non-rewarding archive tombstone for a missing window."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from botocore.exceptions import ClientError

from reliquary.infrastructure.storage import (
    get_s3_client,
    upload_window_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("window", type=int)
    parser.add_argument("--validator-hotkey", required=True)
    parser.add_argument("--failure-stage", required=True)
    parser.add_argument("--failure-type", required=True)
    parser.add_argument(
        "--environment",
        action="append",
        dest="environments",
        required=True,
    )
    parser.add_argument("--reconstructed-from", default="validator_logs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _tombstone(args: argparse.Namespace) -> dict:
    environments = list(dict.fromkeys(args.environments))
    return {
        "archive_schema_version": 2,
        "window_status": "aborted",
        "window_start": args.window,
        "validator_hotkey": args.validator_hotkey,
        "randomness": "",
        "environment": environments[0],
        "environments": environments,
        "failure_stage": args.failure_stage,
        "failure_type": args.failure_type,
        "reconstructed": True,
        "reconstructed_from": args.reconstructed_from,
        "batch": [],
        "runners_up": [],
        "rejected": [],
        "reject_summary": {},
        "server_reject_summary": {},
        "rewards_by_hotkey": {},
        "rewarded_but_not_selected_by_hotkey": {},
        "training_quarantine": {
            "quarantined": True,
            "reasons": ["aborted_window"],
            "metrics": {},
        },
        "training_accumulator": {
            "schema_version": 1,
            "trained": False,
            "blocked_reason": "aborted_window",
        },
    }


async def _object_exists(key: str) -> bool:
    """Return False only for an authoritative object-not-found response."""
    async with get_s3_client() as client:
        bucket = os.environ.get("R2_BUCKET_ID", "reliquary")
        try:
            await client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(
                exc.response.get("ResponseMetadata", {}).get(
                    "HTTPStatusCode", 0
                )
                or 0
            )
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise
        return True


async def _run(args: argparse.Namespace) -> None:
    archive = _tombstone(args)
    key = f"reliquary/dataset/window-{args.window}.json.gz"
    if not args.execute:
        print(json.dumps(archive, indent=2, sort_keys=True))
        print("dry-run: pass --execute to upload")
        return
    if await _object_exists(key) and not args.force:
        raise SystemExit(f"refusing to overwrite existing archive: {key}")
    await upload_window_dataset(args.window, archive)
    print(f"uploaded {key}")


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
