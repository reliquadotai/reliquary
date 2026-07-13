#!/usr/bin/env python3
"""Prewarm exact revision-pinned prompt Parquet blobs in the HF cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment


SOURCE_PINS = {
    "opencodeinstruct": (
        OpenCodeInstructEnvironment._CURATED_REPO,
        OpenCodeInstructEnvironment._CURATED_REVISION,
    ),
    "openmathinstruct": (
        OpenMathInstructEnvironment._OMI_REPO,
        OpenMathInstructEnvironment._OMI_REVISION,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download exact pinned prompt Parquet files into the persistent "
            "Hugging Face cache before opening validator windows."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_PINS),
        help="source to prewarm; repeatable, defaults to both",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="fail unless every pinned file is already in the local cache",
    )
    args = parser.parse_args()

    selected = args.source or list(SOURCE_PINS)
    for name in selected:
        repo, revision = SOURCE_PINS[name]
        snapshot = Path(snapshot_download(
            repo_id=repo,
            revision=revision,
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
            local_files_only=args.verify_only,
            max_workers=max(1, args.max_workers),
        ))
        files = sorted((snapshot / "data").glob("*.parquet"))
        if not files:
            raise RuntimeError(f"no cached Parquet files for {repo}@{revision}")
        total_bytes = sum(path.stat().st_size for path in files)
        print(
            f"{name}: ready repo={repo} revision={revision} "
            f"files={len(files)} bytes={total_bytes}"
        )


if __name__ == "__main__":
    main()
