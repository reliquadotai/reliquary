#!/usr/bin/env python3
"""Rebuild the ``envscaler_tools_v1`` corpus from its pinned upstream release.

The 2026-09-02 feasibility measurement ran against a loose directory that no
longer exists, pinned only by the sha256 of its two files. Those bytes are
not reproducible: upstream ships ``tools`` and ``init_config`` as *JSON
strings*, the loader wants them decoded, and the directory held someone's
re-serialisation of that decode. This script makes the whole chain
reproducible instead — pinned revision in, canonical bytes out.

The loader addresses scenarios **by position**, so list order is part of the
corpus identity. Order is inherited from upstream and never sorted; only the
mapping keys are sorted, which does not move any element.

Usage::

    python scripts/fetch_envscaler_corpus.py --out ~/envscaler-corpus
    export RELIQUARY_ENVSCALER_DATA=~/envscaler-corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Two datasets, each pinned to the revision the digests below were taken at.
# Both files were uploaded on 2026-01-09 and never rewritten; the later
# upstream commits touch README.md only.
SOURCES = {
    "env_meta.json": {
        "repo_id": "XXHStudyHard/EnvScaler-191-Env",
        "revision": "3d30c6ac2446c06b79d727f196b84df90cc9cb69",
        "filename": "191_env_metadata_processed.json",
        "sha256": (
            "600fa6d9613e3a104807edbff863d26e"
            "a48aeb1d5d3ef758c0e6f109f04f49bd"
        ),
        # Upstream ships this field as a JSON string; the loader wants a list.
        "decode": ("tools",),
        "expected_length": 191,
    },
    "rl_scen.json": {
        "repo_id": "XXHStudyHard/EnvScaler-RL-Scenario",
        "revision": "a14061538b0ffd3e84d44edf40fbe67477f83552",
        "filename": "envscaler_rl_scenario_metadata.json",
        "sha256": (
            "99c05a81710da5adc81453a05f166023"
            "f2cd3235a03a9eb8b5736ba8c7c2abe8"
        ),
        # Same shape, on the field ``reset`` feeds to the world constructor.
        "decode": ("init_config",),
        "expected_length": 2550,
    },
}

# sha256 of what this script writes. Canonical serialisation makes these
# reproducible on any machine; a mismatch means upstream moved under a pin.
NORMALIZED_SHA256 = {
    "env_meta.json": (
        "f0e874c28b3592c5820afcdbae28d3c1"
        "0d7687632d5e1f922a6a68a2d904f81d"
    ),
    "rl_scen.json": (
        "a953b969366d0ad0dc6e5262084fc7d4"
        "6b6cefb992bd02ac5083fec537622e06"
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    """The serialisation the pinned digests are taken over.

    ``sort_keys`` orders mapping keys only. Sequence order — the corpus
    identity — is untouched.
    """
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")


def normalize(raw: bytes, spec: dict) -> bytes:
    """Upstream bytes to the bytes the loader reads. Pure, so it is testable.

    Raises on a record count that does not match the pin: a truncated or
    extended corpus silently renumbers every scenario the loader addresses
    by position.
    """
    records = json.loads(raw)
    if isinstance(records, dict):
        records = list(records.values())
    if len(records) != spec["expected_length"]:
        raise SystemExit(
            f"{len(records)} records, expected {spec['expected_length']}"
        )
    for record in records:
        for field in spec["decode"]:
            record[field] = json.loads(record[field])
    return _canonical(records)


def _fetch(spec: dict) -> bytes:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=spec["repo_id"],
        filename=spec["filename"],
        revision=spec["revision"],
        repo_type="dataset",
    )
    return Path(path).read_bytes()


def build(out: Path, *, strict: bool = True) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, spec in SOURCES.items():
        raw = _fetch(spec)
        upstream = _sha256(raw)
        if upstream != spec["sha256"]:
            raise SystemExit(
                f"{spec['repo_id']}@{spec['revision'][:12]} served "
                f"{upstream} for {spec['filename']}, pinned {spec['sha256']}"
            )
        payload = normalize(raw, spec)
        digest = _sha256(payload)
        if strict and digest != NORMALIZED_SHA256[name]:
            raise SystemExit(
                f"{name} normalised to {digest}, pinned "
                f"{NORMALIZED_SHA256[name]}"
            )
        (out / name).write_bytes(payload)
        written[name] = digest
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="directory to write env_meta.json and rl_scen.json into",
    )
    parser.add_argument(
        "--allow-digest-drift", action="store_true",
        help="write the files and report digests instead of failing on a "
             "mismatch; use when deliberately re-pinning",
    )
    args = parser.parse_args(argv)

    written = build(args.out, strict=not args.allow_digest_drift)
    for name, digest in written.items():
        print(f"{digest}  {args.out / name}")
    print(f"\nexport RELIQUARY_ENVSCALER_DATA={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
