"""The commit an installed distribution came from (pip's ``direct_url.json``). Imports nothing heavy, so a
miner whose harness is missing can still be told so."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess


def dist_commit(name: str) -> str | None:
    """The commit a distribution was installed from: its VCS pin, or the HEAD
    of the checkout an editable install points at. None when unknowable."""
    try:
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    info = json.loads(raw)
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url") or ""
    if url.startswith("file://"):
        checkout = url[len("file://"):]
        git = ["git", "-c", "safe.directory=*", "-C", checkout]
        try:
            head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10)
            dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True,
                                   timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return None
        # An editable checkout with local changes is not the pinned code,
        # whatever its HEAD says: report nothing rather than the commit.
        if head.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip():
            return head.stdout.strip()
    return None
