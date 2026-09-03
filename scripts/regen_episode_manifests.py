#!/usr/bin/env python3
"""Regenerate the episode environments' manifests and their pinned digests.

`reliquary/environment/agentic/types.py` is bound by every episode
manifest, so any change to the shared action contract invalidates all of
them at once and the registry refuses to build until the digests are
rewritten. This recomputes each manifest's implementation digest, then the
canonical document digest, and writes the result back into the two places
that pin it — `registry.py` and `profiles.py`.

Substitution is done per `EnvironmentSpec(` / `EnvironmentProfile(` block
rather than by a regex over the whole file. An earlier version of the
logic-environment regenerator matched on a bare contract string and
rewrote a neighbouring environment's digest; splitting on the block
boundary makes that impossible rather than unlikely.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reliquary/environment/registry.py"
PROFILES = ROOT / "reliquary/protocol/profiles.py"
MANIFESTS = ROOT / "reliquary/environment/manifests"

# environment name -> contract id, as pinned in registry.py and profiles.py
EPISODE_ENVIRONMENTS = {
    "reliquary_stateful_tools_v1": "reliquary-stateful-tools-v1",
    "reliquary_retrieval_tools_v1": "reliquary-retrieval-tools-v1",
    "reliquary_workspace_tools_v1": "reliquary-workspace-tools-v1",
}

_DIGEST = re.compile(
    r'(environment_manifest_sha256=\(\n(\s*))"([0-9a-f]+)"\n\s*"([0-9a-f]+)"'
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_document_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(ROOT / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rewrite_block(path: Path, opener: str, contract: str, digest: str) -> None:
    """Replace the digest inside the one block naming this contract."""
    text = path.read_text(encoding="utf-8")
    parts = text.split(opener)
    hits = [
        index for index, part in enumerate(parts)
        if index > 0 and f'"{contract}",' in part
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one {opener} block for {contract} in "
            f"{path.name}, found {len(hits)}"
        )
    index = hits[0]
    # Keep the split shape the file already uses, so the diff is the digest
    # and nothing else.
    def replace(match: re.Match[str]) -> str:
        head_len = len(match.group(3))
        indent = match.group(2)
        return (
            f"{match.group(1)}\"{digest[:head_len]}\"\n"
            f"{indent}\"{digest[head_len:]}\""
        )

    patched, count = _DIGEST.subn(replace, parts[index], count=1)
    if count != 1:
        raise SystemExit(f"no digest pin inside the {contract} block")
    parts[index] = patched
    path.write_text(opener.join(parts), encoding="utf-8")


def main() -> int:
    for environment, contract in EPISODE_ENVIRONMENTS.items():
        manifest_path = MANIFESTS / f"{environment}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        files = list(manifest["implementation_files"])
        before = manifest["implementation_sha256"]
        manifest["implementation_sha256"] = _implementation_sha256(files)

        fixture = manifest.get("golden_fixture")
        if fixture is not None:
            # The goldens are driven by ScriptedPolicy, which hands the runner
            # an action it already holds, so a parsing change cannot move
            # them. Recomputed rather than assumed.
            recomputed = _file_sha256(ROOT / fixture)
            if recomputed != manifest["golden_fixture_sha256"]:
                raise SystemExit(
                    f"{environment}: golden fixture changed on disk; "
                    "regenerate the goldens before the manifest"
                )

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        digest = _canonical_document_sha256(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        _rewrite_block(REGISTRY, "    EnvironmentSpec(", contract, digest)
        _rewrite_block(PROFILES, "EnvironmentProfile(", contract, digest)

        moved = "unchanged" if before == manifest["implementation_sha256"] else "updated"
        print(f"{environment:<32} impl {moved}  manifest {digest[:16]}…")

    check = subprocess.run(
        [sys.executable, "-c",
         "from reliquary.environment.registry import ENVIRONMENT_SPECS;"
         "print(len(ENVIRONMENT_SPECS), 'environments built')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip())
    if check.returncode != 0:
        raise SystemExit("registry rejected the regenerated manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
