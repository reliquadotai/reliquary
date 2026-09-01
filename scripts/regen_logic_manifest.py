#!/usr/bin/env python3
"""Regenerate the ``reliquarylogic_v1`` goldens, manifest and pinned digests.

The registry refuses to build when a bound file changes, so the catalog
cannot be imported while the digests are stale. This loads the environment
modules without the package ``__init__`` that builds the catalog, rebuilds
the goldens and manifest, then writes the new digest back into
``registry.py`` and ``profiles.py``. Importing the real registry afterwards
is the verification: it fails if the digest algorithm inlined here has
drifted from the registry's own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reliquary/environment/registry.py"
PROFILES = ROOT / "reliquary/protocol/profiles.py"
MANIFEST = ROOT / "reliquary/environment/manifests/reliquarylogic_v1.json"
FIXTURE = ROOT / "tests/fixtures/reliquarylogic_v1.jsonl"

IMPLEMENTATION_FILES = [
    "reliquary/environment/logic_tasks.py",
    "reliquary/environment/records_tasks.py",
    "reliquary/environment/reliquarylogic.py",
    "reliquary/environment/structured_output.py",
]

# Anchored on the contract id so the substitution cannot walk onto a
# neighbouring environment's pin: an earlier version keyed off the bare
# string "reliquary-logic-v1", which also names the prompt template, and it
# rewrote reliquaryverifiable_v1's digest instead.
_PIN = re.compile(
    r'((?:contract_version|environment_contract_id)="reliquary-logic-v1",\n'
    r'\s*environment_manifest_sha256=\(\n\s*)'
    r'"[0-9a-f]{31}"(\n\s*)"[0-9a-f]{33}"'
)


def _rewrite_pins(digest: str) -> None:
    head, tail = digest[:31], digest[31:]
    for path in (REGISTRY, PROFILES):
        text = path.read_text(encoding="utf-8")
        patched, count = _PIN.subn(
            lambda m: f'{m.group(1)}"{head}"{m.group(2)}"{tail}"', text
        )
        if count != 1:
            raise SystemExit(
                f"expected exactly one reliquary-logic-v1 pin in {path}, "
                f"found {count}"
            )
        path.write_text(patched, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_document_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_sha256(paths) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(ROOT / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _perturb(reference, spec):
    """A near-miss the checker must reject, whatever the answer shape is."""
    if spec.get("check") == "numbrix_path":
        broken = [list(row) for row in reference]
        size = spec["constraints"]["size"]
        broken[0][0], broken[size - 1][size - 1] = (
            broken[size - 1][size - 1], broken[0][0],
        )
        return broken
    if isinstance(reference, bool):
        return not reference
    if isinstance(reference, str):
        return reference[:-1] if len(reference) > 1 else reference + "x"
    if isinstance(reference, dict):
        broken = dict(reference)
        key = sorted(broken)[0]
        broken[key] = (broken[key] + 1) % 10
        return broken
    if isinstance(reference, list):
        return reference[:-1] if len(reference) > 1 else reference + [0]
    return None


def _import_without_catalog():
    """The package __init__ builds the catalog, which refuses a stale pin."""
    package = types.ModuleType("reliquary.environment")
    package.__path__ = [str(ROOT / "reliquary/environment")]
    sys.modules.setdefault("reliquary", types.ModuleType("reliquary"))
    sys.modules["reliquary"].__path__ = [str(ROOT / "reliquary")]
    sys.modules["reliquary.environment"] = package
    from reliquary.environment.logic_tasks import (  # noqa: E402
        VIRTUAL_LENGTH, generate_logic_task,
    )
    from reliquary.environment.reliquarylogic import (  # noqa: E402
        ReliquaryLogicEnvironment,
    )
    return VIRTUAL_LENGTH, generate_logic_task, ReliquaryLogicEnvironment


def main() -> int:
    VIRTUAL_LENGTH, generate_logic_task, ReliquaryLogicEnvironment = (
        _import_without_catalog()
    )

    environment = ReliquaryLogicEnvironment()

    # One golden per family, taken at the first index that produces it, so
    # coverage follows the generator list instead of a hand-picked tuple.
    indices: dict[str, int] = {}
    for index in range(20000):
        family = generate_logic_task(index).family
        indices.setdefault(family, index)
    expected_families = {
        generate_logic_task(index).family for index in range(20000)
    }
    if set(indices) != expected_families:
        raise SystemExit("golden selection missed a family")

    rows = []
    for index in sorted(indices.values()):
        problem = environment.get_problem(index)
        spec = json.loads(problem["ground_truth"])
        reference = generate_logic_task(index).expected
        accepted = json.dumps({"result": reference}, separators=(",", ":"))
        rejected = [
            json.dumps({"result": _perturb(reference, spec)},
                       separators=(",", ":")),
            '{"result":null}',
            '{"result":[]}',
        ]

        assert environment.compute_reward(problem, accepted) == 1.0, index
        for bad in rejected:
            assert environment.compute_reward(problem, bad) == 0.0, index

        rows.append({
            "index": index,
            "id": problem["id"],
            "family": spec["family"],
            "operation_id": problem["operation_id"],
            "difficulty": problem["difficulty"],
            "prompt_sha256": hashlib.sha256(
                problem["prompt"].encode("utf-8")
            ).hexdigest(),
            "target_sha256": hashlib.sha256(
                problem["ground_truth"].encode("utf-8")
            ).hexdigest(),
            "accepted_completion": accepted,
            "rejected_completions": rejected,
        })

    FIXTURE.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["golden_fixture_sha256"] = _file_sha256(FIXTURE)
    manifest["implementation_files"] = IMPLEMENTATION_FILES
    manifest["implementation_sha256"] = _implementation_sha256(
        IMPLEMENTATION_FILES
    )
    manifest["virtual_length"] = VIRTUAL_LENGTH
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    digest = _canonical_document_sha256(
        json.loads(MANIFEST.read_text(encoding="utf-8"))
    )
    _rewrite_pins(digest)

    check = subprocess.run(
        [sys.executable, "-c",
         "from reliquary.environment.registry import get_environment_spec;"
         "print(get_environment_spec('reliquarylogic_v1').name)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if check.returncode != 0:
        print(check.stderr, file=sys.stderr)
        raise SystemExit("registry rejected the regenerated manifest")
    print(f"golden  {manifest['golden_fixture_sha256']}")
    print(f"impl    {manifest['implementation_sha256']}")
    print(f"manifest {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
