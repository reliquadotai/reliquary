"""A customer's source archive, checked before anything builds it.

Only shape is checked here (paths, sizes, the manifest, hash-pinned requirements, examples): the archive is
read as bytes and never unpacked to disk on the admin host, and nothing in it is executed.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tomllib
from dataclasses import dataclass, field

MAX_PACKED_BYTES = 50 * 1024 * 1024
MAX_UNPACKED_BYTES = 200 * 1024 * 1024
MAX_MEMBERS = 10_000
MIN_EXAMPLES, MAX_EXAMPLES = 2, 64
MAX_EXAMPLE_ANSWER_CHARS = 100_000
KINDS = ("single_turn", "agentic")
_TASKSET_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
# One requirement per logical line, each with at least one --hash=sha256:...
_REQ_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.+!_-]+\Z")
_HASH_RE = re.compile(r"\A--hash=sha256:[0-9a-f]{64}\Z")


class PackageInvalid(ValueError):
    """The archive is refused; the message says why (``package_invalid``)."""


@dataclass(frozen=True)
class Example:
    key: str
    answer: str
    expected_score: float


@dataclass(frozen=True)
class Package:
    sha256: str
    root: str
    taskset: str
    args: dict
    kind: str
    requirements: tuple[str, ...]
    examples: tuple[Example, ...]
    files: tuple[str, ...] = field(repr=False)


def _members(data: bytes) -> list[tuple[tarfile.TarInfo, tarfile.TarFile]]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        members = archive.getmembers()
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise PackageInvalid(f"not a .tar.gz archive: {exc}") from None
    if len(members) > MAX_MEMBERS:
        raise PackageInvalid(f"more than {MAX_MEMBERS} entries")
    return [(m, archive) for m in members]


def _check_path(name: str) -> None:
    parts = name.split("/")
    if name.startswith("/") or "\\" in name or any(p in ("..",) for p in parts) or "\0" in name:
        raise PackageInvalid(f"unsafe path {name!r}")


def _requirements(text: str) -> tuple[str, ...]:
    pinned = []
    for line in text.replace("\\\n", " ").splitlines():
        line = re.sub(r"(^|\s)#.*", "", line).strip()
        if not line:
            continue
        name, *hashes = line.split()
        if name.startswith("-"):
            raise PackageInvalid(f"requirements.lock: option {name!r} is not allowed (no index or url overrides)")
        if not _REQ_NAME_RE.match(name):
            raise PackageInvalid(f"requirements.lock: {name!r} is not pinned as name==version")
        if not hashes or not all(_HASH_RE.match(h) for h in hashes):
            raise PackageInvalid(f"requirements.lock: {name!r} needs --hash=sha256:<64 hex> for every artifact")
        pinned.append(" ".join([name, *hashes]))
    return tuple(pinned)


def _examples(text: str) -> tuple[Example, ...]:
    examples = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise PackageInvalid(f"examples.jsonl line {number}: not JSON") from None
        if not isinstance(row, dict) or set(row) != {"key", "answer", "expected_score"}:
            raise PackageInvalid(f"examples.jsonl line {number}: needs exactly key, answer, expected_score")
        key, answer, score = row["key"], row["answer"], row["expected_score"]
        if not isinstance(key, str) or not key or not isinstance(answer, str) \
                or len(answer) > MAX_EXAMPLE_ANSWER_CHARS:
            raise PackageInvalid(f"examples.jsonl line {number}: key and answer are strings")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise PackageInvalid(f"examples.jsonl line {number}: expected_score is a number in [0, 1]")
        examples.append(Example(key, answer, float(score)))
    if not MIN_EXAMPLES <= len(examples) <= MAX_EXAMPLES:
        raise PackageInvalid(f"examples.jsonl holds {MIN_EXAMPLES} to {MAX_EXAMPLES} examples")
    if not any(e.expected_score >= 0.5 for e in examples) or not any(e.expected_score < 0.5 for e in examples):
        raise PackageInvalid("examples.jsonl needs one example scoring at least 0.5 and one below")
    return tuple(examples)


def read_package(data: bytes) -> Package:
    """The archive's manifest, requirements and examples, or ``PackageInvalid``."""
    if len(data) > MAX_PACKED_BYTES:
        raise PackageInvalid(f"the archive is over {MAX_PACKED_BYTES} bytes")
    sha256 = hashlib.sha256(data).hexdigest()
    total, entries = 0, []
    for member, archive in _members(data):
        _check_path(member.name)
        if not (member.isfile() or member.isdir()):
            raise PackageInvalid(f"{member.name!r}: only regular files and directories")
        total += member.size
        if total > MAX_UNPACKED_BYTES:
            raise PackageInvalid(f"the archive unpacks to over {MAX_UNPACKED_BYTES} bytes")
        name = member.name.removeprefix("./").rstrip("/")
        if name:
            entries.append((name, member, archive))
    files = [name for name, member, _ in entries if member.isfile()]
    # The package sits at the top, or under one top directory (as `tar czf x.tar.gz pkg/` makes it).
    tops = {name.split("/", 1)[0] for name in files}
    root = next(iter(tops)) if len(tops) == 1 and all("/" in f for f in files) else ""
    prefix = f"{root}/" if root else ""
    texts = {}
    for name, member, archive in entries:
        relative = name.removeprefix(prefix)
        if member.isfile() and relative in ("pyproject.toml", "requirements.lock", "examples.jsonl"):
            handle = archive.extractfile(member)
            try:
                texts[relative] = handle.read().decode("utf-8") if handle else ""
            except UnicodeDecodeError:
                raise PackageInvalid(f"{relative} is not UTF-8") from None
    for required in ("pyproject.toml", "requirements.lock", "examples.jsonl"):
        if required not in texts:
            raise PackageInvalid(f"{required} is missing at the archive's top level")
    try:
        project = tomllib.loads(texts["pyproject.toml"])
    except tomllib.TOMLDecodeError as exc:
        raise PackageInvalid(f"pyproject.toml: {exc}") from None
    tool = (project.get("tool") or {}).get("reliquary")
    if not isinstance(tool, dict):
        raise PackageInvalid("pyproject.toml needs a [tool.reliquary] table")
    unknown = sorted(set(tool) - {"taskset", "args", "kind"})
    if unknown:
        raise PackageInvalid(f"[tool.reliquary]: unknown keys {unknown}")
    taskset, args, kind = tool.get("taskset"), tool.get("args", {}), tool.get("kind", "single_turn")
    if not isinstance(taskset, str) or not _TASKSET_RE.match(taskset):
        raise PackageInvalid("[tool.reliquary] taskset is the verifiers v1 taskset id the package registers")
    if not isinstance(args, dict):
        raise PackageInvalid("[tool.reliquary] args is a table")
    if kind not in KINDS:
        raise PackageInvalid(f"[tool.reliquary] kind is one of {KINDS}")
    return Package(sha256=sha256, root=root, taskset=taskset, args=args, kind=kind,
                   requirements=_requirements(texts["requirements.lock"]),
                   examples=_examples(texts["examples.jsonl"]), files=tuple(sorted(files)))


__all__ = ["KINDS", "MAX_PACKED_BYTES", "MAX_UNPACKED_BYTES", "Example", "Package", "PackageInvalid",
           "read_package"]
