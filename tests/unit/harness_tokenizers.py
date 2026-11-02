"""Tokenizers the harness tests run against, found from the environment.

Set RELIQUARY_QWEN38_TOKENIZER and/or RELIQUARY_HARNESS_TOKENIZERS (paths joined with
os.pathsep); tests whose tokenizer is not on this host skip."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def tokenizer_digest(directory: str | Path) -> str:
    """sha256 over the tokenizer files the renderer reads, in a fixed order (first 16 hex)."""
    digest = hashlib.sha256()
    for name in TOKENIZER_FILES:
        data = (Path(directory) / name).read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


def _tokenizers() -> dict[str, str]:
    paths = [os.environ.get("RELIQUARY_QWEN38_TOKENIZER", "")]
    paths += os.environ.get("RELIQUARY_HARNESS_TOKENIZERS", "").split(os.pathsep)
    found = {}
    for path in filter(None, paths):
        if all((Path(path) / name).is_file() for name in TOKENIZER_FILES):
            found[tokenizer_digest(path)] = path
    return found


TOKENIZERS = _tokenizers()
