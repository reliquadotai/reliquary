"""A corpus validator serving several jobs runs their merged contract, whose
`prompt_encoding` is one task's (see `merge_corpus_contracts`). That is safe
only while nothing on the corpus path reads it: the job's `renderer_id`
decides the encoding. This pins that."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "reliquary"
FORBIDDEN = {"RAW_COMPLETION_PROMPTS", "encode_prompt", "prompt_encoding"}
CORPUS_PATH = sorted(
    [*ROOT.glob("validator/corpus_*.py"), *ROOT.glob("corpus/*.py"),
     ROOT / "miner" / "corpus_miner.py"]
)


def _names(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr


def test_the_guard_sees_the_corpus_modules():
    assert len(CORPUS_PATH) > 10
    assert ROOT / "validator" / "corpus_service.py" in CORPUS_PATH


@pytest.mark.parametrize("path", CORPUS_PATH, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_corpus_module_reads_the_prompt_encoding(path):
    used = FORBIDDEN & set(_names(ast.parse(path.read_text())))
    assert not used, f"{path.name} reads {sorted(used)}; the merged contract's is one task's"
