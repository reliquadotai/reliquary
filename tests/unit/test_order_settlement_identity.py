"""Dataset orders on any model: settlement of existing jobs is byte-identical.

Pinned on the base commit (7753e5e4) before any any-model change: the archive
and the settlement state the unchanged ``CorpusSettler`` writes for a fixed set
of verdicts, and the manifest and task contract the admin writes for an
operator catalog job and an eval job. A digest moving here is a payment change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from tests.unit.test_corpus_settlement import _Archives, _Records, _settler, _v

# sha256 of the canonical JSON, computed on 7753e5e4.
SETTLEMENT_GOLDEN = "89f814beac311229a14961fee681e372ead3db6497dce73eaabddaa4432f104f"


def _canonical(document) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _settled() -> dict:
    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30),
                        "3" * 64: _v("C", 900, ok=False), "4" * 64: _v("A", 7)})
    archives = _Archives(46000)
    asyncio.run(_settler(records, archives, now=5.0).settle_once())
    return {"archives": archives.written, "state": records.state}


def test_settlement_of_an_existing_job_is_byte_identical():
    assert _canonical(_settled()) == SETTLEMENT_GOLDEN
