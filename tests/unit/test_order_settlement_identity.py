"""Dataset orders on any model: settlement of a job is byte-identical.

The archive and the settlement state the period settler writes for a fixed set
of verdicts (every corpus task, orders included, is paid by period since
2026-10-08; the window ``CorpusSettler`` pinned here before is gone), and the
manifest and task contract the admin writes for an operator catalog job and an
eval job (tests/unit/test_admin_*). A digest moving here is a payment change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from tests.unit.test_corpus_settlement import _Archives, _Records, _settler, _v

# sha256 of the canonical JSON, computed when the period settler took over (2026-10-08).
SETTLEMENT_GOLDEN = "4a56f422de86aacb2de4c94cfe809dffe237d41473f0af728bc715094d47f465"


def _canonical(document) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _settled() -> dict:
    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30),
                        "3" * 64: _v("C", 900, ok=False), "4" * 64: _v("A", 7)})
    archives = _Archives()
    asyncio.run(_settler(records, archives).settle_once())
    return {"archives": archives.written, "state": records.state}


def test_settlement_of_an_existing_job_is_byte_identical():
    assert _canonical(_settled()) == SETTLEMENT_GOLDEN
