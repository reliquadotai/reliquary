"""Each audit batch says what it cost, so "how long does validating one
submission take" has an answer in the log."""

import asyncio
import logging

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_auditor import CorpusAuditor
from tests.unit.test_corpus_audit import _tiny
from tests.unit.test_corpus_auditor import _Records, _record, _Tokenizer


def test_an_audit_batch_logs_its_size_wait_and_speed(caplog):
    model = _tiny(0)
    first, second = _record(model), _record(model)
    first["received_at"], second["received_at"] = 990.0, 995.0
    records = _Records({"a" * 64: first, "b" * 64: second})
    auditor = CorpusAuditor(job_id="math-v1", records=records, model=model,
                            tokenizer=_Tokenizer(), proof=PROOF, clock=lambda: 1000.0)

    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_auditor"):
        asyncio.run(auditor.audit_many(["a" * 64, "b" * 64]))

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("corpus audit batch:")]
    assert len(lines) == 1
    line = lines[0]
    for part in ("records=2", "completions=2", "completion_tokens=140",
                 "oldest_wait=10.0s", "forward=", "verify=", "tokens_per_s="):
        assert part in line, line
