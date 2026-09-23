"""Every accepted submission re-run through the job's own model, on CPU tiny models."""

import asyncio
import base64

import pytest
import torch

from reliquary.corpus.encoding import prompt_token_ids
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import build_chunk_proofs
from reliquary.validator.corpus_audit import completion_hidden_states
from reliquary.validator.corpus_auditor import CorpusAuditor
from tests.unit.test_corpus_audit import _tiny

ID = "e" * 64


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [10 + (ord(c) % 50) for c in text]


class _Records:
    def __init__(self, submissions):
        self.submissions = submissions
        self.verdicts = {}

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def list_submission_ids(self, job_id):
        return sorted(self.submissions)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def write_verdict(self, job_id, sid, verdict):
        if sid in self.verdicts:
            return False
        self.verdicts[sid] = verdict
        return True

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)


def _record(model, rendered="hello", completion=None):
    prompt = prompt_token_ids(_Tokenizer(), rendered)
    completion = completion or list(range(100, 170))
    hidden = completion_hidden_states(model, prompt + completion, len(prompt))
    proofs = [base64.b64encode(p).decode() for p in build_chunk_proofs(
        hidden, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)]
    return {"hotkey": "5Hot", "rendered_prompt": rendered, "token_count": len(completion),
            "completions": [{"tokens": completion, "text": "", "proofs": proofs}]}


def _auditor(model, records, proof=PROOF):
    return CorpusAuditor(job_id="math-v1", records=records, model=model,
                         tokenizer=_Tokenizer(), proof=proof)


def test_an_honest_submission_passes():
    model = _tiny(0)
    records = _Records({ID: _record(model)})
    verdict = asyncio.run(_auditor(model, records).audit(ID))
    assert verdict["passed"] is True and records.verdicts[ID]["token_count"] == 70


def test_another_models_proofs_fail():
    records = _Records({ID: _record(_tiny(1))})
    verdict = asyncio.run(_auditor(_tiny(0), records).audit(ID))
    assert verdict["passed"] is False and verdict["hotkey"] == "5Hot"


def test_one_failing_completion_fails_the_submission():
    good, bad = _record(_tiny(0)), _record(_tiny(1))
    good["completions"].append(bad["completions"][0])
    records = _Records({ID: good})
    assert asyncio.run(_auditor(_tiny(0), records).audit(ID))["passed"] is False


def test_a_configuration_error_writes_no_verdict():
    from dataclasses import replace

    model = _tiny(0)
    records = _Records({ID: _record(model)})
    wide = replace(PROOF, topk=1024)  # wider than the tiny model's 128 activations per row x 32
    assert asyncio.run(_auditor(model, records, proof=wide).audit(ID)) is None
    assert records.verdicts == {}


def test_pending_ids_are_the_submissions_without_a_verdict():
    records = _Records({ID: {}, "f" * 64: {}})
    records.verdicts[ID] = {}
    assert asyncio.run(_auditor(_tiny(0), records).pending_ids()) == ["f" * 64]


def test_a_second_audit_of_the_same_submission_changes_nothing():
    model = _tiny(0)
    records = _Records({ID: _record(model)})
    auditor = _auditor(model, records)
    first = asyncio.run(auditor.audit(ID))
    asyncio.run(auditor.audit(ID))
    assert records.verdicts[ID] == first
