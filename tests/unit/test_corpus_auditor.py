"""Every accepted submission re-run through the job's own model, on CPU tiny models."""

import asyncio
import base64
import gc
import sys
import weakref

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


def test_no_completions_fails_closed():
    model = _tiny(0)
    empty = _record(model)
    empty["completions"] = []
    records = _Records({ID: empty})
    verdict = asyncio.run(_auditor(model, records).audit(ID))
    assert (verdict["passed"], verdict["reason"]) == (False, "no_completions")


def test_run_survives_a_store_exception_and_keeps_draining():
    model = _tiny(0)
    bad_id = "c" * 64
    good_id = "d" * 64

    class _FlakyRecords(_Records):
        async def read_submission(self, job_id, sid):
            if sid == bad_id:
                raise ConnectionError("transient store failure")
            return await super().read_submission(job_id, sid)

    records = _FlakyRecords({good_id: _record(model)})
    auditor = _auditor(model, records)
    auditor.enqueue(bad_id)
    auditor.enqueue(good_id)

    async def _wait_for_the_good_verdict():
        task = asyncio.create_task(auditor.run())
        try:
            while good_id not in records.verdicts:
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(asyncio.wait_for(_wait_for_the_good_verdict(), timeout=5))
    assert records.verdicts[good_id]["passed"] is True


# --- final review, finding 4: retry what failed, and stop loudly on a broken validator ---


def test_a_submission_whose_read_failed_is_audited_on_the_next_rescan():
    model = _tiny(0)
    flaky_id = "c" * 64

    class _OnceFlakyRecords(_Records):
        failed = False

        async def read_submission(self, job_id, sid):
            if sid == flaky_id and not self.failed:
                self.failed = True
                raise ConnectionError("transient store failure")
            return await super().read_submission(job_id, sid)

    records = _OnceFlakyRecords({flaky_id: _record(model)})
    auditor = CorpusAuditor(job_id="math-v1", records=records, model=model,
                            tokenizer=_Tokenizer(), proof=PROOF, rescan_every_seconds=0.05)

    async def _wait():
        task = asyncio.create_task(auditor.run())
        try:
            while flaky_id not in records.verdicts:
                assert not task.done(), task
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(asyncio.wait_for(_wait(), timeout=10))
    assert records.failed and records.verdicts[flaky_id]["passed"] is True


def test_consecutive_validator_side_errors_stop_the_auditor_loudly():
    from dataclasses import replace

    from reliquary.validator.corpus_auditor import (
        MAX_CONSECUTIVE_VALIDATOR_ERRORS,
        CorpusAuditorHalted,
    )

    model = _tiny(0)
    record = _record(model)
    records = _Records({f"{i:064x}": record for i in range(MAX_CONSECUTIVE_VALIDATOR_ERRORS + 1)})
    wide = replace(PROOF, topk=1024)  # a configuration error on every audit
    auditor = CorpusAuditor(job_id="math-v1", records=records, model=model,
                            tokenizer=_Tokenizer(), proof=wide, rescan_every_seconds=0.05)

    with pytest.raises(CorpusAuditorHalted):
        asyncio.run(asyncio.wait_for(auditor.run(), timeout=10))
    assert records.verdicts == {}


# --- re-review: an out-of-vocabulary id is the miner's fault, not ours ---


def test_out_of_vocab_records_fail_and_never_halt_the_auditor():
    from reliquary.validator.corpus_auditor import MAX_CONSECUTIVE_VALIDATOR_ERRORS

    model = _tiny(0)
    vocabulary = model.get_input_embeddings().num_embeddings
    bad = _record(model)
    bad["completions"][0]["tokens"] = list(range(100, 169)) + [vocabulary]
    ids = [f"{i:064x}" for i in range(MAX_CONSECUTIVE_VALIDATOR_ERRORS + 2)]
    records = _Records({sid: bad for sid in ids})
    auditor = _auditor(model, records)

    async def _drain():
        task = asyncio.create_task(auditor.run())
        try:
            while len(records.verdicts) < len(ids):
                assert not task.done(), task
                await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(asyncio.wait_for(_drain(), timeout=20))
    assert all(v["passed"] is False and v["reason"] == "token_out_of_vocab"
               for v in records.verdicts.values())


# --- batched audit: one bad record must not stall the rest of its batch ---


def test_a_batch_with_one_validator_error_still_judges_the_others(monkeypatch):
    model = _tiny(0)
    good, bad = "a" * 64, "b" * 64
    records = _Records({good: _record(model), bad: _record(model)})
    auditor = _auditor(model, records)
    real = auditor._judge_many

    def flaky(batch):
        if any(r is records.submissions[bad] for r in batch) and len(batch) > 1:
            raise RuntimeError("batch OOM")
        if any(r is records.submissions[bad] for r in batch):
            raise RuntimeError("still failing alone")
        return real(batch)

    monkeypatch.setattr(auditor, "_judge_many", flaky)
    asyncio.run(auditor.audit_many([good, bad]))
    assert records.verdicts[good]["passed"] is True
    assert bad not in records.verdicts


def test_the_solo_retries_run_after_the_failed_batchs_exception_is_forgotten(monkeypatch):
    # A batch-level exception's traceback pins every frame (and tensor) live
    # when it was raised; asyncio.to_thread runs the judged call itself in a
    # worker thread with its own, always-clear exc_info, so the only place
    # that can observe "are we still inside the failed batch's except block"
    # is the coroutine's own thread, right as it hands work to that thread.
    model = _tiny(0)
    good, bad = "a" * 64, "b" * 64
    records = _Records({good: _record(model), bad: _record(model)})
    auditor = _auditor(model, records)
    real_judge_many = auditor._judge_many
    real_to_thread = asyncio.to_thread
    dispatch_exc_info = []
    solo_exc_ref = []

    class _SimulatedOOM(RuntimeError):
        pass  # a plain RuntimeError instance cannot take a weakref

    def flaky(batch):
        if any(r is records.submissions[bad] for r in batch) and len(batch) > 1:
            raise _SimulatedOOM("batch OOM")
        if any(r is records.submissions[bad] for r in batch):
            exc = _SimulatedOOM("still failing alone")
            solo_exc_ref.append(weakref.ref(exc))
            raise exc
        return real_judge_many(batch)

    async def spying_to_thread(func, *args, **kwargs):
        dispatch_exc_info.append(sys.exc_info())
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spying_to_thread)
    monkeypatch.setattr(auditor, "_judge_many", flaky)

    asyncio.run(auditor.audit_many([good, bad]))

    # Dispatch 0 is the whole-batch call (nothing has failed yet); 1 and 2 are
    # the two solo retries, which must run with no exception live in the
    # caller (the failed batch's traceback, and whatever it pinned, forgotten
    # before either retry starts).
    assert len(dispatch_exc_info) == 3
    assert dispatch_exc_info[1:] == [(None, None, None), (None, None, None)]

    gc.collect()
    # audit_many must not keep the failing record's own exception (and its
    # traceback) referenced anywhere once it has recorded the failure as a
    # plain message.
    assert solo_exc_ref and solo_exc_ref[0]() is None
