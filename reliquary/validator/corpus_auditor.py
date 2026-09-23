"""Re-run the job's model over every accepted submission and record a verdict.

V0 audits everything (q = 1): a submission is paid only once a passing verdict
exists for it. A validator-side error writes no verdict, so the submission stays
pending instead of being charged to a miner for our fault.
"""

from __future__ import annotations

import asyncio
import logging
import time

import torch

from reliquary.corpus.encoding import prompt_token_ids
from reliquary.protocol.profiles import ProofProfile
from reliquary.validator.corpus_audit import audit_completion, completion_hidden_states

logger = logging.getLogger(__name__)

VERDICT_SCHEMA = "reliquary/corpus-verdict/v1"


class CorpusAuditor:
    def __init__(self, *, job_id: str, records, model, tokenizer, proof: ProofProfile) -> None:
        self._job_id = job_id
        self._records = records
        self._model = model
        self._tokenizer = tokenizer
        self._proof = proof
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def enqueue(self, submission_id: str) -> None:
        self._queue.put_nowait(submission_id)

    async def pending_ids(self) -> list[str]:
        submitted = await self._records.list_submission_ids(self._job_id)
        judged = set(await self._records.list_verdict_ids(self._job_id))
        return [sid for sid in submitted if sid not in judged]

    def _judge(self, record: dict) -> dict:
        prompt = prompt_token_ids(self._tokenizer, record["rendered_prompt"])
        worst = {"worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0}
        if not record["completions"]:
            # Fail closed like sequence_verdict does for an empty chunk sequence:
            # no completions must never read as a vacuous pass paid like honest work.
            return {"passed": False, "reason": "no_completions", **worst}
        passed, reason = True, None
        for completion in record["completions"]:
            hidden = completion_hidden_states(
                self._model, prompt + list(completion["tokens"]), len(prompt)
            )
            outcome = audit_completion(hidden, completion["proofs"], self._proof)
            for result in outcome.results:
                worst["worst_exp"] = max(worst["worst_exp"], result.exp_mismatches)
                worst["worst_mant_mean"] = max(worst["worst_mant_mean"], float(result.mant_err_mean))
                worst["worst_mant_median"] = max(worst["worst_mant_median"], float(result.mant_err_median))
            if not outcome.passed and passed:
                passed, reason = False, outcome.reason
        return {"passed": passed, "reason": reason, **worst}

    async def audit(self, submission_id: str) -> dict | None:
        record = await self._records.read_submission(self._job_id, submission_id)
        if record is None:
            logger.error("corpus submission %s has no record", submission_id[:12])
            return None
        try:
            judged = await asyncio.to_thread(self._judge, record)
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # Ours, not the miner's: leave it pending for the next start.
            logger.error("corpus audit of %s failed on the validator: %s", submission_id[:12], exc)
            return None
        verdict = {
            "schema": VERDICT_SCHEMA,
            "submission_id": submission_id,
            "hotkey": record["hotkey"],
            "token_count": int(record["token_count"]),
            "audited_at": time.time(),
            **judged,
        }
        if not await self._records.write_verdict(self._job_id, submission_id, verdict):
            return await self._records.read_verdict(self._job_id, submission_id)
        return verdict

    async def run(self) -> None:
        for submission_id in await self.pending_ids():
            self.enqueue(submission_id)
        while True:
            submission_id = await self._queue.get()
            try:
                await self.audit(submission_id)
            except Exception:
                # A store hiccup (e.g. a transient ConnectionError) must not kill the
                # drain loop: leave the submission pending, pending_ids() picks it up
                # again at the next start, and we keep draining the rest of the queue.
                logger.exception("corpus audit of %s crashed the drain loop", submission_id[:12])
