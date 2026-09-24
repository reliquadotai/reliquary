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
from reliquary.validator.corpus_text import REASON_TOKEN_OUT_OF_VOCAB

logger = logging.getLogger(__name__)

VERDICT_SCHEMA = "reliquary/corpus-verdict/v1"

RESCAN_SECONDS = 60.0
MAX_CONSECUTIVE_VALIDATOR_ERRORS = 5


class CorpusAuditorHalted(Exception):
    """Too many audits in a row failed on our side: this validator cannot audit."""


class CorpusAuditor:
    def __init__(self, *, job_id: str, records, model, tokenizer, proof: ProofProfile,
                 rescan_every_seconds: float = RESCAN_SECONDS,
                 max_validator_errors: int = MAX_CONSECUTIVE_VALIDATOR_ERRORS) -> None:
        self._job_id = job_id
        self._records = records
        self._model = model
        self._tokenizer = tokenizer
        self._proof = proof
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        # Queued or in flight: a rescan must not audit the same id twice at once.
        self._queued: set[str] = set()
        self._rescan_every = rescan_every_seconds
        self._max_validator_errors = max_validator_errors
        self._validator_errors = 0

    def enqueue(self, submission_id: str) -> None:
        if submission_id in self._queued:
            return
        self._queued.add(submission_id)
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
        # The miner's fault, not ours: checked before the prefill so it becomes a
        # failed verdict instead of a validator-side error that halts the auditor.
        vocabulary = self._model.get_input_embeddings().num_embeddings
        for completion in record["completions"]:
            tokens = completion["tokens"]
            if tokens and (min(tokens) < 0 or max(tokens) >= vocabulary):
                return {"passed": False, "reason": REASON_TOKEN_OUT_OF_VOCAB, **worst}
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
            # Ours, not the miner's: leave it pending for the next rescan.
            self._validator_errors += 1
            logger.error("corpus audit of %s failed on the validator: %s", submission_id[:12], exc)
            return None
        self._validator_errors = 0
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

    async def _rescan_forever(self) -> None:
        # The only retry for an id whose audit failed (store read or our own
        # error): without it, it would wait for the next process start.
        while True:
            await asyncio.sleep(self._rescan_every)
            try:
                for submission_id in await self.pending_ids():
                    self.enqueue(submission_id)
            except Exception:
                logger.exception("corpus pending rescan failed; retrying next period")

    async def run(self) -> None:
        for submission_id in await self.pending_ids():
            self.enqueue(submission_id)
        rescan = asyncio.create_task(self._rescan_forever())
        try:
            while True:
                submission_id = await self._queue.get()
                try:
                    await self.audit(submission_id)
                except Exception:
                    # A store hiccup (e.g. a transient ConnectionError) must not kill
                    # the drain loop: the submission stays pending and the next
                    # rescan queues it again.
                    logger.exception("corpus audit of %s crashed the drain loop", submission_id[:12])
                finally:
                    self._queued.discard(submission_id)
                if self._validator_errors >= self._max_validator_errors:
                    # Spec §6: a validator-side fault stops the worker loudly. A
                    # process that looks healthy while paying nobody is worse.
                    logger.critical(
                        "corpus audit failed on the validator %d times in a row; stopping",
                        self._validator_errors,
                    )
                    raise CorpusAuditorHalted(
                        f"{self._validator_errors} consecutive validator-side audit errors"
                    )
        finally:
            rescan.cancel()
