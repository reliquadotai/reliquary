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
from reliquary.validator.corpus_audit import audit_completion, batch_completion_hidden_states
from reliquary.validator.corpus_text import REASON_TOKEN_OUT_OF_VOCAB

logger = logging.getLogger(__name__)

VERDICT_SCHEMA = "reliquary/corpus-verdict/v1"

RESCAN_SECONDS = 60.0
MAX_CONSECUTIVE_VALIDATOR_ERRORS = 5
# Padded size (rows x longest sequence) a sub-batch's forward pass may reach.
AUDIT_BATCH_TOKENS = 131072
# `run()` drains the queue into groups no larger than this before auditing.
RUN_BATCH_IDS = 16


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

    def _judge_many(self, records: list[dict]) -> list[dict]:
        """Judge several records at once: every completion of every record that
        needs the GPU is packed, sorted by length, into shared forward passes."""
        worst_zero = {"worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0}
        results: list[dict | None] = [None] * len(records)
        prompts: list[list[int] | None] = [None] * len(records)
        vocabulary = self._model.get_input_embeddings().num_embeddings
        for i, record in enumerate(records):
            if not record["completions"]:
                # Fail closed like sequence_verdict does for an empty chunk sequence:
                # no completions must never read as a vacuous pass paid like honest work.
                results[i] = {"passed": False, "reason": "no_completions", **worst_zero}
                continue
            # The miner's fault, not ours: checked before the prefill so it becomes
            # a failed verdict instead of a validator-side error that halts the
            # auditor.
            out_of_vocab = any(
                completion["tokens"]
                and (min(completion["tokens"]) < 0 or max(completion["tokens"]) >= vocabulary)
                for completion in record["completions"]
            )
            if out_of_vocab:
                results[i] = {"passed": False, "reason": REASON_TOKEN_OUT_OF_VOCAB, **worst_zero}
                continue
            prompts[i] = prompt_token_ids(self._tokenizer, record["rendered_prompt"])

        # Every remaining completion, from every record, packed shortest first so
        # padding waste stays low, into sub-batches under the token budget.
        queue = sorted(
            (len(prompts[i]) + len(completion["tokens"]), i, c_idx)
            for i, record in enumerate(records)
            if results[i] is None
            for c_idx, completion in enumerate(record["completions"])
        )
        sub_batches: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        current_width = 0
        for length, i, c_idx in queue:
            width = max(current_width, length)
            if current and (len(current) + 1) * width > AUDIT_BATCH_TOKENS:
                sub_batches.append(current)
                current, width = [], length
            current.append((i, c_idx))
            current_width = width
        if current:
            sub_batches.append(current)

        outcomes = {}
        for sub_batch in sub_batches:
            sequences = [
                (prompts[i] + list(records[i]["completions"][c_idx]["tokens"]), len(prompts[i]))
                for i, c_idx in sub_batch
            ]
            hidden_states = batch_completion_hidden_states(self._model, sequences)
            for (i, c_idx), hidden in zip(sub_batch, hidden_states):
                completion = records[i]["completions"][c_idx]
                outcomes[i, c_idx] = audit_completion(hidden, completion["proofs"], self._proof)
            # Drop this sub-batch's padded activations before the next one is
            # computed: two final-hidden-state tensors must never be live at once.
            del hidden_states, sequences, hidden

        for i, record in enumerate(records):
            if results[i] is not None:
                continue
            passed, reason = True, None
            worst = dict(worst_zero)
            for c_idx in range(len(record["completions"])):
                outcome = outcomes[i, c_idx]
                for result in outcome.results:
                    worst["worst_exp"] = max(worst["worst_exp"], result.exp_mismatches)
                    worst["worst_mant_mean"] = max(worst["worst_mant_mean"], float(result.mant_err_mean))
                    worst["worst_mant_median"] = max(worst["worst_mant_median"], float(result.mant_err_median))
                if not outcome.passed and passed:
                    passed, reason = False, outcome.reason
            results[i] = {"passed": passed, "reason": reason, **worst}
        return results

    async def audit_many(self, submission_ids: list[str]) -> list[dict | None]:
        ids: list[str] = []
        records: list[dict] = []
        for submission_id in submission_ids:
            try:
                record = await self._records.read_submission(self._job_id, submission_id)
            except Exception:
                # Isolated per id: one bad read must not stall the rest of the batch.
                logger.exception("corpus read of %s failed; leaving it pending", submission_id[:12])
                continue
            if record is None:
                logger.error("corpus submission %s has no record", submission_id[:12])
                continue
            ids.append(submission_id)
            records.append(record)
        if not records:
            return []

        batch_failed, batch_error = False, ""
        try:
            judged: list = await asyncio.to_thread(self._judge_many, records)
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # Record only the message here, then leave the block: `exc` and its
            # traceback pin every frame that was live when the batch failed
            # (including this batch's hidden states), and the retries below
            # must not run while any of that memory is still referenced.
            batch_failed, batch_error = True, str(exc)
            del exc

        if batch_failed:
            # Ours, not the miner's: one record's fault must not stall the rest
            # of the batch, so retry each alone before giving up on any of them.
            logger.error(
                "corpus audit batch of %d failed on the validator: %s; retrying one by one",
                len(records), batch_error,
            )
            if torch.cuda.is_available():
                # The failed batch's activations are unreachable now; hand that
                # memory back before the smaller retries ask for their own.
                torch.cuda.empty_cache()
            judged = []
            for record in records:
                try:
                    judged.append((await asyncio.to_thread(self._judge_many, [record]))[0])
                except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as solo_exc:
                    # A message, not the exception object: so this record's
                    # traceback (and whatever activations it pins) cannot
                    # outlive this line, into the next record's retry.
                    judged.append(str(solo_exc))
                    del solo_exc

        results: list[dict | None] = []
        for submission_id, record, outcome in zip(ids, records, judged):
            if not isinstance(outcome, dict):
                # Ours, not the miner's: leave it pending for the next rescan.
                self._validator_errors += 1
                logger.error(
                    "corpus audit of %s failed on the validator: %s", submission_id[:12], outcome
                )
                results.append(None)
                continue
            self._validator_errors = 0
            verdict = {
                "schema": VERDICT_SCHEMA,
                "submission_id": submission_id,
                "hotkey": record["hotkey"],
                "token_count": int(record["token_count"]),
                "audited_at": time.time(),
                **outcome,
            }
            if not await self._records.write_verdict(self._job_id, submission_id, verdict):
                verdict = await self._records.read_verdict(self._job_id, submission_id)
            results.append(verdict)
        return results

    async def audit(self, submission_id: str) -> dict | None:
        results = await self.audit_many([submission_id])
        return results[0] if results else None

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
                batch = [await self._queue.get()]
                while len(batch) < RUN_BATCH_IDS:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                try:
                    await self.audit_many(batch)
                except Exception:
                    # A store hiccup (e.g. a transient ConnectionError) must not kill
                    # the drain loop: the submissions stay pending and the next
                    # rescan queues them again.
                    logger.exception("corpus audit of a batch crashed the drain loop")
                finally:
                    for submission_id in batch:
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
