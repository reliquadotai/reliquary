"""Judge every accepted submission: audit it with the job's model, wait out its
hold, or pass it unaudited, and record a verdict.

With `audit_q = 1` (the default) every record is audited on arrival, as in V0.
A submission is paid only once a passing verdict exists for it. A
validator-side error writes no verdict, so the submission stays pending instead
of being charged to a miner for our fault.
"""

from __future__ import annotations

import asyncio
import bisect
from dataclasses import replace
import logging
import re
import time
from collections.abc import Callable

import torch

from reliquary.corpus.audit_policy import (
    PASS_IDS,
    AuditParams,
    MinerState,
    after_confirmed_failure,
    after_pass,
    decision,
    drawn,
    effective_state,
)
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
# Propagation slack after the draw round's publication before it is fetched:
# asking too early reads as "no beacon", which audits (safe, but wastes the sampling).
BEACON_GRACE_SECONDS = 2.0
# How long a round that just failed to fetch is left unfetched before the next
# attempt: every sampled submission whose draw lands on a bad round would
# otherwise refetch it once per judging pass.
NEGATIVE_BEACON_CACHE_SECONDS = 30.0
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WORST_ZERO = {"worst_exp": 0, "worst_mant_mean": 0.0, "worst_mant_median": 0.0}
_BANNED_VOID = {"passed": False, "audited": False, "reason": "banned"}


class CorpusAuditorHalted(Exception):
    """Too many audits in a row failed on our side: this validator cannot audit."""


class CorpusAuditor:
    def __init__(self, *, job_id: str, records, model, tokenizer, proof: ProofProfile,
                 rescan_every_seconds: float = RESCAN_SECONDS,
                 max_validator_errors: int = MAX_CONSECUTIVE_VALIDATOR_ERRORS,
                 params: AuditParams = AuditParams(), miner_states=None,
                 beacon: Callable[[int], str | None] | None = None,
                 round_at: Callable[[float], int] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
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
        self._params = params
        self._miner_states = miner_states
        self._beacon = beacon
        self._round_at = round_at
        self._clock = clock
        # Records are immutable: (hotkey, received_at, token_count) read once,
        # so a rescan every minute does not re-read every record from the store.
        self._meta: dict[str, tuple[str, float, int]] = {}
        # Per hotkey, the sorted arrival times still inside the hold window;
        # `recent_submissions` is counted from here, never from a job listing.
        self._arrivals: dict[str, list[float]] = {}
        # Pending records are read once per process to seed `_arrivals`; judged
        # ones are not re-read, so `recent` can only undercount (more audits).
        self._seeded = False
        self._randomness: dict[int, str] = {}
        # Round -> when its fetch last failed; a negative cache, so a bad
        # round is retried at most once every NEGATIVE_BEACON_CACHE_SECONDS.
        self._failed_rounds: dict[int, float] = {}
        # Ids whose verdict stands (written, found, or listed): never judged again.
        self._judged: set[str] = set()
        # Per hotkey, the ids read but not judged yet: the siblings an unaudited
        # pass must wait for (queue lag can exceed the hold).
        self._unjudged: dict[str, set[str]] = {}
        # Pending ids whose read failed, until one succeeds or a verdict stands:
        # their hotkey is unknown, so every unaudited pass waits for them.
        self._unreadable: set[str] = set()

    def enqueue(self, submission_id: str) -> None:
        if submission_id in self._queued:
            return
        self._queued.add(submission_id)
        self._queue.put_nowait(submission_id)

    async def pending_ids(self) -> list[str]:
        submitted = await self._records.list_submission_ids(self._job_id)
        judged = set(await self._records.list_verdict_ids(self._job_id))
        for sid in judged - self._judged:
            self._mark_judged(sid)
        return [sid for sid in submitted if sid not in judged]

    def _mark_judged(self, submission_id: str) -> None:
        self._judged.add(submission_id)
        self._unreadable.discard(submission_id)
        if submission_id in self._meta:
            self._unjudged.get(self._meta[submission_id][0], set()).discard(submission_id)

    def queue_lag(self, pending: list[str]) -> float | None:
        """Seconds since the oldest pending record we have read was received."""
        times = [self._meta[sid][1] for sid in pending if sid in self._meta]
        return self._clock() - min(times) if times else None

    def _judge_many(self, records: list[dict]) -> list[dict]:
        """Judge several records at once: every completion of every record that
        needs the GPU is packed, sorted by length, into shared forward passes."""
        worst_zero = _WORST_ZERO
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

    async def _read(self, submission_id: str) -> dict | None:
        try:
            record = await self._records.read_submission(self._job_id, submission_id)
        except Exception:
            # Isolated per id: one bad read must not stall the rest of the batch.
            logger.exception("corpus read of %s failed; leaving it pending", submission_id[:12])
            self._unreadable.add(submission_id)
            return None
        if record is None:
            logger.error("corpus submission %s has no record", submission_id[:12])
            self._unreadable.add(submission_id)
            return None
        self._unreadable.discard(submission_id)
        if submission_id not in self._meta:
            received = record.get("received_at")
            # An older record carries no arrival time: its hold counts from when
            # we first saw it, never as already over.
            received_at = float(received) if received is not None else self._clock()
            self._meta[submission_id] = (
                record["hotkey"], received_at, int(record["token_count"]),
            )
            bisect.insort(self._arrivals.setdefault(record["hotkey"], []), received_at)
            if submission_id not in self._judged:
                self._unjudged.setdefault(record["hotkey"], set()).add(submission_id)
        return record

    def _recent(self, hotkey: str, now: float) -> int:
        arrivals = self._arrivals.get(hotkey, [])
        # Older than the hold window: never counted again, as `now` only grows.
        del arrivals[:bisect.bisect_left(arrivals, now - self._params.hold_seconds)]
        return bisect.bisect_right(arrivals, now)

    async def _audit_outcomes(self, records: list[dict]) -> list[dict | str]:
        """One outcome per record; a string is a validator-side error message."""
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
        return judged

    def _verdict(self, submission_id: str, hotkey: str, token_count: int, outcome: dict,
                 draw: dict | None) -> dict:
        verdict = {
            "schema": VERDICT_SCHEMA,
            "submission_id": submission_id,
            "hotkey": hotkey,
            "token_count": int(token_count),
            "audited_at": self._clock(),
            **outcome,
        }
        if draw is not None:
            verdict["draw"] = draw
        return verdict

    async def _write(self, submission_id: str, verdict: dict) -> tuple[dict, bool]:
        """Create-only: the verdict that stands, and whether this call wrote it."""
        if await self._records.write_verdict(self._job_id, submission_id, verdict):
            self._mark_judged(submission_id)
            return verdict, True
        standing = await self._records.read_verdict(self._job_id, submission_id)
        self._mark_judged(submission_id)
        return standing, False

    async def _state(self, hotkey: str, now: float) -> MinerState:
        if self._miner_states is None:
            return MinerState()
        state = await self._miner_states.get(hotkey)
        if state.banned_until is not None and now >= state.banned_until:
            # Persist the end of a ban as a fresh probation (§7.3), so passes
            # counted before or during the ban never shorten it.
            def end_ban(m: MinerState) -> MinerState:
                if m.banned_until is not None and now >= m.banned_until:
                    return replace(m, banned_until=None, audited_passed=0)
                return m

            state = await self._miner_states.update(hotkey, end_ban)
        return state

    async def audit_many(self, submission_ids: list[str],
                         draws: dict[str, dict] | None = None) -> list[dict | None]:
        ids: list[str] = []
        records: list[dict] = []
        for submission_id in submission_ids:
            record = await self._read(submission_id)
            if record is not None:
                ids.append(submission_id)
                records.append(record)
        if not records:
            return []
        results, _ = await self._audit_records(ids, records, draws or {})
        return results

    async def _audit_records(self, ids: list[str], records: list[dict],
                             draws: dict[str, dict]) -> tuple[list[dict | None], set[str]]:
        """Audit, re-audit each failure alone, write the verdicts and move each
        hotkey's state. Returns the verdicts and the hotkeys with a confirmed failure."""
        outcomes = await self._audit_outcomes(records)
        for k, outcome in enumerate(outcomes):
            if isinstance(outcome, dict) and not outcome["passed"]:
                # §7.2: only a failure that a second, separate audit repeats counts.
                outcomes[k] = (await self._audit_outcomes([records[k]]))[0]

        for submission_id, outcome in zip(ids, outcomes):
            if isinstance(outcome, dict):
                self._validator_errors = 0
            else:
                # Ours, not the miner's: leave it pending for the next rescan.
                self._validator_errors += 1
                logger.error(
                    "corpus audit of %s failed on the validator: %s", submission_id[:12], outcome
                )

        results: list[dict | None] = [None] * len(ids)
        judged = [k for k, outcome in enumerate(outcomes) if isinstance(outcome, dict)]
        now = self._clock()
        failures: dict[str, list[str]] = {}
        for k in judged:
            if not outcomes[k]["passed"]:
                failures.setdefault(records[k]["hotkey"], []).append(ids[k])
        failed_hotkeys = set(failures)
        states: dict[str, MinerState] = {}
        if failures and self._miner_states is not None:
            # Escalate before the verdicts exist: a crash in between then
            # re-audits the records, and the retry counts nothing twice
            # (idempotent by submission id), instead of leaving a caught
            # cheater unsuspected while its held records are paid. One write
            # for the whole batch: miners.json is one key, rate-limited.
            def escalate(sids: list[str]) -> Callable[[MinerState], MinerState]:
                def change(m: MinerState) -> MinerState:
                    for sid in sids:
                        m = after_confirmed_failure(m, self._params, now, sid)
                    return m
                return change

            states.update(await self._miner_states.update_many(
                {hotkey: escalate(sids) for hotkey, sids in failures.items()}))

        # Failures first: a ban they cause must void this batch's passes of that hotkey.
        passes: dict[str, list[float]] = {}
        for k in sorted(judged, key=lambda k: outcomes[k]["passed"]):
            submission_id, record = ids[k], records[k]
            hotkey = record["hotkey"]
            outcome = {**outcomes[k], "audited": True}
            if outcome["passed"]:
                if hotkey not in states:
                    states[hotkey] = await self._state(hotkey, now)
                if effective_state(states[hotkey], now, self._params) == "banned":
                    outcome = dict(_BANNED_VOID)
            verdict = self._verdict(submission_id, hotkey, record["token_count"], outcome,
                                    draws.get(submission_id) if outcome["audited"] else None)
            results[k], written = await self._write(submission_id, verdict)
            # Only the call that wrote a passing verdict counts it: a repeat audit
            # (stale queue entry, restart) must never count twice.
            if written and outcome["passed"] and outcome["audited"]:
                passes.setdefault(hotkey, []).append((submission_id, outcome["worst_mant_mean"]))

        def count(batch: list[tuple[str, float]]) -> Callable[[MinerState], MinerState]:
            def change(m: MinerState) -> MinerState:
                for sid, mant_mean in batch:
                    if effective_state(m, now, self._params) == "banned":
                        return m
                    m = after_pass(m, self._params, mant_mean, sid)
                return m
            return change

        # At most PASS_IDS per hotkey per write: a retry of a write that landed
        # finds every one of its ids still in pass_ids and counts none twice.
        while passes and self._miner_states is not None:
            chunk = {hotkey: batch[:PASS_IDS] for hotkey, batch in passes.items()}
            passes = {hotkey: batch[PASS_IDS:] for hotkey, batch in passes.items()
                      if batch[PASS_IDS:]}
            await self._miner_states.update_many(
                {hotkey: count(batch) for hotkey, batch in chunk.items()})
        return results, failed_hotkeys

    async def _randomness_for(self, round_number: int) -> str | None:
        """The drand randomness of a round, lowercased, or None (fetch error,
        malformed): the caller then audits. A round that just failed is not
        refetched for NEGATIVE_BEACON_CACHE_SECONDS -- every sampled
        submission whose draw lands on that round would otherwise repeat the
        same failing network call, one per judging pass."""
        randomness = self._randomness.get(round_number)
        if randomness is not None:
            return randomness
        failed_at = self._failed_rounds.get(round_number)
        if failed_at is not None and self._clock() - failed_at < NEGATIVE_BEACON_CACHE_SECONDS:
            return None
        try:
            value = await asyncio.to_thread(self._beacon, round_number)
        except Exception:
            logger.warning("drand round %d unavailable; auditing", round_number, exc_info=True)
            value = None
        if isinstance(value, str) and _HEX64.fullmatch(value.lower()):
            randomness = self._randomness[round_number] = value.lower()
            self._failed_rounds.pop(round_number, None)
        else:
            if value is not None:
                logger.error("drand round %d gave malformed randomness %r; auditing",
                             round_number, value)
            self._failed_rounds[round_number] = self._clock()
        return randomness

    async def _decide(self, submission_id: str, now: float,
                      state: MinerState) -> tuple[str, dict | None]:
        """decision() for one known record, with its draw; "undecidable" while
        its draw round is not out yet (the rescan comes back for it)."""
        hotkey, received_at, _ = self._meta[submission_id]
        randomness, draw, recent = None, None, 0
        if self._params.q < 1.0 and effective_state(state, now, self._params) == "sampled":
            if not self._seeded:
                for sid in await self.pending_ids():
                    if sid not in self._meta:
                        await self._read(sid)
                self._seeded = True
            recent = self._recent(hotkey, now)
            if (recent >= 1.0 / self._params.q and self._beacon is not None
                    and self._round_at is not None):
                # round_at(t) is the first round published strictly after t;
                # one more round keeps the miner signing before its
                # randomness exists even with our clock a period behind (§6).
                # It may raise -- the drand chain's genesis/period can still
                # be unresolved (a lazy `round_at` retries on its own
                # schedule) -- caught here rather than propagated, so an
                # unresolved chain audits this submission (randomness stays
                # None below, decision() then reads that as "audit") instead
                # of crashing the whole batch out of the drain loop.
                try:
                    round_number = int(self._round_at(received_at)) + 1
                    round_not_out_yet = int(self._round_at(now - BEACON_GRACE_SECONDS)) <= round_number
                except Exception:
                    logger.warning(
                        "round_at unavailable for %s; auditing", submission_id[:12],
                        exc_info=True,
                    )
                else:
                    if round_not_out_yet:
                        return "undecidable", None
                    randomness = await self._randomness_for(round_number)
                    if randomness is not None:
                        draw = {"round": round_number, "q": self._params.q,
                                "drawn": drawn(randomness, submission_id, self._params.q)}
        choice = decision(state, params=self._params, now=now, received_at=received_at,
                          recent_submissions=recent, randomness_hex=randomness,
                          submission_id=submission_id)
        return choice, draw

    async def _judge_once(self, submission_ids: list[str]) -> set[str]:
        now = self._clock()
        # The backward audit and the rescan bring a caught hotkey's records back.
        submission_ids = [sid for sid in submission_ids if sid not in self._judged]
        read: dict[str, dict] = {}
        for submission_id in submission_ids:
            if submission_id not in self._meta:
                record = await self._read(submission_id)
                if record is not None:
                    read[submission_id] = record
        known = [sid for sid in dict.fromkeys(submission_ids) if sid in self._meta]
        states = {}
        for hotkey in {self._meta[sid][0] for sid in known}:
            states[hotkey] = await self._state(hotkey, now)

        audit_ids, draws, unaudited, voided = [], {}, [], []
        # Per hotkey, arrival times of records whose draw round is not out yet.
        undecided: dict[str, list[float]] = {}
        for submission_id in known:
            hotkey, received_at, _ = self._meta[submission_id]
            choice, draw = await self._decide(submission_id, now, states[hotkey])
            if choice == "audit":
                audit_ids.append(submission_id)
                if draw is not None:
                    draws[submission_id] = draw
            elif choice == "pass_unaudited":
                unaudited.append((submission_id, draw))
            elif choice == "void_banned":
                voided.append(submission_id)
            elif choice == "undecidable":
                undecided.setdefault(hotkey, []).append(received_at)

        if unaudited:
            # Queue lag can exceed the hold: a drawn sibling received within X's
            # hold may still sit in the queue. Decide every such sibling now and
            # audit the drawn ones in this pass, so a failure among them reaches
            # X through the same-pass guard below instead of after X is paid.
            for sid in sorted(self._queued | self._unreadable):
                if sid not in self._meta and sid not in self._judged:
                    await self._read(sid)
            hold_end: dict[str, float] = {}
            for submission_id, _ in unaudited:
                hotkey, received_at, _ = self._meta[submission_id]
                hold_end[hotkey] = max(hold_end.get(hotkey, 0.0),
                                       received_at + self._params.hold_seconds)
            in_pass = set(known)
            for hotkey, until in hold_end.items():
                for sid in sorted(self._unjudged.get(hotkey, ())):
                    if sid in in_pass or self._meta[sid][1] > until:
                        continue
                    choice, draw = await self._decide(sid, now, states[hotkey])
                    if choice == "audit":
                        audit_ids.append(sid)
                        if draw is not None:
                            draws[sid] = draw
                    elif choice == "undecidable":
                        undecided.setdefault(hotkey, []).append(self._meta[sid][1])

        failed: set[str] = set()
        errored: set[str] = set()
        if audit_ids:
            records = [read.get(sid) or await self._read(sid) for sid in audit_ids]
            pairs = [(sid, r) for sid, r in zip(audit_ids, records) if r is not None]
            errored = {self._meta[sid][0] for sid, r in zip(audit_ids, records) if r is None}
            if pairs:
                results, failed = await self._audit_records(
                    [sid for sid, _ in pairs], [r for _, r in pairs], draws)
                errored |= {r["hotkey"] for (_, r), result in zip(pairs, results) if result is None}
        unreadable = bool(self._unreadable)
        if unaudited and unreadable:
            logger.error(
                "%d pending corpus record(s) unreadable (e.g. %s); every unaudited pass "
                "waits until they read or get a verdict",
                len(self._unreadable), min(self._unreadable)[:12])
        for submission_id, draw in unaudited:
            hotkey, received_at, token_count = self._meta[submission_id]
            if hotkey in failed:
                continue  # now suspect: the backward audit decides it
            # Wait while a sibling that could still catch this record is
            # undecided or could not be audited this pass.
            if unreadable or hotkey in errored or any(
                    t <= received_at + self._params.hold_seconds
                    for t in undecided.get(hotkey, ())):
                continue
            await self._write(submission_id, self._verdict(
                submission_id, hotkey, token_count,
                {"passed": True, "audited": False, "reason": None, **_WORST_ZERO}, draw))
        for submission_id in voided:
            hotkey, _, token_count = self._meta[submission_id]
            await self._write(submission_id, self._verdict(
                submission_id, hotkey, token_count, dict(_BANNED_VOID), None))
        return failed

    async def judge_many(self, submission_ids: list[str]) -> None:
        """Decide each record: audit now, wait out its hold, pass it unaudited, or
        void it for a ban; then audit backwards after every confirmed failure."""
        failed = await self._judge_once(list(submission_ids))
        # At q = 1 every held record is already being audited on arrival.
        while failed and self._params.q < 1.0:
            # §7.2: every record of a hotkey just found cheating that has no
            # verdict yet is audited (it is suspect now) before it can be paid.
            pending = await self.pending_ids()
            for sid in pending:
                if sid not in self._meta:
                    await self._read(sid)
            held = [sid for sid in pending if sid in self._meta and self._meta[sid][0] in failed]
            failed = await self._judge_once(held)

    async def audit(self, submission_id: str) -> dict | None:
        results = await self.audit_many([submission_id])
        return results[0] if results else None

    async def _rescan_forever(self) -> None:
        # The only retry for an id whose audit failed (store read or our own
        # error): without it, it would wait for the next process start.
        while True:
            await asyncio.sleep(self._rescan_every)
            try:
                pending = await self.pending_ids()
                for submission_id in pending:
                    self.enqueue(submission_id)
                lag = self.queue_lag(pending)
                # An undrawn record waits one hold by design; far beyond that,
                # the auditor is not keeping up with the traffic.
                level = (logging.WARNING if lag is not None
                         and lag > self._params.hold_seconds + 2 * self._rescan_every
                         else logging.INFO)
                logger.log(level, "corpus audit queue lag: %d pending, oldest received %s s ago",
                           len(pending), "-" if lag is None else f"{lag:.0f}")
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
                    await self.judge_many(batch)
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
