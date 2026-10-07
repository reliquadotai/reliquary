"""Recovery and fencing around the existing executor decision paths."""

from __future__ import annotations

import asyncio
import copy
import math
from dataclasses import asdict

from reliquary.infrastructure.corpus_attempt_store import AttemptRefused, digest


class DurableAttempts:
    def _init_attempts(self, store):
        self._attempt_store = store
        self._attempt_lock = asyncio.Lock()
        self._attempt_work = {}
        self._pending_finishes = {}

    def _credential(self, executor_id):
        document = self._directory.document(executor_id)
        if not self._directory.is_authorized(executor_id):
            raise AttemptRefused(403, "executor_binding_changed")
        return digest({k: document.get(k) for k in ("executor_id", "token_sha256", "scope",
                       "model_id", "model_revision", "provider_id", "expires_at")})

    def _work_digest(self, work):
        return digest(work.items if hasattr(work, "items") else work.item)

    def _snapshot(self, work):
        if hasattr(work, "items"):
            state = {"schema": "executor-queue-state/v1", "attempts": work.attempts,
                     "recheck_drawn": getattr(work, "recheck_drawn", None)}
            if work.future.done() and not work.future.cancelled() and work.future.exception() is None:
                scores, executor = work.future.result()
                state["decision"] = {"executor": executor, "scores": [
                    {"status": status, "chunks": [[c.exp_mismatches, c.mant_err_mean, c.mant_err_median]
                                                   for c in chunks]} for status, chunks in scores]}
        else:
            state = {k: getattr(work, k) for k in ("results", "providers", "failed_at", "drawn",
                                                  "timeouts", "errors", "unserved", "swept_at")}
            state["excluded"] = sorted(work.excluded)
            state["vote_bindings"] = {eid: getattr(work, "vote_bindings", {}).get(eid)
                                      for eid in work.results}
            state["schema"] = "executor-queue-state/v1"
            if work.future.done() and not work.future.cancelled() and work.future.exception() is None:
                state["decision"] = asdict(work.future.result())
        return copy.deepcopy(state)

    def _restore(self, work, state):
        if state.get("schema") != "executor-queue-state/v1":
            raise ValueError("attempt queue state version is unsupported")
        if hasattr(work, "items"):
            if (type(state.get("attempts")) is not int or not 0 <= state["attempts"] <= 2
                    or (state.get("recheck_drawn") is not None and type(state["recheck_drawn"]) is not bool)):
                raise ValueError("audit attempt counters are invalid")
            work.attempts = int(state.get("attempts", 0))
            work.recheck_drawn = state.get("recheck_drawn")
            decision = state.get("decision")
            if decision is not None:
                from reliquary.protocol.toploc import ChunkResult
                facts = self._result_model.model_validate({"scores": decision["scores"]})
                if len(facts.scores) != len(work.items):
                    raise ValueError("audit decision does not fit its input")
                self._resolve(work, [(s.status, tuple(ChunkResult(*c) for c in s.chunks))
                                     for s in facts.scores], decision["executor"])
        else:
            if (type(state.get("errors")) is not int or not 0 <= state["errors"] <= 3
                    or type(state.get("timeouts")) is not int or not 0 <= state["timeouts"] <= 2
                    or not isinstance(state.get("results"), dict) or len(state["results"]) > 3
                    or not math.isfinite(float(state.get("unserved", float("nan"))))
                    or state["unserved"] < 0):
                raise ValueError("grade attempt counters are invalid")
            for key in ("results", "providers", "failed_at", "drawn", "timeouts", "errors",
                        "unserved", "swept_at"):
                if key in state:
                    setattr(work, key, copy.deepcopy(state[key]))
            work.excluded = set(state.get("excluded", []))
            work.vote_bindings = dict(state.get("vote_bindings", {}))
            removed = set()
            for executor in list(work.results):
                answer = self._result_model.model_validate({"results": [work.results[executor]]}).results[0]
                if self._misfit(work, answer) is not None:
                    raise ValueError("grade snapshot does not fit its input")
                try:
                    valid = (work.vote_bindings.get(executor) == self._credential(executor)
                             and work.providers.get(executor) == self._provider(executor))
                except AttemptRefused:
                    valid = False
                if not valid:
                    removed.add(executor)
                    work.results.pop(executor, None)
                    work.providers.pop(executor, None)
                    work.vote_bindings.pop(executor, None)
                    work.excluded.discard(executor)
            decision = state.get("decision")
            if decision is not None and not removed.intersection(decision["graded_by"]):
                from reliquary.validator.corpus_grade_remote import GradeDecision
                self._resolve(work, GradeDecision(decision["status"], decision["result"],
                              tuple(decision["graded_by"]), tuple(decision["providers"])))

    def _attach(self, work, head):
        if hasattr(work, "items"):
            from reliquary.validator.corpus_audit_remote import _Lease
            lease = _Lease(head["lease_id"], work, head["executor_id"], head["expires_at"])
        else:
            from reliquary.validator.corpus_grade_remote import _Lease
            lease = _Lease(head["lease_id"], work, head["executor_id"], head["expires_at"], head["started_at"])
        work.attempt_generation = head["generation"]
        work.attempt_lease = head["lease_id"]
        self._leases[lease.lease_id] = lease
        return lease

    async def _prepare_work(self, work):
        if self._attempt_store is None:
            return False
        work.attempt_key = self._work_digest(work)
        previous = self._attempt_work.get(work.attempt_key)
        if previous is not None and previous is not work:
            work.future = previous.future
            return True
        self._attempt_work[work.attempt_key] = work
        work.future.add_done_callback(lambda _: self._forget_work(work))
        head = await self._attempt_store.read(work.attempt_key)
        if head is None:
            return False
        work.attempt_generation = head["generation"]
        work.attempt_lease = head["lease_id"]
        try:
            credential = self._credential(head["executor_id"])
        except AttemptRefused:
            credential = None
        if credential != head["credential_sha256"]:
            state = {k: v for k, v in head["snapshot"].items() if k != "decision"}
            self._restore(work, state)
            await self._attempt_store.finish(work.attempt_key, head["lease_id"], head["generation"],
                self._snapshot(work), "invalidated", error=[403, "executor_binding_changed"])
            return False
        self._restore(work, head["snapshot"])
        if work.future.done():
            return True
        if head["status"] == "leased":
            lease = self._attach(work, head)
            if lease.expires_at <= self._clock():
                del self._leases[lease.lease_id]
                self._take_back(lease, expired=True)
                self._attempt_changed(work, "expired", error=[410, "lease_expired"])
                await self._flush_attempts()
            else:
                self.stats["attempts_recovered"] += 1
            return True
        # A completed grading snapshot already includes its vote or retry
        # counters. Only a submitted vote crosses the apply crash gap. Audit
        # acceptance can still await the trusted local recheck after the ACK.
        pending = head["status"] == "submitted" or (
            hasattr(work, "items") and head.get("outcome") == "accepted")
        if pending and head.get("result") is not None and head.get("error") is None:
            lease = self._attach(work, head)
            # This result was reserved before expiry. Recovery does not grant
            # its executor a new lease; it applies the already received facts.
            lease.expires_at = max(lease.expires_at, self._clock() + 1)
            body = self._result_model.model_validate(head["result"])
            if not hasattr(work, "items"):
                work.vote_bindings = {**getattr(work, "vote_bindings", {}),
                                      head["executor_id"]: credential}
            from reliquary.validator.corpus_audit_remote import LeaseRefused
            try:
                outcome = self.result(head["executor_id"], head["lease_id"], body)
                error = None
            except LeaseRefused as exc:
                outcome, error = "refused", [exc.status, exc.detail]
            await self._attempt_store.finish(work.attempt_key, head["lease_id"], head["generation"],
                                             self._snapshot(work), outcome, error=error)
            self.stats["results_recovered"] += 1
            return True
        if hasattr(work, "items") and work.attempts >= 2:
            self._local.append(work)
            return True
        return False

    async def prepare_work(self, work):
        if self._attempt_store is None:
            return False
        async with self._attempt_lock:
            await self._flush_attempts()
            return await self._prepare_work(work)

    def _forget_work(self, work):
        if self._attempt_work.get(work.attempt_key) is work:
            self._attempt_work.pop(work.attempt_key, None)

    def _attempt_changed(self, work, outcome, *, error=None):
        if self._attempt_store is not None and hasattr(work, "attempt_generation"):
            self._pending_finishes[work.attempt_key] = (
                work.attempt_lease, work.attempt_generation, self._snapshot(work), outcome, error)

    async def _flush_attempts(self):
        for key, pending in list(self._pending_finishes.items()):
            lease_id, generation, snapshot, outcome, error = pending
            try:
                await self._attempt_store.finish(key, lease_id, generation, snapshot, outcome, error=error)
            except AttemptRefused as exc:
                if exc.status != 410:
                    raise
            if self._pending_finishes.get(key) is pending:
                self._pending_finishes.pop(key, None)

    async def durable_claim(self, executor_id, *args):
        if self._attempt_store is None:
            return self.claim(executor_id, *args)
        async with self._attempt_lock:
            await self._flush_attempts()
            await self._recover_uncertain()
            document = self.claim(executor_id, *args)
            if document is None:
                return None
            lease = self._leases[document["lease_id"]]
            try:
                head = await self._attempt_store.claim(lease.work.attempt_key,
                    {"lease_id": lease.lease_id, "executor_id": executor_id, "expires_at": lease.expires_at},
                    self._credential(executor_id), self._snapshot(lease.work), now=self._clock())
            except BaseException:
                self._leases.pop(lease.lease_id, None)
                lease.work.attempt_uncertain = True
                if lease.work not in self._queue:
                    self._queue.appendleft(lease.work)
                raise
            if head is None:
                self._leases.pop(lease.lease_id, None)
                self.stats["leased"] -= 1
                if not await self._prepare_work(lease.work):
                    self._queue.appendleft(lease.work)
                return None
            lease.work.attempt_generation = head["generation"]
            lease.work.attempt_lease = head["lease_id"]
            return document

    async def _recover_uncertain(self):
        for work in list(self._queue):
            if not getattr(work, "attempt_uncertain", False):
                continue
            if await self._prepare_work(work):
                # Recovery can requeue an expired lease itself.
                if any(lease.work is work for lease in self._leases.values()) or work.future.done():
                    if work in self._queue:
                        self._queue.remove(work)
            work.attempt_uncertain = False

    async def recover_uncertain(self):
        async with self._attempt_lock:
            await self._recover_uncertain()

    def _prepare_result_draw(self, work, body):
        if hasattr(work, "items"):
            if getattr(work, "recheck_drawn", None) is None:
                work.recheck_drawn = self._rng.random() < self._fraction
        elif work.drawn is None:
            answer = body.results[0]
            if answer.status not in {"error", "timeout"} and self._misfit(work, answer) is None:
                work.drawn = self._recheck_possible() and self._rng.random() < self._fraction

    async def durable_result(self, executor_id, lease_id, body):
        if self._attempt_store is None:
            return self.result(executor_id, lease_id, body)
        async with self._attempt_lock:
            await self._flush_attempts()
            credential = self._credential(executor_id)
            lease = self._leases.get(lease_id)
            if lease is None:
                head = await self._attempt_store.lease(lease_id)
                if head["executor_id"] != executor_id or head["credential_sha256"] != credential:
                    raise AttemptRefused(403, "executor_binding_changed")
                if head["status"] == "completed":
                    if head.get("result_sha256") is None and head.get("error") is not None:
                        raise AttemptRefused(*head["error"])
                    if head.get("result_sha256") != digest(body.model_dump()):
                        raise AttemptRefused(409, "attempt_result_conflict")
                    if head.get("error") is not None:
                        raise AttemptRefused(*head["error"])
                    return head["outcome"]
                if head["expires_at"] <= self._clock():
                    raise AttemptRefused(410, "lease_expired")
                raise AttemptRefused(503, "attempt_recovering")
            work = lease.work
            if self._work_digest(work) != work.attempt_key:
                raise AttemptRefused(410, "work_binding_changed")
            if not hasattr(work, "items"):
                work.vote_bindings = {**getattr(work, "vote_bindings", {}), executor_id: credential}
            self._prepare_result_draw(work, body)
            head = await self._attempt_store.reserve(work.attempt_key, lease_id, executor_id,
                       credential, body.model_dump(), self._snapshot(work), now=self._clock())
            if head["generation"] != work.attempt_generation:
                raise AttemptRefused(410, "lease_superseded")
            from reliquary.validator.corpus_audit_remote import LeaseRefused
            try:
                outcome = self.result(executor_id, lease_id, body)
            except LeaseRefused as exc:
                self._attempt_changed(work, "refused", error=[exc.status, exc.detail])
                await self._flush_attempts()
                raise
            self._attempt_changed(work, outcome)
            await self._flush_attempts()
            return outcome
