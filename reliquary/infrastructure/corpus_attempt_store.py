"""Conditional ownership of actual executor work; payloads remain in the record store."""

from __future__ import annotations

import hashlib
import json
import math
import re

from reliquary.infrastructure.corpus_executor_store import _get, _put

SCHEMA = "executor-attempt/v1"
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_HISTORY = 32
_HEX = re.compile(r"^[0-9a-f]{64}$")
_LEASE = re.compile(r"^[0-9a-f]{32}$")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class AttemptRefused(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class AttemptStore:
    """One CAS head per immutable work digest, with monotonic lease generations.

    Result reservation precedes application. A submitted result is replayable
    after a crash; completed snapshots describe native decisions, not settlement.
    The existing verdict/settlement stores remain the final authority.
    """

    def __init__(self, binding: dict, **client_kwargs):
        self.binding = digest(binding)
        self._kwargs = client_kwargs
        self._prefix = f"reliquary/corpus/attempts/{self.binding}/"

    def _key(self, work: str) -> str:
        if not _HEX.fullmatch(work):
            raise ValueError("work digest must be 64 hex characters")
        return self._prefix + work + ".json"

    def _index(self, lease_id: str) -> str:
        if not _LEASE.fullmatch(lease_id):
            raise AttemptRefused(410, "lease_unknown")
        return self._prefix + "leases/" + lease_id + ".json"

    async def read(self, work: str):
        head, _ = await _get(self._key(work), **dict(self._kwargs))
        if head is not None:
            self._validate(head, work)
        return head

    def _validate(self, head: dict, work: str):
        if (head.get("schema") != SCHEMA or head.get("binding_sha256") != self.binding
                or head.get("work_sha256") != work
                or head.get("status") not in {"leased", "submitted", "completed"}
                or type(head.get("generation")) is not int or head["generation"] < 1
                or head["generation"] > (1 << 63) - 1
                or not _LEASE.fullmatch(str(head.get("lease_id", "")))
                or not isinstance(head.get("executor_id"), str)
                or not _HEX.fullmatch(str(head.get("credential_sha256", "")))
                or not isinstance(head.get("snapshot"), dict)
                or not isinstance(head.get("history"), list)
                or len(head["history"]) > MAX_HISTORY
                or not math.isfinite(float(head.get("started_at", float("nan"))))
                or not math.isfinite(float(head.get("expires_at", float("nan"))))
                or float(head["expires_at"]) <= float(head["started_at"])):
            raise ValueError("attempt head is invalid")
        if len(json.dumps(head, allow_nan=False).encode()) > MAX_DOCUMENT_BYTES:
            raise ValueError("attempt snapshot exceeds its byte bound")
        if head.get("result") is not None and digest(head["result"]) != head.get("result_sha256"):
            raise ValueError("attempt result digest does not match")
        if head["status"] == "submitted" and head.get("result") is None:
            raise ValueError("submitted attempt has no result")

    async def _write(self, key, head, etag):
        encoded = json.dumps(head, sort_keys=True, allow_nan=False).encode()
        if len(encoded) > MAX_DOCUMENT_BYTES:
            raise ValueError("attempt snapshot exceeds its byte bound")
        return await _put(key, head, etag, **dict(self._kwargs))

    async def claim(self, work: str, lease: dict, credential: str, snapshot: dict, *, now: float):
        key = self._key(work)
        for _ in range(5):
            old, etag = await _get(key, **dict(self._kwargs))
            if old is not None:
                self._validate(old, work)
                if old["status"] == "submitted" or (old["status"] == "leased" and old["expires_at"] > now):
                    return None
            history = list((old or {}).get("history", []))
            if old:
                history.append({k: old[k] for k in ("generation", "lease_id", "executor_id",
                               "started_at", "expires_at", "status")})
            head = {"schema": SCHEMA, "binding_sha256": self.binding, "work_sha256": work,
                    "generation": (old or {}).get("generation", 0) + 1,
                    "lease_id": lease["lease_id"], "executor_id": lease["executor_id"],
                    "credential_sha256": credential, "expires_at": lease["expires_at"],
                    "started_at": now, "status": "leased", "snapshot": snapshot,
                    "history": history[-MAX_HISTORY:]}
            self._validate(head, work)
            if await self._write(key, head, etag):
                index = {"work_sha256": work, "generation": head["generation"]}
                if not await self._write(self._index(lease["lease_id"]), index, None):
                    stored, _ = await _get(self._index(lease["lease_id"]), **dict(self._kwargs))
                    if stored != index:
                        raise ValueError("attempt lease index conflicts")
                return head
        raise AttemptRefused(503, "attempt_store_busy")

    async def lease(self, lease_id: str):
        index, _ = await _get(self._index(lease_id), **dict(self._kwargs))
        if index is None:
            raise AttemptRefused(410, "lease_unknown")
        if (not isinstance(index, dict) or not _HEX.fullmatch(str(index.get("work_sha256", "")))
                or type(index.get("generation")) is not int or index["generation"] < 1):
            raise ValueError("attempt lease index is invalid")
        head = await self.read(index["work_sha256"])
        if head is None or head["lease_id"] != lease_id or head["generation"] != index["generation"]:
            if index.get("receipt") is not None:
                return {**index["receipt"], "lease_id": lease_id, "status": "completed"}
            raise AttemptRefused(410, "lease_superseded")
        return head

    async def reserve(self, work, lease_id, executor_id, credential, result, snapshot, *, now):
        key = self._key(work)
        result_digest = digest(result)
        for _ in range(5):
            head, etag = await _get(key, **dict(self._kwargs))
            if head is None or head.get("lease_id") != lease_id or head.get("executor_id") != executor_id:
                raise AttemptRefused(410, "lease_superseded")
            self._validate(head, work)
            if head["credential_sha256"] != credential:
                raise AttemptRefused(403, "executor_binding_changed")
            if head["status"] in {"submitted", "completed"}:
                if head.get("result_sha256") != result_digest:
                    raise AttemptRefused(409, "attempt_result_conflict")
                return head
            if head["expires_at"] <= now:
                raise AttemptRefused(410, "lease_expired")
            updated = {**head, "status": "submitted", "result": result,
                       "result_sha256": result_digest, "received_at": now, "snapshot": snapshot}
            if await self._write(key, updated, etag):
                return updated
        raise AttemptRefused(503, "attempt_store_busy")

    async def _receipt(self, work, head):
        # An old successful result can lose its HTTP acknowledgement while a
        # grading continuation acquires its next generation. Keep its compact
        # acknowledgement by nonce; it can never apply a result to newer work.
        key = self._index(head["lease_id"])
        receipt = {k: head[k] for k in ("executor_id", "credential_sha256", "expires_at", "outcome")}
        for field in ("result_sha256", "error"):
            if field in head:
                receipt[field] = head[field]
        for _ in range(5):
            index, etag = await _get(key, **dict(self._kwargs))
            if index is not None and index.get("receipt") is not None:
                if index["receipt"] != receipt:
                    raise ValueError("attempt receipt conflicts")
                return
            updated = {"work_sha256": work, "generation": head["generation"], "receipt": receipt}
            if await self._write(key, updated, etag):
                return
        raise AttemptRefused(503, "attempt_store_busy")

    async def finish(self, work, lease_id, generation, snapshot, outcome, *, error=None):
        key = self._key(work)
        for _ in range(5):
            head, etag = await _get(key, **dict(self._kwargs))
            if head is None or head.get("lease_id") != lease_id or head.get("generation") != generation:
                raise AttemptRefused(410, "lease_superseded")
            self._validate(head, work)
            updated = {**head, "status": "completed", "snapshot": snapshot,
                       "outcome": head.get("outcome", outcome)}
            if error is not None and head["status"] != "completed":
                updated["error"] = error
            if await self._write(key, updated, etag):
                await self._receipt(work, updated)
                return updated
        raise AttemptRefused(503, "attempt_store_busy")
