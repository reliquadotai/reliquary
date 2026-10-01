"""Read-modify-write on one hotkey's slot inside the job's ``miners.json``
CAS object, retrying a conflict by re-reading rather than ever writing a
stale document over a concurrent writer's (Task 4, spec §5)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import logging
import math
import re
import time

from reliquary.corpus.audit_policy import MinerState
from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

try:
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - botocore ships with the store
    BotoCoreError = ClientError = OSError

logger = logging.getLogger(__name__)

# Throttled (R2 allows about one write per second to a key), 5xx, reset or
# timeout: the same transport errors the corpus route answers 503 for.
_TRANSIENT_ERRORS = (ClientError, BotoCoreError, OSError, asyncio.TimeoutError)
_BACKOFF_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 8.0

# Bounds for the fields a stored entry may carry, keyed by the MinerState
# field name; checked only when the field is present, so a fresh/default
# entry ({}) never trips these.
_NONNEGATIVE_INT_FIELDS = ("audited_passed",)
_FINITE_NUMBER_LIST_FIELDS = ("confirmed_failures", "mant_mean_history")
_FINITE_NUMBER_OR_NONE_FIELDS = ("suspect_until", "banned_until")
_SUBMISSION_ID = re.compile(r"[0-9a-f]{64}")


def _is_finite_number(value: object) -> bool:
    # bool is an int subclass in Python; excluded so a stray `true`/`false`
    # in the JSON is caught rather than silently read as 0/1.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_document(document: object, job_id: str) -> Mapping:
    # This object enforces bans: a document that hand-editing turned into a
    # list or string must fail loudly here, not read back as "no miners yet".
    if not isinstance(document, Mapping):
        raise ValueError(f"miners.json for job {job_id!r} is not an object, got {document!r}")
    return document


def _validate_entry(entry: object, job_id: str, hotkey: str) -> Mapping:
    """The stored shape for one hotkey, checked before it reaches
    ``MinerState.from_dict`` — which would otherwise turn a non-mapping or a
    field of the wrong type into a silently fresh (probation, unbanned)
    state, or a ``MinerState`` whose bad field fails far from here."""
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"miners.json entry for hotkey {hotkey!r} in job {job_id!r} is not an object, got {entry!r}"
        )
    for field in _NONNEGATIVE_INT_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{field} for hotkey {hotkey!r} in job {job_id!r} must be a non-negative int, "
                f"got {value!r}"
            )
    for field in _FINITE_NUMBER_LIST_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, list) or not all(_is_finite_number(v) for v in value):
            raise ValueError(
                f"{field} for hotkey {hotkey!r} in job {job_id!r} must be a list of finite numbers, "
                f"got {value!r}"
            )
    for field in _FINITE_NUMBER_OR_NONE_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if value is not None and not _is_finite_number(value):
            raise ValueError(
                f"{field} for hotkey {hotkey!r} in job {job_id!r} must be a finite number or null, "
                f"got {value!r}"
            )
    for field in ("failure_ids", "pass_ids"):
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, list) or not all(
            isinstance(v, str) and _SUBMISSION_ID.fullmatch(v) for v in value
        ):
            raise ValueError(
                f"{field} for hotkey {hotkey!r} in job {job_id!r} must be a list of "
                f"64-lowercase-hex submission ids, got {value!r}"
            )
    return entry


class MinerStates:
    """The per-job miners document, addressed one hotkey at a time."""

    __slots__ = ("_records", "_job_id", "_sleep", "_clock", "_mirror", "_mirror_lock")

    def __init__(self, records, job_id: str, *, sleep=asyncio.sleep, clock=time.time) -> None:
        self._records = records
        self._job_id = job_id
        self._sleep = sleep
        self._clock = clock
        # The last document read or written, and when: the miner status route
        # answers from it instead of reading the store per request.
        self._mirror: tuple[Mapping, float] | None = None
        self._mirror_lock: asyncio.Lock | None = None

    async def _read_document(self) -> tuple[Mapping, str | None]:
        document, etag = await self._records.read_miners(self._job_id)
        document = _validate_document(document, self._job_id)
        self._mirror = (document, self._clock())
        return document, etag

    def _entry(self, document: Mapping, hotkey: str) -> MinerState:
        return MinerState.from_dict(_validate_entry(document.get(hotkey, {}), self._job_id, hotkey))

    def mirror(self) -> tuple[Mapping, float] | None:
        """The last document seen and when, or None before any read."""
        return self._mirror

    async def state_at_most(self, hotkey: str, max_age: float) -> tuple[MinerState, float]:
        """``hotkey``'s state from a document at most ``max_age`` seconds old,
        read only when the mirror is older; one read at a time."""
        if self._mirror_lock is None:
            self._mirror_lock = asyncio.Lock()
        async with self._mirror_lock:
            if self._mirror is None or self._clock() - self._mirror[1] > max_age:
                await self._read_document()
        document, at = self._mirror
        return self._entry(document, hotkey), at

    async def get(self, hotkey: str) -> MinerState:
        document, _ = await self._read_document()
        return self._entry(document, hotkey)

    async def hotkeys(self) -> list[str]:
        document, _ = await self._read_document()
        return sorted(document)

    async def update(
        self, hotkey: str, change: Callable[[MinerState], MinerState], attempts: int = 6
    ) -> MinerState:
        return (await self.update_many({hotkey: change}, attempts))[hotkey]

    async def update_many(
        self, changes: Mapping[str, Callable[[MinerState], MinerState]], attempts: int = 6
    ) -> dict[str, MinerState]:
        """Apply each hotkey's ``change`` to its current state and write them
        all back in one compare-and-swap. On a conflict, re-read the whole
        document (another writer may have landed) and re-apply every change
        to the fresh state, so a ban set by an earlier update is never
        clobbered by a write built from a stale read. A transport error
        (throttling above all) backs off and retries, within ``attempts``."""
        for attempt in range(attempts):
            try:
                document, etag = await self._read_document()
                updated = {hotkey: change(self._entry(document, hotkey))
                           for hotkey, change in changes.items()}
                written = {**document,
                           **{hotkey: state.to_dict() for hotkey, state in updated.items()}}
                await self._records.write_miners(self._job_id, written, etag)
            except CorpusStoreConflict:
                continue
            except _TRANSIENT_ERRORS:
                if attempt + 1 >= attempts:
                    raise
                delay = min(_MAX_BACKOFF_SECONDS, _BACKOFF_SECONDS * 2**attempt)
                logger.warning("miners.json of job %s unavailable; retrying in %.1f s",
                               self._job_id, delay, exc_info=True)
                await self._sleep(delay)
                continue
            self._mirror = (written, self._clock())
            return updated
        raise CorpusStoreConflict(
            f"could not update hotkeys {sorted(changes)!r} in job {self._job_id!r} "
            f"after {attempts} attempts"
        )
