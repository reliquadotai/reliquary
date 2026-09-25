"""Read-modify-write on one hotkey's slot inside the job's ``miners.json``
CAS object, retrying a conflict by re-reading rather than ever writing a
stale document over a concurrent writer's (Task 4, spec §5)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

from reliquary.corpus.audit_policy import MinerState
from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

# Bounds for the fields a stored entry may carry, keyed by the MinerState
# field name; checked only when the field is present, so a fresh/default
# entry ({}) never trips these.
_NONNEGATIVE_INT_FIELDS = ("audited_passed",)
_FINITE_NUMBER_LIST_FIELDS = ("confirmed_failures", "mant_mean_history")
_FINITE_NUMBER_OR_NONE_FIELDS = ("suspect_until", "banned_until")


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
    return entry


class MinerStates:
    """The per-job miners document, addressed one hotkey at a time."""

    __slots__ = ("_records", "_job_id")

    def __init__(self, records, job_id: str) -> None:
        self._records = records
        self._job_id = job_id

    async def _read_entry(self, hotkey: str) -> tuple[MinerState, Mapping, str | None]:
        document, etag = await self._records.read_miners(self._job_id)
        document = _validate_document(document, self._job_id)
        entry = _validate_entry(document.get(hotkey, {}), self._job_id, hotkey)
        return MinerState.from_dict(entry), document, etag

    async def get(self, hotkey: str) -> MinerState:
        state, _, _ = await self._read_entry(hotkey)
        return state

    async def update(
        self, hotkey: str, change: Callable[[MinerState], MinerState], attempts: int = 5
    ) -> MinerState:
        """Apply ``change`` to ``hotkey``'s current state and write it back.
        On a conflict, re-read the whole document (another hotkey's write may
        have landed) and re-apply ``change`` to the fresh state, so a ban set
        by an earlier update is never clobbered by a write built from a stale
        read."""
        for _ in range(attempts):
            current, document, etag = await self._read_entry(hotkey)
            updated = change(current)
            try:
                await self._records.write_miners(
                    self._job_id, {**document, hotkey: updated.to_dict()}, etag
                )
            except CorpusStoreConflict:
                continue
            return updated
        raise CorpusStoreConflict(
            f"could not update hotkey {hotkey!r} in job {self._job_id!r} after {attempts} attempts"
        )
