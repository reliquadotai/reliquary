"""Read-modify-write on one hotkey's slot inside the job's ``miners.json``
CAS object, retrying a conflict by re-reading rather than ever writing a
stale document over a concurrent writer's (Task 4, spec §5)."""

from __future__ import annotations

from collections.abc import Callable

from reliquary.corpus.audit_policy import MinerState
from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict


class MinerStates:
    """The per-job miners document, addressed one hotkey at a time."""

    __slots__ = ("_records", "_job_id")

    def __init__(self, records, job_id: str) -> None:
        self._records = records
        self._job_id = job_id

    async def get(self, hotkey: str) -> MinerState:
        document, _ = await self._records.read_miners(self._job_id)
        return MinerState.from_dict(document.get(hotkey, {}))

    async def update(
        self, hotkey: str, change: Callable[[MinerState], MinerState], attempts: int = 5
    ) -> MinerState:
        """Apply ``change`` to ``hotkey``'s current state and write it back.
        On a conflict, re-read the whole document (another hotkey's write may
        have landed) and re-apply ``change`` to the fresh state, so a ban set
        by an earlier update is never clobbered by a write built from a stale
        read."""
        for _ in range(attempts):
            document, etag = await self._records.read_miners(self._job_id)
            updated = change(MinerState.from_dict(document.get(hotkey, {})))
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
