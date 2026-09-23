"""Pay the corpus task's cap by verified tokens, in ordinary per-task archives.

The weight-only replay pays these archives with no change. The one coupling
with other tasks is the replay horizon (the highest index across tasks), so
the index rules here keep the corpus from ever moving it while another task is
alive. Settlement is two-phase so a crash can delay a payment, never repeat it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import logging
import time

logger = logging.getLogger(__name__)

SETTLEMENT_SCHEMA = "reliquary/corpus-settlement/v1"


def rewards_for(verdicts: Iterable[Mapping], cap: float) -> dict[str, float]:
    tokens: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.get("passed"):
            tokens[verdict["hotkey"]] = tokens.get(verdict["hotkey"], 0) + int(verdict["token_count"])
    total = sum(tokens.values())
    if total <= 0:
        return {}
    return {hotkey: cap * count / total for hotkey, count in tokens.items()}


def choose_window(*, last_window, other_max, other_max_seen_at, now, stall_seconds):
    if other_max is None:
        return 0 if last_window is None else last_window + 1
    if last_window is None or other_max > last_window:
        return other_max
    if other_max_seen_at is not None and now - other_max_seen_at > stall_seconds:
        # Every other task is idle: advancing alone decays it the way a
        # retired task already decays.
        return last_window + 1
    return None


class CorpusSettler:
    def __init__(self, *, task_id, job_id, cap, records, archives,
                 stall_seconds: float = 3 * 16 * 60, clock=time.time) -> None:
        self._task_id = task_id
        self._job_id = job_id
        self._cap = float(cap)
        self._records = records
        self._archives = archives
        self._stall = stall_seconds
        self._clock = clock

    def _archive(self, window: int, rewards: Mapping[str, float]) -> dict:
        return {
            "window_start": int(window),
            "window_status": "completed",
            "rewards_by_hotkey": dict(rewards),
            "task_id": self._task_id,
            "mechanism": "corpus-generation",
            "job_id": self._job_id,
        }

    async def _finish(self, state: dict, etag) -> int:
        pending = state["pending"]
        # Idempotent: the same window and the same rewards, however often a
        # crash makes this run again.
        await self._archives.write(self._task_id, pending["window"], self._archive(pending["window"], pending["rewards"]))
        final = {
            **state,
            "last_window": pending["window"],
            "settled": sorted(set(state.get("settled") or []) | set(pending["ids"])),
            "pending": None,
        }
        await self._records.write_settlement(self._job_id, final, etag)
        return pending["window"]

    async def settle_once(self) -> int | None:
        state, etag = await self._records.read_settlement(self._job_id)
        state = {"schema": SETTLEMENT_SCHEMA, "last_window": None, "settled": [],
                 "other_max_seen": None, "other_max_seen_at": None, "pending": None, **state}
        if state["pending"]:
            return await self._finish(state, etag)

        now = self._clock()
        other_max = await self._archives.other_max(self._task_id)
        clock_changed = other_max != state["other_max_seen"]
        if clock_changed:
            state["other_max_seen"], state["other_max_seen_at"] = other_max, now

        settled = set(state["settled"])
        new_ids = [sid for sid in await self._records.list_verdict_ids(self._job_id) if sid not in settled]
        window = choose_window(last_window=state["last_window"], other_max=other_max,
                               other_max_seen_at=state["other_max_seen_at"], now=now,
                               stall_seconds=self._stall)

        if new_ids and window is not None:
            verdicts = [await self._records.read_verdict(self._job_id, sid) for sid in new_ids]
            rewards = rewards_for(verdicts, self._cap)
            if rewards:
                state["pending"] = {"window": window, "ids": new_ids, "rewards": rewards}
                etag = await self._records.write_settlement(self._job_id, state, etag)
                return await self._finish(state, etag)
            # Every verdict this period failed (spec §7): no archive, the
            # index does not move, but these ids must not be reconsidered
            # forever, so mark them settled in this same CAS write.
            state["settled"] = sorted(settled | set(new_ids))
            await self._records.write_settlement(self._job_id, state, etag)
            return None

        if clock_changed:
            # Nothing settles this call, but other_max genuinely moved: CAS
            # it in now. Otherwise the next call finds the persisted
            # other_max_seen still stale, "changes" again, and keeps
            # resetting the stall clock to "now" forever — the corpus is
            # never paid again once the other task goes idle (§7b rule 3).
            await self._records.write_settlement(self._job_id, state, etag)
        return None


class R2Archives:
    """The two archive calls the settler makes, against the real bucket."""

    async def other_max(self, task_id: str) -> int | None:
        from reliquary.infrastructure import storage

        best = None
        for other in await storage.list_task_ids(strict=True):
            if other == task_id:
                continue
            windows = await storage.list_all_window_keys(task_id=other, strict=True)
            if windows:
                best = max(best or 0, max(windows))
        return best

    async def write(self, task_id: str, window: int, data: dict) -> None:
        import os

        from reliquary.infrastructure import storage

        # upload_window_dataset keys by RELIQUARY_TASK_ID; the corpus validator
        # runs under its own task id, so refuse to write anywhere else.
        if os.getenv("RELIQUARY_TASK_ID") != task_id:
            raise RuntimeError(f"RELIQUARY_TASK_ID is not {task_id!r}; refusing to archive")
        await storage.upload_window_dataset(window, data)
