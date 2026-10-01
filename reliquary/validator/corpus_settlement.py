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

# Wall time of one RL window: the V1 cycle measured on 2026-09-13 (proof
# ~11.3 min + rotation ~4.6 min). Alone, the corpus advances at most this often,
# so the shared replay horizon never moves faster than RL itself moved it.
RL_WINDOW_SECONDS = 16 * 60


def rewards_for(verdicts: Iterable[Mapping], cap: float) -> dict[str, float]:
    tokens: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.get("passed"):
            tokens[verdict["hotkey"]] = tokens.get(verdict["hotkey"], 0) + int(verdict["token_count"])
    total = sum(tokens.values())
    if total <= 0:
        return {}
    return {hotkey: cap * count / total for hotkey, count in tokens.items()}


def _stalled(other_max_seen_at, now, stall_seconds) -> bool:
    return other_max_seen_at is not None and now - other_max_seen_at > stall_seconds


def choose_window(*, last_window, other_max, other_max_seen_at, now, stall_seconds,
                  last_advanced_at=None, advance_every_seconds=0.0):
    if other_max is not None and (last_window is None or other_max > last_window):
        return other_max
    if other_max is None and last_window is None:
        return 0
    if other_max is not None and not _stalled(other_max_seen_at, now, stall_seconds):
        return None
    # Every other task is idle (or none exists): advancing alone decays it the
    # way a retired task already decays, but no faster than RL itself would.
    if last_advanced_at is not None and now - last_advanced_at < advance_every_seconds:
        return None
    return last_window + 1


class CorpusSettler:
    def __init__(self, *, task_id, job_id, cap, records, archives,
                 stall_seconds: float = 3 * RL_WINDOW_SECONDS,
                 advance_every_seconds: float = RL_WINDOW_SECONDS, clock=time.time,
                 on_settled=None) -> None:
        self._task_id = task_id
        self._job_id = job_id
        self._cap = float(cap)
        self._records = records
        self._archives = archives
        self._stall = stall_seconds
        self._advance_every = advance_every_seconds
        self._clock = clock
        # How many verdicts stand settled, as of the last settlement read.
        self.settled_count: int | None = None
        # Verdict totals of everything settled, persisted in the settlement
        # object (the status route's counts, with no listing).
        self.totals: dict | None = None
        # Told which ids each settlement moved, once it is written.
        self.on_settled = on_settled

    def set_cap(self, cap: float) -> None:
        """A cap changed in the registry: the next settlement pays under it."""
        self._cap = float(cap)

    def _archive(self, window: int, rewards: Mapping[str, float]) -> dict:
        return {
            "window_start": int(window),
            "window_status": "completed",
            "rewards_by_hotkey": dict(rewards),
            "task_id": self._task_id,
            "mechanism": "corpus-generation",
            "job_id": self._job_id,
        }

    async def _finish(self, state: dict, etag, now: float) -> int:
        pending = state["pending"]
        # Idempotent: the same window and the same rewards, however often a
        # crash makes this run again.
        await self._archives.write(self._task_id, pending["window"], self._archive(pending["window"], pending["rewards"]))
        final = {
            **state,
            "last_window": pending["window"],
            "settled": sorted(set(state.get("settled") or []) | set(pending["ids"])),
            "pending": None,
            # Carried in the pending step, so a repeated finish adds nothing twice.
            "totals": pending.get("totals", state.get("totals")),
        }
        if pending.get("alone"):
            # The finish time, not the choice time: a finish delayed by a crash
            # or a hold must still be one RL window from the next lone advance.
            final["advanced_at"] = now
        await self._records.write_settlement(self._job_id, final, etag)
        self._settled(final, pending["ids"])
        return pending["window"]

    def _settled(self, state: dict, ids) -> None:
        self.settled_count = len(state["settled"] or ())
        self.totals = state.get("totals")
        if self.on_settled is not None and ids:
            self.on_settled(list(ids))

    @staticmethod
    def _add(totals: dict, verdicts) -> dict:
        passed = [v for v in verdicts if v and v.get("passed")]
        return {**totals, "verdicts": totals["verdicts"] + len(verdicts),
                "passed": totals["passed"] + len(passed),
                "verified_tokens": totals["verified_tokens"]
                + sum(int(v["token_count"]) for v in passed)}

    async def settle_once(self) -> int | None:
        state, etag = await self._records.read_settlement(self._job_id)
        state = {"schema": SETTLEMENT_SCHEMA, "last_window": None, "settled": [],
                 "other_max_seen": None, "other_max_seen_at": None, "advanced_at": None,
                 "pending": None, "totals": None, **state}
        if state["totals"] is None:
            # A job settled before totals were kept: counted from here on.
            state["totals"] = {"verdicts": 0, "passed": 0, "verified_tokens": 0,
                               "complete": not state["settled"]}
        self._settled(state, ())

        now = self._clock()
        other_max = await self._archives.other_max(self._task_id)
        clock_changed = other_max != state["other_max_seen"]
        if clock_changed:
            # Another task sealed: a new stall, if one comes, starts its own
            # RL-cadence spacing from scratch.
            state["other_max_seen"], state["other_max_seen_at"] = other_max, now
            state["advanced_at"] = None

        if state["pending"]:
            window = state["pending"]["window"]
            live = other_max is not None and not _stalled(state["other_max_seen_at"], now, self._stall)
            if live and window > other_max:
                # Chosen alone during a stall and interrupted; the other task
                # has since revived. Re-targeting the ids to another index could
                # pay them twice if the archive already landed, so hold them
                # until the live task reaches this window (the next corpus index
                # would have waited for exactly that anyway).
                if clock_changed:
                    await self._records.write_settlement(self._job_id, state, etag)
                return None
            return await self._finish(state, etag, now)

        settled = set(state["settled"])
        new_ids = [sid for sid in await self._records.list_verdict_ids(self._job_id) if sid not in settled]
        window = choose_window(last_window=state["last_window"], other_max=other_max,
                               other_max_seen_at=state["other_max_seen_at"], now=now,
                               stall_seconds=self._stall, last_advanced_at=state["advanced_at"],
                               advance_every_seconds=self._advance_every)

        if new_ids and window is not None:
            verdicts = [await self._records.read_verdict(self._job_id, sid) for sid in new_ids]
            lister = getattr(self._records, "list_voided_ids", None)
            if lister is not None:
                # Withdrawn after a quarantined executor's re-audit: settled, never paid.
                voided = set(await lister(self._job_id))
                verdicts = [v if sid not in voided else {**(v or {}), "passed": False}
                            for sid, v in zip(new_ids, verdicts)]
            rewards = rewards_for(verdicts, self._cap)
            totals = self._add(state["totals"], verdicts)
            if rewards:
                alone = other_max is None or window > other_max
                state["pending"] = {"window": window, "ids": new_ids, "rewards": rewards,
                                    "alone": alone, "at": now, "totals": totals}
                etag = await self._records.write_settlement(self._job_id, state, etag)
                return await self._finish(state, etag, now)
            # Every verdict this period failed (spec §7): no archive, the
            # index does not move, but these ids must not be reconsidered
            # forever, so mark them settled in this same CAS write.
            state["settled"] = sorted(settled | set(new_ids))
            state["totals"] = totals
            await self._records.write_settlement(self._job_id, state, etag)
            self._settled(state, new_ids)
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
    """The two archive calls the settler makes, against the real bucket.

    ``served`` names the tasks this process wired after boot, which
    ``RELIQUARY_TASK_ID`` cannot list.
    """

    def __init__(self, *, served=None) -> None:
        self._served = served

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

        from reliquary.shared.task_id import parse_task_ids

        # The corpus validator runs under its own task id(s), so refuse to
        # write under any task RELIQUARY_TASK_ID does not name.
        # Unset is refused as before, never read as the legacy task.
        served = os.getenv("RELIQUARY_TASK_ID")
        hot = set(self._served()) if self._served is not None else set()
        if not served or (task_id not in parse_task_ids(served) and task_id not in hot):
            raise RuntimeError(f"RELIQUARY_TASK_ID does not name {task_id!r}; refusing to archive")
        await storage.upload_window_dataset(window, data, task_id=task_id)
