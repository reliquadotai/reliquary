"""Per signed-episode order, what the platform sizes its sandbox gateways from (design
``docs/design/2026-11-12-sandbox-capacity-autoscale.md``): the ``sandbox`` object of the order control's status.

Built every STATUS_SECONDS from what the control already holds in memory (the job's status counts, the session
book, the quota's refusals, the dispatcher's throughput): no read, no network call."""
from __future__ import annotations

import collections
import math
import statistics
from collections.abc import Mapping
from typing import Any

EPISODE_TIMES_KEPT = 50        # episode_seconds: the median of the job's last this many finished episodes
EPISODE_TOKENS_KEPT = 1000     # tokens_per_episode: the mean over the job's last this many audited episodes
FLEET_FULL_WINDOW_S = 600      # quota_refusals_fleet_full: the last ten minutes


class EpisodeTokens:
    """A job's audited prompt + completion tokens per episode, from its verdicts (the auditor's ``on_verdict``
    and the settler's reads: the same id seen twice counts once)."""

    def __init__(self, kept: int = EPISODE_TOKENS_KEPT) -> None:
        self._kept = int(kept)
        self._tokens: collections.OrderedDict[str, int] = collections.OrderedDict()

    def observe(self, submission_id: str, verdict: Mapping | None) -> None:
        if not verdict:
            return
        try:
            tokens = int(verdict.get("token_count") or 0) + int(verdict.get("prompt_token_count") or 0)
        except (TypeError, ValueError):
            return
        if tokens <= 0:
            return
        self._tokens[str(submission_id)] = tokens
        self._tokens.move_to_end(str(submission_id))
        while len(self._tokens) > self._kept:
            self._tokens.popitem(last=False)

    def mean(self) -> float | None:
        return math.fsum(self._tokens.values()) / len(self._tokens) if self._tokens else None


def book_sessions(book, now: int) -> dict[str, tuple[int, list[tuple[int, int]]]]:
    """Per job, in one pass over the session book: its held sessions, and (submitted at, wall seconds) of its
    finished episodes (submitted, not handed back: open to the final signed record)."""
    from reliquary.sandbox.sessions import HANDED_BACK, SUBMITTED, SessionBook

    jobs: dict[str, tuple[int, list[tuple[int, int]]]] = {}
    for record in book.records():
        if record.job_id is None:
            continue
        held, finished = jobs.get(record.job_id, (0, []))
        if SessionBook._holds(record, now):
            held += 1
        elif record.state == SUBMITTED and record.closed_status != HANDED_BACK and record.closed_at is not None:
            finished.append((int(record.closed_at), max(0, int(record.closed_at) - int(record.issued_at))))
        jobs[record.job_id] = (held, finished)
    return jobs


def episode_seconds(finished: list[tuple[int, int]], kept: int = EPISODE_TIMES_KEPT) -> float | None:
    """The median wall time of the last ``kept`` finished episodes; None before any."""
    if not finished:
        return None
    return float(statistics.median(wall for _at, wall in sorted(finished)[-kept:]))


def episode_counts(job, status: Mapping | None, *, reopens: bool) -> tuple[int, int | None, int | None]:
    """(total, finished, left) episodes: problems x samples; the accepted signed episodes (less the failed audits
    when a failure reopens its slot); none left once the job takes no more. None while its status is unknown."""
    total = int(job.prompt_count) * int(job.slots_per_prompt)
    if status is None or status.get("submissions_accepted") is None:
        return total, None, None
    finished = int(status["submissions_accepted"])
    if reopens:
        finished -= max(0, int(status.get("audited") or 0) - int(status.get("passed") or 0))
    finished = max(0, min(total, finished))
    done = status.get("complete") is True or status.get("state") in ("full", "retired", "drained")
    return total, finished, 0 if done else total - finished


def audit_episodes_per_second(tokens_per_second: tuple[float, float] | None, executors_wanted: int | None,
                              tokens_per_episode: float | None) -> float | None:
    """Episodes the audit decides per second: one executor's audited (completion + prompt) tokens per second x
    the executors wanted for the model / the job's tokens per episode; None when any is unknown."""
    if tokens_per_second is None or executors_wanted is None or not tokens_per_episode:
        return None
    return sum(tokens_per_second) * int(executors_wanted) / float(tokens_per_episode)


def sandbox_status(*, job, status: Mapping | None, reopens: bool, held: int, finished: list[tuple[int, int]],
                   tokens_per_episode: float | None, tokens_per_second: tuple[float, float] | None,
                   executors_wanted: int | None, fleet_refusals: int) -> dict[str, Any]:
    total, done, left = episode_counts(job, status, reopens=reopens)
    return {"episodes_total": total, "episodes_finished": done, "episodes_left": left,
            "sessions_held": int(held), "episode_seconds": episode_seconds(finished),
            "tokens_per_episode": tokens_per_episode,
            "audit_episodes_per_second": audit_episodes_per_second(tokens_per_second, executors_wanted,
                                                                   tokens_per_episode),
            "quota_refusals_fleet_full": int(fleet_refusals)}


__all__ = ["EpisodeTokens", "audit_episodes_per_second", "book_sessions", "episode_counts", "episode_seconds",
           "sandbox_status"]
