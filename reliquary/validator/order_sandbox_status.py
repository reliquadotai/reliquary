"""Per signed-episode order, what the platform sizes its sandbox gateways from (designs
``docs/design/2026-11-12-sandbox-capacity-autoscale.md`` and ``docs/design/2026-11-13-sandbox-sizing-miner-demand.md``):
the ``sandbox`` object of the order control's status.

The sessions a job should have at once (``sessions_target``) are the least of three terms, and ``binding`` names it:

* ``demand``: what miners want on it. Each hotkey counts its declared want (``SandboxEngagement.want``, at least what
  it holds) while it declared it within WANT_TTL_S or still holds a session of the job; a hotkey that declares
  nothing (an older miner) counts what it holds. A hotkey that declared a want on an earlier job of the same model
  in the last FORECAST_WINDOW_S and is busy on no other served job counts that want too (the forecast: a new order
  is sized before its first open). Low-yield hotkeys count nothing. None while no hotkey declared anything (the
  platform keeps its earlier rule).
* ``episodes``: the episodes left.
* ``audit``: the concurrent sessions the audit can keep up with at the most executors the order may have: one
  executor's audited tokens per second x ``executors_allowed`` / (tokens per episode x the share of episodes audited)
  x the episode wall time. None while any is unmeasured (it then never binds).

Built every STATUS_SECONDS from what the control already holds in memory (the job's status counts, the session
book, the quota's refusals, the dispatcher's throughput): no read, no network call."""
from __future__ import annotations

import collections
import math
import statistics
from collections.abc import Callable, Mapping
from typing import Any

EPISODE_TIMES_KEPT = 50        # episode_seconds: the median of the job's last this many finished episodes
EPISODE_TOKENS_KEPT = 1000     # tokens_per_episode: the mean over the job's last this many audited episodes
FLEET_FULL_WINDOW_S = 600      # quota_refusals_fleet_full: the last ten minutes
WANT_TTL_S = 1800              # a declared want counts this long after the open that carried it (or while held)


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


def held_by_hotkey(book, now: int) -> dict[str, dict[str, int]]:
    """job id -> hotkey -> the sessions it holds there now."""
    from reliquary.sandbox.sessions import SessionBook

    held: dict[str, dict[str, int]] = {}
    for record in book.records():
        if record.job_id is not None and SessionBook._holds(record, now):
            per_job = held.setdefault(record.job_id, {})
            per_job[record.hotkey] = per_job.get(record.hotkey, 0) + 1
    return held


def job_demand(job_id: str, *, now: int, declarations: Mapping[str, Mapping[str, tuple[int, int]]],
               held: Mapping[str, Mapping[str, int]], model_of: Mapping[str, Any], served: set[str],
               excluded: Callable[[str], bool] = lambda hotkey: False,
               want_ttl_s: float = WANT_TTL_S) -> dict[str, Any]:
    """The miners' demand on ``job_id`` (module docstring): {sessions (None: nothing declared), declared, forecast,
    held_undeclared, hotkeys}. ``declarations``: the quota's (job -> hotkey -> (want, at)); ``held``:
    ``held_by_hotkey``; ``model_of``: job id -> its model (jobs served lately); ``served``: the jobs served now;
    ``excluded``: a hotkey counting nothing (low yield)."""
    here, mine = declarations.get(job_id, {}), held.get(job_id, {})
    counted: set[str] = set()
    declared = undeclared = forecast = 0
    for hotkey in set(here) | set(mine):
        if excluded(hotkey):
            continue
        holds = int(mine.get(hotkey, 0))
        want, at = here.get(hotkey, (0, None))
        if at is not None and (at >= now - want_ttl_s or holds > 0):
            declared += max(int(want), holds)
        elif holds:
            undeclared += holds
        else:
            continue
        counted.add(hotkey)
    model = model_of.get(job_id)
    if model is not None:
        busy = {hotkey for other in served if other != job_id
                for hotkey, (_want, at) in declarations.get(other, {}).items() if at >= now - want_ttl_s}
        busy |= {hotkey for other in served if other != job_id for hotkey in held.get(other, {})}
        latest: dict[str, tuple[int, int]] = {}
        for other, wants in declarations.items():
            if other == job_id or other in served or model_of.get(other) != model:
                continue
            for hotkey, (want, at) in wants.items():
                if hotkey not in latest or at > latest[hotkey][1]:
                    latest[hotkey] = (int(want), int(at))
        for hotkey, (want, _at) in latest.items():
            if hotkey in counted or hotkey in busy or excluded(hotkey):
                continue
            forecast += want
            counted.add(hotkey)
    known = declared > 0 or forecast > 0
    return {"sessions": declared + undeclared + forecast if known else None, "declared": declared,
            "forecast": forecast, "held_undeclared": undeclared, "hotkeys": len(counted)}


def audit_capacity_sessions(tokens_per_second: tuple[float, float] | None, executors_allowed: int | None,
                            tokens_per_episode: float | None, audit_share: float | None,
                            seconds_per_episode: float | None) -> int | None:
    """The concurrent sessions whose episodes the audit keeps up with at ``executors_allowed`` executors: one
    executor's audited tokens per second x executors / (tokens per episode x the audited share) x the episode wall
    time. None while any is unknown."""
    if (tokens_per_second is None or not executors_allowed or not tokens_per_episode or not audit_share
            or not seconds_per_episode):
        return None
    per_second = sum(tokens_per_second) * int(executors_allowed) / (float(tokens_per_episode)
                                                                     * min(1.0, max(1e-3, float(audit_share))))
    return max(1, math.floor(per_second * float(seconds_per_episode)))


TERMS = ("demand", "episodes", "audit")   # the order a tie names


def sessions_target(demand: int | None, episodes_left: int | None,
                    audit_capacity: int | None) -> tuple[int | None, str | None]:
    """(the least of the three terms, the term that binds); (None, None) while the demand is unknown."""
    if demand is None:
        return None, None
    terms = [(value, name) for name, value in zip(TERMS, (demand, episodes_left, audit_capacity), strict=True)
             if value is not None]
    value, name = min(terms, key=lambda t: (t[0], TERMS.index(t[1])))
    return max(0, int(value)), name


def sandbox_status(*, job, status: Mapping | None, reopens: bool, held: int, finished: list[tuple[int, int]],
                   tokens_per_episode: float | None, tokens_per_second: tuple[float, float] | None,
                   executors_wanted: int | None, fleet_refusals: int, demand: Mapping[str, Any] | None = None,
                   executors_allowed: int | None = None, audit_share: float | None = None) -> dict[str, Any]:
    total, done, left = episode_counts(job, status, reopens=reopens)
    seconds = episode_seconds(finished)
    audit_capacity = audit_capacity_sessions(tokens_per_second, executors_allowed, tokens_per_episode, audit_share,
                                             seconds)
    wanted = None if demand is None else demand.get("sessions")
    target, binding = sessions_target(wanted, left, audit_capacity)
    return {"episodes_total": total, "episodes_finished": done, "episodes_left": left,
            "sessions_held": int(held), "episode_seconds": seconds,
            "tokens_per_episode": tokens_per_episode,
            "audit_episodes_per_second": audit_episodes_per_second(tokens_per_second, executors_wanted,
                                                                   tokens_per_episode),
            "quota_refusals_fleet_full": int(fleet_refusals),
            "demand_sessions": wanted,
            "demand_declared_sessions": None if demand is None else demand.get("declared"),
            "demand_forecast_sessions": None if demand is None else demand.get("forecast"),
            "demand_hotkeys": None if demand is None else demand.get("hotkeys"),
            "audit_share": None if audit_share is None else round(float(audit_share), 4),
            "executors_allowed": executors_allowed,
            "audit_capacity_sessions": audit_capacity,
            "sessions_target": target, "binding": binding}


__all__ = ["WANT_TTL_S", "EpisodeTokens", "audit_capacity_sessions", "audit_episodes_per_second", "book_sessions",
           "episode_counts", "episode_seconds", "held_by_hotkey", "job_demand", "sandbox_status", "sessions_target"]
