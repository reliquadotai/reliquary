"""Per signed-episode order, what the platform sizes its sandbox gateways from (designs
``docs/design/2026-11-12-sandbox-capacity-autoscale.md`` and ``docs/design/2026-11-13-sandbox-sizing-miner-demand.md``):
the ``sandbox`` object of the order control's status.

The sessions a job should have at once (``sessions_target``) are the least of three terms, and ``binding`` names it:

* ``demand``: what miners want on it. Each hotkey counts its declared want (``SandboxEngagement.want``) while it
  declared it within WANT_TTL_S or still holds a session of the job, bounded by what it showed it runs: at most
  WANT_GROWTH x the larger of its sessions held in the last RECENT_S and its episodes finished in the last
  FINISHED_S (at least WANT_FLOOR), never below what it holds. A hotkey whose declaration has not turned into a
  session counts one session for NEW_WANT_S, then nothing. So the fleet grows with what miners prove they use (a
  hotkey's count doubles as its sessions do), never with what one says. A hotkey that declares nothing (an older
  miner) counts what it holds. A hotkey that declared a want on an earlier job of the same model in the last
  FORECAST_WINDOW_S and is busy on no other served job counts that want too, bounded the same way by the most
  sessions it held there (the forecast: a new order is sized before its first open). Low-yield hotkeys count
  nothing. None while no hotkey declared anything (the platform keeps its earlier rule).
* ``episodes``: the episodes left.
* ``audit``: the concurrent sessions the audit can keep up with at the most executors the order may have: one
  executor's audited tokens per second x ``executors_allowed`` / (tokens per episode x the share of episodes audited)
  x the episode wall time. While any is unmeasured, a conservative prior stands in (``audit_prior``:
  AUDIT_PRIOR_SESSIONS_PER_EXECUTOR per executor allowed).

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
RECENT_S = 600                 # a hotkey's recent sessions: held at some point in the last ten minutes
FINISHED_S = 1800              # its recent episodes: finished (submitted) in the last half hour
WANT_GROWTH = 2                # a want counts at most this many times what the hotkey showed it runs
WANT_FLOOR = 4                 # ... and at least this many once it ran anything
NEW_WANT_S = 600               # a want with no session yet counts one session this long, then nothing
AUDIT_PRIOR_SESSIONS_PER_EXECUTOR = 64   # the audit term before its throughput is measured, per executor allowed


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


def activity_by_hotkey(book, now: int) -> dict[str, dict[str, tuple[int, int]]]:
    """job id -> hotkey -> (its sessions held at some point in the last RECENT_S, its episodes finished in the last
    FINISHED_S): what a hotkey showed it runs, the bound on what its want counts for."""
    from reliquary.sandbox.sessions import HANDED_BACK, SUBMITTED, SessionBook

    out: dict[str, dict[str, tuple[int, int]]] = {}
    for record in book.records():
        if record.job_id is None:
            continue
        closed = record.closed_at
        recent = SessionBook._holds(record, now) or (closed is not None and int(closed) >= now - RECENT_S)
        finished = (record.state == SUBMITTED and record.closed_status != HANDED_BACK and closed is not None
                    and int(closed) >= now - FINISHED_S)
        if not (recent or finished):
            continue
        per_job = out.setdefault(record.job_id, {})
        r, f = per_job.get(record.hotkey, (0, 0))
        per_job[record.hotkey] = (r + int(bool(recent)), f + int(finished))
    return out


def bounded_want(want: int, shown: int) -> int:
    """A declared want counted against what the hotkey showed it runs (``shown`` > 0)."""
    return min(int(want), max(WANT_FLOOR, WANT_GROWTH * int(shown)))


def job_demand(job_id: str, *, now: int, declarations: Mapping[str, Mapping[str, tuple]],
               held: Mapping[str, Mapping[str, int]], model_of: Mapping[str, Any], served: set[str],
               activity: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
               excluded: Callable[[str], bool] = lambda hotkey: False,
               want_ttl_s: float = WANT_TTL_S) -> dict[str, Any]:
    """The miners' demand on ``job_id`` (module docstring): {sessions (None: nothing declared), declared, forecast,
    held_undeclared, hotkeys}. ``declarations``: the quota's (job -> hotkey -> (want, at, peak held)); ``held``:
    ``held_by_hotkey``; ``activity``: ``activity_by_hotkey``; ``model_of``: job id -> its model (jobs served
    lately); ``served``: the jobs served now; ``excluded``: a hotkey counting nothing (low yield)."""
    here, mine = declarations.get(job_id, {}), held.get(job_id, {})
    shown_here = (activity or {}).get(job_id, {})
    counted: set[str] = set()
    declared = undeclared = forecast = 0
    for hotkey in set(here) | set(mine):
        if excluded(hotkey):
            continue
        holds = int(mine.get(hotkey, 0))
        want, at = here[hotkey][:2] if hotkey in here else (0, None)
        if at is not None and (at >= now - want_ttl_s or holds > 0):
            shown = max(holds, *shown_here.get(hotkey, (0, 0)))
            if shown:
                declared += max(holds, bounded_want(want, shown))
            elif at >= now - NEW_WANT_S:
                declared += 1                       # its first open: one session until it runs one
            else:
                continue                            # it declared, and never ran a session: nothing
        elif holds:
            undeclared += holds
        else:
            continue
        counted.add(hotkey)
    model = model_of.get(job_id)
    if model is not None:
        busy = {hotkey for other in served if other != job_id
                for hotkey, entry in declarations.get(other, {}).items() if entry[1] >= now - want_ttl_s}
        busy |= {hotkey for other in served if other != job_id for hotkey in held.get(other, {})}
        latest: dict[str, tuple[int, int, int]] = {}
        for other, wants in declarations.items():
            if other == job_id or other in served or model_of.get(other) != model:
                continue
            for hotkey, entry in wants.items():
                want, at = int(entry[0]), int(entry[1])
                peak = int(entry[2]) if len(entry) > 2 else 0
                if hotkey not in latest or at > latest[hotkey][1]:
                    latest[hotkey] = (want, at, peak)
        for hotkey, (want, _at, peak) in latest.items():
            if hotkey in counted or hotkey in busy or excluded(hotkey) or peak <= 0:
                continue                            # never held a session there: no forecast from its word
            forecast += bounded_want(want, peak)
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


TERMS = ("demand", "episodes", "audit", "audit_prior")   # the order a tie names


def audit_prior_sessions(executors_allowed: int | None) -> int:
    """The audit term before its throughput is measured: conservative, so a declared demand cannot rent much
    before the audit shows it can keep up (at 5 % audited the measured term is far larger)."""
    return AUDIT_PRIOR_SESSIONS_PER_EXECUTOR * max(1, int(executors_allowed or 1))


def sessions_target(demand: int | None, episodes_left: int | None, audit_capacity: int | None,
                    audit_prior: int | None = None) -> tuple[int | None, str | None]:
    """(the least of the terms, the term that binds); (None, None) while the demand is unknown. ``audit_prior``
    stands in for an unmeasured ``audit_capacity`` (binding ``audit_prior``)."""
    if demand is None:
        return None, None
    prior = audit_prior if audit_capacity is None else None
    terms = [(value, name) for name, value in zip(TERMS, (demand, episodes_left, audit_capacity, prior), strict=True)
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
    target, binding = sessions_target(wanted, left, audit_capacity, audit_prior_sessions(executors_allowed))
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
            "audit_capacity_measured": audit_capacity is not None,
            "sessions_target": target, "binding": binding}


__all__ = ["AUDIT_PRIOR_SESSIONS_PER_EXECUTOR", "NEW_WANT_S", "WANT_FLOOR", "WANT_GROWTH", "WANT_TTL_S", "EpisodeTokens",
           "activity_by_hotkey", "audit_capacity_sessions", "audit_episodes_per_second", "audit_prior_sessions",
           "book_sessions", "bounded_want", "episode_counts", "episode_seconds", "held_by_hotkey", "job_demand",
           "sandbox_status", "sessions_target"]
