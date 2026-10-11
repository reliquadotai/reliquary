"""The validator's live view of the sandbox machines (spec §7).

Every POLL_SECONDS the validator fetches each listed machine's signed capacity report
from the machine's DIRECTORY address (`GET {address}/capacity`; the push to
HEARTBEAT_URL is not used, so the validator exposes no heartbeat endpoint). A report
counts only when `heartbeat_valid` accepts it with the validator's clock and
HEARTBEAT_MAX_AGE_S, it names its own machine, and its signed `public_base_url` is the
directory's address. A machine with no accepted report for STALE_AFTER_S is drained:
no new session, and `on_drained(machine_id)` voids its live sessions with no fault to
their miners. A session goes to the least-loaded eligible machine: active in the
directory, fresh, holding the image and the env package at the pinned version (and,
when the task names one, serving the env with the options digest the validator resolved
it with), the tools version this build renders, caps no lower than the budgets and token validity,
and a free slot once the tokens issued since its report are counted; for an RL session, also CPU and memory
room on a box at least the task's size (its report's `resources`, with the reservations issued here and not yet
in a report: `pick(reserve=)`). "Least loaded" is
the largest FREE FRACTION (free / capacity); equal fractions go to the smallest machine
id, so ties bias toward lexicographically small ids (deterministic, not balanced). A machine whose
`max_transcript_bytes` is refused by `transcript_cap_refusal` is never placed.

The directory snapshot (keys, addresses, statuses) is rebuilt from R2 every
`directory_refresh_s` (setting `sandbox_directory_refresh_s`, default
DIRECTORY_REFRESH_SECONDS): a revoked machine or an ended key stops counting within that
bound. Each read is bounded by `directory_read_timeout_s` (default
DIRECTORY_READ_TIMEOUT_SECONDS) and runs beside the polls, never in front of them, so a
hung R2 read cannot stop heartbeats or drain checks. A failed read is retried with
exponential backoff (DIRECTORY_BACKOFF_BASE_SECONDS doubling, capped at
DIRECTORY_BACKOFF_CAP_SECONDS), one error line per attempt. The snapshot's age is kept. Once it is older than `directory_max_age_s` (default
DIRECTORY_MAX_AGE_SECONDS), whether it was never read or the R2 reads keep failing,
the fleet fails closed: `directory()` is an empty snapshot (no key verifies, so no
heartbeat and no transcript does), `pick` places nothing, polling stops (no machine is
drained for the validator's own outage), and an error is logged (before the first read
attempt has finished, only "not loaded yet" at info). It reopens on the first successful
read. A machine's silence is counted from the latest of the fleet's start, the moment
it entered the directory, and the moment the directory last became usable.

Our own egress. When a poll gets no accepted report from ANY listed machine, the
validator suspects its own network rather than every machine at once: nothing is
drained (an ALERT is logged once) until at least one machine reports again, and the
silence of the others is then counted from that recovery. A single-machine fleet is
therefore never drained for silence (it still gets no new session: placement needs a
fresh report); its sessions end by their close or their lapse.

Heartbeat summaries (observability) are written to R2 at most once a minute per
machine, each write bounded by HEARTBEAT_RECORD_TIMEOUT_S so a slow R2 never stalls a
poll.

A key end backdated by the operator (compromise) refuses signatures from the next
refresh on. It does NOT reach transcripts that were already admitted: those were
verified against the snapshot of their time and stay admitted (and paid); undoing
them is a manual, out-of-band decision.

Polling is concurrent and bounded per machine: each fetch has FETCH_TIMEOUT_S in total
(a machine that drips bytes cannot stall the poll or the others) and reads at most
MAX_REPORT_BYTES (a machine cannot exhaust the validator's memory). Redirects are not
followed: the report comes from the directory's address or not at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.attest import heartbeat_valid

from reliquary.sandbox import transcript_cap_refusal
from reliquary.sandbox.machines import DirectorySnapshot, snapshot_from_documents

logger = logging.getLogger(__name__)

OPEN_REFUSAL_SKEW_S = 120
"""How far a signed open refusal's time may lie outside its token's validity (clock skew)."""

HEARTBEAT_MAX_AGE_S = 30
STALE_AFTER_S = 60
POLL_SECONDS = 5.0
DIRECTORY_REFRESH_SECONDS = 30.0
DIRECTORY_MAX_AGE_SECONDS = 120.0
DIRECTORY_READ_TIMEOUT_SECONDS = 15.0
DIRECTORY_BACKOFF_BASE_SECONDS = 5.0
DIRECTORY_BACKOFF_CAP_SECONDS = 30.0
HEARTBEAT_RECORD_SECONDS = 60.0
HEARTBEAT_RECORD_TIMEOUT_S = 5.0
TOOLS_VERSION = "reliquary-tools/1"
# Per machine, per poll: the whole fetch (connect, headers, body) must fit in this.
FETCH_TIMEOUT_S = 4.0
# A capacity report is a few KiB (its image list dominates); anything past this is refused.
MAX_REPORT_BYTES = 1024 * 1024
# Budget name -> the cap a capacity report states for it.
_CAPS = {"max_calls": "max_calls", "cpu_s": "max_cpu_s", "per_call_timeout_s": "max_call_timeout_s",
         "wall_s": "max_wall_s", "memory_bytes": "max_memory_bytes", "pids": "max_pids",
         "disk_bytes": "max_disk_bytes"}
# The fields of a report's ``resources`` block an RL placement reads (all non-negative ints).
_RESOURCE_FIELDS = ("cpu_millis", "memory_bytes", "free_cpu_millis", "free_memory_bytes", "max_box_cpu_millis",
                    "boxes_per_episode")
MIN_BOX_CPU_MILLIS = 100          # the gateway never sizes a box below 0.1 CPU
def reservation_need(resources: Mapping[str, Any], reserve: Mapping[str, Any]) -> tuple[int, int] | None:
    """(CPU millis, memory bytes) a session with ``reserve`` takes on a machine reporting ``resources`` (the
    gateway's own rule: its agent box of the task's CPUs, at least 0.1 CPU, and its memory budget), or None when
    the machine's box is smaller than the task declares (never clamped: such a machine never gets it)."""
    box = int(resources["max_box_cpu_millis"])
    asked = reserve.get("cpu_millis")
    if asked is not None and int(asked) > box:
        return None
    per_box = box if asked is None else max(MIN_BOX_CPU_MILLIS, int(asked))
    boxes = int(resources["boxes_per_episode"])
    return boxes * per_box, boxes * int(reserve["memory_bytes"])


def _valid_resources(value: Any) -> bool:
    return (isinstance(value, Mapping)
            and all(type(value.get(name)) is int and value[name] >= 0 for name in _RESOURCE_FIELDS)
            and value["cpu_millis"] > 0 and value["memory_bytes"] > 0 and value["boxes_per_episode"] > 0
            and value["max_box_cpu_millis"] > 0)


def _listed(resources: Any) -> frozenset[str]:
    """The live session ids a ``resources`` block names (its ``used_*`` counts them)."""
    listed = resources.get("sessions") if isinstance(resources, Mapping) else None
    if not isinstance(listed, list):
        return frozenset()
    return frozenset(s for s in listed if isinstance(s, str))


@dataclass(frozen=True)
class Placement:
    machine_id: str
    address: str


async def http_fetch_report(address: str, *, timeout: float = FETCH_TIMEOUT_S,
                            max_bytes: int | None = None, transport: Any = None) -> dict | None:
    """`GET {address}/capacity`, or None on any failure. At most `max_bytes` are read
    (MAX_REPORT_BYTES by default) and the whole exchange is bounded by `timeout`."""
    import httpx

    limit = MAX_REPORT_BYTES if max_bytes is None else max_bytes

    async def fetch() -> dict | None:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False,
                                     transport=transport) as client:
            async with client.stream("GET", f"{address.rstrip('/')}/capacity",
                                     headers={"accept-encoding": "identity"}) as response:
                if response.status_code != 200:
                    return None
                # Raw bytes only: a compressed body could expand past the cap in one chunk.
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    return None
                announced = response.headers.get("content-length")
                if announced is not None and (not announced.isdigit() or int(announced) > limit):
                    return None
                body = bytearray()
                async for chunk in response.aiter_raw():
                    body += chunk
                    if len(body) > limit:
                        return None
        return json.loads(bytes(body))

    try:
        return await asyncio.wait_for(fetch(), timeout=timeout)
    except (httpx.HTTPError, ValueError, RecursionError, TimeoutError):
        return None


class Fleet:
    def __init__(self, *, read_documents: Callable[[], Awaitable[list[dict]]],
                 fetch_report: Callable[[str], Awaitable[dict | None]] = http_fetch_report,
                 clock: Callable[[], float] = time.time,
                 record_heartbeat: Callable[..., Awaitable[Any]] | None = None,
                 on_drained: Callable[[str], None] | None = None,
                 directory_refresh_s: float = DIRECTORY_REFRESH_SECONDS,
                 directory_max_age_s: float = DIRECTORY_MAX_AGE_SECONDS,
                 directory_read_timeout_s: float = DIRECTORY_READ_TIMEOUT_SECONDS) -> None:
        if not directory_refresh_s > 0 or not directory_max_age_s > directory_refresh_s:
            raise ValueError("the directory needs 0 < refresh period < max age, got "
                             f"{directory_refresh_s} and {directory_max_age_s}")
        if not directory_read_timeout_s > 0:
            raise ValueError(f"the directory read timeout must be positive, got "
                             f"{directory_read_timeout_s}")
        self.directory_read_timeout_s = float(directory_read_timeout_s)
        self._next_refresh_at: float | None = None   # None: due now
        self._refresh_failures = 0
        self._read_attempted = False                 # the first read has finished (either way)
        self._not_loaded_logged = False
        self._first_seen: dict[str, float] = {}
        self._refresh_s = float(directory_refresh_s)
        self._max_age_s = float(directory_max_age_s)
        self._directory_at: float | None = None      # clock time of the last good read
        self._fresh_since: float | None = None       # when the snapshot last became usable
        self._stale_alerted = False
        self._read_documents = read_documents
        self._fetch = fetch_report
        self._clock = clock
        self._record = record_heartbeat
        self.on_drained = on_drained
        # Called after every poll, early returns included (the RL open queue's wake-up).
        self.on_polled: Callable[[], None] | None = None
        self._directory = DirectorySnapshot()
        self._reports: dict[str, dict] = {}
        self._last_ok: dict[str, float] = {}
        self._recorded: dict[str, float] = {}
        self._issued: dict[str, list[int]] = {}
        # RL reservations issued here and not yet in a report: machine -> [(at, until, reserve, session id)]
        self._reserved: dict[str, list[tuple[int, int, dict, str | None]]] = {}
        # RL placement only on machines reporting a valid ``resources`` block (the production RL wiring sets it).
        self.resources_required = False
        self._no_resources_logged: set[str] = set()
        self._listed: dict[str, float] = {}     # session id -> when an accepted report first listed it live
        self._started = clock()
        self.drained: set[str] = set()
        self._egress_suspect = False      # the last poll heard no machine at all
        self._options_alerted: set[tuple[str, str, str | None, str]] = set()
        self._refusal_alerted = False     # the "cannot verify open refusals" error is logged once

    def directory_age(self, now: float | None = None) -> float | None:
        """Seconds since the last successful directory read, or None if never read."""
        if self._directory_at is None:
            return None
        return (self._clock() if now is None else now) - self._directory_at

    def directory_ready(self, now: float | None = None) -> bool:
        """Whether the snapshot is young enough to place sessions and admit transcripts."""
        age = self.directory_age(now)
        return age is not None and age <= self._max_age_s

    def directory(self) -> DirectorySnapshot:
        """The snapshot to verify against, or an empty one (fail closed) when it is
        older than the max age."""
        if not self.directory_ready():
            self._alert_stale(self._clock())
            return _EMPTY_DIRECTORY
        return self._directory

    def directory_if_ready(self, now: float | None = None) -> DirectorySnapshot | None:
        """The snapshot, or None when it is not young enough: one call for a readiness
        check and the verify that follows it, so they cannot see two directories. The
        read time is taken before the snapshot (a refresh assigns the snapshot first),
        so a fresh time never pairs with an older snapshot."""
        at = self._directory_at
        snapshot = self._directory
        now = self._clock() if now is None else now
        if at is None or now - at > self._max_age_s:
            self._alert_stale(now)
            return None
        return snapshot

    def _alert_stale(self, now: float) -> None:
        if not self._read_attempted:
            if not self._not_loaded_logged:
                self._not_loaded_logged = True
                logger.info("sandbox machine directory not loaded yet; no session is placed")
            return
        if self._stale_alerted:
            return
        self._stale_alerted = True
        age = self.directory_age(now)
        logger.error("ALERT sandbox machine directory stale (%s): no session is placed and "
                     "no signed transcript is admitted until R2 is read again",
                     "never read" if age is None else f"last read {age:.0f} s ago, "
                     f"max {self._max_age_s:.0f} s")

    async def refresh_directory(self) -> None:
        """Read the directory once, within the read timeout (TimeoutError past it)."""
        try:
            documents = await asyncio.wait_for(self._read_documents(),
                                               timeout=self.directory_read_timeout_s)
        finally:
            self._read_attempted = True
        snapshot = snapshot_from_documents(documents)
        now = self._clock()
        for entry in snapshot.entries():
            self._first_seen.setdefault(entry.machine_id, now)
        if not self.directory_ready(now):
            self._fresh_since = now
            if self._stale_alerted:
                logger.warning("sandbox machine directory read again; placement reopens")
        self._directory, self._directory_at, self._stale_alerted = snapshot, now, False

    def accept_report(self, machine_id: str, report: Any, now: float) -> bool:
        if not self.directory_ready(now):
            self._alert_stale(now)
            return False
        entry = self._directory.entry(machine_id)
        if entry is None:
            return False
        check = heartbeat_valid(report, self._directory, now=int(now), max_age_s=HEARTBEAT_MAX_AGE_S)
        if not check:
            logger.warning("heartbeat of machine %s refused: %s", machine_id, check.reason.value)
            return False
        document = report["document"]
        refusal = transcript_cap_refusal(document.get("caps"))
        if refusal is not None:
            logger.warning("heartbeat of machine %s refused: %s", machine_id, refusal)
            return False
        if (document.get("machine_id") != machine_id
                or str(document.get("public_base_url") or "").rstrip("/") != entry.address):
            logger.warning("heartbeat of machine %s names another machine or address; refused",
                           machine_id)
            return False
        self._reports[machine_id] = dict(document)
        self._last_ok[machine_id] = now
        listed = _listed(document.get("resources"))
        if listed:
            for session_id in listed:
                self._listed.setdefault(session_id, now)   # it reached its gateway: its miner had the token
            if len(self._listed) > 200_000:                  # bounded: forget what was first listed a day ago
                self._listed = {sid: at for sid, at in self._listed.items() if at > now - 86_400}
        if machine_id in self.drained:
            self.drained.discard(machine_id)
            logger.info("machine %s heartbeats again; it gets sessions again", machine_id)
        return True

    async def poll_once(self) -> None:
        try:
            await self._poll_once()
        finally:
            if self.on_polled is not None:
                try:
                    self.on_polled()
                except Exception:
                    logger.exception("the poll listener failed")

    async def _poll_once(self) -> None:
        now = self._clock()
        if not self.directory_ready(now):
            # Nothing can be verified; and a machine is not drained for our own outage.
            self._alert_stale(now)
            return
        entries = [e for e in self._directory.entries() if e.status != "revoked"]
        reports = await asyncio.gather(*(self._bounded_fetch(e.address) for e in entries),
                                       return_exceptions=True)
        heard = 0
        for entry, report in zip(entries, reports):
            if report is None or isinstance(report, BaseException):
                continue
            if self.accept_report(entry.machine_id, report, now):
                heard += 1
                await self._maybe_record(entry.machine_id, now)
        if entries and heard == 0:
            # Every machine silent in the same poll: our egress, most likely. Drain
            # nothing until one of them is heard again.
            if not self._egress_suspect:
                self._egress_suspect = True
                logger.error("ALERT no sandbox machine answered this poll (%d listed): "
                             "suspecting this validator's own egress; no machine is "
                             "drained until one answers again", len(entries))
            return
        if self._egress_suspect:
            self._egress_suspect = False
            # Silence is counted from the recovery, not from before the outage.
            self._fresh_since = now
            logger.warning("sandbox machines answer again; drain checks resume")
        for entry in entries:
            # Silence is counted from when the directory last became usable at the earliest.
            baseline = max(self._started, self._first_seen.get(entry.machine_id, self._started))
            last = max(self._last_ok.get(entry.machine_id, baseline),
                       self._fresh_since if self._fresh_since is not None else self._started)
            if now - last > STALE_AFTER_S and entry.machine_id not in self.drained:
                self.drained.add(entry.machine_id)
                logger.warning("machine %s has sent no valid heartbeat for %.0f s: drained",
                               entry.machine_id, now - last)
                if self.on_drained is not None:
                    try:
                        self.on_drained(entry.machine_id)
                    except Exception:
                        logger.exception("draining machine %s failed", entry.machine_id)

    async def _bounded_fetch(self, address: str) -> Any:
        # Bounded here as well as in http_fetch_report, so an injected fetcher that hangs
        # costs one machine its heartbeat, never the poll.
        try:
            return await asyncio.wait_for(self._fetch(address), timeout=FETCH_TIMEOUT_S)
        except TimeoutError:
            return None

    async def _maybe_record(self, machine_id: str, now: float) -> None:
        if self._record is None or now - self._recorded.get(machine_id, -math.inf) < HEARTBEAT_RECORD_SECONDS:
            return
        self._recorded[machine_id] = now
        document = self._reports[machine_id]
        summary = {name: document.get(name) for name in
                   ("capacity", "active", "free", "env_packages", "tools_version", "runsc_version")}
        summary["images"] = len(document.get("images") or ())
        try:
            await asyncio.wait_for(self._record(machine_id, at=now, summary=summary),
                                   HEARTBEAT_RECORD_TIMEOUT_S)
        except Exception as exc:  # observability only (a timeout included)
            logger.warning("heartbeat summary of %s not written: %s", machine_id, type(exc).__name__)

    def note_issued(self, machine_id: str, at: int, *, reserve: Mapping[str, Any] | None = None,
                    hold_s: int | None = None, session_id: str | None = None) -> None:
        """A session was issued on the machine at ``at``: a slot until its next report. With ``reserve`` (RL),
        its CPU and memory are also held here until a report lists the session (its ``used`` counts it then),
        the session leaves the book (``note_settled``), or a report made after ``at + hold_s`` (the token's
        present window: the gateway refuses it later) does not list it, whichever comes first."""
        issued = self._issued.setdefault(machine_id, [])
        issued.append(int(at))
        horizon = int(at) - 4 * HEARTBEAT_MAX_AGE_S
        self._issued[machine_id] = [t for t in issued if t >= horizon]
        if reserve is not None:
            until = int(at) + int(hold_s if hold_s is not None else 4 * HEARTBEAT_MAX_AGE_S)
            # A hold past its window counts only while the report before it is fresh: forget older ones.
            reserved = [r for r in self._reserved.get(machine_id, ()) if r[1] >= int(at) - HEARTBEAT_MAX_AGE_S]
            reserved.append((int(at), until, dict(reserve), session_id))
            self._reserved[machine_id] = reserved

    def note_settled(self, session_id: str) -> None:
        """The session left the book (closed, lapsed, voided): its reservation is no longer held here; the
        gateway's next report is the truth. A session holding nothing here (corpus) changes nothing."""
        for machine_id, reserved in list(self._reserved.items()):
            kept = [r for r in reserved if r[3] != session_id]
            if len(kept) != len(reserved):
                self._reserved[machine_id] = kept

    def session_reported(self, session_id: str) -> bool:
        """Whether a report this process accepted listed the session live on its machine: its miner held the
        token (a restart did not lose the open's answer; the session held its box)."""
        return session_id in self._listed

    def _resources(self, machine_id: str, document: Mapping[str, Any]) -> Mapping[str, Any] | None:
        resources = document.get("resources")
        if _valid_resources(resources):
            return resources
        if machine_id not in self._no_resources_logged:
            self._no_resources_logged.add(machine_id)
            logger.error("sandbox machine %s reports no valid resources: it gets no RL session "
                         "(set episode_cpu_millis_total and episode_memory_bytes_total on it)", machine_id)
        return None

    def _by_resources(self, document: Mapping[str, Any], reserve: Mapping[str, Any] | None) -> bool:
        """Whether this placement is judged by CPU and memory (an RL session, on a machine that reports them or
        once they are required) rather than by slots."""
        return reserve is not None and (self.resources_required or "resources" in document)

    def _margin(self, machine_id: str, document: Mapping[str, Any], reserve: Mapping[str, Any]) -> float | None:
        """The machine's remaining fraction (the scarcer of CPU and memory) once this session is placed; negative
        when it has no room now; None when it never could (no valid ``resources``, a box smaller than the task,
        totals below the need). The reservations issued here count until a report shows them (see
        ``note_issued``): never twice past a report that lists them, never zero times before."""
        resources = self._resources(machine_id, document)
        if resources is None:
            return None
        need = reservation_need(resources, reserve)
        if need is None or need[0] > resources["cpu_millis"] or need[1] > resources["memory_bytes"]:
            return None
        report_at = document["at"]
        held = [r for r in self._reserved.get(machine_id, ()) if r[3] not in self._listed and report_at <= r[1]]
        cpu = memory = 0
        for _at, _until, other, _sid in held:
            taken = reservation_need(resources, other) or (0, 0)
            cpu, memory = cpu + taken[0], memory + taken[1]
        free_cpu = resources["free_cpu_millis"] - cpu - need[0]
        free_memory = resources["free_memory_bytes"] - memory - need[1]
        if free_cpu < 0 or free_memory < 0:
            return min(free_cpu, 0) / resources["cpu_millis"] + min(free_memory, 0) / resources["memory_bytes"]
        return min(free_cpu / resources["cpu_millis"], free_memory / resources["memory_bytes"])

    def session_capacity(self, now: float | None = None) -> int:
        """The session slots (``capacity`` of their last report) of the fresh, active, undrained machines: the
        fleet size the RL yield quota shares. Memory only."""
        now = self._clock() if now is None else now
        if not self.directory_ready(now):
            return 0
        total = 0
        for entry in self._directory.entries():
            if entry.status != "active" or entry.machine_id in self.drained:
                continue
            document = self._reports.get(entry.machine_id)
            if document is None or now - document["at"] > HEARTBEAT_MAX_AGE_S:
                continue
            total += max(0, int(document.get("capacity") or 0))
        return total

    def _options_differ(self, machine_id: str, document: Mapping[str, Any], env: str,
                        expected: str) -> bool:
        """Whether the machine serves `env` with other options than the ones this
        validator resolves its tasks with (or publishes none for it): logged once per
        (machine, env, published, expected), at error level."""
        published = (document.get("env_options_sha256") or {}).get(env)
        if published == expected:
            return False
        key = (machine_id, env, published, expected)
        if key not in self._options_alerted:
            self._options_alerted.add(key)
            logger.error(
                "machine %s is refused for %s: it %s, this validator builds the env with "
                "options sha256 %s; the gateway's episode_env_options and the validator's "
                "options must agree (no session is placed there until they do)",
                machine_id, env,
                "publishes no options digest for it" if published is None
                else f"serves it with options sha256 {published}", expected)
        return True

    def pick(self, *, image: str, env: str, env_package: str, budgets: Mapping[str, int],
             validity_s: int, now: float,
             env_options_sha256: str | None = None, exclude: Collection[str] = (),
             reserve: Mapping[str, Any] | None = None) -> Placement | None:
        """The machine a new session goes to (see the module docstring), or None. With
        `env_options_sha256`, a machine is eligible only if its report publishes that
        digest for `env` (the gateway is the authority on the env's options). `exclude`: machines
        held for another open (the RL open queue). With `reserve` (RL), a machine reporting `resources` (every
        machine, once `resources_required`) must have room for the session's CPU and memory on a box at least the
        task's CPUs; the least loaded is then the one with the largest remaining fraction."""
        if not self.directory_ready(now):
            self._alert_stale(now)
            return None
        best: tuple[float, str] | None = None
        chosen: Placement | None = None
        for entry in self._directory.entries():
            if entry.machine_id in exclude:
                continue
            document = self._reports.get(entry.machine_id)
            if not self._fits(entry, document, image=image, env=env, env_package=env_package, budgets=budgets,
                              validity_s=validity_s, now=now, env_options_sha256=env_options_sha256):
                continue
            free = self._free(entry.machine_id, document)
            if free <= 0:
                continue
            if self._by_resources(document, reserve):
                margin = self._margin(entry.machine_id, document, reserve)
                if margin is None or margin < 0:
                    continue
                key = (-margin, entry.machine_id)
            else:
                key = (-free / max(1, document["capacity"]), entry.machine_id)
            if best is None or key < best:
                best, chosen = key, Placement(entry.machine_id, entry.address)
        return chosen

    def _free(self, machine_id: str, document: Mapping[str, Any]) -> int:
        issued = sum(1 for at in self._issued.get(machine_id, ()) if at >= document["at"])
        return document["free"] - issued

    def _fits(self, entry, document, *, image: str, env: str, env_package: str, budgets: Mapping[str, int],
              validity_s: int, now: float, env_options_sha256: str | None) -> bool:
        """The checks of a machine that do not depend on its free room (shared by ``pick`` and ``could_place``)."""
        if entry.status != "active" or entry.machine_id in self.drained:
            return False
        if document is None or now - document["at"] > HEARTBEAT_MAX_AGE_S:
            return False
        return self._serves(entry, document, image=image, env=env, env_package=env_package, budgets=budgets,
                            validity_s=validity_s, env_options_sha256=env_options_sha256)

    def _serves(self, entry, document, *, image: str, env: str, env_package: str, budgets: Mapping[str, int],
                validity_s: int, env_options_sha256: str | None) -> bool:
        """What the machine's (last) report says it serves: the image, the env, its options, the tools and caps."""
        if image not in (document.get("images") or ()):
            return False
        if (document.get("env_packages") or {}).get(env) != env_package:
            return False
        if env_options_sha256 is not None and self._options_differ(
                entry.machine_id, document, env, env_options_sha256):
            return False
        if document.get("tools_version") != TOOLS_VERSION:
            return False
        caps = document.get("caps") or {}
        if any(type(caps.get(cap)) is not int or budgets[name] > caps[cap] for name, cap in _CAPS.items()):
            return False
        if type(caps.get("max_token_validity_s")) is not int or validity_s > caps["max_token_validity_s"]:
            return False
        return True

    def could_place(self, *, image: str, env: str, env_package: str, budgets: Mapping[str, int], validity_s: int,
                    now: float, env_options_sha256: str | None = None,
                    reserve: Mapping[str, Any] | None = None) -> bool | None:
        """Whether some machine could ever take this session, its load aside: True (a fresh active machine
        serves it), False (no active machine serves it, by what every one of them reports) or None when it
        cannot be judged now: the directory is stale, no machine report was accepted yet (a start) or the last
        poll heard none (our egress), no report is fresh, or a machine whose last report serves it is drained or
        stale. The RL open queue refuses at once only on False (final for the seed); None is retryable. With
        `reserve` (RL), a machine judged by its `resources` (see `pick`) serves the session only if its box is at
        least the task's CPUs and its totals hold the session's CPU and memory."""
        if not self.directory_ready(now) or self._egress_suspect or not self._last_ok:
            return None
        fresh = unknown = False
        kw = dict(image=image, env=env, env_package=env_package, budgets=budgets, validity_s=validity_s,
                  env_options_sha256=env_options_sha256)
        for entry in self._directory.entries():
            document = self._reports.get(entry.machine_id)
            if entry.status != "active" or document is None:
                continue
            current = entry.machine_id not in self.drained and now - document["at"] <= HEARTBEAT_MAX_AGE_S
            fresh = fresh or current
            if self._serves(entry, document, **kw) and self._could_hold(entry.machine_id, document, reserve):
                if current:
                    return True
                unknown = True              # it served it when last heard: drained or stale for now
        return None if unknown or not fresh else False

    def _could_hold(self, machine_id: str, document: Mapping[str, Any], reserve: Mapping[str, Any] | None) -> bool:
        """Whether the machine could ever hold this session by its resources (its box size and totals, its load
        aside); always True for a placement judged by slots."""
        if not self._by_resources(document, reserve):
            return True
        resources = self._resources(machine_id, document)
        if resources is None:
            return False
        need = reservation_need(resources, reserve)
        return need is not None and need[0] <= resources["cpu_millis"] and need[1] <= resources["memory_bytes"]

    def nearest(self, *, image: str, env: str, env_package: str, budgets: Mapping[str, int], validity_s: int,
                now: float, env_options_sha256: str | None = None,
                reserve: Mapping[str, Any] | None = None) -> str | None:
        """The fresh machine serving this session that is closest to having room for it (the most free room,
        full or not): the one the open queue holds for a head waiter that keeps missing. With `reserve` (RL), room
        is CPU and memory (see `pick`), and a machine that could never hold the session is not one."""
        best: tuple[float, str] | None = None
        for entry in self._directory.entries():
            document = self._reports.get(entry.machine_id)
            if not self._fits(entry, document, image=image, env=env, env_package=env_package, budgets=budgets,
                              validity_s=validity_s, now=now, env_options_sha256=env_options_sha256):
                continue
            if self._by_resources(document, reserve):
                margin = self._margin(entry.machine_id, document, reserve)
                if margin is None:
                    continue
                key = (-margin, entry.machine_id)
            else:
                key = (-self._free(entry.machine_id, document) / max(1, document["capacity"]), entry.machine_id)
            if best is None or key < best:
                best = key
        return None if best is None else best[1]

    def open_refusal_valid(self, receipt: Mapping[str, Any], record: Any) -> bool | None:
        """Whether ``receipt`` is the gateway's signed refusal of ``record``'s token on its machine, signed
        while that token was valid. None while the directory is not ready (the close is then refused
        retryably); False when this reliquary-sandbox cannot verify receipts (one error log)."""
        now = int(self._clock())
        directory = self.directory_if_ready(now)
        if directory is None:
            return None
        try:
            from reliquary_sandbox.attest import open_refusal_valid
        except ImportError:
            if not self._refusal_alerted:
                self._refusal_alerted = True
                logger.error("this reliquary-sandbox cannot verify signed open refusals: re-pin it")
            return False
        document = receipt.get("document") if isinstance(receipt, Mapping) else None
        at = document.get("at") if isinstance(document, Mapping) else None
        if (type(at) is not int or at < int(record.issued_at) - OPEN_REFUSAL_SKEW_S
                or at > int(record.expires_at) + OPEN_REFUSAL_SKEW_S):
            return False             # a gateway signs only while the token is admissible
        return bool(open_refusal_valid(receipt, directory, machine_id=record.machine_id,
                                       session_id=record.session_id, token_sha256=record.token_sha256,
                                       now=now))

    async def refresh_if_due(self) -> None:
        """Read the directory if its period (or, after failures, its backoff) has
        elapsed. Never raises; one error line per failed attempt."""
        now = self._clock()
        if self._next_refresh_at is not None and now < self._next_refresh_at:
            return
        try:
            await self.refresh_directory()
        except Exception as exc:
            self._refresh_failures += 1
            delay = min(DIRECTORY_BACKOFF_CAP_SECONDS,
                        DIRECTORY_BACKOFF_BASE_SECONDS * 2 ** (self._refresh_failures - 1))
            self._next_refresh_at = self._clock() + delay
            logger.error("sandbox machine directory read failed (%s, attempt %d); next try in "
                         "%.0f s; keeping the last snapshot until it is %.0f s old",
                         type(exc).__name__, self._refresh_failures, delay, self._max_age_s)
            return
        self._refresh_failures = 0
        self._next_refresh_at = self._clock() + self._refresh_s

    async def step(self) -> None:
        """One sequential tick: poll the machines, then refresh the directory if due.
        `run` runs the two side by side instead, so a slow read never delays a poll."""
        await self._poll_guarded()
        await self.refresh_if_due()

    async def _poll_guarded(self) -> None:
        try:
            await self.poll_once()
        except Exception:
            logger.exception("machine heartbeat poll failed")

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Poll every POLL_SECONDS and refresh the directory on its own schedule, in two
        independent loops; both end (an in-flight read is cancelled) when `stop` is set."""
        async def every_tick(action: Callable[[], Awaitable[None]]) -> None:
            while stop is None or not stop.is_set():
                await action()
                if stop is None:
                    await asyncio.sleep(POLL_SECONDS)
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
                    except TimeoutError:
                        pass

        loops = [asyncio.create_task(every_tick(self.refresh_if_due)),
                 asyncio.create_task(every_tick(self._poll_guarded))]
        try:
            if stop is None:
                await asyncio.gather(*loops)
            else:
                await stop.wait()
        finally:
            for loop in loops:
                loop.cancel()
            await asyncio.gather(*loops, return_exceptions=True)


_EMPTY_DIRECTORY = DirectorySnapshot()

__all__ = ["DIRECTORY_BACKOFF_BASE_SECONDS", "DIRECTORY_BACKOFF_CAP_SECONDS",
           "DIRECTORY_MAX_AGE_SECONDS", "DIRECTORY_READ_TIMEOUT_SECONDS", "DIRECTORY_REFRESH_SECONDS", "FETCH_TIMEOUT_S", "HEARTBEAT_MAX_AGE_S",
           "HEARTBEAT_RECORD_SECONDS", "HEARTBEAT_RECORD_TIMEOUT_S", "MAX_REPORT_BYTES", "MIN_BOX_CPU_MILLIS", "POLL_SECONDS",
           "STALE_AFTER_S", "TOOLS_VERSION", "Fleet", "Placement", "http_fetch_report", "reservation_need"]
