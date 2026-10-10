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
and a free slot once the tokens issued since its report are counted. "Least loaded" is
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.attest import heartbeat_valid

from reliquary.sandbox import transcript_cap_refusal
from reliquary.sandbox.machines import DirectorySnapshot, snapshot_from_documents

logger = logging.getLogger(__name__)

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
        self._directory = DirectorySnapshot()
        self._reports: dict[str, dict] = {}
        self._last_ok: dict[str, float] = {}
        self._recorded: dict[str, float] = {}
        self._issued: dict[str, list[int]] = {}
        self._started = clock()
        self.drained: set[str] = set()
        self._egress_suspect = False      # the last poll heard no machine at all
        self._options_alerted: set[tuple[str, str, str | None, str]] = set()

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
        if machine_id in self.drained:
            self.drained.discard(machine_id)
            logger.info("machine %s heartbeats again; it gets sessions again", machine_id)
        return True

    async def poll_once(self) -> None:
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

    def note_issued(self, machine_id: str, at: int) -> None:
        issued = self._issued.setdefault(machine_id, [])
        issued.append(int(at))
        horizon = int(at) - 4 * HEARTBEAT_MAX_AGE_S
        self._issued[machine_id] = [t for t in issued if t >= horizon]

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
             env_options_sha256: str | None = None) -> Placement | None:
        """The machine a new session goes to (see the module docstring), or None. With
        `env_options_sha256`, a machine is eligible only if its report publishes that
        digest for `env` (the gateway is the authority on the env's options)."""
        if not self.directory_ready(now):
            self._alert_stale(now)
            return None
        best: tuple[float, str] | None = None
        chosen: Placement | None = None
        for entry in self._directory.entries():
            if entry.status != "active" or entry.machine_id in self.drained:
                continue
            document = self._reports.get(entry.machine_id)
            if document is None or now - document["at"] > HEARTBEAT_MAX_AGE_S:
                continue
            if image not in (document.get("images") or ()):
                continue
            if (document.get("env_packages") or {}).get(env) != env_package:
                continue
            if env_options_sha256 is not None and self._options_differ(
                    entry.machine_id, document, env, env_options_sha256):
                continue
            if document.get("tools_version") != TOOLS_VERSION:
                continue
            caps = document.get("caps") or {}
            if any(type(caps.get(cap)) is not int or budgets[name] > caps[cap]
                   for name, cap in _CAPS.items()):
                continue
            if type(caps.get("max_token_validity_s")) is not int or validity_s > caps["max_token_validity_s"]:
                continue
            issued = sum(1 for at in self._issued.get(entry.machine_id, ()) if at >= document["at"])
            free = document["free"] - issued
            if free <= 0:
                continue
            key = (-free / max(1, document["capacity"]), entry.machine_id)
            if best is None or key < best:
                best, chosen = key, Placement(entry.machine_id, entry.address)
        return chosen

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
           "HEARTBEAT_RECORD_SECONDS", "HEARTBEAT_RECORD_TIMEOUT_S", "MAX_REPORT_BYTES", "POLL_SECONDS", "STALE_AFTER_S",
           "TOOLS_VERSION", "Fleet", "Placement", "http_fetch_report"]
