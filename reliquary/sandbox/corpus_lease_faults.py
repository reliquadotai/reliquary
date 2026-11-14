"""Draw leases our side failed: never charged to the miner.

A signed-sandbox draw job leases ``(prompt, slot)`` before the episode opens its session. A lease released or
expired unused is charged on a job priced by auction (debt, halved capacity: ``corpus.token_auction``) and counts
against the hotkey's unused leases otherwise. When the episode failed on our side, that charge is ours to bear, as
the RL lease book does for its ``box_failed`` (``validator.rl_leases``). Judged from the issuer's own records only,
never the miner's word, over what happened to that hotkey's prompt since the lease was granted:

* the last session issued for it decides when there is one after our last refusal: ours when our side ended it
  (a drained machine or lost gateway, a signed ``box_failed`` final, an abort the gateway gave because other miners'
  gradings held its grading pool), the miner's otherwise (graded, expired, withdrawn, any other abort);
* else an open our side refused (``SessionIssuer.refused_ours``: the fleet full while the hotkey was within its
  allowance, a task or ledger unreadable, our directory or store down) excuses it.

A ``box_failed`` (which the gateway also gives when the agent broke its own box) is free at most
``BOX_FAILED_FREE_PER_DAY`` times per hotkey across jobs (``FaultBudget``, kept across restarts in the auction
state directory), then charged.

The issuer keys refusals and sessions by ``(hotkey, job, prompt)``, not by lease: a corpus engagement names no lease.
Two leases of one hotkey on one prompt at once (two slots) are judged on the same records.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BOX_FAILED_FREE_PER_DAY = 48
_DAY_S = 86400.0
BUDGET_FILE = "lease-fault-budget.json"


class FaultBudget:
    """The free ``box_failed`` leases of each hotkey in the last day, across jobs, saved to ``path`` (when given)
    at each use so a restart does not reset it."""

    def __init__(self, per_day: int = BOX_FAILED_FREE_PER_DAY, *, path: str | os.PathLike | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.per_day, self._path, self._clock = int(per_day), (Path(path) if path else None), clock
        self._lock = threading.Lock()
        self._used: dict[str, list[float]] = {}
        if self._path is not None:
            try:
                raw = json.loads(self._path.read_text())
                self._used = {str(h): [float(t) for t in ts] for h, ts in (raw.get("used") or {}).items()}
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001 - a corrupt file starts empty, said once
                logger.exception("lease fault budget %s unreadable: starting empty", self._path)

    def take(self, hotkey: str) -> bool:
        """One free lease for ``hotkey`` if its day has one left (and count it)."""
        now = self._clock()
        with self._lock:
            for key in list(self._used):
                kept = [t for t in self._used[key] if t >= now - _DAY_S]
                if kept:
                    self._used[key] = kept
                else:
                    del self._used[key]
            times = self._used.setdefault(str(hotkey), [])
            if len(times) >= self.per_day:
                return False
            times.append(now)
            self._save()
            return True

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"used": self._used}))
            os.replace(temporary, self._path)
        except OSError:
            logger.exception("lease fault budget %s not saved", self._path)


_shared: FaultBudget | None = None
_shared_lock = threading.Lock()


def shared_budget() -> FaultBudget:
    """This process's one budget (every job of a validator), saved in ``RELIQUARY_AUCTION_STATE_DIR`` when set."""
    global _shared
    with _shared_lock:
        if _shared is None:
            from reliquary.corpus.token_auction import STATE_DIR_ENV

            root = os.environ.get(STATE_DIR_ENV)
            _shared = FaultBudget(path=Path(root) / BUDGET_FILE if root else None)
        return _shared


def _ours(record) -> str | None:
    """Why our side ended this corpus session, or None (the miner's)."""
    from reliquary.sandbox.rl_engagements import _session_failure

    cause = _session_failure(record)
    if cause == "aborted" and getattr(record, "fault", None) != "grading_pool":
        return None            # an abort for anything but other miners' gradings: the miner's, as its abort cap
    return cause


def corpus_lease_fault(issuer: Any, job_id: str, *,
                       budget: FaultBudget | None = None) -> Callable[[Any], tuple[str, str] | None]:
    """``DrawLeases``' ``validator_fault`` for one signed job served by ``issuer``."""

    def fault(lease) -> tuple[str, str] | None:
        refused = issuer.refused_ours(lease.hotkey, job_id, lease.prompt_index, lease.granted_at)
        records = ()
        for attempt in range(3):
            try:
                records = issuer.book.of_hotkey(lease.hotkey)
                break
            except RuntimeError:                  # the index changed during the read (another thread)
                if attempt == 2:
                    raise
        mine = [r for r in records if r.job_id == job_id and r.prompt_index == lease.prompt_index
                and r.issued_at >= int(lease.granted_at)]
        if refused is not None and not any(r.issued_at >= refused[0] for r in mine):
            # Nothing ran after our refusal: the episode could not run.
            return "open_refused_ours", f"an open of prompt {lease.prompt_index} was refused {refused[1]}"
        if not mine:
            return None
        last = max(mine, key=lambda r: (r.issued_at, r.session_id))
        cause = _ours(last)
        if cause is None:
            return None
        if cause == "box_failed" and not (budget or shared_budget()).take(lease.hotkey):
            logger.warning("draw lease %s of %s: box_failed past the %d free a day: charged", lease.lease_id[:8],
                           lease.hotkey[:12], (budget or shared_budget()).per_day)
            return None
        return cause, f"session {last.session_id} on {last.machine_id}: {last.state}/{last.closed_status}"

    return fault


__all__ = ["BOX_FAILED_FREE_PER_DAY", "FaultBudget", "corpus_lease_fault", "shared_budget"]
