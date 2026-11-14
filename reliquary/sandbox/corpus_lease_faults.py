"""Draw leases our side failed: never charged to the miner.

A signed-sandbox draw job leases ``(prompt, slot)`` before the episode opens its session. A lease released or
expired unused is charged on a job priced by auction (debt, halved capacity: ``corpus.token_auction``) and counts
against the hotkey's unused leases otherwise. When the episode failed on our side, that charge is ours to bear, as
the RL lease book does for its ``box_failed`` (``validator.rl_leases``):

* an open of that prompt our side refused since the grant (``SessionIssuer.refused_ours``: the fleet full or
  short, a task or ledger unreadable, our directory or store down);
* the last session of that prompt since the grant ended by our side (``rl_engagements._session_failure``: a
  drained machine or lost gateway, a signed ``box_failed`` or ``aborted`` final).

Only the issuer's own records count, never the miner's word. A ``box_failed`` or ``aborted`` (which the gateway also
gives when the agent broke its own box) is free at most ``BOX_FAILED_FREE_PER_DAY`` times per hotkey and job, then
charged, as in RL.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

BOX_FAILED_FREE_PER_DAY = 48
_DAY_S = 86400.0
# Causes the agent can also bring on itself: budgeted.
BUDGETED = frozenset({"box_failed", "aborted"})


def corpus_lease_fault(issuer: Any, job_id: str, *, free_per_day: int = BOX_FAILED_FREE_PER_DAY,
                       clock: Callable[[], float] = time.time) -> Callable[[Any], tuple[str, str] | None]:
    """``DrawLeases``' ``validator_fault`` for one signed job served by ``issuer``."""
    from reliquary.sandbox.rl_engagements import _session_failure

    used: dict[str, collections.deque] = {}
    lock = threading.Lock()

    def budgeted(hotkey: str, cause: str) -> bool:
        now = clock()
        with lock:
            times = used.setdefault(hotkey, collections.deque())
            while times and times[0] < now - _DAY_S:
                times.popleft()
            if len(times) >= free_per_day:
                return False
            times.append(now)
            return True

    def fault(lease) -> tuple[str, str] | None:
        refused = issuer.refused_ours(lease.hotkey, job_id, lease.prompt_index, lease.granted_at)
        if refused is not None:
            return "open_refused_ours", f"an open of prompt {lease.prompt_index} was refused {refused}"
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
        if not mine:
            return None
        last = max(mine, key=lambda r: (r.issued_at, r.session_id))
        cause = _session_failure(last)
        if cause is None:
            return None
        if cause in BUDGETED and not budgeted(lease.hotkey, cause):
            logger.warning("draw lease %s of %s: %s past the %d free a day: charged", lease.lease_id[:8],
                           lease.hotkey[:12], cause, free_per_day)
            return None
        return cause, f"session {last.session_id} on {last.machine_id}: {last.state}/{last.closed_status}"

    return fault


__all__ = ["BOX_FAILED_FREE_PER_DAY", "corpus_lease_fault"]
