"""Opt-in RL service state (service-contract/v2), on the existing window/proof/weight clocks.

``ServiceRuntime`` is the validator-side state of one RL service order. It composes three
modules over ONE SQLite connection (WAL), serialized by ``ServiceRuntime.lock``:

* ``run_log.RunObservationLog``    the run-wide observation log (never reset on adoption);
* ``exploration.ExplorationLedger`` first-scan exploration pay and its sampled audit;
* ``settlement``                    the per-window money (proportional split) and its validation.

What a caller can rely on:

* Window envelope. ``open_window`` freezes, per window: the order hash, the schedule in force
  (and its hash), one pool per ACTIVE env exactly, the slot geometry and the checkpoint. A
  schedule applied later takes effect at the next ``open_window``; an env deactivated mid-window
  stays in the running envelope and its entitlements still settle. Every amount, cap and price
  handed to the ledger comes from that frozen envelope, the same one ``settle_window`` receives.
* Announcement. ``announcement`` is the v2 ``ServicePolicyAnnouncement`` built from the frozen
  envelope plus the window's beacon; the first beacon stored for a window wins, so the policy a
  batcher holds never changes between admission and a deferred proof.
* Observations. One transaction per observation. Exploration goes through
  ``exploration.record_exploration`` only (record + reservation + first-scan release together).
  Pay is keyed on (env, prompt), never on the group id. The audit draw is keyed with the run's
  secret salt (R18): no miner can tell which of its rows are drawn. The audit draw round is the drand round
  of the VALIDATOR-clock arrival + 2. A refusal is a ``ServicePolicyLimit``: nothing was written.
* Audits. ``queued_audits`` lists what to audit (hotkeys past probation first, then by draw
  round) straight from the ledger rows, so it survives a restart. ``record_audit`` applies a
  verdict, releases the forfeited first scans and publishes their settle events atomically.
* Seal. See ``finalize_exploration`` for the contract the batcher owes before settlement.
* Settlement. ``reconcile_archive`` finalizes what is left, prices the ledger's integer
  entitlement counts with ``settle_window``, self-checks the result with
  ``validate_service_archive_v2`` under the protocol slot geometry, and closes the window to
  new observations. Calling it again recomputes from the caller's batch and the same finalized
  entitlement counts (same input, same bytes); a verdict arriving after finalize changes nothing.
* Scans. A prompt with a proven training observation is scanned for the run (aborted windows
  excepted: they trained nothing); exploration on a
  prompt trained in the same window is unpaid (``trained``, no sanction). See ``run_log``.

Only ``service-contract/v2`` runs RL here; ``service-contract/v1`` (dataset mapping / curation)
is refused.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass


logger = logging.getLogger(__name__)

QUALIFICATION_SCHEMA = "service-runtime-qualification/v2"
_EPS = 1e-12
@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """What ``ServiceRuntime.record_audit`` did with one verdict.

    ``kind``: ``"passed"`` (the group is audited-and-passed), ``"failed"`` (a failure was applied:
    ``forfeited`` holds the ids the hotkey lost, their first scans released; banned or put in
    re-probation according to the failure class, R31) or
    ``"not_applied"`` (nothing changed: unknown id, row not awaiting an audit, a verdict that is
    moot after finalize or on a row already unaudited, or SQLite busy -- the caller may retry).
    """
    kind: str
    forfeited: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.kind == "passed"

    @property
    def failed(self) -> bool:
        return self.kind == "failed"

    @property
    def applied(self) -> bool:
        return self.kind != "not_applied"


_MAX_NUMBER = 1e15  # clocks, pools and rewards are far below; beyond it a number is not one of them


def _finite(value) -> bool:
    """True for a finite int/float within ``_MAX_NUMBER``. Never raises: ``math.isfinite`` and
    ``float()`` raise OverflowError on a huge int, which would escape an ``except ValueError``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return -_MAX_NUMBER <= value <= _MAX_NUMBER
    return isinstance(value, float) and math.isfinite(value) and abs(value) <= _MAX_NUMBER


class EpisodePrecommitRefused(ValueError):
    """An episode precommit the frozen window does not allow; nothing was written. ``reason`` is
    the wire reason; ``existing`` names the live precommit of the same (hotkey, env, task, window)."""

    def __init__(self, reason: str, existing: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.existing = existing

def protocol_slot_geometry() -> tuple[int, int]:
    """``(picks_target, batch_slots)`` of one env in one window, from the protocol constants."""
    from reliquary.constants import B_BATCH, FILL_CLOSED_PICKS_PER_WINDOW
    return int(FILL_CLOSED_PICKS_PER_WINDOW), int(B_BATCH)


