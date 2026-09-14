"""Miner-facing lifecycle descriptions and compact durable candidate outcomes."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

DETAIL_FIELDS = frozenset({
    "stage", "is_final", "selection_status", "outcome_code", "explanation",
    "reason_details", "environment", "prompt_idx", "checkpoint_revision", "receipt_id",
    "ordering_policy", "rank_scope", "proof_status", "proof_reason", "body_received_ts",
    "proof_recorded_ts", "proof_duration_seconds", "finalized_ts", "batch_index",
    "selection_target", "selected_count",
})


def public_verdict(entry, *, detailed=False):
    # Existing miner SDKs forbid unknown fields. Extended fields are opt-in.
    return {k: v for k, v in entry.items()
            if k != "_sequence" and (detailed or k not in DETAIL_FIELDS)}


def lifecycle_fields(*, accepted, selected, selection_reason, reason, now):
    final = selected is not None or not accepted
    code = selection_reason or (reason if not accepted else "admitted_pending_selection")
    if selected is True:
        code = selection_reason or "selected"
    elif selected is False and not selection_reason and accepted:
        code = "not_selected_status_unavailable"
    explanations = {
        "admitted_pending_selection": "Admitted; selection has not been finalized.",
        "selected_fifo": "Selected into a durable training batch; this is not confirmation of training consumption or an on-chain payout.",
        "selected": "Selected for the training batch.",
        "proof_not_needed_target_reached": "The proof plan reached its required number of passing groups; this candidate was not needed.",
        "same_prompt_or_content_already_proven": "An earlier-ranked candidate already proved this prompt or content.",
        "proof_failure_debt": "A hotkey or operator proof-failure budget was exhausted for this environment and proof plan.",
        "proof_rejected": "The deferred proof failed; proof_reason contains the verifier result when available.",
        "proven_not_selected_before_window_close": "Proof passed, but this group did not receive a training seat before window closure.",
        "proof_not_completed_before_window_close": "The window closed without a completed proof for this candidate.",
        "picked_but_unpaid_incomplete_cross_environment_batch": "Picked locally, but the cross-environment batch was not durably completed.",
        "not_selected_status_unavailable": "Not selected; the validator did not retain the detailed cause.",
    }
    return {
        "stage": "final" if final else "admission",
        "is_final": final,
        "selection_status": "selected" if selected is True else "not_selected" if final else "pending",
        "outcome_code": code,
        "explanation": explanations.get(code, f"Validator outcome: {code}."),
        **({"finalized_ts": now} if final else {}),
    }


class FinalVerdictStore:
    """Store only bounded, admitted-candidate outcomes, never raw submissions.

    One transaction per published batch/window. The existing state volume
    provides restart persistence; this is not an independent off-host backup.
    """
    RETENTION_WINDOWS = 2048

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=1)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS verdicts (window INTEGER, hotkey TEXT, root TEXT, payload TEXT, PRIMARY KEY(window,hotkey,root))")
        self._db.execute("CREATE TABLE IF NOT EXISTS coverage (id INTEGER PRIMARY KEY CHECK(id=1), first_window INTEGER, last_window INTEGER)")
        self._db.commit()

    def put_many(self, records):
        rows = [(int(v["window_n"]), hk, v["merkle_root"].lower(), json.dumps(v, separators=(",", ":")))
                for hk, v in records if v.get("is_final") and v.get("accepted_into_pool") and v.get("window_n") is not None]
        if not rows:
            return
        with self._lock, self._db:
            self._db.executemany("INSERT INTO verdicts VALUES (?,?,?,?) ON CONFLICT(window,hotkey,root) DO UPDATE SET payload=excluded.payload WHERE json_extract(verdicts.payload,'$.selected_for_batch') IS NOT 1 OR json_extract(excluded.payload,'$.selected_for_batch')=1", rows)
            self._db.execute("INSERT INTO coverage VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET first_window=min(first_window,excluded.first_window), last_window=max(last_window,excluded.last_window)", (min(r[0] for r in rows), max(r[0] for r in rows)))
            self._db.execute("DELETE FROM verdicts WHERE window < (SELECT last_window FROM coverage WHERE id=1)-?+1", (self.RETENTION_WINDOWS,))

    def lookup(self, hotkey, window, root):
        with self._lock:
            row = self._db.execute("SELECT payload FROM verdicts WHERE window=? AND hotkey=? AND root=?", (window, hotkey, root.lower())).fetchone()
            coverage = self._db.execute("SELECT first_window,last_window FROM coverage WHERE id=1").fetchone()
        oldest = max(coverage[0], coverage[1] - self.RETENTION_WINDOWS + 1) if coverage else None
        status = "found" if row else "expired" if coverage and coverage[0] <= window < oldest else "not_recorded"
        return {"status": status, "verdict": json.loads(row[0]) if row else None,
                "oldest_retained_window": oldest, "retention_windows": self.RETENTION_WINDOWS}
