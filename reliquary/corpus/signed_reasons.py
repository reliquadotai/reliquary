"""Names shared by the signed-sandbox intake, admission, the session book and the miner.
Pure (hashlib only): `corpus.admission` imports it, and replay validators never
install reliquary-sandbox."""

from __future__ import annotations

import hashlib

REASON_SANDBOX_TRANSCRIPT = "sandbox_transcript_invalid"
REASON_SANDBOX_EXPIRED = "sandbox_session_expired"
REASON_SANDBOX_SESSION_REUSED = "sandbox_session_reused"
REASON_SANDBOX_CALL_MISMATCH = "sandbox_call_mismatch"
REASON_SANDBOX_STATE_MISMATCH = "sandbox_state_mismatch"
SANDBOX_REASONS = (REASON_SANDBOX_TRANSCRIPT, REASON_SANDBOX_EXPIRED,
                   REASON_SANDBOX_SESSION_REUSED, REASON_SANDBOX_CALL_MISMATCH,
                   REASON_SANDBOX_STATE_MISMATCH)

_SESSION_DOMAIN = b"reliquary/sandbox-session/v1\x00"


def session_seen_key(session_id: str) -> str:
    """The session's entry in the job ledger's seen set: written in the same
    compare-and-swap that consumes the slot, so a session pays once. Domain-separated,
    so it never equals a completion digest."""
    return hashlib.sha256(_SESSION_DOMAIN + session_id.encode("utf-8")).hexdigest()


def corpus_engagement(job_id: str, prompt_index: int) -> str:
    """A corpus session token's `engagement` claim."""
    return f"corpus:{job_id}:{int(prompt_index)}"


def state_matches(state_sha256: str | None, final_diff: str) -> bool:
    """§5.D: the submitted diff is the state the sandbox graded. A graded final with no
    state (graded 0.0 during extract) matches only an empty diff (ruling 8)."""
    if state_sha256 is None:
        return final_diff == ""
    return hashlib.sha256(final_diff.encode("utf-8")).hexdigest() == state_sha256
