"""Bad-envelope connection-priming bypass — defence-in-depth tests.

PR #35 closed the trivial spoof-DoS on ``/submit`` by verifying an
sr25519 envelope signature BEFORE bumping ``_per_window_counts``.
That made BAD_ENVELOPE_SIGNATURE rejects deliberately quota-free,
which is correct for the threat model PR #35 targeted (an anonymous
attacker spoofing a victim's hotkey cannot drain the victim's
8-submission quota).

A follow-on side-channel was observed in the wild: the LEGITIMATE
hotkey owner can fire a burst of BAD_ENVELOPE_SIGNATURE packets at
window OPEN under their own hotkey to warm HTTP/1.1 keep-alive
sockets at zero quota cost, then ride those warm sockets for the
real signed POSTs that follow. The pattern logged on the network
is 24 bad-envelope packets followed by 8 properly-signed ones,
giving the exploiter a ~20-30 ms RTT edge on the seal-trigger
race against honest single-instance miners.

These tests pin two defences that close the bypass without
re-introducing the spoof-DoS that PR #35 fixed:

  1. ``Connection: close`` is emitted on every BAD_ENVELOPE_SIGNATURE
     response, so an attacker cannot use a bad packet to warm a
     keep-alive socket for the next packet. Honest accepted
     submissions do NOT carry the header.
  2. Per-hotkey BAD_ENVELOPE_SIGNATURE budget
     (``MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW``) — past the cap the
     verdict is still BAD_ENVELOPE_SIGNATURE but is silently dropped
     from the per-hotkey verdict ring so an anonymous spoofer
     cannot flood a victim's ``/verdicts/{hotkey}`` history.

PR #35's invariant is verified end-to-end: even when an anonymous
spoofer sends well above the cap, ``_per_window_counts[victim]``
stays at zero and the victim's first legitimate signed submission
is accepted.
"""

from __future__ import annotations

import bittensor as bt
from fastapi.testclient import TestClient

from reliquary.constants import (
    MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    M_ROLLOUTS,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    RejectReason,
    WindowState,
)
from reliquary.validator.server import ValidatorServer

# Reuse the helpers already exercised by test_envelope_signature.py.
from tests.unit.test_envelope_signature import (
    _batcher,
    _make_commit,
    _new_keypair,
    _signed_request,
)


def _unsigned_bad_request(victim_hotkey: str) -> BatchSubmissionRequest:
    """Build a request that will fail the envelope-signature gate.

    Mirrors the on-wire shape PR #35 sees from an anonymous spoofer:
    valid pydantic schema, claims ``victim_hotkey`` as ``miner_hotkey``,
    but ``envelope_signature`` is empty so verification rejects it
    before any quota counter is touched.
    """
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=(i < 4), total_reward=float(i < 4))
        rollouts.append(RolloutSubmission(
            tokens=commit["tokens"],
            reward=float(i < 4),
            commit=commit,
            env_name="openmathinstruct",
        ))
    return BatchSubmissionRequest(
        miner_hotkey=victim_hotkey,
        prompt_idx=42,
        window_start=500,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
        drand_round=0,
        nonce="attack",
        envelope_signature="",
    )


# ---------------------------------------------------------------------------
# Fix 1 — Connection: close header on BAD_ENVELOPE_SIGNATURE responses
# ---------------------------------------------------------------------------


def test_bad_envelope_response_carries_connection_close(enforce_envelope):
    """A BAD_ENVELOPE_SIGNATURE response MUST instruct the client to
    tear the socket down. Without this header, the attacker can fire
    the next bad-envelope packet over the same warm TCP+TLS connection
    and use the zero-quota channel as a free keep-alive priming
    mechanism for the real signed POSTs that follow."""
    kp_victim = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    resp = client.post(
        "/submit",
        json=_unsigned_bad_request(kp_victim.ss58_address).model_dump(mode="json"),
    )
    assert resp.json()["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value
    # Header lookup is case-insensitive on httpx/requests Response.
    assert resp.headers.get("connection", "").lower() == "close", (
        "BAD_ENVELOPE_SIGNATURE response is missing Connection: close — "
        "attacker can warm a keep-alive pool with bad packets at zero "
        "quota cost and gain an RTT edge on signed POSTs that follow."
    )


def test_accepted_submission_does_not_force_connection_close(enforce_envelope):
    """The defence MUST be scoped to bad-envelope rejects only.
    Legitimate signed submissions must not pay a forced handshake
    on every call — that would crater throughput for honest miners
    who submit multiple prompts per window over a keep-alive pool.
    """
    kp = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    req = _signed_request(kp, window_start=500)
    resp = client.post("/submit", json=req.model_dump(mode="json"))
    assert resp.json()["accepted"] is True, resp.json()
    # Either the header is absent entirely, or it isn't ``close`` — the
    # server may legitimately omit it (HTTP/1.1 default = keep-alive).
    assert resp.headers.get("connection", "").lower() != "close", (
        "Connection: close leaked onto an ACCEPTED response — would "
        "force honest miners off keep-alive and tank their throughput."
    )


# ---------------------------------------------------------------------------
# Fix 2 — per-hotkey BAD_ENVELOPE budget caps verdict-ring noise
# ---------------------------------------------------------------------------


def test_bad_envelope_verdict_ring_capped_per_hotkey(enforce_envelope):
    """Past the cap, BAD_ENVELOPE_SIGNATURE responses are still emitted
    but no longer write to the per-hotkey verdict ring. Without this
    cap, a spoofer firing N bad packets at the victim's hotkey would
    flood ``_verdicts[victim]`` with N junk entries (capped only by
    the 200-entry ring buffer), displacing any legitimate verdicts the
    victim might want to see — a noticeable observability DoS even
    though the rate-limit quota is preserved.
    """
    kp_victim = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    burst = MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW + 10
    payload = _unsigned_bad_request(kp_victim.ss58_address).model_dump(mode="json")
    for _ in range(burst):
        resp = client.post("/submit", json=payload)
        # Response shape is unchanged past the cap — still a bad-envelope
        # reject. Only the side-effect (verdict-ring write) is gated.
        assert resp.json()["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value
        assert resp.headers.get("connection", "").lower() == "close"

    recorded = list(server._verdicts.get(kp_victim.ss58_address, []))
    assert len(recorded) == MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW, (
        f"Verdict ring recorded {len(recorded)} bad-envelope entries "
        f"for one hotkey in one window; cap is "
        f"{MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW}. Without the cap, "
        f"spoofers can flood /verdicts/{{victim}} with junk."
    )


def test_bad_envelope_counter_resets_on_batcher_swap(enforce_envelope):
    """The cap is per-window, not per-server-lifetime. When the active
    batcher swaps (= window boundary) the counter must reset so a
    miner that hit the cap last window isn't permanently muted from
    /verdicts in subsequent windows."""
    kp_victim = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    payload = _unsigned_bad_request(kp_victim.ss58_address).model_dump(mode="json")
    # Burn through the cap in window 500.
    for _ in range(MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW + 5):
        client.post("/submit", json=payload)

    assert (
        server._bad_envelope_counts.get(kp_victim.ss58_address, 0)
        == MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW
    )

    # Window boundary — operator swaps in a new batcher.
    server.set_active_batcher(_batcher(window_start=501))
    assert server._bad_envelope_counts == {}, (
        "_bad_envelope_counts must reset on batcher swap or last "
        "window's bad-envelope budget leaks into the new window."
    )


# ---------------------------------------------------------------------------
# Regression: PR #35's anti-spoof-DoS invariant survives the defences
# ---------------------------------------------------------------------------


def test_priming_cap_does_not_re_enable_spoof_dos(enforce_envelope):
    """End-to-end invariant: even with the new per-hotkey
    bad-envelope cap and forced socket teardown in place, an
    anonymous spoofer firing far more bad packets than the legitimate
    quota MUST NOT consume the victim's
    ``MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW`` slot — and the
    victim's first legitimate signed submission must still be
    accepted. This is the PR #35 invariant; the new defences must not
    leak the quota counter under any path.
    """
    kp_victim = _new_keypair()
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # Spoofer fires several times the legitimate quota in bad packets.
    bad_payload = _unsigned_bad_request(kp_victim.ss58_address).model_dump(mode="json")
    burst = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW * 3
    for _ in range(burst):
        resp = client.post("/submit", json=bad_payload)
        assert resp.json()["reason"] == RejectReason.BAD_ENVELOPE_SIGNATURE.value

    # Victim's quota counter is untouched.
    assert server._per_window_counts.get(kp_victim.ss58_address, 0) == 0

    # And the victim's first signed submission still goes through.
    good = _signed_request(kp_victim, window_start=500, prompt_idx=42)
    resp = client.post("/submit", json=good.model_dump(mode="json"))
    body = resp.json()
    assert body["accepted"] is True, body
    assert body["reason"] == RejectReason.ACCEPTED.value
    # Honest path must not carry Connection: close.
    assert resp.headers.get("connection", "").lower() != "close"
