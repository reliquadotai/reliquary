"""Proven-dominance early close of a full auction window.

The fixed collection deadline exists so an early seal can never cut off a
slow-but-hard submission that could still land (see the docstring of
``test_collection_deadline.py``). Dominance close respects that reasoning by
inverting it: it only fires when NO submission could still land — productive
capacity is fully charged by terminal work, nothing in flight can refund a
slot, and every signed upload receipt has been honoured to the letter.

Measured in production 2026-08-26: both environments fill at +19s to +45s and
the window then spends a median 79 s of its 102 s cycle rejecting everything
with ``batch_filled``. Dominance is reached there; the close waits out the
last receipt graces (≤33 s) on top, landing around +55-80 s.

Once dominance holds it is permanent for the window: a late reveal is rejected
``proof_grading_attempts_full`` before it can touch capacity (the register
path checks capacity first), and refunds only ever come from in-flight work,
which dominance requires to be empty.
"""

from __future__ import annotations

import pytest

from tests.unit.test_grpo_window_batcher import _make_batcher


def _dominant(batcher) -> None:
    """Put the batcher in the proven-dominance state via its own accounting.

    64 terminal charges, nothing pending, nothing in flight. Tests reach the
    counters directly (existing style in test_collection_deadline.py) because
    driving 64 full precommit/reveal/grade cycles would test the pipeline,
    not the close condition.
    """
    from reliquary.constants import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    batcher._proof_grading_charged = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    assert not batcher._pending_proof_reservations
    assert not batcher._inflight_proof_reservations


def _clock_batcher(mode: str, monkeypatch, **overrides):
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(batcher_module, "AUCTION_EARLY_CLOSE_MODE", mode)
    now = [1000.0]
    wall = [10_000.0]
    b = _make_batcher(
        time_fn=lambda: now[0],
        wall_clock_fn=lambda: wall[0],
        **overrides,
    )
    b.mark_window_opened()

    def advance(seconds: float) -> None:
        now[0] += seconds
        wall[0] += seconds

    return b, advance


def test_dominance_requires_full_terminal_capacity(monkeypatch):
    """64 charges alone are not dominance — a refund could reopen a slot.

    Refunds only come from in-flight work (``_grading_refundable`` is applied
    when an inflight reservation is released), so dominance additionally
    requires both reservation maps empty.
    """
    from reliquary.constants import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(25.0)

    b._proof_grading_charged = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW - 1
    assert b.poll_deadline() is False
    assert b.early_close_eligible_at is None

    b._proof_grading_charged = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    b._pending_proof_reservations["r1"] = object()
    assert b.poll_deadline() is False
    assert b.early_close_eligible_at is None
    del b._pending_proof_reservations["r1"]

    b._inflight_proof_reservations["r2"] = object()
    assert b.poll_deadline() is False
    assert b.early_close_eligible_at is None
    del b._inflight_proof_reservations["r2"]

    assert b.poll_deadline() is True
    assert b.is_sealed() is True
    assert b.early_close_sealed is True


def test_shadow_records_eligibility_without_sealing(monkeypatch):
    """Shadow is the deployment default: measure, change nothing.

    ``early_close_eligible_at`` pins the first moment dominance held, so the
    shadow data answers "how much would we have saved" before enforce is ever
    switched on.
    """
    b, advance = _clock_batcher("shadow", monkeypatch)
    advance(30.0)
    _dominant(b)

    assert b.poll_deadline() is False
    assert b.is_sealed() is False
    eligible_at = b.early_close_eligible_at
    assert eligible_at == pytest.approx(1030.0)

    advance(5.0)
    assert b.poll_deadline() is False
    assert b.early_close_eligible_at == eligible_at  # pinned once
    assert b.early_close_sealed is False

    from reliquary.constants import WINDOW_COLLECTION_SECONDS

    advance(WINDOW_COLLECTION_SECONDS)
    assert b.poll_deadline() is True  # the deadline still seals normally
    assert b.early_close_sealed is False


def test_off_mode_never_early_seals(monkeypatch):
    from reliquary.constants import WINDOW_COLLECTION_SECONDS

    b, advance = _clock_batcher("off", monkeypatch)
    advance(30.0)
    _dominant(b)

    assert b.poll_deadline() is False
    assert b.early_close_eligible_at is None
    advance(WINDOW_COLLECTION_SECONDS)
    assert b.poll_deadline() is True


def test_enforce_honours_receipt_graces_before_sealing(monkeypatch):
    """A signed receipt is a bounded right to upload; the close waits it out.

    The reveal was doomed the moment dominance held (capacity is terminal),
    but the promise is honoured to the LETTER: the window only seals once
    every accepted receipt has resolved — revealed, finished, or expired at
    its own grace deadline. This is what bounds the close at fill+grace and
    keeps the generation contract untouched.
    """
    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(25.0)

    accepted, reason, _deadline = b.try_register_upload_precommit(
        "receipt-slow", "miner-slow",
        t_arrival_wall=b.window_opened_wall_ts + 25.0,
        payload_bytes=100,
    )
    assert accepted is True and reason is None

    _dominant(b)
    assert b.poll_deadline() is False        # the receipt is still armed
    assert b.is_sealed() is False
    assert b.early_close_eligible_at is not None

    from reliquary.constants import SUBMISSION_UPLOAD_GRACE_SECONDS

    advance(SUBMISSION_UPLOAD_GRACE_SECONDS + 1.0)
    assert b.poll_deadline() is True         # grace over, nothing left armed
    assert b.is_sealed() is True
    assert b.early_close_sealed is True


def test_refusal_is_derived_at_register_time_not_polled(monkeypatch):
    """Dominance flips the instant a worker releases the last reservation.

    poll_deadline runs every 0.5 s (service.py:2281). A precommit landing in
    that gap must not be accepted: its 33 s grace would then hold the seal for
    a third of the window, and its reveal is doomed anyway
    (proof_grading_attempts_full). Register time must ask the same question the
    poll asks, not read a cached answer.
    """
    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(25.0)
    _dominant(b)
    # No poll_deadline() call: the flag, if any, was never set.
    accepted, reason, _ = b.try_register_upload_precommit(
        "receipt-gap", "miner-gap",
        t_arrival_wall=b.window_opened_wall_ts + 25.0,
        payload_bytes=100,
    )
    assert accepted is False
    assert reason == "collection_sealed"


def test_enforce_seals_despite_an_unprunable_receipt(monkeypatch):
    """A receipt that finished transport but never terminalised is unprunable.

    ``_prune_upload_precommits_locked`` only expires reservations whose body
    never arrived (``body_completed_at_wall is None and not revealed``). A raw
    upload that reaches transport-complete and then dies — client disconnect,
    handler cancellation — can never be pruned. Without a time bound the
    enforce path would refuse every new precommit from dominance onward and
    still never seal: strictly worse than off. The wait is bounded at
    dominance + one grace, which is what the docstring promises.
    """
    from reliquary.constants import SUBMISSION_UPLOAD_GRACE_SECONDS

    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(20.0)
    accepted, _reason, _ = b.try_register_upload_precommit(
        "receipt-wedged", "miner-wedged",
        t_arrival_wall=b.window_opened_wall_ts + 20.0,
        payload_bytes=100,
    )
    assert accepted is True
    # Transport completed, terminalisation never happened: prune cannot touch it.
    reservation = b._upload_precommits["receipt-wedged"]
    reservation.body_completed_at_wall = b.window_opened_wall_ts + 21.0
    reservation.revealed = True

    _dominant(b)
    assert b.poll_deadline() is False          # still inside the grace
    advance(SUBMISSION_UPLOAD_GRACE_SECONDS + 1.0)
    assert b.poll_deadline() is True           # bounded: seals anyway
    assert b.early_close_sealed is True


def test_enforce_refuses_new_receipts_once_dominant(monkeypatch):
    """Dominance closes the door to NEW receipts so the close converges.

    Without this, a steady precommit stream keeps the receipt map non-empty
    forever and the close never fires. Refusing is also strictly better for
    the miner: today they are accepted, upload their body, and only learn at
    reveal that capacity was full; ``batch_filled`` is the existing wire
    reason for exactly this situation.
    """
    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(25.0)
    _dominant(b)

    assert b.poll_deadline() is True  # no receipts pending: seals immediately
    assert b.is_sealed() is True


def test_enforce_refusal_reason_is_terminal_for_the_miner(monkeypatch):
    """The refusal must not be one the reference miner retries.

    ``submitter.py:395`` retries BATCH_FILLED through _RETRY_DELAYS
    (1+2+4 s, four attempts) because upload capacity is normally a live pool
    that can free a slot. Under dominance the refusal is PERMANENT for the
    window, so BATCH_FILLED would quadruple refused-precommit traffic for the
    whole refusal period and never deliver the early heads-up this feature
    promises. ``collection_sealed`` is the existing internal reason for a
    closed window and the server already maps it to PRECOMMIT_EXPIRED, which
    the miner treats as terminal.
    """
    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(25.0)

    accepted, reason, _ = b.try_register_upload_precommit(
        "receipt-a", "miner-a",
        t_arrival_wall=b.window_opened_wall_ts + 25.0,
        payload_bytes=100,
    )
    assert accepted is True

    _dominant(b)
    assert b.poll_deadline() is False        # armed receipt holds the seal

    accepted, reason, _ = b.try_register_upload_precommit(
        "receipt-b", "miner-b",
        t_arrival_wall=b.window_opened_wall_ts + 26.0,
        payload_bytes=100,
    )
    assert accepted is False
    assert reason == "collection_sealed"


def test_shadow_does_not_refuse_receipts(monkeypatch):
    """Shadow must be observationally identical to off — that is the point."""
    b, advance = _clock_batcher("shadow", monkeypatch)
    advance(25.0)
    _dominant(b)
    assert b.poll_deadline() is False

    accepted, reason, _ = b.try_register_upload_precommit(
        "receipt-a", "miner-a",
        t_arrival_wall=b.window_opened_wall_ts + 26.0,
        payload_bytes=100,
    )
    assert accepted is True and reason is None


def test_deadline_seal_is_untouched_without_dominance(monkeypatch):
    """Enforce mode without dominance is exactly today's validator."""
    from reliquary.constants import WINDOW_COLLECTION_SECONDS

    b, advance = _clock_batcher("enforce", monkeypatch)
    advance(30.0)
    assert b.poll_deadline() is False
    advance(WINDOW_COLLECTION_SECONDS)
    assert b.poll_deadline() is True
    assert b.early_close_sealed is False


def test_conservation_snapshot_reports_early_close(monkeypatch):
    b, advance = _clock_batcher("shadow", monkeypatch)
    advance(25.0)
    _dominant(b)
    b.poll_deadline()

    stats = b.upload_precommit_conservation()
    early = stats["early_close"]
    assert early["mode"] == "shadow"
    assert early["eligible_offset_seconds"] == pytest.approx(25.0)
    assert early["sealed_early"] is False


def test_empty_mode_falls_back_to_the_safe_default(monkeypatch):
    """``RELIQUARY_AUCTION_EARLY_CLOSE_MODE=`` is a bare line in an env file.

    os.environ.get's default only applies when the key is ABSENT. Raising on
    "" would crash-loop the validator at import on a knob whose whole point is
    to be safe by default — and .env files ship bare-empty lines. The file's
    own convention (DIFFICULTY_AUCTION_ENFORCE, ENFORCE_ENVELOPE_SIGNATURE)
    accepts "" explicitly.
    """
    import importlib

    monkeypatch.setenv("RELIQUARY_AUCTION_EARLY_CLOSE_MODE", "")
    import reliquary.constants as constants

    reloaded = importlib.reload(constants)
    try:
        assert reloaded.AUCTION_EARLY_CLOSE_MODE == "shadow"
    finally:
        monkeypatch.delenv("RELIQUARY_AUCTION_EARLY_CLOSE_MODE", raising=False)
        importlib.reload(constants)


def test_a_wrong_mode_still_fails_loudly(monkeypatch):
    import importlib

    monkeypatch.setenv("RELIQUARY_AUCTION_EARLY_CLOSE_MODE", "sometimes")
    import reliquary.constants as constants

    try:
        with pytest.raises(ValueError, match="EARLY_CLOSE_MODE"):
            importlib.reload(constants)
    finally:
        monkeypatch.delenv("RELIQUARY_AUCTION_EARLY_CLOSE_MODE", raising=False)
        importlib.reload(constants)
