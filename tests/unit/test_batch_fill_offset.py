"""How long a window takes to become fillable, measured in shadow.

The fill-closed window design closes on a COUNT (every environment holds its
target of usable groups) instead of on a clock. Sizing it needs one number the
validator does not record today: how far into a window the training batch
first becomes fillable.

PR #212 measured the adjacent quantity — productive CAPACITY charges at +19 s
to +45 s — but capacity is 64 receipts, not B_BATCH distinct prompts, and a
batch needs distinct prompts. This records the batch-shaped number.

Measured by polling rather than by hooking the admission path: the resolution
needed is seconds, and the admission locks have a documented convoy history
(see the lock-order note on ``_poll_early_close``).
"""

from __future__ import annotations

import hashlib

from reliquary.constants import B_BATCH
from reliquary.validator.batcher import PendingSubmission

from tests.unit.test_grpo_window_batcher import _make_batcher


def _pending(prompt_idx: int) -> PendingSubmission:
    root = str(prompt_idx).encode().ljust(32, b"\x00")
    return PendingSubmission(
        hotkey=f"hk{prompt_idx}",
        prompt_idx=prompt_idx,
        request=None,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        drand_round=1,
        merkle_root=root,
        selection_digest=root,
        prompt_content_sha256=hashlib.sha256(
            f"prompt:{prompt_idx}".encode()
        ).hexdigest(),
        target_content_sha256=hashlib.sha256(b"target").hexdigest(),
    )


def _clock_batcher():
    now = [1000.0]
    b = _make_batcher(time_fn=lambda: now[0], wall_clock_fn=lambda: now[0])
    b.mark_window_opened()

    def advance(seconds: float) -> None:
        now[0] += seconds

    return b, advance


def test_fill_offset_records_when_b_batch_distinct_prompts_are_pending():
    b, advance = _clock_batcher()
    for prompt_idx in range(B_BATCH):
        b._pending.append(_pending(prompt_idx))
    advance(31.0)

    b.poll_deadline()

    assert b.graded_batch_fill_offset_s == 31.0


def test_repeated_prompts_do_not_fill_the_batch():
    """A batch needs B_BATCH DISTINCT prompts.

    ``MAX_SUBMISSIONS_PER_PROMPT`` lets ten miners answer the same prompt, so
    counting submissions instead of prompts would report a batch fillable when
    it holds one prompt ten times over.
    """
    b, advance = _clock_batcher()
    for copy in range(B_BATCH * 2):
        b._pending.append(_pending(copy % 3))
    advance(31.0)

    b.poll_deadline()

    assert b.graded_batch_fill_offset_s is None


def test_fill_offset_latches_at_the_first_fillable_moment():
    """It answers "when did it BECOME fillable", so later polls must not move it.

    Submissions keep arriving after the batch is fillable — that is the whole
    finding of PR #212 — so a re-recording poll would report the end of the
    window rather than the fill.
    """
    b, advance = _clock_batcher()
    for prompt_idx in range(B_BATCH):
        b._pending.append(_pending(prompt_idx))
    advance(31.0)
    b.poll_deadline()

    for prompt_idx in range(B_BATCH, B_BATCH * 2):
        b._pending.append(_pending(prompt_idx))
    advance(40.0)
    b.poll_deadline()

    assert b.graded_batch_fill_offset_s == 31.0


def test_fill_offset_is_exposed_for_analysis():
    """A measurement nobody can read is dead code.

    ``upload_precommit_conservation`` is the per-environment window telemetry
    channel — it already carries ``early_close``, which is not about receipt
    conservation either — and it reaches both the R2 archive and ``/health``.
    """
    b, advance = _clock_batcher()
    for prompt_idx in range(B_BATCH):
        b._pending.append(_pending(prompt_idx))
    advance(31.0)
    b.poll_deadline()

    assert b.upload_precommit_conservation()["graded_batch_fill_offset_seconds"] == 31.0


def test_fill_offset_is_absent_until_the_batch_is_fillable():
    b, advance = _clock_batcher()
    advance(31.0)
    b.poll_deadline()

    assert b.upload_precommit_conservation()["graded_batch_fill_offset_seconds"] is None


def test_prefix_fill_is_recorded_separately_and_later():
    """B_BATCH graded prompts is a floor, not the answer.

    Nothing in ``_pending`` is proven — GRAIL runs at seal — and a group that
    fails its proof is not a group. The system already budgets for that:
    ``MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW`` is ``2 * B_BATCH``, "B_BATCH
    winners plus B_BATCH possible failed candidates".

    So the offset at which a PROVEN batch could have been filled is bracketed:
    at best the B_BATCH offset (every proof passes), at worst the ranked-prefix
    offset (half fail). Recording only the floor would size the design on its
    most optimistic case.
    """
    from reliquary.constants import MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW

    b, advance = _clock_batcher()
    for prompt_idx in range(B_BATCH):
        b._pending.append(_pending(prompt_idx))
    advance(31.0)
    b.poll_deadline()

    for prompt_idx in range(B_BATCH, MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW):
        b._pending.append(_pending(prompt_idx))
    advance(19.0)
    b.poll_deadline()

    assert b.graded_batch_fill_offset_s == 31.0
    assert b.graded_prefix_fill_offset_s == 50.0
