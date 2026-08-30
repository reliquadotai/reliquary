"""The seam between the v6 fill-closed path and the existing seal path.

Under v6 a window's batches are proven on arrival, assembled by the
service-level ``FillClosedBatchAssembler``, emitted to the trainer journal
under the encoded key space, and PAID from the assembler's token split.
The seal path -- which is the auction -- therefore has nothing left to do:
proving again re-charges proof-failure debt and submits a second plan, and
writing again lands a duplicate payload under a RAW journal key that
collides with window ``window_n // 16``'s encoded slot.

These tests pin the seam shut in both directions: v6 seals inert, and
v4/v5 keep every byte of their behaviour.
"""
import hashlib
from unittest.mock import MagicMock

from reliquary.validator.batcher import PendingSubmission

from tests.unit.test_grpo_window_batcher import _make_batcher


def _pending(prompt_idx: int, hotkey: str = "hk") -> PendingSubmission:
    root = str(prompt_idx).encode().ljust(32, b"\x00")
    return PendingSubmission(
        hotkey=hotkey,
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


class _RecordingScheduler:
    """Only records; a real seal would call ``submit``."""

    def __init__(self):
        self.submitted = []

    def submit(self, plan):
        self.submitted.append(plan)
        raise AssertionError("v6 seal must not submit a proof plan")


# ---------------------------------------------------------------- C1


def test_a_v6_seal_proves_nothing_and_selects_nothing(monkeypatch):
    """C1: ``_prove_ranked`` over ``_pending`` would re-prove every group
    the arrival path already proved -- a second plan under
    ``{w}:{env}:auction-winners``, a second charge of proof-failure debt,
    and a selection that contradicts what was already paid."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    scheduler = _RecordingScheduler()
    batcher = _make_batcher(proof_scheduler=scheduler)
    batcher.difficulty_auction_enabled = True
    batcher._pending = [_pending(i, hotkey=f"hk{i}") for i in range(4)]

    batch, rewards = batcher.seal_batch(pool=0.5)

    assert batch == []
    assert rewards == {}
    assert scheduler.submitted == []
    assert batcher.proof_attempts == 0
    assert dict(batcher._expensive_proof_failures_by_operator) == {}


def test_a_v6_seal_still_records_the_cooldown_of_what_it_paid(monkeypatch):
    """The seal path is the only writer of prompt/content cooldown and the
    rollout-hash dedup set. v6 selects nothing NEW, but the groups it
    already proved and paid must still cool their prompts down, or the
    next window re-serves them."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.difficulty_auction_enabled = True
    paid = MagicMock()
    paid.prompt_idx = 17
    paid.prompt_content_sha256 = "c" * 64
    paid.rollout_hashes = [b"\x01" * 32]
    batcher._proven_groups = {"openmathinstruct": [paid]}

    batcher.seal_batch(pool=0.5)

    assert 17 in batcher._cooldown.current_cooldown_set(
        batcher.window_start + 1
    )


def test_the_auction_seal_is_untouched_when_the_gate_is_off():
    """v4/v5 regression pin: with the gate off the seal path still proves
    and still selects."""
    batcher = _make_batcher()
    batcher.difficulty_auction_enabled = True
    batcher._pending = [_pending(i, hotkey=f"hk{i}") for i in range(2)]

    calls = []
    batcher._prove_ranked = lambda pool: calls.append(pool)
    batcher._prove_forensic_sample = lambda: None

    batcher.seal_batch(pool=0.5)

    assert calls == [0.5]
