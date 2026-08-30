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
from unittest.mock import MagicMock, patch

from reliquary import constants
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


# ---------------------------------------------------------------- C2


def test_a_v6_window_writes_no_raw_key_payload_at_seal(monkeypatch):
    """C2: ``enqueue_payload(window_n, ...)`` uses the BARE window number.
    Under v6 the key space is ``window * EMISSIONS + batch``, so a raw-key
    write for window ``w`` lands in window ``w // EMISSIONS``'s slot -- a
    duplicate payload carrying the auction's selection instead of the
    assembler's. The assembler already wrote this window's payloads."""
    _arm_v6(monkeypatch)
    monkeypatch.setattr(constants, "WRITE_TRAINING_PAYLOADS", True)

    svc = _service()
    queue = _RecordingQueue()
    svc._training_payload_queue = queue

    svc._write_training_payload(
        {"openmathinstruct": []}, 30_000, "rev", {"quarantined": False},
    )

    assert queue.payloads == []
    assert queue.tombstones == []


def test_an_aborted_v6_window_tombstones_every_encoded_key(monkeypatch):
    """C2: the trainer never advances on absence and walks the encoded key
    space one integer at a time. An aborted v6 window owns EMISSIONS keys,
    so one raw-key tombstone leaves the whole range unwritten and parks the
    cursor there forever."""
    _arm_v6(monkeypatch)
    monkeypatch.setattr(constants, "WRITE_TRAINING_PAYLOADS", True)

    svc = _service()
    queue = _RecordingQueue()
    svc._training_payload_queue = queue

    svc._write_training_tombstone(30_000, "admission_drain", "Timeout")

    emissions = constants.FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert {key for key, _ in queue.tombstones} == {
        30_000 * emissions + index for index in range(emissions)
    }


def test_the_abort_tombstone_never_covers_a_slot_the_assembler_used(
    monkeypatch,
):
    """A window can abort after emitting real batches. Those slots hold
    payloads already; padding starts at the assembler's next index, exactly
    as ``close()`` pads (R18)."""
    _arm_v6(monkeypatch)
    monkeypatch.setattr(constants, "WRITE_TRAINING_PAYLOADS", True)

    svc = _service()
    queue = _RecordingQueue()
    svc._training_payload_queue = queue
    svc._fill_closed_assemblers[30_000] = MagicMock(next_batch_index=2)

    svc._write_training_tombstone(30_000, "admission_drain", "Timeout")

    emissions = constants.FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert {key for key, _ in queue.tombstones} == {
        30_000 * emissions + index for index in range(2, emissions)
    }


def test_an_aborted_v6_window_drops_its_assembler(monkeypatch):
    """A window that aborts never reaches ``_archive_window``, which is the
    only other place an assembler is popped -- without this the dict grows
    one closed assembler per consecutive abort."""
    _arm_v6(monkeypatch)

    svc = _service()
    svc._training_payload_queue = _RecordingQueue()
    svc._window_iteration_stage = "seal_train_archive"
    svc._fill_closed_assemblers[30_000] = MagicMock(next_batch_index=0)
    batcher = MagicMock()
    batcher.window_start = 30_000

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=MagicMock(),
    ):
        svc._enqueue_aborted_window(
            failure_stage="admission_drain",
            failure_type="Timeout",
            batchers={"openmathinstruct": batcher},
        )

    assert 30_000 not in svc._fill_closed_assemblers


def test_the_seal_path_still_writes_a_raw_key_when_the_gate_is_off(monkeypatch):
    """v4/v5 regression pin: one tombstone, one raw key, unchanged."""
    monkeypatch.setattr(constants, "WRITE_TRAINING_PAYLOADS", True)

    svc = _service()
    queue = _RecordingQueue()
    svc._training_payload_queue = queue

    svc._write_training_tombstone(30_000, "admission_drain", "Timeout")

    assert [key for key, _ in queue.tombstones] == [30_000]


def test_the_seal_path_still_writes_a_payload_when_the_gate_is_off(monkeypatch):
    monkeypatch.setattr(constants, "WRITE_TRAINING_PAYLOADS", True)

    svc = _service()
    queue = _RecordingQueue()
    svc._training_payload_queue = queue

    svc._write_training_payload(
        {"openmathinstruct": []}, 30_000, "rev", {"quarantined": False},
    )

    assert [key for key, _ in queue.payloads] == [30_000]


def _arm_v6(monkeypatch):
    """Arm the gate everywhere the write path reads it: the service decides
    WHETHER to write, and ``encoded_window_journal_key`` decides WHERE."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.service as service_module

    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)


class _RecordingQueue:
    def __init__(self):
        self.payloads = []
        self.tombstones = []

    def enqueue_payload(self, key, data):
        self.payloads.append((key, data))

    def enqueue_tombstone(self, key, data):
        self.tombstones.append((key, data))


def _service():
    from reliquary.validator.service import ValidationService
    from tests.unit.test_archive_window_content import _FakeEnv, _FakeWallet

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    return ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=tokenizer,
        env=_FakeEnv(), netuid=99,
    )
