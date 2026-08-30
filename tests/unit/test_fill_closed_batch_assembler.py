"""R13: the service-level join of per-environment emission chunks into
one cross-environment DAPO training batch.

A GrpoWindowBatcher only ever hands the assembler its OWN environment's
next B_BATCH chunk (see batcher.py:_emit_training_batch); chunks from two
environments can arrive in any interleaving. The assembler must pair them
up correctly regardless of arrival order, write exactly one payload per
completed cycle, and never skip or duplicate a cycle's chunk.
"""
from reliquary.constants import B_BATCH
from reliquary.infrastructure.training_payload_queue import (
    encoded_window_journal_key,
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
    decode_tombstone,
    decode_training_payload,
)
from reliquary.validator.fill_closed_batch_assembler import (
    FillClosedBatchAssembler,
)
from tests.unit.test_training_payload_codec import _group, _roll

ENV_ORDER = ["openmathinstruct", "opencodeinstruct"]


def _chunk(tag: int, env: str) -> list:
    return [
        _group([_roll(1.0, 4, env=env)], prompt_idx=tag * 1000 + i)
        for i in range(B_BATCH)
    ]


def _prompt_ids(groups: list) -> list:
    return [g.prompt_idx for g in groups]


def _partial_chunk(tag: int, env: str, n: int) -> list:
    return [
        _group([_roll(1.0, 4, env=env)], prompt_idx=tag * 1000 + i)
        for i in range(n)
    ]


def test_interleaved_chunks_join_into_one_ordered_payload_per_cycle(
    monkeypatch,
):
    """(a)+(b): math k=0, math k=1, code k=0 -> exactly one payload
    written, index 0, containing math's FIRST chunk and code's, not
    math's second. Then code k=1 arrives -> payload index 1 is written,
    and the journal keys are encoded(window, 0) then encoded(window, 1)
    in write order.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    written: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: written.append((key, data)),
        tombstone_fn=lambda key, data: None,
    )

    math_k0 = _chunk(0, "openmathinstruct")
    math_k1 = _chunk(1, "openmathinstruct")
    code_k0 = _chunk(0, "opencodeinstruct")
    code_k1 = _chunk(1, "opencodeinstruct")

    assembler.accept("openmathinstruct", math_k0, window, "rev")
    assembler.accept("openmathinstruct", math_k1, window, "rev")
    assert written == []  # code hasn't contributed anything yet

    assembler.accept("opencodeinstruct", code_k0, window, "rev")

    assert len(written) == 1
    key0, data0 = written[0]
    assert key0 == encoded_window_journal_key(window, 0)
    batches0 = decode_training_payload(data0).batches()
    assert len(batches0["openmathinstruct"]) == B_BATCH
    assert len(batches0["opencodeinstruct"]) == B_BATCH
    # math's FIRST chunk, not the second, is what went out.
    assert _prompt_ids(batches0["openmathinstruct"]) == _prompt_ids(math_k0)
    assert _prompt_ids(batches0["opencodeinstruct"]) == _prompt_ids(code_k0)
    assert assembler.next_batch_index == 1

    assembler.accept("opencodeinstruct", code_k1, window, "rev")

    assert len(written) == 2
    key1, data1 = written[1]
    assert key1 == encoded_window_journal_key(window, 1)
    # Write order: index 0 before index 1.
    assert [k for k, _ in written] == [key0, key1]
    batches1 = decode_training_payload(data1).batches()
    assert _prompt_ids(batches1["openmathinstruct"]) == _prompt_ids(math_k1)
    assert _prompt_ids(batches1["opencodeinstruct"]) == _prompt_ids(code_k1)


def test_assembled_payload_round_trips_through_the_trainer_journal(
    monkeypatch,
):
    """(c): the encoded payload round-trips through the decoder
    train_runner uses (decode_training_payload -> .batches()) and yields
    B_BATCH groups per env. Extended per the coordinator's note: the
    trainer's own WindowJournal.next_entry, with a cursor of
    encoded(window, 0) - 1 and stride 1, must return the index-0 entry --
    not None, and not raise on the window/key mismatch check.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.trainer.journal as journal_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(journal_module, "FILL_CLOSED_ENABLED", True)

    from reliquary.trainer.journal import WindowJournal

    window = 42
    written: dict[str, bytes] = {}
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: written.__setitem__(
            payload_key(key), data
        ),
        tombstone_fn=lambda key, data: None,
    )
    for env in ENV_ORDER:
        assembler.accept(env, _chunk(0, env), window, "rev")

    key0 = encoded_window_journal_key(window, 0)
    assert payload_key(key0) in written

    journal = WindowJournal(fetch_fn=written.get)
    kind, decoded = journal.next_entry(key0 - 1, stride=1)

    assert kind == "payload"
    assert decoded.window_start == window
    batches = decoded.batches()
    assert len(batches["openmathinstruct"]) == B_BATCH
    assert len(batches["opencodeinstruct"]) == B_BATCH


def test_accept_rejects_a_chunk_for_a_different_window():
    assembler = FillClosedBatchAssembler(
        window_start=42,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
    )
    import pytest

    with pytest.raises(ValueError):
        assembler.accept(
            "openmathinstruct", _chunk(0, "openmathinstruct"), 43, "rev",
        )


def test_remainder_snapshot_reports_what_never_became_a_payload():
    """(5): at seal, a partial remainder (one env full, the other short)
    is observable but never emitted as a partial batch."""
    assembler = FillClosedBatchAssembler(
        window_start=42,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
    )
    assembler.accept(
        "openmathinstruct", _chunk(0, "openmathinstruct"), 42, "rev",
    )

    snapshot = assembler.remainder_snapshot()

    assert snapshot["in_accumulator"]["openmathinstruct"] == B_BATCH
    assert snapshot["in_accumulator"]["opencodeinstruct"] == 0
    assert snapshot["pending"] == {
        "openmathinstruct": 0, "opencodeinstruct": 0,
    }


def _chunk_with_suspicious_reward_shape(tag: int, env: str, n: int) -> list:
    """Same shape as ``_chunk`` but the first ``n`` groups carry a
    ``reward_shape`` that ``assess_training_batch`` treats as suspicious
    (short zero-length tail, so this trips ``reward_shape_density``
    rather than ``long_zero_tail_reward_shape``) -- the cheapest real
    quarantine trigger buildable from these fixtures, per
    TRAINING_QUARANTINE_REWARD_SHAPE_MIN_GROUPS=2.
    """
    groups = _chunk(tag, env)
    for group in groups[:n]:
        group.reward_shape = {
            "suspicious": True,
            "zero_length_mode": 120,
            "zero_length_mode_count": 4,
        }
    return groups


def test_quarantined_batch_is_tombstoned_not_enqueued(monkeypatch):
    """R14: assess_training_batch runs on every assembled batch, not just
    at seal. A batch it quarantines must never reach ``enqueue_fn`` --
    that is exactly the poisoned-data path the seal-time gate exists to
    close, and v6 bypassed it entirely. Instead a tombstone is written
    under the batch's OWN encoded journal key, so the trainer's cursor
    still advances (it never advances on absence, only on an explicit
    marker).
    """
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    # Two suspicious groups in math, two in code -> reward_shape_groups=4
    # in the flat batch, well past TRAINING_QUARANTINE_REWARD_SHAPE_MIN_
    # GROUPS=2. This is the REAL rule firing, not a mock.
    math_chunk = _chunk_with_suspicious_reward_shape(
        0, "openmathinstruct", 2
    )
    code_chunk = _chunk_with_suspicious_reward_shape(
        0, "opencodeinstruct", 2
    )
    assembler.accept("openmathinstruct", math_chunk, window, "rev")
    assembler.accept("opencodeinstruct", code_chunk, window, "rev")

    assert enqueued == []
    assert len(tombstoned) == 1
    key, data = tombstoned[0]
    assert key == encoded_window_journal_key(window, 0)
    assert tombstone_key(key) != tombstone_key(window)  # under the batch key
    decoded = decode_tombstone(data)
    assert decoded["window_start"] == window
    assert assembler.next_batch_index == 1


def test_clean_batch_enqueues_with_the_real_quarantine_decision(monkeypatch):
    """R14: a batch assess_training_batch does NOT quarantine still
    enqueues, and its ``window_quarantine`` carries the real decision's
    ``to_archive()`` -- not the previously-hardcoded
    ``{"quarantined": False, "reasons": []}``, which papered over every
    signal assess_training_batch computes (metrics included).
    """
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: None,
    )
    for env in ENV_ORDER:
        assembler.accept(env, _chunk(0, env), window, "rev")

    assert len(enqueued) == 1
    _, data = enqueued[0]
    decoded = decode_training_payload(data)
    assert decoded.window_quarantine["quarantined"] is False
    assert decoded.window_quarantine["reasons"] == []
    # The real decision carries metrics; the old hardcoded dict had none.
    assert "metrics" in decoded.window_quarantine
    assert decoded.window_quarantine["metrics"]["n_groups"] == 2 * B_BATCH


def test_batch_verdict_log_includes_the_batch_index(monkeypatch, caplog):
    """R14: the verdict is logged with the batch index -- v6 writes up
    to FILL_CLOSED_EMISSIONS_PER_WINDOW payloads per window, so a log
    line naming only the window cannot tell which batch was assessed."""
    import logging

    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
    )
    with caplog.at_level(
        logging.INFO,
        logger="reliquary.validator.fill_closed_batch_assembler",
    ):
        for env in ENV_ORDER:
            assembler.accept(env, _chunk(0, env), window, "rev")

    assert any(
        str(window) in record.getMessage()
        and "batch 0" in record.getMessage()
        and "quarantin" in record.getMessage().lower()
        for record in caplog.records
    )


def test_close_emits_the_remainder_as_a_payload_when_every_env_has_groups(
    monkeypatch,
):
    """R16: at window close, math is short (B_BATCH - 2 groups) but code
    already sits at a full B_BATCH held in the accumulator, blocked
    because math never reached B_BATCH to complete the cycle -- exactly
    the ``remainder_snapshot`` scenario that used to be read-only and
    reported one window later as a WARNING with no marker written. Every
    environment has at least one group, so ``close()`` emits it as one
    final, partial payload rather than dropping proven, paid rollouts.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    math_partial = _partial_chunk(0, "openmathinstruct", B_BATCH - 2)
    code_full = _chunk(0, "opencodeinstruct")
    assembler.accept("openmathinstruct", math_partial, window, "rev")
    assembler.accept("opencodeinstruct", code_full, window, "rev")

    # Neither chunk alone completes a cycle (math is short), so nothing
    # has been written yet -- this remainder is genuinely unaddressed
    # until close() runs.
    assert enqueued == []
    assert tombstoned == []

    assembler.close()

    assert len(enqueued) == 1
    key, data = enqueued[0]
    assert key == encoded_window_journal_key(window, 0)
    batches = decode_training_payload(data).batches()
    assert _prompt_ids(batches["openmathinstruct"]) == _prompt_ids(
        math_partial
    )
    assert _prompt_ids(batches["opencodeinstruct"]) == _prompt_ids(code_full)
    # R18: everything past the remainder's own slot is padded with
    # tombstones so the journal is gapless -- see the dedicated padding
    # tests below for the full key-accounting assertions.
    assert len(tombstoned) == FILL_CLOSED_EMISSIONS_PER_WINDOW - 1
    assert assembler.next_batch_index == FILL_CLOSED_EMISSIONS_PER_WINDOW


def test_close_tombstones_when_one_env_contributed_nothing(monkeypatch):
    """R16: opencodeinstruct never called accept() this cycle at all --
    zero groups, not just a short chunk. There is no batch to assemble
    (a DAPO step still needs every environment represented), so close()
    writes a tombstone under the next batch's key instead, so the
    trainer's journal cursor still advances rather than stalling.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    assembler.accept(
        "openmathinstruct", _partial_chunk(0, "openmathinstruct", 5),
        window, "rev",
    )

    assembler.close()

    assert enqueued == []
    # R18: the remainder tombstone plus padding for every slot after it.
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert len(tombstoned) == FILL_CLOSED_EMISSIONS_PER_WINDOW
    key, data = tombstoned[0]
    assert key == encoded_window_journal_key(window, 0)
    decoded = decode_tombstone(data)
    assert decoded["window_start"] == window
    assert assembler.next_batch_index == FILL_CLOSED_EMISSIONS_PER_WINDOW


def test_a_second_close_is_a_noop(monkeypatch):
    """R16: close() must be idempotent -- a second call (or one racing a
    final in-flight accept()) must not write a second payload/tombstone
    under a fresh batch index."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    for env in ENV_ORDER:
        assembler.accept(
            env, _partial_chunk(0, env, 3), window, "rev",
        )

    assembler.close()
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert (
        len(enqueued) + len(tombstoned) == FILL_CLOSED_EMISSIONS_PER_WINDOW
    )
    next_index_after_first_close = assembler.next_batch_index

    assembler.close()

    assert (
        len(enqueued) + len(tombstoned) == FILL_CLOSED_EMISSIONS_PER_WINDOW
    )
    assert assembler.next_batch_index == next_index_after_first_close


def test_enqueue_fn_is_invoked_without_the_lock_held(monkeypatch):
    """R17: accept() used to hold ``self._lock`` through ``_drain_locked``
    all the way into the call to ``_enqueue_fn`` -- a blocking filesystem
    write -- on whichever proof-worker thread happened to call
    ``accept``. The fake ``enqueue_fn`` below asserts the lock is free
    the instant it runs, which fails on the pre-fix code (the call was
    made from inside ``with self._lock:``).
    """
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    calls: list[int] = []

    def enqueue_fn(key: int, data: bytes) -> None:
        assert not assembler._lock.locked()
        calls.append(key)

    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=enqueue_fn,
        tombstone_fn=lambda key, data: None,
    )

    for env in ENV_ORDER:
        assembler.accept(env, _chunk(0, env), window, "rev")

    assert calls == [encoded_window_journal_key(window, 0)]


def test_close_pads_every_remaining_slot_when_remainder_has_every_env(
    monkeypatch,
):
    """R18: WindowJournal.next_entry walks the encoded key space one
    integer at a time with no jump logic, so any slot this window's
    range never writes parks the trainer's cursor there forever -- not
    just for this window's tail, but for every window after it too.
    One full cycle writes a payload at index 0; a remainder with every
    env represented gets its own payload at index 1 (R16); everything
    from there through the window's ceiling must be tombstoned so the
    journal is gapless, with no key skipped or written twice.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    for env in ENV_ORDER:
        assembler.accept(env, _chunk(0, env), window, "rev")
    assert len(enqueued) == 1
    assert assembler.next_batch_index == 1

    for env in ENV_ORDER:
        assembler.accept(
            env, _partial_chunk(1, env, B_BATCH - 2), window, "rev",
        )

    assembler.close()

    remainder_index = 1
    assert len(enqueued) == 2
    payload_keys = [key for key, _ in enqueued]
    assert payload_keys == [
        encoded_window_journal_key(window, 0),
        encoded_window_journal_key(window, remainder_index),
    ]
    expected_padding_keys = {
        encoded_window_journal_key(window, idx)
        for idx in range(
            remainder_index + 1, FILL_CLOSED_EMISSIONS_PER_WINDOW
        )
    }
    tombstone_keys = {key for key, _ in tombstoned}
    assert tombstone_keys == expected_padding_keys

    all_written = [key for key, _ in enqueued] + [
        key for key, _ in tombstoned
    ]
    assert len(all_written) == len(set(all_written))  # no duplicate key
    assert set(all_written) == {
        encoded_window_journal_key(window, idx)
        for idx in range(FILL_CLOSED_EMISSIONS_PER_WINDOW)
    }
    assert assembler.next_batch_index == FILL_CLOSED_EMISSIONS_PER_WINDOW


def test_close_pads_every_remaining_slot_when_remainder_is_empty(
    monkeypatch,
):
    """R18: same gaplessness requirement as above, but the remainder
    itself is the R16 "one env empty" tombstone case -- so index 1
    through the ceiling are ALL tombstones (the remainder marker plus
    the padding), and only index 0's earlier full cycle is a payload.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    for env in ENV_ORDER:
        assembler.accept(env, _chunk(0, env), window, "rev")
    assert len(enqueued) == 1
    assert assembler.next_batch_index == 1

    # opencodeinstruct never contributes to the final cycle at all.
    assembler.accept(
        "openmathinstruct", _partial_chunk(1, "openmathinstruct", 3),
        window, "rev",
    )

    assembler.close()

    assert len(enqueued) == 1  # only the earlier full cycle
    tombstone_keys = {key for key, _ in tombstoned}
    assert tombstone_keys == {
        encoded_window_journal_key(window, idx)
        for idx in range(1, FILL_CLOSED_EMISSIONS_PER_WINDOW)
    }

    all_written = [key for key, _ in enqueued] + [
        key for key, _ in tombstoned
    ]
    assert len(all_written) == len(set(all_written))
    assert set(all_written) == {
        encoded_window_journal_key(window, idx)
        for idx in range(FILL_CLOSED_EMISSIONS_PER_WINDOW)
    }
    assert assembler.next_batch_index == FILL_CLOSED_EMISSIONS_PER_WINDOW


def test_close_drains_pending_before_deciding_the_remainder(monkeypatch):
    """R19 Critical: opencodeinstruct sends TWO full B_BATCH chunks --
    the second queues into ``_pending`` because the accumulator's code
    slot from the first chunk is already full -- while math sends only
    a partial chunk. ``close()`` used to look only at
    ``self._accumulator``, so the queued second code chunk was silently
    dropped: ``remainder_snapshot()`` read right after ``close()`` kept
    reporting it as outstanding pending work, contradicting its own
    docstring. ``close()`` must drain ``_pending`` first (looping, since
    a forced remainder write frees capacity for the next pass) so both
    ``_pending`` and the accumulator are genuinely empty once it
    returns, and every key this window ever touches is written exactly
    once -- nothing vanishes, and nothing is double-counted.
    """
    import reliquary.infrastructure.training_payload_queue as queue_module
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    enqueued: list[tuple[int, bytes]] = []
    tombstoned: list[tuple[int, bytes]] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append((key, data)),
        tombstone_fn=lambda key, data: tombstoned.append((key, data)),
    )

    code_chunk_1 = _chunk(0, "opencodeinstruct")
    code_chunk_2 = _chunk(1, "opencodeinstruct")
    math_partial = _partial_chunk(0, "openmathinstruct", B_BATCH - 2)

    assembler.accept("opencodeinstruct", code_chunk_1, window, "rev")
    assembler.accept("opencodeinstruct", code_chunk_2, window, "rev")
    assembler.accept("openmathinstruct", math_partial, window, "rev")

    pre_close = assembler.remainder_snapshot()
    assert pre_close["pending"]["opencodeinstruct"] == 1
    assert enqueued == []
    assert tombstoned == []

    assembler.close()

    post_close = assembler.remainder_snapshot()
    assert post_close["pending"] == {env: 0 for env in ENV_ORDER}
    assert post_close["in_accumulator"] == {env: 0 for env in ENV_ORDER}

    all_written = [key for key, _ in enqueued] + [
        key for key, _ in tombstoned
    ]
    assert len(all_written) == len(set(all_written))
    assert set(all_written) == {
        encoded_window_journal_key(window, idx)
        for idx in range(FILL_CLOSED_EMISSIONS_PER_WINDOW)
    }

    # math's only chunk was proven exactly once -- it must appear in
    # exactly one written payload, not zero (dropped) and not two
    # (duplicated).
    math_prompt_ids_seen: list[int] = []
    for _, data in enqueued:
        batches = decode_training_payload(data).batches()
        math_prompt_ids_seen.extend(_prompt_ids(batches["openmathinstruct"]))
    assert math_prompt_ids_seen == _prompt_ids(math_partial)


def test_accept_after_close_records_the_straggler_instead_of_raising(
    monkeypatch, caplog,
):
    """R19 (amended): a proof finishing just after the window seals is a
    NORMAL race, not a fault -- raising here (the original R19) escalated
    an ordinary timing race into a full proof-plane fault. A straggler
    landing after close() must return normally, bump the straggler
    counter (exposed via remainder_snapshot() for the service to carry
    into the window's archive/telemetry), and log the loss at ERROR --
    lost to training since the window is sealed, but never silent and
    never a restart.
    """
    import logging

    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    window = 42
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
    )
    assembler.accept(
        "openmathinstruct", _partial_chunk(0, "openmathinstruct", 3),
        window, "rev",
    )
    assembler.close()

    straggler_chunk = _chunk(0, "opencodeinstruct")
    with caplog.at_level(
        logging.ERROR,
        logger="reliquary.validator.fill_closed_batch_assembler",
    ):
        assembler.accept(
            "opencodeinstruct", straggler_chunk, window, "rev",
        )  # must NOT raise

    assert any(
        str(window) in record.getMessage()
        and "opencodeinstruct" in record.getMessage()
        and record.levelno == logging.ERROR
        for record in caplog.records
    )
    stragglers = assembler.remainder_snapshot()["stragglers"]
    assert stragglers["count"] == 1
    assert stragglers["last_environment"] == "opencodeinstruct"

    # A second straggler keeps counting rather than raising or resetting.
    assembler.accept(
        "opencodeinstruct", _chunk(1, "opencodeinstruct"), window, "rev",
    )
    assert (
        assembler.remainder_snapshot()["stragglers"]["count"] == 2
    )
