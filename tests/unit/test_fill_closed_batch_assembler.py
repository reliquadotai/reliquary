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
)
from reliquary.shared.training_payload import decode_training_payload
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
