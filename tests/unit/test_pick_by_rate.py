"""Seats are granted at PICKS, by precommit rate -- not in arrival order.

Amendment v6.1, point 1: proving still happens on arrival, but a proven
group only joins a POOL. A *pick* then selects the ``B_BATCH`` best by
precommit rate among the proven-and-unemitted groups of one environment.
That is what converts excess candidate capacity into length diversity:
every later pick chooses from a population that has had strictly more
time to generate, so a long answer that could never finish before the
old first-come-first-served admission filled up can now land.

The tests below pin the three things the amendment actually promises --
the rate decides, ties do NOT fall back to arrival, and a pick is an
external call rather than an automatic consequence of a proof finishing
-- plus the close consequence of R32: whatever is proven but never
picked is burned, counted and logged, and never paid.
"""
import logging
from types import SimpleNamespace

import pytest

from reliquary.constants import B_BATCH
from reliquary.validator.proof_scheduler import (
    ProofDecision,
    ProofDecisionStatus,
)
from tests.unit.test_grpo_window_batcher import (
    PrivateRewardFakeEnv, _make_batcher,
)

ENV = "openmathinstruct"


def _fill_closed_batcher(monkeypatch, *, picks_target=4, budget=512, **kw):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    batcher = _make_batcher(**kw)
    batcher.fill_state = batcher_module.FillState(
        budgets={ENV: budget}, picks_target=picks_target
    )
    batcher.mark_window_opened()
    return batcher


def _capture_picks(batcher) -> list:
    picked = []
    batcher._emit_training_batch_fn = (
        lambda environment, groups, window_start, checkpoint_revision: (
            picked.append((environment, groups))
        )
    )
    return picked


def _prove(batcher, name, *, rate, payload_bytes, eos_tokens=0, env=ENV):
    """Append one PASSED group to the pick pool, exactly as
    ``_reconcile_fill_state_decisions`` does -- rate and payload size
    carried on the record so a pick can sort without re-deriving them."""
    import reliquary.validator.batcher as batcher_module

    value = SimpleNamespace(name=name, eos_tokens=eos_tokens)
    with batcher.fill_state.lock:
        batcher.fill_state.record_proven(env)
        batcher._proven_groups.setdefault(env, []).append(
            batcher_module._ProvenGroup(
                value=value,
                rate=rate,
                payload_bytes=payload_bytes,
                receipt_id=name,
            )
        )
    return value


def _names(groups) -> set[str]:
    return {group.name for group in groups}


def test_a_late_full_rate_group_beats_an_early_low_rate_group(monkeypatch):
    """The bias this amendment removes.

    Under the watermark-order emission this replaces, the first B_BATCH
    proven groups WERE the batch -- so the slow, early group was in and
    the fast, late one (arriving after the seats were gone) could never
    be. The pick reverses that: the late group's precommit rate is what
    buys the seat, and the early group's low rate is what loses it.
    """
    batcher = _fill_closed_batcher(monkeypatch)
    picked = _capture_picks(batcher)

    # Appended FIRST, so append order alone would have seated it.
    _prove(batcher, "early-slow", rate=1.0, payload_bytes=1_000)
    for i in range(B_BATCH - 1):
        _prove(batcher, f"filler-{i}", rate=50.0, payload_bytes=5_000)
    # Appended LAST: a long answer that only finished near the close.
    _prove(batcher, "late-fast", rate=900.0, payload_bytes=90_000)

    assert batcher.pick_training_batch() is True

    assert len(picked) == 1
    environment, groups = picked[0]
    assert environment == ENV
    assert len(groups) == B_BATCH
    assert "late-fast" in _names(groups)
    assert "early-slow" not in _names(groups)


def test_a_rate_tie_goes_to_the_larger_payload_never_the_earlier_arrival(
    monkeypatch,
):
    """The rate is length-neutral (bytes over elapsed), so equal rates are
    the COMMON case, not a corner. Falling back to arrival there would
    hand the seat straight back to the shortest answer -- the very bias
    the pick exists to remove -- so the larger payload wins instead."""
    batcher = _fill_closed_batcher(monkeypatch)
    picked = _capture_picks(batcher)

    # Same rate everywhere. The SMALLEST payload arrived first.
    _prove(batcher, "early-short", rate=100.0, payload_bytes=1_000)
    for i in range(B_BATCH - 1):
        _prove(batcher, f"filler-{i}", rate=100.0, payload_bytes=50_000)
    _prove(batcher, "late-long", rate=100.0, payload_bytes=90_000)

    assert batcher.pick_training_batch() is True

    _environment, groups = picked[0]
    assert "late-long" in _names(groups)
    assert "early-short" not in _names(groups)


def test_an_unknown_rate_sorts_last_instead_of_crashing_the_pick(monkeypatch):
    """``rate_of`` misses when a receipt fell out of the admission queue.
    The buffered arrival entry already degrades that to lowest priority;
    the pick must degrade it the same way rather than raise."""
    batcher = _fill_closed_batcher(monkeypatch)
    picked = _capture_picks(batcher)

    _prove(batcher, "unknown", rate=None, payload_bytes=90_000)
    for i in range(B_BATCH):
        _prove(batcher, f"known-{i}", rate=1.0, payload_bytes=1_000)

    assert batcher.pick_training_batch() is True

    _environment, groups = picked[0]
    assert "unknown" not in _names(groups)


def test_a_pick_never_emits_a_partial_batch(monkeypatch):
    """A partial batch exists only at the backstop close, which keeps its
    own path. One group short of B_BATCH, a pick refuses outright: no
    callback, no pick recorded, and the pool is left intact for the next
    pick to choose from."""
    batcher = _fill_closed_batcher(monkeypatch)
    picked = _capture_picks(batcher)

    for i in range(B_BATCH - 1):
        _prove(batcher, f"g{i}", rate=10.0, payload_bytes=1_000)

    assert batcher.pick_training_batch() is False
    assert picked == []
    assert batcher.fill_state.snapshot()["picks_emitted"] == 0

    _prove(batcher, "last", rate=10.0, payload_bytes=1_000)

    assert batcher.pick_training_batch() is True
    assert len(picked) == 1
    assert len(picked[0][1]) == B_BATCH
    assert batcher.fill_state.snapshot()["picks_emitted"] == 1


def test_a_second_pick_never_reuses_the_first_picks_groups(monkeypatch):
    """The watermark this replaces could only ever claim a contiguous
    prefix. A pick claims an arbitrary subset, so "already picked" has to
    travel on the group itself -- or the best group is picked again in
    every later pick and the rest are never trained on at all."""
    batcher = _fill_closed_batcher(monkeypatch)
    picked = _capture_picks(batcher)

    for i in range(2 * B_BATCH):
        _prove(batcher, f"g{i}", rate=float(i), payload_bytes=1_000)

    assert batcher.pick_training_batch() is True
    assert batcher.pick_training_batch() is True
    assert batcher.pick_training_batch() is False

    first, second = _names(picked[0][1]), _names(picked[1][1])
    assert first & second == set()
    assert first | second == {f"g{i}" for i in range(2 * B_BATCH)}
    # Best-by-rate first: the top half went out in the first pick.
    assert first == {f"g{i}" for i in range(B_BATCH, 2 * B_BATCH)}


def test_a_completed_proof_no_longer_emits_on_its_own(monkeypatch):
    """Under v6.1 a completed proof only GROWS the pool. Emission is the
    service's call (Task 12 paces it on the trainer's cursor), so the
    automatic ``proven // B_BATCH`` trigger on the proof-completion path
    is gone -- reaching a full batch of proven groups emits nothing until
    somebody picks."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "B_BATCH", 1)

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _fill_closed_batcher(
            monkeypatch, budget=4, proof_scheduler=scheduler
        )
        picked = _capture_picks(batcher)

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _proven() -> bool:
            batcher._drain_arrival_proof_buffer(ENV)
            return batcher.fill_state.snapshot()["proven"][ENV] == 1

        _wait_until(_proven, timeout=5.0)

        assert picked == []

        assert batcher.pick_training_batch() is True
        assert len(picked) == 1
        assert len(picked[0][1]) == 1
    finally:
        assert scheduler.close()


def test_the_rate_and_payload_size_travel_with_the_proven_group(monkeypatch):
    """The precommit is long gone by pick time: ``rate_of`` is keyed by a
    receipt the admission queue holds for the window, and the payload size
    lives on the same entry. Both are read ONCE, at arrival, and carried
    onto the proven record -- the pick re-derives neither."""
    import reliquary.validator.batcher as batcher_module
    from tests.unit.test_rate_ordered_admission import _pending_for_receipt

    batcher = _fill_closed_batcher(monkeypatch)
    batcher.admission_queue = batcher_module.ThroughputAdmissionQueue(
        window_opened_at=batcher.window_opened_at
    )
    batcher.admission_queue.offer(
        receipt_id="fast", environment=ENV, payload_bytes=9_000,
        precommit_arrived_at=batcher.window_opened_at + 90.0,
    )
    extended = []
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    batcher._submit_arrival_proof(_pending_for_receipt(1, "fast"))

    assert len(extended) == 1
    batcher._open_proof_plan_handle = SimpleNamespace(
        decisions=lambda: (
            ProofDecision(
                job_id=extended[0].job_id,
                rank=extended[0].rank,
                prompt_key=extended[0].prompt_key,
                status=ProofDecisionStatus.PASSED,
                device_id="gpu-0",
                started_at=0.0,
                finished_at=1.0,
                value=SimpleNamespace(name="fast", eos_tokens=0),
            ),
        )
    )

    batcher._reconcile_fill_state_decisions(ENV)

    group = batcher._proven_groups[ENV][0]
    assert group.value.name == "fast"
    assert group.payload_bytes == 9_000
    assert group.rate == pytest.approx(100.0)
    assert group.receipt_id == "fast"


def test_a_window_wide_pick_is_counted_once_across_environments(monkeypatch):
    """R35 closes the window at the Nth pick, and ``picks_emitted`` is
    window-wide (one shared ``FillState``) while a batcher only ever holds
    its OWN environment's pool. One pick k is therefore N batcher calls --
    one per environment, joined into a single DAPO batch by the assembler
    -- and must advance the window-wide count ONCE, not once per
    environment, or a two-environment run would close its window after
    half the batches it is supposed to emit.

    R37 realizes that count as the MIN over per-environment ordinals: an
    event half-taken is still the event in flight, so it does not move
    until its second half lands."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 512, "opencodeinstruct": 512},
        picks_target=4,
    )
    math_batcher = _make_batcher()
    code_batcher = _make_batcher(env=PrivateRewardFakeEnv())
    for batcher in (math_batcher, code_batcher):
        batcher.fill_state = shared
        batcher.mark_window_opened()
        _capture_picks(batcher)
    for i in range(B_BATCH):
        _prove(math_batcher, f"m{i}", rate=1.0, payload_bytes=10)
        _prove(code_batcher, f"c{i}", rate=1.0, payload_bytes=10,
               env="opencodeinstruct")

    assert math_batcher.pick_training_batch() is True
    assert shared.snapshot()["picks_emitted"] == 0  # event 1 still in flight
    assert code_batcher.pick_training_batch() is True

    assert shared.snapshot()["picks_emitted"] == 1
    assert shared.snapshot()["picks_by_environment"] == {
        "openmathinstruct": 1, "opencodeinstruct": 1,
    }


def test_no_pick_happens_once_the_window_has_closed(monkeypatch):
    """``record_pick`` raises past ``picks_target``. A pick arriving after
    the close (a paced call racing the seal) is refused, not fatal."""
    batcher = _fill_closed_batcher(monkeypatch, picks_target=1)
    picked = _capture_picks(batcher)

    for i in range(2 * B_BATCH):
        _prove(batcher, f"g{i}", rate=1.0, payload_bytes=10)

    assert batcher.pick_training_batch() is True
    assert batcher.fill_state.is_closed() is True

    assert batcher.pick_training_batch() is False
    assert len(picked) == 1


def test_the_close_burns_the_proven_groups_no_pick_ever_took(
    monkeypatch, caplog
):
    """R32: over-collection is the point, so a closing window normally has
    proven surplus left. That surplus is BURNED -- counted, logged, and
    never handed to the assembler, which is the only thing that pays."""
    batcher = _fill_closed_batcher(monkeypatch, picks_target=1)
    picked = _capture_picks(batcher)

    for i in range(B_BATCH):
        _prove(batcher, f"picked-{i}", rate=100.0, payload_bytes=10,
               eos_tokens=5)
    _prove(batcher, "burned-a", rate=1.0, payload_bytes=10, eos_tokens=11)
    _prove(batcher, "burned-b", rate=1.0, payload_bytes=10, eos_tokens=13)

    assert batcher.pick_training_batch() is True

    with caplog.at_level(logging.INFO, logger="reliquary.validator.batcher"):
        assert batcher.poll_deadline() is True

    assert len(picked) == 1
    assert "burned-a" not in _names(picked[0][1])
    assert "burned-b" not in _names(picked[0][1])

    conservation = batcher.upload_precommit_conservation()
    assert conservation["fill_closed_burned_groups"] == 2
    assert conservation["fill_closed_burned_eos_tokens"] == 24
    assert any(
        "burn" in record.message.lower() and record.levelno == logging.INFO
        for record in caplog.records
    )


def test_the_backstop_close_burns_the_pool_too(monkeypatch):
    """The backstop seals a window whose picks never finished (a stalled
    trainer cursor, per the amendment's failure modes). Everything proven
    is then unpicked -- and burned, for the same reason: nothing that
    skips the assembler was ever paid."""
    import reliquary.validator.batcher as batcher_module

    batcher = _fill_closed_batcher(monkeypatch, picks_target=4)
    picked = _capture_picks(batcher)

    for i in range(B_BATCH - 1):
        _prove(batcher, f"g{i}", rate=1.0, payload_bytes=10, eos_tokens=2)

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_MAX_SECONDS", 0.0)
    assert batcher.poll_deadline() is True

    assert picked == []
    conservation = batcher.upload_precommit_conservation()
    assert conservation["fill_closed_burned_groups"] == B_BATCH - 1
    assert conservation["fill_closed_burned_eos_tokens"] == 2 * (B_BATCH - 1)


def test_the_burn_is_counted_once_however_often_the_close_is_polled(
    monkeypatch,
):
    """``poll_deadline`` is polled on a loop; the burn is a one-shot
    accounting of what the window ended with, not a per-poll tally."""
    import reliquary.validator.batcher as batcher_module

    batcher = _fill_closed_batcher(monkeypatch, picks_target=4)
    _capture_picks(batcher)
    _prove(batcher, "left-over", rate=1.0, payload_bytes=10, eos_tokens=7)

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_MAX_SECONDS", 0.0)
    assert batcher.poll_deadline() is True
    assert batcher.poll_deadline() is True

    conservation = batcher.upload_precommit_conservation()
    assert conservation["fill_closed_burned_groups"] == 1
    assert conservation["fill_closed_burned_eos_tokens"] == 7


def test_the_auction_path_has_no_picks_and_no_burn():
    """Gate off: no ``fill_state``, so a pick is refused outright and the
    burn counters stay at zero -- v4/v5 windows are untouched."""
    batcher = _make_batcher()

    assert batcher.fill_state is None
    assert batcher.pick_training_batch() is False

    conservation = batcher.upload_precommit_conservation()
    assert conservation["fill_closed_burned_groups"] == 0
    assert conservation["fill_closed_burned_eos_tokens"] == 0


def test_both_environments_take_the_final_pick_event(monkeypatch):
    """The Critical R37 fixes, reproduced end to end.

    With one window-wide counter, the first environment's Nth
    ``record_pick`` flipped ``is_closed()`` and ``_claim_pick_chunk``
    refused its sibling IN THE SAME EVENT: the sibling's B_BATCH groups
    were tombstoned unpaid and the environment that did pick had its half
    batch written alone -- 1/16 of every window's training data, silently.
    The gate is this environment's OWN ordinal now, so the last event
    completes on both sides.
    """
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 512, "opencodeinstruct": 512},
        picks_target=1,
    )
    math_batcher = _make_batcher()
    code_batcher = _make_batcher(env=PrivateRewardFakeEnv())
    picked = []
    for batcher in (math_batcher, code_batcher):
        batcher.fill_state = shared
        batcher.mark_window_opened()
        batcher._emit_training_batch_fn = (
            lambda environment, groups, window_start, revision: picked.append(
                (environment, len(groups))
            )
        )
    for i in range(B_BATCH):
        _prove(math_batcher, f"m{i}", rate=1.0, payload_bytes=10)
        _prove(code_batcher, f"c{i}", rate=1.0, payload_bytes=10,
               env="opencodeinstruct")

    assert math_batcher.pick_training_batch() is True
    # The window must NOT be closed here: its second half is still owed.
    assert shared.is_closed() is False
    assert math_batcher.poll_deadline() is False
    assert code_batcher.pick_training_batch() is True

    assert sorted(picked) == [
        ("opencodeinstruct", B_BATCH), ("openmathinstruct", B_BATCH),
    ]
    assert shared.is_closed() is True


def test_a_pick_is_refused_once_this_environment_has_taken_them_all(
    monkeypatch,
):
    """The own-ordinal gate has to bound picks by itself: the window-wide
    ``is_closed()`` stays False while a sibling lags, so it cannot be what
    stops an environment from taking a pick it no longer owns."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 512, "opencodeinstruct": 512},
        picks_target=1,
    )
    batcher = _make_batcher()
    batcher.fill_state = shared
    batcher.mark_window_opened()
    picked = _capture_picks(batcher)
    for i in range(2 * B_BATCH):
        _prove(batcher, f"m{i}", rate=1.0, payload_bytes=10)

    assert batcher.pick_training_batch() is True
    assert shared.is_closed() is False  # the code sibling still owes one

    assert batcher.pick_training_batch() is False
    assert batcher.can_pick() is False
    assert len(picked) == 1


def test_identical_groups_are_ordered_by_sequence_not_by_list_position(
    monkeypatch,
):
    """Minor (c): with an empty ``receipt_id`` and equal payload bytes the
    first three key components all tie, and a stable sort then silently
    fell back to pool order -- the arrival tie-break the docstring
    forswears. A monotone per-batcher sequence, assigned when the group is
    appended to the pool, makes the order total and explicit.

    Pinned in both directions: the same two groups in the opposite pool
    order must still resolve the same way.
    """
    import reliquary.validator.batcher as batcher_module

    def _contested(order):
        batcher = _fill_closed_batcher(monkeypatch)
        picked = _capture_picks(batcher)
        for i in range(B_BATCH - 1):
            _prove(batcher, f"filler-{i}", rate=99.0, payload_bytes=10)
        pool = batcher._proven_groups[ENV]
        first = batcher_module._ProvenGroup(
            value=SimpleNamespace(name="first", eos_tokens=0),
            rate=1.0, payload_bytes=10, receipt_id="", sequence=1,
        )
        second = batcher_module._ProvenGroup(
            value=SimpleNamespace(name="second", eos_tokens=0),
            rate=1.0, payload_bytes=10, receipt_id="", sequence=2,
        )
        with batcher.fill_state.lock:
            for group in order(first, second):
                batcher.fill_state.record_proven(ENV)
                pool.append(group)
        assert batcher.pick_training_batch() is True
        return _names(picked[0][1])

    assert "first" in _contested(lambda a, b: (a, b))
    assert "second" not in _contested(lambda a, b: (a, b))
    assert "first" in _contested(lambda a, b: (b, a))
    assert "second" not in _contested(lambda a, b: (b, a))


def test_the_real_appender_hands_out_increasing_sequences(monkeypatch):
    """The sequence is only a total order if the pool's own appender
    assigns it -- a default that never moves would tie every group."""
    batcher = _fill_closed_batcher(monkeypatch)
    for i in range(3):
        _prove(batcher, f"g{i}", rate=1.0, payload_bytes=10)

    sequences = [group.sequence for group in batcher._proven_groups[ENV]]

    assert sequences == sorted(set(sequences))


def test_a_raising_callback_returns_its_chunk_to_the_pool(monkeypatch):
    """Minor (3): the chunk is flagged picked under the lock, then handed
    to a callback that writes to the trainer journal OUTSIDE it. If that
    write raises, the groups are neither paid (nothing landed in the
    assembler) nor burned (they look picked) -- they vanish from both
    sides of the accounting. Unmark them: they stay pickable, and if no
    later pick takes them the close burns them like any other surplus."""
    batcher = _fill_closed_batcher(monkeypatch, picks_target=4)

    def _explode(environment, groups, window_start, revision):
        raise RuntimeError("journal write failed")

    batcher._emit_training_batch_fn = _explode
    for i in range(B_BATCH):
        _prove(batcher, f"g{i}", rate=1.0, payload_bytes=10, eos_tokens=3)

    with pytest.raises(RuntimeError):
        batcher.pick_training_batch()

    assert all(
        not group.picked for group in batcher._proven_groups[ENV]
    )

    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_MAX_SECONDS", 0.0)
    assert batcher.poll_deadline() is True

    conservation = batcher.upload_precommit_conservation()
    assert conservation["fill_closed_burned_groups"] == B_BATCH
    assert conservation["fill_closed_burned_eos_tokens"] == 3 * B_BATCH


def test_a_diverged_pick_is_refused_and_logged_not_raised(
    monkeypatch, caplog
):
    """The divergence guard is loud at the ``FillState`` level (it raises)
    but must not travel: ``_drive_fill_closed_picks`` runs unguarded on
    the service's 0.5 s poll, so an exception there would take the whole
    window wait down. The batcher catches it, logs ERROR and refuses the
    pick -- nothing is claimed, the sibling can still take its own half,
    and the state stays consistent."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 512, "opencodeinstruct": 512},
        picks_target=16,
    )
    batcher = _make_batcher()
    batcher.fill_state = shared
    batcher.mark_window_opened()
    picked = _capture_picks(batcher)
    for i in range(2 * B_BATCH):
        _prove(batcher, f"m{i}", rate=1.0, payload_bytes=10)

    assert batcher.pick_training_batch() is True  # math takes event 1

    # The code sibling never took its half, so math taking event 2 would
    # put the ordinals two apart -- a service miscall.
    with caplog.at_level(logging.ERROR, logger="reliquary.validator.batcher"):
        assert batcher.pick_training_batch() is False

    assert len(picked) == 1
    assert shared.snapshot()["picks_by_environment"] == {
        "openmathinstruct": 1, "opencodeinstruct": 0,
    }
    assert sum(
        1 for group in batcher._proven_groups[ENV] if group.picked
    ) == B_BATCH
    assert any(
        record.levelno == logging.ERROR and "pick" in record.getMessage()
        for record in caplog.records
    )
