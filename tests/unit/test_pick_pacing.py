"""Picks are WINDOW events, paced by the trainer's own consumption.

Amendment v6.1 points 2 and 3, rulings R34/R35/R36. Task 11 built the
mechanism (``GrpoWindowBatcher.pick_training_batch``); nothing called it.
This is the caller: on the service's existing 0.5 s poll cadence, one
pick EVENT fires when every environment can seat a full ``B_BATCH`` and
the pacing gate for that pick is open --

  * picks 1..depth  -> ``FILL_CLOSED_FIRST_PICK_SECONDS`` after the
    window opened (nothing has been emitted yet, so there is no cursor
    to wait on);
  * pick k > depth  -> the trainer's cursor has reached THIS window's
    batch ``k - depth - 1``, i.e. the trainer has actually consumed the
    batch ``depth`` picks back and still holds ``depth - 1`` in hand.

No constant encodes the trainer's step time (R31): the cadence is
measured off the cursor, so it survives any model/hardware change on the
train worker unchanged.

And R39, which replaced R35's checkpoint reading: after a v6 window
closes, the next one does not open until the trainer has CONSUMED the
batches that window emitted. Waiting on a PUBLICATION was structurally
wrong -- the trainer publishes at 16 TRAINED batches cumulative, not per
window, so an underfilled window waited a full backstop for a checkpoint
that was never coming and a mid-window publish over-armed the next
window. Consumption has neither failure mode.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from reliquary.constants import B_BATCH
from reliquary.infrastructure.training_payload_queue import (
    encoded_window_journal_key,
)
from reliquary.validator.service import ValidationService
from tests.unit.test_grpo_window_batcher import (
    PrivateRewardFakeEnv, _make_batcher,
)

MATH = "openmathinstruct"
CODE = "opencodeinstruct"
WINDOW = 500


# ------------------------------------------------------------------ #
# constants                                                          #
# ------------------------------------------------------------------ #

def test_the_pacing_constants_have_the_amendments_defaults():
    """Spec 'Configuration added': first pick 30 s after open, pipeline
    depth 2 (R34 -- the trainer always holds one batch in hand)."""
    from reliquary.constants import (
        FILL_CLOSED_FIRST_PICK_SECONDS,
        FILL_CLOSED_PICK_PIPELINE_DEPTH,
    )
    assert FILL_CLOSED_FIRST_PICK_SECONDS == 30.0
    assert FILL_CLOSED_PICK_PIPELINE_DEPTH == 2


# ------------------------------------------------------------------ #
# harness                                                            #
# ------------------------------------------------------------------ #

class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _two_env_window(monkeypatch, *, picks_target=16, clock=None):
    """One shared FillState across a math and a code batcher, exactly as
    ``_build_window_batchers`` wires them."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    # The journal key encoding is itself gated: with the flag off it
    # collapses every batch of a window onto the bare window number, so a
    # cursor test would be comparing 500 to 500. Production runs it armed.
    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    clock = clock or _Clock()
    shared = batcher_module.FillState(
        budgets={MATH: 512, CODE: 512}, picks_target=picks_target
    )
    math = _make_batcher(window_start=WINDOW, time_fn=clock)
    code = _make_batcher(
        window_start=WINDOW, env=PrivateRewardFakeEnv(), time_fn=clock
    )
    for batcher in (math, code):
        batcher.fill_state = shared
        batcher.mark_window_opened()
    return math, code, shared, clock


def _prove(batcher, environment, count, *, rate=1.0):
    """Grow the pick pool the way ``_reconcile_fill_state_decisions``
    does: a proof completing only ADDS to the pool, never emits."""
    import reliquary.validator.batcher as batcher_module
    with batcher.fill_state.lock:
        for index in range(count):
            batcher.fill_state.record_proven(environment)
            batcher._proven_groups.setdefault(environment, []).append(
                batcher_module._ProvenGroup(
                    value=SimpleNamespace(
                        name=f"{environment}-{index}", eos_tokens=1
                    ),
                    rate=rate,
                    payload_bytes=100,
                    receipt_id=f"{environment}-{index}",
                )
            )


def _capture(*batchers):
    events = []
    for batcher in batchers:
        batcher._emit_training_batch_fn = (
            lambda environment, groups, window_start, revision: events.append(
                (environment, len(groups))
            )
        )
    return events


class _CursorQueue:
    def __init__(self, cursor=None):
        self.cursor = cursor
        self.reads = 0

    def fetch_step_cursor(self, fetch_fn=None):
        # R38: the validator reads the trainer's cursor via a bounded
        # remote GET (fetch_step_cursor), not the local-file read
        # (read_step_cursor) -- the two run on different hosts. This
        # fake stands in for whichever transport call
        # _read_trainer_step_cursor makes.
        self.reads += 1
        return self.cursor


def _service(monkeypatch, *, cursor=None, enabled=True, **attrs):
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", enabled)
    service = ValidationService.__new__(ValidationService)
    service._training_payload_queue = _CursorQueue(cursor)
    for key, value in attrs.items():
        setattr(service, key, value)
    return service


# ------------------------------------------------------------------ #
# A. the pick event loop                                             #
# ------------------------------------------------------------------ #

def test_the_first_pick_waits_for_the_first_pick_floor(monkeypatch):
    """Pick 1 is gated on wall time, not on the cursor -- the window has
    emitted nothing, so there is nothing for the trainer to have
    consumed. Before the floor: no event, even with both pools full."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    events = _capture(math, code)
    service = _service(monkeypatch)

    clock.now += 29.0
    assert service._drive_fill_closed_picks([math, code]) is False
    assert events == []
    assert shared.snapshot()["picks_emitted"] == 0

    clock.now += 2.0
    assert service._drive_fill_closed_picks([math, code]) is True
    assert sorted(events) == [(CODE, B_BATCH), (MATH, B_BATCH)]
    assert shared.snapshot()["picks_emitted"] == 1


def test_the_second_pick_waits_for_the_end_of_the_first_step_both_sides(
    monkeypatch,
):
    """R41: only pick 1 rides the time floor. Pick 2 is gated on the
    trainer having CONSUMED batch 0 -- the end of the first training
    step -- both boundary sides pinned. One below the encoded key is
    not enough; the key itself is."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch)

    # Burn the ONE free pick off the time floor.
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 1

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    batch_zero = encoded_window_journal_key(WINDOW, 0)

    service._training_payload_queue.cursor = batch_zero - 1
    assert service._drive_fill_closed_picks([math, code]) is False
    assert shared.snapshot()["picks_emitted"] == 1

    service._training_payload_queue.cursor = batch_zero
    assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 2

    # Picks 2 and 3 share the batch-0 gate (the max(0, ...) clamp), which
    # is what refills the depth-2 buffer right after the step-1 bubble.
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 3


def test_pick_depth_plus_two_waits_for_batch_one(monkeypatch):
    """The off-by-one walks: pick k waits on batch ``k - depth - 1``, so
    a cursor parked on batch 0 releases pick 3 and NOT pick 4."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(
        monkeypatch, cursor=encoded_window_journal_key(WINDOW, 0)
    )
    for _ in range(3):
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 3

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is False

    service._training_payload_queue.cursor = encoded_window_journal_key(
        WINDOW, 1
    )
    assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 4


def test_one_environment_short_blocks_the_whole_event(monkeypatch):
    """R36: a pick is a WINDOW event. The ready environment is NOT picked
    alone -- one DAPO batch is every environment's k-th chunk, and half
    a batch is not a batch."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH - 1)
    events = _capture(math, code)
    service = _service(monkeypatch)
    clock.now += 60.0

    assert service._drive_fill_closed_picks([math, code]) is False
    assert events == []
    assert shared.snapshot()["picks_emitted"] == 0

    _prove(code, CODE, 1)
    assert service._drive_fill_closed_picks([math, code]) is True
    assert sorted(events) == [(CODE, B_BATCH), (MATH, B_BATCH)]


def test_a_window_wide_event_advances_the_count_exactly_once(monkeypatch):
    """R36 again, from the counter's side: two environments, one event,
    ``picks_emitted`` +1 -- not +2. Counting per environment would close
    a two-environment window at half its batches."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch, cursor=10**9)
    for expected in (1, 2, 3):
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        assert service._drive_fill_closed_picks([math, code]) is True
        assert shared.snapshot()["picks_emitted"] == expected


def test_every_event_walks_its_own_cursor_boundary_and_closes(monkeypatch):
    """R35 end to end, with NO shortcut cursor: every pick past the depth
    is stepped over its own boundary, so the LAST event's gate
    (k = picks_target -> batch ``k - depth - 1``) is pinned on both sides
    like every other one. The window then closes and ``poll_deadline``
    seals it on the very next poll."""
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)

    target = FILL_CLOSED_EMISSIONS_PER_WINDOW
    math, code, shared, clock = _two_env_window(
        monkeypatch, picks_target=target
    )
    clock.now += 60.0
    service = _service(monkeypatch, cursor=None)
    queue = service._training_payload_queue

    for k in range(1, target + 1):
        assert math.poll_deadline() is False
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        if k >= 2:
            consumed = encoded_window_journal_key(WINDOW, max(0, k - 3))
            # One short of the boundary: still gated, including on the
            # window's LAST event.
            queue.cursor = consumed - 1
            assert service._drive_fill_closed_picks([math, code]) is False
            queue.cursor = consumed
        assert service._drive_fill_closed_picks([math, code]) is True
        assert shared.snapshot()["picks_emitted"] == k

    assert shared.is_closed() is True
    assert service._drive_fill_closed_picks([math, code]) is False
    assert math.poll_deadline() is True
    assert code.poll_deadline() is True


def test_the_backstop_still_seals_a_stalled_window(monkeypatch):
    """No pick ever fires (empty supply), and the window still ends:
    ``FILL_CLOSED_MAX_SECONDS`` is untouched by the pacing."""
    from reliquary.constants import FILL_CLOSED_MAX_SECONDS

    math, code, shared, clock = _two_env_window(monkeypatch)
    service = _service(monkeypatch)
    clock.now += FILL_CLOSED_MAX_SECONDS - 1.0
    assert service._drive_fill_closed_picks([math, code]) is False
    assert math.poll_deadline() is False

    clock.now += 2.0
    assert math.poll_deadline() is True
    assert shared.is_closed() is False


def test_an_absent_cursor_still_lets_the_first_pick_fire(monkeypatch):
    """Failure mode from the spec: cursor stale/absent -> picks stop.
    But pick 1 never depended on it (R41: the only floor pick), so it
    still fires and the trainer gets something to chew on."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch, cursor=None)

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 1

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is False
    assert shared.snapshot()["picks_emitted"] == 1


def test_a_stale_cursor_from_an_older_window_does_not_release_a_pick(
    monkeypatch,
):
    """The encoded key is monotone across windows, so a cursor left on
    the PREVIOUS window's last batch is simply too small -- no special
    staleness rule needed."""
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(
        monkeypatch,
        cursor=encoded_window_journal_key(
            WINDOW - 1, FILL_CLOSED_EMISSIONS_PER_WINDOW - 1
        ),
    )
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is True

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is False


def test_the_cursor_is_read_once_per_tick_and_only_when_gating(monkeypatch):
    """The read is an I/O hop on a 0.5 s loop: never taken while the pick
    is gated on the clock, at most once when it is gated on the cursor."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch, cursor=10**9)
    queue = service._training_payload_queue

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    service._drive_fill_closed_picks([math, code])
    assert queue.reads == 0

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    service._drive_fill_closed_picks([math, code])
    assert queue.reads == 1


def test_a_pool_short_window_never_reads_the_cursor(monkeypatch):
    """Readiness is checked before the gate, so candidate supply that is not
    producing costs no cursor I/O at all."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch, cursor=10**9)
    assert service._drive_fill_closed_picks([math, code]) is False
    assert service._training_payload_queue.reads == 0


def test_an_environment_that_refuses_after_a_sibling_picked_logs_error(
    monkeypatch, caplog,
):
    """Should be impossible -- the pool only grows between the readiness
    check and the pick -- so if it ever happens it is a real fault and
    must be loud, not a silently half-filled batch."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    clock.now += 60.0
    service = _service(monkeypatch)
    monkeypatch.setattr(code, "pick_training_batch", lambda: False)

    with caplog.at_level(logging.ERROR):
        assert service._drive_fill_closed_picks([math, code]) is True
    assert any(
        record.levelno == logging.ERROR
        and "pick event" in record.getMessage()
        for record in caplog.records
    )
    # R37: the event is INCOMPLETE, which is what the ERROR reports --
    # math took its half, code never did, and the window-wide count (the
    # min over environments) therefore does not move at all. It used to
    # read 1 here, off a single window-wide counter that could not tell a
    # half-taken event from a finished one.
    assert shared.snapshot()["picks_by_environment"] == {MATH: 1, CODE: 0}
    assert shared.snapshot()["picks_emitted"] == 0


def test_every_environment_refusing_is_logged_too(monkeypatch, caplog):
    """A total refusal is not "nothing happened": readiness said every
    environment could seat a batch and then none did, which is the same
    broken invariant as a partial refusal and has to be as loud."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    clock.now += 60.0
    service = _service(monkeypatch)
    monkeypatch.setattr(math, "pick_training_batch", lambda: False)
    monkeypatch.setattr(code, "pick_training_batch", lambda: False)

    with caplog.at_level(logging.ERROR):
        assert service._drive_fill_closed_picks([math, code]) is False
    assert any(
        record.levelno == logging.ERROR
        and "pick event" in record.getMessage()
        for record in caplog.records
    )
    assert shared.snapshot()["picks_emitted"] == 0


def test_with_the_gate_off_no_pick_ever_fires(monkeypatch):
    """v4/v5 stay byte-identical: the loop is inert without the flag."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    events = _capture(math, code)
    clock.now += 60.0
    service = _service(monkeypatch, enabled=False)

    assert service._drive_fill_closed_picks([math, code]) is False
    assert events == []
    assert service._training_payload_queue.reads == 0


def test_batchers_without_a_fill_state_are_not_driven(monkeypatch):
    """Legacy batchers (``fill_state is None``) are skipped entirely --
    the same shape ``poll_deadline``'s v6 branch uses."""
    service = _service(monkeypatch)
    legacy = SimpleNamespace(fill_state=None)
    assert service._drive_fill_closed_picks([legacy]) is False


def test_can_pick_answers_the_readiness_question_cheaply(monkeypatch):
    """The batcher-side read the event loop is built on: True exactly
    when this environment holds ``B_BATCH`` proven-and-unpicked groups,
    and False again once a pick has claimed them."""
    math, code, shared, clock = _two_env_window(monkeypatch)
    assert math.can_pick() is False
    _prove(math, MATH, B_BATCH - 1)
    assert math.can_pick() is False
    _prove(math, MATH, 1)
    assert math.can_pick() is True

    _capture(math)
    assert math.pick_training_batch() is True
    assert math.can_pick() is False


def test_can_pick_is_false_once_the_window_is_closed(monkeypatch):
    """R37: closing takes BOTH environments' halves of the last event --
    and either half alone is already enough to stop the environment that
    took it, since ``can_pick`` gates on that environment's own ordinal."""
    math, code, shared, clock = _two_env_window(monkeypatch, picks_target=1)
    _prove(math, MATH, B_BATCH)
    assert math.can_pick() is True

    shared.record_pick(MATH)
    assert math.can_pick() is False
    assert shared.is_closed() is False

    shared.record_pick(CODE)
    assert shared.is_closed() is True
    assert math.can_pick() is False




# ------------------------------------------------------------------ #
# B. rotation waits on CONSUMPTION (R39)                              #
# ------------------------------------------------------------------ #

def _run(coro):
    return asyncio.run(coro)


def _rotation_service(monkeypatch, *, key, cursor=None, enabled=True,
                      freeze=False, requires_successor=False):
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.service as service_module
    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", enabled)
    monkeypatch.setattr(
        service_module, "FILL_CLOSED_ROTATION_POLL_SECONDS", 0.0
    )
    monkeypatch.setenv("RELIQUARY_DISABLE_TRAIN", "1" if freeze else "0")
    service = ValidationService.__new__(ValidationService)
    service._training_payload_queue = _CursorQueue(cursor)
    service._fill_closed_rotation_store = None
    if key is None:
        service._fill_closed_rotation_gate = None
    else:
        from reliquary.validator.fill_closed_rotation import (
            FillClosedRotationGate,
        )
        service._fill_closed_rotation_gate = FillClosedRotationGate(
            source_window=WINDOW,
            required_journal_key=key,
            parent_checkpoint_n=7,
            parent_revision="7" * 40,
            durable_payload_count=3,
            requires_successor=requires_successor,
        )
    service._checkpoint_store = SimpleNamespace(
        repo_id="org/repo",
        current_manifest=lambda: SimpleNamespace(
            checkpoint_n=7, revision="7" * 40
        )
    )
    service._window_n = WINDOW
    service._window_preparation_stage = None
    service.stages = []
    service._set_window_preparation_stage = service.stages.append
    return service


def test_an_underfilled_window_waits_only_for_its_own_batches(monkeypatch):
    """R39, the case revision-comparison got wrong. This window emitted 3
    batches, not 16, so the trainer will NOT publish a checkpoint off it
    -- and rotation must not wait for one. It waits for exactly what this
    window put in the journal: the cursor reaching batch 2."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(monkeypatch, key=last, cursor=last - 1)

    async def drive():
        task = asyncio.create_task(service._wait_for_fill_closed_rotation())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()  # still holding on batch 2
        service._training_payload_queue.cursor = last
        return await task

    assert _run(drive()) == "batches_consumed"


def test_rotation_releases_at_once_when_the_cursor_is_already_past(
    monkeypatch,
):
    """The common case: the trainer consumed the window's last batch
    while the GPU half was still archiving, so there is nothing to wait
    for and no poll loop is entered."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(monkeypatch, key=last, cursor=last + 4)
    assert _run(service._wait_for_fill_closed_rotation()) == (
        "batches_consumed"
    )
    assert service._training_payload_queue.reads == 1
    assert service.stages == []


def test_a_window_that_emitted_nothing_does_not_wait_at_all(monkeypatch):
    """Zero emitted -> nothing for the trainer to consume -> rotate
    immediately. Arming on "a window closed" instead would stall a dead
    window for a whole backstop."""
    service = _rotation_service(monkeypatch, key=None, cursor=None)
    assert _run(service._wait_for_fill_closed_rotation()) == "not_armed"
    assert service._training_payload_queue.reads == 0


def test_rotation_is_bounded_by_the_fill_closed_backstop(monkeypatch):
    """A dead trainer reaches a visible backstop but admission stays closed."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_MAX_SECONDS", 0.0)
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(monkeypatch, key=last, cursor=None)
    assert _run(service._wait_for_fill_closed_rotation()) == (
        "rotation_blocked_timeout"
    )
    assert service._fill_closed_rotation_gate is not None
    assert service.stages[-1] == "fill_closed_rotation_blocked"


def test_an_emergency_freeze_keeps_admission_closed(monkeypatch, caplog):
    """The incident switch freezes learning and cannot bypass rotation."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(
        monkeypatch, key=last, cursor=None, freeze=True
    )
    with caplog.at_level(logging.WARNING):
        assert _run(service._wait_for_fill_closed_rotation()) == (
            "emergency_freeze"
        )
    assert any(
        "freeze" in record.getMessage() for record in caplog.records
    )
    assert service._training_payload_queue.reads == 0
    assert service._fill_closed_rotation_gate is not None
    assert service.stages[-1] == "fill_closed_rotation_frozen"


def test_the_wait_publishes_a_preparation_stage(monkeypatch):
    """A wait that can last the whole backstop must say so on /state, or
    a stalled rotation is indistinguishable from a hung validator."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(monkeypatch, key=last, cursor=last - 1)

    async def drive():
        task = asyncio.create_task(service._wait_for_fill_closed_rotation())
        for _ in range(5):
            await asyncio.sleep(0)
        assert service.stages == ["fill_closed_rotation_wait"]
        service._training_payload_queue.cursor = last
        await task

    _run(drive())


def test_with_the_gate_off_rotation_never_waits(monkeypatch):
    service = _rotation_service(
        monkeypatch, key=encoded_window_journal_key(WINDOW, 2), cursor=None,
        enabled=False,
    )
    assert _run(service._wait_for_fill_closed_rotation()) == "disabled"
    assert service._training_payload_queue.reads == 0


def test_the_arming_records_the_last_emitted_batchs_journal_key(monkeypatch):
    """Arming binds the last durable key and its parent checkpoint."""
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.service as service_module
    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    service = ValidationService.__new__(ValidationService)
    service._fill_closed_rotation_store = None
    service._fill_closed_rotation_gate = SimpleNamespace(stale=True)
    service._checkpoint_store = SimpleNamespace(
        current_manifest=lambda: SimpleNamespace(
            checkpoint_n=7, revision="7" * 40
        )
    )

    service._fill_closed_assembler = SimpleNamespace(
        window_start=WINDOW, next_batch_index=0, durable_payload_count=0,
    )
    service._arm_fill_closed_rotation_gate()
    assert service._fill_closed_rotation_gate is None

    service._fill_closed_assembler = SimpleNamespace(
        window_start=WINDOW, next_batch_index=3, durable_payload_count=3,
    )
    service._arm_fill_closed_rotation_gate()
    gate = service._fill_closed_rotation_gate
    assert gate.required_journal_key == encoded_window_journal_key(WINDOW, 2)
    assert gate.parent_checkpoint_n == 7
    assert gate.parent_revision == "7" * 40
    assert gate.requires_successor is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("next_batch_index", True),
        ("next_batch_index", 3.0),
        ("next_batch_index", "3"),
        ("window_start", True),
        ("window_start", 42.0),
        ("window_start", "42"),
        ("durable_payload_count", False),
        ("durable_payload_count", 3.0),
        ("durable_payload_count", "3"),
    ],
)
def test_rotation_arming_does_not_coerce_persisted_fields(
    monkeypatch,
    field,
    value,
):
    import reliquary.infrastructure.training_payload_queue as queue_module
    import reliquary.validator.service as service_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    service = ValidationService.__new__(ValidationService)
    service._fill_closed_rotation_store = None
    service._fill_closed_rotation_gate = None
    assembler = {
        "window_start": WINDOW,
        "next_batch_index": 3,
        "durable_payload_count": 3,
    }
    assembler[field] = value
    service._fill_closed_assembler = SimpleNamespace(**assembler)
    service._checkpoint_store = SimpleNamespace(
        current_manifest=lambda: SimpleNamespace(
            checkpoint_n=7,
            revision="7" * 40,
        )
    )

    with pytest.raises(RuntimeError, match="not canonical"):
        service._arm_fill_closed_rotation_gate()

    assert service._fill_closed_rotation_gate is None


def test_the_arming_is_consumed_by_one_wait(monkeypatch):
    """One close, one wait: a second rotation must not re-wait on a key
    the trainer already passed (or, worse, on a stale window's key)."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(monkeypatch, key=last, cursor=last)
    assert _run(service._wait_for_fill_closed_rotation()) == (
        "batches_consumed"
    )
    assert service._fill_closed_rotation_gate is None
    assert _run(service._wait_for_fill_closed_rotation()) == "not_armed"


def test_full_window_requires_covering_checkpoint_adoption(monkeypatch):
    """Cursor consumption alone cannot open work derived from stale weights."""
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(
        monkeypatch,
        key=last,
        cursor=last,
        requires_successor=True,
    )

    async def no_poll():
        return None

    service._advance_fill_closed_checkpoint_adoption = no_poll

    async def drive():
        task = asyncio.create_task(service._wait_for_fill_closed_rotation())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
        gate = service._fill_closed_rotation_gate.record_adoption(
            checkpoint_n=8,
            revision="8" * 40,
            trained_cursor=last,
        )
        service._fill_closed_rotation_gate = gate
        service._checkpoint_store = SimpleNamespace(
            current_manifest=lambda: SimpleNamespace(
                checkpoint_n=8, revision="8" * 40
            )
        )
        return await task

    assert _run(drive()) == "checkpoint_adopted"
    assert service._fill_closed_rotation_gate is None


def test_candidate_cursor_must_cover_the_rotation_key(monkeypatch):
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(
        monkeypatch,
        key=last,
        cursor=last,
        requires_successor=True,
    )

    service._record_fill_closed_checkpoint_candidate({
        "checkpoint_n": 8,
        "repo_id": "org/repo",
        "revision": "8" * 40,
        "trained_window_cursor": last - 1,
    })
    assert service._fill_closed_rotation_gate.adopted_revision is None

    service._record_fill_closed_checkpoint_candidate({
        "checkpoint_n": 8,
        "repo_id": "org/repo",
        "revision": "9" * 40,
        "trained_window_cursor": last,
    })
    assert service._fill_closed_rotation_gate.adopted_revision == "9" * 40
    # Recording a covering candidate is not adoption.  The active manifest
    # still names the parent, so readiness remains false.
    assert service._fill_closed_rotation_adoption_ready(
        service._fill_closed_rotation_gate
    ) is False


@pytest.mark.parametrize("checkpoint_n", [True, 8.0, "8", -1])
def test_candidate_identity_is_not_coerced(monkeypatch, checkpoint_n):
    last = encoded_window_journal_key(WINDOW, 2)
    service = _rotation_service(
        monkeypatch,
        key=last,
        cursor=last,
        requires_successor=True,
    )

    service._record_fill_closed_checkpoint_candidate({
        "checkpoint_n": checkpoint_n,
        "repo_id": "org/repo",
        "revision": "9" * 40,
        "trained_window_cursor": last,
    })

    assert service._fill_closed_rotation_gate.adopted_revision is None
