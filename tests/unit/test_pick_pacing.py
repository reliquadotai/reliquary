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

And R35's other half: after a v6 window closes, the next one does not
open until the service has DETECTED the checkpoint the closed window's
16 batches produced -- the synchronisation point that already exists
because miners need the new revision to generate against.
"""
import asyncio
import logging
from types import SimpleNamespace

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

    def read_step_cursor(self):
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


def test_a_pick_past_the_depth_waits_for_the_cursor_both_sides(monkeypatch):
    """R34: pick ``depth + 1`` is gated on the trainer having CONSUMED
    this window's batch 0 -- both boundary sides pinned. One below the
    encoded key is not enough; the key itself is."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch)

    # Burn the two free picks off the time floor.
    for _ in range(2):
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        assert service._drive_fill_closed_picks([math, code]) is True
    assert shared.snapshot()["picks_emitted"] == 2

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    batch_zero = encoded_window_journal_key(WINDOW, 0)

    service._training_payload_queue.cursor = batch_zero - 1
    assert service._drive_fill_closed_picks([math, code]) is False
    assert shared.snapshot()["picks_emitted"] == 2

    service._training_payload_queue.cursor = batch_zero
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


def test_sixteen_events_close_the_window(monkeypatch):
    """R35: the window closes at the 16th PICK, and ``poll_deadline``
    seals it on the very next poll."""
    from reliquary.constants import FILL_CLOSED_EMISSIONS_PER_WINDOW

    math, code, shared, clock = _two_env_window(
        monkeypatch, picks_target=FILL_CLOSED_EMISSIONS_PER_WINDOW
    )
    clock.now += 60.0
    service = _service(monkeypatch, cursor=10**9)
    for _ in range(FILL_CLOSED_EMISSIONS_PER_WINDOW):
        assert math.poll_deadline() is False
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        assert service._drive_fill_closed_picks([math, code]) is True

    assert shared.is_closed() is True
    assert service._drive_fill_closed_picks([math, code]) is False
    assert math.poll_deadline() is True
    assert code.poll_deadline() is True


def test_the_backstop_still_seals_a_stalled_window(monkeypatch):
    """No pick ever fires (an empty fleet), and the window still ends:
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


def test_an_absent_cursor_still_lets_the_first_picks_fire(monkeypatch):
    """Failure mode from the spec: cursor stale/absent -> picks stop.
    But the first ``depth`` picks never depended on it, so they still
    fire on the time floor and the trainer gets something to chew on."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_PICK_PIPELINE_DEPTH", 2)
    math, code, shared, clock = _two_env_window(monkeypatch)
    clock.now += 60.0
    service = _service(monkeypatch, cursor=None)

    for expected in (1, 2):
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        assert service._drive_fill_closed_picks([math, code]) is True
        assert shared.snapshot()["picks_emitted"] == expected

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    assert service._drive_fill_closed_picks([math, code]) is False
    assert shared.snapshot()["picks_emitted"] == 2


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
    for _ in range(2):
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

    for _ in range(2):
        _prove(math, MATH, B_BATCH)
        _prove(code, CODE, B_BATCH)
        service._drive_fill_closed_picks([math, code])
    assert queue.reads == 0

    _prove(math, MATH, B_BATCH)
    _prove(code, CODE, B_BATCH)
    service._drive_fill_closed_picks([math, code])
    assert queue.reads == 1


def test_a_pool_short_window_never_reads_the_cursor(monkeypatch):
    """Readiness is checked before the gate, so a fleet that is not
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
# B. the next window opens on checkpoint detection (R35)              #
# ------------------------------------------------------------------ #

def _run(coro):
    return asyncio.run(coro)


class _Store:
    def __init__(self, revision):
        self.revision = revision

    def current_manifest(self):
        return (
            None if self.revision is None
            else SimpleNamespace(revision=self.revision)
        )


class _Intake:
    def __init__(self, manifests=()):
        self._manifests = list(manifests)
        self.polls = 0
        self.staged = []
        self.installed_revision = None
        self.staged_revision = None
        self._staging_revision = None

    def poll(self):
        self.polls += 1
        return self._manifests.pop(0) if self._manifests else None

    def stage(self, manifest):
        self.staged.append(manifest)
        self.staged_revision = str(manifest["revision"])
        return True

    def snapshot(self):
        return {
            "installed_revision": self.installed_revision,
            "staged_revision": self.staged_revision,
            "staging_revision": self._staging_revision,
        }


def _rotation_service(monkeypatch, *, baseline, store, intake=None,
                      enabled=True):
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", enabled)
    monkeypatch.setattr(
        service_module, "FILL_CLOSED_CHECKPOINT_POLL_SECONDS", 0.0
    )
    service = ValidationService.__new__(ValidationService)
    service._fill_closed_checkpoint_baseline = baseline
    service._checkpoint_store = store
    service._checkpoint_intake = intake
    service._intake_stage_task = None
    service._window_n = 500
    return service


def test_rotation_waits_until_a_new_revision_is_detected(monkeypatch):
    """R35: the closed window's 16 batches are in the journal; the next
    window must not open on the OLD policy. The wait releases the moment
    the intake's poll reports checkpoint N+1."""
    store = _Store("old" * 13 + "a")
    intake = _Intake([None, None, {"revision": "new" * 13 + "b"}])
    service = _rotation_service(
        monkeypatch, baseline=store.revision, store=store, intake=intake
    )
    reason = _run(service._wait_for_fill_closed_checkpoint())
    assert reason == "checkpoint_detected"
    assert intake.polls == 3
    # DETECTION releases the gate; the multi-gigabyte download it just
    # started overlaps the next window's collection rather than blocking
    # the open on it.
    assert service._intake_stage_task is not None
    assert service._fill_closed_checkpoint_baseline is None


def test_rotation_returns_at_once_when_the_swap_already_happened(
    monkeypatch,
):
    """The serial beat usually swaps in ``_train_and_publish`` BEFORE the
    loop comes round: the manifest already names N+1, so there is nothing
    to wait for and no intake poll at all."""
    store = _Store("new")
    intake = _Intake()
    service = _rotation_service(
        monkeypatch, baseline="old", store=store, intake=intake
    )
    assert _run(service._wait_for_fill_closed_checkpoint()) == (
        "checkpoint_detected"
    )
    assert intake.polls == 0


def test_rotation_is_bounded_by_the_fill_closed_backstop(monkeypatch):
    """A dead trainer would otherwise stall rotation forever. The wait
    reuses ``FILL_CLOSED_MAX_SECONDS`` and opens the next window anyway,
    saying so."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_MAX_SECONDS", 0.0)
    store = _Store("old")
    intake = _Intake()
    service = _rotation_service(
        monkeypatch, baseline="old", store=store, intake=intake
    )
    assert _run(service._wait_for_fill_closed_checkpoint()) == (
        "checkpoint_wait_timeout"
    )


def test_rotation_does_not_wait_when_no_v6_window_has_closed(monkeypatch):
    """No baseline armed (first window ever, or a window that emitted
    nothing) -> nothing to wait for."""
    store = _Store("old")
    intake = _Intake()
    service = _rotation_service(
        monkeypatch, baseline=None, store=store, intake=intake
    )
    assert _run(service._wait_for_fill_closed_checkpoint()) == "not_armed"
    assert intake.polls == 0


def test_with_the_gate_off_rotation_never_waits(monkeypatch):
    store = _Store("old")
    intake = _Intake([{"revision": "new"}])
    service = _rotation_service(
        monkeypatch, baseline="old", store=store, intake=intake,
        enabled=False,
    )
    assert _run(service._wait_for_fill_closed_checkpoint()) == "disabled"
    assert intake.polls == 0


def test_a_staged_but_unswapped_checkpoint_counts_as_detected(monkeypatch):
    """DETECTION, not installation, is the release condition (R35): the
    download can finish under the next window's collection."""
    store = _Store("old")
    intake = _Intake()
    intake.staged_revision = "new"
    service = _rotation_service(
        monkeypatch, baseline="old", store=store, intake=intake
    )
    assert _run(service._wait_for_fill_closed_checkpoint()) == (
        "checkpoint_detected"
    )
    assert intake.polls == 0


def test_a_closed_window_that_emitted_batches_arms_the_gate(monkeypatch):
    """The baseline is the revision the CLOSED window collected against,
    captured at seal -- and only when the window actually emitted
    batches, since a window that emitted nothing produces no checkpoint
    to wait for."""
    import reliquary.validator.service as service_module
    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    service = ValidationService.__new__(ValidationService)
    service._checkpoint_store = _Store("rev-a")
    service._fill_closed_checkpoint_baseline = None

    service._fill_closed_assembler = SimpleNamespace(next_batch_index=0)
    service._arm_fill_closed_checkpoint_gate()
    assert service._fill_closed_checkpoint_baseline is None

    service._fill_closed_assembler = SimpleNamespace(next_batch_index=16)
    service._arm_fill_closed_checkpoint_gate()
    assert service._fill_closed_checkpoint_baseline == "rev-a"
