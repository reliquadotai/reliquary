"""Pipelined window collection (v2) — orchestration units.

The loop itself is integration-scale; these tests pin the load-bearing
pieces: the flag default, routing/state ownership of the parameterized GPU
half (a stashed window's half must not touch the collecting window's
routing or FSM state), the serial-beat publication forecast, and the
tombstone helper's explicit-batchers + per-window dedup semantics.
"""
import asyncio
from types import SimpleNamespace

from reliquary.validator.service import ValidationService


def test_flag_defaults_off():
    from reliquary.constants import PIPELINED_WINDOWS
    assert PIPELINED_WINDOWS is False


def _run(coro):
    return asyncio.run(coro)


class _RoutingStub:
    """Host for the real _train_and_publish, wired to fail on any routing
    or FSM mutation (those belong to the collecting window)."""

    def __init__(self, batcher):
        self._active_batchers = {"env": "SENTINEL"}
        self._window_n = 999
        self._verify_task = None
        self.server = SimpleNamespace(
            set_active_batchers=lambda *_: (_ for _ in ()).throw(
                AssertionError("routing cleared by non-owning GPU half")
            )
        )
        self.tombstones = []
        self._batcher = batcher

    def _set_state(self, state):
        raise AssertionError(f"FSM mutated by non-owning GPU half: {state}")

    def _enqueue_aborted_window(self, **kwargs):
        self.tombstones.append(kwargs)


def test_beacon_invalid_pipelined_half_owns_nothing():
    """A stashed window's half hitting beacon-invalid must tombstone WITH
    its own batchers and leave routing + FSM state untouched."""
    bad = SimpleNamespace(beacon_invalid=True, window_start=123)
    stub = _RoutingStub(bad)
    _run(ValidationService._train_and_publish(
        stub, batchers={"math": bad}, window_n=123,
    ))
    assert stub._active_batchers == {"env": "SENTINEL"}
    assert len(stub.tombstones) == 1
    t = stub.tombstones[0]
    assert t["failure_stage"] == "beacon_verification"
    assert t["batchers"] == {"math": bad}


def test_pipelined_half_ignores_collecting_windows_verify_task():
    """verify_task=None + owns_routing=False must NOT fall back to
    self._verify_task (that task belongs to the collecting window)."""
    bad = SimpleNamespace(beacon_invalid=True, window_start=124)
    stub = _RoutingStub(bad)

    async def _boom():
        raise AssertionError("collecting window's verify task was awaited")

    async def _main():
        task = asyncio.get_running_loop().create_task(_boom())
        stub._verify_task = task
        await ValidationService._train_and_publish(
            stub, batchers={"math": bad}, window_n=124,
        )
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, AssertionError):
            pass

    _run(_main())
    assert len(stub.tombstones) == 1


class _ForecastStub:
    _publication_due_next_half = ValidationService._publication_due_next_half

    def __init__(self, since=0, every=16, adaptive=False, manifest=object()):
        self._checkpoint_n = 0
        self._trained_windows_since_publish = since
        self._publish_every = every
        self._adaptive_publication_pending = adaptive
        self._checkpoint_store = SimpleNamespace(
            current_manifest=lambda: manifest
        )


def test_publication_forecast_serial_beat():
    assert _ForecastStub(since=14)._publication_due_next_half() is False
    # counter+1 reaches the interval -> this half may publish -> serial beat
    assert _ForecastStub(since=15)._publication_due_next_half() is True
    assert _ForecastStub(since=0, adaptive=True)._publication_due_next_half() is True
    # bootstrap: no manifest yet -> first publish must be serial
    assert _ForecastStub(since=0, manifest=None)._publication_due_next_half() is True


class _TombstoneStub:
    _enqueue_aborted_window = ValidationService._enqueue_aborted_window
    # Bound so the aborted path can emit the detached-trainer tombstone;
    # WRITE_TRAINING_PAYLOADS is off in tests, so it early-returns.
    _write_training_tombstone = ValidationService._write_training_tombstone

    def __init__(self):
        self._active_batchers = {}
        self._archive_enqueued_windows = set()
        self._window_iteration_stage = "pipelined_train_archive"
        self.wallet = None
        self.kl_reference_state = {}
        self._late_drops = {}

    def _proof_scheduler_health_snapshot(self):
        return {}


def _fake_batcher(window_start):
    return SimpleNamespace(
        window_start=window_start,
        randomness="r" * 64,
        force_seal_reason=None,
        auction_seal_drain={},
        reward_alignment={},
    )


def test_tombstone_explicit_batchers_and_per_window_dedup(monkeypatch):
    enqueued = []

    class _StubQueue:
        def enqueue(self, window_start, archive):
            enqueued.append((window_start, archive))

    monkeypatch.setattr(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        lambda: _StubQueue(),
    )
    stub = _TombstoneStub()
    stashed = {"math": _fake_batcher(200)}

    # explicit batchers: tombstone carries the STASHED window's metadata even
    # though _active_batchers is empty (or points elsewhere).
    stub._enqueue_aborted_window(
        failure_stage="pipelined_train_archive",
        failure_type="PipelinedTrainFailure",
        batchers=stashed,
    )
    assert len(enqueued) == 1
    assert enqueued[0][0] == 200
    assert enqueued[0][1]["window_status"] == "aborted"

    # per-window dedup: same window again -> suppressed
    stub._enqueue_aborted_window(
        failure_stage="pipelined_train_archive",
        failure_type="PipelinedTrainFailure",
        batchers=stashed,
    )
    assert len(enqueued) == 1

    # a DIFFERENT window is not suppressed by the first one's archive
    # (this is the shared-boolean bug the per-window set fixes).
    other = {"math": _fake_batcher(300)}
    stub._enqueue_aborted_window(
        failure_stage="pipelined_train_archive",
        failure_type="PipelinedTrainFailure",
        batchers=other,
    )
    assert len(enqueued) == 2
    assert enqueued[1][0] == 300


class _SealCloseStub:
    _seal_wait_and_close = ValidationService._seal_wait_and_close

    def __init__(self):
        self.states = []

    async def _wait_for_window_seal(self):
        return "sealed"

    def _set_state(self, state):
        self.states.append(state)


def test_seal_wait_and_close_flips_fsm_at_deadline():
    """Miners need the OPEN -> not-OPEN edge ON the collection deadline,
    not when the concurrent GPU half joins ~a half later."""
    from reliquary.protocol.submission import WindowState

    stub = _SealCloseStub()
    reason = _run(_SealCloseStub._seal_wait_and_close(stub))
    assert reason == "sealed"
    assert stub.states == [WindowState.READY]


class _AdaptiveSealCloseStub:
    _seal_wait_and_close = ValidationService._seal_wait_and_close

    def __init__(self):
        self.states = []
        self.observed_ready = None

    async def _wait_for_window_seal(self, *, early_close_ready=None):
        self.observed_ready = early_close_ready()
        return "sealed"

    def _set_state(self, state):
        self.states.append(state)


def test_pipelined_seal_wait_forwards_gpu_completion_gate():
    """The collecting window observes completion of the previous GPU half."""
    from reliquary.protocol.submission import WindowState

    stub = _AdaptiveSealCloseStub()
    gpu_done = False
    reason = _run(stub._seal_wait_and_close(
        early_close_ready=lambda: gpu_done
    ))

    assert reason == "sealed"
    assert stub.observed_ready is False
    assert stub.states == [WindowState.READY]


def test_publication_forecast_false_under_freeze(monkeypatch):
    """RELIQUARY_DISABLE_TRAIN freezes the counter; without the gate a
    counter stuck at publish_every-1 would pin the loop serial forever."""
    monkeypatch.setenv("RELIQUARY_DISABLE_TRAIN", "1")
    stub = _ForecastStub(since=15)
    stub._checkpoint_n = 0
    assert stub._publication_due_next_half() is False


def test_publication_forecast_false_at_checkpoint_ceiling(monkeypatch):
    import reliquary.validator.service as service_mod

    monkeypatch.delenv("RELIQUARY_DISABLE_TRAIN", raising=False)
    monkeypatch.setattr(service_mod, "TRAIN_UNTIL_CHECKPOINT_N", 5)
    stub = _ForecastStub(since=15)
    stub._checkpoint_n = 5
    assert stub._publication_due_next_half() is False
