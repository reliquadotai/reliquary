"""The window ends on its fill, and batches leave while it is still open."""
import threading

from reliquary.constants import B_BATCH
from tests.unit.test_grpo_window_batcher import (
    FakeEnv, PrivateRewardFakeEnv, _make_batcher,
)


def test_the_window_seals_at_the_nth_pick(monkeypatch):
    """R35: the window used to seal when every environment reached its
    proven target -- it no longer does. Proven groups now only
    accumulate in the pool; ``record_pick()`` is called directly here to
    pin the FillState/``poll_deadline`` seam (the real pick-by-rate call
    that drives it in production is ``pick_training_batch``). R37: a pick
    ordinal is per environment, so the seal takes every environment's
    half of the last event."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=1
    )
    batcher.mark_window_opened()

    batcher.fill_state.record_proven("openmathinstruct")
    batcher.fill_state.record_proven("opencodeinstruct")
    assert batcher.poll_deadline() is False

    batcher.fill_state.record_pick("openmathinstruct")
    assert batcher.poll_deadline() is False

    batcher.fill_state.record_pick("opencodeinstruct")

    assert batcher.poll_deadline() is True


def test_two_env_batchers_share_one_fill_state_for_is_closed(monkeypatch):
    """R10: the service builds one ``GrpoWindowBatcher`` per environment,
    but ``FillState`` is shared and ``is_closed()`` is window-wide (R35:
    gated on picks, not a per-environment proven count). One shared
    instance, injected into both batchers' ``.fill_state``, is what makes
    ``is_closed()`` see across them -- and under R37 seeing across them
    means asking EVERY environment for its own ordinal, so one
    environment racing ahead never closes the window under its sibling."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=2
    )
    math_batcher = _make_batcher()
    code_batcher = _make_batcher(env=PrivateRewardFakeEnv())
    math_batcher.fill_state = shared
    code_batcher.fill_state = shared

    shared.record_pick("openmathinstruct")
    shared.record_pick("opencodeinstruct")
    shared.record_pick("openmathinstruct")

    assert math_batcher.fill_state.is_closed() is False
    assert code_batcher.fill_state.is_closed() is False

    shared.record_pick("opencodeinstruct")

    assert math_batcher.fill_state.is_closed() is True
    assert code_batcher.fill_state.is_closed() is True


def test_the_shared_fill_state_lock_is_the_same_object_on_both_batchers(
    monkeypatch,
):
    """The lock has to live ON the shared ``FillState`` instance (moved
    there per R10) -- a per-batcher ``_fill_state_lock`` would give each
    batcher its own, separate lock around the SAME shared object, which is
    no lock at all."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    math_batcher = _make_batcher()
    code_batcher = _make_batcher(env=PrivateRewardFakeEnv())
    math_batcher.fill_state = shared
    code_batcher.fill_state = shared

    assert math_batcher.fill_state.lock is code_batcher.fill_state.lock
    assert not hasattr(math_batcher, "_fill_state_lock")
    assert not hasattr(code_batcher, "_fill_state_lock")


def test_service_builds_one_shared_fill_state_and_injects_every_batcher(
    monkeypatch,
):
    """R10 (4): the service is the single place a v6 window's
    ``FillState`` is constructed. ``_build_window_batchers`` builds one
    ``GrpoWindowBatcher`` per environment (unrelated to this test); this
    pins that every one of them gets the SAME ``FillState`` instance,
    with every environment's admission budget and the window's picks
    target, assigned to ``.fill_state``."""
    import reliquary.validator.service as service_module
    from reliquary.constants import (
        FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
        FILL_CLOSED_EMISSIONS_PER_WINDOW,
    )
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap
    from tests.unit.test_service_v2 import _build_late_drop_service

    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)

    svc = _build_late_drop_service()
    math_env = FakeEnv()
    code_env = PrivateRewardFakeEnv()
    svc.envs = {"openmathinstruct": math_env, "opencodeinstruct": code_env}
    svc.env_mix = [
        ("openmathinstruct", B_BATCH), ("opencodeinstruct", B_BATCH),
    ]
    svc.env = math_env
    svc._cooldown_per_env = {
        name: CooldownMap(cooldown_windows=1_000_000) for name in svc.envs
    }
    svc._content_cooldown_per_env = {
        name: ContentCooldownMap(cooldown_windows=1_000_000)
        for name in svc.envs
    }

    batchers = svc._build_window_batchers(999)

    assert set(batchers) == {"openmathinstruct", "opencodeinstruct"}
    shared = batchers["openmathinstruct"].fill_state
    assert shared is not None
    assert batchers["opencodeinstruct"].fill_state is shared
    assert shared.snapshot()["budgets"] == {
        "openmathinstruct": FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
        "opencodeinstruct": FILL_CLOSED_ADMISSION_BUDGET_PER_ENV,
    }
    assert shared.snapshot()["picks_target"] == FILL_CLOSED_EMISSIONS_PER_WINDOW


def _build_two_env_fill_closed_service(monkeypatch, *, enabled: bool):
    """Shared setup for the R13 wiring tests below."""
    import reliquary.validator.service as service_module
    from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap
    from tests.unit.test_service_v2 import _build_late_drop_service

    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", enabled)

    svc = _build_late_drop_service()
    math_env = FakeEnv()
    code_env = PrivateRewardFakeEnv()
    svc.envs = {"openmathinstruct": math_env, "opencodeinstruct": code_env}
    svc.env_mix = [
        ("openmathinstruct", B_BATCH), ("opencodeinstruct", B_BATCH),
    ]
    svc.env = math_env
    svc._cooldown_per_env = {
        name: CooldownMap(cooldown_windows=1_000_000) for name in svc.envs
    }
    svc._content_cooldown_per_env = {
        name: ContentCooldownMap(cooldown_windows=1_000_000)
        for name in svc.envs
    }
    return svc


def test_service_wires_one_assembler_into_every_batcher(monkeypatch):
    """R13 (4): the assembler is constructed beside the shared FillState,
    same place, same gate, and injected as every batcher's
    ``emit_training_batch_fn`` -- the SAME bound method, so both
    batchers' chunks join in the same assembler instance."""
    svc = _build_two_env_fill_closed_service(monkeypatch, enabled=True)

    batchers = svc._build_window_batchers(999)

    math_fn = batchers["openmathinstruct"]._emit_training_batch_fn
    code_fn = batchers["opencodeinstruct"]._emit_training_batch_fn
    assert math_fn is not None
    assert code_fn is not None
    assert math_fn.__self__ is code_fn.__self__
    from reliquary.validator.fill_closed_batch_assembler import (
        FillClosedBatchAssembler,
    )
    assert isinstance(math_fn.__self__, FillClosedBatchAssembler)
    assert math_fn.__self__.window_start == 999


def test_no_assembler_and_no_callback_with_the_gate_off(monkeypatch):
    """(d): with FILL_CLOSED_ENABLED off, no assembler is created and
    every batcher's ``emit_training_batch_fn`` is None."""
    svc = _build_two_env_fill_closed_service(monkeypatch, enabled=False)

    batchers = svc._build_window_batchers(999)

    assert getattr(svc, "_fill_closed_assembler", None) is None
    for batcher in batchers.values():
        assert batcher._emit_training_batch_fn is None


def _pool_record(batcher_module, name, *, rate=1.0, payload_bytes=1_000):
    """One entry of the pick pool, as ``_reconcile_fill_state_decisions``
    builds it: the proven value plus the rate and payload size that
    travelled with it from its precommit."""
    return batcher_module._ProvenGroup(
        value=name, rate=rate, payload_bytes=payload_bytes, receipt_id=name,
    )


def test_a_pick_hands_over_one_b_batch_chunk_of_its_own_environment(
    monkeypatch,
):
    """16 x 32 = 512 is arithmetic, not a schedule the miner can see.

    R13: a batcher's own emission depends ONLY on its own environment
    (assembly across environments moved to the service), so the callback
    signature is ``(environment, groups, window_start,
    checkpoint_revision)`` for ONE chunk of THIS batcher's own
    environment -- not a cross-env dict.

    Amendment v6.1: what triggers that callback is a PICK, not the
    arrival of the B_BATCH-th proven group. Proving a full batch's worth
    here emits nothing on its own; the single ``pick_training_batch()``
    call below is what hands the chunk over.
    """
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    emitted = []
    batcher = _make_batcher()  # openmathinstruct
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 64}, picks_target=16
    )
    # Stub the injected callback, not ``_claim_pick_chunk`` itself: the
    # selection, the claim and the pick accounting all live INSIDE it,
    # under ``fill_state.lock`` (Task 7 review, Critical) -- stubbing the
    # method out would bypass that gating entirely instead of exercising
    # it.
    batcher._emit_training_batch_fn = (
        lambda environment, groups, window_start, checkpoint_revision: (
            emitted.append((environment, groups, window_start))
        )
    )

    for i in range(B_BATCH):
        with batcher.fill_state.lock:
            batcher.fill_state.record_proven("openmathinstruct")
            batcher._proven_groups.setdefault(
                "openmathinstruct", []
            ).append(_pool_record(batcher_module, f"g{i}"))

    assert emitted == []

    assert batcher.pick_training_batch() is True

    assert len(emitted) == 1
    assert emitted[0][0] == "openmathinstruct"
    assert len(emitted[0][1]) == B_BATCH
    assert emitted[0][2] == batcher.window_start


def test_concurrent_proven_writes_and_picks_never_drop_or_duplicate_groups(
    monkeypatch,
):
    """Task 7 review, Critical: the readiness check and the claim used to
    read ``fill_state.snapshot()['proven']`` and ``self._proven_groups``
    WITHOUT ``fill_state``'s lock, while
    ``_reconcile_fill_state_decisions`` mutates both together UNDER it, on
    a possibly different proof-worker thread -- a reader could slice a
    list shorter than the count it had already believed and permanently
    discard the group arriving right after: silent loss of trained data.

    The lock discipline is unchanged by amendment v6.1, but what it now
    protects is the pick's selection and its ``picked`` flags rather than
    a watermark. A writer thread and a PICKING thread hammer the same
    batcher concurrently; every proven group must come out of the pick
    callback exactly once (no drops, no duplicates).
    """
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    n_cycles = 13  # 13 * B_BATCH == 208, close to the requested ~200
    n_groups = n_cycles * B_BATCH
    env = "openmathinstruct"

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={env: n_groups}, picks_target=n_cycles
    )
    emitted_chunks: list[list] = []
    batcher._emit_training_batch_fn = (
        lambda environment, groups, window_start, checkpoint_revision: (
            emitted_chunks.append(groups)
        )
    )

    appended: list[str] = []

    def writer() -> None:
        for i in range(n_groups):
            name = f"g{i}"
            with batcher.fill_state.lock:
                batcher.fill_state.record_proven(env)
                batcher._proven_groups.setdefault(env, []).append(
                    _pool_record(batcher_module, name)
                )
            appended.append(name)

    stop = threading.Event()

    def picker() -> None:
        while not stop.is_set():
            batcher.pick_training_batch()

    writer_thread = threading.Thread(target=writer)
    picker_thread = threading.Thread(target=picker)
    picker_thread.start()
    writer_thread.start()
    writer_thread.join()
    stop.set()
    picker_thread.join()
    while batcher.pick_training_batch():  # drain what the close still allows
        pass

    emitted_flat = [group for chunk in emitted_chunks for group in chunk]
    assert sorted(emitted_flat) == sorted(appended), (
        len(emitted_flat), len(appended),
    )


def test_state_advertises_which_environments_are_still_admitting(monkeypatch):
    """Math has historically under-filled, so Code will close first and
    Math will set the window duration. Code miners need to see that."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 1, "opencodeinstruct": 1}, picks_target=16
    )
    batcher.fill_state.record_proven("opencodeinstruct")

    fill = batcher.upload_precommit_conservation()["fill_state"]

    assert fill["proven"]["opencodeinstruct"] == 1
    assert fill["closed"] is False


def test_v6_does_not_consult_the_seal_time_proof_wall(monkeypatch):
    """The wall bounded ONE seal burst. v6 has no burst -- it proves
    continuously -- so the window backstop is the only time bound.

    Task 6's brief specified this as its own Step 3b and skipped it
    (NEEDS_CONTEXT); it belongs here since ``poll_deadline`` is this
    task's file (see task-7-addendum.md, "Unchanged from the brief").
    """
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "MAX_PROOF_WALL_SECONDS", 0.0)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        budgets={"openmathinstruct": 4, "opencodeinstruct": 4}, picks_target=16
    )
    batcher.mark_window_opened()

    # A zero wall would abort the auction path instantly; v6 must ignore it.
    assert batcher.poll_deadline() is False


def test_the_pool_grows_from_the_reconcile_walk_not_from_record_proven(
    monkeypatch,
):
    """Pins the hook point named in task-7-addendum.md: a PASSED group
    reaches the pool from ``_reconcile_fill_state_decisions``'s walk of
    the scheduler's own terminal decisions -- not from
    ``FillState.record_proven``, which a test double (or any future
    caller) could invoke with no scheduler decision behind it and no
    group to add.

    Amendment v6.1: the walk no longer emits anything either way. It
    grows the pool; picks are the service's call.
    """
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "B_BATCH", 1)
    env = "openmathinstruct"

    emitted = []
    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        batcher.fill_state = batcher_module.FillState(
            budgets={env: 1}, picks_target=16
        )
        batcher._emit_training_batch_fn = (
            lambda environment, groups, window_start, checkpoint_revision: (
                emitted.append((environment, groups))
            )
        )

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _proven() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            return batcher.fill_state.snapshot()["proven"][env] == 1

        _wait_until(_proven, timeout=5.0)

        # The walk put the group in the pool -- and emitted nothing.
        assert len(batcher._proven_groups[env]) == 1
        assert emitted == []

        # Bypasses the walk entirely: it moves the COUNT and nothing
        # else, so the pool still holds exactly one pickable group.
        batcher.fill_state.record_proven(env)
        assert len(batcher._proven_groups[env]) == 1

        assert batcher.pick_training_batch() is True
        assert len(emitted) == 1
        assert emitted[0] == (env, [batcher._proven_groups[env][0].value])
    finally:
        assert scheduler.close()


def test_backstop_seals_the_open_plan_so_it_finalises_instead_of_hanging(
    monkeypatch,
):
    """Nothing previously called ``scheduler.seal(...)`` when a v6 window
    closes. An open-ended plan does not finalise on exhaustion -- that is
    its whole point -- so at the backstop (target not reached) the plan
    stayed open forever: in-flight proofs still completed and got
    reconciled, but the plan never reported terminal, ``drain`` could wait
    on it forever, and the next window's plan for the same environment
    would be refused (one active plan per environment).
    """
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler, ProofPlanOutcome,
    )
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        # target=4: far short of what we ever submit, so the plan can only
        # finalise via an explicit seal, never by reaching its target.
        batcher.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 4}, picks_target=16
        )
        batcher.mark_window_opened()

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _proven() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            return batcher.fill_state.snapshot()["proven"][env] == 1

        _wait_until(_proven, timeout=5.0)

        handle = batcher._open_proof_plan_handle
        assert handle is not None
        # Exhaustion is not terminal on its own while a window still
        # admits -- confirms this test would hang without the fix, not
        # pass vacuously because the plan was already done.
        assert handle.done() is False

        # The scheduler's own plan deadline was fixed, at submission time,
        # using the REAL clock and the default FILL_CLOSED_MAX_SECONDS --
        # comfortably in the future. Patching the constant AFTER the plan
        # already exists only moves the WINDOW's own backstop check,
        # isolating this test from the scheduler's independent deadline
        # mechanism.
        monkeypatch.setattr(batcher_module, "FILL_CLOSED_MAX_SECONDS", 0.0)

        assert batcher.poll_deadline() is True

        assert handle.done() is True
        result = handle.result(timeout=1.0)
        # NEEDS_CONTEXT resolved by direct measurement (see report): with
        # the real v6 plan config (``allow_shortfall=True``, set by
        # ``_extend_proof_plan``), a short, sealed, open-ended plan
        # finalises COMPLETED with a shortfall completion_reason, not
        # CAPACITY_ABORTED -- CAPACITY_ABORTED is only what
        # ``allow_shortfall=False`` produces.
        assert result.outcome is ProofPlanOutcome.COMPLETED
        assert batcher.fill_state.snapshot()["in_flight"][env] == 0
    finally:
        assert scheduler.close()


def test_fill_close_also_seals_the_plan_which_finalises_completed(
    monkeypatch,
):
    """Reaching the plan's own ``required_passes`` (still sized off the
    admission budget, see ``_extend_proof_plan``) already finalises the
    plan on its own (see ``_finalize_if_terminal_locked``); this only
    confirms the new, unconditional seal call at fill-close is a safe,
    idempotent no-op there, not a second, conflicting way to close it.
    ``record_pick()`` is called directly to reach fill-close here (Task
    11 wires the real pick-by-rate call in production)."""
    import reliquary.validator.batcher as batcher_module
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler, ProofPlanOutcome,
    )
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload, _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    env = "openmathinstruct"

    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
    )
    try:
        batcher = _make_batcher(proof_scheduler=scheduler)
        batcher.fill_state = batcher_module.FillState(
            budgets={"openmathinstruct": 1}, picks_target=1
        )
        batcher.mark_window_opened()

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _closed() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            snap = batcher.fill_state.snapshot()
            if snap["proven"][env] >= 1 and snap["picks_emitted"] == 0:
                batcher.fill_state.record_pick(env)
            return batcher.fill_state.is_closed()

        _wait_until(_closed, timeout=5.0)

        assert batcher.poll_deadline() is True

        handle = batcher._open_proof_plan_handle
        assert handle is not None
        assert handle.done() is True
        assert (
            handle.result(timeout=1.0).outcome is ProofPlanOutcome.COMPLETED
        )
        assert batcher.fill_state.snapshot()["in_flight"][env] == 0
    finally:
        assert scheduler.close()
