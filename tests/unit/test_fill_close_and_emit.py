"""The window ends on its fill, and batches leave while it is still open."""
import threading

from reliquary.constants import B_BATCH
from tests.unit.test_grpo_window_batcher import (
    FakeEnv, PrivateRewardFakeEnv, _make_batcher,
)


def test_the_window_seals_when_every_environment_is_full(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher.mark_window_opened()

    batcher.fill_state.record_proven("openmathinstruct")
    assert batcher.poll_deadline() is False

    batcher.fill_state.record_proven("opencodeinstruct")

    assert batcher.poll_deadline() is True


def test_two_env_batchers_share_one_fill_state_for_is_closed(monkeypatch):
    """R10: the service builds one ``GrpoWindowBatcher`` per environment,
    but ``FillState`` is multi-key and ``is_closed()`` needs every
    environment full. One shared instance, injected into both batchers'
    ``.fill_state``, is what makes ``is_closed()`` see across them."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    shared = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    math_batcher = _make_batcher()
    code_batcher = _make_batcher(env=PrivateRewardFakeEnv())
    math_batcher.fill_state = shared
    code_batcher.fill_state = shared

    shared.record_proven("openmathinstruct")

    assert math_batcher.fill_state.is_closed() is False
    assert code_batcher.fill_state.is_closed() is False

    shared.record_proven("opencodeinstruct")

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
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
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
    with every environment's target, assigned to ``.fill_state``."""
    import reliquary.validator.service as service_module
    from reliquary.constants import FILL_CLOSED_TARGET_GROUPS_PER_ENV
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
    assert shared.snapshot()["targets"] == {
        "openmathinstruct": FILL_CLOSED_TARGET_GROUPS_PER_ENV,
        "opencodeinstruct": FILL_CLOSED_TARGET_GROUPS_PER_ENV,
    }


def test_a_batch_is_emitted_every_b_batch_proven_groups(monkeypatch):
    """16 x 32 = 512 is arithmetic, not a schedule the miner can see."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    emitted = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 64, "opencodeinstruct": 64}
    )
    # Stub the injected callback, not ``_emit_training_batch`` itself: the
    # counter check-and-increment now lives INSIDE ``_emit_training_batch``,
    # under ``fill_state.lock``, together with the slice and watermark
    # advance (Task 7 review, Critical) -- stubbing the method out would
    # bypass that gating entirely instead of exercising it.
    batcher._emit_training_batch_fn = lambda batch: emitted.append(batch)

    for _ in range(B_BATCH):
        with batcher.fill_state.lock:
            batcher.fill_state.record_proven("openmathinstruct")
            batcher._proven_groups.setdefault(
                "openmathinstruct", []
            ).append(object())
            batcher.fill_state.record_proven("opencodeinstruct")
            batcher._proven_groups.setdefault(
                "opencodeinstruct", []
            ).append(object())
        batcher._maybe_emit_batch()

    assert len(emitted) == 1


def test_concurrent_proven_writes_and_emission_never_drop_or_duplicate_groups(
    monkeypatch,
):
    """Task 7 review, Critical: ``_maybe_emit_batch`` read
    ``fill_state.snapshot()['proven']`` and ``_emit_training_batch`` read
    ``self._proven_groups[env]`` WITHOUT ``fill_state``'s lock, while
    ``_reconcile_fill_state_decisions`` mutates both together UNDER it, on
    a possibly different proof-worker thread. A reader could observe a
    ``proven`` count that had advanced past a ``_proven_groups`` append not
    yet made, slice a short list, advance the watermark anyway, and
    permanently discard the group arriving right after -- silent loss of
    trained data. A writer thread and an emitter thread hammer the same
    batcher concurrently; every proven group must come out of the emit
    callback exactly once (no drops, no duplicates).
    """
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    n_cycles = 13  # 13 * B_BATCH == 208, close to the requested ~200
    n_groups = n_cycles * B_BATCH
    envs = ("openmathinstruct", "opencodeinstruct")

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={env: n_groups for env in envs}
    )
    emitted_batches: list[dict] = []
    batcher._emit_training_batch_fn = lambda batch: emitted_batches.append(
        batch
    )

    appended: dict[str, list[object]] = {env: [] for env in envs}

    def writer() -> None:
        for _ in range(n_groups):
            for env in envs:
                group = object()
                with batcher.fill_state.lock:
                    batcher.fill_state.record_proven(env)
                    batcher._proven_groups.setdefault(env, []).append(group)
                appended[env].append(group)

    stop = threading.Event()

    def emitter() -> None:
        while not stop.is_set():
            batcher._maybe_emit_batch()

    writer_thread = threading.Thread(target=writer)
    emitter_thread = threading.Thread(target=emitter)
    emitter_thread.start()
    writer_thread.start()
    writer_thread.join()
    stop.set()
    emitter_thread.join()
    batcher._maybe_emit_batch()  # drain whatever became ready right at the end

    emitted_by_env: dict[str, list[object]] = {env: [] for env in envs}
    for batch in emitted_batches:
        for env, groups in batch.items():
            emitted_by_env[env].extend(groups)

    for env in envs:
        emitted_ids = sorted(id(g) for g in emitted_by_env[env])
        appended_ids = sorted(id(g) for g in appended[env])
        assert emitted_ids == appended_ids, (
            env, len(emitted_ids), len(appended_ids),
        )


def test_state_advertises_which_environments_are_still_admitting(monkeypatch):
    """Math has historically under-filled, so Code will close first and
    Math will set the window duration. Code miners need to see that."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
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
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    batcher.mark_window_opened()

    # A zero wall would abort the auction path instantly; v6 must ignore it.
    assert batcher.poll_deadline() is False


def test_emission_is_hooked_from_the_reconcile_walk_not_record_proven(
    monkeypatch,
):
    """Pins the hook point named in task-7-addendum.md: ``_maybe_emit_batch``
    must be called from ``_reconcile_fill_state_decisions``, after its walk
    -- not from ``FillState.record_proven`` directly, which is now the
    scheduler's only path to accounting a PASSED group. A direct
    ``record_proven`` call (bypassing the walk entirely) must NOT emit on
    its own; only a REAL scheduler's terminal decision, walked, does.
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
            targets={"openmathinstruct": 1, "opencodeinstruct": 1}
        )
        # Stub the injected callback, not ``_emit_training_batch`` itself:
        # the counter check-and-increment now lives INSIDE
        # ``_emit_training_batch``, under ``fill_state.lock`` (Task 7
        # review, Critical), so it must stay live for repeated
        # ``_maybe_emit_batch`` calls (via ``_wait_until``'s polling) to
        # self-gate correctly instead of re-firing on every poll.
        batcher._emit_training_batch_fn = lambda batch: emitted.append(batch)

        # Bypasses the reconcile walk entirely -- must not trigger emission.
        # Also seeds a matching proven GROUP (not just the count) for the
        # other environment, so the real accumulator this test now
        # exercises can actually reach ``ready`` once math's real proof
        # completes below.
        batcher.fill_state.record_proven("opencodeinstruct")
        batcher._proven_groups.setdefault("opencodeinstruct", []).append(
            object()
        )
        assert emitted == []

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _proven() -> bool:
            batcher._drain_arrival_proof_buffer(env)
            return batcher.fill_state.snapshot()["proven"][env] == 1

        _wait_until(_proven, timeout=5.0)

        assert len(emitted) == 1
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
            targets={"openmathinstruct": 4}
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
    """Reaching the target already finalises the plan on its own (see
    ``_finalize_if_terminal_locked``); this only confirms the new,
    unconditional seal call at fill-close is a safe, idempotent no-op
    there, not a second, conflicting way to close it."""
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
            targets={"openmathinstruct": 1}
        )
        batcher.mark_window_opened()

        assert batcher.accept_submission(
            _request(prompt_idx=21, hotkey="miner")
        ).accepted

        def _closed() -> bool:
            batcher._drain_arrival_proof_buffer(env)
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
