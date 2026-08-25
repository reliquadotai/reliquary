"""Several proof processes per GPU.

``ProofWorkerPool`` was built as isolation, not parallelism, and its docstring
records why: *"two workers on one GPU measure x1.04 because the CUDA contexts
time-slice"*. The mechanism is right and the conclusion was right for the
configuration measured — without an MPS server two CUDA contexts genuinely
take turns.

Re-measured 2026-08-25 on an H100 PCIe, 192 real archived rollouts, the
production stack (torch 2.7.0+cu128 / transformers 5.9.0 / flash_attn 2.8.3)
and the production checkpoint:

    workers   without MPS   with MPS
          1        12.5 s     12.1 s
          2         9.1 s      6.3 s
          4         8.3 s      5.7 s
          8             -      5.8 s   (plateau)

A single Python thread cannot keep the card fed: one proof costs
``60 ms + 0.0145 ms x token``, i.e. 87 % fixed dispatch at the median rollout
length, and the GPU idles at 39 % utilisation while it runs. Extra slots fill
those gaps.

Every slot runs the unmodified ``batch=1`` proof path, so no shape, kernel or
dtype changes and no accept/reject decision can shift — verified across the
matrix, 192/192 rollouts bit-identical on every digest (all_passed, sketch
diff, p_stop, the seed counters, argmax ids and chosen probabilities).

These tests pin the slot identity rules, that a slot loads onto its physical
device, that capacity keeps counting physical GPUs, and that slots of one GPU
actually prove concurrently.
"""

from __future__ import annotations

import threading
import time

import pytest

from reliquary.validator.proof_capacity import (
    ProofCapacityQualificationError,
    expand_proof_slots,
    physical_proof_device,
    resolve_cuda_proof_devices,
)
from reliquary.validator.proof_scheduler import (
    GlobalProofScheduler,
    ProofPlan,
    ProofPlanOutcome,
    RankedProof,
)

MATH = "openmathinstruct"
CODE = "opencodeinstruct"


class _FakeCuda:
    """One physical GPU, as the production validator has."""

    @staticmethod
    def device_count():
        return 2

    @staticmethod
    def get_device_name(index):
        return "NVIDIA H100 PCIe"

    @staticmethod
    def get_device_properties(index):
        return type("Properties", (), {"uuid": f"GPU-{index}"})()


def test_a_single_slot_keeps_the_bare_device_id():
    """Existing deployments must not see their device ids change."""
    assert expand_proof_slots(("cuda:0",), 1) == ("cuda:0",)
    assert expand_proof_slots(("cuda:0", "cuda:1"), 1) == ("cuda:0", "cuda:1")


def test_extra_slots_are_suffixed_per_device():
    assert expand_proof_slots(("cuda:0",), 4) == (
        "cuda:0#0", "cuda:0#1", "cuda:0#2", "cuda:0#3",
    )
    assert expand_proof_slots(("cuda:0", "cuda:1"), 2) == (
        "cuda:0#0", "cuda:0#1", "cuda:1#0", "cuda:1#1",
    )


def test_slot_count_must_be_positive():
    with pytest.raises(ValueError, match="at least one proof slot"):
        expand_proof_slots(("cuda:0",), 0)


def test_a_slot_resolves_to_its_physical_device():
    assert physical_proof_device("cuda:0#3") == "cuda:0"
    assert physical_proof_device("cuda:1") == "cuda:1"


def test_proof_devices_names_cards_and_never_slots():
    """RELIQUARY_PROOF_DEVICES is the physical fleet; slots come from the count.

    Slot ids resolved here would flow straight into
    ``ProofCapacityQualification.validate``, which refuses non-canonical
    indices and duplicate GPU uuids — so an operator who wrote them would get
    a boot crash-loop. Refuse them at the point they are read, and keep the
    error message pointing at the one syntax that works.
    """
    with pytest.raises(
        ProofCapacityQualificationError, match="explicit cuda:<index>"
    ):
        resolve_cuda_proof_devices(("cuda:0#0", "cuda:0#1"), cuda=_FakeCuda())


def test_an_accidental_duplicate_device_is_still_refused():
    with pytest.raises(
        ProofCapacityQualificationError, match="duplicate CUDA indices"
    ):
        resolve_cuda_proof_devices(("cuda:0", "cuda:0"), cuda=_FakeCuda())


def test_slots_with_no_proof_device_are_flagged(caplog):
    """isolation on + slots > 1 + no resolved device = the slots are dropped.

    A protocol profile below v3 resolves no proof device at all, so
    ``expand_proof_slots((), 4)`` is empty, no plane is built, and the
    operator's RELIQUARY_PROOF_SLOTS_PER_DEVICE is discarded in silence. Say
    so rather than let the box run at one process while the config claims four.
    """
    import logging

    from reliquary.validator.proof_worker import assert_proof_slots_supported

    with caplog.at_level(
        logging.WARNING, logger="reliquary.validator.proof_worker"
    ):
        assert_proof_slots_supported(
            slots_per_device=4, isolation=True, proof_devices=(),
        )
    assert "no proof device" in caplog.text

    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="reliquary.validator.proof_worker"
    ):
        assert_proof_slots_supported(
            slots_per_device=4, isolation=True, proof_devices=("cuda:0",),
        )
    assert "RELIQUARY_PROOF_SLOTS_PER_DEVICE" not in caplog.text


def test_extra_slots_require_an_isolated_plane():
    """Slots only buy anything as separate interpreters.

    In-process they would be threads of the validator's own interpreter, which
    is the GIL convoy the isolated plane was built to escape — the same forward
    measured 28.7 ms alone and 29.6 s against one CPU-bound python thread. Ask
    for slots without isolation and you get the convoy, not the speed-up, so
    refuse the combination rather than quietly serving it.
    """
    from reliquary.validator.proof_worker import assert_proof_slots_supported

    assert_proof_slots_supported(slots_per_device=1, isolation=False)
    assert_proof_slots_supported(slots_per_device=4, isolation=True)
    with pytest.raises(RuntimeError, match="PROOF_PROCESS_ISOLATION"):
        assert_proof_slots_supported(slots_per_device=2, isolation=False)


def test_a_slot_loads_onto_its_physical_device(monkeypatch):
    """``.to("cuda:0#1")`` is not a device torch understands.

    The slot id exists for the pool and the scheduler; the worker that owns it
    must resolve it before touching CUDA.
    """
    import reliquary.shared.modeling as modeling
    from reliquary.validator import proof_worker

    placed: list[str] = []

    class _Model:
        def to(self, device):
            placed.append(device)
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    monkeypatch.setattr(modeling, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **k: _Model()
    )
    context = proof_worker.build_proof_context(
        checkpoint="owner/repo", device="cuda:0#1"
    )

    assert placed == ["cuda:0"]
    assert context["device"] == "cuda:0#1"


def test_hub_reload_frees_the_old_replica_before_the_new_one_lands(monkeypatch):
    """Two replicas on one card at once is 20.4 GB.

    ``_install_from_hub`` is the durable fallback taken whenever the staged
    snapshot has been rmtree'd — i.e. after any worker respawn. Holding the old
    replica while the replacement lands doubles that slot's footprint, and with
    four slots sized at 10.2 GB each the spike crosses the ~88% occupancy where
    the allocator cliff starts.
    """
    import reliquary.shared.modeling as modeling
    from reliquary.validator import proof_worker

    events: list[tuple[str, bool]] = []

    class _Replica:
        def eval(self):
            return self

        def parameters(self):
            return iter(())

    old = _Replica()
    context: dict = {"model": old, "device": "cuda:0#1", "revision": None}

    class _New(_Replica):
        def to(self, device):
            events.append(("landed", context.get("model") is old))
            return self

    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **k: _New()
    )
    proof_worker._install_from_hub(context, "owner/repo", "a" * 40)

    assert events == [("landed", False)], (
        "the old replica was still on the card when the new one landed"
    )
    assert context["revision"] == "a" * 40


def test_reload_rebuilds_a_slot_whose_replica_was_dropped(tmp_path, monkeypatch):
    """A hub install that fails after releasing the old replica leaves the slot
    with no model. The next staged-snapshot reload takes the in-place branch,
    so it must rebuild rather than die on ``None.load_state_dict`` — otherwise
    every later swap fails, ``_swap_staged_checkpoint`` rmtree's the staged dir
    and the revision never advances.
    """
    import torch
    from safetensors.torch import save_file

    import reliquary.shared.modeling as modeling
    from reliquary.validator import proof_worker

    save_file({"w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))

    class _Replica:
        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **k: _Replica()
    )
    context: dict = {"model": None, "device": "cuda:0#1", "revision": None}
    proof_worker.reload_proof_context(
        context, str(tmp_path), "a" * 40, "owner/repo",
    )

    assert context["model"] is not None
    assert context["revision"] == "a" * 40


def test_close_retires_every_slot_even_if_the_map_moves(monkeypatch):
    """``close`` must not be derailed by a concurrent slot mutation.

    It iterated ``self._workers.values()`` unlocked. With the pool-wide spawn
    lock that never mattered; with per-slot locks another dispatch thread can
    insert or pop mid-iteration, and the resulting "dictionary changed size
    during iteration" aborts the loop before the remaining slots are killed —
    exactly in the ``force=True`` path, which runs when the scheduler failed
    to quiesce and threads are still in flight.
    """
    from reliquary.validator.proof_worker import ProofWorkerPool

    support = "tests.unit.proof_worker_support"
    slots = expand_proof_slots(("cuda:0",), 3)
    pool = ProofWorkerPool(
        devices=slots,
        context_factory=f"{support}:build_counter_context",
        handler=f"{support}:echo_handler",
    )
    pool.start()
    joined: list[str] = []
    workers = dict(pool._workers)
    for device_id, worker in workers.items():
        original = worker.process.join

        def _join(timeout=None, _d=device_id, _o=original):
            joined.append(_d)
            # Stand in for another dispatch thread respawning a sibling slot.
            pool._workers[f"{_d}-late"] = worker
            return _o(timeout)

        monkeypatch.setattr(worker.process, "join", _join)

    pool.close()

    assert sorted(joined) == sorted(slots)


def test_each_slot_of_one_gpu_gets_its_own_process():
    """The pool keys on the slot, not on the card.

    Two slots sharing ``cuda:0`` must be two interpreters — one process serving
    both would put us back behind a single GIL, which is the whole point.
    """
    from reliquary.validator.proof_worker import ProofWorkerPool

    support = "tests.unit.proof_worker_support"
    pool = ProofWorkerPool(
        devices=expand_proof_slots(("cuda:0",), 2),
        context_factory=f"{support}:build_counter_context",
        handler=f"{support}:echo_handler",
    )
    pool.start()
    try:
        first = pool.call("cuda:0#0", "a")
        second = pool.call("cuda:0#1", "b")
    finally:
        pool.close()

    assert first["pid"] != second["pid"]


def test_a_slow_respawn_does_not_block_the_other_slots():
    """A replica load must not park every other slot behind it.

    ``_request`` used to take the pool-wide spawn lock around ``_worker_for``,
    which can block for ``start_timeout_seconds`` (wired to 900 s) while a
    replacement worker loads its replica. With one slot there was one dispatch
    thread and it never showed. With N slots one dead worker would stall every
    other slot far past MAX_PROOF_WALL_SECONDS, whose jobs then trip the
    active-proof deadline and fault the whole plane.
    """
    from reliquary.validator.proof_worker import ProofWorkerPool

    support = "tests.unit.proof_worker_support"
    slots = expand_proof_slots(("cuda:0",), 2)
    pool = ProofWorkerPool(
        devices=slots,
        context_factory=f"{support}:build_slow_context",
        handler=f"{support}:echo_handler",
        factory_kwargs={"slow_device": slots[0], "seconds": "5"},
    )
    # Deliberately not started: the first request spawns the worker, which is
    # the respawn path a mid-window worker death takes.
    done = threading.Event()

    def call_slow():
        try:
            pool.call(slots[0], "slow")
        finally:
            done.set()

    slow = threading.Thread(target=call_slow, daemon=True)
    slow.start()
    time.sleep(0.5)  # let the slow spawn get under way
    try:
        started = time.monotonic()
        pool.call(slots[1], "fast")
        elapsed = time.monotonic() - started
    finally:
        done.wait(30)
        pool.close()

    assert elapsed < 4.0, (
        f"the second slot waited {elapsed:.1f}s behind the first slot's spawn"
    )


def test_slots_of_one_gpu_prove_concurrently():
    """The payoff: two slots on ONE card overlap in time.

    With a single slot the scheduler runs exactly one proof at a time — that
    is what the current ``devices=("cuda:0",)`` deployment does, and what
    leaves the card at 39 % utilisation.
    """
    started = threading.Barrier(2, timeout=5)
    devices = expand_proof_slots(("cuda:0",), 2)

    def prove(invocation):
        started.wait()
        return True

    scheduler = GlobalProofScheduler(
        devices=devices,
        environments=(MATH, CODE),
        proof_callable=prove,
        checkpoint_revision="rev-a",
    )
    try:
        handle = scheduler.submit(
            ProofPlan(
                plan_id="math-window",
                environment=MATH,
                checkpoint_revision="rev-a",
                candidates=[
                    RankedProof(
                        job_id=f"job-{i}",
                        rank=i,
                        prompt_key=f"prompt-{i}",
                        payload={"rank": i},
                    )
                    for i in range(2)
                ],
                required_passes=2,
                deadline_at=time.monotonic() + 10,
            )
        )
        result = handle.result(5)

        assert result.outcome is ProofPlanOutcome.COMPLETED
    finally:
        assert scheduler.close()


def test_a_closed_pool_refuses_to_spawn_a_replacement():
    """``close`` clears the map, so a late dispatch thread finds it empty.

    Without a closed flag ``_worker_for`` would happily start a fresh process
    and load a 10.2 GB replica onto a card being torn down — after ``close``
    already took its snapshot, so nothing would ever retire it. This is the
    ``force=True`` path, which runs exactly when the scheduler failed to
    quiesce and threads are still in flight.
    """
    from reliquary.validator.proof_worker import (
        ProofWorkerPool,
        ProofWorkerUnavailable,
    )

    support = "tests.unit.proof_worker_support"
    slots = expand_proof_slots(("cuda:0",), 2)
    pool = ProofWorkerPool(
        devices=slots,
        context_factory=f"{support}:build_counter_context",
        handler=f"{support}:echo_handler",
    )
    pool.start()
    pool.close()

    with pytest.raises(ProofWorkerUnavailable, match="closed"):
        pool.call(slots[0], "late")


def test_concurrent_replica_loads_are_bounded():
    """Per-slot locks removed the only serialisation of replica loads.

    ``_spawn`` materialises the whole replica on the host before moving it to
    the card, so four slots respawning at once would spike host RSS by ~32 GB
    on a validator already sitting on a ~24 GB floor with a known leak. Bound
    the concurrent loads instead of leaving them unbounded — but not at one,
    or a single slow spawn parks the others again (see the test above).
    """
    from reliquary.validator.proof_worker import ProofWorkerPool

    support = "tests.unit.proof_worker_support"
    slots = expand_proof_slots(("cuda:0",), 4)
    pool = ProofWorkerPool(
        devices=slots,
        context_factory=f"{support}:build_slow_context",
        handler=f"{support}:echo_handler",
        factory_kwargs={"slow_device": "", "seconds": "2"},
    )
    lock = threading.Lock()
    active = 0
    peak = 0
    # Wrap the load itself, not the entry point: the bound is applied around
    # _spawn_locked, so counting _spawn calls would measure arrivals instead.
    original = pool._spawn_locked

    def _counting(device_id):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return original(device_id)
        finally:
            with lock:
                active -= 1

    pool._spawn_locked = _counting
    threads = [
        threading.Thread(target=pool.call, args=(slot, "x"), daemon=True)
        for slot in slots
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
    finally:
        pool.close()

    assert peak <= 2, f"{peak} replicas loaded at once"


def test_retire_never_kills_a_replacement_worker():
    """``_retire`` popped by device id, not by worker identity.

    A thread holding a stale worker can fail its exchange and retire the
    healthy replacement another thread just spawned and is using. With one
    slot and one dispatch thread this was unreachable; N slots plus the
    round-robin device-less paths put several threads on every slot.
    """
    from reliquary.validator.proof_worker import ProofWorkerPool

    support = "tests.unit.proof_worker_support"
    slots = expand_proof_slots(("cuda:0",), 2)
    pool = ProofWorkerPool(
        devices=slots,
        context_factory=f"{support}:build_counter_context",
        handler=f"{support}:echo_handler",
    )
    pool.start()
    try:
        stale = pool._workers[slots[0]]
        pool._retire(slots[0])
        fresh = pool._worker_for(slots[0])
        assert fresh is not stale

        pool._retire(slots[0], worker=stale)

        assert pool._workers.get(slots[0]) is fresh
        assert pool.call(slots[0], "still alive")["payload"] == "still alive"
    finally:
        pool.close()
