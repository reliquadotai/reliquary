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


def test_slots_of_one_gpu_resolve_without_a_duplicate_gpu_error():
    """The duplicate-GPU guard exists to catch two ids pointing at one card by
    accident. Slots say so on purpose, so they must resolve — while keeping
    their distinct ids, which is what gives the pool one process each."""
    resolved = resolve_cuda_proof_devices(
        ("cuda:0#0", "cuda:0#1"), cuda=_FakeCuda()
    )

    assert [identity.device_id for identity in resolved] == [
        "cuda:0#0", "cuda:0#1",
    ]
    assert {identity.device_uuid for identity in resolved} == {"gpu-0"}


def test_an_accidental_duplicate_device_is_still_refused():
    with pytest.raises(
        ProofCapacityQualificationError, match="duplicate CUDA"
    ):
        resolve_cuda_proof_devices(("cuda:0", "cuda:0"), cuda=_FakeCuda())


def test_two_slots_naming_the_same_index_twice_are_refused():
    """``cuda:0#0`` twice is a config bug, not a request for two processes."""
    with pytest.raises(
        ProofCapacityQualificationError, match="duplicate CUDA"
    ):
        resolve_cuda_proof_devices(("cuda:0#0", "cuda:0#0"), cuda=_FakeCuda())


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
