"""The proof plane must run in its own process.

Measured 2026-08-23: a proof thread sharing an interpreter with the
validator's event loop is convoyed off the GIL — the same GRAIL forward
costs 28.7 ms alone and 29.6 s against one CPU-bound python thread. These
tests pin the isolation and the failure semantics that make it safe:
an unavailable worker must raise (validator infrastructure failure), never
return a rejection that would be blamed on a miner.
"""

from __future__ import annotations

import os

import pytest

from reliquary.validator.proof_worker import (
    ProofWorkerPool,
    ProofWorkerUnavailable,
)

SUPPORT = "tests.unit.proof_worker_support"


def _pool(**kwargs):
    return ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:echo_handler",
        **kwargs,
    )


def test_handler_runs_in_a_separate_process():
    pool = _pool()
    pool.start()
    try:
        result = pool.call("cuda:0", "hello")
    finally:
        pool.close()

    assert result["payload"] == "hello"
    assert result["pid"] != os.getpid()


def test_handler_error_raises_and_leaves_the_pool_usable():
    """A refusal inside the worker is infrastructure failure, not a verdict.

    The scheduler turns an exception into a capacity abort; it turns a
    returned value into a decision about a miner. Relaying a crashed proof
    as anything other than a raise would charge a miner for our fault.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:boom_handler",
    )
    pool.start()
    try:
        with pytest.raises(ProofWorkerUnavailable, match="handler refused"):
            pool.call("cuda:0", "candidate-7")

        # The worker survives a handler-level error: the next call answers.
        with pytest.raises(ProofWorkerUnavailable):
            pool.call("cuda:0", "candidate-8")
    finally:
        pool.close()


def test_dead_worker_raises_then_the_pool_replaces_it():
    """A worker killed mid-proof (CUDA fault, OOM) must not wedge the plane.

    The in-flight window aborts — that is the scheduler contract — but the
    next window has to find a live worker, otherwise one fault costs every
    window until an operator restarts the validator.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:dispatch_handler",
    )
    pool.start()
    try:
        first = pool.call("cuda:0", "echo", "before")
        with pytest.raises(ProofWorkerUnavailable):
            pool.call("cuda:0", "crash", None)
        second = pool.call("cuda:0", "echo", "after")
    finally:
        pool.close()

    assert first["payload"] == "before"
    assert second["payload"] == "after"
    assert second["pid"] != first["pid"], "the pool must have respawned"


def test_unknown_device_raises_instead_of_spawning_one():
    """Respawn-on-demand must not become spawn-anything-on-demand.

    Device identity is qualified against the capacity manifest at startup;
    a routing bug that quietly created a worker for an unqualified device
    would prove rollouts on hardware nobody benchmarked.
    """
    pool = _pool()
    pool.start()
    try:
        with pytest.raises(ProofWorkerUnavailable, match="cuda:3"):
            pool.call("cuda:3", "hello")
    finally:
        pool.close()


def test_hung_worker_times_out_instead_of_blocking_forever():
    """``connection.recv()`` on a wedged child blocks the scheduler thread.

    The proof plane already bounds itself with MAX_PROOF_WALL_SECONDS, but
    that deadline monitor cannot fire if the device thread is parked in a
    blocking read. The pool has to own its own bound.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:dispatch_handler",
        request_timeout_seconds=0.5,
    )
    pool.start()
    try:
        with pytest.raises(ProofWorkerUnavailable, match="timed out"):
            pool.call("cuda:0", "sleep", 5.0)
        # Retired and replaced, so the plane recovers on the next window.
        assert pool.call("cuda:0", "echo", "recovered")["payload"] == "recovered"
    finally:
        pool.close()


def test_reload_swaps_what_subsequent_proofs_see():
    """Publication swaps the verify plane every CHECKPOINT_PUBLISH_INTERVAL.

    The worker holds the only copy of the weights now, so the swap has to
    reach it — a worker left on the parent revision would prove rollouts
    against weights the protocol no longer certifies.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:echo_handler",
        reload_handler=f"{SUPPORT}:reload_handler",
    )
    pool.start()
    try:
        assert pool.call("cuda:0", "before")["revision"] is None
        pool.reload("cuda:0", "/snapshots/abc", "revision-abc")
        assert pool.call("cuda:0", "after")["revision"] == "revision-abc"
    finally:
        pool.close()


def test_remote_verifier_routes_by_proxy_and_returns_the_proof_result():
    """The batcher keeps calling ``verify_commitment_proofs(commit, model, ...)``.

    In isolated mode the validator no longer holds a replica, so ``model`` is
    a proxy naming the device. The adapter has to preserve the signature and
    hand back a ProofResult whose sparse fields survived the pipe intact —
    the behavioural gates read them.
    """
    from reliquary.validator.proof_worker import (
        ProofModelProxy,
        remote_commitment_verifier,
    )

    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:proof_result_handler",
    )
    pool.start()
    try:
        verify = remote_commitment_verifier(pool)
        proxy = ProofModelProxy(device_id="cuda:0", config=None, generation_config=None)
        result = verify(
            {"tokens": [1, 2, 3], "commitments": []},
            proxy,
            "randomness-hex",
            tokenizer=None,
            seed_u_values=[0.25, 0.5, 0.75],
        )
    finally:
        pool.close()

    assert result.all_passed is True
    assert result.checked == 3
    assert result.sketch_diff_max == 17
    assert result.has_sparse_outputs is True
    assert result.p_stop == 0.125
    assert result.completion_chosen_probs == [0.9] * 8
    assert result.seed_n_stochastic == 6 and result.seed_n_match == 5
    assert result.terminal_pick_ok is True


def test_remote_verifier_refuses_a_model_that_is_not_a_proxy():
    """A real model reaching the adapter means the wiring half-applied.

    Silently proving against whatever object arrived would be worse than
    stopping: it is how a replica on stale weights gets used.
    """
    from reliquary.validator.proof_worker import remote_commitment_verifier

    verify = remote_commitment_verifier(object())
    with pytest.raises(ProofWorkerUnavailable, match="ProofModelProxy"):
        verify({"tokens": []}, object(), "randomness-hex")


def test_run_commitment_proof_binds_the_workers_own_model_and_tokenizer(monkeypatch):
    """The worker owns the weights; the caller only names a device.

    Passing anything other than the context's own model would prove the
    rollout against the wrong replica — the exact failure the device/revision
    bookkeeping exists to prevent.
    """
    import reliquary.validator.verifier as verifier_module
    from reliquary.validator.proof_worker import run_commitment_proof

    seen: dict = {}

    def fake_verify(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        seen.update(
            commit=commit, model=model, randomness=randomness,
            tokenizer=tokenizer, seed_u_values=seed_u_values,
        )
        return verifier_module.ProofResult(all_passed=True, passed=1, checked=1)

    monkeypatch.setattr(
        verifier_module, "verify_commitment_proofs", fake_verify,
    )
    context = {
        "model": "the-worker-model",
        "tokenizer": "the-worker-tokenizer",
        "revision": "rev-1",
    }

    result = run_commitment_proof(
        context, {"tokens": [1]}, "randomness-hex", [0.5],
    )

    assert result.all_passed is True
    assert seen["model"] == "the-worker-model"
    assert seen["tokenizer"] == "the-worker-tokenizer"
    assert seen["randomness"] == "randomness-hex"
    assert seen["seed_u_values"] == [0.5]


def _tiny_model():
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    return Tiny()


def test_reload_proof_context_installs_the_snapshot_and_records_the_revision(tmp_path):
    import torch
    from safetensors.torch import save_file

    from reliquary.validator.proof_worker import reload_proof_context

    model = _tiny_model()
    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    context = {"model": model, "tokenizer": None, "revision": "old-rev"}

    reload_proof_context(context, str(tmp_path), "new-rev")

    assert torch.equal(model.weight.detach(), torch.ones(2, 2))
    assert context["revision"] == "new-rev"


def test_reload_proof_context_refuses_a_snapshot_that_does_not_match(tmp_path):
    """Half-installed weights are worse than stale ones.

    A strict mismatch must raise and leave the recorded revision untouched,
    so the validator sees the worker is not certified for the new checkpoint
    instead of proving against a spliced state dict.
    """
    import torch
    from safetensors.torch import save_file

    from reliquary.validator.proof_worker import reload_proof_context

    model = _tiny_model()
    save_file({"not_the_weight": torch.ones(3, 3)}, str(tmp_path / "model.safetensors"))
    context = {"model": model, "tokenizer": None, "revision": "old-rev"}

    with pytest.raises(RuntimeError, match="state mismatch"):
        reload_proof_context(context, str(tmp_path), "new-rev")

    assert context["revision"] == "old-rev"


def test_reload_proof_context_refuses_an_empty_snapshot(tmp_path):
    from reliquary.validator.proof_worker import reload_proof_context

    context = {"model": _tiny_model(), "tokenizer": None, "revision": "old-rev"}
    with pytest.raises(RuntimeError, match="no safetensors"):
        reload_proof_context(context, str(tmp_path), "new-rev")
    assert context["revision"] == "old-rev"


def test_build_proof_context_loads_the_replica_exactly_as_the_validator_did(monkeypatch):
    """Bit-identical decisions are the whole justification for this change.

    The worker's replica must be loaded with the same dtype, the same
    attention implementation and the same pinned revision as the in-process
    replica it replaces — a drift there silently changes verdicts.
    """
    import torch

    import reliquary.shared.modeling as modeling
    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.validator.proof_worker import build_proof_context

    loaded: dict = {}

    class FakeModel:
        def __init__(self):
            self.device_arg = None
            self.evaled = False
            self._param = torch.nn.Parameter(torch.zeros(1))

        def to(self, device):
            self.device_arg = device
            return self

        def eval(self):
            self.evaled = True
            return self

        def parameters(self):
            return iter([self._param])

    def fake_load_model(source, **kwargs):
        loaded.update(source=source, kwargs=kwargs)
        return FakeModel()

    monkeypatch.setattr(modeling, "load_text_generation_model", fake_load_model)
    monkeypatch.setattr(
        modeling, "load_tokenizer", lambda source, **kw: f"tokenizer::{source}",
    )

    context = build_proof_context(
        checkpoint="ReliquaryForge/some-checkpoint",
        device="cuda:0",
        load_kwargs={"revision": "pinned-rev"},
    )

    assert loaded["source"] == "ReliquaryForge/some-checkpoint"
    assert loaded["kwargs"]["torch_dtype"] is torch.bfloat16
    assert loaded["kwargs"]["attn_implementation"] == ATTN_IMPLEMENTATION
    assert loaded["kwargs"]["revision"] == "pinned-rev"
    assert context["model"].device_arg == "cuda:0"
    assert context["model"].evaled is True
    assert context["model"]._param.requires_grad is False
    assert context["tokenizer"] == "tokenizer::ReliquaryForge/some-checkpoint"
    assert context["device"] == "cuda:0"
    # The bootstrap replica is certified by nothing: the resumed snapshot has
    # not been installed yet, and the sync must be forced to install it.
    assert context["revision"] is None


def test_pool_tracks_the_revision_each_worker_holds():
    """The swap path asks 'is this worker certified for revision X?'.

    Without that answer the validator either reloads on every window or,
    worse, marks a worker ready for weights it never received.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_context_installing",
        handler=f"{SUPPORT}:echo_handler",
        reload_handler=f"{SUPPORT}:reload_handler",
        factory_kwargs={"revision": "rev-boot"},
    )
    pool.start()
    try:
        assert pool.revision("cuda:0") == "rev-boot"
        pool.reload("cuda:0", "/snapshots/xyz", "rev-next")
        assert pool.revision("cuda:0") == "rev-next"
    finally:
        pool.close()


def test_a_respawned_worker_reports_the_revision_it_was_rebuilt_with():
    """A worker that died after a swap must come back on the CURRENT weights.

    Respawning from the boot revision would put the plane back on stale
    weights without anyone noticing.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_context_installing_dispatch",
        handler=f"{SUPPORT}:dispatch_handler",
        reload_handler=f"{SUPPORT}:reload_handler",
        factory_kwargs={"revision": "rev-boot"},
    )
    pool.start()
    try:
        pool.reload("cuda:0", "/snapshots/xyz", "rev-next")
        with pytest.raises(ProofWorkerUnavailable):
            pool.call("cuda:0", "crash", None)
        assert pool.revision("cuda:0") is None, (
            "a replacement worker has not been given the current weights yet"
        )
    finally:
        pool.close()


def test_each_worker_is_told_which_device_it_owns():
    """One factory, several devices: the worker cannot guess its own.

    Without this every worker would build its replica on the same device —
    the multi-GPU path would silently collapse onto one card.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0", "cuda:1"),
        context_factory=f"{SUPPORT}:build_device_context",
        handler=f"{SUPPORT}:device_handler",
        factory_kwargs={"tag": "shared"},
    )
    pool.start()
    try:
        first = pool.call("cuda:0", "x")
        second = pool.call("cuda:1", "x")
    finally:
        pool.close()

    assert first["device"] == "cuda:0"
    assert second["device"] == "cuda:1"
    assert first["tag"] == second["tag"] == "shared"


def test_isolated_plane_dotted_paths_resolve_to_the_real_worker_functions():
    """The child resolves these by string; a typo only shows up at spawn.

    Pinning them here turns 'the validator refuses to boot in production'
    into 'a unit test fails'.
    """
    from reliquary.validator import proof_worker as pw

    assert pw._resolve(pw.PROOF_CONTEXT_FACTORY) is pw.build_proof_context
    assert pw._resolve(pw.PROOF_HANDLER) is pw.run_commitment_proof
    assert pw._resolve(pw.PROOF_RELOAD_HANDLER) is pw.reload_proof_context


def test_isolated_plane_hands_the_validator_one_proxy_per_device():
    """The validator keeps a per-device entry, but holds no weights.

    Each proxy has to carry the EOS metadata the termination gates read,
    otherwise every rollout would look unterminated.
    """
    from types import SimpleNamespace

    from reliquary.validator.proof_worker import (
        ProofModelProxy,
        build_isolated_proof_plane,
    )

    reference = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=151643),
        generation_config=SimpleNamespace(eos_token_id=[151643, 151645]),
    )

    pool, proxies = build_isolated_proof_plane(
        devices=("cuda:0", "cuda:1"),
        checkpoint="ReliquaryForge/whatever",
        load_kwargs={"revision": "pinned"},
        reference_model=reference,
    )

    assert pool.devices == ("cuda:0", "cuda:1")
    assert set(proxies) == {"cuda:0", "cuda:1"}
    assert all(isinstance(p, ProofModelProxy) for p in proxies.values())
    assert proxies["cuda:1"].device_id == "cuda:1"

    from reliquary.shared.modeling import resolve_eos_token_ids

    assert resolve_eos_token_ids(proxies["cuda:0"], None) == {151643, 151645}


def test_isolation_requires_a_snapshot_source_for_the_swap():
    """Isolation without the detached-trainer intake breaks publication.

    The worker can only take new weights from a staged snapshot directory.
    In-process training hands the swap an in-memory state dict instead, so
    the pairing has to fail at boot, not 16 windows later at the first
    publication with the plane already live.
    """
    from reliquary.validator.proof_worker import assert_isolation_supported

    assert_isolation_supported(isolation=False, detached_trainer=False)
    assert_isolation_supported(isolation=True, detached_trainer=True)

    with pytest.raises(RuntimeError, match="RELIQUARY_DETACHED_TRAINER"):
        assert_isolation_supported(isolation=True, detached_trainer=False)


def test_real_worker_loads_a_model_and_proves_across_the_process_boundary(tmp_path):
    """End-to-end on the production dotted paths, with a tiny CPU model.

    Everything above this test uses cheap doubles for the worker body. This
    one drives the real chain — build_proof_context loading a checkpoint,
    run_commitment_proof calling verify_commitment_proofs, a populated
    ProofResult coming back over the pipe — so a break in the glue between
    them cannot reach production disguised as a green suite.
    """
    import os

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from reliquary.validator.proof_worker import (
        ProofModelProxy,
        ProofWorkerPool,
        PROOF_CONTEXT_FACTORY,
        PROOF_HANDLER,
        PROOF_RELOAD_HANDLER,
        remote_commitment_verifier,
    )

    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2, tie_word_embeddings=True,
    )
    checkpoint = tmp_path / "tiny"
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    model.save_pretrained(str(checkpoint))
    config.save_pretrained(str(checkpoint))

    # The worker re-imports constants, so the child picks this up on spawn.
    os.environ["GRAIL_ATTN_IMPL"] = "eager"
    pool = ProofWorkerPool(
        devices=("cpu",),
        context_factory=PROOF_CONTEXT_FACTORY,
        handler=PROOF_HANDLER,
        reload_handler=PROOF_RELOAD_HANDLER,
        factory_kwargs={"checkpoint": str(checkpoint), "load_kwargs": {}},
        request_timeout_seconds=300.0,
    )
    pool.start()
    try:
        tokens = [1, 5, 9, 13, 17, 21, 25, 29]
        commit = {
            "tokens": tokens,
            "commitments": [{"sketch": 0} for _ in tokens],
            "rollout": {"prompt_length": 4, "completion_length": 4},
        }
        verify = remote_commitment_verifier(pool)
        result = verify(
            commit,
            ProofModelProxy(device_id="cpu"),
            "a" * 64,
            seed_u_values=[0.1, 0.2, 0.3, 0.4],
        )
    finally:
        pool.close()
        os.environ.pop("GRAIL_ATTN_IMPL", None)

    # A real forward ran in the worker: the validator computed its own
    # sketches (non-zero diff against the planted zeros) and the sparse
    # outputs the behavioural gates read came back whole.
    # (all_passed is not asserted here: the v7 sketch tolerance is 5000+,
    # so planted zeros still pass it — a known property of the gate, not of
    # this transport.)
    assert result.checked == len(tokens)
    assert result.sketch_diff_max > 0
    assert result.has_sparse_outputs is True
    assert len(result.completion_chosen_probs) > 0
    assert all(0.0 <= p <= 1.0 for p in result.completion_chosen_probs)
    assert result.p_stop is not None


def test_pool_learns_the_revision_from_the_worker_not_from_the_caller():
    """The parent must never assert weights on the worker's behalf.

    Production bootstraps the worker from the base model while the caller
    holds the resumed checkpoint's sha. Seeding the pool with the caller's
    sha made `_synchronize_proof_workers` find a match, skip the reload and
    mark the device ready — so every GRAIL proof ran against base weights and
    every honest miner was rejected. The revision has to travel back from the
    worker, which is the only party that knows what it loaded.
    """
    installed = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_context_installing",
        handler=f"{SUPPORT}:echo_handler",
        factory_kwargs={"revision": "installed-by-the-worker"},
    )
    installed.start()
    try:
        assert installed.revision("cuda:0") == "installed-by-the-worker"
    finally:
        installed.close()

    nothing = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_context_installing",
        handler=f"{SUPPORT}:echo_handler",
    )
    nothing.start()
    try:
        assert nothing.revision("cuda:0") is None, (
            "a worker that installed no checkpoint must not be marked certified"
        )
    finally:
        nothing.close()


def test_concurrent_calls_on_one_device_do_not_interleave():
    """Device-less proof paths all route to the first device.

    `_default_proof_proxy` sends the forensic sample and the legacy
    non-auction admission to the same worker, and those run on concurrent
    HTTP threads. Two unserialized send/recv pairs on one pipe interleave
    pickle frames — at best the child dies, at worst one miner receives
    another miner's ProofResult.
    """
    import threading

    pool = _pool()
    pool.start()
    results, errors = {}, []

    def hammer(tag):
        try:
            for _ in range(15):
                got = pool.call("cuda:0", tag)
                if got["payload"] != tag:
                    errors.append(f"{tag} received {got['payload']!r}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tag}: {type(exc).__name__}: {exc}")
        results[tag] = True

    threads = [threading.Thread(target=hammer, args=(f"caller-{i}",)) for i in range(4)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
    finally:
        pool.close()

    assert not errors, errors
    assert len(results) == 4


def test_worker_errors_keep_their_original_class_name():
    """`batcher.py` keys its CUDA-OOM recovery on the exception CLASS name.

    Collapsing everything into ProofWorkerUnavailable silently disables the
    gc.collect() / empty_cache() branch, and the fragmented allocator state
    now lives in a child the parent cannot clean up either.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:boom_handler",
    )
    pool.start()
    try:
        with pytest.raises(ProofWorkerUnavailable) as excinfo:
            pool.call("cuda:0", "candidate-7")
    finally:
        pool.close()

    assert excinfo.value.remote_error_type == "ValueError"
    assert "handler refused" in str(excinfo.value)


def test_proof_error_type_prefers_the_childs_class_over_the_wrapper():
    """What the forensic sampler records and branches on.

    An OOM raised inside the worker must still be recorded as
    "OutOfMemoryError", not as the transport's wrapper type, or the
    allocator-cleanup branch never runs.
    """
    from reliquary.validator.proof_worker import (
        ProofWorkerUnavailable,
        proof_error_type,
    )

    wrapped = ProofWorkerUnavailable(
        "proof worker cuda:0 failed", remote_error_type="OutOfMemoryError",
    )
    assert proof_error_type(wrapped) == "OutOfMemoryError"
    assert proof_error_type(ProofWorkerUnavailable("no remote type")) == (
        "ProofWorkerUnavailable"
    )
    assert proof_error_type(ValueError("local")) == "ValueError"


def test_reload_gets_its_own_longer_deadline():
    """An 8 GB safetensors load is slower than a proof by design.

    Running it under the proof request timeout kills a healthy worker
    mid-reload, and `_swap_staged_checkpoint` then rmtree's the staged
    directory while verify_model has already advanced — recoverable only by
    a restart and a burned window.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_counter_context",
        handler=f"{SUPPORT}:dispatch_handler",
        reload_handler=f"{SUPPORT}:slow_reload_handler",
        request_timeout_seconds=0.5,
        reload_timeout_seconds=20.0,
    )
    pool.start()
    try:
        # Slower than the proof timeout, well inside the reload one.
        pool.reload("cuda:0", "2.0", "rev-slow")
        assert pool.revision("cuda:0") == "rev-slow"
        # The proof timeout still applies to proofs.
        with pytest.raises(ProofWorkerUnavailable, match="timed out"):
            pool.call("cuda:0", "sleep", 5.0)
    finally:
        pool.close()


def test_a_worker_that_dies_while_building_raises_the_documented_error():
    """A child that OOMs or hits a bad snapshot dies during its factory.

    The handshake `recv()` then raises a raw EOFError — not the
    ProofWorkerUnavailable every caller and the class docstring promise —
    and the half-created worker is never retired.
    """
    pool = ProofWorkerPool(
        devices=("cuda:0",),
        context_factory=f"{SUPPORT}:build_context_that_dies",
        handler=f"{SUPPORT}:echo_handler",
    )
    with pytest.raises(ProofWorkerUnavailable, match="cuda:0"):
        pool.start()
    pool.close()


def test_reload_falls_back_to_the_hub_when_the_snapshot_is_gone(tmp_path, monkeypatch):
    """`mark_installed` rmtree's the staged directory after every swap.

    A worker that dies later then respawns with no weights and no local
    source. Without a durable fallback the plane stays down until an
    operator restarts the validator — a recoverable crash turned into an
    outage. HF is where the checkpoint durably lives; it is the same place
    miners pull it from.
    """
    import torch

    import reliquary.shared.modeling as modeling
    from reliquary.validator.proof_worker import reload_proof_context

    class Tiny(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2, 2), float(value)))

        def eval(self):
            return self

    monkeypatch.setattr(
        modeling, "load_text_generation_model",
        lambda source, **kw: Tiny(7.0) if kw.get("revision") == "rev-hub" else Tiny(0.0),
    )
    context = {"model": Tiny(0.0), "tokenizer": None, "device": "cpu",
               "revision": "old-rev"}

    reload_proof_context(
        context, str(tmp_path / "deleted-by-mark-installed"), "rev-hub",
        repo_id="ReliquaryForge/qwen3-4b-base-dapo-v4",
    )

    assert torch.equal(context["model"].weight.detach(), torch.full((2, 2), 7.0))
    assert context["revision"] == "rev-hub"


def test_reload_without_snapshot_or_repo_refuses(tmp_path):
    from reliquary.validator.proof_worker import reload_proof_context

    context = {"model": None, "tokenizer": None, "revision": "old-rev"}
    with pytest.raises(RuntimeError, match="no source"):
        reload_proof_context(context, None, "rev-x")
    assert context["revision"] == "old-rev"
