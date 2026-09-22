"""Run the GRAIL proof plane in its own process, one per proof device.

The proof worker issues hundreds of CUDA ops per forward, releasing and
re-acquiring the GIL each time. Sharing an interpreter with the validator's
event loop — which holds the GIL in long blocks to parse submission bodies and
spawn admission workers — convoys the proof thread off the lock: the same
forward measured 28.7 ms alone and 29.6 s against one CPU-bound python thread
(2026-08-23, H100). Giving the proof plane its own interpreter removes the
contention without changing a single kernel: same device, same weights, same
``batch=1``, so every accept/reject decision is bit-identical.

This module owns only process lifecycle and the request/response contract. The
model, the tokenizer and the proof itself are resolved by dotted path inside
the child, so nothing heavy is pickled across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import importlib
import logging
import multiprocessing
import threading
from typing import Any, Callable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ProofModelProxy",
    "ProofWorkerPool",
    "ProofWorkerUnavailable",
    "assert_isolation_supported",
    "assert_proof_slots_supported",
    "mps_control_pipe_path",
    "assert_host_memory_for_cpu_replicas",
    "validator_replica_device",
    "build_isolated_proof_plane",
    "build_proof_context",
    "reload_proof_context",
    "proof_error_type",
    "remote_commitment_verifier",
    "batch_padding_for",
    "run_commitment_proof",
    "shadow_traversal",
]


class ProofWorkerUnavailable(RuntimeError):
    """The worker owning a device cannot answer.

    Raised — never returned as a rejection. A dead or unreachable worker is
    validator infrastructure failure: the scheduler must abort the proof plane
    rather than blame the miner whose candidate happened to be in flight.

    ``remote_error_type`` carries the class name the child actually raised.
    Callers discriminate on it — the forensic sampler keys its CUDA-OOM
    recovery (``gc.collect`` / ``empty_cache``) on that name, and collapsing
    every child failure into one opaque type would silently disable it.
    """

    def __init__(self, message: str, *, remote_error_type: str | None = None) -> None:
        super().__init__(message)
        self.remote_error_type = remote_error_type


@dataclass(frozen=True)
class ProofModelProxy:
    """Stands in for the replica the validator no longer holds.

    The proof-dependent gates around the GRAIL call (termination, EOS
    padding, cap truncation) only ever read ``config`` / ``generation_config``
    to resolve the EOS set, so they keep working against metadata while the
    weights live in the worker.
    """

    device_id: str
    config: Any = None
    generation_config: Any = None


def remote_commitment_verifier(
    pool: "ProofWorkerPool",
) -> Callable[..., Any]:
    """Adapt a pool to ``verify_commitment_proofs``'s call signature.

    Injected into the batcher as ``verify_commitment_proofs_fn`` so the
    per-rollout proof loop is untouched.
    """

    def _device_of(model: Any) -> str:
        device_id = getattr(model, "device_id", None)
        if not isinstance(model, ProofModelProxy) or not device_id:
            raise ProofWorkerUnavailable(
                "isolated proof plane requires a ProofModelProxy, got "
                f"{type(model).__name__}"
            )
        return device_id

    def verify(
        commit: Any,
        model: Any,
        window_randomness: str,
        *,
        tokenizer: Any = None,
        seed_u_values: Any = None,
    ) -> Any:
        device_id = _device_of(model)
        return pool.call(device_id, commit, window_randomness, seed_u_values)

    def warm(inputs: Sequence[Any], model: Any, window_randomness: str) -> int:
        """Pay one pass for a slice of the request the per-rollout loop is about to prove."""
        device_id = _device_of(model)
        commits = [commit for commit, _ in inputs]
        seeds = [seed for _, seed in inputs]
        return pool.warm(device_id, commits, window_randomness, seeds)

    verify.warm = warm
    return verify


def proof_error_type(exc: BaseException) -> str:
    """Class name to record and branch on for a failed proof.

    With an isolated plane the real failure happened in the child, so the
    transport wrapper's own type says nothing useful — callers that key
    recovery on the name (CUDA OOM cleanup) need what the child raised.
    """
    return getattr(exc, "remote_error_type", None) or type(exc).__name__


def _resolve(dotted: str) -> Callable[..., Any]:
    """Resolve ``package.module:attribute`` inside whichever process asks."""
    module_name, _, attribute = dotted.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"expected 'module:attribute', got {dotted!r}"
        )
    return getattr(importlib.import_module(module_name), attribute)


def _worker_main(
    connection: Any,
    *,
    context_factory: str,
    handler: str,
    reload_handler: str | None,
    warm_handler: str | None,
    factory_kwargs: Mapping[str, Any],
    device: str,
) -> None:
    """Child entrypoint: build the heavy context once, then serve requests."""
    try:
        context = _resolve(context_factory)(device=device, **dict(factory_kwargs))
        handler_fn = _resolve(handler)
        reload_fn = _resolve(reload_handler) if reload_handler else None
        warm_fn = _resolve(warm_handler) if warm_handler else None
    except BaseException as exc:  # noqa: BLE001 - reported, then the child exits
        try:
            connection.send(("start_failed", (type(exc).__name__, str(exc))))
        except Exception:
            pass
        return
    # The revision travels back from the worker: it is the only party that
    # knows which weights it loaded. A parent-asserted label silently skips
    # the reload and proves against the wrong checkpoint.
    connection.send(("ready", context.get("revision")))

    while True:
        request = connection.recv()
        operation, args, kwargs = request
        if operation == "shutdown":
            return
        try:
            if operation == "describe":
                payload = describe_proof_context(context)
            elif operation == "reload":
                if reload_fn is None:
                    raise RuntimeError("worker has no reload handler")
                payload = reload_fn(context, *args, **kwargs)
            elif operation == "warm":
                if warm_fn is None:
                    raise RuntimeError("worker has no warm handler")
                payload = warm_fn(context, *args, **kwargs)
            else:
                payload = handler_fn(context, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - relayed, worker stays up
            connection.send(("error", (type(exc).__name__, str(exc))))
        else:
            connection.send(("ok", payload))


@dataclass
class _Worker:
    device_id: str
    process: Any = None
    connection: Any = None
    # One request at a time per pipe. The scheduler gives one thread per
    # device, but every device-less proof path (forensic sample, legacy
    # non-auction admission) routes to the first device from concurrent
    # HTTP threads; unserialized send/recv pairs interleave pickle frames.
    lock: Any = field(default_factory=threading.Lock)


class ProofWorkerPool:
    """One dedicated process per proof slot.

    Each proof interpreter has no competing python thread. Several slots may
    name the same card (``cuda:0#0``, ``cuda:0#1``, ...): without an MPS server
    their CUDA contexts time-slice and a second slot buys almost nothing —
    which is what the original x1.04 measurement here recorded — but with one
    they overlap, and four slots measure ~x2 (see PROOF_SLOTS_PER_DEVICE).
    Every slot runs the same batch=1 path, so no verdict can shift.
    """

    def __init__(
        self,
        *,
        devices: Sequence[str],
        context_factory: str,
        handler: str,
        reload_handler: str | None = None,
        warm_handler: str | None = None,
        factory_kwargs: Mapping[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
        reload_timeout_seconds: float | None = None,
        warm_timeout_seconds: float | None = None,
        start_timeout_seconds: float = 900.0,
    ) -> None:
        if not devices:
            raise ValueError("ProofWorkerPool requires at least one device")
        if len(set(devices)) != len(devices):
            raise ValueError("proof worker devices must be distinct")
        self._devices = tuple(devices)
        self._context_factory = context_factory
        self._handler = handler
        self._reload_handler = reload_handler
        self._warm_handler = warm_handler
        self._factory_kwargs = dict(factory_kwargs or {})
        self._request_timeout_seconds = (
            None if request_timeout_seconds is None
            else float(request_timeout_seconds)
        )
        self._reload_timeout_seconds = (
            None if reload_timeout_seconds is None
            else float(reload_timeout_seconds)
        )
        # A warm pass drives the model over a whole slice at once, which on a streamed replica is
        # a full traversal: it is closer to a reload than to a single proof, and the request
        # timeout would retire a worker that is doing exactly what it was asked.
        self._warm_timeout_seconds = (
            self._reload_timeout_seconds if warm_timeout_seconds is None
            else float(warm_timeout_seconds)
        )
        self._start_timeout_seconds = float(start_timeout_seconds)
        self._revisions: dict[str, str | None] = {}
        self._workers: dict[str, _Worker] = {}
        # Guards the maps themselves. Held only for dict access — never
        # across a spawn, which can block for start_timeout_seconds while a
        # replica loads. With one slot that was invisible; with N it would
        # park every other slot's dispatch thread behind one respawn, past
        # MAX_PROOF_WALL_SECONDS, and fault the plane.
        self._spawn_lock = threading.Lock()
        self._closed = False
        # One lock per slot, so two threads racing a respawn still do not
        # each start a process for the SAME slot.
        self._device_spawn_locks: dict[str, threading.Lock] = {}
        # ...but bound how many load a replica AT ONCE across slots: _spawn
        # materialises the whole ~8 GB model on the host before moving it to
        # the card, and this process already carries a ~24 GB floor. Two, not
        # one: at one, a single slow spawn parks every other slot again, which
        # is the regression the per-slot locks exist to fix.
        self._replica_load_slots = threading.Semaphore(2)
        self._context = multiprocessing.get_context("spawn")

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    def start(self) -> None:
        for device_id in self._devices:
            # Same lock _request takes. Nothing calls into the pool during
            # startup today, but leaving the spawn invariant resting on that
            # call order lets a future caller spawn a slot twice and drop one
            # child, still holding a full replica, out of the map.
            with self._device_spawn_lock(device_id):
                self._worker_for(device_id)

    def _worker_for(self, device_id: str) -> _Worker:
        """Return the live worker, replacing one that died since last use."""
        if device_id not in self._devices:
            raise ProofWorkerUnavailable(
                f"device {device_id!r} is not a configured proof device "
                f"({', '.join(self._devices)})"
            )
        worker = self._workers.get(device_id)
        if worker is None:
            if self._closed:
                # close() already took its snapshot; a replica started now
                # would land on a card being torn down and never be retired.
                raise ProofWorkerUnavailable(
                    f"proof worker pool is closed, refusing to spawn "
                    f"{device_id}"
                )
            worker = self._spawn(device_id)
            if self._closed:
                # close() holds no lock across a spawn and a spawn can block
                # for start_timeout_seconds, so its snapshot cannot contain
                # this child. Publishing it now would leak a replica and an
                # open pipe past the shutdown meant to reap them.
                self._kill(worker)
                raise ProofWorkerUnavailable(
                    f"proof worker pool closed while {device_id} was starting"
                )
            self._workers[device_id] = worker
        return worker

    def _retire(self, device_id: str, worker: "_Worker | None" = None) -> None:
        """Kill the worker on ``device_id``.

        ``worker`` names WHICH one: a thread that failed an exchange may reach
        here long after another thread respawned the slot, and retiring by id
        alone would kill the healthy replacement out from under it.
        """
        if worker is not None and self._workers.get(device_id) is not worker:
            return
        self._revisions[device_id] = None
        worker = self._workers.pop(device_id, None)
        if worker is None:
            return
        self._kill(worker)

    @staticmethod
    def _kill(worker: "_Worker") -> None:
        """Reap one child, whether or not it ever reached ``_workers``."""
        try:
            worker.connection.close()
        except OSError:
            pass
        if worker.process.is_alive():
            worker.process.kill()
        worker.process.join(timeout=5.0)

    def _spawn(self, device_id: str) -> _Worker:
        with self._replica_load_slots:
            return self._spawn_locked(device_id)

    def _spawn_locked(self, device_id: str) -> _Worker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_conn,),
            kwargs={
                "context_factory": self._context_factory,
                "handler": self._handler,
                "reload_handler": self._reload_handler,
                "warm_handler": self._warm_handler,
                "factory_kwargs": self._factory_kwargs,
                "device": device_id,
            },
            name=f"reliquary-proof-{device_id}",
            daemon=True,
        )
        process.start()
        child_conn.close()

        def _abandon(reason: str, remote_type: str | None = None):
            try:
                parent_conn.close()
            except OSError:
                pass
            if process.is_alive():
                process.kill()
            process.join(timeout=5.0)
            self._revisions[device_id] = None
            return ProofWorkerUnavailable(reason, remote_error_type=remote_type)

        # A replica load is slow; a hung one must not park the caller forever.
        if not parent_conn.poll(self._start_timeout_seconds):
            raise _abandon(
                f"proof worker {device_id} did not come up within "
                f"{self._start_timeout_seconds:g}s"
            )
        try:
            status, payload = parent_conn.recv()
        except (EOFError, OSError) as exc:
            raise _abandon(
                f"proof worker {device_id} died before it was ready: {exc!r}",
                type(exc).__name__,
            ) from exc
        if status != "ready":
            raise _abandon(
                f"proof worker {device_id} failed to start: "
                f"{payload[0]}: {payload[1]}",
                payload[0],
            )
        self._revisions[device_id] = payload
        return _Worker(device_id=device_id, process=process, connection=parent_conn)

    def _device_spawn_lock(self, device_id: str) -> Any:
        with self._spawn_lock:
            lock = self._device_spawn_locks.get(device_id)
            if lock is None:
                lock = self._device_spawn_locks[device_id] = threading.Lock()
            return lock

    def _request(self, device_id: str, operation: str, args, kwargs) -> Any:
        with self._device_spawn_lock(device_id):
            worker = self._worker_for(device_id)
        with worker.lock:
            return self._exchange(worker, device_id, operation, args, kwargs)

    def _exchange(self, worker, device_id: str, operation: str, args, kwargs) -> Any:
        try:
            worker.connection.send((operation, args, kwargs))
            timeout = {
                "reload": self._reload_timeout_seconds,
                "warm": self._warm_timeout_seconds,
            }.get(operation, self._request_timeout_seconds)
            if timeout is not None and not worker.connection.poll(timeout):
                self._retire(device_id, worker=worker)
                raise ProofWorkerUnavailable(
                    f"proof worker {device_id} timed out after {timeout:g}s"
                )
            status, payload = worker.connection.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            # The child died mid-request. Retire it so the NEXT window finds a
            # live worker, and raise: the scheduler must abort this plane
            # rather than turn our fault into a miner's rejection.
            self._retire(device_id, worker=worker)
            raise ProofWorkerUnavailable(
                f"proof worker {device_id} died mid-request: {exc!r}",
                remote_error_type=type(exc).__name__,
            ) from exc
        if status == "ok":
            return payload
        raise ProofWorkerUnavailable(
            f"proof worker {device_id} failed: {payload[0]}: {payload[1]}",
            remote_error_type=payload[0],
        )

    def call(self, device_id: str, *args: Any, **kwargs: Any) -> Any:
        return self._request(device_id, "call", args, kwargs)

    def reload(
        self,
        device_id: str,
        snapshot_dir: str | None,
        checkpoint_revision: str,
        repo_id: str | None = None,
    ) -> None:
        """Install new weights in the worker (checkpoint publication)."""
        self._revisions[device_id] = None
        self._request(
            device_id, "reload",
            (snapshot_dir, checkpoint_revision, repo_id), {},
        )
        self._revisions[device_id] = checkpoint_revision

    def warm(
        self,
        device_id: str,
        commits: Sequence[Any],
        window_randomness: str,
        seed_u_values: Sequence[Any] | None = None,
    ) -> int:
        """Drive the model once for a slice, so the proofs that follow do not each drive it.

        Returns how many rollouts were warmed. A pool with no warm handler warms none, and the
        proofs run exactly as they did before.
        """
        if not self._warm_handler or not commits:
            return 0
        seeds = list(seed_u_values or [None] * len(commits))
        return int(
            self._request(
                device_id, "warm", (list(commits), window_randomness, seeds), {},
            )
        )

    def revision(self, device_id: str) -> str | None:
        """Revision this worker is certified for, or None when unknown."""
        return self._revisions.get(device_id)

    def describe(self, device_id: str) -> dict[str, Any]:
        """Read identity from the process that owns the actual weights."""
        return self._request(device_id, "describe", (), {})

    def is_alive(self, device_id: str) -> bool:
        worker = self._workers.get(device_id)
        return bool(not self._closed and worker is not None and worker.process.is_alive())

    def close(self, force: bool = False) -> None:
        """Retire every worker.

        ``force`` skips the polite shutdown frame: when a device thread may
        still be mid-request, writing into its pipe corrupts the exchange it
        is reading. Kill the child instead and let the caller fail loudly.
        """
        # Snapshot first: with per-slot spawn locks another dispatch thread can
        # insert or pop while this runs, and iterating the live map would abort
        # here and leave the remaining slots alive.
        with self._spawn_lock:
            self._closed = True
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            if not force:
                try:
                    worker.connection.send(("shutdown", (), {}))
                except (OSError, BrokenPipeError, ValueError):
                    pass
            else:
                if worker.process.is_alive():
                    worker.process.kill()
            worker.process.join(timeout=10.0)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(timeout=5.0)
            try:
                worker.connection.close()
            except OSError:
                pass
        self._workers.clear()


# ─────────────────────────  production worker body  ─────────────────────────
# Resolved by dotted path inside the child. Everything below runs in the proof
# process, never in the validator's interpreter.


def batch_padding_for(replica: str | None) -> bool:
    """Whether this slot's passes may mix lengths.

    A streamed slot has to: a traversal is its whole cost, and rollouts that terminate on their
    own almost never share a token count, so an unpadded pass carries one rollout. A resident slot
    does not: its forward is cheap either way, and not padding is how it keeps returning exactly
    what one-at-a-time verification returned before any of this existed.
    """
    from reliquary.constants import PROOF_BATCH_PADDING

    if PROOF_BATCH_PADDING != "auto":
        return PROOF_BATCH_PADDING == "on"
    return replica == STREAMED


def _warm_key(commit: Any) -> tuple:
    """What identifies a rollout among the ones a warm pass covered."""
    return tuple(commit["tokens"])


def warm_commitment_batch(
    context: MutableMapping[str, Any],
    commits: list[dict],
    window_randomness: str,
    seed_u_values: Any = None,
) -> int:
    """Pay one traversal of the model for a whole batch, before its proofs arrive one by one.

    The server proves item by item, and rightly so: that loop carries the receipts, the deadlines
    and the stop at the first failure. Warming leaves it alone and removes the only part that does
    not belong to an item — the pass over the model, which a streamed replica pays in full every
    time. Each proof then finds its rows already computed.
    """
    from reliquary.validator import verifier as verifier_module

    context["_warm"] = {}
    if not commits:
        return 0
    rows = verifier_module.forward_rows_for_batch(
        commits, context["model"], pad=batch_padding_for(context.get("replica")),
    )
    context["_warm"] = {_warm_key(commit): rows[index] for index, commit in enumerate(commits)}
    return len(context["_warm"])


def shadow_traversal(context: MutableMapping[str, Any], commit: Any) -> bool:
    """Run the streamed traversal beside the resident forward and report what differs.

    That the two agree to the bit is the oracle this path rests on, and it exists only while the
    model still fits on a card — which is the window before a model arrives that does not. Running
    it on a sampled share of live traffic is how it gets exercised against the real checkpoint and
    real rollouts while it can still be checked at all.

    It never touches a verdict. A miner is accepted or rejected on the same forward as before; a
    disagreement here is reported and counted, and it is the operator who decides what it means.
    """
    import random

    import torch

    from reliquary.constants import LAYER_INDEX, PROOF_SHADOW_FRACTION
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.streaming_forward import StreamedReplica

    model = context.get("model")
    if (
        PROOF_SHADOW_FRACTION <= 0.0
        or context.get("replica") != RESIDENT
        or model is None
        or isinstance(commit, list)
        or random.random() >= PROOF_SHADOW_FRACTION
    ):
        return False
    report = context.setdefault("shadow", {"checked": 0, "mismatched": 0, "failed": 0})
    try:
        shadow = context.get("_shadow_replica")
        if shadow is None:
            shadow = StreamedReplica.from_model(model, device=next(model.parameters()).device)
            context["_shadow_replica"] = shadow
        tokens = torch.tensor([list(commit["tokens"])], device=next(model.parameters()).device)
        with torch.no_grad():
            resident, _ = forward_single_layer(model, tokens, None, LAYER_INDEX)
            streamed, _ = forward_single_layer(shadow, tokens, None, LAYER_INDEX)
        report["checked"] += 1
        if not torch.equal(streamed, resident):
            difference = float((streamed.float() - resident.float()).abs().max())
            report["mismatched"] += 1
            report["max_difference"] = max(report.get("max_difference", 0.0), difference)
            logger.error(
                "shadow traversal disagrees with the resident forward on %d tokens "
                "(max |difference| %.3e); no verdict was changed",
                tokens.shape[1], difference,
            )
            return False
        return True
    except Exception:  # noqa: BLE001 - a shadow must never cost a proof
        report["failed"] += 1
        logger.exception("shadow traversal failed; the proof itself is unaffected")
        return False


def run_commitment_proof(
    context: MutableMapping[str, Any],
    commit: Any,
    window_randomness: str,
    seed_u_values: Any = None,
) -> Any:
    """Run a GRAIL proof against the weights this worker owns.

    A list of commits is proved with one pass of the model per group and comes back as a list of
    results in the same order. That matters for a streamed replica, where a pass costs a full
    traversal and proving rollouts one at a time would pay it over and over.
    """
    from reliquary.validator import verifier as verifier_module

    shadow_traversal(context, commit)
    warm = context.get("_warm")
    if warm and not isinstance(commit, list):
        # Spent as it is consumed: a second attempt on the same rollout runs its own pass rather
        # than reusing rows that belonged to the first.
        rows = warm.pop(_warm_key(commit), None)
        if rows is not None:
            return verifier_module.verify_commitment_proofs(
                commit,
                context["model"],
                window_randomness,
                tokenizer=context["tokenizer"],
                seed_u_values=seed_u_values,
                forward=rows,
            )
    if isinstance(commit, list):
        seeds = seed_u_values if isinstance(seed_u_values, list) else [None] * len(commit)
        return verifier_module.verify_commitment_proofs_batch(
            commit,
            context["model"],
            window_randomness,
            tokenizer=context["tokenizer"],
            seed_u_values=seeds,
            pad=batch_padding_for(context.get("replica")),
        )
    return verifier_module.verify_commitment_proofs(
        commit,
        context["model"],
        window_randomness,
        tokenizer=context["tokenizer"],
        seed_u_values=seed_u_values,
    )


def describe_proof_context(context: MutableMapping[str, Any]) -> dict[str, Any]:
    """No tensors: metadata is measured inside the GPU-owning interpreter."""
    import torch
    from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint
    from reliquary.validator.proof_capacity import (
        physical_proof_device, resolve_cuda_proof_devices,
    )

    model = context["model"]
    if model is None:
        raise ProofWorkerUnavailable("proof worker has no loaded model")
    device = context["device"]
    identity, = resolve_cuda_proof_devices(
        [physical_proof_device(device)], cuda=torch.cuda,
    )
    return {
        "device_id": device,
        "replica": context.get("replica", RESIDENT),
        "physical_device": identity.device_id,
        "hardware_class": identity.hardware_class,
        "device_uuid": identity.device_uuid,
        "revision": context.get("revision"),
        "shadow": dict(context.get("shadow") or {}),
        "runtime": collect_runtime_fingerprint(generation_model=model, proof_model=model),
        "config": model.config.to_dict(),
        "generation_config": model.generation_config.to_dict(),
    }


def _install_streamed(
    context: MutableMapping[str, Any],
    snapshot_dir: str | None,
    checkpoint_revision: str,
    repo_id: str | None,
) -> None:
    """Point a streamed replica at the checkpoint it must now verify against.

    The staged directory is deleted the moment the swap completes, so the weights are first
    written into this validator's own store, which is where the traversal then reads them from.
    Building it costs one pass over the checkpoint and is paid once per revision, by whichever
    slot rotates first; the slots that follow find it already there.
    """
    from pathlib import Path

    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.fused_layers import fused_store_root, install_fused_store
    from reliquary.shared.streaming_forward import StreamedReplica
    from reliquary.validator.proof_capacity import physical_proof_device

    if not (snapshot_dir and Path(snapshot_dir).is_dir()):
        raise RuntimeError(
            "a streamed replica reloads from a staged checkpoint directory, and "
            f"{snapshot_dir!r} is not one; the durable repo is not a substitute because the "
            "layers are read from disk on every traversal"
        )
    store = install_fused_store(snapshot_dir, fused_store_root(), checkpoint_revision)
    previous = context.get("model")
    context["model"] = None
    context["revision"] = None
    if hasattr(previous, "close"):
        previous.close()
    context["model"] = StreamedReplica.from_checkpoint(
        store,
        device=physical_proof_device(context.get("device")),
        dtype=torch.bfloat16,
        prefetch=True,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    context["revision"] = checkpoint_revision


def reload_proof_context(
    context: MutableMapping[str, Any],
    snapshot_dir: str | None,
    checkpoint_revision: str,
    repo_id: str | None = None,
) -> None:
    """Install a published checkpoint into the worker's replica.

    Mirrors ``_ValidatorService._refresh_verify_model_from_dir``: assemble the
    whole state dict on CPU first, allow exactly the model's declared tied
    keys, and record the revision only once the load succeeded. A partial
    install must surface as a raise, never as a worker that keeps proving.

    ``repo_id`` is the durable fallback. ``CheckpointIntake.mark_installed``
    rmtree's the staged directory after every swap, so a worker respawned
    later has no local source; without this the plane would stay down until an
    operator restarted the validator. HF is where the checkpoint durably
    lives — the same place miners pull it from.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    if not checkpoint_revision:
        raise RuntimeError("proof worker reload requires a checkpoint revision")

    # The loader records what it actually read, but startup still reports no
    # certified revision. Only the authenticated adoption path can bind it.
    initial_source = context.pop("_initial_source", None)
    if (not snapshot_dir and repo_id and context.get("model") is not None
            and initial_source == (repo_id, checkpoint_revision)):
        context["revision"] = checkpoint_revision
        return

    if context.get("replica") == STREAMED:
        # A streamed replica has no layers to load into: its weights live in the checkpoint and
        # are read one at a time. Rotating it means pointing it at the new directory, which costs
        # only the fixed parts — the layers are not held in the first place.
        _install_streamed(context, snapshot_dir, checkpoint_revision, repo_id)
        return

    state: dict[str, Any] = {}
    if snapshot_dir and Path(snapshot_dir).is_dir():
        for path in sorted(Path(snapshot_dir).glob("*.safetensors")):
            state.update(load_file(str(path), device="cpu"))
        if not state and not repo_id:
            raise RuntimeError(f"no safetensors under {snapshot_dir}")

    if not state:
        if not repo_id:
            raise RuntimeError(
                "proof worker reload has no source: snapshot dir "
                f"{snapshot_dir!r} is unusable and no repo_id was given"
            )
        _install_from_hub(context, repo_id, checkpoint_revision)
        return

    model = context.get("model")
    if model is None:
        # A hub install that failed after releasing the old replica leaves the
        # slot empty. The in-place path has nothing to load into, so rebuild
        # rather than fail every swap from here on.
        if not repo_id:
            raise RuntimeError(
                "proof worker holds no replica and no repo_id is available "
                f"to rebuild it for {checkpoint_revision!r}"
            )
        _install_from_hub(context, repo_id, checkpoint_revision)
        return

    tied = set(getattr(model, "_tied_weights_keys", None) or [])
    result = model.load_state_dict(state, strict=False)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    missing = [
        key for key in getattr(result, "missing_keys", []) or []
        if key not in tied
    ]
    if unexpected or missing:
        raise RuntimeError(
            "staged checkpoint state mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if tied and hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    context["revision"] = checkpoint_revision


def _install_from_hub(
    context: MutableMapping[str, Any],
    repo_id: str,
    checkpoint_revision: str,
) -> None:
    """Rebuild the replica straight from the durable checkpoint repo."""
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared import modeling

    from reliquary.validator.proof_capacity import physical_proof_device

    # Assembled on the host first, then the old replica is released before the
    # replacement touches the card: holding both is 20.4 GB on one GPU, and
    # with several slots that spike lands on a card that is already full.
    model = modeling.load_text_generation_model(
        repo_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        revision=checkpoint_revision,
    )
    device = context.get("device")
    # A failed move leaves the slot with no model. A reload reaching it next
    # rebuilds here (see the None guard in reload_proof_context); a PROOF
    # reaching it first raises, and the scheduler turns any proof error into a
    # FAULTED plane, i.e. a validator restart with an unpaid window. That cost
    # is why the concurrent-load bound above matters: the trade is only worth
    # it against an OOM, which takes every other slot down as well.
    # gc.collect first: a reference cycle in the module graph would defer the
    # free past empty_cache and leave the old 10.2 GB resident under the move.
    context["model"] = None
    context["revision"] = None
    gc.collect()
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device is not None and hasattr(model, "to"):
        model = model.to(physical_proof_device(device))
    model = model.eval()
    for parameter in getattr(model, "parameters", list)():
        parameter.requires_grad = False
    context["model"] = model
    context["revision"] = checkpoint_revision


from reliquary.shared.replica_strategy import RESIDENT, STREAMED


def choose_proof_replica(
    checkpoint: str, physical_device: str, declared: str | None = None,
) -> str:
    """Whether this slot can hold the model, or has to walk it one layer at a time.

    Derived from the checkpoint's size against the card's free memory, so no operator has to know
    what a model is made of. A task that pins a replica overrides the derivation — that is how a
    fleet of unequal cards is made to run one path — and ``RELIQUARY_PROOF_REPLICA`` overrides
    both, for a shadow run or an incident.
    """
    import os
    from pathlib import Path

    import torch

    from reliquary.shared.replica_strategy import RESIDENT, checkpoint_weight_bytes, choose_replica

    override = os.environ.get("RELIQUARY_PROOF_REPLICA") or None
    if not str(physical_device).startswith("cuda"):
        # Without a card there is nothing to outgrow, and nothing to refuse a task for.
        return choose_replica(
            weights_bytes=0, free_bytes=1, override=override, declared=declared,
        )
    try:
        weights = checkpoint_weight_bytes(Path(checkpoint))
    except (FileNotFoundError, OSError):
        # A checkpoint we cannot size is a checkpoint we have always loaded resident.
        return choose_replica(
            weights_bytes=0, free_bytes=1, override=override, declared=declared,
        )
    free, _total = torch.cuda.mem_get_info(torch.device(physical_device))
    return choose_replica(
        weights_bytes=weights, free_bytes=free, override=override, declared=declared,
    )


def build_proof_context(
    *,
    checkpoint: str,
    device: str,
    load_kwargs: Mapping[str, Any] | None = None,
    replica: str | None = None,
) -> dict[str, Any]:
    """Load this worker's bootstrap replica, exactly as the in-process path did.

    Same dtype, same attention implementation, same pinned revision: the
    isolated plane must not change a single kernel, only which interpreter
    drives it.

    The returned ``revision`` is deliberately ``None`` until adoption certifies
    the checkpoint. The default loads the base model; a pinned proof startup
    can preload the intended published revision. The initial source records
    only what this loader read, allowing exact adoption to reuse those weights.
    """
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared import modeling
    from reliquary.shared.replica_strategy import RESIDENT, STREAMED
    from reliquary.shared.streaming_forward import StreamedReplica
    from reliquary.validator.proof_capacity import physical_proof_device

    kwargs = dict(load_kwargs or {})
    tokenizer = modeling.load_tokenizer(checkpoint, **kwargs)
    # ``device`` is a proof SLOT id: several slots can share one card, and
    # torch does not understand the ``cuda:0#1`` form. The slot id stays in the
    # context because that is this worker's identity to the pool and scheduler.
    physical = physical_proof_device(device)
    replica = choose_proof_replica(checkpoint, physical, replica)
    if replica == STREAMED:
        # The model outgrew the card: keep one decoder layer on it at a time. What the
        # verification path reads off a model is unchanged, so nothing downstream moves.
        model = StreamedReplica.from_checkpoint(
            checkpoint,
            device=physical,
            dtype=torch.bfloat16,
            prefetch=True,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
    else:
        model = modeling.load_text_generation_model(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **kwargs,
        ).to(physical).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "replica": replica,
        "revision": None,
        "_initial_source": (checkpoint, kwargs.get("revision")),
    }


PROOF_CONTEXT_FACTORY = "reliquary.validator.proof_worker:build_proof_context"
PROOF_HANDLER = "reliquary.validator.proof_worker:run_commitment_proof"
PROOF_RELOAD_HANDLER = "reliquary.validator.proof_worker:reload_proof_context"
PROOF_WARM_HANDLER = "reliquary.validator.proof_worker:warm_commitment_batch"


def build_isolated_proof_plane(
    *,
    devices: Sequence[str],
    checkpoint: str,
    load_kwargs: Mapping[str, Any] | None = None,
    reference_model: Any = None,
    replica: str | None = None,
) -> tuple["ProofWorkerPool", dict[str, ProofModelProxy]]:
    """Assemble the isolated plane: one worker per proof slot, one proxy each.

    The returned pool is NOT started — the caller decides when to pay the
    model load. Proxies carry only the EOS metadata the proof-dependent gates
    read; the weights live in the workers.
    """
    from reliquary.constants import (
        PROOF_WORKER_RELOAD_TIMEOUT_SECONDS,
        PROOF_WORKER_REQUEST_TIMEOUT_SECONDS,
    )

    pool = ProofWorkerPool(
        devices=tuple(devices),
        context_factory=PROOF_CONTEXT_FACTORY,
        handler=PROOF_HANDLER,
        reload_handler=PROOF_RELOAD_HANDLER,
        warm_handler=PROOF_WARM_HANDLER,
        factory_kwargs={
            "checkpoint": checkpoint,
            "load_kwargs": dict(load_kwargs or {}),
            "replica": replica,
        },
        request_timeout_seconds=PROOF_WORKER_REQUEST_TIMEOUT_SECONDS,
        reload_timeout_seconds=PROOF_WORKER_RELOAD_TIMEOUT_SECONDS,
        start_timeout_seconds=PROOF_WORKER_RELOAD_TIMEOUT_SECONDS,
    )
    proxies = {
        device: ProofModelProxy(
            device_id=device,
            config=getattr(reference_model, "config", None),
            generation_config=getattr(
                reference_model, "generation_config", None
            ),
        )
        for device in devices
    }
    return pool, proxies


# CUDA's own default when CUDA_MPS_PIPE_DIRECTORY is unset. A box running the
# daemon on the default path sets nothing, so reading the variable alone would
# report "no MPS" for the common case.
_DEFAULT_MPS_PIPE_DIRECTORY = "/tmp/nvidia-mps"


def mps_control_pipe_path() -> str:
    """The named pipe ``nvidia-cuda-mps-control -d`` creates when it starts."""
    import os

    directory = (
        os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "").strip()
        or _DEFAULT_MPS_PIPE_DIRECTORY
    )
    return os.path.join(directory, "control")


def assert_proof_slots_supported(
    *,
    slots_per_device: int,
    isolation: bool,
    proof_devices: Sequence[str] = (),
) -> None:
    """Extra slots only mean anything as separate interpreters on real cards.

    In-process they would be threads of the validator's own interpreter, i.e.
    the GIL convoy the isolated plane exists to escape (the same forward
    measured 28.7 ms alone and 29.6 s against one CPU-bound python thread).
    Refuse the combination instead of serving the convoy under a name that
    promises parallelism.

    With no card resolved there is no plane at all and the slot count is simply
    dropped — a warning, not a refusal, because a protocol profile below v3
    legitimately configures no proof device.
    """
    if int(slots_per_device) > 1 and not isolation:
        raise RuntimeError(
            "RELIQUARY_PROOF_SLOTS_PER_DEVICE > 1 requires "
            "RELIQUARY_PROOF_PROCESS_ISOLATION: in-process slots would share "
            "this interpreter's GIL, which is what isolation exists to avoid"
        )
    if int(slots_per_device) > 1 and not proof_devices:
        logger.warning(
            "RELIQUARY_PROOF_SLOTS_PER_DEVICE=%d ignored: no proof device is "
            "configured, so no isolated plane is built and proving stays "
            "in-process",
            int(slots_per_device),
        )
        return
    if int(slots_per_device) > 1:
        import os

        pipe = mps_control_pipe_path()
        if not os.path.exists(pipe):
            logger.warning(
                "%d proof slots per GPU, but no CUDA MPS control pipe at %s. "
                "Without an MPS server the slots' CUDA contexts time-slice "
                "instead of overlapping: 4 slots measured 8.3 s against 5.7 s "
                "with one. Start it with nvidia-cuda-mps-control -d and give "
                "the container the same CUDA_MPS_PIPE_DIRECTORY. Nothing else "
                "reports this — the proofs stay correct, only slower.",
                int(slots_per_device),
                pipe,
            )


def validator_replica_device(
    *, isolated_plane: bool, gpu_device: str = "cuda:0",
) -> str:
    """Device for the validator's OWN train/verify replicas.

    With an isolated plane this process neither trains (the detached trainer
    owns that, and isolation requires it — see ``assert_isolation_supported``)
    nor proves (every proof runs in a worker holding its own replica). Measured
    on the live validator 2026-08-25: the main process held 31.4 GB while
    ``nvidia-smi pmon`` reported ``sm = 0`` across a full proof burst. Keeping
    the pair on the CPU hands that budget to the workers instead.

    ``isolated_plane`` is whether a plane was actually BUILT, not whether the
    flag is set: RELIQUARY_PROOF_PROCESS_ISOLATION can be on while no proof
    device resolves (a protocol profile below v3), and then the batcher still
    proves in-process against this replica. Deciding on the flag alone would
    leave a flash-attention-2 model on the CPU and prove there, silently.
    """
    return "cpu" if isolated_plane else gpu_device


# What the validator's own replicas cost in host RAM once they leave the card.
# The train/verify pair is ~16 GB steady, but RELIQUARY_RESUME_FROM rebinds
# train_model only after the replacement has loaded, so three are alive for a
# moment at every boot. A pinned KL reference is a fourth.
VALIDATOR_REPLICA_HOST_MEMORY_FLOOR_GB = 24.0
VALIDATOR_KL_REFERENCE_HOST_MEMORY_GB = 8.0


def _available_host_memory_gb() -> float | None:
    """``MemAvailable`` in GB, or None where /proc/meminfo is not readable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def assert_host_memory_for_cpu_replicas(
    *,
    isolated_plane: bool,
    kl_base_model: bool = False,
    available_gb: float | None = None,
) -> None:
    """Warn when the host cannot carry the replicas the CPU move puts on it.

    An isolated plane hands this process's train/verify pair to host RAM,
    converting VRAM into a PERMANENT RSS floor. This validator has a known
    ~11 GB/h RSS leak and has been OOM-killed before, so the floor shortens
    time-to-OOM proportionally — and the OOM lands mid-window: an aborted,
    unpaid window plus a lost LR warmup.

    Warn, never refuse. A validator running with a tight host still earns; one
    that will not boot does not. This exists so the regression is a decision at
    startup rather than a restart hours later.
    """
    if not isolated_plane:
        return
    floor = VALIDATOR_REPLICA_HOST_MEMORY_FLOOR_GB
    if kl_base_model:
        floor += VALIDATOR_KL_REFERENCE_HOST_MEMORY_GB
    available = (
        _available_host_memory_gb() if available_gb is None
        else float(available_gb)
    )
    if available is None or available >= floor:
        return
    logger.warning(
        "Isolated proof plane: this process's replicas load on the CPU and "
        "need ~%.0f GB of host RAM, but only %.1f GB is available. They are a "
        "permanent RSS floor, not a transient — with the known RSS leak this "
        "box will reach OOM sooner, and an OOM mid-window costs an unpaid "
        "window and the LR warmup. Free host RAM, or run without "
        "RELIQUARY_PROOF_PROCESS_ISOLATION.",
        floor,
        available,
    )


def assert_isolation_supported(
    *, isolation: bool, detached_trainer: bool,
) -> None:
    """Refuse the one combination whose checkpoint swap cannot work.

    An isolated worker takes new weights from a staged snapshot directory,
    which only the detached-trainer intake produces. In-process training
    publishes an in-memory state dict the worker has no way to receive, so
    the first publication would strand the plane on stale weights.
    """
    if isolation and not detached_trainer:
        raise RuntimeError(
            "RELIQUARY_PROOF_PROCESS_ISOLATION requires "
            "RELIQUARY_DETACHED_TRAINER: the isolated proof worker reloads "
            "from the staged snapshot directory, which only the detached "
            "trainer intake stages"
        )
