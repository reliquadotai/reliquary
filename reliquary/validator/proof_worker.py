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
import importlib
import multiprocessing
import threading
from typing import Any, Callable, Mapping, MutableMapping, Sequence

__all__ = [
    "ProofModelProxy",
    "ProofWorkerPool",
    "ProofWorkerUnavailable",
    "assert_isolation_supported",
    "assert_proof_slots_supported",
    "validator_replica_device",
    "build_isolated_proof_plane",
    "build_proof_context",
    "reload_proof_context",
    "proof_error_type",
    "remote_commitment_verifier",
    "run_commitment_proof",
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

    def verify(
        commit: Any,
        model: Any,
        window_randomness: str,
        *,
        tokenizer: Any = None,
        seed_u_values: Any = None,
    ) -> Any:
        device_id = getattr(model, "device_id", None)
        if not isinstance(model, ProofModelProxy) or not device_id:
            raise ProofWorkerUnavailable(
                "isolated proof plane requires a ProofModelProxy, got "
                f"{type(model).__name__}"
            )
        return pool.call(device_id, commit, window_randomness, seed_u_values)

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
    factory_kwargs: Mapping[str, Any],
    device: str,
) -> None:
    """Child entrypoint: build the heavy context once, then serve requests."""
    try:
        context = _resolve(context_factory)(device=device, **dict(factory_kwargs))
        handler_fn = _resolve(handler)
        reload_fn = _resolve(reload_handler) if reload_handler else None
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
            if operation == "reload":
                if reload_fn is None:
                    raise RuntimeError("worker has no reload handler")
                payload = reload_fn(context, *args, **kwargs)
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
    they overlap, and four slots measure x2.19 (see PROOF_SLOTS_PER_DEVICE).
    Every slot runs the same batch=1 path, so no verdict can shift.
    """

    def __init__(
        self,
        *,
        devices: Sequence[str],
        context_factory: str,
        handler: str,
        reload_handler: str | None = None,
        factory_kwargs: Mapping[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
        reload_timeout_seconds: float | None = None,
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
        self._factory_kwargs = dict(factory_kwargs or {})
        self._request_timeout_seconds = (
            None if request_timeout_seconds is None
            else float(request_timeout_seconds)
        )
        self._reload_timeout_seconds = (
            None if reload_timeout_seconds is None
            else float(reload_timeout_seconds)
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
        # One lock per slot, so two threads racing a respawn still do not
        # each start a process for the SAME slot.
        self._device_spawn_locks: dict[str, threading.Lock] = {}
        self._context = multiprocessing.get_context("spawn")

    @property
    def devices(self) -> tuple[str, ...]:
        return self._devices

    def start(self) -> None:
        for device_id in self._devices:
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
            worker = self._spawn(device_id)
            self._workers[device_id] = worker
        return worker

    def _retire(self, device_id: str) -> None:
        self._revisions[device_id] = None
        worker = self._workers.pop(device_id, None)
        if worker is None:
            return
        try:
            worker.connection.close()
        except OSError:
            pass
        if worker.process.is_alive():
            worker.process.kill()
        worker.process.join(timeout=5.0)

    def _spawn(self, device_id: str) -> _Worker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_conn,),
            kwargs={
                "context_factory": self._context_factory,
                "handler": self._handler,
                "reload_handler": self._reload_handler,
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
            timeout = (
                self._reload_timeout_seconds if operation == "reload"
                else self._request_timeout_seconds
            )
            if timeout is not None and not worker.connection.poll(timeout):
                self._retire(device_id)
                raise ProofWorkerUnavailable(
                    f"proof worker {device_id} timed out after {timeout:g}s"
                )
            status, payload = worker.connection.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            # The child died mid-request. Retire it so the NEXT window finds a
            # live worker, and raise: the scheduler must abort this plane
            # rather than turn our fault into a miner's rejection.
            self._retire(device_id)
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

    def revision(self, device_id: str) -> str | None:
        """Revision this worker is certified for, or None when unknown."""
        return self._revisions.get(device_id)

    def close(self, force: bool = False) -> None:
        """Retire every worker.

        ``force`` skips the polite shutdown frame: when a device thread may
        still be mid-request, writing into its pipe corrupts the exchange it
        is reading. Kill the child instead and let the caller fail loudly.
        """
        for worker in self._workers.values():
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


def run_commitment_proof(
    context: MutableMapping[str, Any],
    commit: Any,
    window_randomness: str,
    seed_u_values: Any = None,
) -> Any:
    """Run one GRAIL proof against the weights this worker owns."""
    from reliquary.validator import verifier as verifier_module

    return verifier_module.verify_commitment_proofs(
        commit,
        context["model"],
        window_randomness,
        tokenizer=context["tokenizer"],
        seed_u_values=seed_u_values,
    )


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

    model = context["model"]
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
    # A failed move leaves no model, so the next proof raises and the worker is
    # retired and respawned — the same fail-closed path a dead worker takes,
    # and better than an OOM that takes every other slot down with it.
    context["model"] = None
    context["revision"] = None
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device is not None and hasattr(model, "to"):
        model = model.to(physical_proof_device(device))
    model = model.eval()
    for parameter in getattr(model, "parameters", list)():
        parameter.requires_grad = False
    context["model"] = model
    context["revision"] = checkpoint_revision


def build_proof_context(
    *,
    checkpoint: str,
    device: str,
    load_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load this worker's bootstrap replica, exactly as the in-process path did.

    Same dtype, same attention implementation, same pinned revision: the
    isolated plane must not change a single kernel, only which interpreter
    drives it.

    The returned ``revision`` is deliberately ``None``. Under auction-v3+ the
    validator bootstraps from the BASE model and only then resumes to the
    trained checkpoint, so this worker holds weights no checkpoint certifies.
    Reporting ``None`` forces ``_synchronize_proof_workers`` to install the
    resumed snapshot before the device is ever marked ready — claiming the
    caller's sha here is what made every proof run against base weights.
    """
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared import modeling
    from reliquary.validator.proof_capacity import physical_proof_device

    kwargs = dict(load_kwargs or {})
    tokenizer = modeling.load_tokenizer(checkpoint, **kwargs)
    # ``device`` is a proof SLOT id: several slots can share one card, and
    # torch does not understand the ``cuda:0#1`` form. The slot id stays in the
    # context because that is this worker's identity to the pool and scheduler.
    model = modeling.load_text_generation_model(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        **kwargs,
    ).to(physical_proof_device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "revision": None,
    }


PROOF_CONTEXT_FACTORY = "reliquary.validator.proof_worker:build_proof_context"
PROOF_HANDLER = "reliquary.validator.proof_worker:run_commitment_proof"
PROOF_RELOAD_HANDLER = "reliquary.validator.proof_worker:reload_proof_context"


def build_isolated_proof_plane(
    *,
    devices: Sequence[str],
    checkpoint: str,
    load_kwargs: Mapping[str, Any] | None = None,
    reference_model: Any = None,
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
        factory_kwargs={
            "checkpoint": checkpoint,
            "load_kwargs": dict(load_kwargs or {}),
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


def assert_proof_slots_supported(
    *, slots_per_device: int, isolation: bool,
) -> None:
    """Extra slots only mean anything as separate interpreters.

    In-process they would be threads of the validator's own interpreter, i.e.
    the GIL convoy the isolated plane exists to escape (the same forward
    measured 28.7 ms alone and 29.6 s against one CPU-bound python thread).
    Refuse the combination instead of serving the convoy under a name that
    promises parallelism.
    """
    if int(slots_per_device) > 1 and not isolation:
        raise RuntimeError(
            "RELIQUARY_PROOF_SLOTS_PER_DEVICE > 1 requires "
            "RELIQUARY_PROOF_PROCESS_ISOLATION: in-process slots would share "
            "this interpreter's GIL, which is what isolation exists to avoid"
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
