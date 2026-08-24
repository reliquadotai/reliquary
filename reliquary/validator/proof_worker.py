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
from typing import Any, Callable, Mapping, MutableMapping, Sequence

__all__ = [
    "ProofModelProxy",
    "ProofWorkerPool",
    "ProofWorkerUnavailable",
    "assert_isolation_supported",
    "build_isolated_proof_plane",
    "build_proof_context",
    "reload_proof_context",
    "remote_commitment_verifier",
    "run_commitment_proof",
]


class ProofWorkerUnavailable(RuntimeError):
    """The worker owning a device cannot answer.

    Raised — never returned as a rejection. A dead or unreachable worker is
    validator infrastructure failure: the scheduler must abort the proof plane
    rather than blame the miner whose candidate happened to be in flight.
    """


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
    context = _resolve(context_factory)(device=device, **dict(factory_kwargs))
    handler_fn = _resolve(handler)
    reload_fn = _resolve(reload_handler) if reload_handler else None
    connection.send(("ready", None))

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


class ProofWorkerPool:
    """One dedicated process per proof device.

    Not a parallelism device: two workers on one GPU measure x1.04 because the
    CUDA contexts time-slice. The point is that each proof interpreter has no
    competing python thread.
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
        initial_revision: str | None = None,
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
        self._initial_revision = initial_revision
        self._revisions: dict[str, str | None] = {}
        self._workers: dict[str, _Worker] = {}
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
            # A replacement is only certified for what its factory installed.
            # Anything published since must be re-sent before it proves again.
            self._revisions.setdefault(device_id, self._initial_revision)
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
        parent_conn.recv()
        return _Worker(device_id=device_id, process=process, connection=parent_conn)

    def _request(self, device_id: str, operation: str, args, kwargs) -> Any:
        worker = self._worker_for(device_id)
        try:
            worker.connection.send((operation, args, kwargs))
            timeout = self._request_timeout_seconds
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
                f"proof worker {device_id} died mid-request: {exc!r}"
            ) from exc
        if status == "ok":
            return payload
        raise ProofWorkerUnavailable(
            f"proof worker {device_id} failed: {payload[0]}: {payload[1]}"
        )

    def call(self, device_id: str, *args: Any, **kwargs: Any) -> Any:
        return self._request(device_id, "call", args, kwargs)

    def reload(
        self, device_id: str, snapshot_dir: str, checkpoint_revision: str,
    ) -> None:
        """Install new weights in the worker (checkpoint publication)."""
        self._revisions[device_id] = None
        self._request(
            device_id, "reload", (snapshot_dir, checkpoint_revision), {},
        )
        self._revisions[device_id] = checkpoint_revision

    def revision(self, device_id: str) -> str | None:
        """Revision this worker is certified for, or None when unknown."""
        return self._revisions.get(device_id)

    def close(self) -> None:
        for worker in self._workers.values():
            try:
                worker.connection.send(("shutdown", (), {}))
            except (OSError, BrokenPipeError, ValueError):
                pass
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
    snapshot_dir: str,
    checkpoint_revision: str,
) -> None:
    """Install a published snapshot into the worker's replica.

    Mirrors ``_ValidatorService._refresh_verify_model_from_dir``: assemble the
    whole state dict on CPU first, allow exactly the model's declared tied
    keys, and record the revision only once the load succeeded. A partial
    install must surface as a raise, never as a worker that keeps proving.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    if not checkpoint_revision:
        raise RuntimeError("proof worker reload requires a checkpoint revision")

    state: dict[str, Any] = {}
    for path in sorted(Path(snapshot_dir).glob("*.safetensors")):
        state.update(load_file(str(path), device="cpu"))
    if not state:
        raise RuntimeError(f"no safetensors under {snapshot_dir}")

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


def build_proof_context(
    *,
    checkpoint: str,
    device: str,
    revision: str | None = None,
    load_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load this worker's replica, exactly as the in-process path did.

    Same dtype, same attention implementation, same pinned revision: the
    isolated plane must not change a single kernel, only which interpreter
    drives it.
    """
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared import modeling

    kwargs = dict(load_kwargs or {})
    tokenizer = modeling.load_tokenizer(checkpoint, **kwargs)
    model = modeling.load_text_generation_model(
        checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        **kwargs,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "revision": revision,
    }


PROOF_CONTEXT_FACTORY = "reliquary.validator.proof_worker:build_proof_context"
PROOF_HANDLER = "reliquary.validator.proof_worker:run_commitment_proof"
PROOF_RELOAD_HANDLER = "reliquary.validator.proof_worker:reload_proof_context"


def build_isolated_proof_plane(
    *,
    devices: Sequence[str],
    checkpoint: str,
    load_kwargs: Mapping[str, Any] | None = None,
    revision: str | None = None,
    reference_model: Any = None,
) -> tuple["ProofWorkerPool", dict[str, ProofModelProxy]]:
    """Assemble the isolated plane: one worker per device, one proxy each.

    The returned pool is NOT started — the caller decides when to pay the
    model load. Proxies carry only the EOS metadata the proof-dependent gates
    read; the weights live in the workers.
    """
    from reliquary.constants import PROOF_WORKER_REQUEST_TIMEOUT_SECONDS

    pool = ProofWorkerPool(
        devices=tuple(devices),
        context_factory=PROOF_CONTEXT_FACTORY,
        handler=PROOF_HANDLER,
        reload_handler=PROOF_RELOAD_HANDLER,
        factory_kwargs={
            "checkpoint": checkpoint,
            "revision": revision,
            "load_kwargs": dict(load_kwargs or {}),
        },
        request_timeout_seconds=PROOF_WORKER_REQUEST_TIMEOUT_SECONDS,
        initial_revision=revision,
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
