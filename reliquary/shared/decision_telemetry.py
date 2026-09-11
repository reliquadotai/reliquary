"""Bounded private decision events; observation never participates in consensus."""

from __future__ import annotations

import contextvars
import functools
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

_context = contextvars.ContextVar("decision_context", default={})
_writer = None
_lock = threading.Lock()


def enabled():
    return os.getenv("RELIQUARY_DECISION_TELEMETRY_ENABLED", "0").lower() in {
        "1",
        "true",
        "yes",
    }


class EventWriter:
    """One bounded queue and rotating private file per process, no network."""

    def __init__(self, directory, *, capacity=2048, max_bytes=16_777_216, keep=16):
        self.directory = Path(directory)
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.keep = keep
        self.queue = queue.Queue(maxsize=capacity)
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.put_lock = threading.Lock()
        self.written = self.dropped = self.errors = 0
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="decision-events"
        )
        self.thread.start()

    def put(self, event):
        with self.put_lock:
            self.sequence += 1
            event = {
                **event,
                "sequence": self.sequence,
                "writer_status": self.snapshot(),
            }
            try:
                self.queue.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def snapshot(self):
        return dict(
            run_id=self.run_id,
            written=self.written,
            dropped=self.dropped,
            errors=self.errors,
            pending=self.queue.qsize(),
        )

    def _run(self):
        path = None
        size = 0
        part = 0
        while True:
            event = self.queue.get()
            try:
                raw = (
                    json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
                ).encode()
                if len(raw) > 262144:
                    self.dropped += 1
                    continue
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                if path is None or size + len(raw) > self.max_bytes:
                    part += 1
                    path = self.directory / f"events-{self.run_id}-{part:06d}.jsonl"
                    size = 0
                    # Retention is per process run; host retention spans old runs separately.
                    previous = sorted(
                        self.directory.glob(f"events-{self.run_id}-*.jsonl")
                    )
                    for old in previous[: max(0, len(previous) - self.keep + 1)]:
                        old.unlink(missing_ok=True)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "ab") as stream:
                    stream.write(raw)
                size += len(raw)
                self.written += 1
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()


def writer():
    global _writer
    if _writer is None:
        with _lock:
            if _writer is None:
                root = os.getenv("RELIQUARY_DECISION_TELEMETRY_DIR")
                if not root:
                    root = str(
                        Path(os.getenv("RELIQUARY_STATE_DIR", "/tmp/reliquary"))
                        / "decision_telemetry"
                    )
                _writer = EventWriter(root)
    return _writer


def emit(event, **fields):
    if not enabled():
        return
    try:
        output = writer()
        output.put(
            {
                **_context.get(),
                **fields,
                "schema_version": 1,
                "event": event,
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "run_id": output.run_id,
            }
        )
    except Exception:
        # This path cannot fail training, admission, payment or signatures.
        return


def snapshot():
    return {"enabled": enabled(), **(_writer.snapshot() if _writer else {})}


def capture(event, fields):
    """Defer ALL metadata work so observation failures cannot affect the caller."""
    if enabled():
        try:
            emit(event, **fields())
        except Exception:
            emit("measurement_missing", stage=event)


def group_ref(group, *, window=None, environment=None, checkpoint=None):
    """No tokens, signatures or wallet material; selected identity joins journal data."""
    try:
        origin = getattr(group, "_decision_origin", {})
        rollouts = getattr(
            group, "rollouts", getattr(getattr(group, "request", None), "rollouts", [])
        )
        root = getattr(group, "merkle_root", None)
        if isinstance(root, bytes):
            root = root.hex()
        return dict(
            window=origin.get("window", window),
            environment=origin.get(
                "environment",
                environment
                or (getattr(rollouts[0], "env_name", None) if rollouts else None),
            ),
            checkpoint=origin.get("checkpoint", checkpoint),
            journal_key=origin.get("journal_key"),
            hotkey=getattr(group, "hotkey", None),
            prompt_idx=int(getattr(group, "prompt_idx", -1)),
            root=root if isinstance(root, str) else None,
            rollouts=len(rollouts),
            eos_tokens=int(getattr(group, "eos_tokens", 0) or 0),
        )
    except Exception:
        return {"identity_unavailable": True}


def observe(event):
    """Trace synchronous calls without retaining GPU tensors or changing results."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            call_id = uuid.uuid4().hex
            parent = _context.get()
            context = {**parent, "call_id": call_id}
            try:
                if event == "proof_callable":
                    pending = args[1]
                    context.update(
                        window=args[0].window_start,
                        receipt_id=getattr(
                            pending.request, "_precommit_receipt_id", None
                        ),
                        group=group_ref(pending),
                    )
            except Exception:
                context["identity_unavailable"] = True
            if parent.get("call_id"):
                context["parent_call_id"] = parent["call_id"]
            for key in ("window_index", "env_name", "prompt_idx", "checkpoint_hash"):
                if key in kwargs:
                    context[key] = kwargs[key]
            token = _context.set(context)
            start = time.monotonic()
            emit(event + "_started")
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                emit(
                    event + "_failed",
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - start,
                )
                raise
            else:
                extra = {}
                try:
                    if event == "miner_generation" and isinstance(result, list):
                        extra = {
                            "rollouts": len(result),
                            "generated_tokens": sum(
                                max(0, len(r["tokens"]) - r["prompt_length"])
                                for r in result
                            ),
                        }
                except Exception:
                    extra = {"measurement_unavailable": True}
                emit(event + "_returned", seconds=time.monotonic() - start, **extra)
                return result
            finally:
                _context.reset(token)

        return wrapped

    return decorate


def annotate_origins(batches, decoded):
    if not enabled():
        return batches
    try:
        for env, groups in batches.items():
            for group in groups:
                group._decision_origin = dict(
                    window=decoded.window_start,
                    environment=env,
                    checkpoint=decoded.checkpoint_revision,
                    journal_key=getattr(decoded, "_decision_journal_key", None),
                )
        emit(
            "trainer_payload_groups",
            window=decoded.window_start,
            checkpoint=decoded.checkpoint_revision,
            groups=[
                group_ref(g, environment=env)
                for env, groups in batches.items()
                for g in groups
            ],
        )
    except Exception:
        emit("measurement_missing", stage="trainer_group_origins")
    return batches


def journal_received(decoded, key):
    if enabled():
        try:
            decoded._decision_journal_key = key
        except Exception:
            emit("measurement_missing", stage="journal_identity")


def optimizer_receipt(plan, *, n_processed, window, step_index, item_builder):
    if not enabled():
        return
    # The caller invokes this ONLY after optimizer.step returns successfully.
    # Count actual eligible microbatch rows; never equate a cursor with an update.
    emit(
        "optimizer_success",
        window=window,
        step_index=step_index,
        processed_rollouts=n_processed,
    )
    try:
        groups = []
        for group, advantages, scale in plan:
            items = item_builder([(group, advantages, scale)])
            if items:
                groups.append(
                    {
                        **group_ref(group, window=window),
                        "processed_rollouts": len(items),
                        "trainable_tokens": sum(sum(item[5]) for item in items),
                    }
                )
        emit(
            "optimizer_group_receipt",
            window=window,
            step_index=step_index,
            reconciled=sum(g["processed_rollouts"] for g in groups) == n_processed,
            groups=groups,
        )
    except Exception:
        emit("measurement_missing", stage="optimizer_group_receipt", window=window)


def begin_attempt(*, window, environment, prompt_idx, checkpoint):
    if not enabled():
        return
    try:
        _context.set(
            dict(
                attempt_id=uuid.uuid4().hex,
                window=window,
                environment=environment,
                prompt_idx=prompt_idx,
                checkpoint=checkpoint,
            )
        )
        emit("miner_attempt_started")
    except Exception:
        pass
