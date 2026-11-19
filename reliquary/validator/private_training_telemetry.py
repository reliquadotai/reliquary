"""Bounded best-effort private W&B writer, isolated from the training/publisher threads."""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections.abc import Mapping
from types import MappingProxyType

from reliquary.shared.private_training_wandb import PRIVATE_TRAINING_WANDB_SCHEMA
from reliquary.shared.training_outputs import metric_values

logger = logging.getLogger(__name__)
MAX_PENDING_ROWS = 32
MAX_METRICS = 256
INITIAL_RETRY_SECONDS = 1.0
MAX_RETRY_SECONDS = 60.0
_MAX_SAFE_INTEGER = 2**53 - 1


class _IdentityRefused(ValueError):
    pass


class PrivateTrainingWriter:
    """A stalled SDK consumes one daemon thread and a bounded queue, never the trainer."""

    def __init__(self, identity: Mapping[str, str]) -> None:
        self.identity = MappingProxyType(dict(identity))
        self.active = False
        self._queue = queue.Queue(maxsize=MAX_PENDING_ROWS)
        self._closing = threading.Event()
        self._warned = False
        self._thread = threading.Thread(target=self._work, name="private-training-wandb", daemon=True)
        self._thread.start()

    def prepare(self, metrics: Mapping, checkpoint: int, observed_at: float):
        """Copy just finite allowlisted measurements before the next update changes buffers."""
        if (isinstance(checkpoint, bool) or not isinstance(checkpoint, int)
                or not 0 <= checkpoint <= _MAX_SAFE_INTEGER
                or isinstance(observed_at, bool) or not isinstance(observed_at, (int, float))
                or not math.isfinite(observed_at) or not 0 < observed_at < 1e11):
            return None
        values = metric_values(metrics)
        if not values or len(values) > MAX_METRICS:
            return None
        row = {**values, "reliquary/order_id": self.identity["order_id"],
               "reliquary/task_id": self.identity["task_id"],
               "async/published_checkpoint": checkpoint, "_timestamp": float(observed_at)}

        def after_publication() -> None:
            if self._closing.is_set():
                return
            try:
                self._queue.put_nowait((checkpoint, row))
            except queue.Full:
                # Retain the newest committed observations during a long SDK outage.
                # A missing observation stays a gap, never a synthesized value.
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._queue.put_nowait((checkpoint, row))
                except (queue.Empty, queue.Full):
                    pass
                self._warn()

        return after_publication

    def _warn(self) -> None:
        if not self._warned:
            # Do not include SDK exceptions, credentials, project names or private identities.
            logger.warning("private training telemetry unavailable or full; training continues")
            self._warned = True

    def _work(self) -> None:
        run = None
        try:
            import wandb

            i = self.identity
            config = {"reliquary_order_id": i["order_id"], "reliquary_task_id": i["task_id"],
                      "reliquary_checkpoint_key": "async/published_checkpoint",
                      "reliquary_writer_schema": PRIVATE_TRAINING_WANDB_SCHEMA}
            delay = INITIAL_RETRY_SECONDS
            while not self._closing.is_set():
                try:
                    run = wandb.init(entity=i["entity"], project=i["project"], id=i["run_id"],
                                     resume="allow", reinit=True, job_type="trainer", config=config,
                                     save_code=False,
                                     settings=wandb.Settings(init_timeout=15, console="off", disable_git=True,
                                                             disable_code=True))
                    if (run is None or run.id != i["run_id"] or run.entity != i["entity"]
                            or run.project != i["project"]
                            or any(run.config.get(key) != value for key, value in config.items())):
                        raise _IdentityRefused("private telemetry identity mismatch")
                    next_step = getattr(run, "step", 0)
                    if isinstance(next_step, bool) or not isinstance(next_step, int) or next_step < 0:
                        raise _IdentityRefused("invalid resumed private telemetry cursor")
                    break
                except _IdentityRefused:
                    raise  # Never retry an SDK response bound to a different identity.
                except Exception:
                    self._warn()
                    if self._closing.wait(delay):
                        return
                    delay = min(MAX_RETRY_SECONDS, delay * 2)
            else:
                return
            self.active = True
            while not self._closing.is_set() or not self._queue.empty():
                try:
                    checkpoint, row = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if checkpoint < next_step:
                        continue
                    run.log(row, step=checkpoint, commit=True)
                    next_step = checkpoint + 1
                    self.active = True
                except Exception:
                    self.active = False
                    self._warn()
                finally:
                    self._queue.task_done()
        except Exception:
            self._warn()
        finally:
            self.active = False
            if run is not None:
                try:
                    run.finish()
                except Exception:
                    self._warn()

    def close(self, timeout: float = 0.5) -> None:
        self._closing.set()
        self._thread.join(timeout=timeout)
