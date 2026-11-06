"""`reliquary corpus audit-executor`: a rented GPU that scores audit leases.

It holds one secret, its executor token, and pulls work over HTTPS it opens
itself: heartbeat, claim a lease, score it with the public checkpoint at the
pinned revision, post the per-item chunk comparisons. It never sees a hotkey or
a verdict, and needs no inbound port.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    AUDIT_PROTOCOL_V2,
    AUDIT_PROTOCOL_V3,
    ITEM_ERROR,
    PROTOCOLS,
    AuditLease,
)
from reliquary.validator.lease_executor import TOKEN_ENV, LeaseExecutor, serve_executor

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 20.0
IDLE_SECONDS = 2.0
# After a control refused v3, how long before the executor offers it again.
V3_RETRY_SECONDS = 600.0
AUDIT_PREFIX = "/corpus/internal/audit"
EVAL_AUDIT_PREFIX = "/corpus/internal/eval-audit"


# A rented pod's root disk can be a few dozen GB while its volume holds a terabyte.
VOLUME_DIR = "/workspace"
DISK_MARGIN_BYTES = 4 * 1024**3


class DiskTooSmall(RuntimeError):
    """This pod cannot hold the checkpoint: no retry here will fix it."""


def _cache_dir(volume: str = VOLUME_DIR) -> str | None:
    """The pod's volume when the operator set no HF cache and it is mounted and writable; else the default."""
    import os

    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return None
    if os.path.ismount(volume) and os.access(volume, os.W_OK):
        return os.path.join(volume, "hf", "hub")
    return None


def _check_disk(model_id: str, revision: str, cache_dir: str | None, *, info=None, free=None) -> None:
    """Raise ``DiskTooSmall`` when the checkpoint does not fit where it will be downloaded."""
    import os
    import shutil

    from huggingface_hub import HfApi, constants

    target = cache_dir or constants.HF_HUB_CACHE
    if info is None:
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True, token=False)
    needed = sum(int(f.size or 0) for f in (info.siblings or ())) + DISK_MARGIN_BYTES
    if free is None:
        probe = target
        while not os.path.exists(probe):
            probe = os.path.dirname(probe) or "/"
        free = shutil.disk_usage(probe).free
    if free < needed:
        raise DiskTooSmall(f"{model_id}@{revision[:12]} needs {needed / 1e9:.0f} GB at {target}, "
                           f"{free / 1e9:.0f} GB free")


def load_public_model(model_id: str, revision: str):
    """The checkpoint from the public HF repo at the pinned revision, with no
    credential at all (``token=False``), on the pod's volume when it has one."""
    import torch
    from huggingface_hub import snapshot_download

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_only_model

    cache_dir = _cache_dir()
    try:
        directory = snapshot_download(model_id, revision=revision, token=False, cache_dir=cache_dir,
                                      local_files_only=True)
    except Exception:
        _check_disk(model_id, revision, cache_dir)
        directory = snapshot_download(model_id, revision=revision, token=False, cache_dir=cache_dir)
    return load_text_only_model(
        directory, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda").eval()


class AuditExecutor(LeaseExecutor):
    kind = "audit executor"

    def __init__(self, *, http, executor_id: str, token: str, model_id: str | None = None,
                 model_revision: str | None = None,
                 load_model: Callable[[str, str], Any] = load_public_model,
                 batch_tokens: int | None = None,
                 heartbeat_seconds: float = HEARTBEAT_SECONDS,
                 idle_seconds: float = IDLE_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 prefix: str = AUDIT_PREFIX) -> None:
        # The eval control serves the same protocol under its own prefix.
        super().__init__(http=http, executor_id=executor_id, token=token, prefix=prefix,
                         heartbeat_seconds=heartbeat_seconds, idle_seconds=idle_seconds,
                         clock=clock)
        self.model_id, self.model_revision = model_id, model_revision
        self._load_model = load_model
        self._model = None
        if batch_tokens is None:
            from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

            batch_tokens = AUDIT_BATCH_TOKENS
        self._batch_tokens = batch_tokens
        self._v3 = True
        self._v3_retry_at = 0.0

    def heartbeat_detail(self) -> dict:
        return {"leases": self.leases, "loaded": self._model is not None}

    async def start(self) -> None:
        """Learn the registered model from the control when not given, then load it."""
        answer = await self.heartbeat()
        if self.model_id is None or self.model_revision is None:
            self.model_id, self.model_revision = answer["model_id"], answer["model_revision"]
        elif (self.model_id, self.model_revision) != (answer["model_id"], answer["model_revision"]):
            raise RuntimeError(
                f"this executor is registered for {answer['model_id']}@{answer['model_revision']}, "
                f"not {self.model_id}@{self.model_revision}"
            )
        logger.info("audit executor %s loading %s@%s", self._executor_id, self.model_id,
                    self.model_revision)
        try:
            self._model = await self._load_heartbeating()
        except DiskTooSmall as exc:
            # Exiting would only restart the container into the same full disk, heartbeating each time:
            # go silent instead, so the fleet sees a dead pod and replaces it on another host.
            logger.critical("audit executor %s cannot run on this pod: %s", self._executor_id, exc)
            await asyncio.Event().wait()

    async def _load_heartbeating(self):
        """Load the model while heartbeating, so a slow 27B download is not taken for a dead pod."""
        loading = asyncio.ensure_future(asyncio.to_thread(self._load_model, self.model_id, self.model_revision))
        while True:
            done, _ = await asyncio.wait({loading}, timeout=self._heartbeat_every)
            if done:
                return loading.result()
            try:
                await self.heartbeat()
            except Exception as exc:
                logger.warning("audit executor heartbeat failed while loading: %r", exc)

    def _score(self, lease: AuditLease) -> list[dict]:
        from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
        from reliquary.validator.corpus_audit import score_sequences

        try:
            if lease.min_chunk_tokens not in (None, MIN_CHUNK_TOKENS):
                # The floor is the scorer's own; a lease cannot move it.
                raise ValueError(f"lease chunking floor {lease.min_chunk_tokens} is not this build's")
            scores, _, _ = score_sequences(
                self._model,
                [(i.tokens, i.prompt_len, None, None if i.spans is None else [tuple(s) for s in i.spans], i.draw)
                 if i.draw is not None
                 else (i.tokens, i.prompt_len, i.proofs) if i.spans is None
                 else (i.tokens, i.prompt_len, i.proofs, [tuple(s) for s in i.spans])
                 for i in lease.items],
                chunk_tokens=lease.chunk_tokens, topk=lease.topk,
                batch_tokens=self._batch_tokens)
        except Exception as exc:
            # Ours: the batch goes back to the control, nobody is judged on it.
            logger.exception("audit executor could not score lease %s", lease.lease_id[:8])
            return [{"status": ITEM_ERROR, "chunks": [], "detail": str(exc)[:500]}
                    for _ in lease.items]
        from reliquary.validator.corpus_gpu import scores_to_wire

        return [{"status": status, "chunks": chunks, **({"draw": extra[0]} if extra else {})}
                for status, chunks, *extra in scores_to_wire(scores)]

    async def step(self) -> bool:
        """One claim; True when a lease was scored and posted."""
        await self.heartbeat_if_due()
        body = {"executor_id": self._executor_id, "model_id": self.model_id,
                "model_revision": self.model_revision}
        if not self._v3 and self._clock() >= self._v3_retry_at:
            self._v3 = True
        if self._v3:
            # The eval control scores v1 (pairs) and v3 (forced draws).
            body["protocols"] = list(PROTOCOLS) if self._prefix == AUDIT_PREFIX else [AUDIT_PROTOCOL,
                                                                                       AUDIT_PROTOCOL_V3]
        elif self._prefix == AUDIT_PREFIX:
            body["protocols"] = [AUDIT_PROTOCOL, AUDIT_PROTOCOL_V2]
        response = await self._post(f"{self._prefix}/claim", body)
        if response.status_code == 422 and self._v3:
            # A control that predates v3 refuses it: claim as before (corpus v1+v2; eval, no field), and offer v3
            # again later, so an upgraded control gets it without a restart.
            logger.warning("audit control at %s does not take v3; claiming without it", self._prefix)
            self._v3 = False
            self._v3_retry_at = self._clock() + V3_RETRY_SECONDS
            return await self.step()
        if response.status_code == 204:
            return False
        response.raise_for_status()
        lease = AuditLease.model_validate(response.json())
        scores = await asyncio.to_thread(self._score, lease)
        await self.post_result(lease.lease_id, {"scores": scores})
        return True


def run_audit_executor(*, control_url: str, executor_id: str, model_id: str | None = None,
                       model_revision: str | None = None, prefix: str = AUDIT_PREFIX) -> None:
    serve_executor(control_url, lambda **client: AuditExecutor(
        executor_id=executor_id, model_id=model_id, model_revision=model_revision,
        prefix=prefix, **client))


__all__ = ["AuditExecutor", "DiskTooSmall", "TOKEN_ENV", "load_public_model", "run_audit_executor"]
