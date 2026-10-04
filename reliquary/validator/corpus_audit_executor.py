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

from reliquary.validator.corpus_audit_protocol import ITEM_ERROR, AuditLease
from reliquary.validator.lease_executor import TOKEN_ENV, LeaseExecutor, serve_executor

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 20.0
IDLE_SECONDS = 2.0
AUDIT_PREFIX = "/corpus/internal/audit"
EVAL_AUDIT_PREFIX = "/corpus/internal/eval-audit"


def load_public_model(model_id: str, revision: str):
    """The checkpoint from the public HF repo at the pinned revision, with no
    credential at all (``token=False``)."""
    import torch
    from huggingface_hub import snapshot_download

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_only_model

    directory = snapshot_download(model_id, revision=revision, token=False)
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
        self._model = await asyncio.to_thread(self._load_model, self.model_id, self.model_revision)

    def _score(self, lease: AuditLease) -> list[dict]:
        from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
        from reliquary.validator.corpus_audit import score_sequences

        try:
            if lease.min_chunk_tokens not in (None, MIN_CHUNK_TOKENS):
                # The floor is the scorer's own; a lease cannot move it.
                raise ValueError(f"lease chunking floor {lease.min_chunk_tokens} is not this build's")
            scores, _, _ = score_sequences(
                self._model,
                [(i.tokens, i.prompt_len, i.proofs) if i.spans is None
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

        return [{"status": status, "chunks": chunks} for status, chunks in scores_to_wire(scores)]

    async def step(self) -> bool:
        """One claim; True when a lease was scored and posted."""
        await self.heartbeat_if_due()
        body = {"executor_id": self._executor_id, "model_id": self.model_id,
                "model_revision": self.model_revision}
        if self._prefix == AUDIT_PREFIX:
            # The eval control's claim model knows no protocols field.
            body["protocols"] = ["reliquary.corpus-audit/v1", "reliquary.corpus-audit/v2"]
        response = await self._post(f"{self._prefix}/claim", body)
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


__all__ = ["AuditExecutor", "TOKEN_ENV", "load_public_model", "run_audit_executor"]
