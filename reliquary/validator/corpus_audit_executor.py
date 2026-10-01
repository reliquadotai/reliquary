"""`reliquary corpus audit-executor`: a rented GPU that scores audit leases.

It holds one secret, its executor token, and pulls work over HTTPS it opens
itself: heartbeat, claim a lease, score it with the public checkpoint at the
pinned revision, post the per-item chunk comparisons. It never sees a hotkey or
a verdict, and needs no inbound port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from reliquary.validator.corpus_audit_protocol import ITEM_ERROR, AuditLease

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 20.0
IDLE_SECONDS = 2.0
ERROR_BACKOFF_SECONDS = 10.0
REQUEST_TIMEOUT_SECONDS = 120.0
TOKEN_ENV = "RELIQUARY_EXECUTOR_TOKEN"
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


class AuditExecutor:
    def __init__(self, *, http, executor_id: str, token: str, model_id: str | None = None,
                 model_revision: str | None = None,
                 load_model: Callable[[str, str], Any] = load_public_model,
                 batch_tokens: int | None = None,
                 heartbeat_seconds: float = HEARTBEAT_SECONDS,
                 idle_seconds: float = IDLE_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 prefix: str = AUDIT_PREFIX) -> None:
        if not token:
            raise ValueError(f"{TOKEN_ENV} is empty")
        self._http = http
        self._executor_id = executor_id
        # The eval control serves the same protocol under its own prefix.
        self._prefix = prefix
        self._headers = {"Authorization": f"Bearer {token}"}
        self.model_id, self.model_revision = model_id, model_revision
        self._load_model = load_model
        self._model = None
        if batch_tokens is None:
            from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

            batch_tokens = AUDIT_BATCH_TOKENS
        self._batch_tokens = batch_tokens
        self._heartbeat_every = heartbeat_seconds
        self._idle = idle_seconds
        self._clock = clock
        self._last_heartbeat: float | None = None
        self.leases = 0

    async def _post(self, path: str, body: dict):
        return await self._http.post(path, json=body, headers=self._headers,
                                     timeout=REQUEST_TIMEOUT_SECONDS)

    async def heartbeat(self) -> dict:
        response = await self._post(f"{self._prefix}/heartbeat", {
            "executor_id": self._executor_id,
            "detail": {"leases": self.leases, "loaded": self._model is not None},
        })
        response.raise_for_status()
        self._last_heartbeat = self._clock()
        return response.json()

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
        from reliquary.validator.corpus_audit import score_sequences

        try:
            scores, _, _ = score_sequences(
                self._model, [(i.tokens, i.prompt_len, i.proofs) for i in lease.items],
                chunk_tokens=lease.chunk_tokens, topk=lease.topk,
                batch_tokens=self._batch_tokens)
        except Exception as exc:
            # Ours: the batch goes back to the control, nobody is judged on it.
            logger.exception("audit executor could not score lease %s", lease.lease_id[:8])
            return [{"status": ITEM_ERROR, "chunks": [], "detail": str(exc)[:500]}
                    for _ in lease.items]
        return [{"status": status,
                 "chunks": [[r.exp_mismatches, float(r.mant_err_mean), float(r.mant_err_median)]
                            for r in results]}
                for status, results in scores]

    async def step(self) -> bool:
        """One claim; True when a lease was scored and posted."""
        if self._last_heartbeat is None or self._clock() - self._last_heartbeat >= self._heartbeat_every:
            await self.heartbeat()
        response = await self._post(f"{self._prefix}/claim", {
            "executor_id": self._executor_id, "model_id": self.model_id,
            "model_revision": self.model_revision,
        })
        if response.status_code == 204:
            return False
        response.raise_for_status()
        lease = AuditLease.model_validate(response.json())
        scores = await asyncio.to_thread(self._score, lease)
        posted = await self._post(f"{self._prefix}/{lease.lease_id}/result",
                                  {"scores": scores})
        if posted.status_code in (410, 422):
            logger.warning("audit lease %s not taken: %s", lease.lease_id[:8], posted.text[:200])
        else:
            posted.raise_for_status()
        self.leases += 1
        return True

    async def run(self) -> None:
        await self.start()
        while True:
            try:
                worked = await self.step()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 401:
                    # Revoked, expired or quarantined: nothing left to do here.
                    logger.critical("audit executor %s refused by the control: %s",
                                    self._executor_id, exc)
                    raise
                logger.warning("audit executor step failed: %r; backing off", exc)
                await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                continue
            if not worked:
                await asyncio.sleep(self._idle)


def run_audit_executor(*, control_url: str, executor_id: str, model_id: str | None = None,
                       model_revision: str | None = None, prefix: str = AUDIT_PREFIX) -> None:
    import httpx

    token = os.environ.get(TOKEN_ENV, "").strip()

    async def main() -> None:
        async with httpx.AsyncClient(base_url=control_url.rstrip("/"),
                                     follow_redirects=False) as http:
            await AuditExecutor(http=http, executor_id=executor_id, token=token,
                                model_id=model_id, model_revision=model_revision,
                                prefix=prefix).run()

    asyncio.run(main())


__all__ = ["AuditExecutor", "TOKEN_ENV", "load_public_model", "run_audit_executor"]
