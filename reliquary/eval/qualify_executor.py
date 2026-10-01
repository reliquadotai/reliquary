"""`reliquary corpus qualify --model repo@rev`: the executor half of qualification.

It claims a qualify lease from the eval control, renders the lease's prompts
with the model's chat template exactly as an eval job does, decodes them with
vLLM capturing TOPLOC proofs (the miners' ``Generator``), verifies every
completion with the HF prefill the auditor uses, and posts every chunk's
measures. The control turns them into thresholds; this side decides nothing.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

logger = logging.getLogger(__name__)

EVAL_AUDIT_PREFIX = "/corpus/internal/eval-audit"
REQUEST_TIMEOUT_SECONDS = 120.0


def spread(completions: int, prompts: int) -> list[int]:
    """``completions`` over ``prompts``, as evenly as possible, first ones first."""
    base, extra = divmod(completions, prompts)
    return [base + (1 if k < extra else 0) for k in range(prompts)]


def qualify_lease(lease: dict, *, tokenizer, generator, score: Callable[[list], list],
                  model_info: dict, clock: Callable[[], float] = time.monotonic) -> dict:
    """The result body for a qualify lease. ``generator.generate(prompt_ids, n)``
    returns ``Generation(tokens, proofs)``; ``score(items)`` is the prefill
    verifier over ``(tokens, prompt_len, proofs)``."""
    from reliquary.corpus.encoding import prompt_token_ids
    from reliquary.validator.corpus_audit_protocol import ITEM_OK
    from reliquary.validator.corpus_service import ChatTemplatePromptRenderer

    renderer = ChatTemplatePromptRenderer(tokenizer, thinking=bool(lease["thinking"]))
    items, generated = [], 0
    decode_seconds = 0.0
    counts = spread(int(lease["completions"]), len(lease["prompts"]))
    for prompt, n in zip(lease["prompts"], counts):
        if n == 0:
            continue
        ids = prompt_token_ids(tokenizer, renderer.initial_text(SimpleNamespace(prompt=prompt["text"])))
        started = clock()
        generations = generator.generate(ids, n)
        decode_seconds += clock() - started
        for generation in generations:
            generated += len(generation.tokens)
            items.append((ids + list(generation.tokens), len(ids), list(generation.proofs)))
    scores = score(items)
    chunks = [[int(c.exp_mismatches), float(c.mant_err_mean), float(c.mant_err_median)]
              for status, results in scores if status == ITEM_OK for c in results]
    return {
        "type": "qualify", "chunks": chunks, "completions": len(items),
        "failed_completions": sum(1 for status, _ in scores if status != ITEM_OK),
        "completion_tokens": generated, "decode_seconds": max(decode_seconds, 1e-6),
        **model_info,
    }


class QualifyExecutor:
    """Claims qualify leases for its model and runs them, synchronously."""

    def __init__(self, *, http, executor_id: str, token: str, model_id: str,
                 model_revision: str, run: Callable[[dict], dict]) -> None:
        if not token:
            raise ValueError("RELIQUARY_EXECUTOR_TOKEN is empty")
        self._http = http
        self._headers = {"Authorization": f"Bearer {token}"}
        self._executor_id = executor_id
        self._model = (model_id, model_revision)
        self._run = run

    def step(self) -> dict | None:
        """One claim; the posted answer, or None when nothing was waiting."""
        from reliquary.eval.qualify_protocol import QualifyLease

        response = self._http.post(f"{EVAL_AUDIT_PREFIX}/claim", headers=self._headers, json={
            "executor_id": self._executor_id, "model_id": self._model[0],
            "model_revision": self._model[1], "kind": "qualify"},
            timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        lease = QualifyLease.model_validate(response.json()).model_dump()
        body = self._run(lease)
        posted = self._http.post(f"{EVAL_AUDIT_PREFIX}/{lease['lease_id']}/result",
                                 headers=self._headers, json=body,
                                 timeout=REQUEST_TIMEOUT_SECONDS)
        posted.raise_for_status()
        return posted.json()

    def run(self, *, sleep: Callable[[float], None] = time.sleep, idle_seconds: float = 10.0,
            max_idle: int | None = None) -> dict | None:
        """Claim until one qualification is done (the executor's one job)."""
        idle = 0
        while max_idle is None or idle < max_idle:
            answer = self.step()
            if answer is not None:
                return answer
            idle += 1
            sleep(idle_seconds)
        return None


def load_qualifier(model_id: str, revision: str):
    """The real decode, verifier and model facts for one model (GPU)."""
    import json as _json

    import torch
    import vllm
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_audit_executor import load_public_model

    directory = snapshot_download(model_id, revision=revision, token=False)
    config = _json.loads(open(os.path.join(directory, "config.json")).read())
    tokenizer = load_tokenizer(directory)
    eos = tokenizer.eos_token_id
    info = {"gpu_count": max(1, torch.cuda.device_count()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vllm_version": str(vllm.__version__),
            "checkpoint_sha256": checkpoint_fingerprint(directory),
            "architecture": (config.get("architectures") or ["unknown"])[0],
            "eos_token_id": int(eos)}
    return SimpleNamespace(directory=directory, tokenizer=tokenizer, info=info,
                           load_model=lambda: load_public_model(model_id, revision))


def run_lease_on_gpu(lease: dict, loaded) -> dict:
    """Decode, then free vLLM, then verify with the HF model on the same card."""
    import gc
    from dataclasses import replace

    import torch

    from reliquary.corpus.job import Sampling
    from reliquary.miner.corpus_miner import VllmGenerator
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.validator.corpus_audit import score_sequences
    from reliquary.validator.corpus_auditor import AUDIT_BATCH_TOKENS

    proof = replace(TOPLOC_DEPLOYED_DEFAULTS, chunk_tokens=lease["chunk_tokens"],
                    topk=lease["topk"])
    sampling = lease["sampling"]
    generator = VllmGenerator(loaded.directory, Sampling(
        temperature=float(sampling.get("temperature", 1.0)),
        top_p=float(sampling.get("top_p", 1.0)), top_k=int(sampling.get("top_k") or 0),
        min_new_tokens=2, max_new_tokens=int(lease["max_new_tokens"]), n=1),
        proof, loaded.info["eos_token_id"])
    pending: list = []

    def collect(items):
        pending.extend(items)
        return [("ok", ())] * len(items)

    measured = qualify_lease(lease, tokenizer=loaded.tokenizer, generator=generator,
                             score=collect, model_info=loaded.info)
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    model = loaded.load_model()
    scores, _, _ = score_sequences(model, pending, chunk_tokens=proof.chunk_tokens,
                                   topk=proof.topk, batch_tokens=AUDIT_BATCH_TOKENS)
    from reliquary.validator.corpus_audit_protocol import ITEM_OK

    measured["chunks"] = [[int(c.exp_mismatches), float(c.mant_err_mean),
                           float(c.mant_err_median)]
                          for status, results in scores if status == ITEM_OK for c in results]
    measured["failed_completions"] = sum(1 for status, _ in scores if status != ITEM_OK)
    return measured


def run_qualify(*, control_url: str, executor_id: str, model: str) -> dict | None:
    import httpx

    model_id, _, revision = model.partition("@")
    if not revision:
        raise ValueError("--model must be repo@revision")
    loaded = load_qualifier(model_id, revision)
    with httpx.Client(base_url=control_url.rstrip("/"), follow_redirects=False) as http:
        executor = QualifyExecutor(
            http=http, executor_id=executor_id,
            token=os.environ.get("RELIQUARY_EXECUTOR_TOKEN", "").strip(),
            model_id=model_id, model_revision=revision,
            run=lambda lease: run_lease_on_gpu(lease, loaded))
        return executor.run()


__all__ = [
    "EVAL_AUDIT_PREFIX",
    "QualifyExecutor",
    "qualify_lease",
    "run_qualify",
    "spread",
]
