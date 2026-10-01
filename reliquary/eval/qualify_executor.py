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
    """The result body for a qualify lease. Every prompt is decoded in one
    batch when the generator offers ``generate_many(prompt_ids, ns)``, else
    prompt by prompt with ``generate(prompt_ids, n)``; each returns
    ``Generation(tokens, proofs)``. ``score(items)`` is the prefill verifier
    over ``(tokens, prompt_len, proofs)``."""
    from reliquary.corpus.encoding import prompt_token_ids
    from reliquary.validator.corpus_audit_protocol import ITEM_OK
    from reliquary.validator.corpus_service import ChatTemplatePromptRenderer

    renderer = ChatTemplatePromptRenderer(tokenizer, thinking=bool(lease["thinking"]))
    counts = spread(int(lease["completions"]), len(lease["prompts"]))
    prompts = [(prompt_token_ids(tokenizer, renderer.initial_text(
        SimpleNamespace(prompt=prompt["text"]))), n)
        for prompt, n in zip(lease["prompts"], counts) if n > 0]
    started = clock()
    if callable(getattr(generator, "generate_many", None)):
        grouped = generator.generate_many([ids for ids, _ in prompts], [n for _, n in prompts])
    else:
        grouped = [generator.generate(ids, n) for ids, n in prompts]
    decode_seconds = clock() - started
    items, generated = [], 0
    for (ids, _), generations in zip(prompts, grouped):
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


HEARTBEAT_SECONDS = 20.0


class _Heartbeats:
    """Heartbeats beside the work (model load and qualification included), so
    the platform sees the executor alive: ``{"loaded": bool, "leases": int}``."""

    def __init__(self, send: Callable[[dict], None], every: float) -> None:
        import threading

        self._send, self._every = send, every
        self.state = {"loaded": False, "leases": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="qualify-heartbeat",
                                        daemon=True)

    def _loop(self) -> None:
        while True:
            try:
                self._send(dict(self.state))
            except Exception:
                logger.warning("qualify heartbeat failed", exc_info=True)
            if self._stop.wait(self._every):
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


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
        # Set by run_qualify: the heartbeats this executor keeps up.
        self.beats: _Heartbeats | None = None

    def heartbeat(self, detail: dict) -> None:
        response = self._http.post(f"{EVAL_AUDIT_PREFIX}/heartbeat", headers=self._headers,
                                   json={"executor_id": self._executor_id, "detail": detail},
                                   timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

    def heartbeats(self, every: float = HEARTBEAT_SECONDS) -> _Heartbeats:
        return _Heartbeats(self.heartbeat, every)

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
        if self.beats is not None:
            self.beats.state["leases"] = 1
        try:
            body = self._run(lease)
        finally:
            if self.beats is not None:
                self.beats.state["leases"] = 0
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
    import torch
    import vllm
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_audit_executor import load_public_model

    directory = snapshot_download(model_id, revision=revision, token=False)
    tokenizer = load_tokenizer(directory)
    info = {"gpu_count": max(1, torch.cuda.device_count()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vllm_version": str(vllm.__version__),
            "checkpoint_sha256": checkpoint_fingerprint(directory)}
    # Decoding stops on the model's own eos; the control records it independently.
    eos = int(tokenizer.eos_token_id)
    return SimpleNamespace(directory=directory, tokenizer=tokenizer, info=info, eos=eos,
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
        proof, loaded.eos)
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
    with httpx.Client(base_url=control_url.rstrip("/"), follow_redirects=False) as http:
        state: dict = {}
        executor = QualifyExecutor(
            http=http, executor_id=executor_id,
            token=os.environ.get("RELIQUARY_EXECUTOR_TOKEN", "").strip(),
            model_id=model_id, model_revision=revision,
            run=lambda lease: run_lease_on_gpu(lease, state["loaded"]))
        # Alive from the first second: the download and load take long.
        executor.beats = executor.heartbeats()
        executor.beats.start()
        try:
            state["loaded"] = load_qualifier(model_id, revision)
            executor.beats.state["loaded"] = True
            return executor.run()
        finally:
            executor.beats.stop()


__all__ = [
    "EVAL_AUDIT_PREFIX",
    "QualifyExecutor",
    "qualify_lease",
    "run_qualify",
    "spread",
]
