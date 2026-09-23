"""A corpus miner: walk the job's prompts in this hotkey's order, generate with
the job's sampling, prove every completion from its own decode activations,
sign, submit.

The loop is written against three small seams (generator, client, signer) so
it is tested without a GPU; ``VllmGenerator`` is the real generator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from typing import Protocol

from reliquary.corpus.encoding import completion_text, prompt_token_ids
from reliquary.corpus.walk import walk_index

logger = logging.getLogger(__name__)

# Reasons after which the cursor on the validator is the truth, not ours.
_RESYNC = frozenset({"prompt_full", "bad_cursor", "prompt_mismatch"})


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    proofs: list[str]


class Generator(Protocol):
    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]: ...


class CorpusClient(Protocol):
    def job(self) -> dict: ...

    def cursor(self, hotkey: str) -> int: ...

    def submit(self, body: dict) -> dict: ...


def build_submission(*, job, hotkey, cursor, prompt_index, rendered_prompt, generations,
                     tokenizer, sign) -> dict:
    body = {
        "job_id": job.job_id,
        "miner_hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "checkpoint_sha256": job.checkpoint_sha256,
        "rendered_prompt": rendered_prompt,
        "completions": [
            {"tokens": list(g.tokens), "text": completion_text(tokenizer, g.tokens, job.eos_token_id),
             "proofs": list(g.proofs)}
            for g in generations
        ],
        "signature": "",
    }
    body["signature"] = sign(body)
    return body


def mine_steps(*, job, hotkey, client, generator, tokenizer, render, sign,
               max_steps: int | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    cursor = client.cursor(hotkey)
    steps = 0
    while max_steps is None or steps < max_steps:
        steps += 1
        prompt_index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
        rendered = render(prompt_index)
        generations = generator.generate(prompt_token_ids(tokenizer, rendered), job.sampling.n)
        answer = client.submit(build_submission(
            job=job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=rendered, generations=generations, tokenizer=tokenizer, sign=sign,
        ))
        reason = str(answer.get("reason"))
        counts[reason] += 1
        if reason == "job_complete":
            break
        if answer.get("accepted"):
            cursor += 1
        elif reason in _RESYNC:
            cursor = client.cursor(hotkey)
        else:
            logger.warning("corpus submission refused: %s %s", reason, answer.get("detail"))
            cursor = client.cursor(hotkey)
    return dict(counts)


class VllmGenerator:
    """vLLM in-process on the V1 runner, capturing decode activations for proofs.

    One request per completion (n=1 each), so every captured row set maps to
    exactly one completion; prefix caching is off because cached rows are never
    recomputed and would be missing from the capture.
    """

    def __init__(self, checkpoint_dir: str, sampling, proof) -> None:
        import os

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        from vllm import LLM, SamplingParams

        from reliquary.miner.vllm_hidden_capture import capture_hidden_states

        self._capture_cm = capture_hidden_states()
        self._capture = self._capture_cm.__enter__()
        self._llm = LLM(model=checkpoint_dir, dtype="bfloat16", enable_prefix_caching=False)
        self._params = SamplingParams(
            n=1, temperature=sampling.temperature, top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else -1,
            min_tokens=sampling.min_new_tokens, max_tokens=sampling.max_new_tokens,
        )
        self._proof = proof

    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]:
        import base64

        from vllm.inputs import TokensPrompt

        from reliquary.miner.vllm_hidden_capture import completion_rows
        from reliquary.protocol.toploc_proof import build_chunk_proofs

        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * n, self._params)
        generations = []
        for output in outputs:
            tokens = list(output.outputs[0].token_ids)
            # `pop`, not `for_request`: this generator lives for the whole
            # mining run, and a request's rows are never read again after its
            # proof is built, so keeping them would grow CPU memory unbounded.
            rows = completion_rows(self._capture.pop(output.request_id),
                                   len(prompt_ids), len(prompt_ids) + len(tokens))
            proofs = build_chunk_proofs(rows, chunk_tokens=self._proof.chunk_tokens, topk=self._proof.topk)
            generations.append(Generation(tokens, [base64.b64encode(p).decode() for p in proofs]))
        return generations
