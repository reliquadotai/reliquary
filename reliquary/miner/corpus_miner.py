"""A corpus miner: walk the job's prompts in this hotkey's order, generate with
the job's sampling, prove every completion from its own decode activations,
sign, submit.

The loop is written against three small seams (generator, client, signer) so
it is tested without a GPU; ``VllmGenerator`` is the real generator. A
``CorpusClient`` may raise ``CorpusTransientFailure`` (503 ledger contention,
a timeout, a transport error) -- the signed body is idempotent, so the loop
retries the identical submission rather than paying for a fresh generation --
or ``CorpusPermanentFailure`` (any other HTTP error, or a body this client
cannot make sense of); a run of ``max_consecutive_failures`` of the latter
raises ``CorpusMinerHalted`` rather than spinning forever on a route that
will never answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import time
from typing import Protocol

from reliquary.corpus.encoding import completion_text, prompt_token_ids
from reliquary.corpus.walk import walk_index

logger = logging.getLogger(__name__)

# Reasons after which the cursor on the validator is the truth, not ours.
_RESYNC = frozenset({"prompt_full", "bad_cursor", "prompt_mismatch"})

# Backoff delays for a retried request, in seconds; the last value repeats.
# Bounded so a long outage does not turn into an ever-growing sleep.
_TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)

# How many consecutive permanent failures (or reason-less answers) the loop
# tolerates on one call before giving up on it entirely.
_MAX_CONSECUTIVE_FAILURES = 5


class CorpusTransientFailure(Exception):
    """Ledger contention (HTTP 503) or a transport-level failure (timeout,
    connection error). The request that failed is idempotent -- it is either
    a cursor read or a signed, already-built submission -- so the caller
    retries the SAME request rather than building a new one."""


class CorpusPermanentFailure(Exception):
    """An HTTP error this client has no reason to expect will clear on retry
    (404/422/500/...), or a response this client could not parse. Counted by
    the loop rather than left to crash it outright."""

    def __init__(self, message: str, *, status: int | None = None, detail=None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class CorpusMinerHalted(Exception):
    """Raised out of ``mine_steps`` after too many consecutive permanent
    failures on one call, so the CLI can report why and exit non-zero
    instead of the process looping on a job or route that will never
    answer."""

    def __init__(self, message: str, *, counts: dict[str, int]) -> None:
        super().__init__(message)
        self.counts = dict(counts)


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


def _retry(call, *, sleep, counts, max_consecutive_failures):
    """Run ``call`` (a zero-argument callable bound to one idempotent
    request), retrying on failure.

    A transient failure always retries with bounded exponential backoff --
    nothing is lost by trying the same request again. A permanent failure is
    counted, and after ``max_consecutive_failures`` in a row this call is not
    worth retrying further: raises ``CorpusMinerHalted``. The count resets
    whenever a transient failure or a success intervenes, so it measures a
    genuine consecutive run against this one call.
    """
    attempt = 0
    consecutive_permanent = 0
    while True:
        try:
            return call()
        except CorpusTransientFailure as exc:
            consecutive_permanent = 0
            delay = _TRANSIENT_BACKOFF_SECONDS[min(attempt, len(_TRANSIENT_BACKOFF_SECONDS) - 1)]
            logger.warning("corpus request hit a transient failure: %s; retrying in %.0fs", exc, delay)
            sleep(delay)
            attempt += 1
        except CorpusPermanentFailure as exc:
            consecutive_permanent += 1
            counts["permanent_failure"] += 1
            logger.error("corpus request failed (status=%s): %s", exc.status, exc.detail or exc)
            if consecutive_permanent >= max_consecutive_failures:
                raise CorpusMinerHalted(
                    f"{consecutive_permanent} consecutive permanent failures: {exc}",
                    counts=counts,
                ) from exc
            delay = _TRANSIENT_BACKOFF_SECONDS[min(attempt, len(_TRANSIENT_BACKOFF_SECONDS) - 1)]
            sleep(delay)
            attempt += 1


def mine_steps(*, job, hotkey, client, generator, tokenizer, render, sign,
               max_steps: int | None = None, sleep=time.sleep,
               max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES) -> dict[str, int]:
    counts: Counter[str] = Counter()
    retry_kwargs = dict(sleep=sleep, counts=counts, max_consecutive_failures=max_consecutive_failures)
    cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    steps = 0
    consecutive_unreasoned = 0
    while max_steps is None or steps < max_steps:
        steps += 1
        prompt_index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
        rendered = render(prompt_index)
        try:
            generations = generator.generate(prompt_token_ids(tokenizer, rendered), job.sampling.n)
        except ValueError as exc:
            # E.g. a row-count mismatch out of `completion_rows`: vLLM
            # preempted and recomputed a request mid-batch under KV
            # pressure, so this step's activations cannot be trusted. The
            # generator has already dropped its own captured rows for the
            # request ids in this step; there is nothing here to submit, so
            # resync the cursor (this step consumed nothing) and move on.
            logger.warning("dropping a corpus generation step: %s", exc)
            counts["generation_failed"] += 1
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
            continue
        body = build_submission(
            job=job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=rendered, generations=generations, tokenizer=tokenizer, sign=sign,
        )
        answer = _retry(lambda: client.submit(body), **retry_kwargs)
        reason = answer.get("reason")
        if reason is None:
            # A response this route never sends without one: count it the
            # same way a permanent HTTP failure is counted, rather than loop
            # forever on an answer nothing here can act on.
            consecutive_unreasoned += 1
            counts["permanent_failure"] += 1
            logger.error("corpus submission answered with no reason: %r", answer)
            if consecutive_unreasoned >= max_consecutive_failures:
                raise CorpusMinerHalted(
                    f"{consecutive_unreasoned} consecutive corpus answers carried no reason",
                    counts=dict(counts),
                )
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
            continue
        consecutive_unreasoned = 0
        reason = str(reason)
        counts[reason] += 1
        if reason == "job_complete":
            break
        if answer.get("accepted"):
            cursor += 1
        elif reason in _RESYNC:
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
        else:
            logger.warning("corpus submission refused: %s %s", reason, answer.get("detail"))
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    return dict(counts)


class VllmGenerator:
    """vLLM in-process on the V1 runner, capturing decode activations for proofs.

    One request per completion (n=1 each), so every captured row set maps to
    exactly one completion; prefix caching is off because cached rows are never
    recomputed and would be missing from the capture.
    """

    def __init__(self, checkpoint_dir: str, sampling, proof, eos_token_id: int,
                 gpu_memory_utilization: float | None = None) -> None:
        import os

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        from vllm import LLM, SamplingParams

        from reliquary.miner.vllm_hidden_capture import capture_hidden_states

        self._capture_cm = capture_hidden_states()
        self._capture = self._capture_cm.__enter__()
        # Only passed when set: a miner alone on its card keeps vLLM's own
        # default, one sharing it (e.g. with a validator) asks for less.
        memory = {} if gpu_memory_utilization is None else {"gpu_memory_utilization": gpu_memory_utilization}
        self._llm = LLM(model=checkpoint_dir, dtype="bfloat16", enable_prefix_caching=False, **memory)
        self._params = SamplingParams(
            n=1, temperature=sampling.temperature, top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else -1,
            min_tokens=sampling.min_new_tokens, max_tokens=sampling.max_new_tokens,
            # The job's eos is the only terminator `check_text_matches_tokens`
            # judges against; the checkpoint's own generation_config may list
            # others (e.g. both im_end and endoftext), and vLLM stopping on
            # one of those instead would get an honest completion refused
            # bad_termination. `ignore_eos=True` turns off that model-config
            # default so only `stop_token_ids` below can end generation --
            # `min_tokens` above still masks it until the floor is reached.
            # NOT verified against the installed vLLM API (no GPU on this
            # machine): whether vLLM keeps the stop token itself in the
            # output `token_ids` when `stop_token_ids` fires needs confirming
            # on the H100 box in Task 11; `completion_text`'s "drop a
            # trailing eos" branch assumes it does.
            stop_token_ids=[eos_token_id], ignore_eos=True,
        )
        self._proof = proof

    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]:
        import base64

        from vllm.inputs import TokensPrompt

        from reliquary.miner.vllm_hidden_capture import completion_rows
        from reliquary.protocol.toploc_proof import build_chunk_proofs

        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * n, self._params)
        generations = []
        for index, output in enumerate(outputs):
            tokens = list(output.outputs[0].token_ids)
            # `pop`, not `for_request`: this generator lives for the whole
            # mining run, and a request's rows are never read again after its
            # proof is built, so keeping them would grow CPU memory unbounded.
            try:
                rows = completion_rows(self._capture.pop(output.request_id),
                                       len(prompt_ids), len(prompt_ids) + len(tokens))
            except ValueError:
                # A row-count mismatch usually means vLLM preempted and
                # recomputed this request under KV pressure mid-batch: the
                # rest of this batch's captured rows are just as suspect, and
                # every one still in `self._capture` would otherwise sit
                # there forever. Forget them, then let `mine_steps` drop the
                # whole step rather than submit generations no proof can be
                # trusted for.
                for leftover in outputs[index + 1:]:
                    try:
                        self._capture.pop(leftover.request_id)
                    except KeyError:
                        pass
                raise
            proofs = build_chunk_proofs(rows, chunk_tokens=self._proof.chunk_tokens, topk=self._proof.topk)
            generations.append(Generation(tokens, [base64.b64encode(p).decode() for p in proofs]))
        return generations
