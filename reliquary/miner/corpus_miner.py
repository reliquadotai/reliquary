"""A corpus miner: walk the job's prompts in this hotkey's order, generate with
the job's sampling, prove every completion from its own decode activations,
sign, submit.

The loop is written against three small seams (generator, client, signer) so
it is tested without a GPU; ``VllmGenerator`` is the real generator. A
``CorpusClient`` may raise ``CorpusTransientFailure`` (502/503/504, a
timeout, a transport error) -- the signed body is idempotent, so the loop
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
import secrets
import time
from typing import Protocol

from reliquary.corpus.encoding import completion_text, prompt_token_ids
from reliquary.corpus.walk import job_walk_index

logger = logging.getLogger(__name__)

# Reasons after which the cursor on the validator is the truth, not ours.
_RESYNC = frozenset({"prompt_full", "bad_cursor", "prompt_mismatch"})
# Refusals no retry can change: generating on would only burn the card.
_HALT = frozenset({"hotkey_not_registered", "miner_banned"})
# Skip refusals that only say the read was stale: read `next` again.
_SKIP_REREAD = frozenset({"bad_cursor", "prompt_not_full"})
# How many stale skips in a row before generating anyway; submit then settles it.
_MAX_STALE_SKIPS = 3

# Backoff delays for a retried request, in seconds; the last value repeats.
# Bounded so a long outage does not turn into an ever-growing sleep.
_TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)

# How many consecutive permanent failures (or reason-less answers) the loop
# tolerates on one call before giving up on it entirely.
_MAX_CONSECUTIVE_FAILURES = 5

# Statuses that say "not now" rather than "never": ledger contention or a store
# outage (503), and a proxy in front of the validator timing out or losing it
# (502/504). A short outage must not count toward the permanent-failure halt.
TRANSIENT_STATUSES = frozenset({502, 503, 504})
# How long the read of a job's open prompts may take.
OPEN_READ_TIMEOUT_SECONDS = 15.0


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


class CorpusJobRetired(Exception):
    """The validator answered 410 ``job_retired``: the job admits nothing more.
    A job end, not a failure: the miner stops it without retrying."""


class CorpusMinerHalted(Exception):
    """Raised out of ``mine_steps`` after too many consecutive permanent
    failures on one call, so the CLI can report why and exit non-zero
    instead of the process looping on a job or route that will never
    answer."""

    def __init__(self, message: str, *, counts: dict[str, int]) -> None:
        super().__init__(message)
        self.counts = dict(counts)


def _error_detail(response):
    try:
        return response.json()
    except ValueError:
        return response.text[:500]


def _error_object(response) -> dict:
    detail = _error_detail(response)
    return detail if isinstance(detail, dict) else {}


def issue_corpus_request(request_call):
    """Run one httpx request against the corpus validator, translating its
    outcome into the two exceptions ``mine_steps`` understands. A status in
    ``TRANSIENT_STATUSES`` and a transport failure (timeout, connection error)
    are transient -- the caller retries the SAME idempotent request; every other
    error status, or a body this client cannot parse as JSON, is permanent."""
    import httpx

    try:
        response = request_call()
    except httpx.TransportError as exc:
        raise CorpusTransientFailure(f"transport error: {exc}") from exc
    if response.status_code == 410 and _error_object(response).get("detail") == "job_retired":
        raise CorpusJobRetired(f"410 job_retired from {response.request.url}")
    if response.status_code == 409 and _error_object(response).get("detail") == "job_paused":
        raise CorpusTransientFailure(f"409 job_paused from {response.request.url}")
    if response.status_code in TRANSIENT_STATUSES:
        raise CorpusTransientFailure(f"{response.status_code} from {response.request.url}")
    if response.status_code >= 400:
        raise CorpusPermanentFailure(
            f"{response.status_code} from {response.request.url}",
            status=response.status_code,
            detail=_error_detail(response),
        )
    try:
        return response.json()
    except ValueError as exc:
        raise CorpusPermanentFailure(
            f"non-JSON body from {response.request.url}: {exc}",
            status=response.status_code,
        ) from exc


class CorpusJobSelectionError(Exception):
    """This miner named a job the validator does not serve: pass ``--job-id``
    with one of the listed jobs."""


class HttpCorpusClient:
    """The ``CorpusClient`` over HTTP: the legacy paths (the validator's default
    job), or with ``job_id`` that job's own paths on a validator serving several."""

    def __init__(self, http, *, job_id: str | None = None) -> None:
        self._http = http
        self._job_id = job_id
        # Set from the job's manifest (`submits_scoped`): every other job keeps
        # the legacy /corpus/submit, which deployed corpus controls answer.
        self.scoped_submit = False

    def served_jobs(self) -> list:
        """Every job the validator serves; empty when it cannot say."""
        try:
            return list(self._http.get("/corpus/jobs").json()["jobs"])
        except Exception:
            return []

    def _refuse_unserved(self, response) -> None:
        """A job-scoped 404: a job this validator does not serve, or a validator
        from before several jobs, which has no job-scoped routes at all."""
        if response.status_code != 404:
            return
        if _error_object(response).get("detail") == "corpus_job_not_served":
            raise CorpusJobSelectionError(
                f"the validator does not serve job {self._job_id!r}; it serves {self.served_jobs()}"
            )
        raise CorpusJobSelectionError(
            "this validator serves a single job and has no job-scoped routes; "
            "drop --job-id, or ask its operator to update it"
        )

    def job(self) -> dict:
        if self._job_id is None:
            response = self._http.get("/corpus/job")
        else:
            response = self._http.get(f"/corpus/jobs/{self._job_id}/job")
            self._refuse_unserved(response)
        response.raise_for_status()
        return response.json()

    def contract(self) -> dict:
        """The task contract the validator serves for this job (or its only one)."""
        if self._job_id is None:
            response = self._http.get("/corpus/contract")
        else:
            response = self._http.get(f"/corpus/jobs/{self._job_id}/contract")
            self._refuse_unserved(response)
        response.raise_for_status()
        return response.json()

    def cursor(self, hotkey: str) -> int:
        path = (f"/corpus/cursor/{hotkey}" if self._job_id is None
                else f"/corpus/jobs/{self._job_id}/cursor/{hotkey}")
        return int(issue_corpus_request(lambda: self._http.get(path))["cursor"])

    def submit(self, body: dict) -> dict:
        path = (f"/corpus/jobs/{self._job_id}/submit"
                if self._job_id is not None and self.scoped_submit else "/corpus/submit")
        return issue_corpus_request(lambda: self._http.post(path, json=body))

    def eval_prompts(self) -> bytes:
        """An eval job's prompt lines, which the job's manifest hashes."""
        response = self._http.get(f"/corpus/jobs/{self._job_id}/eval-prompts")
        self._refuse_unserved(response)
        response.raise_for_status()
        return response.content

    def _path(self, tail: str) -> str:
        return (f"/corpus/{tail}" if self._job_id is None
                else f"/corpus/jobs/{self._job_id}/{tail}")

    def next_prompt(self, hotkey: str) -> dict | None:
        """Where this hotkey's walk stands and the slots left there, or None
        from a validator that predates the route (404) or a job that has no
        walk (409)."""
        return _unless_absent(lambda: self._http.get(self._path(f"next/{hotkey}")))

    def skip(self, body: dict) -> dict | None:
        """Step over a full prompt, or None from a validator without the route."""
        return _unless_absent(lambda: self._http.post(self._path("skip"), json=body))

    def open_prompts(self) -> dict | None:
        """Which prompts still have a slot (``slots.parse_open_map`` reads it),
        or None from a validator without the route (404, 405) or a job too
        large for a map (409). Its own short timeout: it is an optimisation,
        and must not hold an episode back as long as a submit may take."""
        return _unless_absent(
            lambda: self._http.get(self._path("open"), timeout=OPEN_READ_TIMEOUT_SECONDS),
            absent=(404, 405, 409))


def _unless_absent(request_call, *, absent=(404, 409)):
    try:
        return issue_corpus_request(request_call)
    except CorpusPermanentFailure as exc:
        # 404: a validator from before the routes. 409: a job with no walk.
        if exc.status in absent:
            return None
        raise


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


def build_skip(*, job, hotkey, cursor, prompt_index, to_cursor, sign) -> dict:
    body = {
        "job_id": job.job_id,
        "miner_hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "to_cursor": to_cursor,
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
               max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES,
               sign_skip=None) -> dict[str, int]:
    """``_mine_steps``, ended cleanly (one log line, no retry) when the
    validator says the job is retired."""
    counts: Counter[str] = Counter()
    try:
        return _mine_steps(job=job, hotkey=hotkey, client=client, generator=generator,
                           tokenizer=tokenizer, render=render, sign=sign, max_steps=max_steps,
                           sleep=sleep, max_consecutive_failures=max_consecutive_failures,
                           sign_skip=sign_skip, counts=counts)
    except CorpusJobRetired as exc:
        counts["job_retired"] += 1
        logger.info("corpus job %s is retired; stopping it (%s)", job.job_id, exc)
        return dict(counts)


def _mine_steps(*, job, hotkey, client, generator, tokenizer, render, sign,
                max_steps: int | None, sleep, max_consecutive_failures: int,
                sign_skip, counts: Counter) -> dict[str, int]:
    """Mine up to ``max_steps`` generations.

    With ``sign_skip`` on a ``miner_walk`` job, each step first asks the
    validator where the walk stands and skips full prompts with a signed skip
    rather than generating for them. A validator without those routes, or one
    that cannot verify a skip, is mined exactly as before.
    """
    retry_kwargs = dict(sleep=sleep, counts=counts, max_consecutive_failures=max_consecutive_failures)
    cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    steps = 0
    consecutive_unreasoned = 0
    skipping = (
        sign_skip is not None
        and getattr(job, "prompt_order", None) == "miner_walk"
        and callable(getattr(client, "next_prompt", None))
        and callable(getattr(client, "skip", None))
    )

    def past_full_prompts(cursor: int) -> int | None:
        """The cursor of the next prompt worth generating for, or None when the
        job is complete. Turns ``skipping`` off for the run on a validator
        that cannot serve it."""
        nonlocal skipping
        stale = 0
        while True:
            position = _retry(lambda: client.next_prompt(hotkey), **retry_kwargs)
            if position is None:
                logger.info("the validator has no next/skip routes: generating for every step")
                skipping = False
                return cursor
            try:
                cursor = int(position["cursor"])
                index = int(position["prompt_index"])
                remaining = int(position["slots_remaining"])
                skip_to = int(position["skip_to"])
                if skip_to <= cursor:
                    raise ValueError(f"skip_to {skip_to} is not past cursor {cursor}")
            except (KeyError, TypeError, ValueError):
                logger.warning("unusable next answer %r: generating for every step", position)
                skipping = False
                return cursor
            if remaining > 0:
                return cursor
            if index != job_walk_index(job, hotkey, cursor):
                # Not our walk: generating lets submit name the disagreement.
                logger.warning("the validator's walk names prompt %d at cursor %d, ours %d",
                               index, cursor, job_walk_index(job, hotkey, cursor))
                return cursor
            # One skip over the whole run of full prompts `next` found.
            body = build_skip(job=job, hotkey=hotkey, cursor=cursor, prompt_index=index,
                              to_cursor=skip_to, sign=sign_skip)
            answer = _retry(lambda: client.skip(body), **retry_kwargs)
            if answer is None:
                logger.info("the validator has no skip route: generating for every step")
                skipping = False
                return cursor
            reason = str(answer.get("reason"))
            if answer.get("skipped"):
                counts["skipped"] += 1
                stale = 0
                continue
            if reason == "job_complete":
                counts["job_complete"] += 1
                return None
            counts[f"skip_{reason}"] += 1
            if reason in _HALT:
                raise CorpusMinerHalted(f"the validator refused this hotkey: {reason}",
                                        counts=dict(counts))
            if reason in _SKIP_REREAD and stale < _MAX_STALE_SKIPS:
                stale += 1
                continue
            if reason not in _SKIP_REREAD:
                logger.warning("corpus skip refused: %s %s; generating for every step",
                               reason, answer.get("detail"))
                skipping = False
            return cursor

    while max_steps is None or steps < max_steps:
        steps += 1
        if skipping:
            cursor = past_full_prompts(cursor)
            if cursor is None:
                break
        # A SOURCE index: the one `render` draws and the route renders again.
        prompt_index = job_walk_index(job, hotkey, cursor)
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
        if reason in _HALT:
            raise CorpusMinerHalted(f"the validator refused this hotkey: {reason}",
                                    counts=dict(counts))
        if answer.get("accepted"):
            cursor += 1
        elif reason in _RESYNC:
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
        else:
            logger.warning("corpus submission refused: %s %s", reason, answer.get("detail"))
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    return dict(counts)


# Room for the rendered prompt beside the job's completion budget.
PROMPT_ALLOWANCE_TOKENS = 8192
# A step runs n sequences; vLLM's default 1,024 overruns a hybrid model's Mamba cache.
MAX_NUM_SEQS = 256


def _has_vision_encoder(checkpoint_dir: str) -> bool:
    import json
    from pathlib import Path

    try:
        config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "vision_config" in config


def submits_scoped(job) -> bool:
    """Whether a job's submissions go to ``/corpus/jobs/{job_id}/submit``: when
    its manifest says ``submit: "scoped"`` (generation orders) or names an eval
    set (eval orders), both served by the order control, which has no legacy
    path. Read from the job, never from the miner's configuration."""
    from reliquary.corpus.job import SUBMIT_SCOPED
    from reliquary.eval.prompt_source import is_eval_source

    return getattr(job, "submit", None) == SUBMIT_SCOPED or is_eval_source(job.prompt_source)


class CorpusContractError(RuntimeError):
    """The validator served a contract that does not describe this job."""


def save_served_contract(contract: dict, job, directory) -> "Path":
    """Keep the contract the validator serves, once it is known to describe the
    job's checkpoint and to carry the toploc proof every submission needs."""
    import json
    import os
    from pathlib import Path
    import tempfile

    from reliquary.corpus.job import JOB_ID_RE

    job_id = getattr(job, "job_id", None)
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise CorpusContractError("the served job has an invalid job id")

    if (contract.get("model_id") != job.checkpoint_repo
            or contract.get("model_revision") != job.checkpoint_revision):
        raise CorpusContractError(
            f"the served contract describes {contract.get('model_id')!r}@"
            f"{contract.get('model_revision')!r}, not the job's "
            f"{job.checkpoint_repo!r}@{job.checkpoint_revision!r}"
        )
    if not any(p.get("scheme") == "toploc-v1" for p in contract.get("proofs") or ()):
        raise CorpusContractError("the served contract carries no toploc proof")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.contract.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=f".{job_id}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(json.dumps(contract, sort_keys=True, separators=(",", ":")))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


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
        # vLLM seeds every engine with 0: two miners sampling one prompt at the
        # same step would submit identical completions, the second refused
        # hash_duplicate. The audit never depends on the seed.
        seed = secrets.randbelow(2**31 - 1) + 1
        # vLLM otherwise reserves the checkpoint's own maximum length, whose KV
        # cache need not fit the card; the job never asks for more than this.
        extra = {"max_model_len": sampling.max_new_tokens + PROMPT_ALLOWANCE_TOKENS,
                 "max_num_seqs": MAX_NUM_SEQS}
        if _has_vision_encoder(checkpoint_dir):
            # The job's prompts are text: skip the vision encoder's profiling.
            extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        self._llm = LLM(model=checkpoint_dir, dtype="bfloat16", enable_prefix_caching=False,
                        seed=seed, **memory, **extra)
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
            # Checked on vLLM 0.30 (H100, 2026-09-24): the stop token is kept
            # at the end of `token_ids`, which `completion_text` relies on.
            stop_token_ids=[eos_token_id], ignore_eos=True,
        )
        self._proof = proof

    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]:
        from vllm.inputs import TokensPrompt

        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * n, self._params)
        return self._generations(outputs, [len(prompt_ids)] * n)

    def generate_many(self, prompts: list[list[int]], ns: list[int]) -> list[list[Generation]]:
        """Several prompts, ``ns[i]`` completions each, decoded in one batch
        (qualification); grouped back per prompt."""
        from vllm.inputs import TokensPrompt

        flat = [ids for ids, n in zip(prompts, ns) for _ in range(n)]
        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in flat],
                                     self._params)
        generations = self._generations(outputs, [len(ids) for ids in flat])
        grouped, start = [], 0
        for n in ns:
            grouped.append(generations[start:start + n])
            start += n
        return grouped

    def _generations(self, outputs, prompt_lengths: list[int]) -> list[Generation]:
        import base64

        from reliquary.miner.vllm_hidden_capture import completion_rows
        from reliquary.protocol.toploc_proof import build_chunk_proofs

        generations = []
        for index, (output, prompt_length) in enumerate(zip(outputs, prompt_lengths)):
            tokens = list(output.outputs[0].token_ids)
            # `pop`, not `for_request`: this generator lives for the whole
            # mining run, and a request's rows are never read again after its
            # proof is built, so keeping them would grow CPU memory unbounded.
            try:
                rows = completion_rows(self._capture.pop(output.request_id),
                                       prompt_length, prompt_length + len(tokens))
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
