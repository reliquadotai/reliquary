"""The agentic miner's signed-sandbox mode (plan 3). No Docker on the miner: per episode,
ask the validator for a session token (a request signed by the hotkey for this
validator and route), open the episode on the machine the validator named, run
verifiers' bash harness through `reliquary_sandbox_verifiers.SandboxEpisodeRunner`
against the local generate endpoint (tokens and proofs come from the engine's session
log, as in replay mode), and return what to submit: the gateway's transcript and the
graded state as the final diff.

The miner never submits what it already knows the validator refuses. An episode whose
final is not `graded`, whose run errored (a gateway close raised as `EpisodeClosed`
ends as a verifiers `TaskError` trace), whose state is not UTF-8, or whose transcript
fails the validator's own checks (`transcript_refusal`: `verify_transcript` with every
`Expected` field, record 0's tools and env package, the grading deadline) is reported
at once (`close`), so its reservation and the hotkey's caps are released. The built
trajectory then passes `signed_trajectory_precheck` (spans, size, §5.C, §5.D) before
it is signed; the mining loop reports it when it does not.

Every 429 and 503 is honoured: a throttled open holds this hotkey's next open until its
`Retry-After` has passed; a throttled close (`close_busy`, a stale directory) is resent
after it. Tokens and signed open bodies are bearer secrets: nothing here logs them, and
errors carry exception types and statuses, never a request body."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from typing import Any

from reliquary.miner.agentic_episode import EpisodeResult
from reliquary.miner.corpus_miner import retry_after_seconds
from reliquary.protocol.sandbox_session import (
    SessionRefused, sandbox_close_path, sandbox_open_path,
)
from reliquary.protocol.signatures import build_sandbox_close_binding, build_sandbox_open_binding

logger = logging.getLogger(__name__)

OPEN_WINDOW_S = 900
GRADING_GRACE_S = 1800
SUBMIT_MARGIN_S = 120
SS58_FORMAT = 42
THROTTLED = frozenset({429, 503})
DEFAULT_RETRY_AFTER_S = 10.0
"""A 429/503 without a usable `Retry-After` (the validator always sends one)."""
OPEN_ATTEMPTS = 3
"""Sends of one signed open after transport errors (idempotent per request id)."""
CLOSE_ATTEMPTS = 6
CANCELLED_CLOSE_S = 30.0
"""How long a cancelled episode spends reporting its session before it propagates."""
MAX_ERROR_CHARS = 300


def ss58_format_42(address: Any) -> str | None:
    """`address` as an ss58 format-42 account address, or None when it does not decode."""
    if not isinstance(address, str) or not address:
        return None
    try:
        from scalecodec.utils.ss58 import ss58_decode, ss58_encode

        public_key = ss58_decode(address)
        if not isinstance(public_key, str) or len(public_key) != 64:
            return None
        return ss58_encode(public_key, SS58_FORMAT)
    except Exception:
        return None


def _compact(body: Mapping[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class HttpSandboxSessions:
    """The validator's session routes. `validator_hotkey` (ss58) and `prefix` are the
    audience every request is signed for."""

    def __init__(self, http, *, validator_hotkey: str, prefix: str = "/corpus") -> None:
        audience = ss58_format_42(validator_hotkey)
        if audience is None:
            raise ValueError("validator_hotkey must be the validator's ss58 hotkey address")
        self._http, self.prefix, self.validator_hotkey = http, prefix, audience

    def _post(self, path: str, body: dict) -> dict:
        response = self._http.post(path, content=_compact(body),
                                   headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json()
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        retry = retry_after_seconds(response.headers.get("Retry-After"))
        if retry is None and response.status_code in THROTTLED:
            retry = DEFAULT_RETRY_AFTER_S
        raise SessionRefused(str(payload.get("reason") or f"http_{response.status_code}"),
                             payload.get("detail") if isinstance(payload.get("detail"), dict)
                             else {},
                             retry_after=retry, status=response.status_code)

    def open(self, body: dict) -> dict:
        return self._post(sandbox_open_path(self.prefix), body)

    def close(self, session_id: str, body: dict) -> dict:
        return self._post(sandbox_close_path(self.prefix, session_id), body)


def signed_open_request(*, hotkey: str, job_id: str, prompt_index: int,
                        sign_binding: Callable[[bytes], str], now: float, request_id: str,
                        validator_hotkey: str, path: str) -> dict:
    body = {"miner_hotkey": hotkey, "request_id": request_id, "at": int(now),
            "engagement": {"kind": "corpus", "job_id": job_id, "prompt_index": int(prompt_index)},
            "signature": ""}
    body["signature"] = sign_binding(build_sandbox_open_binding(
        body, validator_hotkey=validator_hotkey, path=path))
    return body


def signed_close_request(*, hotkey: str, session_id: str, reason: str, transcript: dict | None,
                         sign_binding: Callable[[bytes], str], now: float, request_id: str,
                         validator_hotkey: str, path: str) -> dict:
    body = {"miner_hotkey": hotkey, "request_id": request_id, "at": int(now),
            "session_id": session_id, "reason": reason, "transcript": transcript, "signature": ""}
    body["signature"] = sign_binding(build_sandbox_close_binding(
        body, validator_hotkey=validator_hotkey, path=path))
    return body


class _NoKeys:
    """The miner holds no machine keys: `verify_transcript` then flags `unknown_key`
    (ignored here) and checks everything else."""

    def public_key(self, machine_id: str, key_id: str, at: int):
        return None


def _claims_of(token: Any):
    """The claims of the token this validator just granted (its signature is the
    validator's to check; the miner has no validator key)."""
    from reliquary_sandbox.attest import RecordError, SessionClaims

    if not isinstance(token, Mapping) or set(token) != {"v", "claims", "key_id", "signature"}:
        raise RecordError("not a session token")
    return SessionClaims.from_dict(token["claims"])


def transcript_refusal(transcript: Any, *, job, hotkey: str, index: int,
                       now: float) -> tuple[str, dict] | None:
    """The signed intake's transcript checks the miner can run (all but the machine
    and validator signatures): `verify_transcript` with every `Expected` field and a
    graded final, the grading deadline, record 0's tools version, tools and env
    package. Returns `(reason, detail)` or None."""
    from reliquary_sandbox.attest import Expected, Reason, verify_transcript
    from reliquary_sandbox.observation import TOOLS_VERSIONS

    from reliquary.corpus.job import sandbox_split
    from reliquary.corpus.signed_reasons import (
        REASON_SANDBOX_EXPIRED, REASON_SANDBOX_TRANSCRIPT, corpus_engagement,
    )
    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES

    if not isinstance(transcript, Mapping):
        return "malformed_submission", {"transcript": None}
    size = len(_compact(transcript))
    if size > MAX_TRANSCRIPT_BYTES:
        return "trajectory_too_large", {"transcript_bytes": size}
    episode = job.episode
    spec = episode.sandbox
    expected = Expected(hotkey=hotkey, engagement=corpus_engagement(job.job_id, index),
                        env=spec.env, split=sandbox_split(episode), index=int(index),
                        checkpoint=job.checkpoint_sha256, require_graded=True)
    result = verify_transcript(transcript, _NoKeys(), _claims_of, expected)
    reasons = [reason.value for reason in result.reasons if reason is not Reason.UNKNOWN_KEY]
    if reasons:
        return REASON_SANDBOX_TRANSCRIPT, {"reasons": reasons}
    deadline = result.claims.expires_at + GRADING_GRACE_S
    if now > deadline:
        return REASON_SANDBOX_EXPIRED, {"now": int(now), "deadline": deadline}
    opened = result.open
    if opened.tools_version not in TOOLS_VERSIONS:
        return REASON_SANDBOX_TRANSCRIPT, {"record0": "tools_version", "got": opened.tools_version}
    if tuple(opened.tools) != tuple(spec.tools):
        return REASON_SANDBOX_TRANSCRIPT, {"record0": "tools", "got": list(opened.tools)}
    if opened.env_package != spec.env_package:
        return REASON_SANDBOX_TRANSCRIPT, {"record0": "env_package", "got": opened.env_package}
    return None


def _describe(exc: BaseException) -> str:
    """An error for logs and counts: the type and, bounded, the message; a gateway
    refusal by its status (its detail can carry signed records)."""
    from reliquary_sandbox.episode_client import EpisodeError

    if isinstance(exc, EpisodeError):
        detail = exc.detail if isinstance(exc.detail, str) else ""
        return f"{type(exc).__name__}: {exc.status} {detail[:MAX_ERROR_CHARS]}".rstrip()
    return f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}"


class SignedSweEpisodeRunner:
    """`SweEpisodeRunner`'s interface (`run(index, on_session)`, `deadline(index)`, an
    async context) for a signed-sandbox job, for one hotkey. `sessions` is the
    validator's session client (`HttpSandboxSessions`): its `validator_hotkey` and
    `prefix` are the audience every request is signed for."""

    def __init__(self, *, job, hotkey: str, sign_binding: Callable[[bytes], str], sessions,
                 model_name: str, renderer_model_dir: str, generate_url: str, sampling,
                 harness_env: dict | None = None, source=None, runner_factory=None,
                 clock: Callable[[], float] = time.time,
                 new_request_id: Callable[[], str] = lambda: uuid.uuid4().hex,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        from reliquary.corpus.job import sandbox_split
        from reliquary.environment.agentic_swe import SignedSweSource

        if ss58_format_42(hotkey) != hotkey:
            raise ValueError("a signed-sandbox miner's hotkey must be an ss58 format-42 address "
                             "(the validator compares the token's hotkey to it)")
        validator_hotkey = ss58_format_42(getattr(sessions, "validator_hotkey", None))
        if validator_hotkey is None:
            raise ValueError("the session client names no ss58 validator hotkey")
        self._job, self._hotkey, self._sign = job, hotkey, sign_binding
        self._sessions = sessions
        self._validator = validator_hotkey
        self._prefix = getattr(sessions, "prefix", "/corpus")
        self._model_name, self._renderer_dir = model_name, renderer_model_dir
        self._generate_url, self._sampling = generate_url, sampling
        self._harness_env = dict(harness_env or {})
        self._source = source or SignedSweSource(sandbox_split(job.episode))
        self._runner_factory = runner_factory
        self._clock, self._new_request_id, self._sleep = clock, new_request_id, sleep
        self._ctx = None
        self._runners: dict[str, object] = {}
        self._stack = AsyncExitStack()
        self._not_before = 0.0          # event-loop time before which no open is sent

    async def __aenter__(self) -> SignedSweEpisodeRunner:
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def deadline(self, index: int) -> float:
        """Open, the agent's wall budget, grading inside the verification window, submit."""
        return float(OPEN_WINDOW_S + self._job.episode.sandbox.budgets.wall_s
                     + GRADING_GRACE_S + SUBMIT_MARGIN_S)

    def _context(self):
        if self._ctx is None:
            from renderers.configs import Qwen38RendererConfig
            from verifiers.v1.clients import ModelContext
            from verifiers.v1.configs.client import TrainClientConfig
            from verifiers.v1.types import Sampling

            self._ctx = ModelContext(
                model=self._model_name,
                client=TrainClientConfig(base_url=f"{self._generate_url.rstrip('/')}/v1",
                                         api_key_var="RELIQUARY_GENERATE_KEY",
                                         renderer=Qwen38RendererConfig(),
                                         renderer_model_name=self._renderer_dir),
                sampling=Sampling(temperature=self._sampling.temperature,
                                  top_p=self._sampling.top_p,
                                  max_tokens=self._job.episode.max_tokens_per_turn))
        return self._ctx

    def _default_runner(self, url: str):
        from reliquary_sandbox_verifiers import SandboxEpisodeRunner

        return SandboxEpisodeRunner(ctx=self._context(), gateway_url=url,
                                    max_turns=self._job.episode.max_turns,
                                    harness_env=self._harness_env)

    async def _runner(self, url: str):
        if url not in self._runners:
            factory = self._runner_factory or self._default_runner
            self._runners[url] = await self._stack.enter_async_context(factory(url))
        return self._runners[url]

    def _hold_opens(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        self._not_before = max(self._not_before, loop.time() + max(1.0, float(seconds)))

    async def _open(self, index: int) -> dict:
        wait = self._not_before - asyncio.get_running_loop().time()
        if wait > 0:
            await self._sleep(wait)
        path = sandbox_open_path(self._prefix)
        body = signed_open_request(hotkey=self._hotkey, job_id=self._job.job_id,
                                   prompt_index=index, sign_binding=self._sign,
                                   now=self._clock(), request_id=self._new_request_id(),
                                   validator_hotkey=self._validator, path=path)
        for attempt in range(OPEN_ATTEMPTS):
            try:
                return await asyncio.to_thread(self._sessions.open, body)
            except SessionRefused as refused:
                if refused.retry_after is not None or refused.status in THROTTLED:
                    self._hold_opens(refused.retry_after or DEFAULT_RETRY_AFTER_S)
                raise
            except Exception as exc:
                # A lost answer: the same signed request returns the same grant.
                logger.warning("sandbox session open for prompt %d failed (%s), attempt %d",
                               index, type(exc).__name__, attempt + 1)
                if attempt + 1 < OPEN_ATTEMPTS:
                    await self._sleep(2.0 ** attempt)
        self._hold_opens(DEFAULT_RETRY_AFTER_S)
        raise SessionRefused("validator_unreachable", retry_after=DEFAULT_RETRY_AFTER_S)

    async def _close(self, session_id: str, reason: str, transcript: dict | None) -> None:
        """Report how the session ended; a throttled report is resent after its
        `Retry-After`. Never raises."""
        path = sandbox_close_path(self._prefix, session_id)
        for attempt in range(CLOSE_ATTEMPTS):
            body = signed_close_request(hotkey=self._hotkey, session_id=session_id,
                                        reason=reason, transcript=transcript,
                                        sign_binding=self._sign, now=self._clock(),
                                        request_id=self._new_request_id(),
                                        validator_hotkey=self._validator, path=path)
            last = attempt + 1 == CLOSE_ATTEMPTS
            try:
                answer = await asyncio.to_thread(self._sessions.close, session_id, body)
            except SessionRefused as refused:
                if refused.status in THROTTLED and not last:
                    await self._sleep(refused.retry_after or DEFAULT_RETRY_AFTER_S)
                    continue
                logger.warning("reporting sandbox session %s (%s) refused: %s", session_id,
                               reason, refused.reason)
                return
            except Exception as exc:
                if not last:
                    await self._sleep(2.0 ** attempt)
                    continue
                logger.warning("reporting sandbox session %s (%s) failed: %s", session_id,
                               reason, type(exc).__name__)
                return
            logger.info("sandbox session %s reported %s: %s", session_id, reason,
                        answer.get("state") if isinstance(answer, Mapping) else None)
            return

    async def run(self, index: int, on_session=None) -> EpisodeResult:
        grant = await self._open(index)      # raises SessionRefused
        session_id = str(grant["session_id"])
        logger.info("sandbox session %s for prompt %d on %s", session_id, index,
                    grant["gateway_url"])
        try:
            runner = await self._runner(grant["gateway_url"])
            on_trace = (lambda trace: on_session(trace.id)) if on_session is not None else None
            result = await runner.run(token=grant["token"], prompt=self._source.prompt(index),
                                      on_trace=on_trace)
        except Exception as exc:  # no episode result, no transcript
            await self._close(session_id, "open_failed", None)
            return EpisodeResult(None, "", None, False, None, error=_describe(exc))
        except BaseException:
            # Cancelled (the episode's deadline, the miner stopping): report it, briefly.
            try:
                async with asyncio.timeout(CANCELLED_CLOSE_S):
                    await self._close(session_id, "open_failed", None)
            except Exception:
                pass
            raise
        trace = result.trace
        trace_id = getattr(trace, "id", None)
        stop = getattr(trace, "stop_condition", None)
        final = (result.final or {}).get("body") or {}
        status = final.get("status")
        if result.error or status != "graded":
            # Unpaid: not graded, or the run did not complete (a gateway close raised
            # as EpisodeClosed is a TaskError trace).
            await self._close(session_id, "final", result.transcript)
            return EpisodeResult(trace_id, "", stop, False, None, error=result.error or (
                f"the sandbox episode ended {status} ({final.get('reason')})"))
        try:
            final_diff = (result.state or b"").decode("utf-8")
        except UnicodeDecodeError:
            await self._close(session_id, "final", result.transcript)
            return EpisodeResult(trace_id, "", stop, False, None,
                                 error="the graded state is not UTF-8: no diff can match it")
        refusal = transcript_refusal(result.transcript, job=self._job, hotkey=self._hotkey,
                                     index=index, now=self._clock())
        if refusal is not None:
            await self._close(session_id, "final", result.transcript)
            reason, detail = refusal
            return EpisodeResult(trace_id, "", stop, False, None,
                                 error=f"the validator would refuse the transcript: {reason} "
                                       f"{detail}")
        transcript = result.transcript

        async def release() -> None:
            await self._close(session_id, "final", transcript)

        return EpisodeResult(trace_id, final_diff, stop, True, final.get("reward"),
                             transcript=transcript, release=release)


def signed_trajectory_precheck(renderer, *, max_turns: int):
    """The validator's own refusals of a signed trajectory, run before it is signed:
    spans, transcript size, §5.C and §5.D. Returns `(reason, detail)` or None.
    (`transcript_refusal` ran on the transcript when the episode ended.)"""
    from reliquary.corpus.checks import check_turn_spans
    from reliquary.corpus.signed_parse import parse_signed_trajectory, signed_records
    from reliquary.corpus.signed_reasons import (
        REASON_SANDBOX_STATE_MISMATCH, REASON_SANDBOX_TRANSCRIPT, state_matches,
    )
    from reliquary.corpus.trajectory_parse import TrajectoryRefused
    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES

    def precheck(built):
        spans = [tuple(span) for span in built.spans]
        result = check_turn_spans(spans, len(built.tokens), max_turns)
        if not result.ok:
            return result.reason or "bad_turns", dict(result.detail)
        if built.transcript is None:
            return "malformed_submission", {"transcript": None}
        size = len(_compact(built.transcript))
        if size > MAX_TRANSCRIPT_BYTES:
            return "trajectory_too_large", {"transcript_bytes": size}
        try:
            found = signed_records(built.transcript)
        except (KeyError, TypeError, ValueError) as exc:
            return REASON_SANDBOX_TRANSCRIPT, {"why": type(exc).__name__}
        try:
            parse_signed_trajectory(renderer, prompt_ids=list(built.prompt_ids),
                                    tokens=list(built.tokens), spans=spans, stop=built.stop,
                                    max_turns=max_turns, calls=found.calls, offered=found.tools,
                                    final=found.final)
        except TrajectoryRefused as refused:
            return refused.reason, dict(refused.detail)
        if not state_matches(found.final.state_sha256, built.final_diff):
            return REASON_SANDBOX_STATE_MISMATCH, {}
        return None

    return precheck


__all__ = ["HttpSandboxSessions", "SignedSweEpisodeRunner", "signed_close_request",
           "signed_open_request", "signed_trajectory_precheck", "ss58_format_42",
           "transcript_refusal"]
