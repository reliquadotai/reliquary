"""The agentic corpus miner: episodes in parallel, one signed trajectory per
finished episode (spec §5 N1, N2)."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from reliquary.corpus.job import PROMPT_ORDER_FREE
from reliquary.corpus.slots import OpenMap, parse_open_map
from reliquary.corpus.trajectory import BuiltTrajectory, TrajectoryUnbuildable, build_trajectory
from reliquary.corpus.walk import job_walk_index
from reliquary.miner.corpus_miner import _HALT, CorpusJobRetired, CorpusMinerHalted, _retry
from reliquary.protocol.sandbox_session import SessionRefused

logger = logging.getLogger(__name__)

# Session refusals no later request of this identity can clear: it stops. Waiting
# (Retry-After, backoff) is the signed runner's: it holds the hotkey's next open.
SESSION_HALT = frozenset({
    "bad_signature", "hotkey_not_registered", "miner_banned", "job_not_signed",
    "job_complete", "stale_request", "malformed_request", "engagement_kind_unsupported",
    "validator_unavailable",
})
SESSION_HALT_HINTS = {
    "bad_signature": " (is --validator-hotkey this validator's hotkey?)",
    "stale_request": " (is this machine's clock right?)",
}
# Reporting unsubmitted sessions when an episode crashes or is cancelled.
REPORT_TIMEOUT_S = 30.0
# How old the job's open map may be when an episode starts; older, it is read again.
OPEN_MAX_AGE_SECONDS = 30.0


def _clock() -> float:
    return time.monotonic()


# Walk positions checked against the map before the launcher yields to the loop.
_OPEN_SCAN = 1024


@dataclass
class Identity:
    hotkey: str
    sign: Callable[[dict], str]
    # A test hook (the end-to-end forgeries): rewrites a built trajectory before it is signed.
    transform: Callable[[BuiltTrajectory, int], BuiltTrajectory] | None = None
    # Episodes to run; None: until the job completes.
    episodes: int | None = None
    # Extra environment for the bash harness program (and the commands it runs).
    harness_env: dict | None = None
    # Signed-sandbox jobs: signs a session request's binding with the hotkey (hex).
    sign_binding: Callable[[bytes], str] | None = None


def harness_key(harness_env: dict | None) -> tuple:
    return tuple(sorted((harness_env or {}).items()))


def build_trajectory_submission(*, job, hotkey, cursor, prompt_index, rendered_prompt,
                                trajectory: BuiltTrajectory, sign) -> dict:
    body = {
        "job_id": job.job_id,
        "miner_hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "checkpoint_sha256": job.checkpoint_sha256,
        "rendered_prompt": rendered_prompt,
        "completions": [],
        "trajectory": trajectory.wire(),
        "signature": "",
    }
    body["signature"] = sign(body)
    return body


def trajectory_precheck(renderer, *, max_turns: int) -> Callable[[BuiltTrajectory], tuple[str, dict] | None]:
    """The validator's own refusals, run on a built trajectory before it is
    signed (ruling P14): its span check, ``parse_trajectory`` through the
    same pinned renderer and the grade lease bounds (ruling P23). Returns
    ``(reason, detail)`` or None."""
    from reliquary.corpus.checks import REASON_TRAJECTORY_TOO_LARGE, check_turn_spans
    from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
    from reliquary.validator.corpus_grade_protocol import grade_item_bounds_refusal

    def precheck(built: BuiltTrajectory) -> tuple[str, dict] | None:
        spans = [tuple(span) for span in built.spans]
        result = check_turn_spans(spans, len(built.tokens), max_turns)
        if not result.ok:
            return result.reason or "bad_turns", dict(result.detail)
        try:
            parsed = parse_trajectory(renderer, prompt_ids=list(built.prompt_ids),
                                      tokens=list(built.tokens), spans=spans, stop=built.stop,
                                      max_turns=max_turns)
        except TrajectoryRefused as refused:
            return refused.reason, dict(refused.detail)
        # The intake's lease bounds (ruling P23 d), so no slot is lost to them.
        too_large = grade_item_bounds_refusal(parsed.actions, built.final_diff)
        if too_large is not None:
            return REASON_TRAJECTORY_TOO_LARGE, too_large
        return None

    return precheck


def docker_storage_warning(*, refusal: Callable[[], str | None] | None = None) -> str | None:
    """A startup warning when this miner's Docker storage is not on xfs (F6
    M3, ruling P20): executors replay on xfs, so an ext4 directory order
    mismatches every cut ``find``/``grep -r`` listing and spends the
    episode's replay tolerance. A warning, never a refusal."""
    if refusal is None:
        from reliquary.validator.corpus_grade_executor import docker_storage_refusal as refusal
    why = refusal()
    if why is None:
        return None
    return (f"warning: {why}. Grade executors replay on xfs: on this storage your cut "
            f"`find`/`grep -r` listings mismatch theirs and spend each episode's replay "
            f"tolerance (honest episodes can be voided). Put Docker's storage on xfs.")


class OpenPrompts:
    """The job's open prompts (``GET .../open``), shared by every identity and
    read again once ``max_age`` old, so an episode (minutes of GPU and sandbox)
    is not started on a prompt that would answer ``prompt_full``.

    Only a filter: an unknown prompt is open. A validator without the route, a
    map this miner cannot read or a failed read all leave the miner as it was
    before the route existed. Used on a ``free`` job only: there the validator
    does not check the cursor, so walk positions may be passed over.
    """

    def __init__(self, job, client, *, max_age: float = OPEN_MAX_AGE_SECONDS) -> None:
        read = getattr(client, "open_prompts", None)
        self._job = job
        self._read = read if job.prompt_order == PROMPT_ORDER_FREE else None
        self._max_age = max_age
        self._map: OpenMap | None = None
        self._read_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def exhausted(self) -> bool:
        """The validator said no prompt has a slot left."""
        return self._map is not None and self._map.open_count == 0

    def is_open(self, prompt_index: int) -> bool:
        return self._map is None or self._map.is_open(prompt_index)

    async def refresh(self, counts: Counter) -> None:
        """Read the map if there is none younger than ``max_age``. Raises only
        ``CorpusJobRetired``: every other failure leaves no map (unfiltered)."""
        if self._read is None:
            return
        async with self._lock:
            if self._read is None:
                return
            if self._read_at is not None and _clock() - self._read_at < self._max_age:
                return
            try:
                body = await asyncio.to_thread(self._read)
            except CorpusJobRetired:
                raise
            except Exception as exc:
                counts["open_read_failed"] += 1
                logger.warning("the open prompts could not be read (%r): not filtering", exc)
                self._map, self._read_at = None, _clock()
                return
            self._read_at = _clock()
            if body is None:
                logger.info("the validator does not serve the open prompts: not filtering")
                self._map = self._read = None
                return
            try:
                parsed = parse_open_map(body)
                served = (body.get("job_id"), parsed.prompt_start, parsed.prompt_count)
                mine = (self._job.job_id, self._job.prompt_start, self._job.prompt_count)
                if served != mine:
                    raise ValueError(f"it describes {served}, this miner mines {mine}")
            except ValueError as exc:
                counts["open_map_unusable"] += 1
                logger.warning("the open prompts are unusable (%s): not filtering", exc)
                self._map = self._read = None
                return
            self._map = parsed


class _SubmitDeadline(Exception):
    """The next retry of a submission would land past its `submit_by`."""


def _submit(client, body: dict, counts: Counter, submit_by: float | None = None) -> dict:
    """Submit, retrying; with `submit_by`, a retry that would wait past it stops
    (`_SubmitDeadline`): the validator would refuse it anyway."""
    def sleep(seconds: float) -> None:
        if submit_by is not None and time.time() + seconds > submit_by:
            raise _SubmitDeadline()
        time.sleep(seconds)

    return _retry(lambda: client.submit(body), sleep=sleep, counts=counts,
                  max_consecutive_failures=5)


async def _mine_identity(*, job, identity: Identity, client, engine, runner, decode,
                         slots: asyncio.Semaphore, counts: Counter, stop: asyncio.Event,
                         precheck=None, open_prompts: OpenPrompts | None = None) -> None:
    tasks: set[asyncio.Task] = set()
    if open_prompts is None:
        open_prompts = OpenPrompts(job, client)
    tag = identity.hotkey[:8]

    def halt() -> None:
        """The identity is done (job complete, retired, halted): episodes still
        running would only submit into a refusal, so they are cancelled."""
        stop.set()
        current = asyncio.current_task()
        for task in tasks:
            if task is not current and not task.done():
                task.cancel()

    async def report(pending: list, prompt_index: int) -> None:
        """Report the signed sessions in `pending` (not submitted), each once."""
        while pending:
            release = pending.pop()
            try:
                await release()
            except Exception:
                logger.warning("reporting the sandbox session of episode %d failed",
                               prompt_index)

    async def bounded_report(pending: list, prompt_index: int) -> None:
        try:
            async with asyncio.timeout(REPORT_TIMEOUT_S):
                await report(pending, prompt_index)
        except TimeoutError:
            logger.warning("reporting the sandbox session of episode %d timed out", prompt_index)

    async def episode(cursor: int, prompt_index: int, sessions: list[str],
                      pending: list) -> None:
        deadline = getattr(runner, "deadline", None)
        limit = deadline(prompt_index) if deadline is not None else None
        try:
            async with asyncio.timeout(limit):
                result = await runner.run(prompt_index, on_session=sessions.append)
        except TimeoutError:
            counts["episode_timeout"] += 1
            logger.warning("episode %d of %s passed its %s s deadline", prompt_index, tag, limit)
            return
        except SessionRefused as refused:
            counts[f"session_refused:{refused.reason}"] += 1
            logger.warning("episode %d of %s: no sandbox session (%s)", prompt_index, tag,
                           refused.reason)
            if refused.reason in SESSION_HALT:
                counts["halted"] += 1
                logger.error("%s stops: the validator refuses its sandbox sessions (%s)%s", tag,
                             refused.reason, SESSION_HALT_HINTS.get(refused.reason, ""))
                halt()
            return

        if result.release is not None:
            # Until it is submitted and accepted; `one` reports it if this episode crashes.
            pending.append(result.release)

        async def unsubmitted() -> None:
            """A signed session that will not be submitted is reported, so its
            reservation ends now rather than at its deadline."""
            await report(pending, prompt_index)

        if result.session_id:
            sessions.append(result.session_id)
        # Taken whatever the outcome, so a failed episode's log does not linger.
        session = engine.take_session(result.session_id) if result.session_id else None
        if not result.ok or session is None:
            counts["episode_failed"] += 1
            logger.warning("episode %d not submitted: %s", prompt_index,
                           result.error or "no session was logged")
            await unsubmitted()
            return
        if not session.linear:
            counts["not_linear"] += 1
            await unsubmitted()
            return
        try:
            built = build_trajectory(session.turns, final_diff=result.final_diff, stop=result.stop)
            if result.transcript is not None:
                built = dataclasses.replace(built, transcript=result.transcript)
        except TrajectoryUnbuildable as exc:
            counts["unbuildable"] += 1
            logger.warning("episode %d not submitted: %s", prompt_index, exc)
            await unsubmitted()
            return
        # Before the forgery hook: the hook tests the validator, the check the honest run.
        if precheck is None:
            refusal = None
        elif getattr(precheck, "takes_prompt_index", False):
            refusal = await asyncio.to_thread(precheck, built, prompt_index)
        else:
            refusal = await asyncio.to_thread(precheck, built)
        # A runner that knows its own validator checks (a suite episode's replay).
        runner_precheck = getattr(runner, "precheck", None)
        if refusal is None and runner_precheck is not None:
            refusal = await asyncio.to_thread(runner_precheck, built, prompt_index)
        if refusal is not None:
            reason, detail = refusal
            counts["precheck_refused"] += 1
            counts[f"precheck_refused:{reason}"] += 1
            logger.warning("episode %d of %s not submitted, the validator would refuse it: %s %s",
                           prompt_index, tag, reason, detail)
            await unsubmitted()
            return
        if identity.transform is not None:
            built = identity.transform(built, prompt_index)
        body = build_trajectory_submission(
            job=job, hotkey=identity.hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=decode(list(built.prompt_ids)), trajectory=built, sign=identity.sign)
        if result.submit_by is not None and time.time() > result.submit_by:
            counts["submit_deadline_passed"] += 1
            logger.warning("episode %d of %s not submitted: past the validator's deadline",
                           prompt_index, tag)
            await unsubmitted()
            return
        try:
            answer = await asyncio.to_thread(_submit, client, body, counts, result.submit_by)
        except _SubmitDeadline:
            counts["submit_deadline_passed"] += 1
            logger.warning("episode %d of %s not submitted: its retries reached the validator's "
                           "deadline", prompt_index, tag)
            await unsubmitted()
            return
        except (CorpusJobRetired, CorpusMinerHalted) as exc:
            counts["halted"] += 1
            logger.error("%s stops: %s", tag, exc)
            await unsubmitted()
            halt()
            return
        reason = str(answer.get("reason"))
        if reason == "accepted":
            pending.clear()          # the validator ends an accepted session's reservation
            if result.submitted is not None:
                result.submitted()
        else:
            await unsubmitted()
        counts[reason] += 1
        # The reward SweEnv.finalize graded after the rollout; verifiers' own
        # "rollout done: reward=..." line is logged before that grading.
        logger.info("episode %d of %s: %s %s(reward %s)", prompt_index, tag, reason,
                    f"{answer['detail']} " if answer.get("detail") else "", result.reward)
        if reason == "job_complete" or reason in _HALT:
            halt()

    async def one(cursor: int) -> None:
        """One episode, isolated: whatever it raises is counted and logged
        here, and its generate sessions are dropped, never its siblings'."""
        prompt_index = job_walk_index(job, identity.hotkey, cursor)
        sessions: list[str] = []
        pending: list = []
        try:
            await episode(cursor, prompt_index, sessions, pending)
        except asyncio.CancelledError:
            counts["episode_cancelled"] += 1
            await bounded_report(pending, prompt_index)
            raise
        except Exception:
            counts["episode_crashed"] += 1
            logger.exception("episode %d of %s crashed", prompt_index, tag)
            await bounded_report(pending, prompt_index)
        finally:
            for session_id in dict.fromkeys(sessions):
                try:
                    engine.drop_session(session_id)
                except Exception:
                    logger.exception("dropping session %s failed", session_id)

    # `cursor` is this hotkey's walk position, and the validator of a free job
    # (every episode job) does not check it: positions whose prompt is full
    # are passed over, in the walk's own order. `started` counts episodes.
    cursor = started = 0
    while not stop.is_set() and (identity.episodes is None or started < identity.episodes):
        await slots.acquire()
        if stop.is_set():
            slots.release()
            break
        if not getattr(engine, "healthy", True):
            # Every turn would fail fast: starting episodes would only burn boxes.
            counts["engine_unhealthy"] += 1
            logger.error("the generate engine is unhealthy: %s stops", tag)
            stop.set()
            slots.release()
            break
        # After the wait for a slot, which can last a whole episode: the map
        # this start is decided on is at most OPEN_MAX_AGE_SECONDS old.
        try:
            await open_prompts.refresh(counts)
        except CorpusJobRetired as exc:
            counts["halted"] += 1
            logger.error("%s stops: %s", tag, exc)
            halt()
            slots.release()
            break
        if open_prompts.exhausted:
            # Running episodes finish and submit; each learns `job_complete` itself.
            counts["no_open_prompt"] += 1
            logger.info("no prompt of %s has a slot left: %s starts no more episodes",
                        job.job_id, tag)
            stop.set()
            slots.release()
            break
        for _ in range(_OPEN_SCAN):
            if open_prompts.is_open(job_walk_index(job, identity.hotkey, cursor)):
                break
            cursor += 1
            counts["skipped_full"] += 1
        else:
            # A long run of full prompts: let the loop breathe, then go on.
            slots.release()
            await asyncio.sleep(0)
            continue
        task = asyncio.create_task(one(cursor))
        task.add_done_callback(lambda _t: slots.release())
        tasks.add(task)
        cursor += 1
        started += 1
    # return_exceptions: a cancelled or crashed episode never cancels its siblings.
    await asyncio.gather(*tasks, return_exceptions=True)


async def mine_agentic(*, job, identities, client, engine, runners, decode,
                       concurrency: int, precheck=None,
                       runner_for: Callable[[Identity], object] | None = None) -> dict[str, Counter]:
    """Every identity mines until its episode count or the job's end; at most
    ``concurrency`` episodes run at once across them. ``precheck`` (see
    ``trajectory_precheck``) drops what the validator would refuse, unsigned.
    ``runner_for`` (signed-sandbox jobs) gives each identity its own runner, in
    place of ``runners`` keyed by harness environment."""
    slots = asyncio.Semaphore(concurrency)
    open_prompts = OpenPrompts(job, client)
    counts = {identity.hotkey: Counter() for identity in identities}
    stops = {identity.hotkey: asyncio.Event() for identity in identities}
    outcomes = await asyncio.gather(*(
        _mine_identity(job=job, identity=identity, client=client, engine=engine,
                       runner=(runner_for(identity) if runner_for is not None
                               else runners[harness_key(identity.harness_env)]),
                       decode=decode,
                       slots=slots, counts=counts[identity.hotkey], stop=stops[identity.hotkey],
                       precheck=precheck, open_prompts=open_prompts)
        for identity in identities), return_exceptions=True)
    for identity, outcome in zip(identities, outcomes):
        if isinstance(outcome, BaseException):
            counts[identity.hotkey]["identity_crashed"] += 1
            logger.error("mining for %s stopped: %r", identity.hotkey[:8], outcome)
    return counts


async def serve_loopback(app, port: int):
    """Start ``app`` on 127.0.0.1 only (the generate endpoint has no auth);
    returns the uvicorn server and its serving task once it accepts requests."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        if serving.done():
            serving.result()  # raises the reason it never started
            raise RuntimeError(f"the generate endpoint did not start on 127.0.0.1:{port}")
        await asyncio.sleep(0.1)
    return server, serving


async def run_agentic_miner(*, job, checkpoint_dir: str, proof, tokenizer, identities, client,
                            concurrency: int = 8, port: int = 8011,
                            gpu_memory_utilization: float | None = None,
                            max_num_seqs: int = 16, sessions=None,
                            max_live_per_job: int = 4) -> dict[str, Counter]:
    """The whole miner in one process: engine thread, loopback endpoint, episodes.
    A signed-sandbox job needs ``sessions`` (the validator's session client,
    ``signed_episode.HttpSandboxSessions``) and each identity's ``sign_binding``."""
    import contextlib

    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.miner import agentic_episode
    from reliquary.miner.suite_episode import is_suite_episode

    if is_suite_episode(job.episode):
        return await run_suite_miner(
            job=job, checkpoint_dir=checkpoint_dir, proof=proof, tokenizer=tokenizer,
            identities=identities, client=client, concurrency=concurrency,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs)

    signed = is_signed_sandbox(job)
    if signed:
        if sessions is None:
            raise RuntimeError("a signed-sandbox job needs the validator's session client")
        if any(identity.sign_binding is None for identity in identities):
            raise RuntimeError("a signed-sandbox job needs each identity's sign_binding")
    else:
        refusal = agentic_episode.network_notice_refusal()
        if refusal:
            raise RuntimeError(f"refusing to mine: {refusal}")
    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary.miner.agentic_episode import SweEpisodeRunner
    from reliquary.miner.corpus_generate_server import (
        GenerateEngine, VllmTurnCore, build_generate_app,
    )

    renderer = load_turn_renderer(checkpoint_dir, tools=job.episode.sandbox.tools if signed
                                  else ("bash", "edit"))
    stop_ids = sorted(renderer.stop_ids)
    core = VllmTurnCore(checkpoint_dir, sampling=job.sampling, proof=proof, stop_token_ids=stop_ids,
                        max_total_tokens=job.episode.max_total_tokens, max_num_seqs=max_num_seqs,
                        gpu_memory_utilization=gpu_memory_utilization)
    engine = GenerateEngine(core, max_total_tokens=job.episode.max_total_tokens,
                            max_tokens_per_turn=job.episode.max_tokens_per_turn)
    engine.start()
    server = serving = None
    try:
        server, serving = await serve_loopback(
            build_generate_app(engine, model_name=job.checkpoint_repo), port)
        async with contextlib.AsyncExitStack() as stack:
            if signed:
                from reliquary.miner.signed_episode import (
                    SignedSweEpisodeRunner, signed_trajectory_precheck,
                )

                per_identity = {}
                for identity in identities:
                    per_identity[identity.hotkey] = await stack.enter_async_context(
                        SignedSweEpisodeRunner(
                            job=job, hotkey=identity.hotkey, sign_binding=identity.sign_binding,
                            sessions=sessions, model_name=job.checkpoint_repo,
                            renderer_model_dir=checkpoint_dir,
                            generate_url=f"http://127.0.0.1:{port}", sampling=job.sampling,
                            harness_env=identity.harness_env, max_live=max_live_per_job))
                from reliquary.corpus.job import sandbox_split
                from reliquary.environment.agentic_swe import SignedSweSource

                return await mine_agentic(
                    job=job, identities=identities, client=client, engine=engine, runners={},
                    runner_for=lambda identity: per_identity[identity.hotkey],
                    decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False),
                    concurrency=concurrency,
                    precheck=signed_trajectory_precheck(
                        renderer, max_turns=job.episode.max_turns, job=job, tokenizer=tokenizer,
                        source=SignedSweSource(sandbox_split(job.episode)),
                        chunk_tokens=proof.chunk_tokens))
            runners = {}
            for identity in identities:
                key = harness_key(identity.harness_env)
                if key not in runners:
                    runners[key] = await stack.enter_async_context(SweEpisodeRunner(
                        episode=job.episode, model_name=job.checkpoint_repo,
                        renderer_model_dir=checkpoint_dir, generate_url=f"http://127.0.0.1:{port}",
                        sampling=job.sampling, harness_env=identity.harness_env))
            return await mine_agentic(
                job=job, identities=identities, client=client, engine=engine, runners=runners,
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False),
                concurrency=concurrency,
                precheck=trajectory_precheck(renderer, max_turns=job.episode.max_turns))
    finally:
        if server is not None:
            server.should_exit = True
            await serving
        engine.stop()


async def run_suite_miner(*, job, checkpoint_dir: str, proof, tokenizer, identities, client,
                          concurrency: int = 8, gpu_memory_utilization: float | None = None,
                          max_num_seqs: int = 16) -> dict[str, Counter]:
    """A suite episode job: the engine thread and episodes in one process. No
    endpoint: the runner's policy calls the engine directly, and nothing but
    the pinned environment wheel runs between turns."""
    from reliquary.corpus.suite_episode import TokenizerCodec
    from reliquary.miner.corpus_generate_server import GenerateEngine, VllmTurnCore
    from reliquary.miner.suite_episode import SuiteEpisodeRunner, suite_stop_ids

    # Coded as the validator's intake, replay and export code.
    codec = TokenizerCodec(tokenizer)
    encode, decode = codec.encode, codec.decode
    core = VllmTurnCore(checkpoint_dir, sampling=job.sampling, proof=proof,
                        stop_token_ids=suite_stop_ids(encode),
                        max_total_tokens=job.episode.max_total_tokens, max_num_seqs=max_num_seqs,
                        gpu_memory_utilization=gpu_memory_utilization)
    engine = GenerateEngine(core, max_total_tokens=job.episode.max_total_tokens,
                            max_tokens_per_turn=job.episode.max_tokens_per_turn)
    engine.start()
    try:
        runner = SuiteEpisodeRunner(job=job, engine=engine, encode=encode, decode=decode,
                                    chunk_tokens=proof.chunk_tokens)
        return await mine_agentic(
            job=job, identities=identities, client=client, engine=engine,
            runners={harness_key(identity.harness_env): runner for identity in identities},
            decode=decode, concurrency=concurrency)
    finally:
        engine.stop()


__all__ = ["Identity", "OpenPrompts", "build_trajectory_submission", "harness_key", "mine_agentic",
           "run_agentic_miner", "run_suite_miner", "serve_loopback", "trajectory_precheck"]
