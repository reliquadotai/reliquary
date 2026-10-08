"""The agentic corpus miner: episodes in parallel, one signed trajectory per
finished episode (spec §5 N1, N2)."""

from __future__ import annotations

import asyncio
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

logger = logging.getLogger(__name__)

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


def _submit(client, body: dict, counts: Counter) -> dict:
    return _retry(lambda: client.submit(body), sleep=time.sleep, counts=counts,
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

    async def episode(cursor: int, prompt_index: int, sessions: list[str]) -> None:
        deadline = getattr(runner, "deadline", None)
        limit = deadline(prompt_index) if deadline is not None else None
        try:
            async with asyncio.timeout(limit):
                result = await runner.run(prompt_index, on_session=sessions.append)
        except TimeoutError:
            counts["episode_timeout"] += 1
            logger.warning("episode %d of %s passed its %s s deadline", prompt_index, tag, limit)
            return
        if result.session_id:
            sessions.append(result.session_id)
        # Taken whatever the outcome, so a failed episode's log does not linger.
        session = engine.take_session(result.session_id) if result.session_id else None
        if not result.ok or session is None:
            counts["episode_failed"] += 1
            logger.warning("episode %d not submitted: %s", prompt_index,
                           result.error or "no session was logged")
            return
        if not session.linear:
            counts["not_linear"] += 1
            return
        try:
            built = build_trajectory(session.turns, final_diff=result.final_diff, stop=result.stop)
        except TrajectoryUnbuildable as exc:
            counts["unbuildable"] += 1
            logger.warning("episode %d not submitted: %s", prompt_index, exc)
            return
        # Before the forgery hook: the hook tests the validator, the check the honest run.
        refusal = await asyncio.to_thread(precheck, built) if precheck is not None else None
        if refusal is not None:
            reason, detail = refusal
            counts["precheck_refused"] += 1
            counts[f"precheck_refused:{reason}"] += 1
            logger.warning("episode %d of %s not submitted, the validator would refuse it: %s %s",
                           prompt_index, tag, reason, detail)
            return
        if identity.transform is not None:
            built = identity.transform(built, prompt_index)
        body = build_trajectory_submission(
            job=job, hotkey=identity.hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=decode(list(built.prompt_ids)), trajectory=built, sign=identity.sign)
        try:
            answer = await asyncio.to_thread(_submit, client, body, counts)
        except (CorpusJobRetired, CorpusMinerHalted) as exc:
            counts["halted"] += 1
            logger.error("%s stops: %s", tag, exc)
            halt()
            return
        reason = str(answer.get("reason"))
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
        try:
            await episode(cursor, prompt_index, sessions)
        except asyncio.CancelledError:
            counts["episode_cancelled"] += 1
            raise
        except Exception:
            counts["episode_crashed"] += 1
            logger.exception("episode %d of %s crashed", prompt_index, tag)
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
                       concurrency: int, precheck=None) -> dict[str, Counter]:
    """Every identity mines until its episode count or the job's end; at most
    ``concurrency`` episodes run at once across them. ``precheck`` (see
    ``trajectory_precheck``) drops what the validator would refuse, unsigned."""
    slots = asyncio.Semaphore(concurrency)
    open_prompts = OpenPrompts(job, client)
    counts = {identity.hotkey: Counter() for identity in identities}
    stops = {identity.hotkey: asyncio.Event() for identity in identities}
    outcomes = await asyncio.gather(*(
        _mine_identity(job=job, identity=identity, client=client, engine=engine,
                       runner=runners[harness_key(identity.harness_env)], decode=decode,
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
                            max_num_seqs: int = 16) -> dict[str, Counter]:
    """The whole miner in one process: engine thread, loopback endpoint, episodes."""
    import contextlib

    from reliquary.miner import agentic_episode

    refusal = agentic_episode.network_notice_refusal()
    if refusal:
        raise RuntimeError(f"refusing to mine: {refusal}")
    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary.miner.agentic_episode import SweEpisodeRunner
    from reliquary.miner.corpus_generate_server import (
        GenerateEngine, VllmTurnCore, build_generate_app,
    )

    renderer = load_turn_renderer(checkpoint_dir)
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


__all__ = ["Identity", "OpenPrompts", "build_trajectory_submission", "harness_key", "mine_agentic",
           "run_agentic_miner", "serve_loopback", "trajectory_precheck"]
