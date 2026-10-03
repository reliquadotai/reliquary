"""The agentic corpus miner: episodes in parallel, one signed trajectory per
finished episode (spec §5 N1, N2)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from reliquary.corpus.trajectory import BuiltTrajectory, TrajectoryUnbuildable, build_trajectory
from reliquary.corpus.walk import job_walk_index
from reliquary.miner.corpus_miner import _HALT, CorpusJobRetired, CorpusMinerHalted, _retry

logger = logging.getLogger(__name__)


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


def _submit(client, body: dict, counts: Counter) -> dict:
    return _retry(lambda: client.submit(body), sleep=time.sleep, counts=counts,
                  max_consecutive_failures=5)


async def _mine_identity(*, job, identity: Identity, client, engine, runner, decode,
                         slots: asyncio.Semaphore, counts: Counter, stop: asyncio.Event) -> None:
    async def one(cursor: int) -> None:
        prompt_index = job_walk_index(job, identity.hotkey, cursor)
        try:
            result = await runner.run(prompt_index)
        except Exception:
            logger.exception("episode %d of %s crashed", prompt_index, identity.hotkey[:8])
            counts["episode_crashed"] += 1
            return
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
        if identity.transform is not None:
            built = identity.transform(built, prompt_index)
        body = build_trajectory_submission(
            job=job, hotkey=identity.hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=decode(list(built.prompt_ids)), trajectory=built, sign=identity.sign)
        try:
            answer = await asyncio.to_thread(_submit, client, body, counts)
        except (CorpusJobRetired, CorpusMinerHalted) as exc:
            counts["halted"] += 1
            logger.error("%s stops: %s", identity.hotkey[:8], exc)
            stop.set()
            return
        reason = str(answer.get("reason"))
        counts[reason] += 1
        logger.info("episode %d of %s: %s %s", prompt_index, identity.hotkey[:8], reason,
                    answer.get("detail") or "")
        if reason == "job_complete" or reason in _HALT:
            stop.set()

    tasks: set[asyncio.Task] = set()
    cursor = 0
    while not stop.is_set() and (identity.episodes is None or cursor < identity.episodes):
        await slots.acquire()
        if stop.is_set():
            slots.release()
            break
        if not getattr(engine, "healthy", True):
            # Every turn would fail fast: starting episodes would only burn boxes.
            counts["engine_unhealthy"] += 1
            logger.error("the generate engine is unhealthy: %s stops", identity.hotkey[:8])
            stop.set()
            slots.release()
            break
        task = asyncio.create_task(one(cursor))
        task.add_done_callback(lambda _t: slots.release())
        tasks.add(task)
        cursor += 1
    await asyncio.gather(*tasks)


async def mine_agentic(*, job, identities, client, engine, runners, decode,
                       concurrency: int) -> dict[str, Counter]:
    """Every identity mines until its episode count or the job's end; at most
    ``concurrency`` episodes run at once across them."""
    slots = asyncio.Semaphore(concurrency)
    counts = {identity.hotkey: Counter() for identity in identities}
    stops = {identity.hotkey: asyncio.Event() for identity in identities}
    await asyncio.gather(*(
        _mine_identity(job=job, identity=identity, client=client, engine=engine,
                       runner=runners[harness_key(identity.harness_env)], decode=decode,
                       slots=slots, counts=counts[identity.hotkey], stop=stops[identity.hotkey])
        for identity in identities))
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

    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary.miner.agentic_episode import SweEpisodeRunner
    from reliquary.miner.corpus_generate_server import (
        GenerateEngine, VllmTurnCore, build_generate_app,
    )

    stop_ids = sorted(load_turn_renderer(checkpoint_dir).stop_ids)
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
                concurrency=concurrency)
    finally:
        if server is not None:
            server.should_exit = True
            await serving
        engine.stop()


__all__ = ["Identity", "build_trajectory_submission", "harness_key", "mine_agentic",
           "run_agentic_miner", "serve_loopback"]
