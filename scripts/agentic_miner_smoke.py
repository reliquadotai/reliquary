"""Smoke the agentic miner on the GPU box before the end-to-end run: run a few
real episodes against the local generate endpoint (phase generate), then audit
and parse the trajectories offline with the validator's code (phase verify).

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 HF_HOME=/opt/hf \
      python scripts/agentic_miner_smoke.py generate --env-commit SHA --indexes 0 1 --work /opt/smoke.json
    HF_HOME=/opt/hf python scripts/agentic_miner_smoke.py verify --env-commit SHA --work /opt/smoke.json

Phase generate also asserts what only real vLLM can show about the generate
core: the engine's request ids map back to ours, ``num_cached_tokens`` is
reported and used, the logprob entries are read (not floored), stop tokens
stay in ``token_ids``, no captured activations outlive a drained batch, and an
abort by our id frees the request. It counts preemptions under the run's
concurrency.
"""

import argparse
import asyncio
import json
import math
import time

from reliquary.corpus.job import parse_job

MODEL = "Qwen/Qwen3.8-27B"


def smoke_job(env_commit: str, num_images: int, max_turns: int, prompt_count: int = 1):
    return parse_job({
        "schema": "reliquary/corpus-job/v1", "job_id": "agentic-smoke", "checkpoint_repo": MODEL,
        "checkpoint_revision": "main", "checkpoint_sha256": "0" * 64,
        "prompt_source": "reliquary_agentic_swe_v1", "prompt_count": prompt_count,
        "renderer_id": "renderers:qwen38@0.1.11", "eos_token_id": 0,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                     "max_new_tokens": 8192, "n": 1},
        "slots_per_prompt": 2, "filter": None, "prompt_order": "free", "deadline_round": None,
        "episode": {"env": {"package": "reliquary-swe", "version": env_commit, "split": "train",
                            "num_images": num_images},
                    "harness": "bash", "renderer": "renderers:qwen38@0.1.11",
                    "verifiers": "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf",
                    "max_turns": max_turns, "max_tokens_per_turn": 8192, "max_total_tokens": 60000,
                    "replay_fraction_failed": 1.0},
    })


class CoreProbe:
    """Wraps a live VllmTurnCore: records what every finished turn looked like
    on the way through ``_finish`` and counts the scheduler's preemptions."""

    def __init__(self, core) -> None:
        from reliquary.miner.corpus_generate_server import LOGPROB_FLOOR

        self.core, self.turns, self.preemptions, self.added = core, [], 0, 0
        self._floor = LOGPROB_FLOOR
        finish, add = core._finish, core.add

        def recording_finish(output):
            completion = output.outputs[0]
            entries = list(completion.logprobs or [])
            entry = entries[0] if entries else None
            base = core._own_id(output.request_id)
            prompt_len = core._prompt_len.get(base) if base else None
            done = finish(output)
            floored = sum(1 for lp in done.logprobs if lp <= self._floor)
            self.turns.append({
                "engine_id": output.request_id, "own_id": base, "suffixed": base is not None and output.request_id != base,
                "has_num_cached_tokens": hasattr(output, "num_cached_tokens"),
                "num_cached_tokens": getattr(output, "num_cached_tokens", None),
                "prompt_len": prompt_len, "completion_len": len(completion.token_ids),
                "last_token": int(completion.token_ids[-1]) if completion.token_ids else None,
                "ends_on_stop": bool(completion.token_ids) and int(completion.token_ids[-1]) in core._stops,
                "finish_reason": done.finish_reason, "error": done.error,
                "logprobs_type": type(completion.logprobs).__name__,
                "entry_type": type(entry).__name__,
                "entry_value_type": type(next(iter(entry.values()))).__name__ if isinstance(entry, dict) and entry else None,
                "floored": floored, "proofs": len(done.proofs),
            })
            return done

        def counting_add(request_id, prompt_ids, max_tokens):
            add(request_id, prompt_ids, max_tokens)
            self.added += 1

        core._finish, core.add = recording_finish, counting_add
        scheduler = core._engine.engine_core.engine_core.scheduler
        preempt = scheduler._preempt_request

        def counting_preempt(request, *args, **kwargs):
            self.preemptions += 1
            return preempt(request, *args, **kwargs)

        scheduler._preempt_request = counting_preempt


async def wait_for(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.05)
    return condition()


async def abort_check(engine, core, prompt_ids) -> dict:
    """A turn whose waiter goes away is aborted by our (base) id: vLLM must
    drop the request at once, not run it to its token cap."""
    task = asyncio.create_task(engine.generate("abort-probe", prompt_ids, 4096))
    started = await wait_for(lambda: core.has_unfinished(), 30)
    await asyncio.sleep(2.0)                      # well into decoding
    running_before = core.has_unfinished()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    freed = await wait_for(lambda: not core.has_unfinished(), 5.0)
    await asyncio.sleep(0.5)
    return {"started": started, "running_before_cancel": running_before, "freed_within_5s": freed,
            "pending_capture_after": core.pending_capture_count(),
            "output_states_left": len(core._engine.output_processor.request_states)}


async def generate(args):
    from huggingface_hub import snapshot_download

    from reliquary.corpus.trajectory import TrajectoryUnbuildable, build_trajectory
    from reliquary.environment.agentic_swe import episode_support_refusal, load_turn_renderer
    from reliquary.miner.agentic_episode import SweEpisodeRunner
    from reliquary.miner.agentic_miner import serve_loopback
    from reliquary.miner.corpus_generate_server import GenerateEngine, VllmTurnCore, build_generate_app
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF

    job = smoke_job(args.env_commit, args.num_images, args.max_turns)
    refusal = episode_support_refusal(job.episode, need_verifiers=True)
    if refusal:
        raise SystemExit(refusal)
    directory = snapshot_download(MODEL)
    renderer = load_turn_renderer(directory)
    stops = sorted(renderer.stop_ids)
    core = VllmTurnCore(directory, sampling=job.sampling, proof=PROOF, stop_token_ids=stops,
                        max_total_tokens=60000, max_num_seqs=args.max_num_seqs,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    probe = CoreProbe(core)
    engine = GenerateEngine(core, max_total_tokens=60000, max_tokens_per_turn=8192)
    engine.start()
    server, serving = await serve_loopback(build_generate_app(engine, model_name=MODEL), args.port)
    out, checks = [], {}
    started = time.monotonic()
    try:
        async with SweEpisodeRunner(episode=job.episode, model_name=MODEL, renderer_model_dir=directory,
                                    generate_url=f"http://127.0.0.1:{args.port}",
                                    sampling=job.sampling) as runner:
            results = await asyncio.gather(*(runner.run(i) for i in args.indexes),
                                           return_exceptions=True)
        wall = time.monotonic() - started
        drained = await wait_for(lambda: not core.has_unfinished(), 10)
        checks["drained"] = drained
        checks["pending_capture_after_batch"] = core.pending_capture_count()
        first_prompt = None
        for index, result in zip(args.indexes, results):
            if isinstance(result, BaseException):
                out.append({"index": index, "ok": False, "stop": None, "reward": None,
                            "error": f"crashed: {result!r}", "turns": 0, "linear": None})
                continue
            session = engine.take_session(result.session_id) if result.session_id else None
            row = {"index": index, "ok": result.ok, "stop": result.stop, "reward": result.reward,
                   "error": result.error, "turns": len(session.turns) if session else 0,
                   "linear": session.linear if session else None}
            if session and session.turns and first_prompt is None:
                first_prompt = list(session.turns[0].prompt_ids)
            if result.ok and session and session.linear:
                try:
                    built = build_trajectory(session.turns, final_diff=result.final_diff, stop=result.stop)
                except TrajectoryUnbuildable as exc:
                    row["unbuildable"] = str(exc)
                else:
                    row.update(prompt_ids=list(built.prompt_ids), tokens=list(built.tokens),
                               spans=[list(s) for s in built.spans], proofs=[list(p) for p in built.proofs],
                               final_diff=built.final_diff)
            out.append(row)
        checks["abort"] = await abort_check(engine, core, first_prompt or renderer.initial_ids("Say hello."))
        checks["pending_capture_final"] = core.pending_capture_count()
    finally:
        server.should_exit = True
        await serving
        engine.stop()

    turns = probe.turns
    proven = [t for t in turns if t["error"] is None]
    later = [t for t in turns if t["prompt_len"] and t["num_cached_tokens"]]
    stats = {
        "episodes": len(args.indexes), "wall_s": round(wall, 1), "requests_added": probe.added,
        "turns_finished": len(turns), "turns_unproven": len(turns) - len(proven),
        "unproven_errors": sorted({t["error"] for t in turns if t["error"]})[:5],
        "preemptions": probe.preemptions,
        "preemption_rate_per_turn": round(probe.preemptions / max(1, probe.added), 4),
        "suffixed_engine_ids": sum(t["suffixed"] for t in turns),
        "unmatched_engine_ids": sum(t["own_id"] is None for t in turns),
        "num_cached_tokens_reported": sum(t["has_num_cached_tokens"] for t in turns),
        "turns_with_cache_hits": len(later),
        "cached_share_of_prompt": round(sum(t["num_cached_tokens"] for t in later)
                                        / max(1, sum(t["prompt_len"] for t in turns if t["prompt_len"])), 4),
        "logprob_entry_types": sorted({(t["logprobs_type"], t["entry_type"], t["entry_value_type"]) for t in turns}),
        "floored_logprobs": sum(t["floored"] for t in turns),
        "completion_tokens": sum(t["completion_len"] for t in turns),
        "turns_ending_on_stop": sum(t["ends_on_stop"] for t in turns),
        "finish_reasons": {r: sum(t["finish_reason"] == r for t in turns) for r in {t["finish_reason"] for t in turns}},
        "capped_turns_not_at_cap": sum(1 for t in turns if not t["ends_on_stop"] and t["error"] is None
                                       and t["completion_len"] < 8192 and t["prompt_len"]
                                       and t["prompt_len"] + t["completion_len"] < 60000),
        **checks,
    }
    failures = []
    if stats["unmatched_engine_ids"]:
        failures.append("an engine request id did not map back to ours")
    if stats["num_cached_tokens_reported"] != len(turns):
        failures.append("num_cached_tokens missing from a RequestOutput")
    if not stats["turns_with_cache_hits"]:
        failures.append("no turn reported a prefix-cache hit")
    if stats["floored_logprobs"] > max(1, stats["completion_tokens"] // 1000):
        failures.append("logprob entries were not read (floored)")
    if not stats["turns_ending_on_stop"]:
        failures.append("no completion kept its stop token")
    if stats["capped_turns_not_at_cap"]:
        failures.append("a completion ended before its cap without a stop token (stop token stripped?)")
    if not stats["drained"] or stats["pending_capture_after_batch"]:
        failures.append("captured activations outlived the drained batch")
    if not (stats["abort"]["freed_within_5s"] and stats["abort"]["running_before_cancel"]
            and stats["abort"]["pending_capture_after"] == 0 and stats["abort"]["output_states_left"] == 0):
        failures.append("an abort by our request id did not free the request")
    stats["failures"] = failures
    with open(args.work, "w") as handle:
        json.dump({"rows": out, "stats": stats, "turns": turns}, handle)
    print(json.dumps([{k: r.get(k) for k in ("index", "ok", "stop", "reward", "turns", "linear", "error",
                                             "unbuildable")} for r in out], indent=1))
    print(json.dumps(stats, indent=1, default=str))
    if failures:
        raise SystemExit("smoke failed: " + "; ".join(failures))


def verify(args):
    """The validator's path on each recorded trajectory: the miner's pre-sign
    check, the episode intake (prompt, spans, parse, budgets, proof shape),
    then the per-span TOPLOC audit."""
    import torch
    from huggingface_hub import snapshot_download

    from reliquary.corpus.trajectory import BuiltTrajectory
    from reliquary.environment.agentic_swe import load_swe_source, load_turn_renderer, network_notice
    from reliquary.miner.agentic_miner import build_trajectory_submission, trajectory_precheck
    from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
    from reliquary.shared.modeling import load_text_only_model
    from reliquary.validator.agentic_intake import EpisodeIntake, IntakeFacts
    from reliquary.validator.corpus_audit import score_sequences, trajectory_outcome

    with open(args.work) as handle:
        rows = [r for r in json.load(handle)["rows"] if "tokens" in r]
    if not rows:
        raise SystemExit("no trajectory to verify")
    directory = snapshot_download(MODEL)
    renderer = load_turn_renderer(directory)
    tokenizer = renderer._tokenizer
    source = load_swe_source(args.num_images)
    job = smoke_job(args.env_commit, args.num_images, args.max_turns, prompt_count=len(source))
    intake = EpisodeIntake(job=job, source=source, renderer=renderer, tokenizer=tokenizer,
                           vocab_size=None, chunk_tokens=PROOF.chunk_tokens)
    precheck = trajectory_precheck(renderer, max_turns=job.episode.max_turns)
    model = load_text_only_model(directory, torch_dtype=torch.bfloat16,
                                 attn_implementation="sdpa").to("cuda").eval()
    report = []
    for r in rows:
        prompt = r["prompt_ids"]
        built = BuiltTrajectory(tuple(prompt), tuple(r["tokens"]), tuple(tuple(s) for s in r["spans"]),
                                tuple(tuple(p) for p in r["proofs"]), r["final_diff"], r["stop"])
        refusal = precheck(built)
        body = build_trajectory_submission(
            job=job, hotkey="5Smoke", cursor=0, prompt_index=r["index"],
            rendered_prompt=tokenizer.decode(prompt, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False),
            trajectory=built, sign=lambda body: "00")
        outcome_intake = intake.check(CorpusSubmissionRequest.model_validate(body))
        intake_ok = isinstance(outcome_intake, IntakeFacts)
        absolute = [(len(prompt) + s, len(prompt) + e) for s, e in r["spans"]]
        flat = [p for turn in r["proofs"] for p in turn]
        (scored,), forward, _ = score_sequences(model, [(prompt + r["tokens"], len(prompt), flat, absolute)],
                                                chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk,
                                                batch_tokens=131072)
        outcome = trajectory_outcome(*scored, [e - s for s, e in r["spans"]], PROOF)
        report.append({"index": r["index"],
                       "prompt_is_validator_render": tuple(prompt) == intake.initial_ids(r["index"]),
                       "precheck": refusal,
                       "intake": "accepted" if intake_ok else {"reason": outcome_intake.reason,
                                                               "detail": outcome_intake.detail},
                       "tokens": len(prompt) + len(r["tokens"]),
                       "turns": len(r["spans"]), "audit_passed": outcome.passed,
                       "audit_reason": outcome.reason, "forward_s": round(forward, 2),
                       "worst_exp": max((c.exp_mismatches for c in outcome.results), default=None),
                       "worst_mant_mean": max((c.mant_err_mean for c in outcome.results
                                               if math.isfinite(c.mant_err_mean)), default=None)})
    print(json.dumps({"network_notice_from": network_notice()[1]}))
    print(json.dumps(report, indent=1, default=str))
    if not all(x["prompt_is_validator_render"] and x["precheck"] is None and x["intake"] == "accepted"
               and x["audit_passed"] for x in report):
        raise SystemExit("smoke failed")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("phase", choices=["generate", "verify"])
    p.add_argument("--env-commit", required=True)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--indexes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--work", required=True)
    args = p.parse_args()
    asyncio.run(generate(args)) if args.phase == "generate" else verify(args)


if __name__ == "__main__":
    main()
