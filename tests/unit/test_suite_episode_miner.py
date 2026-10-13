"""The miner of suite episode corpus jobs, on the telecom environment, over a
scripted turn core (no GPU): the real runner, engine, trajectory and replay."""

from __future__ import annotations

import asyncio
import importlib.util
import re
from collections import Counter

import pytest

from reliquary.corpus.job import JobError
from reliquary.corpus.trajectory import build_trajectory
from reliquary.environment.agentic.chatml import ChatMLEpisodeRenderer
from reliquary.environment.agentic.types import AssistantAction
from reliquary.miner.agentic_miner import Identity, harness_key, mine_agentic
from reliquary.miner.corpus_generate_server import Finished, GenerateEngine
from reliquary.miner.suite_episode import (
    SuiteEpisodeRunner,
    _stop_for,
    episode_seed,
    read_episode_job,
    replay_refusal,
    suite_support_refusal,
)
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
from reliquary.protocol.toploc import span_chunk_count

ENV = "reliquary_telecom_solo_v1"
MANIFEST_SHA = "74d0e7569247eabc3d4fb2d909773f2b1f1d8430a4ce4f4fd09c6ad6e6794b05"
RENDERER = "reliquary-chatml-tools-v1"

# A char-level tokenizer whose ChatML specials are one id each, as in Qwen's.
_SPECIALS = {"<|im_start|>": 2_000_000, "<|im_end|>": 2_000_001, "<|endoftext|>": 2_000_002}
_BY_ID = {v: k for k, v in _SPECIALS.items()}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in _SPECIALS) + ")")


def encode(text: str) -> list[int]:
    return [_SPECIALS[p] if p in _SPECIALS else ord(c)
            for p in _SPLIT.split(text) if p for c in ([p] if p in _SPECIALS else p)]


def decode(ids) -> str:
    return "".join(_BY_ID.get(i) or chr(i) for i in ids)


def _manifest(*, max_turns=8, max_total_tokens=60000, version=MANIFEST_SHA, **episode):
    return {
        "schema": "reliquary/corpus-job/v1", "job_id": "telecom-suite-v1",
        "checkpoint_repo": "R/teutonic", "checkpoint_revision": "rev1",
        "checkpoint_sha256": "c" * 64, "prompt_source": ENV, "prompt_count": 100,
        "renderer_id": RENDERER, "eos_token_id": 248046,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                     "max_new_tokens": 4096, "n": 1},
        "slots_per_prompt": 2, "filter": None, "prompt_order": "free", "deadline_round": None,
        "episode": {"env": {"package": "reliquary-suite", "name": ENV, "version": version,
                            "split": "train"},
                    "harness": "suite", "renderer": RENDERER, "max_turns": max_turns,
                    "max_tokens_per_turn": 4096, "max_total_tokens": max_total_tokens,
                    **episode},
    }


def test_a_suite_manifest_is_read_and_others_go_to_parse_job():
    job = read_episode_job(_manifest(replay_fraction_failed=0.1))   # an unknown field is ignored
    assert job.episode.env.name == ENV and job.episode.harness == "suite"
    assert job.episode.max_turns == 8 and job.sampling.n == 1
    with pytest.raises(JobError, match="renderer_id"):
        read_episode_job({**_manifest(), "renderer_id": "other"})
    with pytest.raises(JobError, match="package"):
        bad = _manifest()
        bad["episode"]["env"]["package"] = "reliquary-swe"
        read_episode_job(bad)


def test_the_seed_is_the_job_and_prompt_digest():
    import hashlib

    digest = hashlib.sha256(b"telecom-suite-v1:7").hexdigest()
    assert episode_seed("telecom-suite-v1", 7) == int(digest[:16], 16)


def test_stop_mapping():
    assert _stop_for("turn_limit") == "max_turns"
    for reason in ("finished", "too_many_errors", "invalid_action", "environment_done"):
        assert _stop_for(reason) == "agent_completed"


# Not imported here: the environment loader refuses a module imported unverified.
needs_telecom = pytest.mark.skipif(
    importlib.util.find_spec("reliquary_telecom_solo") is None,
    reason="install the reliquary-telecom-solo wheel (reliquary-environments@0bfffb4)")


@needs_telecom
def test_support_refuses_a_job_pinning_another_manifest():
    assert suite_support_refusal(read_episode_job(_manifest()).episode) is None
    refusal = suite_support_refusal(read_episode_job(_manifest(version="f" * 64)).episode)
    assert refusal and "manifest" in refusal
    unknown = read_episode_job(_manifest())
    unknown = unknown.episode.__class__(**{**{f: getattr(unknown.episode, f) for f in
                                              ("harness", "renderer", "max_turns",
                                               "max_tokens_per_turn", "max_total_tokens")},
                                           "env": unknown.episode.env.__class__(
                                               "reliquary-suite", "no_such_env", MANIFEST_SHA,
                                               "train")})
    assert "no_such_env" in suite_support_refusal(unknown)


def _turn(tool: str | None, *, end="<|im_end|>") -> list[int]:
    action = (AssistantAction.final("All set.") if tool is None
              else AssistantAction(kind="tool", tool=tool, arguments={}))
    return encode(ChatMLEpisodeRenderer.action_text(action, reasoning="Let me check.") + end)


class ScriptedCore:
    """Answers each request with the next scripted completion, cut at its cap,
    with as many proofs as the span needs."""

    def __init__(self, completions):
        self._script = list(completions)
        self._queue = []

    def add(self, request_id, prompt_ids, max_tokens):
        self._queue.append((request_id, max_tokens))

    def has_unfinished(self):
        return bool(self._queue)

    def step(self):
        out = []
        for request_id, cap in self._queue:
            tokens = tuple(self._script.pop(0)[:cap] if self._script else _turn("done"))
            proofs = tuple("P" for _ in range(span_chunk_count(
                len(tokens), TOPLOC_DEPLOYED_DEFAULTS.chunk_tokens)))
            out.append(Finished(request_id, tokens, tuple(-0.1 for _ in tokens), "stop", proofs))
        self._queue = []
        return out


class Client:
    def __init__(self):
        self.bodies = []

    def submit(self, body):
        self.bodies.append(body)
        return {"reason": "accepted", "accepted": True}


def _mine(completions, **manifest):
    job = read_episode_job(_manifest(**manifest))
    engine = GenerateEngine(ScriptedCore(completions),
                            max_total_tokens=job.episode.max_total_tokens,
                            max_tokens_per_turn=job.episode.max_tokens_per_turn)
    engine.start()
    try:
        runner = SuiteEpisodeRunner(job=job, engine=engine, encode=encode, decode=decode)
        client = Client()
        counts = asyncio.run(mine_agentic(
            job=job, identities=[Identity("5Hot", sign=lambda body: "sig", episodes=1)],
            client=client, engine=engine, runners={harness_key(None): runner},
            decode=decode, concurrency=1))
    finally:
        engine.stop()
    return counts["5Hot"], client.bodies, runner, job


@needs_telecom
def test_an_honest_episode_is_submitted_and_replays():
    counts, bodies, runner, job = _mine([_turn("check_status_bar"), _turn("done")])
    assert counts["accepted"] == 1, counts
    (body,) = bodies
    trajectory = body["trajectory"]
    assert trajectory["stop"] == "agent_completed" and trajectory["final_diff"] == ""
    assert len(trajectory["turns"]) == 2 and trajectory["turns"][0]["start"] == 0
    assert trajectory["turns"][-1]["end"] == len(trajectory["tokens"])
    assert trajectory["tokens"][trajectory["turns"][-1]["end"] - 1] == _SPECIALS["<|im_end|>"]
    prompt_index = body["prompt_index"]
    prompt = runner.initial_ids(prompt_index)
    assert body["rendered_prompt"] == decode(prompt)

    # The validator's own replay of exactly what was signed.
    from reliquary.corpus.trajectory import GeneratedTurn

    built = build_trajectory(
        [GeneratedTurn(tuple(prompt + trajectory["tokens"][:t["start"]]),
                       tuple(trajectory["tokens"][t["start"]:t["end"]]), tuple(t["proofs"]))
         for t in trajectory["turns"]], final_diff="", stop="agent_completed")
    environment = runner._environment
    seed = episode_seed(job.job_id, prompt_index)
    assert replay_refusal(environment, prompt_index=prompt_index, seed=seed, built=built,
                          renderer_id=RENDERER, encode=encode, decode=decode) is None
    # Another stop, or another prompt, does not replay.
    from dataclasses import replace

    assert replay_refusal(environment, prompt_index=prompt_index, seed=seed,
                          built=replace(built, stop="max_turns"), renderer_id=RENDERER,
                          encode=encode, decode=decode) is not None
    forged = list(built.tokens)
    forged[built.spans[0][1] + 3] = ord("#")      # an observation token
    assert replay_refusal(environment, prompt_index=prompt_index, seed=seed,
                          built=replace(built, tokens=tuple(forged)), renderer_id=RENDERER,
                          encode=encode, decode=decode)[0] == "replay_mismatch"


@needs_telecom
def test_the_turn_limit_is_submitted_as_max_turns():
    counts, bodies, _, _ = _mine([_turn("check_status_bar")] * 3, max_turns=3)
    assert counts["accepted"] == 1, counts
    assert bodies[0]["trajectory"]["stop"] == "max_turns"
    assert len(bodies[0]["trajectory"]["turns"]) == 3


@needs_telecom
def test_an_episode_over_its_token_budget_is_dropped():
    from reliquary.corpus.walk import job_walk_index

    job = read_episode_job(_manifest())
    runner = SuiteEpisodeRunner(job=job, engine=None, encode=encode, decode=decode)
    # Room for the first turn, not for the observation after it.
    prompt = runner.initial_ids(job_walk_index(job, "5Hot", 0))
    total = len(prompt) + len(_turn("check_status_bar")) + 5
    assert total > 4096
    counts, bodies, _, _ = _mine([_turn("check_status_bar"), _turn("done")],
                                 max_total_tokens=total)
    assert bodies == [] and counts["episode_failed"] == 1, counts

    # The runner's own verdict: the runner's budget, not anything else.
    small = read_episode_job(_manifest(max_total_tokens=total))
    engine = GenerateEngine(ScriptedCore([_turn("check_status_bar")]), max_total_tokens=total,
                            max_tokens_per_turn=4096)
    engine.start()
    try:
        result = asyncio.run(SuiteEpisodeRunner(job=small, engine=engine, encode=encode,
                                                decode=decode).run(job_walk_index(job, "5Hot", 0)))
    finally:
        engine.stop()
    assert not result.ok and "total token budget" in result.error


@needs_telecom
def test_a_turn_ending_on_endoftext_mid_episode_is_dropped():
    counts, bodies, _, _ = _mine([_turn("check_status_bar", end="<|endoftext|>"),
                                  _turn("done")])
    assert bodies == [], counts
    assert counts["precheck_refused:bad_termination"] == 1, counts


@needs_telecom
def test_endoftext_on_the_last_turn_is_submitted():
    counts, bodies, _, _ = _mine([_turn("check_status_bar"), _turn("done", end="<|endoftext|>")])
    assert counts["accepted"] == 1 and bodies[0]["trajectory"]["stop"] == "agent_completed"


@needs_telecom
def test_counts_are_a_counter():
    counts, _, _, _ = _mine([_turn("done")])
    assert isinstance(counts, Counter) and counts["accepted"] == 1
