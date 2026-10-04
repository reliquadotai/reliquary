"""The agentic mining loop over fakes: one trajectory per finished episode."""

import asyncio
from collections import Counter

from reliquary.corpus.job import parse_job
from reliquary.corpus.trajectory import GeneratedTurn
from reliquary.miner.agentic_episode import EpisodeResult
from reliquary.miner.agentic_miner import Identity, harness_key, mine_agentic
from reliquary.miner.corpus_generate_server import SessionLog
from tests.unit.test_corpus_job_episode import _manifest

JOB = parse_job(_manifest(prompt_count=50))


class FakeRunner:
    def __init__(self, ok=True, stop="agent_completed", slow=(), slow_seconds=30.0, deadline=None):
        self.ran, self._ok, self._stop = [], ok, stop
        self._slow, self._slow_seconds, self._deadline = set(slow), slow_seconds, deadline

    def deadline(self, index):
        return self._deadline

    async def run(self, index, on_session=None):
        self.ran.append(index)
        if on_session is not None:
            on_session(f"s{index}")                       # the trace id, known at mint
        await asyncio.sleep(self._slow_seconds if index in self._slow
                            else 0.01 * (index % 3))     # finish out of order
        return EpisodeResult(f"s{index}", f"diff {index}", self._stop, self._ok, 1.0)


class FakeEngine:
    def __init__(self, linear=True):
        self._linear = linear
        self.dropped = []

    def take_session(self, session_id):
        n = int(session_id[1:])
        first = GeneratedTurn((1, 2), (10 + n, 7), ("P",))
        second = GeneratedTurn((1, 2, 10 + n, 7, 30), (11, 7), ("Q",))
        return SessionLog([first, second], self._linear)

    def drop_session(self, session_id):
        self.dropped.append(session_id)


class FakeClient:
    def __init__(self, answers=None):
        self.bodies, self._answers = [], list(answers or [])

    def submit(self, body):
        self.bodies.append(body)
        return self._answers.pop(0) if self._answers else {"reason": "accepted", "accepted": True}


def _run(identities, *, runner=None, engine=None, client=None, concurrency=4):
    runner = runner or FakeRunner()
    client = client or FakeClient()
    counts = asyncio.run(mine_agentic(
        job=JOB, identities=identities, client=client, engine=engine or FakeEngine(),
        runners={harness_key(None): runner}, decode=lambda ids: "".join(map(str, ids)),
        concurrency=concurrency))
    return counts, client, runner


def test_each_finished_episode_is_submitted_as_a_signed_trajectory():
    counts, client, runner = _run([Identity("5Hot", sign=lambda body: "sig", episodes=5)])
    assert counts["5Hot"]["accepted"] == 5 and len(client.bodies) == 5
    body = client.bodies[0]
    assert body["signature"] == "sig" and body["completions"] == []
    assert body["trajectory"]["turns"][1] == {"start": 3, "end": 5, "proofs": ["Q"]}
    assert body["rendered_prompt"] == "12"
    assert sorted(b["prompt_index"] for b in client.bodies) == sorted(runner.ran)


def test_a_failed_or_rewritten_episode_is_not_submitted():
    counts, client, _ = _run([Identity("5Hot", sign=lambda b: "s", episodes=3)],
                             runner=FakeRunner(ok=False))
    assert counts["5Hot"]["episode_failed"] == 3 and client.bodies == []
    counts, client, _ = _run([Identity("5Hot", sign=lambda b: "s", episodes=2)],
                             engine=FakeEngine(linear=False))
    assert counts["5Hot"]["not_linear"] == 2 and client.bodies == []


def test_job_complete_stops_the_identity():
    client = FakeClient([{"reason": "job_complete", "accepted": False}])
    counts, _, runner = _run([Identity("5Hot", sign=lambda b: "s", episodes=50)], client=client,
                             concurrency=1)
    assert counts["5Hot"]["job_complete"] == 1 and len(runner.ran) <= 2


def test_a_transform_rewrites_before_signing():
    from dataclasses import replace

    forge = Identity("5Bad", sign=lambda b: "s", episodes=1,
                     transform=lambda built, index: replace(built, final_diff="forged"))
    _, client, _ = _run([forge])
    assert client.bodies[0]["trajectory"]["final_diff"] == "forged"


def test_identities_walk_their_own_prompts():
    a = Identity("5A", sign=lambda b: "s", episodes=3)
    b = Identity("5B", sign=lambda b: "s", episodes=3)
    counts, client, _ = _run([a, b])
    assert Counter(body["miner_hotkey"] for body in client.bodies) == {"5A": 3, "5B": 3}


def test_the_episode_env_config_pins_the_harness():
    from reliquary.miner.agentic_episode import env_config

    config = env_config(JOB.episode, harness_env={"BASH_ENV": "/x"})
    assert config["agent"]["harness"] == {"id": "bash", "edit": True, "search": False,
                                          "env": {"BASH_ENV": "/x"}}
    assert config["agent"]["max_turns"] == 40 and config["taskset"]["num_images"] == 20


def test_an_unhealthy_engine_starts_no_more_episodes():
    engine = FakeEngine()
    engine.healthy = False
    counts, client, runner = _run([Identity("5Hot", sign=lambda b: "s", episodes=5)], engine=engine)
    assert counts["5Hot"]["engine_unhealthy"] == 1 and runner.ran == [] and client.bodies == []


def test_the_generate_endpoint_listens_on_loopback_only():
    import socket

    from fastapi import FastAPI

    from reliquary.miner.agentic_miner import serve_loopback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    async def scenario():
        server, serving = await serve_loopback(FastAPI(), port)
        try:
            assert server.config.host == "127.0.0.1"
            assert {s.getsockname()[0] for srv in server.servers for s in srv.sockets} == {"127.0.0.1"}
        finally:
            server.should_exit = True
            await serving

    asyncio.run(scenario())


# -- the pre-signing check (ruling P14): the validator's own parse, before signing --

def test_a_trajectory_the_precheck_refuses_is_dropped_and_counted():
    seen = []

    def precheck(built):
        seen.append(built)
        return ("bad_turns", {"turn": 0, "why": "span does not re-render to itself"})

    client = FakeClient()
    counts = asyncio.run(mine_agentic(
        job=JOB, identities=[Identity("5Hot", sign=lambda b: "s", episodes=3)], client=client,
        engine=FakeEngine(), runners={harness_key(None): FakeRunner()},
        decode=lambda ids: "", concurrency=2, precheck=precheck))
    assert client.bodies == [] and len(seen) == 3
    assert counts["5Hot"]["precheck_refused"] == 3
    assert counts["5Hot"]["precheck_refused:bad_turns"] == 3


def test_the_precheck_runs_before_the_forgery_hook():
    forge = Identity("5Bad", sign=lambda b: "s", episodes=1,
                     transform=lambda built, index: __import__("dataclasses").replace(
                         built, final_diff="forged"))
    client = FakeClient()
    asyncio.run(mine_agentic(
        job=JOB, identities=[forge], client=client, engine=FakeEngine(),
        runners={harness_key(None): FakeRunner()}, decode=lambda ids: "", concurrency=1,
        precheck=lambda built: None))
    assert client.bodies[0]["trajectory"]["final_diff"] == "forged"


def _fake_built(turns, stop="agent_completed"):
    from reliquary.corpus.trajectory import BuiltTrajectory
    from tests.unit.test_trajectory_parse import PROMPT, build

    tokens, spans = build(turns)
    return BuiltTrajectory(tuple(PROMPT), tuple(tokens), tuple(spans),
                           tuple(("p",) for _ in spans), "", stop)


def test_trajectory_precheck_is_the_validators_parse_and_span_check():
    from reliquary.miner.agentic_miner import trajectory_precheck
    from tests.unit.test_trajectory_parse import CALL, TERM, TEXT, FakeRenderer

    honest = _fake_built([([TEXT, CALL, TERM], ["out"]), ([TEXT, TERM], None)])
    renderer = FakeRenderer()
    assert trajectory_precheck(renderer, max_turns=40)(honest) is None
    renderer.canonical = False                     # what a dropped malformed call does
    assert trajectory_precheck(renderer, max_turns=40)(honest)[0] == "bad_turns"
    assert trajectory_precheck(FakeRenderer(), max_turns=1)(honest)[0] == "bad_turns"


def test_trajectory_precheck_applies_the_lease_bounds_like_the_intake(monkeypatch):
    """F2 (ruling P23 d): the intake refuses what no grade lease can carry; the
    miner runs the same bound so an honest miner never loses a slot to it."""
    from reliquary.miner.agentic_miner import trajectory_precheck
    from reliquary.validator import corpus_grade_protocol
    from tests.unit.test_trajectory_parse import CALL, TERM, TEXT, FakeRenderer

    honest = _fake_built([([TEXT, CALL, TERM], ["a long output"]), ([TEXT, TERM], None)])
    assert trajectory_precheck(FakeRenderer(), max_turns=40)(honest) is None
    monkeypatch.setattr(corpus_grade_protocol, "MAX_OBSERVATION_CHARS", 5)
    reason, detail = trajectory_precheck(FakeRenderer(), max_turns=40)(honest)
    assert reason == "trajectory_too_large" and detail["why"] == "observation"


def test_a_real_malformed_call_trajectory_is_dropped_not_submitted():
    import os

    import pytest

    pytest.importorskip("renderers")
    tokenizer = os.environ.get("RELIQUARY_QWEN38_TOKENIZER")
    if not tokenizer:
        pytest.skip("set RELIQUARY_QWEN38_TOKENIZER")
    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary.miner.agentic_miner import trajectory_precheck
    from reliquary.protocol.toploc import span_chunk_count

    r = load_turn_renderer(tokenizer)
    enc = lambda text: tuple(r._tokenizer.encode(text, add_special_tokens=False))
    prompt = tuple(r.initial_ids("Fix the bug."))
    # What verifiers' train client did on the real stack: the unnamed call is
    # dropped, the named one answered, then the agent completes.
    first = enc("Look.\n</think>\n\n<tool_call>\n<function=>\n</function>\n</tool_call>\n<tool_call>\n"
                "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>")
    second_prompt = tuple(r.next_prompt(list(prompt), list(first), ["README.md"]))
    second = enc("Done.\n</think>\n\nAll good.<|im_end|>")
    proofs = lambda ids: tuple("p" for _ in range(span_chunk_count(len(ids), 32)))

    class RealEngine(FakeEngine):
        def take_session(self, session_id):
            return SessionLog([GeneratedTurn(prompt, first, proofs(first)),
                               GeneratedTurn(second_prompt, second, proofs(second))], True)

    client = FakeClient()
    counts = asyncio.run(mine_agentic(
        job=JOB, identities=[Identity("5Hot", sign=lambda b: "s", episodes=1)], client=client,
        engine=RealEngine(), runners={harness_key(None): FakeRunner()}, decode=lambda ids: "",
        concurrency=1, precheck=trajectory_precheck(r, max_turns=JOB.episode.max_turns)))
    assert client.bodies == [] and counts["5Hot"]["precheck_refused:bad_turns"] == 1



# -- fix round 2: one episode's failure never stops the miner --

def _mine(runner, *, engine=None, client=None, precheck=None, decode=None, episodes=6,
          concurrency=3, sign=None):
    engine = engine or FakeEngine()
    client = client or FakeClient()
    counts = asyncio.run(mine_agentic(
        job=JOB, identities=[Identity("5Hot", sign=sign or (lambda b: "s"), episodes=episodes)],
        client=client, engine=engine, runners={harness_key(None): runner},
        decode=decode or (lambda ids: ""), concurrency=concurrency, precheck=precheck))
    return counts["5Hot"], client, engine


def _walk(n):
    from reliquary.corpus.walk import job_walk_index

    return [job_walk_index(JOB, "5Hot", cursor) for cursor in range(n)]


def test_a_precheck_crash_drops_that_episode_only():
    bad = _walk(6)[2]

    def precheck(built):
        if built.final_diff == f"diff {bad}":
            raise RuntimeError("renderer exploded")
        return None

    counts, client, engine = _mine(FakeRunner(), precheck=precheck)
    assert counts["episode_crashed"] == 1 and counts["accepted"] == 5
    assert sorted(b["prompt_index"] for b in client.bodies) == sorted(i for i in _walk(6) if i != bad)
    assert f"s{bad}" in engine.dropped


def test_a_decode_or_sign_crash_drops_that_episode_only():
    calls = {"n": 0}

    def decode(ids):
        calls["n"] += 1
        if calls["n"] == 1:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        return ""

    counts, client, _ = _mine(FakeRunner(), decode=decode)
    assert counts["episode_crashed"] == 1 and len(client.bodies) == 5

    def sign(body):
        if body["prompt_index"] == _walk(6)[0]:
            raise ValueError("wallet locked")
        return "s"

    counts, client, _ = _mine(FakeRunner(), sign=sign)
    assert counts["episode_crashed"] == 1 and len(client.bodies) == 5


def test_a_non_halting_submit_error_drops_that_episode_only():
    class Flaky(FakeClient):
        def submit(self, body):
            if body["prompt_index"] == _walk(6)[1]:
                raise KeyError("reason")
            return super().submit(body)

    counts, client, _ = _mine(FakeRunner(), client=Flaky())
    assert counts["episode_crashed"] == 1 and len(client.bodies) == 5


def test_an_episode_past_its_deadline_is_dropped_and_its_session_too():
    slow = _walk(4)[1]
    runner = FakeRunner(slow={slow}, slow_seconds=30.0, deadline=0.2)
    counts, client, engine = _mine(runner, episodes=4)
    assert counts["episode_timeout"] == 1 and len(client.bodies) == 3
    assert f"s{slow}" in engine.dropped


def test_job_complete_cancels_the_episodes_still_running():
    import time

    walk = _walk(4)
    runner = FakeRunner(slow=set(walk[1:]), slow_seconds=30.0)
    client = FakeClient([{"reason": "job_complete", "accepted": False}])
    started = time.monotonic()
    counts, client, engine = _mine(runner, client=client, episodes=4, concurrency=4)
    assert time.monotonic() - started < 10
    assert len(client.bodies) == 1 and counts["job_complete"] == 1
    assert counts["episode_cancelled"] == 3
    assert {f"s{i}" for i in walk[1:]} <= set(engine.dropped)


def test_a_notice_that_differs_from_verifiers_refuses_to_mine(monkeypatch):
    import pytest

    from reliquary.miner import agentic_episode, agentic_miner

    monkeypatch.setattr(agentic_episode, "network_notice_refusal", lambda: "the notice differs")
    with pytest.raises(RuntimeError, match="the notice differs"):
        asyncio.run(agentic_miner.run_agentic_miner(
            job=JOB, checkpoint_dir="/nonexistent", proof=None, tokenizer=None,
            identities=[], client=None))


def test_the_network_notice_check_against_the_installed_verifiers(monkeypatch):
    import pytest

    base = pytest.importorskip("verifiers.v1.dialects.base")
    from reliquary.miner.agentic_episode import network_notice_refusal

    assert network_notice_refusal() is None
    monkeypatch.setattr(base, "CAPABILITY_NOTICE", "Network blocked, differently.")
    assert "differs" in network_notice_refusal()


def test_the_cli_exposes_max_num_seqs():
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    result = CliRunner().invoke(app, ["corpus", "mine-agentic", "--help"])
    assert "--max-num-seqs" in result.output


def test_the_episode_line_logs_the_graded_reward(caplog):
    # verifiers logs "rollout done: reward=0.000" before SweEnv.finalize grades
    # the patch; the miner's own line must carry the graded reward.
    import logging

    with caplog.at_level(logging.INFO, logger="reliquary.miner.agentic_miner"):
        _run([Identity("5Hot", sign=lambda b: "s", episodes=1)])
    lines = [r.getMessage() for r in caplog.records if "accepted" in r.getMessage()]
    assert lines and all("reward 1.0" in line for line in lines)



def test_a_miner_on_non_xfs_docker_storage_is_warned_not_refused():
    """F6 M3: executors replay on xfs; a miner's ext4 order spends tolerance."""
    from reliquary.miner.agentic_miner import docker_storage_warning

    warning = docker_storage_warning(refusal=lambda: "Docker stores images on ext2/ext3 "
                                                     "(/var/lib/docker), not xfs")
    assert "ext2/ext3" in warning and "tolerance" in warning
    assert docker_storage_warning(refusal=lambda: None) is None
