"""The miner in signed-sandbox mode: a signed session request, an episode on the
machine the validator named, the transcript submitted with the tokens, and every
unsubmitted session reported."""

import asyncio
import json
import os
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.job import parse_job, sandbox_split  # noqa: E402
from reliquary.corpus.signed_reasons import corpus_engagement  # noqa: E402
from reliquary.environment.agentic_swe import SignedSweSource  # noqa: E402
from reliquary.miner.agentic_episode import EpisodeResult  # noqa: E402
from reliquary.miner.agentic_miner import Identity, mine_agentic  # noqa: E402
from reliquary.miner.signed_episode import (  # noqa: E402
    HttpSandboxSessions, SignedSweEpisodeRunner,
)
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.protocol.signatures import (  # noqa: E402
    build_sandbox_close_binding, build_sandbox_open_binding,
)
from tests.unit.sandbox_fixtures import NOW, claims, signer, transcript  # noqa: E402
from tests.unit.test_agentic_miner import FakeClient, FakeEngine  # noqa: E402
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import ENV_PACKAGE, signed_episode  # noqa: E402

JOB = parse_job(_manifest(prompt_count=50, episode=signed_episode()))
# Real ss58 format-42 addresses (the validator normalises every hotkey to one).
HOT = "5CT5jwBEAhveEjgiSCQbkaKcKcUyF3VJ8qNXM9rXsuQyn3Kd"
VALIDATOR = "5CqTdCcXCC7kv1Nwq2sCeJUNVqxbBPXjMAUXrbhRJtrUiyPP"
OPEN_PATH = "/corpus/sandbox/sessions"


def _transcript(tmp_path, index=3, **overrides):
    values = dict(hotkey=HOT, engagement=corpus_engagement(JOB.job_id, index), index=index,
                  split=sandbox_split(JOB.episode), env=JOB.episode.sandbox.env,
                  checkpoint=JOB.checkpoint_sha256)
    status = overrides.pop("status", "graded")
    tools = overrides.pop("tools", ("bash", "edit"))
    values.update(overrides)
    return transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(**values),
                      status=status, state=b"diff\n", tools=tools, env_package=ENV_PACKAGE)


def _grant(signed):
    return {"session_id": "s-1", "token": signed["token"], "gateway_url": "http://10.0.0.5:8080",
            "expires_at": NOW + 4500}


class FakeSessions:
    validator_hotkey, prefix = VALIDATOR, "/corpus"

    def __init__(self, grant=None, refuse=None, close_refusals=(), clock=None):
        self.grant, self.refuse, self.opened, self.closed = grant, refuse, [], []
        self.close_refusals = list(close_refusals)
        self.clock, self.times, self.live, self.peak = clock, [], 0, 0

    def open(self, body):
        self.opened.append(body)
        if self.clock is not None:
            self.times.append(self.clock())
        if self.refuse is not None:
            raise self.refuse
        self.live += 1
        self.peak = max(self.peak, self.live)
        return self.grant

    def close(self, session_id, body):
        self.closed.append((session_id, body))
        if self.close_refusals:
            raise self.close_refusals.pop(0)
        self.live -= 1
        # A final close of a graded transcript holds the slot (closed_graded).
        return {"state": "closed_graded" if body["reason"] == "final" else "closed"}


class FakeTime:
    """The runner's monotonic clock and sleep, advanced only by its sleeps."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)
        await asyncio.sleep(0)


class FakeSandboxRunner:
    def __init__(self, signed, status="graded", state=b"diff\n", error=None, raises=None,
                 delay=0.0):
        self.transcript, self.status, self.state = signed, status, state
        self.error, self.raises, self.runs, self.delay = error, raises, [], delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, *, token, prompt, on_trace=None):
        self.runs.append((token, prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        trace = SimpleNamespace(id="t-1", stop_condition="agent_completed")
        if on_trace is not None:
            on_trace(trace)
        reward = 1.0 if self.status == "graded" else None
        return SimpleNamespace(trace=trace, state=self.state, transcript=self.transcript,
                               refusals=(), error=self.error,
                               final={"body": {"status": self.status, "reward": reward,
                                               "reason": None}})


def runner(sessions, sandbox, sleeps=None, fake_time=None, max_live=4, new_request_id=None):
    fake_time = fake_time or FakeTime()
    if sleeps is not None:
        fake_time.sleeps = sleeps

    return SignedSweEpisodeRunner(
        job=JOB, hotkey=HOT, sign_binding=lambda binding: "sig-" + binding.hex()[:16],
        sessions=sessions, model_name="m", renderer_model_dir="/ck",
        generate_url="http://127.0.0.1:1", sampling=JOB.sampling,
        source=SignedSweSource("train", prompt_of=lambda s, i: f"Fix task {i}."),
        runner_factory=lambda url: sandbox, clock=lambda: NOW,
        new_request_id=new_request_id or (lambda: "a" * 32),
        sleep=fake_time.sleep, monotonic=fake_time.monotonic, max_live=max_live)


async def _run(signed, index=3, times=1):
    seen, results = [], []
    async with signed:
        for _ in range(times):
            try:
                results.append(await signed.run(index, on_session=seen.append))
            except SessionRefused as refused:
                results.append(refused)
    return (results[0] if times == 1 else results), seen


def test_a_graded_episode_is_returned_with_its_transcript_and_released_on_demand(tmp_path, caplog):
    caplog.set_level("DEBUG", logger="reliquary.miner.signed_episode")
    signed = _transcript(tmp_path)
    sessions = FakeSessions(grant=_grant(signed))
    sandbox = FakeSandboxRunner(signed)
    result, seen = asyncio.run(_run(runner(sessions, sandbox)))
    assert result.ok and result.final_diff == "diff\n" and result.transcript == signed
    assert result.reward == 1.0 and seen == ["t-1"]
    (body,) = sessions.opened
    assert body["miner_hotkey"] == HOT and type(body["at"]) is int
    assert body["engagement"] == {"kind": "corpus", "job_id": JOB.job_id, "prompt_index": 3}
    binding = build_sandbox_open_binding(body, validator_hotkey=VALIDATOR, path=OPEN_PATH)
    assert body["signature"] == "sig-" + binding.hex()[:16]
    assert sandbox.runs == [(signed["token"], "Fix task 3.")]
    # Closed `final` before it is returned for submission (closed_graded, drain-immune).
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-1", "final")]
    assert result.submit_by == NOW + 4500 + 1800 - 60
    asyncio.run(result.release())
    session_id, close = sessions.closed[1]
    assert session_id == "s-1" and close["reason"] == "withdraw" and close["transcript"] == signed
    assert close["signature"] == "sig-" + build_sandbox_close_binding(
        close, validator_hotkey=VALIDATOR, path="/corpus/sandbox/sessions/s-1/close").hex()[:16]
    assert "sandbox session s-1" in caplog.text            # the log is captured
    assert signed["token"]["signature"] not in caplog.text


@pytest.mark.parametrize("kw,why,close", [
    (dict(status="expired"), "expired", "final"),
    (dict(status="budget_exhausted",
          error="TaskError: EpisodeClosed: 409: budget"), "EpisodeClosed", "final"),
    (dict(error="RoutingError: x"), "RoutingError", "withdraw"),
    (dict(state=b"\xff\xfe"), "UTF-8", "withdraw"),
])
def test_an_unpaid_or_unusable_episode_is_reported_at_once(tmp_path, kw, why, close):
    """A graded transcript that will not be submitted is withdrawn (slot and caps
    freed); any other final is reported as final."""
    status = kw.get("status", "graded")
    signed = _transcript(tmp_path, status=status)
    sessions = FakeSessions(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed, **kw))))
    assert not result.ok and why in result.error and result.release is None
    assert [body["reason"] for _, body in sessions.closed] == [close]
    assert sessions.closed[0][1]["transcript"] == signed


@pytest.mark.parametrize("overrides,reason", [
    (dict(index=4), "task_mismatch"),
    (dict(hotkey="5CqTdCcXCC7kv1Nwq2sCeJUNVqxbBPXjMAUXrbhRJtrUiyPP"), "hotkey_mismatch"),
    (dict(checkpoint="e" * 64), "checkpoint_mismatch"),
    (dict(tools=("bash",)), "tools"),
])
def test_a_transcript_the_validator_would_refuse_is_closed_not_returned(tmp_path, overrides,
                                                                          reason):
    signed = _transcript(tmp_path, **overrides)
    sessions = FakeSessions(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert not result.ok and reason in result.error
    assert [body["reason"] for _, body in sessions.closed] == ["withdraw"]
    assert sessions.closed[0][1]["transcript"] == signed


def test_any_failure_after_the_episode_withdraws_the_graded_session(tmp_path, monkeypatch):
    from reliquary.miner import signed_episode

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(signed_episode, "transcript_refusal", boom)
    signed = _transcript(tmp_path)
    sessions = FakeSessions(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert not result.ok and "RuntimeError" in result.error
    assert [(body["reason"], body["transcript"]) for _, body in sessions.closed] == [
        ("withdraw", signed)]


def test_a_failed_open_is_reported_without_a_transcript(tmp_path):
    signed = _transcript(tmp_path)
    sessions = FakeSessions(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(
        signed, raises=RuntimeError("503"))), ))
    assert not result.ok
    assert sessions.closed[0][1]["reason"] == "open_failed"
    assert sessions.closed[0][1]["transcript"] is None


def test_a_cancelled_episode_still_releases_its_session(tmp_path):
    signed = _transcript(tmp_path)
    sessions = FakeSessions(grant=_grant(signed))

    class Hangs(FakeSandboxRunner):
        async def run(self, **kw):
            await asyncio.sleep(3600)

    async def main():
        signed_runner = runner(sessions, Hangs(signed))
        async with signed_runner:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await signed_runner.run(3)

    asyncio.run(main())
    assert [body["reason"] for _, body in sessions.closed] == ["open_failed"]


def test_a_busy_close_is_retried_after_its_retry_after(tmp_path):
    signed = _transcript(tmp_path)
    sessions = FakeSessions(grant=_grant(signed), close_refusals=[
        SessionRefused("close_busy", retry_after=7.0, status=503),
        SessionRefused("directory_unavailable", retry_after=3.0, status=503),
        SessionRefused("body_timeout", status=408),
        SessionRefused("close_busy", retry_after=600.0, status=503)])
    sleeps = []
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed, status="expired"),
                                        sleeps=sleeps)))
    assert not result.ok
    assert [body["reason"] for _, body in sessions.closed] == ["final"] * 5
    assert sleeps[:2] == [7.0, 3.0] and 1.0 <= sleeps[2] <= 10.0 and sleeps[3] == 60.0


def test_a_refused_session_raises_before_any_episode(tmp_path):
    sandbox = FakeSandboxRunner(_transcript(tmp_path))
    sessions = FakeSessions(refuse=SessionRefused("prompt_unavailable"))
    result, _ = asyncio.run(_run(runner(sessions, sandbox)))
    assert isinstance(result, SessionRefused)
    assert sandbox.runs == []


def test_a_throttled_open_waits_its_retry_after_before_the_next_open(tmp_path):
    sessions = FakeSessions(refuse=SessionRefused("open_rate_cap", retry_after=42.0, status=429))
    sleeps = []
    asyncio.run(_run(runner(sessions, FakeSandboxRunner(_transcript(tmp_path)), sleeps=sleeps),
                     times=2))
    assert len(sessions.opened) == 2 and sleeps == [42.0]


def test_a_cancelled_open_still_closes_the_grant_it_gets(tmp_path):
    import time as wall

    signed = _transcript(tmp_path)

    class Slow(FakeSessions):
        def open(self, body):
            wall.sleep(0.2)
            return super().open(body)

    sessions = Slow(grant=_grant(signed))

    async def main():
        signed_runner = runner(sessions, FakeSandboxRunner(signed))
        async with signed_runner:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await signed_runner.run(3)

    asyncio.run(main())
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-1", "open_failed")]


def test_a_grant_lost_to_transport_errors_is_recovered_and_closed(tmp_path):
    signed = _transcript(tmp_path)
    ids = iter(f"{n:032x}" for n in range(1000))

    class Flaky(FakeSessions):
        failures = 3

        def open(self, body):
            self.opened.append(body)
            if self.failures:
                self.failures -= 1
                raise httpx.ConnectError("lost")
            return dict(self.grant, session_id="s-" + body["request_id"][-2:])

    sessions = Flaky(grant=_grant(signed))
    results, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed),
                                         new_request_id=lambda: next(ids)), times=2))
    assert isinstance(results[0], SessionRefused)
    assert results[0].reason == "validator_unreachable"
    lost = sessions.opened[0]["request_id"]
    assert [body["request_id"] for body in sessions.opened[:4]] == [lost] * 4
    assert sessions.opened[4]["request_id"] != lost
    assert sessions.closed[0] == ("s-" + lost[-2:], sessions.closed[0][1])
    assert sessions.closed[0][1]["reason"] == "open_failed"
    assert results[1].ok


def _mine_signed(signed_runner, *, episodes, concurrency=8):
    identity = Identity(HOT, sign=lambda body: "sig", episodes=episodes,
                        sign_binding=lambda binding: "s")

    async def main():
        async with signed_runner:
            return await mine_agentic(
                job=JOB, identities=[identity], client=FakeClient(), engine=FakeEngine(),
                runners={}, runner_for=lambda identity: signed_runner,
                decode=lambda ids: "".join(map(str, ids)), concurrency=concurrency)

    return asyncio.run(main())[HOT]


@pytest.mark.parametrize("refusal,halts", [
    (SessionRefused("http_404", status=404), True),
    (SessionRefused("internal_error", status=500), True),
    (SessionRefused("http_502", status=502), True),
    (SessionRefused("prompt_unavailable", status=409), False),
])
def test_refusals_without_retry_after_back_off(refusal, halts):
    fake_time = FakeTime()
    sessions = FakeSessions(refuse=refusal, clock=fake_time.monotonic)
    counts = _mine_signed(runner(sessions, FakeSandboxRunner(None), fake_time=fake_time),
                          episodes=40)
    opens = len(sessions.opened)
    assert opens <= 5 * max(fake_time.now, 1.0)
    assert all(later - earlier >= 1.0
               for earlier, later in zip(sessions.times, sessions.times[1:]))
    assert max(fake_time.sleeps) <= 60.0
    if halts:
        assert opens == 10 and counts["halted"] == 1
        assert counts["session_refused:validator_unavailable"] == 1
    else:
        assert opens == 40 and counts["halted"] == 0


def test_a_hold_does_not_end_in_a_burst():
    fake_time = FakeTime()
    sessions = FakeSessions(refuse=SessionRefused("sandbox_capacity", retry_after=42.0,
                                                  status=503), clock=fake_time.monotonic)
    counts = _mine_signed(runner(sessions, FakeSandboxRunner(None), fake_time=fake_time),
                          episodes=20)
    assert len(sessions.opened) == 20 and counts["halted"] == 0
    assert all(later - earlier >= 42.0
               for earlier, later in zip(sessions.times, sessions.times[1:]))


def test_live_sessions_never_exceed_the_per_job_cap(tmp_path):
    signed = _transcript(tmp_path, status="expired")
    sessions = FakeSessions(grant=_grant(signed))
    _mine_signed(runner(sessions, FakeSandboxRunner(signed, status="expired", delay=0.01),
                        max_live=4), episodes=16, concurrency=8)
    assert len(sessions.opened) == 16 and len(sessions.closed) == 16
    assert sessions.peak == 4


def test_a_non_format_42_hotkey_is_refused():
    with pytest.raises(ValueError):
        SignedSweEpisodeRunner(job=JOB, hotkey="5Hot", sign_binding=lambda b: "s",
                               sessions=FakeSessions(), model_name="m", renderer_model_dir="/ck",
                               generate_url="http://127.0.0.1:1", sampling=JOB.sampling)


def test_http_sessions_raise_the_validators_refusal_with_retry_after():
    def handler(request):
        if request.url.path.endswith("/close"):
            return httpx.Response(200, json={"state": "closed"})
        return httpx.Response(503, json={"reason": "sandbox_capacity", "detail": {}},
                              headers={"Retry-After": "10"})

    sessions = HttpSandboxSessions(httpx.Client(base_url="http://validator",
                                                transport=httpx.MockTransport(handler)),
                                   validator_hotkey=VALIDATOR)
    with pytest.raises(SessionRefused) as caught:
        sessions.open({})
    assert caught.value.reason == "sandbox_capacity" and caught.value.retry_after == 10.0
    assert caught.value.status == 503
    assert sessions.close("s-1", {}) == {"state": "closed"}


def test_http_sessions_default_a_throttle_without_a_usable_retry_after():
    def handler(request):
        return httpx.Response(429, json={"reason": "live_cap", "detail": {}},
                              headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})

    sessions = HttpSandboxSessions(httpx.Client(base_url="http://validator",
                                                transport=httpx.MockTransport(handler)),
                                   validator_hotkey=VALIDATOR)
    with pytest.raises(SessionRefused) as caught:
        sessions.open({})
    assert caught.value.retry_after >= 1.0


def test_http_sessions_send_compact_json_and_never_echo_the_body():
    sent = []

    def handler(request):
        sent.append(request.content)
        return httpx.Response(200, json={"state": "closed"})

    sessions = HttpSandboxSessions(httpx.Client(base_url="http://validator",
                                                transport=httpx.MockTransport(handler)),
                                   validator_hotkey=VALIDATOR)
    sessions.close("s-1", {"transcript": {"out": "é"}})
    assert sent == [json.dumps({"transcript": {"out": "é"}}, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")]


class LoopRunner:
    """The mining loop's view: what SignedSweEpisodeRunner returns."""

    def __init__(self, refuse=None, submit_by=None):
        self.released, self.refuse, self.submitted, self.submit_by = [], refuse, [], submit_by

    def deadline(self, index):
        return None

    async def run(self, index, on_session=None):
        if self.refuse is not None:
            raise self.refuse
        on_session(f"s{index}")

        async def release():
            self.released.append(index)

        return EpisodeResult(f"s{index}", f"diff {index}", "agent_completed", True, 1.0,
                             transcript={"token": {}, "records": [index]}, release=release,
                             submitted=lambda: self.submitted.append(index),
                             submit_by=self.submit_by)


def _mine(loop_runner, client=None, episodes=2, precheck=None):
    client = client or FakeClient()
    identity = Identity("5Hot", sign=lambda body: "sig", episodes=episodes,
                        sign_binding=lambda binding: "s")
    counts = asyncio.run(mine_agentic(
        job=JOB, identities=[identity], client=client, engine=FakeEngine(), runners={},
        runner_for=lambda identity: loop_runner, decode=lambda ids: "".join(map(str, ids)),
        concurrency=1, precheck=precheck))
    return counts["5Hot"], client


def test_the_submission_carries_the_transcript_and_keeps_the_session():
    loop_runner = LoopRunner()
    counts, client = _mine(loop_runner)
    assert counts["accepted"] == 2
    assert all(body["trajectory"]["transcript"]["records"] == [body["prompt_index"]]
               for body in client.bodies)
    assert loop_runner.released == [] and len(loop_runner.submitted) == 2


def test_a_submission_past_its_deadline_is_withdrawn_not_sent():
    loop_runner = LoopRunner(submit_by=1.0)
    counts, client = _mine(loop_runner)
    assert client.bodies == [] and len(loop_runner.released) == 2
    assert counts["submit_deadline_passed"] == 2


def test_a_refused_submission_releases_its_session():
    loop_runner = LoopRunner()
    counts, _ = _mine(loop_runner, client=FakeClient([{"reason": "prompt_full", "accepted": False},
                                                      {"reason": "prompt_full", "accepted": False}]))
    assert len(loop_runner.released) == 2


def test_a_precheck_refusal_releases_its_session_unsubmitted():
    loop_runner = LoopRunner()
    counts, client = _mine(loop_runner, precheck=lambda built: ("sandbox_state_mismatch", {}))
    assert client.bodies == [] and len(loop_runner.released) == 2
    assert counts["precheck_refused:sandbox_state_mismatch"] == 2


def test_a_refused_session_is_counted_not_crashed():
    counts, client = _mine(LoopRunner(refuse=SessionRefused("prompt_unavailable", retry_after=0.01)))
    assert counts["session_refused:prompt_unavailable"] == 2 and client.bodies == []
    assert counts["episode_crashed"] == 0


def test_a_permanent_session_refusal_halts_the_identity():
    counts, client = _mine(LoopRunner(refuse=SessionRefused("hotkey_not_registered")), episodes=5)
    assert counts["session_refused:hotkey_not_registered"] == 1 and counts["halted"] == 1


def test_a_bad_signature_halts_with_a_validator_hotkey_hint(caplog):
    caplog.set_level("INFO", logger="reliquary.miner.agentic_miner")
    counts, _ = _mine(LoopRunner(refuse=SessionRefused("bad_signature", status=403)))
    assert counts["halted"] == 1 and "--validator-hotkey" in caplog.text


def test_a_refused_submissions_detail_is_logged_without_its_token(caplog):
    caplog.set_level("INFO", logger="reliquary.miner.corpus_miner")
    from collections import Counter

    from reliquary.miner.corpus_miner import CorpusMinerHalted, CorpusPermanentFailure, _retry

    def call():
        raise CorpusPermanentFailure("409", status=409, detail={
            "reason": "x", "token": {"signature": "SECRET-SIG"}, "more": "y" * 5000})

    with pytest.raises(CorpusMinerHalted):
        _retry(call, sleep=lambda s: None, counts=Counter(), max_consecutive_failures=1)
    assert "corpus request failed" in caplog.text           # the log is captured
    assert "SECRET-SIG" not in caplog.text and "y" * 1000 not in caplog.text


def test_a_throttled_submission_waits_its_retry_after():
    from collections import Counter

    from reliquary.miner.corpus_miner import CorpusTransientFailure, _retry

    calls, sleeps = [], []

    def call():
        calls.append(1)
        if len(calls) == 1:
            raise CorpusTransientFailure("503 sandbox_session_busy", retry_after=25.0)
        return {"reason": "accepted"}

    assert _retry(call, sleep=sleeps.append, counts=Counter(),
                  max_consecutive_failures=5) == {"reason": "accepted"}
    assert sleeps == [25.0]


def test_the_corpus_client_reads_a_throttles_retry_after():
    from reliquary.miner.corpus_miner import CorpusTransientFailure, issue_corpus_request

    def handler(request):
        return httpx.Response(503, json={"detail": "sandbox_session_busy"},
                              headers={"Retry-After": "12"})

    http = httpx.Client(base_url="http://validator", transport=httpx.MockTransport(handler))
    with pytest.raises(CorpusTransientFailure) as caught:
        issue_corpus_request(lambda: http.post("/corpus/submit", json={}))
    assert caught.value.retry_after == 12.0


def test_the_precheck_is_the_validators_signed_parse(tmp_path):
    import dataclasses

    from reliquary.corpus.trajectory import BuiltTrajectory
    from reliquary.miner.signed_episode import signed_trajectory_precheck
    from tests.unit.test_signed_parse import CALL, S, TERM, TEXT

    prompt = S.initial_ids("Fix task 0.")
    first = [TEXT] * 8 + [CALL, TERM]
    full = S.next_prompt(prompt, first, ["a.py\n"])
    tokens = full[len(prompt):] + [TEXT] * 9 + [TERM]
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(),
                        calls=[{"turn": 0, "k": 0, "arguments": {"command": "c0"},
                                "output": "a.py\n"}], state=b"d\n")
    built = BuiltTrajectory(prompt_ids=tuple(prompt), tokens=tuple(tokens),
                            spans=((0, len(first)), (len(full) - len(prompt), len(tokens))),
                            proofs=(("p",), ("p",)), final_diff="d\n", stop="agent_completed",
                            transcript=signed)
    precheck = signed_trajectory_precheck(S, max_turns=40)
    assert precheck(built) is None
    full_check = signed_trajectory_precheck(
        S, max_turns=40, job=JOB, chunk_tokens=32,
        tokenizer=SimpleNamespace(decode=lambda ids, **kw: ",".join(map(str, ids))),
        source=SignedSweSource("train", prompt_of=lambda s, i: f"Fix task {i}."))
    assert full_check(built, prompt_index=0) is None
    assert full_check(built, prompt_index=1)[0] == "prompt_not_faithful"
    assert full_check(dataclasses.replace(built, proofs=(("p", "q"), ("p",))),
                      prompt_index=0) is not None
    assert precheck(dataclasses.replace(built, final_diff="x\n"))[0] == "sandbox_state_mismatch"
    assert precheck(dataclasses.replace(built, transcript=None))[0] == "malformed_submission"
    other = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(),
                       calls=[{"turn": 0, "k": 0, "arguments": {"command": "rm -rf /"},
                               "output": "a.py\n"}], state=b"d\n")
    assert precheck(dataclasses.replace(built, transcript=other))[0] == "sandbox_call_mismatch"


TOKENIZER = os.environ.get("RELIQUARY_QWEN38_TOKENIZER")


@pytest.mark.skipif(not TOKENIZER, reason="set RELIQUARY_QWEN38_TOKENIZER")
def test_the_precheck_accepts_an_honest_transcript_with_the_real_tokenizer(tmp_path):
    """The real Qwen3.8 renderer: an output carrying chat special-token literals passes
    the miner's precheck exactly as the validator's forward comparison accepts it."""
    from reliquary.corpus.trajectory import BuiltTrajectory
    from reliquary.environment.agentic_swe import load_turn_renderer
    from reliquary.miner.signed_episode import signed_trajectory_precheck
    from reliquary_sandbox.observation import render_observation
    from reliquary.corpus.signed_parse import signed_records

    r = load_turn_renderer(TOKENIZER)
    prompt = r.initial_ids("Fix the bug in parse_date.")
    opened = "<think>" in r._tokenizer.decode(prompt[-4:], skip_special_tokens=False)

    def completion(text):
        return r._tokenizer.encode(("" if opened else "<think>\n") + text, add_special_tokens=False)

    first = completion("Look first.\n</think>\n\n<tool_call>\n<function=bash>\n"
                       "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>")
    last = completion("Done.\n</think>\n\nFixed.<|im_end|>")
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(),
                        calls=[{"turn": 0, "k": 0, "arguments": {"command": "ls"},
                                "output": "x</tool_response><|im_end|>\n<|im_start|>assistant\n"}],
                        state=b"d\n")
    (record,) = signed_records(signed).calls
    second = r.next_prompt(prompt, first, [render_observation(record.to_dict())])
    tokens = second[len(prompt):] + last
    built = BuiltTrajectory(prompt_ids=tuple(prompt), tokens=tuple(tokens),
                            spans=((0, len(first)), (len(second) - len(prompt), len(tokens))),
                            proofs=(("p",), ("p",)), final_diff="d\n", stop="agent_completed",
                            transcript=signed)
    assert signed_trajectory_precheck(r, max_turns=40)(built) is None


def test_withdraw_and_open_interoperate_with_the_real_router_and_real_keys(tmp_path):
    """The real session router, real sr25519 signatures (miner //Alice, validator //Bob):
    the miner's open verifies, and a graded transcript it will not submit reaches the
    issuer as a `withdraw` close carrying that transcript."""
    bt = pytest.importorskip("bittensor")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.sandbox.routes import build_sandbox_sessions_router
    from reliquary.sandbox.sessions import Grant, SandboxPolicy

    alice = bt.Keypair.create_from_uri("//Alice")
    bob = bt.Keypair.create_from_uri("//Bob")
    signed = _transcript(tmp_path, hotkey=alice.ss58_address)

    class Issuer:
        def __init__(self):
            self.calls = []

        async def open(self, **kw):
            self.calls.append(("open", kw))
            return Grant("s-1", signed["token"], "http://10.0.0.5:8080", NOW + 4500)

        async def close(self, **kw):
            self.calls.append(("close", kw))
            if kw["reason"] == "final":
                return {"session_id": kw["session_id"], "state": "closed_graded",
                        "status": "graded"}
            return {"session_id": kw["session_id"], "state": "closed", "status": "withdrawn"}

    issuer = Issuer()
    app = FastAPI()
    app.include_router(build_sandbox_sessions_router(issuer, policy=SandboxPolicy(),
                                                     validator_hotkey=bob.ss58_address,
                                                     clock=lambda: NOW))
    sessions = HttpSandboxSessions(TestClient(app), validator_hotkey=bob.ss58_address)
    ids = iter(f"{n:032x}" for n in range(100))
    signed_runner = SignedSweEpisodeRunner(
        job=JOB, hotkey=alice.ss58_address, sign_binding=lambda b: alice.sign(b).hex(),
        sessions=sessions, model_name="m", renderer_model_dir="/ck",
        generate_url="http://127.0.0.1:1", sampling=JOB.sampling,
        source=SignedSweSource("train", prompt_of=lambda s, i: f"Fix task {i}."),
        runner_factory=lambda url: FakeSandboxRunner(signed), clock=lambda: NOW,
        new_request_id=lambda: next(ids))
    result, _ = asyncio.run(_run(signed_runner))
    assert result.ok
    asyncio.run(result.release())
    (_, opened), (_, final), (_, closed) = issuer.calls
    assert final["reason"] == "final" and final["transcript"] == signed
    assert opened["hotkey"] == alice.ss58_address
    assert opened["engagement"] == {"kind": "corpus", "job_id": JOB.job_id, "prompt_index": 3}
    assert closed["reason"] == "withdraw" and closed["transcript"] == signed


# -- final review: I1 (named throttles), I3 (a) (close final before submitting), M1, M2 ---

@pytest.mark.parametrize("reason", [
    "job_not_ready", "ledger_unavailable", "directory_unavailable", "store_unavailable",
    "sandbox_capacity", "close_busy", "session_busy",
])
def test_the_validators_named_throttles_never_stop_the_hotkey(reason):
    fake_time = FakeTime()
    sessions = FakeSessions(refuse=SessionRefused(reason, retry_after=7.0, status=503),
                            clock=fake_time.monotonic)
    counts = _mine_signed(runner(sessions, FakeSandboxRunner(None), fake_time=fake_time),
                          episodes=15)
    assert len(sessions.opened) == 15 and counts["halted"] == 0
    assert all(later - earlier >= 7.0
               for earlier, later in zip(sessions.times, sessions.times[1:]))


class AnsweringSessions(FakeSessions):
    """Closes answer the session's state as the validator would."""

    def __init__(self, *args, close_state="closed_graded", **kw):
        super().__init__(*args, **kw)
        self.close_state = close_state

    def close(self, session_id, body):
        self.closed.append((session_id, body))
        if self.close_refusals:
            raise self.close_refusals.pop(0)
        state = self.close_state if body["reason"] == "final" else "closed"
        return {"session_id": session_id, "state": state, "status": "graded"}


def test_a_graded_transcript_is_closed_final_before_it_is_submitted(tmp_path):
    """I3 (a): `closed_graded` holds the slot and is immune to a drain, so the session
    stays payable while the submission travels."""
    signed = _transcript(tmp_path)
    sessions = AnsweringSessions(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert result.ok and result.transcript == signed and result.release is not None
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-1", "final")]
    assert sessions.closed[0][1]["transcript"] == signed


def test_a_final_close_that_fails_still_lets_the_transcript_be_submitted(tmp_path):
    signed = _transcript(tmp_path)
    refusals = [SessionRefused("transcript_invalid", status=409)]
    sessions = AnsweringSessions(grant=_grant(signed), close_refusals=refusals)
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert result.ok and result.release is not None
    assert [body["reason"] for _, body in sessions.closed] == ["final"]


@pytest.mark.parametrize("state", ["voided", "lapsed", "closed"])
def test_a_session_the_validator_will_not_pay_is_not_submitted(tmp_path, state):
    signed = _transcript(tmp_path)
    sessions = AnsweringSessions(grant=_grant(signed), close_state=state)
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert not result.ok and state in result.error and result.release is None
    assert [body["reason"] for _, body in sessions.closed] == ["final"]


def test_a_reused_request_for_a_live_session_closes_it(tmp_path):
    """M1: after a validator restart a resent open answers `request_reused` (the token
    is gone); its session is still live and holds a slot: the miner closes it."""
    signed = _transcript(tmp_path)
    ids = iter(f"{n:032x}" for n in range(1000))

    class Restarted(FakeSessions):
        sent = 0

        def open(self, body):
            self.opened.append(body)
            self.sent += 1
            if self.sent == 1:
                raise httpx.ConnectError("lost")          # granted, answer lost
            raise SessionRefused("request_reused", {"session_id": "s-lost", "state": "live"},
                                 status=409)

    sessions = Restarted(grant=_grant(signed))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed),
                                        new_request_id=lambda: next(ids))))
    assert isinstance(result, SessionRefused) and result.reason == "request_reused"
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-lost", "open_failed")]


def test_a_reused_request_for_a_session_already_ended_is_not_closed(tmp_path):
    signed = _transcript(tmp_path)
    sessions = FakeSessions(refuse=SessionRefused(
        "request_reused", {"session_id": "s-old", "state": "closed"}, status=409))
    result, _ = asyncio.run(_run(runner(sessions, FakeSandboxRunner(signed))))
    assert isinstance(result, SessionRefused) and sessions.closed == []


def test_a_submission_retried_past_its_deadline_is_withdrawn(monkeypatch):
    """M2: the submit retries stop at `submit_by`; the session is then withdrawn."""
    import time as _time

    from reliquary.miner import agentic_miner
    from reliquary.miner.corpus_miner import CorpusTransientFailure

    slept = []

    def sleep(seconds):                   # never really sleeps; a runaway retry loop ends
        slept.append(seconds)
        if len(slept) > 3:
            raise RuntimeError("still retrying past the deadline")

    monkeypatch.setattr(agentic_miner.time, "sleep", sleep)

    class Busy(FakeClient):
        def submit(self, body):
            self.bodies.append(body)
            raise CorpusTransientFailure("503 sandbox_directory_unavailable", retry_after=600.0)

    loop_runner = LoopRunner(submit_by=_time.time() + 300)
    counts, client = _mine(loop_runner, client=Busy(), episodes=1)
    assert len(client.bodies) == 1 and slept == []           # 600 s would pass the deadline
    assert len(loop_runner.released) == 1
    assert counts["submit_deadline_passed"] == 1 and counts["episode_crashed"] == 0
