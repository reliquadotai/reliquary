"""A broken generation releases its lease and cannot trap the order follower."""

from types import SimpleNamespace

import pytest

from reliquary.miner.corpus_miner import (
    CorpusMinerHalted, CorpusPermanentFailure, CorpusTransientFailure, mine_steps,
)
from tests.unit.test_draw_leases import _Gen, _granted, _LeasingClient, _Tok
from tests.unit.test_eval_draw import _job


def _mine(client, generator, *, render=lambda i: f"prompt {i}", tokenizer=None,
          sign=lambda body: "sig", **kwargs):
    return mine_steps(job=_job(), hotkey="5A", client=client, generator=generator,
                      tokenizer=tokenizer or _Tok(), render=render, sign=sign,
                      sign_lease=lambda body: "lsig", **kwargs)


def _client(count):
    return _LeasingClient([_granted(i, 0, f"{i:032x}") for i in range(count)])


def test_persistent_bad_context_releases_its_leases_and_halts():
    class BadContext:
        def generate_drawn(self, prompt_ids, draws):
            raise ValueError("prompt exceeds configured context")

    client, slept = _client(3), []
    with pytest.raises(CorpusMinerHalted, match="2 consecutive generation failures") as caught:
        _mine(client, BadContext(), sleep=slept.append, max_consecutive_failures=2)
    assert caught.value.counts == {"generation_failed": 2}
    assert client.released == [f"{i:032x}" for i in range(2)]
    assert len(client.lease_bodies) == 2 and client.submitted == []
    assert slept == [1.0]


def test_a_successful_generation_resets_the_failure_streak():
    client, slept = _client(4), []
    counts = _mine(client, _Gen(fail=(1, 3)), sleep=slept.append, max_consecutive_failures=2, max_steps=4)
    assert counts == {"generation_failed": 2, "accepted": 2}
    assert client.released == [f"{i:032x}" for i in (0, 2)]
    assert client.submitted == [(1, 0), (3, 0)] and slept == [1.0, 1.0]


@pytest.mark.parametrize("stage,error", [
    ("render", ValueError("unusable prompt")),
    ("tokenize", RuntimeError("tokenizer failed")),
    ("generate", RuntimeError("CUDA out of memory")),
])
def test_unexpected_generation_errors_release_before_propagating(stage, error):
    def fail(*args, **kwargs):
        raise error

    client = _client(1)
    kwargs = {"render": fail} if stage == "render" else {}
    if stage == "tokenize":
        kwargs["tokenizer"] = SimpleNamespace(encode=fail)
    generator = SimpleNamespace(generate_drawn=fail) if stage == "generate" else _Gen()
    with pytest.raises(type(error)) as caught:
        _mine(client, generator, sleep=lambda seconds: None, max_steps=1, **kwargs)
    assert caught.value is error
    assert client.released == [f"{0:032x}"] and client.submitted == []


def test_a_local_signing_error_releases_before_any_submission():
    error = RuntimeError("signer unavailable")

    def fail(body):
        raise error

    client = _client(1)
    with pytest.raises(RuntimeError) as caught:
        _mine(client, _Gen(), sign=fail, sleep=lambda seconds: None, max_steps=1)
    assert caught.value is error
    assert client.released == [f"{0:032x}"] and client.submitted == []


def test_a_release_error_does_not_mask_the_generation_error():
    error = RuntimeError("CUDA out of memory")

    def fail(*args):
        raise error

    class FailedRelease(_LeasingClient):
        def release_lease(self, body):
            raise ConnectionError("release unavailable")

    client = FailedRelease([_granted(0, 0, f"{0:032x}")])
    with pytest.raises(RuntimeError) as caught:
        _mine(client, SimpleNamespace(generate_drawn=fail), sleep=lambda seconds: None, max_steps=1)
    assert caught.value is error


@pytest.mark.parametrize("release_error", [
    CorpusTransientFailure("503 release unavailable"),
    CorpusPermanentFailure("500 release failed", status=500),
])
def test_expected_release_failures_are_bounded_and_preserve_the_generation_error(release_error):
    error = RuntimeError("CUDA out of memory")

    def fail(*args):
        raise error

    class FailedRelease(_LeasingClient):
        def release_lease(self, body):
            self.released.append(body["lease_id"])
            raise release_error

    client, slept = FailedRelease([_granted(0, 0, f"{0:032x}")]), []
    with pytest.raises(RuntimeError) as caught:
        _mine(client, SimpleNamespace(generate_drawn=fail), sleep=slept.append,
              max_consecutive_failures=3, max_steps=1)
    assert caught.value is error
    assert client.released == [f"{0:032x}"] * 3 and slept == [1.0, 2.0]


def test_a_long_retry_after_does_not_hold_the_generation_error_or_retry_early():
    error = RuntimeError("CUDA out of memory")

    def fail(*args):
        raise error

    class FailedRelease(_LeasingClient):
        def release_lease(self, body):
            self.released.append(body["lease_id"])
            raise CorpusTransientFailure("503 release unavailable", retry_after=3600.0)

    client, slept = FailedRelease([_granted(0, 0, f"{0:032x}")]), []
    with pytest.raises(RuntimeError) as caught:
        _mine(client, SimpleNamespace(generate_drawn=fail), sleep=slept.append, max_steps=1)
    assert caught.value is error
    assert client.released == [f"{0:032x}"] and slept == []


def test_an_uncertain_submission_failure_does_not_release_the_lease():
    class FailedSubmit(_LeasingClient):
        def submit(self, body):
            raise CorpusPermanentFailure("submission response unusable", status=500)

    client = FailedSubmit([_granted(0, 0, f"{0:032x}")])
    with pytest.raises(CorpusMinerHalted):
        _mine(client, _Gen(), sleep=lambda seconds: None, max_consecutive_failures=1, max_steps=1)
    assert client.released == []


@pytest.mark.parametrize("accepted,reason,released", [
    (True, "accepted", []),
    (False, "bad_termination", [f"{0:032x}"]),
])
def test_submission_still_settles_or_releases_its_lease(accepted, reason, released):
    class Answering(_LeasingClient):
        def submit(self, body):
            self.submitted.append((body["prompt_index"], body["cursor"]))
            return {"accepted": accepted, "reason": reason}

    client = Answering([_granted(0, 0, f"{0:032x}")])
    counts = _mine(client, _Gen(), sleep=lambda seconds: None, max_steps=1)
    assert counts == {reason: 1}
    assert client.submitted == [(0, 0)] and client.released == released
