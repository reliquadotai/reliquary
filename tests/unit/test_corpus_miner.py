"""The miner walks its own order, proves what it generated, and resyncs on refusal."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from reliquary.corpus.encoding import prompt_token_ids
from reliquary.corpus.walk import walk_index
from reliquary.miner.corpus_miner import (
    CorpusMinerHalted,
    CorpusPermanentFailure,
    CorpusTransientFailure,
    Generation,
    VllmGenerator,
    build_submission,
    mine_steps,
)

EOS = 99


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def _job(prompt_count=50, n=2):
    return SimpleNamespace(job_id="math-v1", prompt_count=prompt_count, eos_token_id=EOS,
                           checkpoint_sha256="a" * 64, sampling=SimpleNamespace(n=n),
                           prompt_order="miner_walk")


class _Generator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt_ids, n):
        self.prompts.append(prompt_ids)
        return [Generation(tokens=[104, 105, EOS], proofs=["AAAA"]) for _ in range(n)]


class _Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.submitted = []
        self.cursor_reads = 0
        self.position = 0

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return self.position

    def submit(self, body):
        self.submitted.append(body)
        answer = self.answers.pop(0)
        if answer == "accepted":
            self.position += 1
        return {"reason": answer, "accepted": answer == "accepted"}


def test_the_submission_carries_the_walk_prompt_and_its_text():
    body = build_submission(job=_job(), hotkey="5Hot", cursor=0, prompt_index=7, rendered_prompt="q7",
                            generations=[Generation([104, 105, EOS], ["AAAA"])],
                            tokenizer=_Tokenizer(), sign=lambda b: "sig")
    assert body["prompt_index"] == 7 and body["signature"] == "sig"
    assert body["completions"] == [{"tokens": [104, 105, EOS], "text": "hi", "proofs": ["AAAA"]}]


def test_the_miner_follows_its_own_walk():
    client, generator = _Client(["accepted"] * 3), _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator, tokenizer=_Tokenizer(),
               render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=3)
    assert [b["prompt_index"] for b in client.submitted] == [walk_index("math-v1", "5Hot", c, 50) for c in range(3)]
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2]


def test_a_refused_step_resynchronises_the_cursor():
    client = _Client(["prompt_full", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=2)
    assert counts == {"prompt_full": 1, "accepted": 1}
    assert client.cursor_reads >= 2


def test_a_complete_job_stops_the_miner():
    client = _Client(["job_complete", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=5)
    assert counts == {"job_complete": 1} and len(client.submitted) == 1


def test_miner_and_auditor_tokenize_the_prompt_identically():
    generator = _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=_Client(["accepted"]), generator=generator,
               tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=1)
    index = walk_index("math-v1", "5Hot", 0, 50)
    assert generator.prompts == [prompt_token_ids(_Tokenizer(), f"q{index}")]


# --- fix round 1: transient/permanent HTTP failures, and a generation failure ---


class _FlakyOnceClient:
    """``submit`` fails transiently once, then accepts."""

    def __init__(self):
        self.position = 0
        self.cursor_reads = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return self.position

    def submit(self, body):
        self.submit_calls.append(body)
        if len(self.submit_calls) == 1:
            raise CorpusTransientFailure("503 ledger contention")
        self.position += 1
        return {"reason": "accepted", "accepted": True}


def test_a_transient_submit_failure_retries_the_identical_body():
    client, generator, sleeps = _FlakyOnceClient(), _Generator(), []
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=1, sleep=sleeps.append)
    assert len(generator.prompts) == 1, "one generation, not one per retry"
    assert len(client.submit_calls) == 2
    assert client.submit_calls[0] == client.submit_calls[1], "the SAME signed body, not a fresh one"
    assert counts == {"accepted": 1}
    assert sleeps == [1.0]


class _CursorFlakyOnceClient:
    """The initial ``cursor`` read fails transiently once, then succeeds."""

    def __init__(self):
        self.position = 0
        self.cursor_calls = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_calls += 1
        if self.cursor_calls == 1:
            raise CorpusTransientFailure("timeout reading cursor")
        return self.position

    def submit(self, body):
        self.submit_calls.append(body)
        self.position += 1
        return {"reason": "accepted", "accepted": True}


def test_a_transient_cursor_failure_retries_and_does_not_crash():
    client, generator, sleeps = _CursorFlakyOnceClient(), _Generator(), []
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=1, sleep=sleeps.append)
    assert client.cursor_calls == 2
    assert counts == {"accepted": 1}
    assert sleeps == [1.0]


class _AlwaysFailingClient:
    """``submit`` always answers with a permanent failure (e.g. a 404 the
    job route sends once the job is gone)."""

    def __init__(self):
        self.cursor_reads = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return 0

    def submit(self, body):
        self.submit_calls.append(body)
        raise CorpusPermanentFailure(
            "404 corpus_job_unknown", status=404, detail={"detail": "corpus_job_unknown"}
        )


def test_a_permanent_failure_stops_the_miner_after_k_consecutive():
    client, generator, sleeps = _AlwaysFailingClient(), _Generator(), []
    with pytest.raises(CorpusMinerHalted) as excinfo:
        mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                   tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                   max_steps=None, sleep=sleeps.append, max_consecutive_failures=3)
    assert len(client.submit_calls) == 3
    assert len(generator.prompts) == 1, "the retries resend the SAME body, no fresh generation per attempt"
    assert excinfo.value.counts == {"permanent_failure": 3}
    assert len(sleeps) == 2, "no backoff after the failure that crosses the threshold"


class _FlakyGenerator:
    """Raises once (as ``completion_rows`` would on a vLLM preemption), then
    generates normally."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt_ids, n):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("3 rows for 5 tokens: rows are missing")
        return [Generation(tokens=[104, 105, EOS], proofs=["AAAA"]) for _ in range(n)]


def test_a_generation_failure_drops_the_step_and_continues():
    client, generator = _Client(["accepted"]), _FlakyGenerator()
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=2, sleep=lambda s: None)
    assert generator.calls == 2
    assert len(client.submitted) == 1, "the failed step submits nothing"
    assert counts == {"generation_failed": 1, "accepted": 1}
    assert client.cursor_reads == 2, "one initial read, one resync after the dropped step"


def test_the_vllm_generator_stops_only_on_the_jobs_eos(monkeypatch):
    """SamplingParams must stop on the job's eos, not whatever the checkpoint's
    own generation_config lists: vLLM stopping on a DIFFERENT terminator would
    get an honest completion judged (and refused, ``bad_termination``) against
    an eos it never produced."""

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGPUModelRunner:
        def execute_model(self, *a, **kw): ...

        def _model_forward(self, *a, **kw): ...

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_gpu_model_runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
    fake_gpu_model_runner_module.GPUModelRunner = _FakeGPUModelRunner

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", fake_gpu_model_runner_module)

    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    generator = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)

    assert generator._params.stop_token_ids == [EOS]
    assert generator._params.ignore_eos is True
    assert generator._params.min_tokens == 2
    assert generator._params.max_tokens == 64
