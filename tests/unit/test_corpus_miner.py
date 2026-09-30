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


def _job(prompt_count=50, n=2, prompt_start=0):
    return SimpleNamespace(job_id="math-v1", prompt_count=prompt_count,
                           prompt_start=prompt_start, eos_token_id=EOS,
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


def test_a_started_job_renders_and_submits_the_source_row():
    """The miner renders and submits the SOURCE index, the one the route's
    fidelity check renders: the walk shifted by the job's start."""
    rendered = []
    client, generator = _Client(["accepted"] * 4), _Generator()

    def render(index):
        rendered.append(index)
        return f"q{index}"

    mine_steps(job=_job(prompt_start=7000), hotkey="5Hot", client=client, generator=generator,
               tokenizer=_Tokenizer(), render=render, sign=lambda b: "sig", max_steps=4)
    expected = [7000 + walk_index("math-v1", "5Hot", c, 50) for c in range(4)]
    assert rendered == expected
    assert [b["prompt_index"] for b in client.submitted] == expected
    assert [b["rendered_prompt"] for b in client.submitted] == [f"q{i}" for i in expected]
    assert all(7000 <= i < 7050 for i in expected)


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


@pytest.mark.parametrize("reason", ["hotkey_not_registered", "miner_banned"])
def test_a_refusal_no_retry_can_change_stops_the_miner(reason):
    """Generating on would burn the card for nothing: stop and say why."""
    client = _Client([reason, "accepted"])
    with pytest.raises(CorpusMinerHalted, match=reason):
        mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                   tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                   max_steps=5)
    assert len(client.submitted) == 1


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


def _install_fake_vllm(monkeypatch):
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


def test_the_vllm_generator_stops_only_on_the_jobs_eos(monkeypatch):
    """SamplingParams must stop on the job's eos, not whatever the checkpoint's
    own generation_config lists: vLLM stopping on a DIFFERENT terminator would
    get an honest completion judged (and refused, ``bad_termination``) against
    an eos it never produced."""

    _install_fake_vllm(monkeypatch)

    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    generator = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)

    assert generator._params.stop_token_ids == [EOS]
    assert generator._params.ignore_eos is True
    assert generator._params.min_tokens == 2
    assert generator._params.max_tokens == 64


def test_the_vllm_generator_leaves_the_memory_share_to_vllm_unless_told(monkeypatch):
    """A validator sharing the card needs vLLM to take less than its default
    share; unset, vLLM keeps its own default rather than one we guessed."""
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)

    default = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    shared = VllmGenerator("/fake/checkpoint", sampling, proof, EOS, gpu_memory_utilization=0.5)

    assert "gpu_memory_utilization" not in default._llm.kwargs
    assert shared._llm.kwargs["gpu_memory_utilization"] == 0.5


def test_two_vllm_generators_do_not_share_a_sampling_seed(monkeypatch):
    """vLLM seeds every engine with 0 by default, so two miners sampling the
    same prompt at the same step drew byte-identical completions and the second
    was refused `hash_duplicate` (rehearsal 2026-09-25): each process seeds its
    own engine at random."""
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)

    seeds = {VllmGenerator("/fake/checkpoint", sampling, proof, EOS)._llm.kwargs.get("seed")
             for _ in range(4)}

    assert None not in seeds and 0 not in seeds
    assert len(seeds) == 4


# --- final review, finding 3: gateway statuses are transient too ---


def _response(status, body=b'{"ok": true}'):
    import httpx

    return httpx.Response(status, content=body,
                          request=httpx.Request("POST", "http://validator/corpus/submit"))


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_gateway_or_unavailable_status_is_transient(status):
    from reliquary.miner.corpus_miner import issue_corpus_request

    with pytest.raises(CorpusTransientFailure):
        issue_corpus_request(lambda: _response(status, b"{}"))


@pytest.mark.parametrize("status", [400, 404, 422, 500])
def test_other_error_statuses_stay_permanent(status):
    from reliquary.miner.corpus_miner import issue_corpus_request

    with pytest.raises(CorpusPermanentFailure) as caught:
        issue_corpus_request(lambda: _response(status, b'{"detail": "x"}'))
    assert caught.value.status == status


def test_a_transport_error_is_transient_and_a_body_is_returned():
    import httpx

    from reliquary.miner.corpus_miner import issue_corpus_request

    def _fail():
        raise httpx.ConnectError("refused")

    with pytest.raises(CorpusTransientFailure):
        issue_corpus_request(_fail)
    assert issue_corpus_request(lambda: _response(200)) == {"ok": True}


def test_the_vllm_generator_sizes_its_context_to_the_job(monkeypatch):
    """vLLM otherwise reserves the checkpoint's own maximum length (262,144 on
    Qwen3.8-27B), whose KV cache does not fit one H100 next to the weights."""
    from reliquary.miner.corpus_miner import MAX_NUM_SEQS, PROMPT_ALLOWANCE_TOKENS

    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=32768)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    generator = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    assert generator._llm.kwargs["max_model_len"] == 32768 + PROMPT_ALLOWANCE_TOKENS
    # vLLM's default of 1,024 concurrent sequences exceeds a hybrid model's
    # Mamba cache on one H100 (320 on Qwen3.8-27B); a step runs only n of them.
    assert generator._llm.kwargs["max_num_seqs"] == MAX_NUM_SEQS


def test_a_vision_checkpoint_is_served_text_only(monkeypatch, tmp_path):
    """The job's prompts are text: a multimodal checkpoint must not reserve its
    vision encoder's profiling memory."""
    import json

    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": {"depth": 27}}))
    vision = VllmGenerator(str(tmp_path), sampling, proof, EOS)
    text = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    assert vision._llm.kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert "limit_mm_per_prompt" not in text._llm.kwargs


# --------------------------------------------------------------------------
# HttpCorpusClient: the legacy paths, or one job's paths with --job-id
# --------------------------------------------------------------------------


def _validator(jobs):
    """A validator serving ``jobs`` (job_id -> manifest), answering like the real routes."""
    import httpx

    seen = []

    def handle(request):
        seen.append((request.method, request.url.path))
        path = request.url.path
        if path == "/corpus/jobs":
            return httpx.Response(200, json={"jobs": sorted(jobs)})
        if path == "/corpus/job":
            return httpx.Response(200, json=next(iter(jobs.values())))
        if path == "/corpus/cursor/5Hot":
            return httpx.Response(200, json={"hotkey": "5Hot", "cursor": 3})
        if path.startswith("/corpus/jobs/"):
            _, _, _, job_id, what, *rest = path.split("/")
            if job_id not in jobs:
                return httpx.Response(404, json={"detail": "corpus_job_not_served"})
            if what == "job":
                return httpx.Response(200, json=jobs[job_id])
            return httpx.Response(200, json={"hotkey": rest[0], "cursor": 7})
        if path == "/corpus/submit":
            return httpx.Response(200, json={"accepted": True, "reason": "accepted"})
        return httpx.Response(404, json={"detail": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://validator"), seen


def test_without_a_job_id_the_client_uses_the_legacy_paths():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, seen = _validator({"math": {"job_id": "math"}})
    client = HttpCorpusClient(http)

    assert client.job() == {"job_id": "math"}
    assert client.cursor("5Hot") == 3
    assert client.submit({"job_id": "math"})["accepted"] is True
    assert [p for _, p in seen] == ["/corpus/job", "/corpus/cursor/5Hot", "/corpus/submit"]


def test_with_a_job_id_the_client_uses_that_jobs_paths():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, seen = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})
    client = HttpCorpusClient(http, job_id="code")

    assert client.job() == {"job_id": "code"}
    assert client.cursor("5Hot") == 7
    assert client.submit({"job_id": "code"})["accepted"] is True
    assert [p for _, p in seen] == ["/corpus/jobs/code/job", "/corpus/jobs/code/cursor/5Hot",
                                    "/corpus/submit"]


def test_a_multi_job_validator_without_a_job_id_gives_its_first_listed_job():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, _ = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})
    client = HttpCorpusClient(http)

    assert client.job() == {"job_id": "math"}
    assert client.served_jobs() == ["code", "math"]


def test_a_job_id_the_validator_does_not_serve_names_the_ones_it_does():
    from reliquary.miner.corpus_miner import CorpusJobSelectionError, HttpCorpusClient

    http, _ = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})

    with pytest.raises(CorpusJobSelectionError) as caught:
        HttpCorpusClient(http, job_id="nope").job()
    assert "nope" in str(caught.value) and "code" in str(caught.value)


def test_corpus_mine_on_a_multi_job_validator_without_job_id_mines_the_default(monkeypatch):
    """Adding a job must not halt a live miner: it keeps mining the validator's
    default job, and is told the others exist."""
    from types import SimpleNamespace

    import bittensor
    import httpx
    import huggingface_hub
    from typer.testing import CliRunner

    import reliquary.protocol.profiles as profiles
    from reliquary.cli.main import app
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR, TOPLOC_DEPLOYED_DEFAULTS

    class _Downloading(Exception):
        pass

    def download(repo, revision=None):
        raise _Downloading(repo)

    from tests.unit.test_corpus_service import _manifest

    http, seen = _validator({"math": {**_manifest(), "job_id": "math"}, "code": {"job_id": "code"}})
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, "/unused-profile-is-patched")
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE",
                        SimpleNamespace(profile_id="p", proofs=(TOPLOC_DEPLOYED_DEFAULTS,)))
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(httpx, "Client", lambda **kw: http)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    result = CliRunner().invoke(app, ["corpus", "mine", "--validator-url", "http://validator"])

    assert isinstance(result.exception, _Downloading), (result.output, result.exception)
    assert "mining job math" in result.output and "code" in result.output
    assert "--job-id" in result.output


def _single_job_validator():
    """A validator from before several jobs: no /corpus/jobs routes at all."""
    import httpx

    def handle(request):
        if request.url.path == "/corpus/job":
            return httpx.Response(200, json={"job_id": "math"})
        return httpx.Response(404, json={"detail": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://validator")


@pytest.mark.parametrize("read", ["job", "contract"])
def test_a_job_id_against_a_single_job_validator_says_to_drop_it(read):
    from reliquary.miner.corpus_miner import CorpusJobSelectionError, HttpCorpusClient

    client = HttpCorpusClient(_single_job_validator(), job_id="math")
    with pytest.raises(CorpusJobSelectionError) as caught:
        getattr(client, read)()
    assert "single job" in str(caught.value) and "--job-id" in str(caught.value)
