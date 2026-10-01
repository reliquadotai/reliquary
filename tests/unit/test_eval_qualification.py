"""Per-model qualification: band -> thresholds, the control's queue, the
executor's decode-and-verify (design v2, item 4)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from reliquary.eval import qualification as qual
from reliquary.eval.qualify_executor import QualifyExecutor, qualify_lease, spread
from reliquary.eval.qualify_protocol import QualifyResult
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit_remote import LeaseRefused
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2


def test_thresholds_are_the_band_p99_with_margin_between_floor_and_ceiling():
    chunks = [(k, k / 2, k / 4) for k in range(1, 101)]
    band = qual.band_of(chunks)
    assert band["exp_mismatch"] == 99 and band["chunks"] == 100
    verdict = qual.thresholds_from_band(band)
    # 99 x 1.5 = 149 is clamped to the 120 ceiling; 74.25 stands; 37.1 -> floor 40.
    assert verdict["thresholds"] == {"exp_mismatch_threshold": 120,
                                     "mant_mean_threshold": 74.25,
                                     "mant_median_threshold": 40.0}
    assert verdict["clamped"] == ["exp_mismatch"] and verdict["refused"] is None
    tight = qual.thresholds_from_band(qual.band_of([(1, 1.0, 1.0)] * 10))
    assert tight["thresholds"] == {"exp_mismatch_threshold": 60, "mant_mean_threshold": 40.0,
                                   "mant_median_threshold": 40.0}


def test_an_honest_band_over_the_threshold_ceiling_is_refused(monkeypatch):
    monkeypatch.setenv("RELIQUARY_QUALIFY_THRESHOLD_CEILING_EXP", "100")
    verdict = qual.thresholds_from_band(qual.band_of([(150, 1.0, 1.0)] * 10))
    assert verdict["refused"] and "exp_mismatch" in verdict["refused"]
    assert verdict["thresholds"]["exp_mismatch_threshold"] == 100  # never over the ceiling
    with pytest.raises(ValueError):
        qual.band_of([])


def test_job_thresholds_stay_between_floor_and_ceiling():
    good = {"exp_mismatch_threshold": 75, "mant_mean_threshold": 41.0,
            "mant_median_threshold": 40.0}
    assert qual.check_thresholds(good) == good
    with pytest.raises(ValueError, match="ceiling"):
        qual.check_thresholds({**good, "exp_mismatch_threshold": 359})
    with pytest.raises(ValueError, match="ceiling"):
        qual.check_thresholds({**good, "mant_mean_threshold": 238.5})


def test_bands_agree_within_a_ratio():
    a = {"exp_mismatch": 40, "mant_mean": 20.0, "mant_median": 10.0}
    assert qual.bands_agree(a, {"exp_mismatch": 58, "mant_mean": 30.0, "mant_median": 15.0})
    assert not qual.bands_agree(a, {"exp_mismatch": 80, "mant_mean": 20.0, "mant_median": 10.0})
    # Near zero, the slack: 0 and 2 agree.
    zero = {"exp_mismatch": 0, "mant_mean": 0.0, "mant_median": 0.0}
    assert qual.bands_agree(zero, {"exp_mismatch": 2, "mant_mean": 1.0, "mant_median": 1.0})


def test_a_request_bounds_its_decode():
    with pytest.raises(ValueError):
        qual.new_request(qualification_id="q", model="m", revision="r", set_id="s",
                         problems=4, sampling={}, max_new_tokens=64, thinking=False,
                         completions=65)
    with pytest.raises(ValueError):
        qual.new_request(qualification_id="q", model="m", revision="r", set_id="s",
                         problems=4, sampling={}, max_new_tokens=131072, thinking=False,
                         completions=32)
    assert qual.qualify_lease_seconds(16384) == pytest.approx(1800 + 4096)


def test_spread():
    assert spread(32, 32) == [1] * 32
    assert spread(10, 4) == [3, 3, 2, 2]
    assert spread(2, 4) == [1, 1, 0, 0]


class _Tokenizer:
    chat_template = "x"

    def apply_chat_template(self, messages, **kwargs):
        return f"<u>{messages[0]['content']}</u>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _Generator:
    def __init__(self):
        self.calls = []

    def generate_many(self, prompts, ns):
        self.calls.append((len(prompts), list(ns)))
        return [[SimpleNamespace(tokens=[7, 8, 9], proofs=["p"]) for _ in range(n)] for n in ns]


def _lease(**kw):
    return {"protocol": "reliquary.corpus-audit/v1", "type": "qualify", "lease_id": "a" * 32,
            "qualification_id": "order-q1", "model_id": "m", "model_revision": "r" * 40,
            "chunk_tokens": 32, "topk": 128, "expires_at": 1e12,
            "prompts": [{"problem_id": "p0", "text": "one"}, {"problem_id": "p1", "text": "two"}],
            "completions": 3, "sampling": {"temperature": 1.0}, "max_new_tokens": 16,
            "thinking": False, **kw}


INFO = {"gpu_count": 1, "gpu": "H100", "vllm_version": "0.1", "checkpoint_sha256": "d" * 64}


def test_the_executor_renders_like_the_job_decodes_in_one_batch_and_reports_every_chunk():
    seen = []

    def score(items):
        seen.extend(items)
        return [("ok", (ChunkResult(3, 1.5, 1.0),)) if k else ("proof_undecodable", ())
                for k in range(len(items))]

    generator = _Generator()
    ticks = iter(range(100))
    body = qualify_lease(_lease(), tokenizer=_Tokenizer(), generator=generator, score=score,
                         model_info=INFO, clock=lambda: float(next(ticks)))
    assert generator.calls == [(2, [2, 1])]  # one batch
    prompt = [ord(c) for c in "<u>one</u>"]
    assert seen[0] == (prompt + [7, 8, 9], len(prompt), ["p"])
    assert body["completions"] == 3 and body["failed_completions"] == 1
    assert body["chunks"] == [[3, 1.5, 1.0], [3, 1.5, 1.0]]
    assert body["completion_tokens"] == 9 and body["decode_seconds"] == 1.0
    QualifyResult.model_validate(body)


def test_the_miners_generator_decodes_many_prompts_in_one_call(monkeypatch):
    from reliquary.miner.corpus_miner import Generation, VllmGenerator
    from tests.unit.test_corpus_miner import EOS, _install_fake_vllm

    import sys
    from types import ModuleType

    _install_fake_vllm(monkeypatch)
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2,
                               max_new_tokens=8)
    generator = VllmGenerator("/fake", sampling, SimpleNamespace(chunk_tokens=32, topk=8), EOS)
    batches = []

    def generate(prompts, params):
        batches.append([p["prompt_token_ids"] for p in prompts])
        return list(range(len(prompts)))

    generator._llm.generate = generate
    generator._generations = lambda outputs, lengths: [Generation([o], [str(n)])
                                                        for o, n in zip(outputs, lengths)]
    grouped = generator.generate_many([[1, 2], [3]], [2, 1])
    assert batches == [[[1, 2], [1, 2], [3]]]
    assert [[g.proofs for g in group] for group in grouped] == [[["2"], ["2"]], [["1"]]]


@pytest.fixture
def store(monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    return qual.QualificationStore()


def _queue(store, now, facts=None):
    prompts = b"".join(json.dumps({"problem_id": f"s-{k:06d}", "messages": [
        {"role": "user", "content": f"q{k}"}]}).encode() + b"\n" for k in range(10))

    async def read_prompts(set_id):
        return prompts if set_id == "s" else None

    async def model_facts(repo, revision):
        return facts or {"architecture": "Qwen3ForCausalLM", "eos_token_id": 151645}

    return qual.QualificationQueue(store=store, read_prompts=read_prompts,
                                   model_facts=model_facts, clock=lambda: now[0])


def _request(store, model="m"):
    asyncio.run(store.write(qual.new_request(
        qualification_id="order-q1", model=model, revision="r" * 40, set_id="s", problems=4,
        completions=8, sampling={"temperature": 0.6}, max_new_tokens=64, thinking=False,
        clock=lambda: 0.0), None))


def _executor(eid, provider=None, host=None, model="m"):
    return {"executor_id": eid, "model_id": model, "model_revision": "r" * 40,
            "provider_id": provider or f"prov-{eid}", "host": host or f"host-{eid}"}


def _result(chunks=None, **kw):
    return QualifyResult.model_validate({"type": "qualify",
                                         "chunks": chunks or [[10, 5.0, 4.0]] * 50,
                                         "completions": 8, "failed_completions": 0,
                                         "completion_tokens": 3600, "decode_seconds": 2.0,
                                         **INFO, **kw})


def _measure(queue, eid, **kw):
    lease = asyncio.run(queue.claim(_executor(eid, **{k: v for k, v in kw.items()
                                                       if k in ("provider", "host")})))
    assert lease is not None
    return lease


def test_two_qualifiers_on_distinct_providers_agree_and_the_control_reads_the_model_facts(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    assert asyncio.run(queue.claim(_executor("x", model="other"))) is None
    first = asyncio.run(queue.claim(_executor("e1")))
    assert [p["text"] for p in first["prompts"]] == ["q0", "q1", "q2", "q3"]
    assert first["expires_at"] == pytest.approx(qual.qualify_lease_seconds(64))
    # Same provider or same host: never the second qualifier.
    assert asyncio.run(queue.claim(_executor("e2", provider="prov-e1"))) is None
    assert asyncio.run(queue.claim(_executor("e3", host="host-e1"))) is None
    assert asyncio.run(queue.claim({**_executor("e4"), "provider_id": None})) is None
    second = asyncio.run(queue.claim(_executor("e2")))
    assert asyncio.run(queue.claim(_executor("e5"))) is None  # two seats taken
    with pytest.raises(LeaseRefused):
        asyncio.run(queue.result(_executor("e2"), first["lease_id"], _result()))
    after_one = asyncio.run(queue.result(_executor("e1"), first["lease_id"], _result()))
    assert after_one["status"] == qual.PENDING
    final = asyncio.run(queue.result(_executor("e2"), second["lease_id"],
                                     _result(chunks=[[12, 6.0, 4.0]] * 50)))
    assert final["status"] == qual.QUALIFIED
    result = final["result"]
    assert result["band"]["exp_mismatch"] == 12  # the larger agreed band
    assert result["thresholds"]["exp_mismatch_threshold"] == 60
    assert (result["architecture"], result["eos_token_id"]) == ("Qwen3ForCausalLM", 151645)
    assert result["tokens_per_gpu_hour"] == pytest.approx(3600 / 2 * 3600)
    assert set(result["measurements"]) == {"e1", "e2"}
    assert {m["provider_id"] for m in result["measurements"].values()} == {"prov-e1", "prov-e2"}


def test_disagreeing_qualifiers_are_retried_with_another_pair_then_fail(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())

    def pair(a, b, a_result, b_result):
        la = asyncio.run(queue.claim(_executor(a)))
        lb = asyncio.run(queue.claim(_executor(b)))
        asyncio.run(queue.result(_executor(a), la["lease_id"], a_result))
        return asyncio.run(queue.result(_executor(b), lb["lease_id"], b_result))

    wide = _result(chunks=[[100, 5.0, 4.0]] * 50)
    record = pair("e1", "e2", _result(), wide)
    assert record["status"] == qual.PENDING and record["disagreements"] == [["e1", "e2"]]
    # The same pair is never tried again; another may be.
    lease = asyncio.run(queue.claim(_executor("e1")))
    assert asyncio.run(queue.claim(_executor("e2"))) is None
    other = asyncio.run(queue.claim(_executor("e3")))
    asyncio.run(queue.result(_executor("e1"), lease["lease_id"], _result()))
    record = asyncio.run(queue.result(_executor("e3"), other["lease_id"],
                                      _result(checkpoint_sha256="e" * 64)))
    assert record["status"] == qual.PENDING and len(record["disagreements"]) == 2
    record = pair("e4", "e5", _result(), _result(failed_completions=1))
    assert record["status"] == qual.FAILED and "disagreed" in record["result"]["failed_reason"]


def test_both_qualifiers_failing_their_own_proofs_refuse_the_model(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    la = asyncio.run(queue.claim(_executor("e1")))
    lb = asyncio.run(queue.claim(_executor("e2")))
    asyncio.run(queue.result(_executor("e1"), la["lease_id"], _result(failed_completions=2)))
    final = asyncio.run(queue.result(_executor("e2"), lb["lease_id"], _result(failed_completions=1)))
    assert final["status"] == qual.REFUSED


def test_expired_qualify_leases_free_the_seat_then_fail_the_qualification(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    first = asyncio.run(queue.claim(_executor("e1")))
    now[0] = 1e6
    with pytest.raises(LeaseRefused):
        asyncio.run(queue.result(_executor("e1"), first["lease_id"], _result()))
    for eid in ("e2", "e3"):
        assert asyncio.run(queue.claim(_executor(eid))) is not None
        now[0] += 1e6
    assert asyncio.run(queue.claim(_executor("e4"))) is None
    record, _ = asyncio.run(store.read("order-q1"))
    assert record["status"] == qual.FAILED and "expired" in record["result"]["failed_reason"]
    # Terminal: never read again.
    asyncio.run(queue.refresh())
    assert "order-q1" in queue._done and "order-q1" not in queue._open


def test_the_qualify_executor_claims_runs_and_posts():
    posted = []

    def handle(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/claim"):
            assert body["kind"] == "qualify"
            return httpx.Response(200, json=_lease())
        posted.append((request.url.path, body))
        return httpx.Response(200, json={"status": "qualified"})

    http = httpx.Client(base_url="http://c", transport=httpx.MockTransport(handle))
    executor = QualifyExecutor(http=http, executor_id="e1", token="t", model_id="m",
                               model_revision="r" * 40, run=lambda lease: {"type": "qualify"})
    assert executor.run(sleep=lambda s: None) == {"status": "qualified"}
    assert posted[0][0] == f"/corpus/internal/eval-audit/{'a' * 32}/result"


def test_the_qualify_command_needs_its_token_and_a_pinned_model(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    monkeypatch.delenv("RELIQUARY_EXECUTOR_TOKEN", raising=False)
    result = CliRunner().invoke(app, ["corpus", "qualify", "--model", "o/m@abc",
                                      "--control", "http://c", "--executor-id", "e"])
    assert result.exit_code == 1 and "RELIQUARY_EXECUTOR_TOKEN" in result.output
    monkeypatch.setenv("RELIQUARY_EXECUTOR_TOKEN", "t")
    result = CliRunner().invoke(app, ["corpus", "qualify", "--model", "o/m",
                                      "--control", "http://c", "--executor-id", "e"])
    assert result.exit_code == 1 and "repo@revision" in result.output
