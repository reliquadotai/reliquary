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


def test_thresholds_are_the_band_p99_with_margin_never_under_the_floor():
    # 100 chunks: p99 is the 99th smallest.
    chunks = [(k, k / 2, k / 4) for k in range(1, 101)]
    verdict = qual.thresholds_from_band(chunks)
    assert verdict["band"]["exp_mismatch"] == 99
    assert verdict["thresholds"] == {"exp_mismatch_threshold": 149,
                                     "mant_mean_threshold": 74.25,
                                     "mant_median_threshold": 40.0}  # 37.125 < floor 40
    assert verdict["refused"] is None
    tight = qual.thresholds_from_band([(1, 1.0, 1.0)] * 10)
    assert tight["thresholds"] == {"exp_mismatch_threshold": 60, "mant_mean_threshold": 40.0,
                                   "mant_median_threshold": 40.0}


def test_a_band_over_the_ceiling_is_refused(monkeypatch):
    monkeypatch.setenv("RELIQUARY_QUALIFY_CEILING_EXP", "100")
    verdict = qual.thresholds_from_band([(150, 1.0, 1.0)] * 10)
    assert verdict["refused"] and "exp_mismatch" in verdict["refused"]
    with pytest.raises(ValueError):
        qual.thresholds_from_band([])


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
    def generate(self, prompt_ids, n):
        return [SimpleNamespace(tokens=[7, 8, 9], proofs=["p"]) for _ in range(n)]


def _lease(**kw):
    return {"protocol": "reliquary.corpus-audit/v1", "type": "qualify", "lease_id": "a" * 32,
            "qualification_id": "order-q1", "model_id": "m", "model_revision": "r" * 40,
            "chunk_tokens": 32, "topk": 128, "expires_at": 1e12,
            "prompts": [{"problem_id": "p0", "text": "one"}, {"problem_id": "p1", "text": "two"}],
            "completions": 3, "sampling": {"temperature": 1.0}, "max_new_tokens": 16,
            "thinking": False, **kw}


INFO = {"gpu_count": 1, "gpu": "H100", "vllm_version": "0.1", "checkpoint_sha256": "d" * 64,
        "architecture": "Qwen3ForCausalLM", "eos_token_id": 2}


def test_the_executor_renders_like_the_job_decodes_and_reports_every_chunk():
    seen = []

    def score(items):
        seen.extend(items)
        return [("ok", (ChunkResult(3, 1.5, 1.0),)) if k else ("proof_undecodable", ())
                for k in range(len(items))]

    ticks = iter(range(100))
    body = qualify_lease(_lease(), tokenizer=_Tokenizer(), generator=_Generator(), score=score,
                         model_info=INFO, clock=lambda: float(next(ticks)))
    prompt = [ord(c) for c in "<u>one</u>"]
    assert seen[0] == (prompt + [7, 8, 9], len(prompt), ["p"])
    assert body["completions"] == 3 and body["failed_completions"] == 1
    assert body["chunks"] == [[3, 1.5, 1.0], [3, 1.5, 1.0]]
    assert body["completion_tokens"] == 9 and body["decode_seconds"] == 2.0
    QualifyResult.model_validate(body)


@pytest.fixture
def store(monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    return qual.QualificationStore()


def _queue(store, now):
    prompts = b"".join(json.dumps({"problem_id": f"s-{k:06d}", "messages": [
        {"role": "user", "content": f"q{k}"}]}).encode() + b"\n" for k in range(10))

    async def read_prompts(set_id):
        return prompts if set_id == "s" else None

    return qual.QualificationQueue(store=store, read_prompts=read_prompts,
                                   clock=lambda: now[0], lease_seconds=100)


def _request(store, model="m"):
    asyncio.run(store.write(qual.new_request(
        qualification_id="order-q1", model=model, revision="r" * 40, set_id="s", problems=4,
        completions=8, sampling={"temperature": 0.6}, max_new_tokens=64, thinking=False,
        clock=lambda: 0.0), None))


EXECUTOR = {"executor_id": "e1", "model_id": "m", "model_revision": "r" * 40,
            "provider_id": "lium-a", "host": "h1"}


def _result(**kw):
    return QualifyResult.model_validate({"type": "qualify", "chunks": [[10, 5.0, 4.0]] * 50,
                                         "completions": 8, "failed_completions": 0,
                                         "completion_tokens": 3600, "decode_seconds": 2.0,
                                         **INFO, **kw})


def test_the_queue_leases_to_the_models_executor_and_records_the_verdict(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    assert asyncio.run(queue.claim({**EXECUTOR, "model_id": "other"})) is None
    lease = asyncio.run(queue.claim(EXECUTOR))
    assert [p["text"] for p in lease["prompts"]] == ["q0", "q1", "q2", "q3"]
    assert lease["completions"] == 8 and lease["type"] == "qualify"
    assert asyncio.run(queue.claim(EXECUTOR)) is None  # leased
    with pytest.raises(LeaseRefused):
        asyncio.run(queue.result({**EXECUTOR, "executor_id": "e2"}, lease["lease_id"],
                                 _result()))
    final = asyncio.run(queue.result(EXECUTOR, lease["lease_id"], _result()))
    assert final["status"] == qual.QUALIFIED
    assert final["result"]["thresholds"]["exp_mismatch_threshold"] == 60
    assert final["result"]["tokens_per_gpu_hour"] == pytest.approx(3600 / 2 * 3600)
    assert final["result"]["provider_id"] == "lium-a"
    stored, _ = asyncio.run(store.read("order-q1"))
    assert stored["status"] == qual.QUALIFIED


def test_failed_proofs_or_a_wide_band_refuse_the_model(store, monkeypatch):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    lease = asyncio.run(queue.claim(EXECUTOR))
    final = asyncio.run(queue.result(EXECUTOR, lease["lease_id"],
                                     _result(failed_completions=1)))
    assert final["status"] == qual.REFUSED and "failed their own proofs" in \
        final["result"]["refused_reason"]


def test_an_expired_qualify_lease_goes_back_to_the_queue(store):
    now = [0.0]
    queue = _queue(store, now)
    _request(store)
    asyncio.run(queue.refresh())
    first = asyncio.run(queue.claim(EXECUTOR))
    now[0] = 200.0
    with pytest.raises(LeaseRefused):
        asyncio.run(queue.result(EXECUTOR, first["lease_id"], _result()))
    second = asyncio.run(queue.claim({**EXECUTOR, "executor_id": "e2"}))
    assert second is not None and second["lease_id"] != first["lease_id"]


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
