"""Remote audit v2: span items only go to executors that ask for v2."""

import asyncio

import pytest
from pydantic import ValidationError

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    AUDIT_PROTOCOL_V2,
    MAX_SEQUENCE_TOKENS,
    AuditItem,
    AuditLease,
    ClaimRequest,
)
from reliquary.validator.corpus_audit_remote import RemoteAuditDispatcher
from tests.unit.test_corpus_audit_remote import _Clock, _directory, _doc

V1_ITEM = {"tokens": list(range(10, 30)), "prompt_len": 5, "proofs": ["A" * 8]}
V2_ITEM = {"tokens": list(range(10, 60)), "prompt_len": 5, "proofs": ["A" * 8, "B" * 8],
           "spans": [[5, 20], [30, 50]]}
BOTH = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V2)


def _dispatcher(local=None, **kw):
    clock = _Clock()

    async def default_local(items):
        return [("ok", ()) for _ in items]

    d = RemoteAuditDispatcher(directory=_directory([_doc()], clock), proof=PROOF,
                              local_scores=local or default_local, clock=clock, **kw)
    return d, clock


async def test_a_v1_executor_is_never_given_a_v2_lease():
    d, _ = _dispatcher()
    task = asyncio.ensure_future(d.score([V2_ITEM]))
    await asyncio.sleep(0)
    assert d.claim("pod-1") is None
    lease = d.claim("pod-1", protocols=BOTH)
    assert lease["protocol"] == AUDIT_PROTOCOL_V2 and lease["items"][0]["spans"] == [[5, 20], [30, 50]]
    assert lease["min_chunk_tokens"] == 8
    AuditLease.model_validate(lease)
    task.cancel()


async def test_a_v1_lease_carries_no_v2_fields():
    d, _ = _dispatcher()
    task = asyncio.ensure_future(d.score([V1_ITEM]))
    await asyncio.sleep(0)
    lease = d.claim("pod-1")
    assert lease["protocol"] == AUDIT_PROTOCOL
    assert "min_chunk_tokens" not in lease and "spans" not in lease["items"][0]
    task.cancel()


async def test_mixed_batches_split_so_v1_rows_stay_claimable_by_v1():
    d, _ = _dispatcher()
    task = asyncio.ensure_future(d.score([V1_ITEM, V2_ITEM]))
    await asyncio.sleep(0)
    lease = d.claim("pod-1")
    assert lease["protocol"] == AUDIT_PROTOCOL and len(lease["items"]) == 1
    task.cancel()


async def test_v2_work_is_scored_locally_when_only_v1_executors_are_connected():
    seen = []

    async def local(items):
        seen.append(items)
        return [("ok", ()) for _ in items]

    d, _ = _dispatcher(local)
    d.heartbeat("pod-1")
    d._directory._by_id["pod-1"] = _doc()
    assert d.connected()
    task = asyncio.ensure_future(d.score([V2_ITEM]))
    await asyncio.sleep(0)
    assert d.claim("pod-1") is None            # a v1 claim: nothing for it
    await d.sweep()
    (scored,) = await asyncio.wait_for(task, 1)
    assert scored[2] is None and seen and seen[0][0]["spans"] == [[5, 20], [30, 50]]


async def test_v2_work_waits_for_a_connected_v2_executor():
    d, _ = _dispatcher()
    d._directory._by_id["pod-1"] = _doc()
    d.claim("pod-1", protocols=BOTH)           # a v2-capable executor makes contact
    task = asyncio.ensure_future(d.score([V2_ITEM]))
    await asyncio.sleep(0)
    await d.sweep()
    assert not task.done()                     # it waits for that executor
    lease = d.claim("pod-1", protocols=BOTH)
    assert lease["protocol"] == AUDIT_PROTOCOL_V2
    task.cancel()


async def test_a_rechecked_v2_batch_is_rescored_locally_with_its_spans():
    from reliquary.validator.corpus_audit_protocol import AuditResult

    seen = []

    from reliquary.protocol.toploc import ChunkResult

    async def local_cr(items):
        seen.append(items)
        return [("ok", (ChunkResult(0, 0.0, 0.0), ChunkResult(0, 0.0, 0.0))) for _ in items]

    d, _ = _dispatcher(local_cr, recheck_fraction=1.0)
    d._directory._by_id["pod-1"] = _doc()
    task = asyncio.ensure_future(d.score([V2_ITEM]))
    await asyncio.sleep(0)
    lease = d.claim("pod-1", protocols=BOTH)
    chunks = [[0, 0.0, 0.0], [0, 0.0, 0.0]]
    d.result("pod-1", lease["lease_id"], AuditResult(scores=[{"status": "ok", "chunks": chunks}]))
    (scored,) = await asyncio.wait_for(task, 1)
    assert scored[2] is None and seen[0][0]["spans"] == [[5, 20], [30, 50]]


def test_old_claims_default_to_v1_and_a_v2_lease_needs_spans():
    assert ClaimRequest(executor_id="pod-1", model_id="m", model_revision="r").protocols == [AUDIT_PROTOCOL]
    with pytest.raises(ValidationError):
        AuditLease(protocol=AUDIT_PROTOCOL_V2, lease_id="a" * 32, model_id="m", model_revision="r",
                   chunk_tokens=32, topk=128, expires_at=1.0, items=[V1_ITEM], min_chunk_tokens=8)


def test_an_item_fits_a_60k_trajectory_with_its_prompt():
    AuditItem(tokens=list(range(60_000)), prompt_len=8_000, proofs=[], spans=[(8_000, 60_000)])
    assert MAX_SEQUENCE_TOKENS == 65_536


@pytest.mark.parametrize("spans", [[(3, 20)], [(5, 5)], [(20, 30), (5, 10)], [(5, 30), (25, 40)],
                                   [(5, 21)]])
def test_malformed_spans_are_refused_at_parse_time(spans):
    with pytest.raises(ValidationError):
        AuditItem(tokens=list(range(20)), prompt_len=5, proofs=[], spans=spans)


async def test_each_v2_row_gets_its_own_lease():
    d, _ = _dispatcher()
    task = asyncio.ensure_future(d.score([V2_ITEM, V2_ITEM, V2_ITEM]))
    await asyncio.sleep(0)
    leases = [d.claim("pod-1", protocols=BOTH) for _ in range(2)]
    assert all(len(lease["items"]) == 1 for lease in leases)
    assert len(d._queue) == 1
    task.cancel()


async def test_a_v1_claim_still_gets_v1_work_queued_behind_v2_work():
    d, _ = _dispatcher()
    d._queue.clear()
    t2 = asyncio.ensure_future(d.score([V2_ITEM]))
    await asyncio.sleep(0)
    t1 = asyncio.ensure_future(d.score([V1_ITEM]))
    await asyncio.sleep(0)
    assert d._queue[0].protocol == AUDIT_PROTOCOL_V2
    lease = d.claim("pod-1")
    assert lease["protocol"] == AUDIT_PROTOCOL
    assert len(d._queue) == 1 and d._queue[0].protocol == AUDIT_PROTOCOL_V2
    t1.cancel(); t2.cancel()


async def _router_claim(body):
    import httpx
    from fastapi import FastAPI

    from reliquary.validator.corpus_audit_remote import build_audit_executor_router
    from tests.unit.test_corpus_audit_remote import GOOD, MODEL, REVISION

    clock = _Clock()
    directory = _directory([_doc()], clock)
    await directory.refresh()
    seen = []

    class Stub:
        def claim(self, executor_id, protocols=(AUDIT_PROTOCOL,)):
            seen.append(tuple(protocols))
            return None

    app = FastAPI()
    app.include_router(build_audit_executor_router(Stub(), directory))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://control") as client:
        await client.post("/corpus/internal/audit/claim", headers={"Authorization": f"Bearer {GOOD}"},
                          json={"executor_id": "pod-1", "model_id": MODEL,
                                "model_revision": REVISION, **body})
    return seen


async def test_the_router_passes_the_claimed_protocols_and_defaults_to_v1():
    assert await _router_claim({"protocols": list(BOTH)}) == [BOTH]
    assert await _router_claim({}) == [(AUDIT_PROTOCOL,)]


async def _claim_body(prefix):
    from reliquary.validator.corpus_audit_executor import AuditExecutor

    posts = []

    class Http:
        async def post(self, path, json=None, **kw):
            posts.append((path, json))
            return type("R", (), {"status_code": 204, "raise_for_status": lambda self: None,
                                  "json": lambda self: {}})()

    executor = AuditExecutor(http=Http(), executor_id="pod-1", token="x", model_id="m",
                             model_revision="r", load_model=lambda m, r: None, prefix=prefix)
    executor._last_heartbeat = executor._clock()
    await executor.step()
    return next(body for path, body in posts if path.endswith("/claim"))


async def test_the_corpus_claim_carries_protocols_and_the_eval_claim_does_not():
    from reliquary.validator.corpus_audit_executor import AUDIT_PREFIX, EVAL_AUDIT_PREFIX

    assert (await _claim_body(AUDIT_PREFIX))["protocols"] == list(BOTH)
    assert "protocols" not in await _claim_body(EVAL_AUDIT_PREFIX)
