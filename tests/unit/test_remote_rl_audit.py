"""Out-of-zone audits on the remote proof worker (``remote_rl_audit``): the wire carries the group and the lease
draw params (never the uniforms), the worker syncs its verify model from the signed index the controller forwards and
judges with ``rl_audit``; a restarted worker is re-adopted once; the service picks remote, local or none."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from reliquary.services.exploration import AUDIT_FAILURE_STATISTICAL
from reliquary.validator.remote_proof_server import create_proof_app
from reliquary.validator.remote_rl_audit import RemoteAuditModel, RlAuditBackend, RlAuditRequest
from reliquary.validator.rl_async_service import AUDIT_DEVICE_ENV, audit_source
from tests.unit.test_rl_audit import OTHER_NONCE, TinyLM, _group, _transitions, gumbel  # noqa: F401 (autouse)

WORKER = "proof-test"


class _ProofSlots:
    devices = ("cuda:0",)


class _Sync:
    """``CheckpointSync`` stand-in: version v of the run is ``TinyLM(table)``'s weights."""

    def __init__(self, table):
        self.state = TinyLM(table).state_dict()
        self.calls = []

    def sync(self, answer, state, version):
        self.calls.append(version)
        return {k: v.clone() for k, v in self.state.items()}, int(answer["index"]["latest"]["version"]), None


class _Board:
    def __init__(self, version):
        self.version = version

    async def answer(self):
        return {"index": {"latest": {"version": self.version}}, "signature": "00", "validator_hotkey": "5V",
                "urls": {"rl/run/anchors/00000000/a.safetensors": "https://r2.test/a"}, "urls_expire_at": 0}


class _Submissions:
    def __init__(self, groups):
        self.groups, self.results = groups, []

    def run_audits(self, forward, limit):
        self.results += [forward(g) for g in self.groups[:limit]]
        return len(self.results)


class _NoTimeout(TestClient):
    def post(self, url, timeout=None, **kwargs):          # the TestClient warns on per-request timeouts
        return super().post(url, **kwargs)


def _remote(table):
    import torch

    torch.manual_seed(1)
    backend = RlAuditBackend(TinyLM(torch.randn_like(table)), _Sync(table))   # starts at other weights
    app = create_proof_app(backend=_ProofSlots(), worker_id=WORKER, profile_id="p", generation_contract_sha256="b" * 64,
                           training_run_id="r", repo_id="t/c", software_revision="c" * 40, proof_path_hash="d" * 64,
                           rl_audit=backend)
    client = _NoTimeout(app)
    return backend, RemoteAuditModel(base_url="https://unused", worker_id=WORKER, client=client), client


def test_the_wire_carries_the_lease_draw_params_not_the_uniforms():
    group = _group(_transitions())
    request = RlAuditRequest.of(group, worker_id=WORKER, version=4)
    body = RlAuditRequest.read(__import__("json").dumps(request.model_dump()).encode())
    assert not any("uniform" in name for name in body.model_dump())
    rebuilt = body.group()
    assert (rebuilt.nonce, rebuilt.prompt_idx, rebuilt.checkpoint, rebuilt.completions, rebuilt.force_spans) == (
        group.nonce, group.prompt_idx, group.checkpoint, group.completions, group.force_spans)
    assert rebuilt.draw_uniforms(3) == group.draw_uniforms(3)
    for bad in ({"force_spans": [[0, 999]] + [None] * 15}, {"force_spans": [None]}, {"nonce": "XY"},
                {"completions": [[-1]] * 16}, {"uniforms": [0.5]}):
        with pytest.raises(ValueError):
            RlAuditRequest.model_validate({**request.model_dump(), **bad})


def test_audits_run_on_the_worker_at_the_latest_checkpoint():
    table = _transitions()
    backend, remote, client = _remote(table)
    honest, foreign = _group(table), _group(table, nonce_for_draw=OTHER_NONCE)
    empty = type(honest)(**{**honest.__dict__, "completions": tuple(c[:1] for c in honest.completions)})
    submissions = _Submissions([honest, foreign, empty])
    with client:
        assert remote.audit(4, submissions, 8) is None                  # not synced yet: nothing runs
        assert asyncio.run(remote.follow_once(_Board(4))) == 4
        assert asyncio.run(remote.follow_once(_Board(4))) is None       # already there: no second sync
        assert backend.sync.calls == [None] and backend.version == 4
        assert remote.audit(5, submissions, 8) is None                  # lagging the latest: nothing runs
        assert remote.audit(4, submissions, 8) == 3
    passed, failed, abstained = submissions.results
    assert (passed.passed, passed.rollout_match) == (True, (1.0,) * 16)
    assert failed.passed is False and failed.failure_class == AUDIT_FAILURE_STATISTICAL
    assert abstained.passed is None and abstained.suspect                # 16 drawn positions: the wire keeps it


def test_a_restarted_worker_is_readopted_once_then_audits():
    table = _transitions()
    backend, remote, client = _remote(table)
    with client:
        asyncio.run(remote.follow_once(_Board(4)))
        backend.version = None                                           # the worker restarted on its base weights
        submissions = _Submissions([_group(table)])
        assert remote.audit(4, submissions, 1) == 1
    assert submissions.results[0].passed is True and backend.sync.calls == [None, None]


def test_the_worker_refuses_another_identity_and_plain_http_objects():
    table = _transitions()
    backend, remote, client = _remote(table)
    remote.worker_id = "someone-else"
    with client:
        with pytest.raises(Exception):
            asyncio.run(remote.follow_once(_Board(4)))
        board = _Board(4)
        plain = asyncio.run(board.answer())
        plain["urls"] = {"k": "http://r2.test/a"}
        with pytest.raises(ValueError, match="https"):
            backend.adopt(plain)
    assert backend.version is None and backend.sync.calls == []


def test_the_service_runs_audits_remote_local_or_not_at_all():
    assert audit_source({"RELIQUARY_PROOF_EXECUTOR_MODE": "remote"}) == "remote"
    assert audit_source({AUDIT_DEVICE_ENV: "cuda:0"}) == "local"
    assert audit_source({"RELIQUARY_PROOF_EXECUTOR_MODE": "shadow"}) is None
    assert audit_source({}) is None
    with pytest.raises(ValueError):
        audit_source({"RELIQUARY_PROOF_EXECUTOR_MODE": "remote", AUDIT_DEVICE_ENV: "cuda:0"})
