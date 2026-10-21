"""Actual signed evaluation grants are isolated, source-bound and replay-safe."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.delivery import LocalDirectorySink  # noqa: E402
from reliquary.environment.agentic_swe import SUPPORTED_RENDERER, harness_tools  # noqa: E402
from reliquary.eval.protocol import SIGNED_SOURCE_POLICIES, canonical, digest, seal_set  # noqa: E402
from reliquary.eval.signed_agentic import resolve_task  # noqa: E402
from reliquary.eval.storage import subnet_key  # noqa: E402
from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.sandbox.evaluation_engagements import (  # noqa: E402
    EVALUATION_SESSION_PREFIX, EvaluationSessionRequest, EvaluationSessions, from_environment,
)
from reliquary.sandbox.sessions import SUBMITTED, SandboxPolicy, SessionBook, SessionIssuer  # noqa: E402
from reliquary.sandbox.tasks import ResolvedTask  # noqa: E402
from tests.unit.sandbox_fixtures import IMAGE, NOW, directory, signer, transcript  # noqa: E402
from tests.unit.test_admin_eval_sandbox import request_body  # noqa: E402
from tests.unit.test_sandbox_sessions import Clock, FakeFleet  # noqa: E402


@pytest.fixture(autouse=True)
def installed_package_identity(monkeypatch):
    # Unit fixtures supply optional distribution identities; signatures use real keys.
    monkeypatch.setattr("reliquary.eval.verifiers_source.installed_artifacts", lambda package: "9" * 64)
    monkeypatch.setattr("reliquary.eval.protocol.renderer_sha256", lambda: "2" * 64)


class Source:
    def __init__(self):
        self.task = ResolvedTask(IMAGE, dict(max_calls=40, per_call_timeout_s=60, wall_s=300,
            memory_bytes=4 * 1024**3, pids=1024, disk_bytes=10 * 1024**3),
            env_options_sha256="7" * 64, env_lock_sha256="8" * 64,
            prepare_timeout_s=30, cpu_millis=2000)

    def prompt(self, index):
        return "Solve held-out task " + str(index)

    async def resolve(self, index):
        return self.task


def build(tmp_path):
    source = Source()
    rows = [dict(asyncio.run(resolve_task(source, i)), problem_id=f"problem-{i + 1}") for i in range(2)]
    for row in rows:
        row.pop("task")
    grading = b"".join(canonical(row) + b"\n" for row in rows)
    policy_sha = SIGNED_SOURCE_POLICIES[("reliquary-terminal==0.3.1", "tmax")]
    card = seal_set({"set_id": "heldout-1", "count": 2, "source": "terminal", "source_kind": "signed_episode",
        "interaction": "signed_episode", "harness_max_turns": 40, "env": "terminal", "split": "tmax",
        "prompts_sha256": "0" * 64, "grading_sha256": hashlib.sha256(grading).hexdigest(),
        "environment_manifest_sha256": "1" * 64, "disjointness": {"disjoint": None},
        "signed_episode": {"package": "reliquary-terminal==0.3.1", "source_policy_sha256": policy_sha, "options": {}, "image_lock_sha256": "8" * 64,
            "env_package": "reliquary-terminal==0.3.1+frozen", "env_options_sha256": "7" * 64,
            "renderer_id": SUPPORTED_RENDERER, "tools_version": "reliquary-tools/1", "tools": ["bash", "edit"]},
        "episode": {"renderer_id": SUPPORTED_RENDERER, "renderer_sha256": "2" * 64,
            "tools_sha256": digest(list(harness_tools(["bash", "edit"]))), "runtime_image_sha256": "3" * 64}},
        reviewed={"data": {"environment.toml": policy_sha}})
    store = LocalDirectorySink(tmp_path / "sets")
    store._path(subnet_key(card["set_id"], "set.json")).write_bytes(canonical(card))
    store._path(subnet_key(card["set_id"], "grading.jsonl")).write_bytes(grading)
    validator, machine = signer(tmp_path, "eval", "eval-key"), signer(tmp_path, "machine", "machine-key")
    clock = Clock()
    fleet = FakeFleet(directory(machine))
    sessions = EvaluationSessions(issuer=None, fleet=fleet, subnet=store,
                                  source_factory=lambda card: source, clock=clock)
    policy = SandboxPolicy(open_window_s=120)
    persisted = MemorySessionStore()
    sessions.issuer = SessionIssuer(book=SessionBook(policy), store=persisted, fleet=fleet, signer=validator,
        token_verifier=attest.Ed25519TokenVerifier({"eval-key": validator.public_key_b64}),
        engagements={"evaluation": sessions}, policy=policy, clock=clock)
    body = request_body()
    protocol = {"schema": "reliquary/evaluation-request/v1", "sets": [{"set_id": "heldout-1", "env": "terminal",
        "problems": 1, "samples": 1}], "protocol_bindings": {"heldout-1": card["protocol_sha256"]},
        "sampling": {"temperature": 0.6, "top_p": 1, "top_k": 0, "seed": 7}, "thinking": False,
        "max_new_tokens": 64, "harness": "signed-bash-v1", "max_turns": 40,
        "max_episode_tokens": 4096, "episode_timeout_seconds": 300, "interaction": "signed_episode"}
    body.update(protocol=protocol, protocol_json=canonical(protocol).decode(), protocol_sha256=digest(protocol),
                protocol_bindings=protocol["protocol_bindings"], expires_at=NOW + 1800,
                set_protocol={key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")})
    return SimpleNamespace(sessions=sessions, body=body, clock=clock, fleet=fleet, persisted=persisted,
                           validator=validator, machine=machine, card=card, rows=rows, source=source)


def open_(env, changes=None):
    body = {**env.body, **(changes or {})}
    if "request_id" not in (changes or {}):
        body["request_id"] = hashlib.sha256(json.dumps([body["task_id"], body["lease_fence"],
            body["set_id"], body["problem_id"], body["sample_index"]], separators=(",", ":")).encode()).hexdigest()
    return asyncio.run(env.sessions.open(EvaluationSessionRequest.model_validate(body)))


def test_actual_grant_and_lost_ack_replay_use_only_evaluation_authority(tmp_path):
    env = build(tmp_path)
    first = open_(env)
    second = open_(env)
    assert first == second
    assert len(env.persisted.documents) == 1 and len(env.fleet.issued) == 1
    claims = attest.SessionClaims.from_dict(first["grant"]["token"]["claims"])
    assert claims.hotkey == "pod-1"
    assert claims.engagement.startswith("evaluation:")
    assert claims.engagement.endswith(":" + digest(first["binding"]))
    assert first["binding"]["authority_kind"] == "leased_worker"
    assert first["binding"]["identity_kind"] == "hf-snapshot/v1"
    assert len(claims.checkpoint) == 64 and claims.checkpoint != "a" * 40 + "0" * 24
    assert claims.expires_at <= env.body["expires_at"]
    assert {record["kind"] for record in env.persisted.documents.values()} == {"evaluation"}
    assert env.sessions.issuer._quota is None
    assert set(env.sessions.issuer._engagements) == {"evaluation"}
    changed = open_(env, {"model": {"repo": "owner/other", "revision": "a" * 40}})
    assert changed["refusal"]["reason"] == "request_conflict"
    assert len(env.persisted.documents) == 1
    assert env.fleet.picked[0]["env_options_sha256"] == "7" * 64
    assert env.fleet.picked[0]["env_lock_sha256"] == "8" * 64


def test_unselected_problem_sample_and_expired_lease_do_not_issue(tmp_path):
    env = build(tmp_path)
    for changes in ({"problem_id": "problem-2"}, {"sample_index": 1}, {"expires_at": NOW - 1},
                    {"set_id": "another-set"}, {"request_id": "0" * 64}):
        with pytest.raises(ValueError):
            open_(env, changes)
    assert not env.persisted.documents
    first = open_(env)
    env.clock.now = env.body["expires_at"] + 1
    with pytest.raises(ValueError):
        open_(env)
    assert len(env.persisted.documents) == 1
    assert first["grant"]["expires_at"] < env.clock.now


def test_frozen_source_and_harness_changes_refuse_before_reservation(tmp_path):
    env = build(tmp_path)
    env.source.task = dataclasses.replace(env.source.task, image="registry.example/changed@sha256:" + "4" * 64)
    with pytest.raises(ValueError, match="task_changed"):
        open_(env)
    assert not env.persisted.documents
    other = copy.deepcopy(env.body["set_protocol"])
    other["protocol"]["harness"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        open_(env, {"set_protocol": other})
    assert not env.persisted.documents


def signed_receipt(env, grant):
    from reliquary_sandbox.attest import sign_record, chain_hash
    from reliquary_sandbox.attest.records import parse_body

    claims = attest.SessionClaims.from_dict(grant["grant"]["token"]["claims"])
    receipt = transcript(env.validator, env.machine, claims, tools=("bash", "edit"), state=b"final\x00archive")
    opened = dataclasses.replace(parse_body(receipt["records"][0]["body"]), env_package="reliquary-terminal==0.3.1+frozen")
    final = dataclasses.replace(parse_body(receipt["records"][-1]["body"]), prev=chain_hash(opened))
    receipt["records"] = [sign_record(env.machine, claims.machine_id, opened),
                          sign_record(env.machine, claims.machine_id, final)]
    return receipt


def test_signed_receipt_checks_state_signatures_identity_and_durable_settlement(tmp_path):
    env = build(tmp_path)
    grant = open_(env)
    binding = grant["binding"]
    claims = attest.SessionClaims.from_dict(grant["grant"]["token"]["claims"])
    receipt = signed_receipt(env, grant)
    verify = env.sessions.verify_result
    for changes in ({"binding": {**binding, "sample_index": 1}}, {"final_state": b"wrong"}):
        with pytest.raises(ValueError):
            asyncio.run(verify(changes.get("binding", binding), receipt, card=env.card,
                               row=env.rows[0], final_state=changes.get("final_state", b"final\x00archive")))
    assert asyncio.run(verify(binding, receipt, card=env.card, row=env.rows[0], final_state=b"final\x00archive")) == 1
    record = env.sessions.issuer.book.get(claims.session_id)
    assert record.state == SUBMITTED
    assert env.persisted.documents[claims.session_id]["state"] == SUBMITTED
    assert asyncio.run(verify(binding, receipt, card=env.card, row=env.rows[0], final_state=b"final\x00archive")) == 1
    forged = copy.deepcopy(receipt)
    forged["records"][-1]["signature"] = "0" * 128
    with pytest.raises(ValueError):
        asyncio.run(verify(binding, forged, card=env.card, row=env.rows[0], final_state=b"final\x00archive"))


def test_constructed_service_filters_assigned_machines_and_uses_its_own_persistence(tmp_path, monkeypatch):
    from reliquary.infrastructure import sandbox_store

    key = tmp_path / "evaluation-key.pem"
    key.write_bytes(attest.generate_private_key_pem())
    key.chmod(0o600)

    async def machines():
        return [{"machine_id": "evaluation-machine"}, {"machine_id": "active-rl-machine"}]

    monkeypatch.setattr(sandbox_store, "list_machines", machines)
    service = from_environment(subnet=object(), environ={
        "RELIQUARY_EVAL_SANDBOX_MACHINE_IDS": "evaluation-machine",
        "RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE": str(key),
        "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID": "evaluation-key",
        "RELIQUARY_SANDBOX_CORPUS_QUOTA": "1"})
    assert asyncio.run(service.fleet._read_documents()) == [{"machine_id": "evaluation-machine"}]
    assert service.fleet.resources_required
    assert service.issuer._quota is None
    assert set(service.issuer._engagements) == {"evaluation"}
    assert service.issuer._store._prefix == EVALUATION_SESSION_PREFIX
    assert EVALUATION_SESSION_PREFIX not in (sandbox_store.SESSION_PREFIX, sandbox_store.RL_SESSION_PREFIX)


def test_receipt_is_not_accepted_when_durable_settlement_fails(tmp_path, monkeypatch):
    env = build(tmp_path)
    grant = open_(env)
    session_id = grant["grant"]["session_id"]

    async def unavailable(ids):
        return 0

    monkeypatch.setattr(env.sessions.issuer, "persist_submitted", unavailable)
    with pytest.raises(RuntimeError, match="not_persisted"):
        asyncio.run(env.sessions.verify_result(grant["binding"], signed_receipt(env, grant),
                    card=env.card, row=env.rows[0], final_state=b"final\x00archive"))
    assert env.sessions.issuer.book.get(session_id).state != SUBMITTED
    assert not env.sessions.issuer.book.is_claimed(session_id)
    assert env.persisted.documents[session_id]["state"] != SUBMITTED


def test_restored_issuance_never_grants_the_same_sample_again(tmp_path):
    env = build(tmp_path)
    open_(env)
    issuer = env.sessions.issuer
    restored = SessionIssuer(book=SessionBook(issuer.policy), store=env.persisted, fleet=env.fleet,
        signer=env.validator, token_verifier=issuer._token_verifier,
        engagements={"evaluation": env.sessions}, policy=issuer.policy, clock=env.clock)
    assert asyncio.run(restored.restore()) == 1
    env.sessions.issuer = restored
    assert open_(env)["refusal"]["reason"] == "request_reused"
    assert open_(env, {"lease_fence": "e" * 64})["refusal"]["reason"] == "engagement_taken"
    assert len(env.persisted.documents) == 1 and len(env.fleet.issued) == 1


def test_late_whole_order_grading_accepts_original_receipt_but_not_voided_session(tmp_path):
    from reliquary.sandbox.sessions import LAPSED, VOIDED
    from reliquary_sandbox.attest import GRADING_GRACE_S

    env = build(tmp_path)
    grant = open_(env)
    receipt = signed_receipt(env, grant)
    session_id = grant["grant"]["session_id"]
    env.clock.now = grant["grant"]["expires_at"] + GRADING_GRACE_S + 3600
    asyncio.run(env.sessions.issuer.maintain())
    assert env.sessions.issuer.book.get(session_id).state == LAPSED
    assert asyncio.run(env.sessions.verify_result(grant["binding"], receipt, card=env.card,
        row=env.rows[0], final_state=b"final\x00archive")) == 1
    assert env.sessions.issuer.book.get(session_id).state == LAPSED
    # A revoked placement cannot become accepted merely because its transcript is signed.
    env.sessions.issuer.book.get(session_id).state = VOIDED
    with pytest.raises(ValueError, match="receipt_changed"):
        asyncio.run(env.sessions.verify_result(grant["binding"], receipt, card=env.card,
            row=env.rows[0], final_state=b"final\x00archive"))


def test_one_sample_cannot_mint_another_grant_after_lease_or_worker_rotation(tmp_path):
    env = build(tmp_path)
    first = open_(env)
    for change in ({"lease_fence": "e" * 64},
                   {"lease_fence": "d" * 64, "worker_id": "replacement-pod"}):
        assert open_(env, change)["refusal"]["reason"] == "engagement_taken"
    assert open_(env) == first
    assert len(env.persisted.documents) == 1 and len(env.fleet.issued) == 1
    env.clock.now = first["grant"]["expires_at"] + 1
    assert open_(env, {"lease_fence": "c" * 64})["refusal"]["reason"] == "engagement_taken"
    assert len(env.persisted.documents) == 1
