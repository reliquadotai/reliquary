"""Evaluation-owned signed sessions, reached only through the administrator corridor.

The platform authenticates its current leased worker before signing this request.
The issuer resolves task authority from the native frozen set, not caller budgets.
Its book, persistence prefix and explicitly assigned machines are evaluation-only.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA = r"^[0-9a-f]{64}$"
_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
EVALUATION_SESSION_PREFIX = "reliquary/sandbox/evaluation-sessions/"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationModel(_Strict):
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class EvaluationSelection(_Strict):
    env: str = Field(min_length=1, max_length=128)
    problems: int = Field(ge=1, le=1_000_000, strict=True)
    samples: int = Field(ge=1, le=64, strict=True)


class EvaluationSessionRequest(_Strict):
    request_id: str = Field(pattern=_SHA)
    eval_id: str = Field(pattern=_ID)
    task_id: str = Field(pattern=r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
    set_id: str = Field(min_length=1, max_length=128)
    problem_id: str = Field(min_length=1, max_length=256)
    sample_index: int = Field(ge=0, le=63, strict=True)
    lease_fence: str = Field(pattern=_SHA)
    expires_at: int = Field(ge=1, strict=True)
    worker_id: str = Field(pattern=_ID)
    authority_kind: Literal["leased_worker"]
    model: EvaluationModel
    model_access: Literal["public", "private"] = "public"
    selection: EvaluationSelection
    protocol: dict[str, Any]
    protocol_json: str = Field(max_length=512 * 1024)
    protocol_sha256: str = Field(pattern=_SHA)
    protocol_bindings: dict[str, str]
    set_protocol: dict[str, Any]

    @field_validator("set_id")
    @classmethod
    def _set_id(cls, value):
        from reliquary.eval.sets import validated_set_id

        return validated_set_id(value)



def _engagement(binding):
    from reliquary.eval.protocol import digest

    sample = {key: binding[key] for key in
              ("eval_id", "task_id", "set_id", "problem_id", "sample_index")}
    return f"evaluation:{digest(sample)}:{digest(binding)}"


class EvaluationSessions:
    kind = "evaluation"

    def __init__(self, *, issuer, fleet, subnet, source_factory=None,
                 resolve_task=None, clock=time.time) -> None:
        self.issuer, self.fleet, self.subnet = issuer, fleet, subnet
        self._source_factory = source_factory or _open_source
        self._resolve_task = resolve_task
        self._clock = clock
        self._pending: dict[str, tuple[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._background: list[asyncio.Task] = []

    async def terms(self, worker: str, engagement: Mapping[str, Any]):
        from reliquary.sandbox.sessions import Refusal

        saved = self._pending.get(engagement.get("binding_sha256"))
        if set(engagement) != {"kind", "binding_sha256"} or saved is None or saved[0] != worker:
            return Refusal("engagement_kind_unsupported", {})
        return saved[1]

    async def start(self) -> None:
        await self.fleet.refresh_if_due()
        await self.issuer.restore()
        self._background = [asyncio.create_task(self.fleet.run()),
                            asyncio.create_task(self.issuer.maintain_forever(30))]

    async def stop(self) -> None:
        for task in self._background:
            task.cancel()
        for task in self._background:
            with suppress(asyncio.CancelledError):
                await task
        await self.issuer.drain(10)

    async def verify_result(self, binding, transcript, *, card, row, final_state,
                            received=None) -> float:
        """Verify an issued evaluation's receipt; this does not prove model generation."""
        from reliquary.eval.signed_agentic import checkpoint_identity
        from reliquary.sandbox.sessions import CLOSED_GRADED, LAPSED, LIVE, SUBMITTED
        from reliquary_sandbox.attest import Expected, GRADING_GRACE_S, token_sha256, verify_transcript

        received = self._clock() if received is None else received
        if not self.fleet.directory_ready(received):
            raise ValueError("evaluation_directory_unavailable")
        expected_row = {"set_id": card["set_id"], "problem_id": row["problem_id"],
                        "source_index": row["source_index"], "environment": card["source"],
                        "environment_manifest_sha256": card["environment_manifest_sha256"]}
        if any(binding.get(k) != v for k, v in expected_row.items()):
            raise ValueError("evaluation_result_identity_changed")
        checkpoint = checkpoint_identity(binding["model"])
        if binding.get("identity_kind") != "hf-snapshot/v1" or binding.get("checkpoint") != checkpoint:
            raise ValueError("evaluation_checkpoint_identity_changed")
        from reliquary.eval.protocol import CLAUDE_CLI_HARNESS, signed_cli_capability
        if card["protocol"]["harness"]["id"] == CLAUDE_CLI_HARNESS:
            from reliquary_sandbox_verifiers.cli import require_context_date

            cap, _, cap_sha = signed_cli_capability()
            episode = card["signed_episode"]
            require_context_date(binding.get("context_date"))
            if (binding.get("harness") != CLAUDE_CLI_HARNESS
                    or any(binding.get(key) != row.get(key) for key in ("prompt_sha256", "reference_sha256"))
                    or binding.get("cli_capability_sha256") != cap_sha
                    or binding.get("cli_runtime_image") != cap.runtime_image
                    or binding.get("evaluation_policy_sha256") != episode["evaluation_policy_sha256"]
                    or type(binding.get("max_turns")) is not int or not 1 <= binding["max_turns"] <= 48
                    or type(binding.get("max_episode_tokens")) is not int
                    or not 1 <= binding["max_episode_tokens"] <= 65536):
                raise ValueError("evaluation_cli_identity_changed")
        engagement = _engagement(binding)
        result = await asyncio.to_thread(
            verify_transcript, transcript, self.fleet.directory(), self.issuer._token_verifier,
            Expected(hotkey=binding["worker_id"], engagement=engagement, env=card["source"],
                     split=card["split"], index=row["source_index"], checkpoint=checkpoint,
                     require_graded=True))
        if not result.ok:
            raise ValueError("evaluation_transcript_invalid")
        claims, opened, final = result.claims, result.open, result.final
        record = self.issuer.book.get(claims.session_id)
        if (record is None or record.kind != "evaluation" or record.engagement != engagement
                or record.token_sha256 != token_sha256(transcript["token"])
                or record.state not in (LIVE, CLOSED_GRADED, LAPSED, SUBMITTED)
                or claims.image != binding["image"] or dataclasses.asdict(claims.budgets) != binding["budgets"]
                or opened.env_package != card["signed_episode"]["env_package"]
                or opened.tools_version != binding["tools_version"]
                or tuple(opened.tools) != tuple(binding["tools"])
                or not isinstance(final_state, bytes)
                or hashlib.sha256(final_state).hexdigest() != final.state_sha256):
            raise ValueError("evaluation_receipt_changed")
        # Whole-order grading may follow other episodes by hours. Expired evaluation
        # receipts remain verifiable; report acceptance owns deduplication, not miner payment.
        if record.state == LAPSED or received > claims.expires_at + GRADING_GRACE_S:
            return float(final.reward)
        if record.state != SUBMITTED:
            refusal = await self.issuer.claim(claims.session_id, hotkey=binding["worker_id"], received=received)
            if refusal is not None:
                raise ValueError("evaluation_session_not_submittable")
            try:
                if await self.issuer.persist_submitted([claims.session_id]) != 1:
                    raise RuntimeError("evaluation_session_not_persisted")
                await self.issuer.submitted(claims.session_id)
            except BaseException:
                await self.issuer.release_claim(claims.session_id)
                raise
        return float(final.reward)

    async def open(self, request: EvaluationSessionRequest) -> dict:
        from reliquary.eval.grading import load_sets
        from reliquary.eval.signed_agentic import checkpoint_identity
        from reliquary.eval.protocol import (BOUNDED_SIGNED_HARNESSES, CLAUDE_CLI_HARNESS,
            SIGNED_HARNESSES, _envelope, digest, signed_cli_capability, validate_execution, validate_request)
        from reliquary.sandbox.sessions import EngagementTerms, Refusal, token_validity_s

        now = int(self._clock())
        if request.expires_at <= now or request.expires_at > now + 86400:
            raise ValueError("evaluation_lease_expired")
        wanted_id = hashlib.sha256(json.dumps(
            [request.task_id, request.lease_fence, request.set_id,
             request.problem_id, request.sample_index], separators=(",", ":"),
            ensure_ascii=False).encode()).hexdigest()
        if request.request_id != wanted_id:
            raise ValueError("evaluation_request_identity_changed")
        protocol = _envelope(request.model_dump())
        validate_request({**request.model_dump(), **{key: protocol.get(key) for key in
                         ("sampling", "thinking", "max_new_tokens")}})
        if protocol.get("interaction") != "signed_episode" or protocol.get("harness") not in SIGNED_HARNESSES:
            raise ValueError("evaluation_harness_unsupported")
        if request.protocol_bindings.get(request.set_id) != request.set_protocol.get("protocol_sha256"):
            raise ValueError("evaluation_set_protocol_changed")
        selected = request.selection
        matches = [row for row in protocol.get("sets", []) if row.get("set_id") == request.set_id]
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in
                                   (("env", selected.env), ("problems", selected.problems),
                                    ("samples", selected.samples))):
            raise ValueError("evaluation_selection_changed")
        if request.sample_index >= selected.samples:
            raise ValueError("evaluation_sample_not_selected")
        loaded = await load_sets([request.set_id], {request.set_id: selected.problems},
                                 {request.set_id: selected.samples}, subnet=self.subnet)
        card, rows = loaded[request.set_id]
        if (card.get("source_kind") != "signed_episode" or card.get("env") != selected.env
                or card.get("protocol_sha256") != request.protocol_bindings[request.set_id]
                or any(card.get(k) != request.set_protocol.get(k)
                       for k in ("protocol", "protocol_json", "protocol_sha256"))):
            raise ValueError("evaluation_frozen_set_changed")
        validate_execution({request.set_id: {key: card[key] for key in
                           ("protocol", "protocol_json", "protocol_sha256")}},
                           {request.set_id: request.protocol_bindings[request.set_id]}, request=protocol)
        rows = [row for row in rows if row.get("problem_id") == request.problem_id]
        if len(rows) != 1:
            raise ValueError("evaluation_problem_not_selected")
        row = rows[0]
        source = await asyncio.to_thread(self._source_factory, card)
        resolve = self._resolve_task
        if resolve is None:
            from reliquary.eval.signed_agentic import resolve_task as resolve
        resolved = await resolve(source, row["source_index"])
        if any(resolved.get(key) != row.get(key) for key in
               ("source_index", "prompt_sha256", "reference_sha256")):
            raise ValueError("evaluation_task_changed")
        episode, task = card["signed_episode"], resolved["task"]
        from reliquary.environment.agentic_swe import SUPPORTED_RENDERER, harness_tools
        from reliquary.sandbox.fleet import TOOLS_VERSION

        public_episode = card["protocol"]["source"]["episode"]
        cli_cap = None
        if protocol["harness"] == CLAUDE_CLI_HARNESS:
            cli_cap, _, cli_sha = signed_cli_capability()
        model_tools_sha = (cli_cap.model_tools_sha256 if cli_cap is not None
                           else digest(list(harness_tools(episode["tools"]))))
        if (episode["renderer_id"] != SUPPORTED_RENDERER
                or public_episode["renderer_id"] != episode["renderer_id"]
                or model_tools_sha != public_episode["tools_sha256"]
                or episode["tools_version"] != TOOLS_VERSION):
            raise ValueError("evaluation_tools_or_renderer_changed")
        names = ("max_calls", "per_call_timeout_s", "wall_s", "memory_bytes", "pids", "disk_bytes")
        budget = {name: task["limits"].get(name) for name in names}
        wall = protocol["episode_timeout_seconds"]
        if any(type(v) is not int or v <= 0 for v in budget.values()) or budget["wall_s"] > wall:
            raise ValueError("evaluation_budget_unsupported")
        budget["per_call_timeout_s"] = min(budget["per_call_timeout_s"], budget["wall_s"])
        cpu = task.get("cpu_millis")
        if type(cpu) is not int or cpu <= 0:
            raise ValueError("evaluation_task_cpu_unbounded")
        budget["cpu_s"] = budget["wall_s"] * max(1, math.ceil(cpu / 1000))
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", task["image"]):
            raise ValueError("evaluation_image_unpinned")
        if cli_cap is not None and task["image"].rsplit("sha256:", 1)[1] == cli_cap.runtime_image.rsplit("sha256:", 1)[1]:
            raise ValueError("evaluation_task_and_cli_images_must_differ")
        prepare = task.get("prepare_timeout_s")
        span = token_validity_s(self.issuer.policy.open_window_s, budget["wall_s"], prepare)
        if now + span > request.expires_at:
            raise ValueError("evaluation_lease_too_short")
        identity = checkpoint_identity(request.model.model_dump())
        binding = {key: getattr(request, key) for key in
                   ("eval_id", "task_id", "set_id", "problem_id", "sample_index", "lease_fence",
                    "worker_id", "authority_kind", "protocol_sha256")}
        binding.update(model=request.model.model_dump(), identity_kind="hf-snapshot/v1", checkpoint=identity,
                       environment=card["source"], environment_manifest_sha256=card["environment_manifest_sha256"],
                       source_index=row["source_index"], image=task["image"], tools=episode["tools"],
                       tools_version=episode["tools_version"], harness=protocol["harness"],
                       renderer_id=episode["renderer_id"], harness_sha256=card["protocol"]["harness"]["sha256"],
                       episode=card["protocol"]["source"]["episode"], budgets=budget)
        if protocol["harness"] in BOUNDED_SIGNED_HARNESSES:
            binding.update(max_turns=protocol["max_turns"], max_episode_tokens=protocol["max_episode_tokens"],
                           evaluation_policy_sha256=episode["evaluation_policy_sha256"])
        if cli_cap is not None:
            from reliquary_sandbox_verifiers.cli import utc_context_date

            binding.update(cli_capability_sha256=cli_sha, cli_runtime_image=cli_cap.runtime_image,
                           context_date=utc_context_date(), prompt_sha256=row["prompt_sha256"],
                           reference_sha256=row["reference_sha256"])
        binding_sha = digest(binding)
        engagement = _engagement(binding)
        sample_prefix = engagement.rsplit(":", 1)[0] + ":"

        def current():
            if int(self._clock()) + span > request.expires_at:
                return Refusal("evaluation_lease_expired", {})
            # One quoted attempt per sample across lease churn, also after expiry/restart.
            if any(record.kind == "evaluation" and record.engagement.startswith(sample_prefix)
                   for record in self.issuer.book._sessions.values()):
                return Refusal("engagement_taken", {})
            return None

        terms = EngagementTerms(
            engagement=engagement, env=card["source"], split=card["split"],
            index=row["source_index"], checkpoint=identity, image=task["image"],
            env_package=episode["env_package"], budgets=budget, exclusive=True,
            still_valid=current, prepare_timeout_s=prepare,
            env_options_sha256=task.get("env_options_sha256"), env_lock_sha256=task.get("env_lock_sha256"),
            reserve={"cpu_millis": task.get("cpu_millis"), "memory_bytes": budget["memory_bytes"]})
        async with self._lock:
            self._pending[binding_sha] = (request.worker_id, terms)
            try:
                grant = await self.issuer.open(hotkey=request.worker_id, request_id=request.request_id,
                                              engagement={"kind": "evaluation", "binding_sha256": binding_sha})
            finally:
                del self._pending[binding_sha]
        if isinstance(grant, Refusal):
            return {"refusal": {"reason": grant.reason, "retry_after": grant.retry_after}}
        return {"grant": {"session_id": grant.session_id, "gateway_url": grant.gateway_url,
                           "token": grant.token, "expires_at": grant.expires_at}, "binding": binding}


def _open_source(card):
    from reliquary.environment.bridged_episode import BridgedEnvConfig, BridgedEpisodeSource

    episode = card["signed_episode"]
    source = BridgedEpisodeSource(BridgedEnvConfig(
        environment=card["source"], package=episode["package"], rl_split=card["split"],
        options=episode["options"], image_lock_sha256=episode["image_lock_sha256"]))
    if (source.manifest_sha256() != card["environment_manifest_sha256"]
            or source.identity != episode["env_package"]
            or source.options_sha256 != episode["env_options_sha256"]):
        raise ValueError("evaluation_environment_changed")
    return source


def from_environment(*, subnet, environ=os.environ, clock=time.time):
    """Opt-in only: existing signer settings plus explicitly assigned sandbox IDs."""
    allowed = {item.strip() for item in environ.get("RELIQUARY_EVAL_SANDBOX_MACHINE_IDS", "").split(",") if item.strip()}
    if not allowed:
        return None
    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionIssuer
    from reliquary.validator.sandbox_wiring import SandboxValidatorConfig
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    config = SandboxValidatorConfig.from_env(environ)
    if config is None:
        raise ValueError("evaluation sandboxes require the existing validator signing key")
    policy = dataclasses.replace(config.policy or SandboxPolicy(), open_window_s=120)
    signer = Signer(config.key_id, load_private_key(config.key_file))
    verifier = Ed25519TokenVerifier({**config.retired_keys, config.key_id: signer.public_key_b64})

    async def documents():
        return await sandbox_store.list_machines()

    fleet = Fleet(read_documents=documents, fetch_report=http_fetch_report, clock=clock,
                  directory_refresh_s=config.directory_refresh_s,
                  directory_max_age_s=config.directory_max_age_s,
                  directory_read_timeout_s=config.directory_read_timeout_s,
                  machine_ids=allowed, additional_machine_scope=config.machine_ids)
    fleet.resources_required = True
    service = EvaluationSessions(issuer=None, fleet=fleet, subnet=subnet, clock=clock)
    service.issuer = SessionIssuer(book=SessionBook(policy), store=sandbox_store.R2SessionStore(
        prefix=EVALUATION_SESSION_PREFIX), fleet=fleet, signer=signer, token_verifier=verifier,
        engagements={"evaluation": service}, policy=policy, clock=clock)
    fleet.on_drained = service.issuer.void_machine
    return service
