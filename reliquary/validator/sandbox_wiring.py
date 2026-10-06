"""A corpus validator's signed-sandbox side (plan 3): the validator's token key, the
machine directory and fleet, the session book and issuer with the route, and, per
signed job, its intake, its engagement view and its grader.

Settings:
* `RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE`: an Ed25519 PKCS#8 PEM, mode 0600, owned by
  the validator's user (`python -m reliquary_sandbox.attest.signing generate PATH`);
  the printed public key goes into every machine's VALIDATOR_PUBLIC_KEYS. Only its
  path is ever logged, never its contents;
* `RELIQUARY_SANDBOX_VALIDATOR_KEY_ID`: its key id;
* `RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS`: JSON `{key_id: base64 public key}` of
  earlier keys whose tokens may still be in flight after a rotation;
* `RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S` (30), `RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S`
  (120), `RELIQUARY_SANDBOX_DIRECTORY_READ_TIMEOUT_S` (15): the machine directory's
  re-read period, the age past which the fleet fails closed, and each read's bound
  (`sandbox.fleet.Fleet`);
* `RELIQUARY_SANDBOX_CLOSE_CONCURRENCY` (4): closes verified at once;
* `RELIQUARY_SANDBOX_CLOSE_BODY_TIMEOUT_S` (30): the time a close's body has to arrive;
* `RELIQUARY_SANDBOX_*` policy settings (`sandbox.sessions.SandboxPolicy`), among them
  `RELIQUARY_SANDBOX_CLAIM_TTL_S` (300).

This module imports reliquary-sandbox: a validator imports it only when a signed job
is served or the key settings are present.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reliquary.corpus.job import sandbox_split
from reliquary.sandbox.fleet import (
    DIRECTORY_MAX_AGE_SECONDS, DIRECTORY_READ_TIMEOUT_SECONDS, DIRECTORY_REFRESH_SECONDS,
)
from reliquary.sandbox.routes import CLOSE_BODY_TIMEOUT_S
from reliquary.sandbox.sessions import (
    CorpusEngagements, JobNotReady, RlPrecommitEngagements, SandboxPolicy, SessionBook, SessionIssuer,
    SignedJobView,
)
from reliquary.sandbox.tasks import SweTaskResolver
from reliquary.validator.signed_grading import SignedEpisodeGrader
from reliquary.validator.signed_intake import build_signed_episode_intake

logger = logging.getLogger(__name__)

CLOSE_CONCURRENCY = 4
SESSION_MAINTAIN_SECONDS = 60.0
# The session restore at start: each read bounded, retried with backoff, then the
# start aborts.
RESTORE_ATTEMPTS = 3
RESTORE_TIMEOUT_S = 60.0
RESTORE_BACKOFF_S = 5.0
# At shutdown, the background state writes (a drained machine's voids) get this long.
STOP_TIMEOUT_S = 10.0


def _positive(environ: Mapping[str, str], name: str, default, kind=float):
    raw = environ.get(name)
    if raw is None:
        return default
    value = kind(raw)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive, finite number")
    return value


@dataclass(frozen=True)
class SandboxValidatorConfig:
    key_file: Path
    key_id: str
    retired_keys: dict[str, str]
    policy: SandboxPolicy | None = None
    directory_refresh_s: float = DIRECTORY_REFRESH_SECONDS
    directory_max_age_s: float = DIRECTORY_MAX_AGE_SECONDS
    directory_read_timeout_s: float = DIRECTORY_READ_TIMEOUT_SECONDS
    close_concurrency: int = CLOSE_CONCURRENCY
    close_body_timeout_s: float = CLOSE_BODY_TIMEOUT_S

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> SandboxValidatorConfig | None:
        key_file = environ.get("RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE")
        key_id = environ.get("RELIQUARY_SANDBOX_VALIDATOR_KEY_ID")
        if not key_file and not key_id:
            return None
        if not (key_file and key_file.strip() and key_id and key_id.strip()):
            raise ValueError("set both RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE and "
                             "RELIQUARY_SANDBOX_VALIDATOR_KEY_ID (neither blank)")
        retired = json.loads(environ.get("RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS") or "{}")
        if not isinstance(retired, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in retired.items()):
            raise ValueError("RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS must map key ids to keys")
        return cls(
            Path(key_file), key_id, retired, SandboxPolicy.from_env(environ),
            directory_refresh_s=_positive(environ, "RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S",
                                          DIRECTORY_REFRESH_SECONDS),
            directory_max_age_s=_positive(environ, "RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S",
                                          DIRECTORY_MAX_AGE_SECONDS),
            directory_read_timeout_s=_positive(
                environ, "RELIQUARY_SANDBOX_DIRECTORY_READ_TIMEOUT_S",
                DIRECTORY_READ_TIMEOUT_SECONDS),
            close_concurrency=_positive(environ, "RELIQUARY_SANDBOX_CLOSE_CONCURRENCY",
                                        CLOSE_CONCURRENCY, int),
            close_body_timeout_s=_positive(environ, "RELIQUARY_SANDBOX_CLOSE_BODY_TIMEOUT_S",
                                           CLOSE_BODY_TIMEOUT_S))


@dataclass
class SandboxServices:
    signer: Any = field(repr=False)
    token_verifier: Any = field(repr=False)
    fleet: Any
    book: SessionBook
    issuer: SessionIssuer
    router: Any
    jobs: dict[str, SignedJobView] = field(default_factory=dict)
    # The app's CorpusJobRoutes, read at call time (set by `wire_signed_job`).
    routes: Callable[[], Any] | None = None
    restore_attempts: int = RESTORE_ATTEMPTS
    restore_timeout_s: float = RESTORE_TIMEOUT_S
    restore_backoff_s: float = RESTORE_BACKOFF_S

    def view(self, job_id: Any) -> SignedJobView | None:
        """The engagement book's lookup: a served job's view; a retired (or removed)
        job's view is dropped and the job is no longer served."""
        view = self.jobs.get(job_id) if isinstance(job_id, str) else None
        if view is None:
            return None
        table = self.routes() if self.routes is not None else None
        if table is not None and job_id in table.retired:
            self.forget(job_id)
            return None
        return view

    def forget(self, job_id: str) -> None:
        """A job that failed to wire, or retired: no session is opened for it again."""
        if self.jobs.pop(str(job_id), None) is not None:
            logger.info("sandbox sessions no longer opened for job %s", job_id)

    async def start(self) -> None:
        """Before serving: the directory read once (a failure is the fleet's to retry,
        on its backoff; it fails closed meanwhile), then the sessions restored. The
        restore is bounded and retried; when it still fails it raises and the
        validator must not start: it would serve opens with empty reservations and caps."""
        await self.fleet.refresh_if_due()
        for attempt in range(1, self.restore_attempts + 1):
            try:
                restored = await asyncio.wait_for(self.issuer.restore(), self.restore_timeout_s)
                break
            except Exception as exc:
                if attempt == self.restore_attempts:
                    logger.error("sandbox sessions not restored after %d attempts (%s): the "
                                 "validator does not start", attempt, type(exc).__name__)
                    raise
                delay = self.restore_backoff_s * 2 ** (attempt - 1)
                logger.warning("sandbox session restore failed (%s, attempt %d); retrying in "
                               "%.0f s", type(exc).__name__, attempt, delay)
                await asyncio.sleep(delay)
        logger.info("sandbox sessions restored: %d", restored)

    async def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        """At shutdown, after the background loops are cancelled: the state writes
        still in flight get `timeout` seconds."""
        await self.issuer.drain(timeout)

    def background(self) -> list:
        """The fleet's loops and the session maintenance: started after `start`, they
        end when cancelled (the validator's shutdown)."""
        return [self.fleet.run(), self.issuer.maintain_forever(SESSION_MAINTAIN_SECONDS)]


def build_sandbox_services(config: SandboxValidatorConfig, *, validator_hotkey: str,
                           store_kwargs=None, session_store=None, read_documents=None,
                           fetch_report=None, registration=None,
                           clock: Callable[[], float] = time.time) -> SandboxServices:
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox import require_sandbox
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.routes import build_sandbox_sessions_router

    require_sandbox()
    policy = config.policy or SandboxPolicy()
    signer = Signer(config.key_id, load_private_key(config.key_file))
    token_verifier = Ed25519TokenVerifier({**config.retired_keys,
                                           config.key_id: signer.public_key_b64})
    kw = dict(store_kwargs or {})

    async def documents():
        return await sandbox_store.list_machines(**kw)

    async def record(machine_id, *, at, summary):
        return await sandbox_store.record_machine_heartbeat(machine_id, at=at, summary=summary, **kw)

    fleet = Fleet(read_documents=read_documents or documents,
                  fetch_report=fetch_report or http_fetch_report, clock=clock,
                  record_heartbeat=record if read_documents is None else None,
                  directory_refresh_s=config.directory_refresh_s,
                  directory_max_age_s=config.directory_max_age_s,
                  directory_read_timeout_s=config.directory_read_timeout_s)
    book = SessionBook(policy)
    jobs: dict[str, SignedJobView] = {}
    services: SandboxServices | None = None

    def view(job_id):
        return None if services is None else services.view(job_id)

    issuer = SessionIssuer(
        book=book, store=session_store or sandbox_store.R2SessionStore(**kw), fleet=fleet,
        signer=signer, token_verifier=token_verifier,
        engagements={"corpus": CorpusEngagements(view, book, clock),
                     "rl_precommit": RlPrecommitEngagements()},
        policy=policy, clock=clock)
    fleet.on_drained = issuer.void_machine
    router = build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=validator_hotkey, prefix="/corpus",
        registration=registration, clock=clock,
        max_concurrent_closes=config.close_concurrency,
        close_body_timeout_s=config.close_body_timeout_s)
    logger.info("sandbox services built: validator key %s, %d retired keys", config.key_id,
                len(config.retired_keys))
    services = SandboxServices(signer=signer, token_verifier=token_verifier, fleet=fleet,
                               book=book, issuer=issuer, router=router, jobs=jobs)
    return services


def wire_signed_job(w, *, services: SandboxServices, routes: Callable[[], Any],
                    checkpoint_dir: str, tokenizer, vocab_size, chunk_tokens: int,
                    intake_factory=build_signed_episode_intake,
                    resolver_factory=SweTaskResolver) -> None:
    """`w.episode_intake` for a signed job, and its view for the session issuer. `routes`
    returns the app's CorpusJobRoutes (built after the intakes), read at call time.
    Until the job's router is adopted there (a hot add in progress), an open for it is
    refused `job_not_ready` (503, retried); once it is retired, `job_not_served`."""
    job = w.job
    job_id = str(job.job_id)
    services.routes = routes
    w.episode_intake = intake_factory(
        job, checkpoint_dir=checkpoint_dir, tokenizer=tokenizer, vocab_size=vocab_size,
        chunk_tokens=chunk_tokens, directory=services.fleet.directory_if_ready,
        token_verifier=services.token_verifier, sessions=services.issuer,
        seen=services.book.submitted_ids, retry_after_s=services.issuer.policy.retry_after_s)

    async def slots_remaining(index: int) -> int | None:
        table = routes()
        router = None if table is None else table.routers.get(job_id)
        if router is None:
            raise JobNotReady(job_id)
        return await router.slots_remaining(index)

    async def is_banned(hotkey: str) -> bool:
        # Read at call time: the job's ban check (`w.is_banned`) is wired with its
        # auditor, after its intake.
        check = getattr(w, "is_banned", None)
        return False if check is None else bool(await check(hotkey))

    services.jobs[job_id] = SignedJobView(
        job=job, resolve_task=resolver_factory(sandbox_split(job.episode)).resolve,
        slots_remaining=slots_remaining, is_banned=is_banned)


def wire_signed_grader(w, *, judge_records, parse_executor=None) -> None:
    if getattr(w, "grader", None) is None:
        w.grader = SignedEpisodeGrader(job=w.job, records=judge_records,
                                       source=w.episode_intake.source,
                                       parse_executor=parse_executor)


__all__ = ["CLOSE_CONCURRENCY", "SandboxServices", "SandboxValidatorConfig",
           "build_sandbox_services", "wire_signed_grader", "wire_signed_job"]
