"""Signed-episode envs on the async RL validator (rl-async-design.md §4.10; implementation map §9).

Where the policies come from. The window path reads an episode env's policy from the order contract (its env
entry's ``episode`` block and public ``catalogue``). The async path has no order: the task's registry entry
carries the same blocks, under ``params["rl_episode_envs"]`` (``EPISODE_ENVS_PARAM``)::

    {"<env>": {"episode": {kind, sandbox_env, split, env_package, tools, max_turns, max_tokens_per_turn,
                           max_episode_tokens, budgets},          # the order's episode block, validated as such
               "catalogue": {package, options, rl_split, image_lock_sha256},   # optional: a bridged env
               "version": "<manifest sha256>"}}                   # optional: the bridged install's manifest

The blocks are validated with the contract's own rules (``service_contract._validate_episode``,
``validate_catalogue``), so a block that an order would refuse is refused here too. The env name must be in
the entry's ``env_split`` (the trainer's mix). The registry is the one document validators, miners and the
trainer already read, signed off by the operator who declared the task: no second source of truth. Miners read
it back from the validator (``GET /rl/episodes``, ``episodes_answer``), which serves exactly the blocks it grades
with. The seed pool of an async policy is ``2 x GROUP_ROLLOUTS`` (a lease has 16 rollouts, its sessions are
numbered 0..15; ``pool_seeds`` only sizes the session policy's per-key caps).

The task source: a bridged env (a ``catalogue``) is built from it (``bridged_episode``: the same
``BridgedEpisodeSource`` the window validator and miners build) and registered as a catalogue spec; its
installed identity must be the block's ``env_package`` and, with ``version``, its manifest that digest. An env
without a catalogue must be a registered signed-episode env (``load_environment(name).source``).

Lease lifetime (``episode_lease_expiry_s``): sandbox time counts in a lease's latency (§4.10). Default for an
episode env: the episode's whole token budget at the reference decode speed times the margin (the single-turn
rule, with ``max_episode_tokens``), plus the session's ``wall_s`` budget (tools may use all of it), plus
``EPISODE_SESSION_SLACK_S`` (opens queued behind the fleet, image prepare, transcript upload). The 16 episodes of
a lease run side by side, so one session's budget, not 16. ``params["rl_lease_expiry_s"] = {env: seconds}``
(``LEASE_EXPIRY_PARAM``) overrides any env's lifetime, single-turn ones included (calibrated at qualification).

The sandbox side (``build_rl_lease_sandbox``) is the window's (``rl_sandbox_wiring``) without windows: the
validator's token key, the machine directory and fleet, a session book and issuer whose only engagement book is
``RlLeaseEngagements`` (kind ``rl_lease``: the lease is the precommit), sessions stored under their own prefix
(``RL_LEASE_SESSION_PREFIX``), the session open / close routes under ``/rl`` (no precommit route), and one
``LeaseEpisodeChecker`` per env. Per-hotkey live and hourly-open caps are lifted: the lease quota bounds a hotkey
(at most 16 sessions per open lease, ``exclusive``), the daily abort cap is kept, scaled for 16-session groups.

Not ported, by design: the window's session settlement (sessions marked submitted or handed back on a group's
verdict) and the yield ledger / yield quota. The engagement already makes a transcript single-use (one session
per (lease, rollout), a lease closes at its first submission) and the lease quota decides how many sessions a
hotkey holds; settlement and yield are the window's capacity accounting (who may open how much), which the
lease quota replaces. Sessions end by their close (final record) or their expiry (``maintain_forever``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EPISODE_ENVS_PARAM = "rl_episode_envs"
LEASE_EXPIRY_PARAM = "rl_lease_expiry_s"
EPISODE_SESSION_SLACK_S = 300.0
"""Beyond the session's wall budget: an open waiting for a fleet slot, the image prepare, the transcript upload."""
SESSION_MAINTAIN_SECONDS = 60.0
RESTORE_ATTEMPTS = 3
RESTORE_TIMEOUT_S = 60.0
RESTORE_BACKOFF_S = 5.0
STOP_TIMEOUT_S = 10.0
_BLOCK_FIELDS = frozenset({"episode", "catalogue", "version"})


def _rollouts() -> int:
    from reliquary.protocol.rl_submit import GROUP_ROLLOUTS

    return GROUP_ROLLOUTS


def validate_episode_block(name: str, block: Any) -> dict:
    """One ``rl_episode_envs`` entry, checked with the order's rules; a canonical copy. ValueError otherwise."""
    from reliquary.protocol.service_contract import (
        _ENV_ID, PUBLIC_SEED_POOL, ServiceContractError, _validate_episode, validate_catalogue,
    )

    if not isinstance(name, str) or not _ENV_ID.fullmatch(name):
        raise ValueError(f"invalid episode environment id {name!r}")
    if not isinstance(block, Mapping) or "episode" not in block or not set(block) <= _BLOCK_FIELDS:
        raise ValueError(f"{name}: an episode env entry is {{episode, catalogue?, version?}}")
    copy = json.loads(json.dumps(dict(block)))
    env = {"episode": copy["episode"], "sampling": {"kind": PUBLIC_SEED_POOL}, "missing_box": "graded"}
    try:
        _validate_episode(name, env)
        if "catalogue" in copy:
            validate_catalogue(name, copy["catalogue"], copy["episode"])
    except ServiceContractError as exc:
        raise ValueError(str(exc)) from exc
    version = copy.get("version")
    if version is not None and (not isinstance(version, str) or len(version) != 64
                                or any(c not in "0123456789abcdef" for c in version)):
        raise ValueError(f"{name}.version must be a sha256 hex digest")
    return copy


def episode_blocks(params: Mapping[str, Any], envs=None) -> dict[str, dict]:
    """The task's episode env blocks (``EPISODE_ENVS_PARAM``), validated; every one must be an env of ``envs``
    (the entry's env split) when given."""
    raw = params.get(EPISODE_ENVS_PARAM) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{EPISODE_ENVS_PARAM} must map env names to episode blocks")
    blocks = {str(name): validate_episode_block(name, block) for name, block in raw.items()}
    if envs is not None:
        missing = sorted(set(blocks) - set(envs))
        if missing:
            raise ValueError(f"episode envs {missing} are not in the task's env_split")
    return blocks


def async_episode_policy(name: str, block: Mapping[str, Any], *, rollouts: int | None = None):
    """The ``EpisodeEnvPolicy`` of a validated block on the async path (its pool: 2 x the lease's rollouts)."""
    from reliquary.protocol.service_contract import EPISODE_BUDGET_FIELDS, EpisodeEnvPolicy

    episode = block["episode"]
    return EpisodeEnvPolicy(
        environment=name, sandbox_env=episode["sandbox_env"], split=episode["split"],
        env_package=episode["env_package"], tools=tuple(episode["tools"]), max_turns=episode["max_turns"],
        max_tokens_per_turn=episode["max_tokens_per_turn"], max_episode_tokens=episode["max_episode_tokens"],
        budgets=tuple((f, episode["budgets"][f]) for f in EPISODE_BUDGET_FIELDS),
        pool_seeds=2 * int(rollouts or _rollouts()))


def episode_policies(blocks: Mapping[str, Mapping[str, Any]]) -> dict:
    return {name: async_episode_policy(name, block) for name, block in blocks.items()}


def episode_lease_expiry_s(policy, *, reference_tokens_per_s: float, margin: float,
                           slack_s: float = EPISODE_SESSION_SLACK_S) -> float:
    """An episode lease's lifetime: its token budget at the reference speed x margin, plus the session's wall
    budget and the open/prepare/upload slack (the module doc)."""
    from reliquary.validator.rl_leases import lease_expiry_s

    decode = lease_expiry_s(max_tokens=int(policy.max_episode_tokens), reference_tokens_per_s=reference_tokens_per_s,
                            margin=margin)
    return float(decode + float(policy.budgets_dict()["wall_s"]) + float(slack_s))


def expiry_overrides(params: Mapping[str, Any]) -> dict[str, float]:
    raw = params.get(LEASE_EXPIRY_PARAM) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{LEASE_EXPIRY_PARAM} must map env names to seconds")
    out = {}
    for name, seconds in raw.items():
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not float(seconds) > 0:
            raise ValueError(f"{LEASE_EXPIRY_PARAM}[{name!r}] must be a positive number of seconds")
        out[str(name)] = float(seconds)
    return out


def load_episode_environment(name: str, block: Mapping[str, Any], *, build_source=None):
    """The env's ``SignedEpisodeEnvironment`` (its ``source`` is the task source). Blocking: a bridged env loads
    its package and its taskset. ValueError when the install is not the block's."""
    from reliquary.environment.registry import find_environment_spec
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    catalogue = block.get("catalogue")
    if catalogue is None:
        from reliquary.environment import load_environment

        spec = find_environment_spec(name)
        if spec is None or spec.interaction_mode != "signed_episode":
            raise ValueError(f"{name}: no catalogue and not a registered signed-episode environment")
        return load_environment(name)
    from reliquary.environment.bridged_episode import (
        BridgedEnvConfig, BridgedEpisodeSource, catalogue_spec,
    )
    from reliquary.environment.registry import register_catalogue_spec

    config = BridgedEnvConfig.from_catalogue(name, catalogue)
    source = (build_source or BridgedEpisodeSource)(config)
    pinned = block["episode"]["env_package"]
    if source.identity != pinned:
        raise ValueError(f"{name}: {source.identity} is installed, the task pins {pinned}")
    manifest = source.manifest_sha256()
    if block.get("version") is not None and manifest != block["version"]:
        raise ValueError(f"{name}: the installed taskset's manifest {manifest[:16]} is not the task's "
                         f"{str(block['version'])[:16]}")
    register_catalogue_spec(catalogue_spec(name, manifest))
    return SignedEpisodeEnvironment(name, source)


def episodes_answer(blocks: Mapping[str, Mapping[str, Any]], expiry: Mapping[str, float]) -> dict:
    """``GET /rl/episodes``: what a miner builds its episode policies and task prompts from."""
    return {"environments": {name: {**dict(block), "lease_expiry_s": float(expiry[name])}
                             for name, block in sorted(blocks.items()) if name in expiry}}


# --- the sandbox side ---------------------------------------------------------------------------------------------
@dataclass
class RlLeaseSandbox:
    signer: Any = field(repr=False)
    token_verifier: Any = field(repr=False)
    fleet: Any
    book: Any
    issuer: Any
    routers: tuple
    checkers: dict
    environments: tuple[str, ...] = ()
    restore_attempts: int = RESTORE_ATTEMPTS
    restore_timeout_s: float = RESTORE_TIMEOUT_S
    restore_backoff_s: float = RESTORE_BACKOFF_S

    async def start(self) -> None:
        """Before serving: the directory read once, then the lease sessions restored (bounded, retried; a restore
        that still fails raises: a validator that lost its sessions would let a rollout be played twice)."""
        await self.fleet.refresh_if_due()
        restored = 0
        for attempt in range(1, self.restore_attempts + 1):
            try:
                restored = await asyncio.wait_for(self.issuer.restore(), self.restore_timeout_s)
                break
            except Exception as exc:
                if attempt == self.restore_attempts:
                    logger.error("rl lease sessions not restored after %d attempts (%s): the validator does not "
                                 "start", attempt, type(exc).__name__)
                    raise
                await asyncio.sleep(self.restore_backoff_s * 2 ** (attempt - 1))
        logger.info("rl lease sessions restored: %d", restored)

    def background(self) -> list:
        return [self.fleet.run(), self.issuer.maintain_forever(SESSION_MAINTAIN_SECONDS)]

    async def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        await self.issuer.drain(timeout)


def build_rl_lease_sandbox(
    config, *, validator_hotkey: str, lease_book, environments: Mapping[str, Any], policies: Mapping[str, Any],
    renderer_for: Callable[[Any], Any], registration=None, signer=None, store_kwargs=None, session_store=None,
    read_documents=None, fetch_report=None, clock: Callable[[], float] = time.time,
    require_resources: bool = False,
) -> RlLeaseSandbox:
    """``config``: ``sandbox_wiring.SandboxValidatorConfig``; ``lease_book``: the async ``LeaseBook``;
    ``environments[env].source``: each episode env's task source; ``policies[env]``: its ``EpisodeEnvPolicy``
    (``async_episode_policy``); ``renderer_for(policy)``: the turn renderer of the run's tokenizer over the
    policy's tools. ``read_documents`` / ``fetch_report`` / ``session_store`` / ``signer``: tests."""
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox import SANDBOX_COMMIT, require_sandbox, sandbox_commit_refusal
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.rl_engagements import EpisodeEnvironmentView, RlLeaseEngagements
    from reliquary.sandbox.rl_routes import BODY_TIMEOUT_S, MAX_PREAUTH
    from reliquary.sandbox.routes import build_sandbox_sessions_router
    from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionIssuer
    from reliquary.validator.rl_episode import LeaseEpisodeChecker
    from reliquary.validator.rl_sandbox_wiring import NO_PER_KEY_CAP, RL_PREFIX, rl_sandbox_policy

    require_sandbox()
    refusal = sandbox_commit_refusal(SANDBOX_COMMIT)
    if refusal is not None:     # an older verifier would charge gateway-signed refusals to miners
        logger.error("ALERT signed-episode RL cannot start: %s (sync the validator's environment)", refusal)
        raise ValueError(f"signed episodes cannot start: {refusal}")
    missing = sorted(set(policies) - set(environments))
    if missing or not policies:
        raise ValueError(f"episode envs without a loaded environment: {missing or 'none declared'}")
    rollouts = _rollouts()
    policy = rl_sandbox_policy(config.policy or SandboxPolicy(), 2 * rollouts)
    # The lease quota decides how many sessions a hotkey holds (16 per open lease, one per rollout): no fixed
    # per-key live or hourly-open cap; the daily abort cap stays (scaled for 16-session groups).
    policy = dataclasses.replace(policy, max_live_per_hotkey=NO_PER_KEY_CAP, max_opens_per_hour=NO_PER_KEY_CAP)
    signer = signer or Signer(config.key_id, load_private_key(config.key_file))
    token_verifier = Ed25519TokenVerifier({**config.retired_keys, signer.key_id: signer.public_key_b64})
    kw = dict(store_kwargs or {})

    async def documents():
        return await sandbox_store.list_machines(**kw)

    async def record(machine_id, *, at, summary):
        return await sandbox_store.record_machine_heartbeat(machine_id, at=at, summary=summary, **kw)

    fleet = Fleet(read_documents=read_documents or documents, fetch_report=fetch_report or http_fetch_report,
                  clock=clock, record_heartbeat=record if read_documents is None else None,
                  directory_refresh_s=config.directory_refresh_s, directory_max_age_s=config.directory_max_age_s,
                  directory_read_timeout_s=config.directory_read_timeout_s)
    fleet.resources_required = bool(require_resources)
    book = SessionBook(policy)
    views = {name: EpisodeEnvironmentView(policy=p, resolve_task=environments[name].source.resolve)
             for name, p in policies.items()}
    engagements = RlLeaseEngagements(book=lease_book, environments=views.get, sessions=book, rollouts=rollouts,
                                     clock=clock)
    issuer = SessionIssuer(
        book=book, store=session_store or sandbox_store.R2SessionStore(prefix=sandbox_store.RL_LEASE_SESSION_PREFIX,
                                                                       **kw),
        fleet=fleet, signer=signer, token_verifier=token_verifier, engagements={"rl_lease": engagements},
        policy=policy, clock=clock,
        # a failed open's signed receipt is checked and recorded: it frees the rollout (``engagement_held``)
        record_faults=True, open_refusal_check=fleet.open_refusal_valid)
    fleet.on_drained = issuer.void_machine
    router = build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=validator_hotkey, prefix=RL_PREFIX, registration=registration,
        clock=clock, max_concurrent_closes=config.close_concurrency,
        close_body_timeout_s=config.close_body_timeout_s, max_preauth_closes=config.close_preauth_concurrency,
        open_body_timeout_s=BODY_TIMEOUT_S, max_preauth_opens=MAX_PREAUTH)
    checkers = {name: LeaseEpisodeChecker(policy=p, renderer=renderer_for(p), source=environments[name].source,
                                          directory=fleet.directory_if_ready, token_verifier=token_verifier)
                for name, p in policies.items()}
    logger.info("rl lease sandbox built for %s (validator key %s)", sorted(policies), signer.key_id)
    return RlLeaseSandbox(signer=signer, token_verifier=token_verifier, fleet=fleet, book=book, issuer=issuer,
                          routers=(router,), checkers=checkers, environments=tuple(sorted(policies)))


__all__ = [
    "EPISODE_ENVS_PARAM", "EPISODE_SESSION_SLACK_S", "LEASE_EXPIRY_PARAM", "RlLeaseSandbox", "async_episode_policy",
    "build_rl_lease_sandbox", "episode_blocks", "episode_lease_expiry_s", "episode_policies", "episodes_answer",
    "expiry_overrides", "load_episode_environment", "validate_episode_block",
]
