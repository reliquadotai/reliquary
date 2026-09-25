"""Reliquary CLI — mine and validate commands."""

import asyncio
import atexit
import logging
import math
import os
import resource
import shutil
import socket as _socket
import subprocess
import sys
import threading
import time as _time
from pathlib import Path

import typer

from reliquary.constants import (
    B_BATCH,
    PROOF_PROCESS_ISOLATION,
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_ENVIRONMENTS,
    DEFAULT_HF_REPO_ID,
    MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
    MINER_GENERATION_BACKEND,
    MINER_VLLM_MAX_NUM_SEQS,
    PROOF_SLOTS_PER_DEVICE,
    PROTOCOL_MODEL_ID,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
    VALIDATOR_HTTP_PORT,
)
from reliquary.environment.registry import resolve_environment_mix
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
from reliquary.validator.errors import FatalProofPlaneError

_DEFAULT_ENVS = DEFAULT_ENVIRONMENTS

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")

logger = logging.getLogger(__name__)

_grader_proc: "subprocess.Popen | None" = None


# Startup registry read: how hard we try before refusing to boot. 3 sleeps of
# 2/4/8s bound the delay at 14s -- far below the time the model load below
# takes anyway, and far above an R2 503 that clears on its own.
REGISTRY_READ_ATTEMPTS = 4
REGISTRY_READ_BACKOFF_SECONDS = 2.0


async def read_task_registry_with_retry(
    read_registry,
    *,
    attempts: int = REGISTRY_READ_ATTEMPTS,
    backoff_seconds: float = REGISTRY_READ_BACKOFF_SECONDS,
):
    """Read the registry, retrying a RAISING client a bounded number of times.

    Refusing to start is correct when we cannot learn what we may pay, but a
    transient R2 error during an ordinary restart is not that, and the V1
    controller runs ``restart: no`` -- an unretried 503 leaves the validator
    down until a human notices. An ABSENT registry is not an error: it returns
    ``({}, None)`` and is handed straight back, never retried.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await read_registry()
        except Exception as exc:
            last = exc
            if attempt >= attempts:
                break
            delay = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "task registry read failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt, attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    assert last is not None
    raise last


def build_task_entry(*, task_id, profile_id, cap, overrides, env_split=None, verification=None):
    """One registry entry: shipped controller defaults, then explicit overrides."""
    from dataclasses import asdict

    from reliquary.environment.abi import canonical_sha256
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_id import normalise_task_id
    from reliquary.shared.task_registry import (
        KNOWN_VERIFICATION,
        MECHANISM_RL_DISCOVERED_PRICE,
        TaskEntry,
    )
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    # The entry's id IS the registry key. Normalise it here, at the one place
    # an operator's typing becomes an entry, so a stray space can never key a
    # task under something `resolve_task_config` will not find; a value that
    # is not a usable id at all still raises.
    task_id = normalise_task_id(task_id)
    profile = resolve_protocol_profile(profile_id)
    if env_split is not None:
        # Fail fast here; `resolve_task_config` is the runtime authority.
        declared = set(profile.environments)
        named = set(env_split)
        unknown = named - declared
        if unknown:
            raise ValueError(
                f"env_split names {sorted(unknown)}, which profile "
                f"{profile.profile_id!r} does not declare; it has "
                f"{sorted(declared)}"
            )
        # A partial split is refused at WRITE time too, not only on read: the
        # registry is shared, so an entry that omits an environment exits
        # every validator on the task with code 4 at its next restart.
        uncovered = declared - named
        if uncovered:
            raise ValueError(
                f"task {task_id!r} declares env_split but it does not cover "
                f"{sorted(uncovered)}, which profile {profile.profile_id!r} "
                f"also declares; env_split must name every profile environment"
            )
    if verification is not None and verification not in KNOWN_VERIFICATION:
        raise ValueError(
            f"--verification must be one of {', '.join(sorted(KNOWN_VERIFICATION))}, "
            f"got {verification!r}; omit it to let each validator derive it from its card"
        )
    params = asdict(PRODUCTION_PRICE_PARAMS)
    params.update(overrides)
    params["cap"] = float(cap)
    return TaskEntry(
        task_id=task_id,
        profile_id=profile.profile_id,
        profile_sha256=canonical_sha256(profile.to_generation_contract()),
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=params,
        status="active",
        retired_at=None,
        env_split=env_split,
        verification=verification,
    )


def build_contract_task_entry(
    *,
    task_id,
    from_profile,
    model_id,
    model_revision,
    model_architecture,
    environments,
    cap,
    overrides,
    verification=None,
):
    """One registry entry that CARRIES its contract, seeded from a template.

    The template is a starting point, never the authority: the entry's contract
    is what the fleet will run, and its digest is computed from that contract.

    ``verification`` stays OUTSIDE the contract, beside it on the entry: it says
    how validators check the work, not what the work is, so it must not change
    the contract's digest. A task generating on a large mixture-of-experts model
    is the case that needs it.
    """
    from dataclasses import asdict

    from reliquary.environment.abi import canonical_sha256
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_id import normalise_task_id
    from reliquary.shared.task_registry import (
        MECHANISM_RL_DISCOVERED_PRICE,
        TaskEntry,
    )
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    task_id = normalise_task_id(task_id)
    contract = dict(resolve_protocol_profile(from_profile).to_generation_contract())

    # The task id IS the contract's profile id: two tasks seeded from one
    # template must stay distinguishable to the checks that compare them.
    contract["profile_id"] = task_id
    contract["model_id"] = model_id
    contract["model_revision"] = model_revision
    # Knowing an arbitrary HF repo's architecture means fetching its config,
    # which this builder cannot do and stay pure (no network, no filesystem).
    # The operator states it; whether THIS image can run it is checked at
    # startup in `resolve_task_config`, against the image's own capability
    # list, not duplicated here where it could drift out of sync.
    contract["model_architecture"] = model_architecture

    if environments is not None:
        if not environments:
            raise ValueError(
                "--envs must name at least one environment, or be omitted "
                "to keep the template's full set"
            )
        declared = contract["environments"]
        unknown = sorted(set(environments) - set(declared))
        if unknown:
            raise ValueError(
                f"template {from_profile!r} does not declare {unknown}; "
                f"it has {sorted(declared)}"
            )
        contract["environments"] = {
            name: declared[name] for name in sorted(environments)
        }

    params = asdict(PRODUCTION_PRICE_PARAMS)
    params.update(overrides)
    params["cap"] = float(cap)
    return TaskEntry(
        task_id=task_id,
        profile_id=task_id,
        profile_sha256=canonical_sha256(contract),
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=params,
        status="active",
        retired_at=None,
        env_split=None,
        contract=contract,
        verification=verification,
    )


def build_corpus_task_entry(
    *,
    task_id,
    job_id,
    from_profile,
    model_id,
    model_revision,
    model_architecture,
    prompt_source,
    cap,
    overrides,
    verification=None,
    min_incentive_share=0.0,
):
    """One registry entry for a corpus generation job.

    The contract is built exactly as an RL task's is, narrowed to the single
    environment the job draws its prompts from, so a validator that boots this
    task installs precisely what the job reads. ``job_id`` stays OUTSIDE the
    contract, beside it on the entry: the contract says how generation happens
    and is compared against what the binary derives at startup, while the job
    says which work to do.
    """
    from dataclasses import replace

    from reliquary.environment.abi import canonical_sha256
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

    # A corpus task's price IS its cap, so accepting a floor and then
    # overwriting it would be the silent drop this CLI refuses elsewhere.
    if "floor" in overrides and float(overrides["floor"]) != float(cap):
        raise ValueError(
            f"a corpus task pins its price at its cap, so floor "
            f"{overrides['floor']} cannot be declared against cap {cap}"
        )
    entry = build_contract_task_entry(
        task_id=task_id,
        from_profile=from_profile,
        model_id=model_id,
        model_revision=model_revision,
        model_architecture=model_architecture,
        environments=[prompt_source],
        cap=cap,
        overrides=overrides,
        verification=verification,
    )
    # V0 has no price discovery: floor == cap is what keeps `advance()` still.
    params = {**entry.params, "floor": entry.params["cap"]}
    # Every verified token is paid: a floor cut here would drop small miners'
    # work, so a corpus task starts with none unless the operator names one.
    params["min_incentive_share"] = float(min_incentive_share)
    params["min_incentive_ramp_start"] = min(
        float(params.get("min_incentive_ramp_start", 0.0)), float(min_incentive_share)
    )
    contract = _with_enforced_toploc(entry.contract)
    return replace(
        entry,
        mechanism=MECHANISM_CORPUS_GENERATION,
        params=params,
        job_id=job_id,
        contract=contract,
        profile_sha256=canonical_sha256(contract),
    )


def _with_enforced_toploc(contract):
    """A corpus task is paid only on audited work, and the corpus validator
    refuses a contract without an enforced toploc proof. No compiled template
    carries one, so the template's own toploc entry is enforced if it has one,
    and Prime Intellect's deployed defaults are added otherwise."""
    from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC, TOPLOC_DEPLOYED_DEFAULTS

    proofs = [dict(p) for p in contract.get("proofs") or ()]
    toploc = [p for p in proofs if p.get("scheme") == PROOF_SCHEME_TOPLOC]
    if toploc:
        for proof in toploc:
            proof["mode"] = "enforce"
    else:
        proofs.append(TOPLOC_DEPLOYED_DEFAULTS.to_contract())
    return {**contract, "proofs": proofs}


tasks_app = typer.Typer(name="tasks", help="Declare and retire subnet tasks")
app.add_typer(tasks_app)


def _parse_env_split_option(value: str | None) -> dict[str, float] | None:
    """``"math=0.6,code=0.4"`` -> ``{"math": 0.6, "code": 0.4}``, or None."""
    if value is None:
        return None
    shares: dict[str, float] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"--env-split entries must be name=share, got {chunk!r}"
            )
        name, _, raw_share = chunk.partition("=")
        name = name.strip()
        try:
            shares[name] = float(raw_share.strip())
        except ValueError as exc:
            raise ValueError(
                f"--env-split share for {name!r} is not a number: {raw_share!r}"
            ) from exc
    if not shares:
        raise ValueError("--env-split must name at least one environment")
    return shares


@tasks_app.command("create")
def tasks_create(
    task_id: str = typer.Option(..., "--task-id"),
    profile_id: str = typer.Option(
        None, "--profile-id", help="Compiled profile to pin; refused with --model"
    ),
    cap: float = typer.Option(..., "--cap", help="Most of the pool this task may pay"),
    start: float = typer.Option(None, "--start"),
    decay: float = typer.Option(None, "--decay"),
    env_split: str = typer.Option(
        None,
        "--env-split",
        help="How the cap divides between environments, e.g. math=0.6,code=0.4",
    ),
    model: str = typer.Option(
        None, "--model", help="Model id; implies a carried contract"
    ),
    model_revision: str = typer.Option(None, "--model-revision"),
    model_architecture: str = typer.Option(
        None,
        "--model-architecture",
        help="Architecture class the model config declares, e.g. Qwen3ForCausalLM",
    ),
    from_profile: str = typer.Option(
        None, "--from-profile", help="Template to seed the contract from"
    ),
    envs: str = typer.Option(
        None, "--envs", help="Comma-separated subset of the template's environments"
    ),
    verification: str = typer.Option(
        None,
        "--verification",
        help=(
            "Pin how rollouts are verified: 'resident' holds the model on the card, "
            "'streamed' walks it one layer at a time. Omit to let each validator derive "
            "it from its own card."
        ),
    ),
) -> None:
    from reliquary.infrastructure.task_registry_store import create_task
    from reliquary.shared.task_registry import RegistryError

    overrides = {k: v for k, v in (("start", start), ("decay", decay)) if v is not None}
    try:
        if model is not None:
            if profile_id is not None:
                # An operator who passes an option believes it does
                # something; silently dropping --profile-id here would be
                # the same trap this branch keeps finding elsewhere. This
                # path is new, so nothing can already depend on the
                # permissive behaviour.
                typer.echo(
                    "error: --profile-id has no effect with --model; use "
                    "--from-profile to select the template",
                    err=True,
                )
                raise typer.Exit(code=1)
            if env_split is not None:
                # Same trap as --profile-id: the contract path builds
                # env_split=None, so the shares an operator typed would be
                # dropped without a word.
                typer.echo(
                    "error: --env-split has no effect with --model; a carried "
                    "contract declares its own environment set, selected with "
                    "--envs",
                    err=True,
                )
                raise typer.Exit(code=1)
            # The builder cannot infer any of these (a template is not a
            # network call, and architecture needs one) -- so all three are
            # required together, and each missing one is named, not guessed.
            missing = [
                flag
                for flag, value in (
                    ("--model-revision", model_revision),
                    ("--from-profile", from_profile),
                    ("--model-architecture", model_architecture),
                )
                if value is None
            ]
            if missing:
                typer.echo(
                    f"error: --model requires {', '.join(missing)}", err=True,
                )
                raise typer.Exit(code=1)
            entry = build_contract_task_entry(
                task_id=task_id,
                from_profile=from_profile,
                model_id=model,
                model_revision=model_revision,
                model_architecture=model_architecture,
                environments=(
                    None
                    if envs is None
                    else [e.strip() for e in envs.split(",") if e.strip()]
                ),
                cap=cap,
                overrides=overrides,
                verification=verification,
            )
        else:
            if profile_id is None:
                typer.echo(
                    "error: --profile-id is required unless --model is given",
                    err=True,
                )
                raise typer.Exit(code=1)
            entry = build_task_entry(
                task_id=task_id,
                profile_id=profile_id,
                cap=cap,
                overrides=overrides,
                env_split=_parse_env_split_option(env_split),
                verification=verification,
            )
        asyncio.run(create_task(entry))
    except (RegistryError, ValueError) as exc:
        # Declaring the first task is the one CLI command that can stop the
        # whole fleet: both legacy fallbacks are armed by an EMPTY registry,
        # so a first entry that is not `default` un-arms them for a task
        # nobody declared. Refuse here rather than weaken the fallbacks.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"declared task {entry.task_id} on {entry.profile_id} with cap {cap}"
        + (f", verified {entry.verification}" if entry.verification else "")
    )


@tasks_app.command("list")
def tasks_list() -> None:
    from reliquary.infrastructure.task_registry_store import read_registry
    from reliquary.shared.task_registry import total_cap

    entries, _ = asyncio.run(read_registry(strict=False))
    for task_id, entry in sorted(entries.items()):
        typer.echo(
            f"{task_id:24s} {entry.status:8s} cap={entry.params['cap']:.3f} "
            f"{entry.profile_id}"
        )
    typer.echo(f"total declared cap: {total_cap(entries):.4f} / 1.0")


@tasks_app.command("set-cap")
def tasks_set_cap(
    task_id: str = typer.Option(..., "--task-id"),
    cap: float = typer.Option(..., "--cap", help="The task's new share of the pool"),
    floor: float = typer.Option(
        None, "--floor",
        help="New price floor; omitted, an RL task keeps its floor and a corpus task's follows the cap",
    ),
    min_incentive_share: float = typer.Option(
        None, "--min-incentive-share",
        help="Minimum share of THIS task a hotkey needs to be paid; 0 pays everyone",
    ),
) -> None:
    """Change a live task's cap; its contract and digest are untouched."""
    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    try:
        asyncio.run(store.set_task_cap(
            task_id, cap, floor=floor, min_incentive_share=min_incentive_share
        ))
    except (RegistryError, store.RegistryConflict) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"task {task_id} now has cap {cap}" + (f" and floor {floor}" if floor is not None else ""))


@tasks_app.command("retire")
def tasks_retire(
    task_id: str = typer.Option(..., "--task-id"),
    retired_at: int = typer.Option(..., "--retired-at", help="drand round"),
) -> None:
    from reliquary.infrastructure.task_registry_store import retire_task_entry

    asyncio.run(retire_task_entry(task_id, retired_at))
    typer.echo(
        f"retired {task_id}; its cap stays reserved until its EMA tail decays"
    )


@tasks_app.command("contract")
def tasks_contract(task_id: str = typer.Option(..., "--task-id")) -> None:
    """Print a task's carried contract, for a deployment to mount."""
    import json

    from reliquary.infrastructure.task_registry_store import read_registry

    entries, _ = asyncio.run(read_registry(strict=False))
    entry = entries.get(task_id)
    if entry is None:
        typer.echo(f"error: no task {task_id!r} in the registry", err=True)
        raise typer.Exit(code=1)
    if entry.contract is None:
        typer.echo(
            f"error: task {task_id!r} is a legacy entry and carries no contract",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(json.dumps(entry.contract, sort_keys=True, separators=(",", ":")))


jobs_app = typer.Typer(
    name="jobs", help="Declare and cancel corpus generation jobs"
)
app.add_typer(jobs_app)


def build_job_manifest(
    *,
    job_id,
    checkpoint_repo,
    checkpoint_revision,
    checkpoint_sha256,
    prompt_source,
    prompt_count,
    renderer_id,
    eos_token_id,
    slots_per_prompt,
    temperature,
    top_p,
    top_k,
    min_new_tokens,
    max_new_tokens,
    n,
    grader_id,
    threshold,
    prompt_order,
    deadline_round,
    from_profile=None,
):
    """The manifest as the job store will hold it, refused unless every
    submission it will ever be paid for could be admitted.

    Field-level refusals live in `parse_job`, which this runs itself rather
    than leaving to the store: the rule below needs a parsed job, and a
    manifest that only fails at the store is one the source check never saw.
    The filter pairing is the one rule `parse_job` cannot see, because by then
    the filter is either built or absent.
    """
    from reliquary.corpus.job import JOB_SCHEMA, parse_job
    from reliquary.validator.corpus_service import prompt_job_for_spec

    if (grader_id is None) != (threshold is None):
        raise ValueError(
            "--grader-id and --threshold go together: a filter needs both, and "
            "a job that keeps every completion declares neither"
        )
    manifest = {
        "schema": JOB_SCHEMA,
        "job_id": job_id,
        "checkpoint_repo": checkpoint_repo,
        "checkpoint_revision": checkpoint_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "prompt_source": prompt_source,
        "prompt_count": prompt_count,
        "renderer_id": renderer_id,
        "eos_token_id": eos_token_id,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "n": n,
        },
        "slots_per_prompt": slots_per_prompt,
        "filter": (
            None
            if grader_id is None
            else {"grader_id": grader_id, "threshold": threshold}
        ),
        "prompt_order": prompt_order,
        "deadline_round": deadline_round,
    }
    # Resolving RENDERS the source's rule and BUILDING it counts its rows, and
    # both are refusals the operator would otherwise meet one submission at a
    # time: an unrenderable source fails fidelity forever, and a prompt_count
    # above the source's length is a 500 on the first submission and every one
    # after it. The profile checked against is the template the TASK is seeded
    # from, not whichever one this CLI process happens to run: it is the one
    # the fleet will render these prompts with.
    prompt_job_for_spec(parse_job(manifest), profile=from_profile)
    return manifest


@jobs_app.command("create")
def jobs_create(
    job_id: str = typer.Option(..., "--job-id", help="Name of the corpus job"),
    task_id: str = typer.Option(
        None, "--task-id", help="Registry key; defaults to the job id"
    ),
    model: str = typer.Option(
        ..., "--model", help="Frozen checkpoint repo; also the job's checkpoint"
    ),
    model_revision: str = typer.Option(..., "--model-revision"),
    model_architecture: str = typer.Option(
        ...,
        "--model-architecture",
        help="Architecture class the model config declares, e.g. Qwen3ForCausalLM",
    ),
    checkpoint_sha256: str = typer.Option(
        ..., "--checkpoint-sha256", help="64 lowercase hex characters"
    ),
    from_profile: str = typer.Option(
        ..., "--from-profile", help="Template to seed the contract from"
    ),
    prompt_source: str = typer.Option(
        ...,
        "--prompt-source",
        help="The installed environment the job draws prompts from; it becomes "
        "the contract's single environment",
    ),
    prompt_count: int = typer.Option(
        ...,
        "--prompt-count",
        help="Rows of the source this job owns; checked against the source's "
        "own length, which BUILDS it -- a dataset-backed source must be "
        "readable from here to declare a job over it",
    ),
    renderer_id: str = typer.Option(..., "--renderer-id"),
    eos_token_id: int = typer.Option(..., "--eos-token-id"),
    slots_per_prompt: int = typer.Option(..., "--slots-per-prompt"),
    max_new_tokens: int = typer.Option(
        None,
        "--max-new-tokens",
        help="Omit to take the budget the template gives this prompt source",
    ),
    cap: float = typer.Option(
        ..., "--cap", help="The task's share of the pool; also its pinned price"
    ),
    min_incentive_share: float = typer.Option(
        0.0,
        "--min-incentive-share",
        help="Minimum share of this task a hotkey needs to be paid; 0 pays every verified token",
    ),
    min_new_tokens: int = typer.Option(
        2,
        "--min-new-tokens",
        help="Tokens a completion must reach, terminator included; 2 is the "
        "lowest a job may declare, because 1 would pay for a completion whose "
        "only token is the terminator",
    ),
    temperature: float = typer.Option(1.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    top_k: int = typer.Option(0, "--top-k"),
    n: int = typer.Option(1, "--n", help="Completions per submitted slot"),
    grader_id: str = typer.Option(
        None, "--grader-id", help="Rejection sampling: what decides membership"
    ),
    threshold: float = typer.Option(None, "--threshold"),
    prompt_order: str = typer.Option("free", "--prompt-order"),
    deadline_round: int = typer.Option(None, "--deadline-round"),
    start: float = typer.Option(None, "--start"),
    decay: float = typer.Option(None, "--decay"),
    verification: str = typer.Option(
        None,
        "--verification",
        help=(
            "Pin how rollouts are verified: 'resident' holds the model on the "
            "card, 'streamed' walks it one layer at a time. Omit to let each "
            "validator derive it from its own card."
        ),
    ),
    fleet_knows_corpus_generation: bool = typer.Option(
        False,
        "--fleet-knows-corpus-generation",
        help=(
            "Required. Confirms that every validator already runs a binary "
            "that knows the 'corpus-generation' mechanism; one corpus entry "
            "makes the whole registry unreadable to any that does not, and "
            "those validators refuse to start."
        ),
    ),
) -> None:
    """Write the job manifest and the registry entry that pays for it."""
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.task_registry_store import (
        RegistryConflict,
        create_task,
    )
    from reliquary.shared.task_registry import (
        RegistryError,
        require_fleet_knows_corpus_generation,
    )

    overrides = {
        k: v for k, v in (("start", start), ("decay", decay)) if v is not None
    }
    try:
        if max_new_tokens is None:
            # The template budgets each environment for its model, and the RL
            # task on the same source generates to that length already.
            from reliquary.protocol.profiles import resolve_protocol_profile

            environments = resolve_protocol_profile(from_profile).environments
            if prompt_source not in environments:
                raise ValueError(
                    f"template {from_profile!r} does not declare {prompt_source!r}; "
                    "pass --max-new-tokens"
                )
            max_new_tokens = environments[prompt_source].max_new_tokens
        manifest = build_job_manifest(
            job_id=job_id,
            # The contract's model IS the job's frozen checkpoint. Taking both
            # from one flag is what makes them unable to disagree: a validator
            # verifying one model while admitting against another job would
            # pay for work nobody can reproduce.
            checkpoint_repo=model,
            checkpoint_revision=model_revision,
            checkpoint_sha256=checkpoint_sha256,
            prompt_source=prompt_source,
            prompt_count=prompt_count,
            renderer_id=renderer_id,
            # The same template the entry's contract is built from, so the
            # manifest is checked against the contract this command declares.
            from_profile=from_profile,
            eos_token_id=eos_token_id,
            slots_per_prompt=slots_per_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            n=n,
            grader_id=grader_id,
            threshold=threshold,
            prompt_order=prompt_order,
            deadline_round=deadline_round,
        )
        entry = build_corpus_task_entry(
            task_id=task_id or job_id,
            job_id=job_id,
            from_profile=from_profile,
            model_id=model,
            model_revision=model_revision,
            model_architecture=model_architecture,
            prompt_source=prompt_source,
            cap=cap,
            overrides=overrides,
            verification=verification,
            min_incentive_share=min_incentive_share,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Before either write, so a refusal leaves nothing behind. The guard is in
    # `task_registry` and does not know this CLI, so the flag is named here.
    try:
        require_fleet_knows_corpus_generation(
            entry, acknowledged=fleet_knows_corpus_generation
        )
    except RegistryError as exc:
        typer.echo(
            f"error: {exc} Pass --fleet-knows-corpus-generation.", err=True
        )
        raise typer.Exit(code=1) from exc

    try:
        asyncio.run(job_store.write_job(manifest, None))
    except job_store.CorpusStoreConflict as exc:
        # Replacing a live job's manifest would change the work under miners
        # already holding slots against it.
        typer.echo(
            f"error: job {job_id!r} already has a manifest; pick another job id",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # The registry write goes last because it is the one that can lose a race
    # or break the sum rule.
    try:
        asyncio.run(create_task(entry))
    except (RegistryError, RegistryConflict) as exc:
        # These two refuse INSTEAD of putting: a rule rejected the entry, or
        # every attempt lost its compare-and-swap. Nothing landed, so the
        # manifest is a job nobody pays for and is safe to take back.
        try:
            asyncio.run(job_store.delete_job(job_id))
        except Exception:
            typer.echo(
                f"error: the task was not declared AND its manifest could not "
                f"be removed; delete job {job_id!r} by hand before retrying",
                err=True,
            )
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        # Transport, timeout, anything else: the put MAY have landed. Deleting
        # the manifest now is the worst outcome available -- a declared task
        # holding a cap share and refusing every submission it is paid for --
        # so leave it and make the operator look.
        typer.echo(f"error: {exc}", err=True)
        typer.echo(
            f"error: the registry write for job {job_id!r} did not confirm, so "
            f"its manifest is LEFT IN PLACE. Run `reliquary jobs list` to see "
            f"whether the task landed before retrying.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"declared job {job_id} as task {entry.task_id} on {prompt_source} "
        f"with cap {cap} pinned as its price"
        + (f", verified {entry.verification}" if entry.verification else "")
    )


@jobs_app.command("list")
def jobs_list() -> None:
    """Every job with a manifest, and the task that declares it, if any."""
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.task_registry_store import read_registry

    entries, _ = asyncio.run(read_registry(strict=False))
    declared = {
        entry.job_id: entry
        for _, entry in sorted(entries.items(), reverse=True)
        if entry.job_id
    }
    stored = asyncio.run(job_store.list_jobs())
    for job_id in stored:
        entry = declared.get(job_id)
        if entry is None:
            # An orphan is what a failed rollback leaves; it must be visible.
            typer.echo(f"{job_id:24s} no task entry")
        else:
            typer.echo(
                f"{job_id:24s} {entry.status:8s} task={entry.task_id} "
                f"cap={entry.params['cap']:.3f}"
            )
    for job_id, entry in sorted(declared.items()):
        if job_id not in stored:
            # The mirror image: a task that would refuse its first submission.
            typer.echo(f"{job_id:24s} declared by {entry.task_id}, NO MANIFEST")


def _job_grader(job):
    """The grader `--apply-filter` scores every completion with: the job's own
    prompt source, at its own filter's threshold. Episode-mode sources cannot
    grade a single completion text this way (there is no single-turn
    `get_problem`/`compute_reward` for them), so this refuses instead of
    grading wrongly."""
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.validator.corpus_service import _owned_position

    if job.filter is None:
        raise typer.BadParameter(f"job {job.job_id!r} has no filter to apply")
    spec = ENVIRONMENT_SPECS[job.prompt_source]
    if spec.interaction_mode == "episode":
        raise typer.BadParameter(
            f"prompt source {job.prompt_source!r} is episode-mode; "
            "--apply-filter cannot grade a single completion text against it"
        )
    environment = spec.create()
    threshold = job.filter.threshold

    def grade(prompt_index: int, text: str) -> tuple[bool, float]:
        problem = environment.get_problem(_owned_position(job, prompt_index))
        reward = environment.compute_reward(problem, text)
        return reward >= threshold, reward

    return grade


@jobs_app.command("export")
def jobs_export(
    job_id: str = typer.Argument(...),
    out: str = typer.Option(..., "--out"),
    apply_filter: bool = typer.Option(False, "--apply-filter"),
    only_accepted: bool = typer.Option(False, "--only-accepted"),
) -> None:
    """Write the verified completions of a job as JSON lines.

    Written to a temporary file beside `--out` and swapped in with
    `os.replace` only once the export completes, so a mid-stream failure (the
    record store, the grader) never leaves a truncated, valid-looking dataset
    in its place -- and any pre-existing `--out` is untouched until then.
    """
    import json

    from reliquary.corpus.export import export_rows
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    async def _run() -> int:
        job, _ = await job_store.read_job(job_id)
        if job is None:
            raise typer.BadParameter(f"no job {job_id!r}")
        grade = _job_grader(job) if apply_filter else None
        temporary = f"{out}.{os.getpid()}.tmp"
        written = 0
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                async for row in export_rows(
                    job=job, records=BucketRecordStore(), grade=grade
                ):
                    if only_accepted and not row.get("accepted", True):
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
            os.replace(temporary, out)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return written

    typer.echo(f"{asyncio.run(_run())} rows written to {out}")


@jobs_app.command("status")
def jobs_status(job_id: str = typer.Argument(...)) -> None:
    """How far a job is from drained: accepted, audited and settled counts.

    Read-only. The stop procedure waits for `drained: yes` before
    `jobs cancel`: retiring the task is a boot gate, so anything not yet
    audited or settled when the corpus validator stops is never paid.
    """
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    async def _read():
        records = BucketRecordStore()
        submissions = set(await records.list_submission_ids(job_id))
        verdicts = set(await records.list_verdict_ids(job_id))
        state, _ = await records.read_settlement(job_id)
        return submissions, verdicts, state or {}

    submissions, verdicts, state = asyncio.run(_read())
    settled = verdicts & set(state.get("settled") or ())
    pending = state.get("pending")
    pending_window = pending["window"] if pending else None
    last_window = state.get("last_window")
    unaudited = len(submissions - verdicts)
    unsettled = len(verdicts - settled)
    typer.echo(
        f"{job_id}: submissions={len(submissions)} verdicts={len(verdicts)} "
        f"unaudited={unaudited} settled={len(settled)} unsettled={unsettled} "
        f"pending={'none' if pending_window is None else pending_window} "
        f"last_window={'none' if last_window is None else last_window}"
    )
    drained = unaudited == 0 and unsettled == 0 and pending is None
    typer.echo(f"drained: {'yes' if drained else 'no'}")


@jobs_app.command("fingerprint")
def jobs_fingerprint(
    checkpoint: str = typer.Argument(..., help="HF repo id or local directory"),
    revision: str = typer.Option("", "--revision", help="HF revision (repo ids only)"),
) -> None:
    """Print the value `jobs create --checkpoint-sha256` expects for a checkpoint."""
    from pathlib import Path

    from reliquary.corpus.encoding import checkpoint_fingerprint

    directory = Path(checkpoint)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(checkpoint, revision=revision or None,
                                           allow_patterns=["*.safetensors"]))
    typer.echo(checkpoint_fingerprint(directory))


@jobs_app.command("cancel")
def jobs_cancel(
    job_id: str = typer.Option(..., "--job-id"),
    retired_at: int = typer.Option(..., "--retired-at", help="drand round"),
) -> None:
    """Retire the task entry. It is a BOOT gate, not a stop.

    `resolve_task_config` refuses a retired entry at startup and `admit()`
    never reads `status`, so a validator already serving this job keeps
    admitting submissions until it restarts. The manifest stays either way:
    settlement still reads it.
    """
    from reliquary.infrastructure.task_registry_store import (
        read_registry,
        retire_task_entry,
    )

    entries, _ = asyncio.run(read_registry(strict=False))
    named = [entry for entry in entries.values() if entry.job_id == job_id]
    if not named:
        typer.echo(
            f"error: no task in the registry names job {job_id!r}", err=True
        )
        raise typer.Exit(code=1)
    for entry in named:
        asyncio.run(retire_task_entry(entry.task_id, retired_at))
    typer.echo(
        f"retired "
        + ", ".join(sorted(entry.task_id for entry in named))
        + f" for job {job_id}. This stops validators that START from now on; "
        "one already running keeps admitting submissions until it restarts. "
        "The manifest stays for settlement."
    )


corpus_app = typer.Typer(name="corpus", help="Mine a corpus generation task")
app.add_typer(corpus_app)


@corpus_app.command("mine")
def corpus_mine(
    validator_url: str = typer.Option(..., "--validator-url"),
    wallet_name: str = typer.Option("default"),
    hotkey: str = typer.Option("default"),
    wallet_path: str = typer.Option(os.getenv("BT_WALLET_PATH", "")),
    max_steps: int = typer.Option(0, help="0 = until the job completes"),
    gpu_memory_utilization: float = typer.Option(
        None, "--gpu-memory-utilization",
        help="Share of the card vLLM may take; omit for vLLM's own default",
    ),
) -> None:
    """Generate for the corpus job the validator serves, and submit it."""
    import bittensor as bt
    import httpx
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.corpus.job import parse_job
    from reliquary.miner.corpus_miner import (
        CorpusMinerHalted,
        VllmGenerator,
        issue_corpus_request as _issue,
        mine_steps,
    )
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import sign_corpus_submission
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job

    # Every corpus completion is proved from its own decode activations, so a
    # process whose active contract carries no toploc entry has nothing to
    # submit with: refuse to start rather than generate work it can't sign.
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    if proof is None:
        typer.echo(
            "error: the active protocol profile "
            f"{ACTIVE_PROTOCOL_PROFILE.profile_id!r} declares no toploc proof; "
            "corpus mining has no way to prove a completion under it",
            err=True,
        )
        raise typer.Exit(code=4)

    wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
    if wallet_path:
        wallet_kwargs["path"] = wallet_path
    wallet = bt.Wallet(**wallet_kwargs)
    http = httpx.Client(base_url=validator_url, timeout=120.0)

    class _Client:
        def job(self):
            response = http.get("/corpus/job")
            response.raise_for_status()
            return response.json()

        def cursor(self, hk):
            return int(_issue(lambda: http.get(f"/corpus/cursor/{hk}"))["cursor"])

        def submit(self, body):
            return _issue(lambda: http.post("/corpus/submit", json=body))

    client = _Client()
    job = parse_job(client.job())
    directory = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
    if checkpoint_fingerprint(directory) != job.checkpoint_sha256:
        typer.echo("error: the downloaded checkpoint does not match the job's fingerprint", err=True)
        raise typer.Exit(code=4)
    tokenizer = load_tokenizer(directory)

    def encode(text):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    renderer = renderer_for_job(job, encode, tokenizer=tokenizer)
    prompts = prompt_job_for_spec(job)
    try:
        counts = mine_steps(
            job=job, hotkey=wallet.hotkey.ss58_address, client=client,
            generator=VllmGenerator(directory, job.sampling, proof, job.eos_token_id,
                                    gpu_memory_utilization=gpu_memory_utilization),
            tokenizer=tokenizer, render=lambda i: renderer.initial_text(prompts.task_for(i)),
            sign=lambda body: sign_corpus_submission(wallet, body),
            max_steps=max_steps or None,
        )
    except CorpusMinerHalted as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(dict(exc.counts))
        raise typer.Exit(code=1) from exc
    typer.echo(counts)


@app.command("watch-verdicts")
def watch_verdicts(
    hotkey: str = typer.Option(..., help="Public miner SS58 address; no wallet or private key required"),
    validator_url: str = typer.Option(..., help="Validator HTTP(S) base URL"),
    window: int | None = typer.Option(None, min=0, help="Read all stored final outcomes for one window, then exit"),
):
    """Watch verdicts as JSON lines. Run once per hotkey; Ctrl-C stops it."""
    import httpx
    from reliquary.miner.submitter import monitor_submission_verdicts

    if not validator_url.startswith(("http://", "https://")):
        raise typer.BadParameter("Use an http:// or https:// validator URL")
    logging.basicConfig(level=logging.WARNING)

    async def run():
        submitted = asyncio.Event()
        submitted.set()
        async with httpx.AsyncClient(
            timeout=2, limits=httpx.Limits(max_connections=2, keepalive_expiry=30),
        ) as client:
            if window is not None:
                from urllib.parse import quote
                cursor = ""
                while True:
                    response = await client.get(
                        f"{validator_url.rstrip('/')}/miner-verdict-history/{quote(hotkey, safe='')}/{window}",
                        params={"after": cursor, "limit": 100},
                    )
                    response.raise_for_status()
                    page = response.json()
                    for verdict in page["verdicts"]:
                        import json
                        typer.echo(json.dumps(verdict, separators=(",", ":")))
                    next_cursor = page.get("next_cursor")
                    if not next_cursor:
                        if not page.get("snapshot_complete"):
                            typer.echo("Window history is not marked complete; missing records are not rejections.", err=True)
                        return
                    if next_cursor <= cursor:
                        raise ValueError("history cursor did not advance")
                    cursor = next_cursor
                    await asyncio.sleep(0.2)
            async with monitor_submission_verdicts(
                validator_url.rstrip("/"), hotkey, client, submitted,
                on_verdict=lambda verdict: typer.echo(verdict.model_dump_json(exclude_none=True)),
            ) as task:
                await task

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


def _resolve_cli_environment_mix(value: str) -> list[tuple[str, int]]:
    names = [name.strip() for name in value.split(",")]
    return resolve_environment_mix(
        names,
        profile_environments=ACTIVE_PROTOCOL_PROFILE.environments,
        default_batch_target=B_BATCH,
    )


def _raise_open_file_limit() -> None:
    """Lift the soft RLIMIT_NOFILE to the hard cap; Docker's 1024 default
    starved the controller of sockets (EMFILE) on 2026-09-22."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 1_048_576 if hard == resource.RLIM_INFINITY else hard
        if soft != resource.RLIM_INFINITY and soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info("Raised open file limit %d -> %d", soft, target)
    except (OSError, ValueError):
        logger.warning("Could not raise the open file limit", exc_info=True)


def _run_validator_event_loop(coroutine) -> None:
    """Run the validator and hard-exit after an unrecoverable proof fault.

    ``FatalProofPlaneError`` is raised only after ``ValidationService.run`` has
    executed its best-effort cleanup.  A faulted proof worker can still be
    blocked inside a synchronous CUDA call, though, so normal interpreter and
    extension teardown is not safe to rely on.  ``os._exit`` gives the shell
    supervisor an actual child exit and lets Docker apply its restart policy.
    """

    _raise_open_file_limit()
    try:
        asyncio.run(coroutine)
    except FatalProofPlaneError:
        logger.critical(
            "Fatal proof-plane cleanup completed; forcing process exit for "
            "supervisor restart",
            exc_info=True,
        )
        # Logging handlers flush each record, but flush the standard streams
        # explicitly because os._exit deliberately skips interpreter cleanup.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(1)


def _configured_proof_device_identities(torch_module):
    raw = os.environ.get("RELIQUARY_PROOF_DEVICES", "")
    requested = tuple(
        device.strip() for device in raw.split(",") if device.strip()
    )
    if PROTOCOL_VERSION < 3:
        if requested:
            logger.warning(
                "Ignoring RELIQUARY_PROOF_DEVICES under protocol profile %s",
                PROTOCOL_PROFILE_ID,
            )
        return ()
    if not requested:
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} requires explicit proof replicas; "
            "set RELIQUARY_PROOF_DEVICES after capacity qualification"
        )

    from reliquary.validator.proof_capacity import (
        resolve_cuda_proof_devices,
    )

    return resolve_cuda_proof_devices(
        requested,
        cuda=torch_module.cuda,
    )


def _v3_activation_checkpoint_revision(
    checkpoint: str,
    resume_from: str,
) -> str | None:
    if PROTOCOL_VERSION < 3:
        return None
    if checkpoint != PROTOCOL_MODEL_ID:
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} must bootstrap from "
            f"{PROTOCOL_MODEL_ID}@{PROTOCOL_MODEL_REVISION}"
        )
    prefix = "sha:"
    revision = (
        resume_from[len(prefix):].strip().lower()
        if resume_from.startswith(prefix)
        else ""
    )
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            f"{PROTOCOL_PROFILE_ID} requires "
            "RELIQUARY_RESUME_FROM=sha:<stamped-40-char-checkpoint>"
        )
    return revision


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _miner_requires_grader(env_names: list[str]) -> bool:
    # Miners never grade: opencode reward is validator-authoritative, so the
    # reference miner only generates rollouts. The gVisor grader runs on the
    # validator side. (Operators self-testing best-of-n run their own grader.)
    return False


def _grader_bundle_python() -> Path:
    bundle = os.environ.get(
        "GRADER_BUNDLE_PATH",
        "/opt/reliquary/reliquary/environment/grader/bundle",
    )
    return Path(bundle) / "rootfs" / "usr" / "local" / "bin" / "python3"


def _grader_is_running(socket_path: str, timeout: float = 0.5) -> bool:
    """Return True iff the grader is reachable on the Unix socket."""
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(socket_path)
        return True
    except (FileNotFoundError, ConnectionRefusedError, _socket.timeout, OSError):
        return False


def _ensure_grader_running(use_runsc: "bool | None" = None) -> None:
    """Start the grader server in the background if no one is listening.

    The grader is required for reward computation on code-execution envs
    (OpenCodeInstruct). Without it, OCI rewards silently return 0.0 and
    the validator rejects every OCI submission as a reward-claim mismatch.

    If `use_runsc` is None, auto-detect: use runsc when both the binary
    and the OCI bundle are present. Plain Python fallback is refused unless
    RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 is set for an isolated lab.
    """
    global _grader_proc
    from reliquary.constants import GRADER_SOCKET_PATH

    _logger = logging.getLogger("reliquary.cli")
    remote_executor_url = os.environ.get(
        "RELIQUARY_GRADER_EXECUTOR_URL",
        "",
    ).strip()
    remote_executor_mode = os.environ.get(
        "RELIQUARY_GRADER_EXECUTOR_MODE",
        "shadow",
    ).strip().lower()
    if remote_executor_url and remote_executor_mode not in {"shadow", "remote"}:
        raise RuntimeError(
            "RELIQUARY_GRADER_EXECUTOR_MODE must be 'shadow' or 'remote'"
        )
    needs_local_executor = (
        not remote_executor_url or remote_executor_mode == "shadow"
    )

    if _grader_is_running(GRADER_SOCKET_PATH):
        _logger.info("Grader already running at %s; reusing it", GRADER_SOCKET_PATH)
        return

    if remote_executor_url and remote_executor_mode == "remote" and use_runsc is True:
        raise RuntimeError(
            "authoritative remote grader cannot be combined with local runsc"
        )
    if remote_executor_url and remote_executor_mode == "remote":
        use_runsc = False
    elif needs_local_executor and use_runsc is None:
        use_runsc = bool(shutil.which("runsc")) and _grader_bundle_python().exists()
    if needs_local_executor and not use_runsc:
        if not _env_flag("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "0"):
            raise RuntimeError(
                "opencodeinstruct requires the gVisor/runsc grader sandbox. "
                "Install runsc and build the grader bundle, or set "
                "RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 only on isolated throwaway labs."
            )
        _logger.warning("Launching UNSANDBOXED grader because RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1 is set.")

    cmd = [sys.executable, "-m", "reliquary.environment.grader.server"]
    if use_runsc:
        cmd.append("--use-runsc")

    sanitized_env = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": os.environ.get("GRADER_HOME", "/tmp/reliquary-grader-home"),
        "GRADER_SOCKET_PATH": GRADER_SOCKET_PATH,
        "GRADER_BUNDLE_PATH": os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        ),
    }
    for name in (
        "RELIQUARY_GRADER_EXECUTOR_URL",
        "RELIQUARY_GRADER_EXECUTOR_MODE",
        "RELIQUARY_GRADER_EXECUTOR_CA",
        "RELIQUARY_GRADER_EXECUTOR_CERT",
        "RELIQUARY_GRADER_EXECUTOR_KEY",
        "RELIQUARY_GRADER_EXECUTOR_ALLOW_INSECURE_LOOPBACK",
        "RELIQUARY_GRADER_RUNTIME_ID",
        "GRADER_METRICS_PORT",
        "GRADER_HEALTH_PATH",
    ):
        value = os.environ.get(name)
        if value:
            sanitized_env[name] = value

    _logger.info(
        "Launching grader server (backend=%s, scrubbed_env=1) ...",
        (
            "local-shadow"
            if remote_executor_url and remote_executor_mode == "shadow"
            else (
                "remote"
                if remote_executor_url
                else ("runsc" if use_runsc else "python")
            )
        ),
    )
    _grader_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=sanitized_env,
        start_new_session=True,
    )

    def _cleanup() -> None:
        if _grader_proc is not None and _grader_proc.poll() is None:
            try:
                _grader_proc.terminate()
                _grader_proc.wait(timeout=5)
            except Exception:
                try:
                    _grader_proc.kill()
                except Exception:
                    pass
    atexit.register(_cleanup)

    deadline = _time.time() + 15.0
    while _time.time() < deadline:
        if _grader_is_running(GRADER_SOCKET_PATH):
            _logger.info("Grader server ready at %s", GRADER_SOCKET_PATH)
            return
        _time.sleep(0.2)

    _logger.error(
        "Grader server failed to bind %s within 15s. OCI rewards will "
        "be 0 and all OCI submissions will be rejected. Diagnose by "
        "running `python -m reliquary.environment.grader.server%s` manually.",
        GRADER_SOCKET_PATH,
        " --use-runsc" if use_runsc else "",
    )


def setup_logging(level: str = "INFO"):
    # ``%(threadName)s`` distinguishes the main asyncio loop from the
    # dedicated ``weight-setter`` thread (see ``validate`` below) when
    # tailing logs.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    wallet_path: str = typer.Option(
        os.getenv("BT_WALLET_PATH", ""),
        help="Optional wallet base path",
    ),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environments: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENTS", _DEFAULT_ENVS),
        help="Comma-separated environment names (env: RELIQUARY_ENVIRONMENTS)",
    ),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    mix = _resolve_cli_environment_mix(environments)
    env_names = [name for name, _target in mix]
    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, envs=%s)",
        network, netuid, env_names,
    )

    # Miners never grade (opencode reward is validator-authoritative), so this
    # stays False; the gVisor grader runs on the validator only.
    if _miner_requires_grader(env_names):
        _ensure_grader_running()
    elif "opencodeinstruct" in env_names:
        logger.info("OpenCode miner: reward is validator-authoritative; skipping local grader launch.")

    async def _run():
        import bittensor as bt
        import torch
        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environments
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.checkpoint_identity import (
            CheckpointIdentityError,
            MinerCheckpointIdentityStore,
            checkpoint_identity_from_state,
            default_checkpoint_identity_path,
        )
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2
        from reliquary.shared.modeling import (
            MODEL_SNAPSHOT_ALLOW_PATTERNS,
            load_text_generation_model,
            load_tokenizer,
        )

        wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        wallet = bt.Wallet(**wallet_kwargs)
        subtensor = await get_subtensor()
        checkpoint_identity_store = MinerCheckpointIdentityStore(
            default_checkpoint_identity_path(wallet.hotkey.ss58_address)
        )
        persisted_identity = checkpoint_identity_store.load()

        # --- Resolve initial checkpoint from validator if available ---
        initial_path = checkpoint  # fallback to --checkpoint arg
        initial_checkpoint_identity = None
        try:
            if validator_url:
                url = validator_url
            else:
                metagraph = await get_metagraph(subtensor, NETUID)
                url = discover_validator_url(metagraph)

            import httpx
            from huggingface_hub import snapshot_download
            async with httpx.AsyncClient(timeout=30) as client:
                state = await get_window_state_v2(url, client=client)
            advertised_identity = checkpoint_identity_from_state(state)
            if advertised_identity is not None:
                checkpoint_identity_store.assert_advertisement(
                    advertised_identity
                )
                logger.info(
                    "Validator at %s is on checkpoint %d (%s@%s). "
                    "Downloading to seed the miner model.",
                    url,
                    advertised_identity.checkpoint_n,
                    advertised_identity.repo_id,
                    advertised_identity.oid[:12],
                )
                initial_path = snapshot_download(
                    repo_id=advertised_identity.repo_id,
                    revision=advertised_identity.oid,
                    allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
                )
                initial_checkpoint_identity = advertised_identity
                logger.info("Using initial checkpoint path: %s", initial_path)
            elif persisted_identity is not None:
                raise CheckpointIdentityError(
                    "validator omitted a previously activated checkpoint"
                )
            else:
                logger.info(
                    "Validator has no published checkpoint yet — using --checkpoint=%s",
                    checkpoint,
                )
        except CheckpointIdentityError:
            raise
        except Exception as e:
            if persisted_identity is None:
                logger.warning(
                    "Could not fetch validator checkpoint (%s); falling back "
                    "to --checkpoint=%s",
                    e,
                    checkpoint,
                )
            else:
                logger.warning(
                    "Could not fetch validator checkpoint (%s); reloading "
                    "the last durably activated revision",
                    e,
                )
                initial_path = snapshot_download(
                    repo_id=persisted_identity.repo_id,
                    revision=persisted_identity.oid,
                    allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
                )
                initial_checkpoint_identity = persisted_identity

        # --- Load models from resolved path ---
        logger.info("Loading models from %s...", initial_path)
        base_load_kwargs = (
            {"revision": DEFAULT_BASE_MODEL_REVISION}
            if initial_path == DEFAULT_BASE_MODEL
            else {}
        )
        tokenizer = load_tokenizer(initial_path, **base_load_kwargs)

        # Use 2 GPUs when available (vllm on 0, HF proof on 1). Fall back to
        # sharing GPU 0 for test boxes that only expose one device.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        vllm_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **base_load_kwargs,
        ).to("cuda:0").eval()

        hf_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
            **base_load_kwargs,
        ).to(proof_device).eval()

        if initial_checkpoint_identity is not None:
            checkpoint_identity_store.commit(initial_checkpoint_identity)

        envs = load_environments(env_names)
        generator = None
        if MINER_GENERATION_BACKEND == "vllm":
            from reliquary.miner.vllm_generation import VLLMRolloutGenerator

            # vLLM owns cuda:0, where the transformers generation copy also
            # sits; on a single-device box that copy is only read for its eos
            # ids and device, so the two coexist at a lower utilisation.
            generator = VLLMRolloutGenerator(
                initial_path,
                revision=base_load_kwargs.get("revision"),
                max_num_seqs=MINER_VLLM_MAX_NUM_SEQS,
                gpu_memory_utilization=(
                    0.85 if proof_device != "cuda:0" else 0.6
                ),
            )

        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            envs=envs,
            mix=mix,
            generator=generator,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
            validator_url_override=validator_url or None,
            checkpoint_identity_store=checkpoint_identity_store,
            initial_checkpoint_identity=initial_checkpoint_identity,
        )

        # Seed engine's _loaded_checkpoint_path so the first
        # maybe_pull_checkpoint sees we're already synced (skips redundant reload).
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path

        logger.info("Miner ready. Entering main loop.")
        try:
            await engine.mine_window(subtensor, 0, use_drand=use_drand)
        except KeyboardInterrupt:
            logger.info("Miner interrupted by user")
        except Exception as e:
            logger.error("Mining loop crashed: %s", e, exc_info=True)
            raise

    asyncio.run(_run())


async def mount_corpus_service(server, entry, *, tokenizer, verify_signature=None):
    """Bind the corpus submission route to the one job this task declares.

    Everything the route needs is derived from that declaration rather than
    configured beside it: the job id comes from the registry entry, and the
    renderer from that job's own manifest, so a validator cannot be serving a
    renderer -- or a job -- the declaration did not name. Returns False, having
    done nothing, for any task that is not a corpus one.

    False is reserved for exactly that case. A corpus task that cannot be
    served RAISES, because the alternative is a validator that boots, holds
    its share of the pool and exposes no route, with a missing log line as the
    only evidence.
    """
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
    from reliquary.validator.corpus_service import renderer_for_job
    from reliquary.validator.task_config import TaskConfigError

    if entry is None:
        # The legacy fallback resolves no registry entry at all.
        return False
    mechanism = getattr(entry, "mechanism", None)
    if mechanism is None:
        # `TaskConfig` is not a `TaskEntry`, and passing the wrapper would
        # read as "not a corpus task" and mount nothing at all.
        raise TaskConfigError(
            f"the corpus mount takes the registry entry, not "
            f"{type(entry).__name__}"
        )
    if mechanism != MECHANISM_CORPUS_GENERATION:
        return False

    store = BucketJobStore()
    job, _ = await store.read_job(str(entry.job_id))
    if job is None:
        # The task is declared and would take its share of the pool, so a
        # missing manifest is a refusal to start, not a route that 404s.
        raise TaskConfigError(
            f"task {entry.task_id!r} declares corpus job {entry.job_id!r} but "
            f"the job store has no manifest for it"
        )

    def encode(text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    if verify_signature is None:
        from reliquary.protocol.signatures import verify_corpus_signature

        verify_signature = verify_corpus_signature
    mounted = server.mount_corpus_router(
        entry,
        store=store,
        tokenizer=tokenizer,
        # The job's own source decides the renderer: this validator's active
        # profile is its task contract, so a manifest naming a rendering that
        # contract does not declare refuses the mount rather than serving
        # prompts nobody declared.
        renderer=renderer_for_job(job, encode, tokenizer=tokenizer),
        verify_signature=verify_signature,
    )
    if not mounted:
        # The server applies the same rule to the same entry, so a refusal
        # here means the two disagree -- never something to walk past.
        raise TaskConfigError(
            f"task {entry.task_id!r} declares corpus job {job.job_id!r} but "
            f"the server refused to mount its route"
        )
    logger.info(
        "corpus job %s mounted: source %s, renderer %s, checkpoint %s@%s",
        job.job_id,
        job.prompt_source,
        job.renderer_id,
        job.checkpoint_repo,
        job.checkpoint_revision,
    )
    return mounted


@app.command()
def validate(
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help=(
            "Run full trainer mode (default). "
            "Pass --no-train for weight-only mode: reads R2 archives, "
            "computes EMA, submits weights. No GPU, no HF, no HTTP server."
        ),
    ),
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    wallet_path: str = typer.Option(
        os.getenv("BT_WALLET_PATH", ""),
        help="Optional wallet base path",
    ),
    checkpoint: str = typer.Option(DEFAULT_BASE_MODEL, help="HF repo id or local path of the model to load (trainer mode only)"),
    environments: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENTS", _DEFAULT_ENVS),
        help="Comma-separated environment names (trainer mode only; env: RELIQUARY_ENVIRONMENTS)",
    ),
    http_host: str = typer.Option("0.0.0.0", help="HTTP bind address (trainer mode only)"),
    http_port: int = typer.Option(VALIDATOR_HTTP_PORT, help="HTTP listen port (trainer mode only)"),
    external_ip: str = typer.Option(
        "",
        help=(
            "Public IP this validator is reachable at. Published on-chain via "
            "serve_axon so miners can discover it through the metagraph. "
            "Leave empty to skip publishing (miners then need --validator-url). "
            "Trainer mode only."
        ),
    ),
    external_port: int = typer.Option(
        0,
        help="Public port to advertise on-chain; defaults to --http-port when 0. Trainer mode only.",
    ),
    hf_repo_id: str = typer.Option(
        DEFAULT_HF_REPO_ID,
        help="HuggingFace repo ID to publish checkpoints to (must be writable with HF_TOKEN). Trainer mode only.",
    ),
    resume_from: str = typer.Option(
        os.getenv("RELIQUARY_RESUME_FROM", ""),
        help=(
            "Resume trainer from a checkpoint instead of the base model. "
            "Accepts 'sha:<40-hex>' (HF commit on --hf-repo-id) or "
            "'path:<dir>' (local ckpt_<N> directory). Trainer mode only."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
    set_weights: bool = typer.Option(
        False, "--set-weights/--no-set-weights",
        help="Corpus tasks only: also set weights from this process. Off by default: the RL validator's setter already pays every task.",
    ),
):
    """Run Reliquary validator (trainer mode by default; --no-train for weight-only)."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    # The RL environment mix (and the code grader) is resolved inside `_run`,
    # after the corpus branch: `--environments` defaults to an RL source a
    # corpus task's contract need not declare.
    if train:
        logger.info(
            "Starting Reliquary validator [trainer] (network=%s, netuid=%d, http=%s:%d)",
            network, netuid, http_host, http_port,
        )
    else:
        logger.info(
            "Starting Reliquary validator [weight-only] (network=%s, netuid=%d)",
            network, netuid,
        )

    async def _run():
        nonlocal resume_from
        from reliquary.infrastructure.chain import get_subtensor

        signer_client = None
        if os.environ.get("RELIQUARY_SIGNER_URL", "").strip():
            from reliquary.signer.client import RemoteSignerClient

            signer_client = RemoteSignerClient.from_environment(
                network=network,
                netuid=netuid,
                repo_id=hf_repo_id,
            )
            health = await asyncio.to_thread(signer_client.assert_ready)
            wallet = signer_client.public_wallet
            logger.info(
                "Remote signer ready (hotkey=%s protocol=%d)",
                health.signer_hotkey,
                health.protocol_version,
            )
        else:
            import bittensor as bt

            wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
            if wallet_path:
                wallet_kwargs["path"] = wallet_path
            wallet = bt.Wallet(**wallet_kwargs)
        subtensor = await get_subtensor()

        if train:
            from reliquary.constants import (
                PROTOCOL_GENERATION_CONTRACT,
                PROTOCOL_PROFILE_ID,
                TASK_ID,
            )
            from reliquary.infrastructure.task_registry_store import read_registry
            from reliquary.validator.task_config import (
                TaskConfigError,
                legacy_registry_fallback,
                legacy_task_config,
                resolve_task_config,
            )

            try:
                # `_run` is itself the coroutine `_run_validator_event_loop`
                # drives with `asyncio.run`, so a loop is already running here;
                # the registry read is awaited in place rather than started
                # with a second, nested `asyncio.run`. Done before any GPU or
                # model work below so an undeclared task fails fast.
                #
                # An R2 outage must still refuse (we cannot tell what we may
                # pay), which is why the read stays inside this try -- but a
                # registry that reads back wholly EMPTY, for the legacy
                # "default" task only, is not that: it is every validator
                # running today, before anyone has ever written one. Falling
                # back there is what keeps this branch from taking `default`
                # down the day it ships.
                registry_entries, _ = await read_task_registry_with_retry(
                    read_registry
                )
                if legacy_registry_fallback(registry_entries, [TASK_ID]):
                    logger.warning(
                        "No task registry in R2; starting the legacy task at "
                        "the full pool. Declare it with `reliquary tasks "
                        "create --task-id default --profile-id %s --cap 1.0` "
                        "and this fallback stops being used.",
                        PROTOCOL_PROFILE_ID,
                    )
                    task_config = legacy_task_config()
                else:
                    task_config = resolve_task_config(
                        registry_entries,
                        TASK_ID,
                        profile_id=PROTOCOL_PROFILE_ID,
                        generation_contract=PROTOCOL_GENERATION_CONTRACT,
                    )
            except TaskConfigError as exc:
                # Unlike a missing GPU lease, this is not an environment
                # fault we can run through: we would not know what we are
                # allowed to pay. 3 is the device lease, 2 is click.
                logger.critical(
                    "%s; declare it with `reliquary tasks create` before "
                    "starting this validator",
                    exc,
                )
                raise typer.Exit(code=4) from exc
            except Exception as exc:
                logger.critical(
                    "task registry could not be read (%s); refusing to start "
                    "rather than pay under unknown rules",
                    exc,
                )
                raise typer.Exit(code=4) from exc

            from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

            if getattr(task_config.entry, "mechanism", None) == MECHANISM_CORPUS_GENERATION:
                # A corpus task runs one process on one card with no RL
                # machinery at all: branch before any of it -- model load,
                # proof plane, batching -- is even imported.
                from reliquary.validator.corpus_validator import run_corpus_validator

                try:
                    await run_corpus_validator(
                        entry=task_config.entry, wallet=wallet, netuid=netuid,
                        signer_client=signer_client, http_host=http_host,
                        http_port=http_port, cap=task_config.emission_cap,
                        set_weights=set_weights,
                    )
                except RuntimeError as exc:
                    logger.critical("%s; fix the declaration before starting this validator", exc)
                    raise typer.Exit(code=4) from exc
                return

            mix = _resolve_cli_environment_mix(environments)
            env_names = [name for name, _target in mix]
            if "opencodeinstruct" in env_names:
                _ensure_grader_running()
            logger.info("RL environments: %s", env_names)

            import torch
            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
            from reliquary.validator.service import (
                ValidationService,
                load_validator_replica,
            )

            activation_checkpoint_revision = (
                _v3_activation_checkpoint_revision(checkpoint, resume_from)
            )
            logger.info("Loading model from %s...", checkpoint)
            base_load_kwargs = (
                {"revision": DEFAULT_BASE_MODEL_REVISION}
                if checkpoint == DEFAULT_BASE_MODEL
                else {}
            )
            tokenizer = load_tokenizer(checkpoint, **base_load_kwargs)

            from reliquary.validator.remote_proof import (
                RemoteProofPool, ShadowProofPool, executor_mode,
            )
            from reliquary.constants import DETACHED_TRAINER, KL_BASE_MODEL

            proof_mode = executor_mode()
            remote_pool = None
            if proof_mode != "local":
                if not DETACHED_TRAINER or KL_BASE_MODEL:
                    raise RuntimeError(
                        "remote/shadow proofs require detached training and no "
                        "controller-side fixed KL model"
                    )
                remote_pool = RemoteProofPool.from_environment(repo_id=hf_repo_id)
                remote_pool.start()
            if proof_mode == "remote":
                # No device lease here: every slot below is a card on the
                # executor host, reached over HTTPS, and this controller holds
                # no local CUDA context at all. Cards are leased where they are
                # actually bound -- in the local-proof branch below, which also
                # covers shadow mode because its local pool is authoritative.
                proof_worker_pool = remote_pool
                proof_slots = remote_pool.dispatch_devices
                proof_models = remote_pool.proxies()
                model = next(iter(proof_models.values()))
                from reliquary.validator.observed_proof_rollout import (
                    authorize_observed_live, observed_live_requested,
                    observed_restart_checkpoint,
                )
                if observed_live_requested():
                    recovered_checkpoint = observed_restart_checkpoint(remote_pool)
                    if recovered_checkpoint is not None:
                        activation_checkpoint_revision = recovered_checkpoint.revision
                        resume_from = f"sha:{activation_checkpoint_revision}"
                proof_capacity_qualification = (
                    authorize_observed_live(remote_pool, activation_checkpoint_revision)
                    if observed_live_requested()
                    else remote_pool.qualify(activation_checkpoint_revision)
                )
                if proof_capacity_qualification.get("mode") == "observed_live":
                    logger.warning("Explicit observed live rollout, capacity NOT qualified: %s",
                                   proof_capacity_qualification)
                logger.info("CPU controller: remote proof slots %s", proof_slots)
            else:
                # Resolve the proof plane's topology BEFORE loading this process's
                # replica: whether the plane is isolated decides which device that
                # replica belongs on, and "isolated" means a plane was actually
                # built, not merely that the flag is set.
                from reliquary.constants import DETACHED_TRAINER
                from reliquary.validator.proof_capacity import expand_proof_slots
                from reliquary.validator.proof_worker import (
                    assert_isolation_supported,
                    assert_proof_slots_supported,
                )

                assert_isolation_supported(
                    isolation=PROOF_PROCESS_ISOLATION,
                    detached_trainer=DETACHED_TRAINER,
                )
                proof_device_identities = _configured_proof_device_identities(
                    torch
                )
                if proof_device_identities:
                    from reliquary.constants import TASK_ID
                    from reliquary.validator.device_lease import (
                        DeviceLeaseError,
                        acquire_device_leases,
                        default_lease_directory,
                    )

                    try:
                        acquire_device_leases(
                            [identity.device_uuid for identity in proof_device_identities],
                            task_id=TASK_ID,
                            directory=default_lease_directory(),
                        )
                    except DeviceLeaseError as exc:
                        # A raw traceback under `restart: unless-stopped` is a
                        # crash loop that says nothing. Name the card, the
                        # holder and the remedy once, then exit on a code of
                        # our own (1 is the fatal proof plane, 2 is click's
                        # usage error).
                        logger.critical(
                            "%s; stop that task or point this one at free cards "
                            "with RELIQUARY_PROOF_DEVICES before starting it again",
                            exc,
                        )
                        raise typer.Exit(code=3) from exc
                proof_devices = tuple(
                    identity.device_id for identity in proof_device_identities
                )
                assert_proof_slots_supported(
                    slots_per_device=PROOF_SLOTS_PER_DEVICE,
                    isolation=PROOF_PROCESS_ISOLATION,
                    proof_devices=proof_devices,
                )
                # Capacity is validated against the PHYSICAL devices below and must
                # stay that way — it is a claim about cards, not processes. Only
                # the plane is widened to one entry per proof slot.
                proof_slots = expand_proof_slots(
                    proof_devices, PROOF_SLOTS_PER_DEVICE
                )
                isolated_plane = bool(PROOF_PROCESS_ISOLATION and proof_slots)

                # The CPU move turns VRAM into a permanent host-RSS floor. Say so
                # before paying for it, not hours later through an OOM restart.
                from reliquary.constants import KL_BASE_MODEL as _kl_base
                from reliquary.validator.proof_worker import (
                    assert_host_memory_for_cpu_replicas,
                )

                assert_host_memory_for_cpu_replicas(
                    isolated_plane=isolated_plane,
                    kl_base_model=bool(_kl_base),
                )
                model = load_validator_replica(
                    checkpoint,
                    isolated_plane=isolated_plane,
                    **base_load_kwargs,
                )

                proof_capacity_qualification = None
                if PROTOCOL_VERSION >= 3:
                    from reliquary.shared.runtime_fingerprint import (
                        collect_runtime_fingerprint,
                    )
                    from reliquary.validator.observability import (
                        immutable_build_revision,
                    )
                    from reliquary.validator.proof_capacity import (
                        load_proof_capacity_qualification,
                    )

                    manifest_path = os.environ.get(
                        "RELIQUARY_PROOF_CAPACITY_MANIFEST", ""
                    ).strip()
                    manifest_sha256 = os.environ.get(
                        "RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256", ""
                    ).strip()
                    if not manifest_path or not manifest_sha256:
                        raise RuntimeError(
                            f"{PROTOCOL_PROFILE_ID} requires a pinned "
                            "proof-capacity manifest"
                        )
                    qualification = load_proof_capacity_qualification(
                        manifest_path,
                        expected_sha256=manifest_sha256,
                    )
                    hardware = tuple(
                        identity.hardware_class
                        for identity in proof_device_identities
                    )
                    device_uuids = tuple(
                        identity.device_uuid
                        for identity in proof_device_identities
                    )
                    runtime_fingerprint_hash = collect_runtime_fingerprint(
                        generation_model=model,
                        proof_model=model,
                    )["profile_hash"]
                    from reliquary.validator.proof_capacity import (
                        capacity_budget, compute_proof_path_hash,
                    )

                    budget = capacity_budget()
                    proof_capacity_qualification = qualification.validate(
                        profile_id=PROTOCOL_PROFILE_ID,
                        model_revision=PROTOCOL_MODEL_REVISION,
                        software_revision=immutable_build_revision(),
                        checkpoint_revision=(
                            activation_checkpoint_revision or ""
                        ),
                        runtime_fingerprint_hash=runtime_fingerprint_hash,
                        proof_path_hash=compute_proof_path_hash(),
                        configured_devices=proof_devices,
                        configured_hardware=hardware,
                        configured_device_uuids=device_uuids,
                        proof_wall_seconds=budget["wall_seconds"],
                        minimum_proofs_per_environment=budget["proofs_per_environment"],
                        minimum_completion_tokens_per_environment={
                            environment: math.ceil(cap * 0.9)
                            for environment, cap in (
                                MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.items()
                            )
                        },
                    )
                    carried_from = proof_capacity_qualification.get(
                        "qualification_carried_over_from"
                    )
                    if carried_from:
                        logger.info(
                            "Proof capacity qualification carried over from "
                            "image %s: this image's proof path is byte-identical",
                            carried_from,
                        )
                    logger.info(
                        "Proof capacity qualified: %s",
                        proof_capacity_qualification,
                    )
                proof_models = {}
                proof_worker_pool = None
                if isolated_plane:
                    # The proof plane leaves this interpreter: every replica is
                    # loaded by a worker, this process keeps its own pair on the
                    # CPU, and the event loop can no longer convoy a proof thread
                    # off the GIL. Several slots may share one card.
                    from reliquary.validator.proof_worker import (
                        build_isolated_proof_plane,
                    )

                    logger.info(
                        "Starting isolated proof plane: %d slot(s) over %d GPU(s) "
                        "(%s); replicas load in the workers",
                        len(proof_slots),
                        len(proof_devices),
                        ", ".join(proof_slots),
                    )
                    proof_worker_pool, proof_models = build_isolated_proof_plane(
                        devices=proof_slots,
                        checkpoint=checkpoint,
                        load_kwargs=base_load_kwargs,
                        reference_model=model,
                        replica=task_config.verification,
                    )
                    proof_worker_pool.start()
                    logger.info(
                        "Isolated proof plane ready on %s",
                        ", ".join(proof_slots),
                    )
                else:
                    for device in proof_devices:
                        if device == "cuda:0":
                            continue
                        logger.info(
                            "Loading frozen proof replica on %s from %s",
                            device,
                            checkpoint,
                        )
                        proof_models[device] = load_text_generation_model(
                            checkpoint,
                            torch_dtype=torch.bfloat16,
                            attn_implementation=ATTN_IMPLEMENTATION,
                            **base_load_kwargs,
                        ).to(device).eval()

                if proof_mode == "shadow":
                    if proof_worker_pool is None:
                        raise RuntimeError("shadow proofs require local process isolation")
                    proof_worker_pool = ShadowProofPool(proof_worker_pool, remote_pool)

            service = ValidationService(
                wallet,
                model,
                tokenizer,
                netuid=netuid,
                use_drand=use_drand,
                http_host=http_host,
                http_port=http_port,
                external_ip=external_ip or None,
                external_port=(external_port or http_port) if external_ip else None,
                hf_repo_id=hf_repo_id,
                resume_from=resume_from or None,
                env_mix=mix,
                proof_devices=proof_slots or None,
                proof_models=proof_models or None,
                proof_capacity_qualification=(
                    proof_capacity_qualification
                ),
                emission_cap=task_config.emission_cap,
                price_params=task_config.price_params,
                env_caps=task_config.env_caps,
                proof_worker_pool=proof_worker_pool,
                signer_client=signer_client,
            )
            from reliquary.validator.corpus_service import CorpusPromptSourceError

            try:
                # After the server exists and before it is served, so the
                # route's one fidelity cache lives on the loop that answers.
                await mount_corpus_service(
                    service.server, task_config.entry, tokenizer=tokenizer
                )
            # `CorpusPromptSourceError` beside it, not under it: a renderer
            # this validator's own profile does not declare is a declaration
            # to fix, and on `TaskConfigError` alone it left as a traceback.
            except (TaskConfigError, CorpusPromptSourceError) as exc:
                logger.critical(
                    "%s; fix the declaration with `reliquary jobs` before "
                    "starting this validator",
                    exc,
                )
                raise typer.Exit(code=4) from exc
            # Run the weight setter in a dedicated OS thread with its own
            # event loop. asyncio is single-threaded, so any sync blocking
            # call on the trainer's loop (e.g. /state acquiring a lock the
            # GRAIL verifier is holding) would stall set_weights too. The
            # weight setter's own subtensor (see WeightOnlyValidator.run)
            # plus its own loop here means neither side can block the other.
            from reliquary.validator.weight_only import WeightOnlyValidator

            def _run_weight_setter() -> None:
                try:
                    worker = WeightOnlyValidator(
                        wallet=wallet,
                        netuid=netuid,
                        signer_client=signer_client,
                    )
                    asyncio.run(worker.run())
                except Exception:
                    logger.exception("weight-setter thread crashed")

            threading.Thread(
                target=_run_weight_setter,
                name="weight-setter",
                daemon=True,
            ).start()
            await service.run(subtensor)
        else:
            from reliquary.validator.weight_only import WeightOnlyValidator

            validator = WeightOnlyValidator(
                wallet=wallet,
                netuid=netuid,
                signer_client=signer_client,
            )
            await validator.run()

    _run_validator_event_loop(_run())


@app.command("proof-worker")
def proof_worker() -> None:
    """Serve typed GRAIL proofs on a dedicated, mutually authenticated GPU host."""
    from reliquary.validator.remote_proof_server import main
    main()


@app.command("train-worker")
def train_worker(
    shadow: bool = typer.Option(
        False,
        "--shadow",
        help=(
            "Consume payloads and train but never publish — for the "
            "pre-cutover comparison against the in-process trainer."
        ),
    ),
) -> None:
    """Detached trainer: consume R2 training payloads, publish checkpoints.

    See docs/superpowers/specs/2026-08-21-detached-trainer-r2-design.md.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(threadName)s | %(name)s | "
               "%(levelname)s | %(message)s",
    )
    from reliquary.trainer.cli import run_train_worker

    run_train_worker(shadow=shadow)


if __name__ == "__main__":
    app()
