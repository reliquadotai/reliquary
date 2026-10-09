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
from collections.abc import Mapping
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
    from_profile=None,
    model_id,
    model_revision,
    model_architecture,
    environments,
    cap,
    overrides,
    verification=None,
    base=None,
):
    """One registry entry that CARRIES its contract, seeded from a template.

    The template is ``base`` (a profile, e.g. a composed one) or ``from_profile``
    (a compiled profile id resolved here); exactly one is given.

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
    if (base is None) == (from_profile is None):
        raise ValueError("a contract is seeded from exactly one of base or from_profile")
    if base is None:
        base = resolve_protocol_profile(from_profile)
    from_profile = base.profile_id
    contract = dict(base.to_generation_contract())

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
    from_profile=None,
    model_id,
    model_revision,
    model_architecture,
    prompt_source,
    cap,
    overrides,
    verification=None,
    min_incentive_share=0.0,
    audit_params: Mapping | None = None,
    base=None,
    toploc_thresholds: Mapping | None = None,
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
        base=base,
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
    # Absent keys mean V0 (full audit); the caller (`jobs create`) is the one
    # that writes q/probation/hold defaults, so this builder itself declares
    # none unless told to.
    if audit_params:
        params.update(audit_params)
    contract = _with_enforced_toploc(entry.contract, toploc_thresholds)
    return replace(
        entry,
        mechanism=MECHANISM_CORPUS_GENERATION,
        params=params,
        job_id=job_id,
        contract=contract,
        profile_sha256=canonical_sha256(contract),
    )


def _with_enforced_toploc(contract, thresholds=None):
    """A corpus task is paid only on audited work, and the corpus validator
    refuses a contract without an enforced toploc proof. No compiled template
    carries one, so the template's own toploc entry is enforced if it has one,
    and Prime Intellect's deployed defaults are added otherwise. ``thresholds``
    (a qualification's, never under the floors) replace the proof's own."""
    from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC, TOPLOC_DEPLOYED_DEFAULTS

    proofs = [dict(p) for p in contract.get("proofs") or ()]
    toploc = [p for p in proofs if p.get("scheme") == PROOF_SCHEME_TOPLOC]
    if not toploc:
        proofs.append(TOPLOC_DEPLOYED_DEFAULTS.to_contract())
        toploc = [proofs[-1]]
    for proof in toploc:
        proof["mode"] = "enforce"
        if thresholds:
            from reliquary.eval.qualification import check_thresholds

            proof.update(check_thresholds(dict(thresholds)))
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


def _parse_set_options(values) -> dict[str, dict[str, object]]:
    """``["code.max_new_tokens=16384"]`` -> ``{"code": {"max_new_tokens": 16384}}``.

    Values are typed here because the contract is: ``thinking`` is a JSON
    boolean and every other tunable field a whole number. Which fields are
    tunable at all is ``compose_profile``'s refusal, not this parser's.
    """
    from reliquary.protocol.environment_catalog import TUNABLE_FIELDS

    overrides: dict[str, dict[str, object]] = {}
    for raw in values or ():
        target, sep, value = raw.partition("=")
        name, dot, field = target.strip().partition(".")
        if not sep or not dot or not name or not field:
            raise ValueError(f"--set takes ENV.FIELD=VALUE, got {raw!r}")
        value = value.strip()
        if field not in TUNABLE_FIELDS:
            parsed: object = value  # left for compose_profile to refuse by name
        elif field == "thinking":
            if value not in ("true", "false"):
                raise ValueError(f"--set {target}: thinking is true or false, got {value!r}")
            parsed = value == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                raise ValueError(
                    f"--set {target}: expected a whole number, got {value!r}"
                ) from None
        overrides.setdefault(name, {})[field] = parsed
    return overrides


def _run_policy_named(name, *, rollouts=None, temperature=None, top_p=None, top_k=None):
    """A named run policy, with any sampling flag the operator gave applied."""
    from dataclasses import replace

    from reliquary.protocol.composition import RUN_POLICIES

    if name not in RUN_POLICIES:
        raise ValueError(
            f"unknown run policy {name!r}; expected one of {', '.join(sorted(RUN_POLICIES))}"
        )
    run = RUN_POLICIES[name]
    sampling = {
        k: v for k, v in (
            ("rollouts", rollouts), ("temperature", temperature),
            ("top_p", top_p), ("top_k", top_k),
        ) if v is not None
    }
    if sampling:
        run = replace(run, sampling=replace(run.sampling, **sampling))
    return run


def _proofs_named(proof, proof_mode):
    """``--proof toploc --proof-mode M``: the deployed defaults in mode M."""
    from dataclasses import replace

    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS

    if proof is None and proof_mode is None:
        return ()
    if proof != "toploc":
        raise ValueError(f"--proof takes 'toploc', got {proof!r}")
    # Named, never defaulted: on an RL task enforce moves the decision off GRAIL.
    if proof_mode not in ("shadow", "enforce"):
        raise ValueError("--proof toploc needs --proof-mode shadow or enforce")
    return (replace(TOPLOC_DEPLOYED_DEFAULTS, mode=proof_mode),)


def _composed_task_entry(
    *, task_id, model, model_revision, model_architecture, prompt_encoding, envs,
    run_policy, set_values, rollouts, temperature, top_p, top_k, proof, proof_mode,
    env_split, cap, overrides, verification,
):
    """``tasks create --model`` without a template: model + run policy + catalog."""
    from reliquary.protocol.composition import ModelSpec, compose_profile
    from reliquary.shared.task_id import normalise_task_id

    if env_split is not None:
        raise ValueError(
            "--env-split has no effect with --model; a carried contract declares "
            "its own environment set, selected with --envs"
        )
    missing = [
        flag for flag, value in (
            ("--model-revision", model_revision),
            ("--model-architecture", model_architecture),
            ("--prompt-encoding", prompt_encoding),
            ("--envs", envs),
            ("--run-policy", run_policy),
        ) if value is None
    ]
    if missing:
        raise ValueError(
            f"composing a contract from --model requires {', '.join(missing)} "
            "(or seed it from a template with --from-profile)"
        )
    task_id = normalise_task_id(task_id)
    profile = compose_profile(
        profile_id=task_id,
        model=ModelSpec(
            model_id=model, model_revision=model_revision,
            model_architecture=model_architecture, prompt_encoding=prompt_encoding,
            proofs=_proofs_named(proof, proof_mode),
        ),
        run=_run_policy_named(
            run_policy, rollouts=rollouts, temperature=temperature, top_p=top_p, top_k=top_k,
        ),
        environments=[e.strip() for e in envs.split(",") if e.strip()],
        overrides=_parse_set_options(set_values),
    )
    return build_contract_task_entry(
        task_id=task_id, base=profile, model_id=model, model_revision=model_revision,
        model_architecture=model_architecture, environments=None, cap=cap,
        overrides=overrides, verification=verification,
    )


@tasks_app.command("create")
def tasks_create(
    task_id: str = typer.Option(..., "--task-id"),
    service_contract_file: Path = typer.Option(None, "--service-contract", exists=True, dir_okay=False, readable=True),
    ack_service_fleet: bool = typer.Option(False, "--ack-service-fleet"),
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
        None,
        "--envs",
        help="Comma-separated environments: a subset of the template's, or the catalog's when composing",
    ),
    prompt_encoding: str = typer.Option(
        None, "--prompt-encoding", help="Composing: raw (base model) or chat_template"
    ),
    run_policy: str = typer.Option(
        None, "--run-policy", help="Composing: named run policy, e.g. dapo-v6 or suite-v9"
    ),
    set_values: list[str] = typer.Option(
        None, "--set", help="Composing: ENV.FIELD=VALUE override of a tunable field; repeatable"
    ),
    rollouts: int = typer.Option(None, "--rollouts", help="Composing: overrides the run policy"),
    temperature: float = typer.Option(None, "--temperature", help="Composing: overrides the run policy"),
    top_p: float = typer.Option(None, "--top-p", help="Composing: overrides the run policy"),
    top_k: int = typer.Option(None, "--top-k", help="Composing: overrides the run policy"),
    proof: str = typer.Option(None, "--proof", help="Composing: 'toploc' adds the deployed TOPLOC defaults"),
    proof_mode: str = typer.Option(None, "--proof-mode", help="With --proof: shadow or enforce"),
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
    # Flags that only mean something when the contract is composed.
    compose_flags = [
        flag for flag, value in (
            ("--prompt-encoding", prompt_encoding), ("--run-policy", run_policy),
            ("--set", set_values or None), ("--rollouts", rollouts),
            ("--temperature", temperature), ("--top-p", top_p), ("--top-k", top_k),
            ("--proof", proof), ("--proof-mode", proof_mode),
        ) if value is not None
    ]
    try:
        if compose_flags and (model is None or from_profile is not None):
            # The template (or compiled profile) would silently win over them.
            typer.echo(
                f"error: {', '.join(compose_flags)} only apply when composing a "
                "contract: give --model without --from-profile",
                err=True,
            )
            raise typer.Exit(code=1)
        if model is not None and from_profile is None and profile_id is None:
            entry = _composed_task_entry(
                task_id=task_id, model=model, model_revision=model_revision,
                model_architecture=model_architecture, prompt_encoding=prompt_encoding,
                envs=envs, run_policy=run_policy, set_values=set_values,
                rollouts=rollouts, temperature=temperature, top_p=top_p, top_k=top_k,
                proof=proof, proof_mode=proof_mode, env_split=env_split, cap=cap,
                overrides=overrides, verification=verification,
            )
        elif model is not None:
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
        if service_contract_file is not None:
            from dataclasses import replace
            from reliquary.protocol.service_contract import ServiceContract
            from reliquary.shared.strict_json import strict_json_loads
            from reliquary.shared.task_registry import MECHANISM_SERVICE_RL
            if not ack_service_fleet:
                raise ValueError("service declaration requires --ack-service-fleet for upgraded registry readers")
            if service_contract_file.stat().st_size > 65536:
                raise ValueError("service contract exceeds 64 KiB")
            contract = ServiceContract.from_dict(strict_json_loads(service_contract_file.read_bytes()))
            entry = replace(entry, mechanism=MECHANISM_SERVICE_RL, service_contract=contract.to_dict(),
                            params={**entry.params, "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0})
        elif ack_service_fleet:
            raise ValueError("--ack-service-fleet requires --service-contract")
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
            f"{task_id:24s} {entry.status:8s} admission={entry.admission} cap={entry.params['cap']:.3f} "
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
    audit_q: float = typer.Option(
        None, "--audit-q",
        help="Sampled fraction of audits once a hotkey is out of probation; 1.0 audits everything",
    ),
    audit_probation_submissions: int = typer.Option(
        None, "--audit-probation-submissions",
        help="Audited passes a new hotkey needs, with no confirmed failure, before sampling starts",
    ),
    audit_hold_seconds: float = typer.Option(
        None, "--audit-hold-seconds",
        help="Hold before an unaudited (sampled and not drawn) submission is payable",
    ),
    audit_suspect_seconds: float = typer.Option(
        None, "--audit-suspect-seconds",
        help="How long a hotkey with one confirmed failure is audited at 100%",
    ),
    audit_ban_after_failures: int = typer.Option(
        None, "--audit-ban-after-failures",
        help="Confirmed failures inside the ban window that ban the hotkey",
    ),
    audit_ban_window_seconds: float = typer.Option(
        None, "--audit-ban-window-seconds",
        help="The window confirmed failures are counted in for a ban",
    ),
    audit_ban_seconds: float = typer.Option(
        None, "--audit-ban-seconds",
        help="How long a ban lasts",
    ),
) -> None:
    """Change a live task's cap; its contract and digest are untouched."""
    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    try:
        asyncio.run(store.set_task_cap(
            task_id, cap, floor=floor, min_incentive_share=min_incentive_share, audit_q=audit_q,
            audit_probation_submissions=audit_probation_submissions,
            audit_hold_seconds=audit_hold_seconds,
            audit_suspect_seconds=audit_suspect_seconds,
            audit_ban_after_failures=audit_ban_after_failures,
            audit_ban_window_seconds=audit_ban_window_seconds,
            audit_ban_seconds=audit_ban_seconds,
        ))
    except (RegistryError, store.RegistryConflict) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"task {task_id} now has cap {cap}" + (f" and floor {floor}" if floor is not None else ""))


def _task_admission(task_id: str, admission: str) -> None:
    from reliquary.infrastructure.task_registry_store import RegistryConflict, set_task_admission
    from reliquary.shared.task_registry import RegistryError

    try:
        asyncio.run(set_task_admission(task_id, admission))
    except (RegistryError, RegistryConflict) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"task {task_id} admission {admission}; active state and cap unchanged")


@tasks_app.command("pause")
def tasks_pause(task_id: str = typer.Option(..., "--task-id")) -> None:
    """Stop new corpus admissions on the control's next registry refresh; drain existing work."""
    _task_admission(task_id, "paused")


@tasks_app.command("resume")
def tasks_resume(task_id: str = typer.Option(..., "--task-id")) -> None:
    """Reopen a paused active corpus task; terminal retirement cannot resume."""
    _task_admission(task_id, "open")


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


@tasks_app.command("close")
def tasks_close(
    task_id: str = typer.Option(..., "--task-id"),
    cut_tail: bool = typer.Option(
        False, "--cut-tail",
        help="A task settled by RL window: stop paying its frozen tail now",
    ),
) -> None:
    """Close a finished corpus task: cap 0 and retired, so it pays nothing more
    and its share of the pool is free. Refused while its job is not drained, or
    while a period-settled task still pays what it earned."""
    from reliquary.validator.corpus_close import TaskNotClosable, close_task

    try:
        message = asyncio.run(close_task(task_id, cut_tail=cut_tail))
    except TaskNotClosable as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(message)


@tasks_app.command("contract")
def tasks_contract(
    task_ids: list[str] = typer.Option(
        ..., "--task-id",
        help="Repeat for corpus tasks one validator serves together: prints their merged contract",
    ),
) -> None:
    """Print a task's carried contract, for a deployment to mount."""
    import json

    from reliquary.infrastructure.task_registry_store import read_registry
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
    from reliquary.validator.task_config import merge_corpus_contracts

    entries, _ = asyncio.run(read_registry(strict=False))
    contracts = {}
    for task_id in task_ids:
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
        if len(task_ids) > 1 and entry.mechanism != MECHANISM_CORPUS_GENERATION:
            typer.echo(
                f"error: task {task_id!r} is {entry.mechanism!r}; only corpus tasks "
                "share one validator's contract",
                err=True,
            )
            raise typer.Exit(code=1)
        contracts[task_id] = entry.contract
    if len(contracts) == 1:
        contract = next(iter(contracts.values()))
    else:
        try:
            contract = merge_corpus_contracts(contracts)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(contract, sort_keys=True, separators=(",", ":")))


envs_app = typer.Typer(
    name="envs", help="The environment catalog composed contracts are built from"
)
app.add_typer(envs_app)


@envs_app.command("list")
def envs_list() -> None:
    """Every catalogued environment, its default budget, source profile and digest."""
    from reliquary.protocol.environment_catalog import (
        CATALOG_PROVENANCE,
        ENVIRONMENT_CATALOG,
        environment_body_contract,
    )
    from reliquary.protocol.release_contract import canonical_sha256

    typer.echo(f"{'environment':<38} {'max_new_tokens':>14}  {'sha256':<12}  provenance")
    for name in sorted(ENVIRONMENT_CATALOG):
        digest = canonical_sha256(environment_body_contract(name))
        typer.echo(
            f"{name:<38} {ENVIRONMENT_CATALOG[name].max_new_tokens:>14}  "
            f"{digest[:12]}  {CATALOG_PROVENANCE[name]}"
        )


@envs_app.command("show")
def envs_show(name: str = typer.Argument(...)) -> None:
    """One catalog body as JSON, with its digest, provenance and tunable fields."""
    import json

    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.protocol.environment_catalog import (
        CATALOG_PROVENANCE,
        ENVIRONMENT_CATALOG,
        TUNABLE_FIELDS,
        environment_body_contract,
    )
    from reliquary.protocol.release_contract import canonical_sha256

    if name not in ENVIRONMENT_CATALOG:
        reason = (
            "is installed but has no catalog entry; add and review one first"
            if name in ENVIRONMENT_SPECS else "is not an environment"
        )
        typer.echo(
            f"error: {name!r} {reason}; catalogued: {', '.join(sorted(ENVIRONMENT_CATALOG))}",
            err=True,
        )
        raise typer.Exit(code=1)
    body = environment_body_contract(name)
    typer.echo(json.dumps({
        "name": name,
        "body": body,
        "canonical_sha256": canonical_sha256(body),
        "provenance": CATALOG_PROVENANCE[name],
        "tunable_fields": sorted(TUNABLE_FIELDS),
    }, indent=2, sort_keys=True))


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
    profile=None,
    prompt_start=0,
    seed=None,
    submit=None,
    episode=None,
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

    # Neither means the active profile; both would leave one silently unused.
    if profile is not None and from_profile is not None:
        raise ValueError("pass the job's profile or its template id, not both")
    if (grader_id is None) != (threshold is None):
        raise ValueError(
            "--grader-id and --threshold go together: a filter needs both, and "
            "a job that keeps every completion declares neither"
        )
    from reliquary.eval.sets import refuse_held_out_overlap

    # Eval sets hold some rows out; a job may never sell them.
    refuse_held_out_overlap(prompt_source, prompt_start, prompt_count)
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
    if prompt_start != 0:
        # Written only when set, so a job declared without it stores the bytes
        # it always did; a negative start is left for `parse_job` to name.
        manifest["prompt_start"] = prompt_start
    if seed is not None:
        manifest["seed"] = seed
    if submit is not None:
        manifest["submit"] = submit
    if episode is not None:
        manifest["episode"] = episode
    # Resolving RENDERS the source's rule and BUILDING it counts its rows, and
    # both are refusals the operator would otherwise meet one submission at a
    # time: an unrenderable source fails fidelity forever, and a range
    # [prompt_start, prompt_start + prompt_count) running past the source's
    # length is a 500 on the first submission and every one
    # after it. The profile checked against is the template the TASK is seeded
    # from, not whichever one this CLI process happens to run: it is the one
    # the fleet will render these prompts with.
    prompt_job_for_spec(
        parse_job(manifest), profile=from_profile if profile is None else profile
    )
    return manifest


def _corpus_base_profile(
    *, task_id, from_profile, model, model_revision, model_architecture,
    prompt_encoding, renderer_id, prompt_source, external_eval=False,
    agentic=False,
):
    """The profile a corpus job's contract is built from: the named template, or
    one composed from the model, the ``corpus-v1`` run policy and the catalog."""
    from reliquary.protocol.composition import RUN_POLICIES, ModelSpec, compose_profile
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_id import normalise_task_id
    from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

    if from_profile is not None:
        if prompt_encoding is not None:
            raise ValueError(
                "--prompt-encoding has no effect with --from-profile: the template's "
                "encoding is kept; omit --from-profile to compose the contract"
            )
        return resolve_protocol_profile(from_profile)
    if prompt_encoding is None:
        prompt_encoding = (
            "chat_template" if renderer_id in CHAT_TEMPLATE_RENDERERS else "raw"
        )
    # No environment overrides: two jobs on one source must carry one body to merge.
    return compose_profile(
        profile_id=normalise_task_id(task_id),
        model=ModelSpec(model, model_revision, model_architecture, prompt_encoding),
        run=RUN_POLICIES["corpus-v1"],
        environments=[prompt_source],
        external_eval=external_eval,
        agentic=agentic,
    )


def prepare_corpus_job(
    *, job_id, task_id, model, model_revision, model_architecture, checkpoint_sha256,
    from_profile, prompt_encoding, prompt_source, prompt_count, prompt_start, renderer_id,
    eos_token_id, slots_per_prompt, max_new_tokens, cap, min_incentive_share, audit_params,
    min_new_tokens=2, temperature=1.0, top_p=1.0, top_k=0, n=1, grader_id=None,
    threshold=None, prompt_order="free", deadline_round=None, overrides=None,
    verification=None, seed=None, contract_environment=None, toploc_thresholds=None,
    submit=None, episode=None,
):
    """The manifest and the registry entry `jobs create` writes, built and
    checked without writing either (the admin service declares jobs with it).

    ``contract_environment`` is the catalog environment the contract declares
    when the prompt source is not one (an eval set: its own environment);
    ``toploc_thresholds`` replaces the proof's thresholds (from qualification)."""
    from reliquary.eval.prompt_source import is_eval_source

    environment = contract_environment or prompt_source
    base = _corpus_base_profile(
        task_id=task_id or job_id, from_profile=from_profile, model=model,
        model_revision=model_revision, model_architecture=model_architecture,
        prompt_encoding=prompt_encoding, renderer_id=renderer_id,
        prompt_source=environment, external_eval=is_eval_source(prompt_source), agentic=episode is not None,
    )
    if max_new_tokens is None:
        # The template or catalog budgets each environment; the length stays
        # a manifest field, so the contract body is not overridden.
        environments = base.environments
        if environment not in environments:
            raise ValueError(
                f"template {from_profile!r} does not declare {environment!r}; "
                "pass --max-new-tokens"
            )
        max_new_tokens = environments[environment].max_new_tokens
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
        prompt_start=prompt_start,
        renderer_id=renderer_id,
        # The profile the entry's contract is built from, so the manifest is
        # checked against the contract this command declares.
        profile=base,
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
        seed=seed,
        submit=submit,
        episode=episode,
    )
    entry = build_corpus_task_entry(
        task_id=task_id or job_id,
        job_id=job_id,
        base=base,
        model_id=model,
        model_revision=model_revision,
        model_architecture=model_architecture,
        prompt_source=environment,
        cap=cap,
        overrides=dict(overrides or {}),
        verification=verification,
        min_incentive_share=min_incentive_share,
        audit_params=dict(audit_params),
        toploc_thresholds=toploc_thresholds,
    )
    return manifest, entry


def _eval_job_source(*, job_id, eval_set, prompt_count, prompt_start, renderer_id, audit_q,
                     grader_id, threshold, from_profile):
    """What an evaluation job served by our own validator is declared from: the
    set's first N problems as its prompt source (checked against the lines'
    sha256), the set's environment as its contract's, a seed from its id.
    Refuses what would make it something else than a measurement."""
    import hashlib

    from reliquary.eval.prompt_source import eval_source_for, register_eval_prompts
    from reliquary.eval.storage import read_published_set
    from reliquary.protocol.external_eval import contract_environment_for
    from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

    if renderer_id not in CHAT_TEMPLATE_RENDERERS:
        raise ValueError(f"an eval set's rows render only through the model's chat template "
                         f"({sorted(CHAT_TEMPLATE_RENDERERS)}), not {renderer_id!r}")
    if prompt_start:
        raise ValueError("an eval job starts at the set's first problem (--prompt-start 0)")
    if audit_q != 1.0:
        raise ValueError("an eval job audits every submission (--audit-q 1.0): an unaudited "
                         "completion would be graded as the model's")
    if grader_id is not None or threshold is not None:
        raise ValueError("an eval job has no filter: every completion is graded, "
                         "right or wrong")
    if from_profile is not None:
        raise ValueError("an eval job's contract is composed from the model and the set's "
                         "environment: omit --from-profile")
    card, prompts = read_published_set(eval_set)
    count = int(card["count"]) if prompt_count is None else int(prompt_count)
    if not 1 <= count <= int(card["count"]):
        raise ValueError(f"set {eval_set} holds {card['count']} problems, not {count}")
    source = eval_source_for(eval_set, prompts, count)
    register_eval_prompts(source, prompts)
    seed = int(hashlib.sha256(job_id.encode()).hexdigest()[:15], 16)
    return source.name, count, contract_environment_for(card), seed


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
        None,
        "--from-profile",
        help="Legacy: seed the contract from a compiled template. Omit to compose "
        "it from the model flags, the corpus-v1 run policy and the catalog",
    ),
    prompt_encoding: str = typer.Option(
        None,
        "--prompt-encoding",
        help="Composing: raw or chat_template; defaults to chat_template for a "
        "chat-template renderer, raw otherwise",
    ),
    prompt_source: str = typer.Option(
        None,
        "--prompt-source",
        "--env",
        help="The installed environment the job draws prompts from; it becomes "
        "the contract's single environment",
    ),
    eval_set: str = typer.Option(
        None,
        "--eval-set",
        help="An evaluation: a published eval set (reliquary eval build-set / "
        "publish-set) instead of --prompt-source. Every submission is audited "
        "and graded later with `reliquary eval grade`",
    ),
    prompt_count: int = typer.Option(
        None,
        "--prompt-count",
        help="Rows of the source this job owns; checked against the source's "
        "own length, which BUILDS it -- a dataset-backed source must be "
        "readable from here to declare a job over it. With --eval-set: the "
        "set's first N problems (default: all of them)",
    ),
    prompt_start: int = typer.Option(
        0,
        "--prompt-start",
        help="First source row this job owns; it serves rows [start, start + "
        "count). Written to the manifest only when positive. Miners and "
        "validators of a job with a start need a build that knows the field",
    ),
    renderer_id: str = typer.Option(..., "--renderer-id"),
    eos_token_id: int = typer.Option(..., "--eos-token-id"),
    slots_per_prompt: int = typer.Option(..., "--slots-per-prompt"),
    max_new_tokens: int = typer.Option(
        None,
        "--max-new-tokens",
        help="Omit to take the budget the template (or the catalog) gives this prompt source",
    ),
    cap: float = typer.Option(
        ..., "--cap", help="The task's share of the pool; also its pinned price"
    ),
    min_incentive_share: float = typer.Option(
        0.0,
        "--min-incentive-share",
        help="Minimum share of this task a hotkey needs to be paid; 0 pays every verified token",
    ),
    audit_q: float = typer.Option(
        1.0,
        "--audit-q",
        help="Sampled fraction of audits once a hotkey is out of probation; 1.0 (default) audits everything",
    ),
    audit_probation_submissions: int = typer.Option(
        100,
        "--audit-probation-submissions",
        help="Audited passes a new hotkey needs, with no confirmed failure, before sampling starts",
    ),
    audit_hold_seconds: int = typer.Option(
        4320,
        "--audit-hold-seconds",
        help="Hold before an unaudited (sampled and not drawn) submission is payable",
    ),
    audit_suspect_seconds: int = typer.Option(
        86400,
        "--audit-suspect-seconds",
        help="How long a hotkey with one confirmed failure is audited at 100%",
    ),
    audit_ban_after_failures: int = typer.Option(
        3,
        "--audit-ban-after-failures",
        help="Confirmed failures inside the ban window that ban the hotkey",
    ),
    audit_ban_window_seconds: int = typer.Option(
        604800,
        "--audit-ban-window-seconds",
        help="The window confirmed failures are counted in for a ban",
    ),
    audit_ban_seconds: int = typer.Option(
        604800,
        "--audit-ban-seconds",
        help="How long a ban lasts",
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
    episode_file: str = typer.Option(
        None,
        "--episode-file",
        help="An agentic job: a JSON file holding the manifest's `episode` object "
        "(spec section 6). Its prompt source is reliquary_agentic_swe_v1",
    ),
    settlement: str = typer.Option(
        "period-ema-v1",
        "--settlement",
        help="How the task is paid: period-ema-v1 (its own 72-minute drand periods, "
        "design 2026-10-03) or windows (the RL window index, as tasks declared "
        "before it). Every validator serving it must know period-ema-v1",
    ),
    fleet_knows_period_settlement: bool = typer.Option(
        False,
        "--fleet-knows-period-settlement",
        help=(
            "Required with --settlement period-ema-v1 (the default). Confirms that "
            "the corpus validator serving the job and every weight setter run a "
            "binary that settles and replays period-ema-v1; an older one settles "
            "it by window, or does not pay it at all."
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
    from reliquary.eval.prompt_source import is_order_job_id

    if is_order_job_id(job_id) or is_order_job_id(task_id):
        # An order job is served only from its qualification record.
        typer.echo(f"error: {task_id or job_id!r} is an order id: only the admin service "
                   "declares order jobs (POST /admin/v1/jobs)", err=True)
        raise typer.Exit(code=2)
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
    contract_environment = seed = episode = None
    try:
        if (prompt_source is None) == (eval_set is None):
            raise ValueError("give exactly one of --prompt-source and --eval-set")
        if eval_set is not None:
            prompt_source, prompt_count, contract_environment, seed = _eval_job_source(
                job_id=job_id, eval_set=eval_set, prompt_count=prompt_count,
                prompt_start=prompt_start, renderer_id=renderer_id, audit_q=audit_q,
                grader_id=grader_id, threshold=threshold, from_profile=from_profile)
        elif prompt_count is None:
            raise ValueError("--prompt-count is required with --prompt-source")
        if episode_file is not None:
            import json

            with open(episode_file, encoding="utf-8") as handle:
                episode = json.load(handle)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        manifest, entry = prepare_corpus_job(
            job_id=job_id, task_id=task_id, model=model, model_revision=model_revision,
            model_architecture=model_architecture, checkpoint_sha256=checkpoint_sha256,
            from_profile=from_profile, prompt_encoding=prompt_encoding,
            prompt_source=prompt_source, prompt_count=prompt_count, prompt_start=prompt_start,
            renderer_id=renderer_id, eos_token_id=eos_token_id,
            slots_per_prompt=slots_per_prompt, max_new_tokens=max_new_tokens, cap=cap,
            min_incentive_share=min_incentive_share,
            audit_params={
                "audit_q": audit_q,
                "audit_probation_submissions": audit_probation_submissions,
                "audit_hold_seconds": audit_hold_seconds,
                "audit_suspect_seconds": audit_suspect_seconds,
                "audit_ban_after_failures": audit_ban_after_failures,
                "audit_ban_window_seconds": audit_ban_window_seconds,
                "audit_ban_seconds": audit_ban_seconds,
            },
            min_new_tokens=min_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            n=n, grader_id=grader_id, threshold=threshold, prompt_order=prompt_order,
            deadline_round=deadline_round, overrides=overrides, verification=verification,
            contract_environment=contract_environment, seed=seed, episode=episode,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if settlement not in ("period-ema-v1", "windows"):
        typer.echo(f"error: --settlement is period-ema-v1 or windows, not {settlement!r}",
                   err=True)
        raise typer.Exit(code=1)
    if settlement == "period-ema-v1" and not fleet_knows_period_settlement:
        typer.echo("error: a period-ema-v1 job needs every validator to know it: pass "
                   "--fleet-knows-period-settlement, or --settlement windows", err=True)
        raise typer.Exit(code=1)
    if settlement == "period-ema-v1":
        from dataclasses import replace as _replace

        # Outside the contract: how the task is paid, not how it generates.
        entry = _replace(entry, params={**entry.params, "settlement": settlement})

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
        + (f"rows [{prompt_start}, {prompt_start + prompt_count}) " if prompt_start else "")
        + f"with cap {cap} pinned as its price"
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
    """The grader `--apply-filter` scores every completion with (`job_grader`),
    refused as a bad parameter rather than graded wrongly."""
    from reliquary.corpus.export import job_grader

    try:
        return job_grader(job)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# The tokenizer and chat template files of a checkpoint: what the turn
# renderer loads, never the weights.
_TOKENIZER_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


def _episode_tokenizer_dir(job) -> str:
    """The job's pinned checkpoint, tokenizer files only."""
    from huggingface_hub import snapshot_download

    return snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision,
                             allow_patterns=_TOKENIZER_PATTERNS)


async def _quarantined_grade_executors() -> list[str]:
    """The grade executors the registry says are quarantined: what one of them
    decided alone is held until a regrade replaces it."""
    from reliquary.infrastructure import corpus_executor_store as executor_store

    return sorted(str(d.get("executor_id")) for d in await executor_store.list_executors()
                  if executor_store.scope_of(d) == "grade" and d.get("status") == "quarantined")


@jobs_app.command("export")
def jobs_export(
    job_id: str = typer.Argument(...),
    out: str = typer.Option(..., "--out"),
    apply_filter: bool = typer.Option(False, "--apply-filter"),
    only_accepted: bool = typer.Option(False, "--only-accepted"),
    sft: bool = typer.Option(
        False, "--sft",
        help="An episode job: keep the certified successes (graded_success and "
             "replay_certified), the SFT set"),
    allow_incomplete: bool = typer.Option(
        False, "--allow-incomplete",
        help="An episode job: export before the job is drained (the counts file says so)"),
) -> None:
    """Write the verified completions of a job as JSON lines.

    An episode job writes one row per replay-certified trajectory (messages
    rebuilt from the proven tokens, tokens, assistant mask, grade); `--sft`
    keeps the successes. It refuses a job not yet drained unless
    `--allow-incomplete`, and always writes `{out}.counts.json`: drained, what
    was exported and what was left out (ungraded, held, voided, uncertified,
    unparseable...), when, and the quarantined executors it held.

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
        episode = job.episode is not None
        if episode and (apply_filter or only_accepted):
            raise typer.BadParameter("an episode job is filtered by its grades: use --sft")
        if sft and not episode:
            raise typer.BadParameter("--sft is for episode jobs; use --apply-filter")
        if episode:
            from reliquary.corpus.delivery import episode_rows
            from reliquary.environment import agentic_swe
            from reliquary.validator import corpus_job_status

            drained = bool((await corpus_job_status.stored_job_counts(
                BucketRecordStore(), job_id))["drained"])
            if not drained and not allow_incomplete:
                raise typer.BadParameter(
                    f"job {job_id!r} is not drained: grades, regrades and voids may still "
                    "change; pass --allow-incomplete to export what is final so far")
            quarantined = await _quarantined_grade_executors()

            from reliquary.corpus.job import is_signed_sandbox, sandbox_split

            signed_job = is_signed_sandbox(job)
            tokenizer_dir = await asyncio.to_thread(_episode_tokenizer_dir, job)
            # A replay job renders with the loader's default tools, as it always did.
            renderer = await asyncio.to_thread(
                agentic_swe.load_turn_renderer, tokenizer_dir,
                *((job.episode.sandbox.tools,) if signed_job else ()))
            source = (agentic_swe.SignedSweSource(sandbox_split(job.episode)) if signed_job
                      else await asyncio.to_thread(agentic_swe.load_swe_source,
                                                   job.episode.env.num_images))
            counts: dict = {}
            rows = episode_rows(job=job, records=BucketRecordStore(), renderer=renderer,
                                source=source, counts=counts, sft_only=sft,
                                quarantined=quarantined)
        else:
            grade = _job_grader(job) if apply_filter else None
            rows = export_rows(job=job, records=BucketRecordStore(), grade=grade)
        temporary = f"{out}.{os.getpid()}.tmp"
        written = 0
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                async for row in rows:
                    if episode:
                        row = {**row, "messages": json.loads(row["messages"]),
                               "turns": json.loads(row["turns"])}
                    elif only_accepted and not row.get("accepted", True):
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
            os.replace(temporary, out)
            if episode:
                sidecar = {
                    "job_id": job_id, "drained": drained, "sft_only": sft,
                    "exported": counts.get("rows", 0),
                    **{k: counts.get(k, 0) for k in ("ungraded", "held", "voided",
                                                     "uncertified", "unparseable")},
                    "counts": counts, "exported_at": _time.time(),
                    "quarantined_executors": list(quarantined),
                }
                side_temporary = f"{out}.counts.json.{os.getpid()}.tmp"
                with open(side_temporary, "w", encoding="utf-8") as handle:
                    json.dump(sidecar, handle, sort_keys=True, indent=1)
                os.replace(side_temporary, f"{out}.counts.json")
                typer.echo(json.dumps(counts, sort_keys=True), err=True)
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
    from reliquary.validator.corpus_job_status import stored_job_counts

    counts = asyncio.run(stored_job_counts(BucketRecordStore(), job_id))
    pending_window, last_window = counts["pending_window"], counts["last_window"]
    typer.echo(
        f"{job_id}: submissions={counts['submissions']} verdicts={counts['verdicts']} "
        f"unaudited={counts['unaudited']} settled={counts['settled']} "
        f"unsettled={counts['unsettled']} "
        f"pending_records={counts['pending_records']} "
        f"pending={'none' if pending_window is None else pending_window} "
        f"last_window={'none' if last_window is None else last_window}"
    )
    drained = counts["drained"]
    typer.echo(f"drained: {'yes' if drained else 'no'}")


@jobs_app.command("miner-reset")
def jobs_miner_reset(
    job_id: str = typer.Option(..., "--job-id"),
    hotkeys: list[str] = typer.Option(None, "--hotkey", help="Repeatable"),
    all_hotkeys: bool = typer.Option(False, "--all", help="Every hotkey in the job's miners.json"),
) -> None:
    """Clear hotkeys' suspect, ban and confirmed failures in the job's miners.json.

    For a validator-side systematic failure (wrong card or kernel band, wrong
    checkpoint) that failed honest miners. Only the state is reset: verdicts
    are write-once, so records already failed or voided `banned` stay unpaid.
    """
    from dataclasses import replace

    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    if bool(hotkeys) == all_hotkeys:
        typer.echo("error: name hotkeys with --hotkey, or pass --all (not both)", err=True)
        raise typer.Exit(code=1)
    states = MinerStates(BucketRecordStore(), job_id)

    def clear(m):
        return replace(m, suspect_until=None, banned_until=None, confirmed_failures=[])

    async def _run() -> list[str]:
        stored = await states.hotkeys()
        unknown = sorted(set(hotkeys or ()) - set(stored))
        if unknown:
            # A typo must not read as a reset of a hotkey that was never caught.
            raise typer.BadParameter(f"no miner state for {', '.join(unknown)} in job {job_id!r}")
        targets = stored if all_hotkeys else sorted(set(hotkeys))
        if targets:
            await states.update_many({hotkey: clear for hotkey in targets})
        return targets

    try:
        targets = asyncio.run(_run())
    except typer.BadParameter as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"reset {len(targets)} hotkey(s) in job {job_id}: {', '.join(targets) or '-'}")
    typer.echo(
        "suspect, ban and confirmed failures cleared; audited_passed is kept, so a "
        "hotkey a failure reset to 0 goes through probation again (audited in full, "
        "paid normally). Verdicts already written stay: records failed or voided "
        "'banned' stay unpaid."
    )


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


admin_app = typer.Typer(name="admin", help="The subnet admin service the platform calls")
app.add_typer(admin_app)


def build_admin_app_from_environment():
    """The admin app as `admin serve` runs it, configured from the environment.

    ``RELIQUARY_ADMIN_SECRET``, ``RELIQUARY_ADMIN_POOL_MAX`` and
    ``RELIQUARY_ADMIN_MODELS`` (a JSON file of qualified models) are required;
    ``RELIQUARY_ADMIN_TASK_PREFIX`` (default ``order-``) bounds the task and job
    ids the platform may touch; deliveries and evaluation grading need
    ``RELIQUARY_PLATFORM_BUCKET`` and its scoped ``RELIQUARY_PLATFORM_R2_*``
    credentials. Subnet-run deliveries may instead use the origin and secret in
    ``RELIQUARY_PLATFORM_DELIVERY_URL`` and ``RELIQUARY_PLATFORM_DELIVERY_SECRET``.
    Delivery is off without either sink. Eval sets' grading files are read
    from the subnet bucket (``R2_*``).
    """
    import json

    from reliquary.admin.service import create_admin_app

    secret = os.getenv("RELIQUARY_ADMIN_SECRET", "")
    if len(secret) < 32:
        raise ValueError("RELIQUARY_ADMIN_SECRET must be set, at least 32 characters")
    pool = os.getenv("RELIQUARY_ADMIN_POOL_MAX", "").strip()
    if not pool:
        raise ValueError("RELIQUARY_ADMIN_POOL_MAX must be set: the corpus caps' total budget")
    models_path = os.getenv("RELIQUARY_ADMIN_MODELS", "").strip()
    if not models_path:
        raise ValueError("RELIQUARY_ADMIN_MODELS must name the qualified models JSON file")
    with open(models_path, encoding="utf-8") as handle:
        models = json.load(handle)
    deliveries = None
    if (os.getenv("RELIQUARY_PLATFORM_DELIVERY_URL", "").strip()
            or os.getenv("RELIQUARY_PLATFORM_DELIVERY_SECRET", "")):
        from reliquary.corpus.delivery import HTTPDeliverySink

        deliveries = HTTPDeliverySink.from_environment()
    elif os.getenv("RELIQUARY_PLATFORM_BUCKET", "").strip():
        from reliquary.corpus.delivery import R2DeliverySink

        deliveries = R2DeliverySink.from_environment()
    prefix = os.getenv("RELIQUARY_ADMIN_TASK_PREFIX", "order-")
    return create_admin_app(secret=secret.encode(), pool_max=float(pool), models=models,
                            deliveries=deliveries, task_prefix=prefix,
                            work_dir=os.getenv("RELIQUARY_ADMIN_WORK_DIR") or None)


@admin_app.command("serve")
def admin_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8790, "--port"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Serve the signed admin routes: jobs, caps, retirement, executors, deliveries."""
    import uvicorn

    setup_logging(log_level)
    try:
        admin = build_admin_app_from_environment()
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    uvicorn.run(admin, host=host, port=port, log_level=log_level.lower())


eval_app = typer.Typer(name="eval", help="Evaluation orders: frozen sets and the pod runner")
app.add_typer(eval_app)


@eval_app.command("build-set")
def eval_build_set(
    out: str = typer.Option(..., "--out", help="An empty directory for the three files"),
    preset: str | None = typer.Option(
        None, "--preset", "--env",
        help="A held-out preset (math, code, logic, instruction_following): the platform's sets"),
    source: str | None = typer.Option(
        None, "--source",
        help="A catalog environment, or verifiers:<taskset-id> for an installed taskset"),
    split: str | None = typer.Option(None, "--split", help="Catalog only; default train"),
    start: int = typer.Option(0, "--start", min=0, help="First row of the range"),
    count: int | None = typer.Option(
        None, "--count", min=1,
        help="Rows in the range (default: to the end); with --preset, the problems drawn"),
    sample: int | None = typer.Option(None, "--sample", min=1,
                                      help="Draw this many rows from the range (needs --seed)"),
    seed: int | None = typer.Option(None, "--seed"),
    taskset_args: str | None = typer.Option(
        None, "--taskset-args",
        help="Verifiers only: the taskset config as JSON, frozen into the set"),
    set_id: str | None = typer.Option(None, "--set-id"),
) -> None:
    """Freeze problems into a set: prompts.jsonl, grading.jsonl, set.json.

    Either a held-out --preset (COUNT problems drawn with SEED), or any --source
    over [START, START+COUNT), whole or --sample'd. Overlap with training is
    written on the card, never refused."""
    import json

    from reliquary.eval import sets

    try:
        if (preset is None) == (source is None):
            raise ValueError("give exactly one of --preset and --source")
        if preset is not None:
            if count is None or seed is None or sample is not None or start or split \
                    or taskset_args:
                raise ValueError("a --preset takes --count and --seed only")
            card = sets.build_set(preset, count=count, seed=seed, out=out, set_id=set_id,
                                  open_environment=sets.open_source)
        else:
            args = json.loads(taskset_args) if taskset_args else None
            if args is not None and not isinstance(args, dict):
                raise ValueError("--taskset-args must be a JSON object")
            card = sets.build_source_set(source, out=out, split=split, start=start,
                                         count=count, sample=sample, seed=seed,
                                         set_id=set_id, taskset_args=args,
                                         open_environment=sets.open_source)
    except (ValueError, FileExistsError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    keys = ("set_id", "env", "source", "split", "count", "index_range", "prompts_sha256",
            "grading_sha256", "disjointness", "taskset", "needs_runtime")
    typer.echo(json.dumps({k: card[k] for k in keys if k in card}, indent=1))


def _admin_client(admin_url: str):
    from reliquary.eval.operator import AdminClient

    secret = os.getenv("RELIQUARY_ADMIN_SECRET", "")
    if len(secret) < 32:
        raise ValueError("RELIQUARY_ADMIN_SECRET must be set: the admin service's secret")
    return AdminClient(admin_url, secret.encode())


_ADMIN_URL = typer.Option("http://127.0.0.1:8790", "--admin-url",
                          help="The admin service (reliquary admin serve)")


@eval_app.command("create")
def eval_create(
    set_ids: list[str] = typer.Option(..., "--set", help="A published set; repeat for several"),
    model: str = typer.Option(..., "--model", help="repo@<40-hex commit>"),
    samples: int = typer.Option(..., "--samples", min=1, help="Completions per problem"),
    max_new_tokens: int = typer.Option(..., "--max-new-tokens", min=1),
    thinking: bool = typer.Option(False, "--thinking/--no-thinking"),
    temperature: float = typer.Option(..., "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    top_k: int = typer.Option(0, "--top-k", min=0),
    count: int | None = typer.Option(None, "--count", min=1,
                                     help="The set's first N problems (default: all)"),
    cap: float | None = typer.Option(None, "--cap", help="The job's share (admin default 0.02)"),
    seed: int | None = typer.Option(None, "--seed"),
    job_id: str | None = typer.Option(None, "--job-id", help="With one --set only"),
    completions: int = typer.Option(32, "--qualify-completions", min=1, max=64),
    attempt: int = typer.Option(0, "--attempt", min=0,
                                help="Ask for a new qualification after a failed one"),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds"),
    admin_url: str = _ADMIN_URL,
) -> None:
    """Qualify MODEL on each set, wait, then declare one eval job per set.

    Running it again finds the same qualification and the same job."""
    import json

    from reliquary.eval import operator
    from reliquary.eval.prompt_source import TASK_PREFIX_ENV, DEFAULT_TASK_PREFIX

    try:
        repo, revision = operator.split_model(model)
        client = _admin_client(admin_url)
        cards = [operator.read_set_card(set_id) for set_id in set_ids]
        prefix = os.environ.get(TASK_PREFIX_ENV, "").strip() or DEFAULT_TASK_PREFIX
        created = operator.create_evaluations(
            client, cards=cards, model=repo, revision=revision, samples=samples,
            max_new_tokens=max_new_tokens, thinking=thinking,
            sampling={"temperature": temperature, "top_p": top_p, "top_k": top_k},
            count=count, cap=cap, seed=seed, job_id=job_id, completions=completions,
            prefix=prefix, poll_seconds=poll_seconds, attempt=attempt,
            log=lambda line: typer.echo(line, err=True))
    except (ValueError, RuntimeError, TimeoutError, operator.AdminError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps([{k: c[k] for k in ("job_id", "set_id", "qualification_id")}
                           for c in created], indent=1))


@eval_app.command("status")
def eval_status(job_id: str = typer.Option(..., "--job"), admin_url: str = _ADMIN_URL) -> None:
    """An eval job's counts: submissions, verdicts, settled, drained."""
    import json

    from reliquary.eval import operator

    try:
        status = _admin_client(admin_url).json("GET", f"/admin/v1/jobs/{job_id}/status")
    except (ValueError, operator.AdminError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({k: v for k, v in status.items() if k != "manifest"}, indent=1))


@eval_app.command("grade")
def eval_grade(
    job_id: str = typer.Option(..., "--job"),
    out: str | None = typer.Option(None, "--out", help="Write report.json, manifest.json, "
                                                       "graded.parquet here"),
    eval_id: str | None = typer.Option(None, "--eval-id", help="Default: the job id"),
    allow_incomplete: bool = typer.Option(False, "--allow-incomplete",
                                          help="Grade a drained job missing samples"),
    poll_seconds: float = typer.Option(10.0, "--poll-seconds"),
    admin_url: str = _ADMIN_URL,
) -> None:
    """Grade a drained eval job and write its report, manifest and graded rows.

    A job our validator served (jobs create --eval-set) is graded here, on this
    host's CPU, from the subnet bucket (R2_*); an order job (order-eval-*) by
    the admin service, which then sends its files home."""
    import json

    from reliquary.eval import operator
    from reliquary.eval.job_grading import JobNotGradable, grade_served_job
    from reliquary.eval.prompt_source import is_order_job_id

    try:
        if not is_order_job_id(job_id):
            if out is None:
                raise ValueError("--out is required: the grading is written here")
            answer = asyncio.run(grade_served_job(job_id, out=out,
                                                  allow_incomplete=allow_incomplete))
        else:
            answer = operator.grade_job(_admin_client(admin_url), job_id, out=out,
                                        eval_id=eval_id, allow_incomplete=allow_incomplete,
                                        poll_seconds=poll_seconds)
    except (ValueError, TimeoutError, operator.AdminError, JobNotGradable) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(answer, indent=1))


@eval_app.command("compare")
def eval_compare(
    a: str = typer.Argument(..., help="A graded directory (eval grade --out)"),
    b: str = typer.Argument(..., help="Another, on the same sets and conditions"),
    allow_ungraded: bool = typer.Option(False, "--allow-ungraded",
                                        help="Count ungraded rows as failures instead of refusing"),
) -> None:
    """pass@1 of two gradings and their difference, with a paired bootstrap interval."""
    import json

    from reliquary.eval.operator import compare_reports

    try:
        result = compare_reports(a, b, allow_ungraded=allow_ungraded)
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=1))


@eval_app.command("publish-set")
def eval_publish_set(directory: str = typer.Argument(..., help="A directory build-set wrote")) -> None:
    """Upload a set: its three files to the subnet bucket (R2_*); a platform
    preset's prompts.jsonl and set.json to the platform bucket too
    (RELIQUARY_PLATFORM_*), which makes it orderable. An operator's set
    (build-set --source) never goes there."""
    import json
    from pathlib import Path

    from reliquary.corpus.delivery import R2DeliverySink
    from reliquary.eval.storage import (
        SetConflict, SubnetEvalStore, is_operator_set, publish_set,
    )

    try:
        card = json.loads((Path(directory) / "set.json").read_text())
        platform = None if is_operator_set(card) else R2DeliverySink.from_environment()
        answer = asyncio.run(publish_set(directory, platform=platform,
                                         subnet=SubnetEvalStore()))
    except (ValueError, OSError, RuntimeError, SetConflict) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(answer, indent=1))


@eval_app.command("run")
def eval_run(
    platform: str = typer.Option(..., "--platform", help="The platform's base URL"),
    executor_id: str = typer.Option(..., "--executor-id"),
    work_dir: str = typer.Option("/opt/reliquary-eval", "--work-dir",
                                 help="Chunks and resume state; keep it across restarts"),
    chunk_problems: int = typer.Option(64, "--chunk-problems", min=1),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Claim one evaluation task and generate it with vLLM (token in
    RELIQUARY_EXECUTOR_TOKEN)."""
    import json

    from reliquary.eval.platform_client import LeaseLost
    from reliquary.eval.runner import run_evaluation

    setup_logging(log_level)
    try:
        result = run_evaluation(platform=platform, executor_id=executor_id, work_dir=work_dir,
                                chunk_problems=chunk_problems)
    except (ValueError, LeaseLost) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result) if result is not None else "no evaluation task to claim")


sandbox_app = typer.Typer(help="Signed-episode sandboxes (reliquary-sandbox machines)")
sandbox_machines_app = typer.Typer(help="The machine directory in R2")
sandbox_app.add_typer(sandbox_machines_app, name="machines")
app.add_typer(sandbox_app, name="sandbox")


def _sandbox_store_call(coroutine):
    from reliquary.infrastructure.sandbox_store import MachineConflict

    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return asyncio.run(coroutine)
    except (ValueError, MachineConflict) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ClientError as exc:
        # The error code only: a storage message may echo request details.
        code = exc.response.get("Error", {}).get("Code", "?")
        typer.echo(f"error: R2 refused the request ({code})", err=True)
        raise typer.Exit(code=1) from exc
    except BotoCoreError as exc:                # credentials, endpoint, connection
        typer.echo(f"error: R2 is unreachable ({type(exc).__name__})", err=True)
        raise typer.Exit(code=1) from exc


def _echo_machine(document) -> None:
    import json

    if document is None:
        typer.echo("error: no such machine", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(document, sort_keys=True))


@sandbox_machines_app.command("register")
def sandbox_machine_register(
    machine_id: str = typer.Option(..., "--machine-id"),
    address: str = typer.Option(..., "--address", help="scheme://host[:port] miners reach"),
    provider: str = typer.Option(..., "--provider"),
    capacity: int = typer.Option(..., "--capacity", help="Concurrent episodes"),
    key_id: str = typer.Option(..., "--key-id"),
    public_key: str = typer.Option(..., "--public-key", help="base64 Ed25519 public key"),
    valid_from: int = typer.Option(..., "--valid-from", help="unix seconds"),
) -> None:
    """Register a machine with its first signing key (create-only)."""
    from reliquary.infrastructure import sandbox_store

    document, _ = _sandbox_store_call(sandbox_store.register_machine(
        machine_id=machine_id, address=address, provider=provider, capacity=capacity,
        key_id=key_id, public_key_b64=public_key, valid_from=valid_from, now=_time.time()))
    _echo_machine(document)


@sandbox_machines_app.command("add-key")
def sandbox_machine_add_key(machine_id: str = typer.Option(..., "--machine-id"),
                            key_id: str = typer.Option(..., "--key-id"),
                            public_key: str = typer.Option(..., "--public-key"),
                            valid_from: int = typer.Option(..., "--valid-from")) -> None:
    """Add a rotated key (switch the machine to it when it has no live episode)."""
    from reliquary.infrastructure import sandbox_store

    _echo_machine(_sandbox_store_call(sandbox_store.add_machine_key(
        machine_id, key_id=key_id, public_key_b64=public_key, valid_from=valid_from)))


@sandbox_machines_app.command("end-key")
def sandbox_machine_end_key(
    machine_id: str = typer.Option(..., "--machine-id"),
    key_id: str = typer.Option(..., "--key-id"),
    valid_until: int = typer.Option(..., "--valid-until", help=(
        "unix seconds. Rotation: when the new key starts. COMPROMISE: the earliest time "
        "the compromise is suspected, even in the past, with --compromise")),
    compromise: bool = typer.Option(False, "--compromise", help=(
        "The key is compromised: end it at --valid-until minus the verifier's clock skew "
        "(CLOCK_SKEW_S, 30 s), since an open may precede its token by that much")),
) -> None:
    """End a key's validity; an end only ever moves earlier. On compromise, pass the
    suspected compromise time with --compromise: the end becomes that time minus
    CLOCK_SKEW_S, so no forged open stamped up to the skew before it verifies."""
    from reliquary.infrastructure import sandbox_store

    if compromise:
        try:
            valid_until = sandbox_store.compromise_valid_until(valid_until)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    _echo_machine(_sandbox_store_call(sandbox_store.end_machine_key(
        machine_id, key_id=key_id, valid_until=valid_until)))


@sandbox_machines_app.command("status")
def sandbox_machine_status(machine_id: str = typer.Option(..., "--machine-id"),
                           status: str = typer.Option(..., "--status",
                                                      help="active, draining or revoked"),
                           reason: str = typer.Option(None, "--reason")) -> None:
    from reliquary.infrastructure import sandbox_store

    _echo_machine(_sandbox_store_call(sandbox_store.set_machine_status(machine_id, status, reason=reason)))


@sandbox_machines_app.command("list")
def sandbox_machine_list() -> None:
    from reliquary.infrastructure import sandbox_store

    for document in _sandbox_store_call(sandbox_store.list_machines()):
        _echo_machine(document)


corpus_app = typer.Typer(name="corpus", help="Mine a corpus generation task")
app.add_typer(corpus_app)


def _restart_with_served_contract(validator_url: str, job_id: str | None = None) -> None:
    """Take the task's contract from the validator and restart with it: the
    active profile is fixed when this process imports it. With ``job_id``,
    that job's own task contract on a validator serving several."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import httpx

    from reliquary.miner.corpus_miner import (
        CorpusContractError,
        CorpusJobSelectionError,
        HttpCorpusClient,
        save_served_contract,
    )
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    client = HttpCorpusClient(httpx.Client(base_url=validator_url, timeout=60.0), job_id=job_id)
    try:
        raw = client.job()
        contract = client.contract()
    except CorpusJobSelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    job = SimpleNamespace(job_id=raw.get("job_id"), checkpoint_repo=raw.get("checkpoint_repo"),
                          checkpoint_revision=raw.get("checkpoint_revision"))
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "reliquary" / "corpus"
    try:
        path = save_served_contract(contract, job, cache)
    except CorpusContractError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    typer.echo(f"using the contract served by {validator_url}: {path}")
    os.environ[TASK_CONTRACT_ENV_VAR] = str(path)
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


ledgers_app = typer.Typer(
    name="ledgers", help="Migrate, verify or downgrade a corpus job's ledgers"
)
corpus_app.add_typer(ledgers_app)


def _run_on_ledgers(job_id: str, action):
    """Run ``action(store, job)`` against the job's bucket, turning a missing
    job or a corrupt ledger into a named exit rather than a traceback."""
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.validator.corpus_service import LedgerSnapshotError

    async def _run():
        store = BucketJobStore()
        job, _ = await store.read_job(job_id)
        if job is None:
            raise typer.BadParameter(f"no job {job_id!r}")
        return await action(store, job)

    try:
        return asyncio.run(_run())
    except LedgerSnapshotError as exc:
        typer.echo(f"error: ledgers of {job_id!r} are corrupt: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@ledgers_app.command("migrate")
def ledgers_migrate(job_id: str = typer.Option(..., "--job")) -> None:
    """Rewrite a v1 ledger as v2 (backup first). Validators also do this at
    startup; running it by hand is for a rehearsal or a stopped fleet."""
    from reliquary.validator.corpus_service import ensure_ledgers_v2

    typer.echo(f"{job_id}: {_run_on_ledgers(job_id, ensure_ledgers_v2)}")


@ledgers_app.command("verify")
def ledgers_verify(job_id: str = typer.Option(..., "--job")) -> None:
    """Load every segment the ledger names, check them, and print sizes.
    Exits 1 on a missing, altered or overlapping segment, or a seen count the
    filled slots do not imply."""
    import json

    from reliquary.validator.corpus_service import verify_ledgers

    report = _run_on_ledgers(job_id, verify_ledgers)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if report["problems"]:
        raise typer.Exit(code=1)


@ledgers_app.command("downgrade")
def ledgers_downgrade(job_id: str = typer.Option(..., "--job")) -> None:
    """Rewrite a v2 ledger as v1 so a pre-v2 image can serve the job. Stop
    every validator serving the job first: one still running would migrate it
    straight back."""
    from reliquary.validator.corpus_service import downgrade_ledgers_v1

    typer.echo(f"{job_id}: {_run_on_ledgers(job_id, downgrade_ledgers_v1)}")


@corpus_app.command("audit-executor")
def corpus_audit_executor(
    control: str = typer.Option(..., "--control", help="The corpus control's HTTPS origin"),
    executor_id: str = typer.Option(..., "--executor-id"),
    model_id: str = typer.Option(
        None, "--model-id", help="Defaults to the model this executor is registered for"),
    model_revision: str = typer.Option(None, "--model-revision"),
    eval_control: bool = typer.Option(
        False, "--eval", help="Score for the eval control (its /corpus/internal/eval-audit routes)"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Score corpus audit leases on this GPU. The only secret is the executor
    token, in RELIQUARY_EXECUTOR_TOKEN; the model comes from the public HF repo."""
    from reliquary.validator.corpus_audit_executor import (
        EVAL_AUDIT_PREFIX, TOKEN_ENV, run_audit_executor,
    )

    setup_logging(log_level)
    if not os.environ.get(TOKEN_ENV, "").strip():
        typer.echo(f"error: {TOKEN_ENV} is not set", err=True)
        raise typer.Exit(code=1)
    # The corpus control's call is left exactly as it was.
    route = {"prefix": EVAL_AUDIT_PREFIX} if eval_control else {}
    run_audit_executor(control_url=control, executor_id=executor_id, model_id=model_id,
                       model_revision=model_revision, **route)


@corpus_app.command("register-grade-executor")
def corpus_register_grade_executor(
    executor_id: str = typer.Option(..., "--executor-id"),
    env_version: str = typer.Option(
        ..., "--env-version", help="reliquary-environments commit the job pins"),
    provider_id: str = typer.Option(
        ..., "--provider-id",
        help="Who runs the box (e.g. hetzner); agreement counts distinct providers only"),
    env_package: str = typer.Option("reliquary-swe", "--env-package"),
    days: float = typer.Option(30.0, "--days"),
) -> None:
    """Register a grade executor in the bucket and print its token once."""
    import hashlib
    import json
    import secrets
    import time

    from reliquary.infrastructure import corpus_executor_store

    token = secrets.token_urlsafe(32)
    now = time.time()
    document, created = asyncio.run(corpus_executor_store.register_executor(
        executor_id=executor_id, token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        model_id=env_package, model_revision=env_version, expires_at=now + days * 86400.0,
        now=now, provider_id=provider_id, scope="grade"))
    typer.echo(json.dumps({"executor_id": document["executor_id"], "created": created,
                           "token": token if created else None}))


@corpus_app.command("grade-executor")
def corpus_grade_executor(
    control_url: str = typer.Option(..., "--control-url"),
    executor_id: str = typer.Option(..., "--executor-id"),
    concurrency: int = typer.Option(4, "--concurrency", help="Items graded or replayed at once"),
    cpus: float = typer.Option(2.0, "--cpus", help="CPUs per box"),
    memory_gb: float = typer.Option(
        6.0, "--memory-gb", help="Memory per box, no swap; concurrency x this must fit the host"),
    pids_limit: int = typer.Option(1024, "--pids-limit", help="Processes per box"),
    disk_gb: float = typer.Option(
        10.0, "--disk-gb",
        help="Writable layer per box: the Docker daemon's default overlay2.size (xfs, pquota), "
             "checked at start and in every box"),
    disk_probe_image: str = typer.Option(
        "alpine:3.22", "--disk-probe-image", help="Image of the start-up disk-limit probe box"),
    allow_non_xfs: bool = typer.Option(
        False, "--allow-non-xfs",
        help="TESTS ONLY: start although Docker's storage is not on xfs (replays then disagree "
             "with honest miners on directory order) and box disks are not bounded"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Grade and replay agentic trajectories for a corpus control. The only
    secret is the executor token, in RELIQUARY_EXECUTOR_TOKEN; boxes come from
    public images, each under --cpus/--memory-gb/--pids-limit. Run one executor
    per Docker host: it removes leftover boxes of its own at start."""
    from reliquary.validator import corpus_grade_executor as grade
    from reliquary.validator.agentic_replay import BoxLimits

    setup_logging(log_level)
    if not os.environ.get(grade.TOKEN_ENV, "").strip():
        typer.echo(f"error: {grade.TOKEN_ENV} is not set", err=True)
        raise typer.Exit(code=1)
    try:
        limits = BoxLimits(cpu=cpus, memory_gb=memory_gb, pids=pids_limit,
                           disk_gb=None if allow_non_xfs else disk_gb)
        BoxLimits(disk_gb=disk_gb)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    refusal = grade.docker_storage_refusal()
    if refusal and not allow_non_xfs:
        typer.echo(f"error: {refusal}. Put Docker's data root on xfs "
                   "(--allow-non-xfs is for tests only)", err=True)
        raise typer.Exit(code=1)
    if refusal:
        typer.echo(f"warning: --allow-non-xfs (tests only): {refusal}", err=True)
    if allow_non_xfs:
        typer.echo("warning: --allow-non-xfs (tests only): box disks are not checked", err=True)
    else:
        refusal = grade.docker_disk_refusal(disk_gb, image=disk_probe_image)
        if refusal:
            typer.echo(f"error: {refusal} (--allow-non-xfs is for tests only)", err=True)
            raise typer.Exit(code=1)
    grade.run_grade_executor(control_url=control_url, executor_id=executor_id,
                             concurrency=concurrency, limits=limits)


@corpus_app.command("order-control")
def corpus_order_control(
    netuid: int = typer.Option(81, "--netuid"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8791, "--port"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Serve every order job, eval (${RELIQUARY_ADMIN_TASK_PREFIX}eval-) and
    generation (${RELIQUARY_ADMIN_TASK_PREFIX}gen-), order-eval- and order-gen-
    by default, whatever its model, with no GPU: tokenizers on CPU, audits by
    executor pairs on distinct providers. Route ^/corpus/jobs/<prefix>(eval|gen)-
    and ^/corpus/internal/eval-audit/ here; the corpus control keeps everything
    else. `eval-control` is the same command."""
    from reliquary.validator import eval_control

    setup_logging(log_level)
    asyncio.run(eval_control.run_order_control(netuid=netuid, http_host=host, http_port=port))


@corpus_app.command("generation-control")
def corpus_generation_control(
    task_ids: list[str] = typer.Option([], "--task-id"),
    checkpoint_dir: str = typer.Option(..., "--checkpoint-dir"),
    gpu_run_dir: str = typer.Option(..., "--gpu-run-dir"),
    netuid: int = typer.Option(81, "--netuid"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8792, "--port"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Serve operator generation orders with a shared, already loaded GPU scorer.

    Owns only <admin prefix>gen-ops- jobs on the configured pinned model. Supply
    the checkpoint and existing GPU socket directories; optional initial task
    ids must be distinct. It may start empty and hot-add actual registry tasks.
    The original corpus supervisor, judges and tasks retain their ownership.
    Route only these generation jobs to this control.
    """
    from reliquary.validator.generation_control import run_generation_control

    setup_logging(log_level)
    asyncio.run(run_generation_control(task_ids=task_ids, checkpoint_dir=checkpoint_dir,
                                      gpu_run_dir=gpu_run_dir, netuid=netuid,
                                      http_host=host, http_port=port))


# The command's first name, kept for existing deployments.
corpus_app.command("eval-control", help="Alias of `order-control`.")(corpus_order_control)


@corpus_app.command("order-control-check")
def corpus_order_control_check() -> None:
    """Check the order control's runtime (drand, hub, tokenizers, and every
    order source's package against the catalog): JSON, exit 1 if anything is
    missing. Run at image build."""
    import json as _json

    from reliquary.validator.eval_control import order_control_runtime_check

    report = order_control_runtime_check()
    typer.echo(_json.dumps(report, indent=1, sort_keys=True))
    if not report["ok"]:
        raise typer.Exit(code=1)


@corpus_app.command("order-nginx")
def corpus_order_nginx(
    port: int = typer.Option(8791, "--port", help="The order control's local port"),
) -> None:
    """Print the nginx locations for the order control, built from
    RELIQUARY_ADMIN_TASK_PREFIX: the one source of the routing regex."""
    from reliquary.eval.prompt_source import order_routes_nginx

    typer.echo(order_routes_nginx(port=port), nl=False)


@corpus_app.command("qualify")
def corpus_qualify(
    model: str = typer.Option(..., "--model", help="repo@revision of the model to qualify"),
    control: str = typer.Option(..., "--control", help="The eval control's HTTPS origin"),
    executor_id: str = typer.Option(..., "--executor-id"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Run one qualification of MODEL leased by the eval control: decode the
    eval prompts with vLLM, verify them with the HF prefill, post the measured
    TOPLOC band. The token is in RELIQUARY_EXECUTOR_TOKEN."""
    import json

    from reliquary.eval.qualify_executor import run_qualify

    setup_logging(log_level)
    if not os.environ.get("RELIQUARY_EXECUTOR_TOKEN", "").strip():
        typer.echo("error: RELIQUARY_EXECUTOR_TOKEN is not set", err=True)
        raise typer.Exit(code=1)
    try:
        answer = run_qualify(control_url=control, executor_id=executor_id, model=model)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(answer))


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
    job_id: str = typer.Option(
        None, "--job-id",
        help="The job to mine on a validator serving several; one process mines one job",
    ),
) -> None:
    """Generate for the corpus job the validator serves, and submit it."""
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    if TASK_CONTRACT_ENV_VAR not in os.environ:
        _restart_with_served_contract(validator_url, job_id)
    import bittensor as bt
    import httpx
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.corpus.job import parse_job
    from reliquary.miner.corpus_miner import (
        CorpusJobSelectionError,
        CorpusMinerHalted,
        HttpCorpusClient,
        VllmGenerator,
        mine_steps,
    )
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import sign_corpus_skip, sign_corpus_submission
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
    client = HttpCorpusClient(http, job_id=job_id)
    try:
        job = parse_job(client.job())
    except CorpusJobSelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if job_id is None:
        others = [served for served in client.served_jobs() if served != job.job_id]
        if others:
            typer.echo(f"mining job {job.job_id}, the validator's default; it also serves "
                       f"{others}: pass --job-id to mine one of those", err=True)
    directory = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
    if checkpoint_fingerprint(directory) != job.checkpoint_sha256:
        typer.echo("error: the downloaded checkpoint does not match the job's fingerprint", err=True)
        raise typer.Exit(code=4)
    tokenizer = load_tokenizer(directory)

    def encode(text):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    from reliquary.eval.prompt_source import (
        is_eval_source, parse_eval_source, register_eval_prompts,
    )
    from reliquary.miner.corpus_miner import submits_scoped

    # Only a job whose manifest says so (order jobs) leaves /corpus/submit.
    client.scoped_submit = submits_scoped(job)
    if is_eval_source(job.prompt_source):
        # An eval job's prompts come from the control serving it, checked
        # against the sha256 its manifest names.
        try:
            register_eval_prompts(parse_eval_source(job.prompt_source), client.eval_prompts())
        except (ValueError, CorpusJobSelectionError) as exc:
            typer.echo(f"error: the eval job's prompts are unusable: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    renderer = renderer_for_job(job, encode, tokenizer=tokenizer)
    prompts = prompt_job_for_spec(job)
    try:
        counts = mine_steps(
            job=job, hotkey=wallet.hotkey.ss58_address, client=client,
            generator=VllmGenerator(directory, job.sampling, proof, job.eos_token_id,
                                    gpu_memory_utilization=gpu_memory_utilization),
            tokenizer=tokenizer, render=lambda i: renderer.initial_text(prompts.task_for(i)),
            sign=lambda body: sign_corpus_submission(wallet, body),
            # Full prompts are skipped, not generated for; an older validator
            # without the routes is mined as before.
            sign_skip=lambda body: sign_corpus_skip(wallet, body),
            max_steps=max_steps or None,
        )
    except CorpusMinerHalted as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(dict(exc.counts))
        raise typer.Exit(code=1) from exc
    typer.echo(counts)


@corpus_app.command("mine-agentic")
def corpus_mine_agentic(
    validator_url: str = typer.Option(..., "--validator-url"),
    job_id: str = typer.Option(..., "--job-id", help="The episode job to mine"),
    wallet_name: str = typer.Option("default"),
    hotkey: str = typer.Option("default"),
    wallet_path: str = typer.Option(os.getenv("BT_WALLET_PATH", "")),
    concurrency: int = typer.Option(
        8, "--concurrency", help="Episodes at once (spec section 9: 8 to 11 on one H100)"),
    episodes: int = typer.Option(0, "--episodes", help="0 = until the job completes"),
    port: int = typer.Option(8011, "--port", help="Loopback port of the generate endpoint"),
    gpu_memory_utilization: float = typer.Option(None, "--gpu-memory-utilization"),
    max_num_seqs: int = typer.Option(
        16, "--max-num-seqs",
        help="vLLM's concurrent sequences; lower it if turns fail as preempted (unprovable)"),
    validator_hotkey: str = typer.Option(
        None, "--validator-hotkey",
        help="The validator's ss58 hotkey; signed-sandbox jobs sign their session requests "
             "for it"),
    max_live_per_job: int = typer.Option(
        4, "--max-live-per-job", min=1,
        help="Signed-sandbox jobs: live sessions at once for this hotkey on the job (match "
             "the validator's per-job cap, 4 by default)"),
) -> None:
    """Mine an agentic (episode) corpus job: verifiers + reliquary-swe episodes
    against a local vLLM with per-turn proofs. Replay jobs need Docker and the
    job's pinned reliquary-swe, verifiers and renderers installed. Signed-sandbox
    jobs need no Docker: every tool call runs on the validator's sandbox machines
    (they need `--validator-hotkey` and the reliquary[sandbox-miner] extra)."""
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    if TASK_CONTRACT_ENV_VAR not in os.environ:
        _restart_with_served_contract(validator_url, job_id)
    import bittensor as bt
    import httpx
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.corpus.job import parse_job
    from reliquary.environment.agentic_swe import episode_support_refusal
    from reliquary.miner.agentic_miner import Identity, run_agentic_miner
    from reliquary.miner.corpus_miner import (
        CorpusJobSelectionError,
        HttpCorpusClient,
        submits_scoped,
    )
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import sign_corpus_submission
    from reliquary.shared.modeling import load_tokenizer

    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    if proof is None:
        typer.echo("error: the active contract declares no toploc proof", err=True)
        raise typer.Exit(code=4)
    wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
    if wallet_path:
        wallet_kwargs["path"] = wallet_path
    wallet = bt.Wallet(**wallet_kwargs)
    client = HttpCorpusClient(httpx.Client(base_url=validator_url, timeout=300.0), job_id=job_id)
    try:
        job = parse_job(client.job())
    except CorpusJobSelectionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if job.episode is None:
        typer.echo(f"error: job {job.job_id!r} is not an episode job; use `corpus mine`", err=True)
        raise typer.Exit(code=2)
    client.scoped_submit = submits_scoped(job)
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.environment.agentic_swe import sandbox_support_refusal

    signed = is_signed_sandbox(job)
    refusal = episode_support_refusal(job.episode, need_verifiers=True) or (
        sandbox_support_refusal(job.episode, need_bridge=True) if signed else None)
    if refusal:
        typer.echo(f"error: {refusal}", err=True)
        raise typer.Exit(code=4)
    sessions = None
    if signed:
        from reliquary.miner.signed_episode import HttpSandboxSessions

        if not validator_hotkey:
            typer.echo("error: a signed-sandbox job needs --validator-hotkey (the validator's "
                       "ss58 hotkey: session requests are signed for it)", err=True)
            raise typer.Exit(code=2)
        try:
            sessions = HttpSandboxSessions(httpx.Client(base_url=validator_url, timeout=60.0),
                                           validator_hotkey=validator_hotkey)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    else:
        from reliquary.miner.agentic_miner import docker_storage_warning

        warning = docker_storage_warning()
        if warning:
            typer.echo(warning, err=True)
    directory = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
    if checkpoint_fingerprint(directory) != job.checkpoint_sha256:
        typer.echo("error: the downloaded checkpoint does not match the job's fingerprint", err=True)
        raise typer.Exit(code=4)
    identity = Identity(hotkey=wallet.hotkey.ss58_address,
                        sign=lambda body: sign_corpus_submission(wallet, body),
                        episodes=episodes or None,
                        sign_binding=lambda binding: wallet.hotkey.sign(binding).hex())
    counts = asyncio.run(run_agentic_miner(
        job=job, checkpoint_dir=directory, proof=proof, tokenizer=load_tokenizer(directory),
        identities=[identity], client=client, concurrency=concurrency, port=port,
        gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
        sessions=sessions, max_live_per_job=max_live_per_job))
    typer.echo({hotkey_: dict(c) for hotkey_, c in counts.items()})
    halted = [hotkey_ for hotkey_, c in counts.items()
              if c.get("halted") or c.get("identity_crashed")]
    if halted:
        hint = (" (a bad_signature: is --validator-hotkey this validator's hotkey?)"
                if any(c.get("session_refused:bad_signature") for c in counts.values()) else "")
        typer.echo(f"error: mining stopped for {', '.join(h[:8] for h in halted)}{hint}",
                   err=True)
        raise typer.Exit(code=3)


@corpus_app.command("status")
def corpus_status(
    validator_url: str = typer.Option(..., "--validator-url"),
    job_id: str = typer.Option(
        None, "--job-id", help="The job; omit for the validator's default job",
    ),
    hotkey: str = typer.Option(
        None, "--hotkey", help="SS58 address; omit to use the wallet's hotkey",
    ),
    wallet_name: str = typer.Option("default", "--wallet-name"),
    wallet_hotkey: str = typer.Option("default", "--wallet-hotkey"),
    wallet_path: str = typer.Option(os.getenv("BT_WALLET_PATH", ""), "--wallet-path"),
    as_json: bool = typer.Option(False, "--json", help="Print the validator's JSON as is"),
) -> None:
    """One hotkey's audit state, counts, recent failures and pay on a corpus job."""
    import json

    from reliquary.miner import corpus_status as status_client

    if hotkey is None:
        import bittensor as bt

        wallet_kwargs = {"name": wallet_name, "hotkey": wallet_hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        wallet = bt.Wallet(**wallet_kwargs)
        # The public half only: no password asked for a read.
        hotkey = (getattr(wallet, "hotkeypub", None) or wallet.hotkey).ss58_address
    try:
        status = status_client.fetch_miner_status(validator_url, hotkey, job_id)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(status, indent=2) if as_json else status_client.format_miner_status(status))


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


def _corpus_hot_registry_reader():
    """The registry reader that makes a corpus validator's job set hot, or None.

    Off unless ``RELIQUARY_CORPUS_HOT_JOBS=1``: a hot validator starts serving
    (and paying) any active corpus entry on its model, which an operator opts into.
    """
    if not _env_flag("RELIQUARY_CORPUS_HOT_JOBS"):
        return None
    from reliquary.infrastructure.task_registry_store import read_registry

    async def entries():
        found, _ = await read_registry()
        return found

    return entries


def _corpus_remote_audit_options() -> dict:
    """Remote audit executors, on with ``RELIQUARY_CORPUS_REMOTE_AUDIT=1``;
    ``RELIQUARY_CORPUS_RECHECK_FRACTION`` (default 0.05) is the share of their
    results this GPU recomputes."""
    if not _env_flag("RELIQUARY_CORPUS_REMOTE_AUDIT"):
        return {}
    fraction = float(os.getenv("RELIQUARY_CORPUS_RECHECK_FRACTION", "0.05"))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("RELIQUARY_CORPUS_RECHECK_FRACTION must be in (0, 1]")
    return {"remote_audit": True, "recheck_fraction": fraction}


async def _run_corpus(*, jobs, wallet, netuid, signer_client, http_host, http_port,
                      set_weights) -> None:
    """The corpus validator: one process, or with ``RELIQUARY_CORPUS_SPLIT=1``
    a supervisor over front, judge and GPU processes
    (``RELIQUARY_CORPUS_SPLIT_JUDGES`` says which jobs leave the front)."""
    read_registry = _corpus_hot_registry_reader()
    remote = _corpus_remote_audit_options()
    # Intake and grading only, no model and no audit (the end-to-end run).
    intake_only = _env_flag("RELIQUARY_CORPUS_INTAKE_ONLY")
    if intake_only and _env_flag("RELIQUARY_CORPUS_SPLIT"):
        raise RuntimeError("RELIQUARY_CORPUS_INTAKE_ONLY serves no audit: the split validator "
                           "is not intake-only; unset RELIQUARY_CORPUS_SPLIT")
    if _env_flag("RELIQUARY_CORPUS_SPLIT"):
        from reliquary.validator.corpus_split import run_corpus_split

        await run_corpus_split(
            served=jobs, netuid=netuid, http_host=http_host, http_port=http_port,
            set_weights=set_weights, hot=read_registry is not None,
            remote_audit=bool(remote.get("remote_audit")),
        )
        return
    from reliquary.validator.corpus_validator import run_corpus_validator

    if len(jobs) == 1:
        (entry, cap), = jobs
        await run_corpus_validator(
            entry=entry, cap=cap, wallet=wallet, netuid=netuid, signer_client=signer_client,
            http_host=http_host, http_port=http_port, set_weights=set_weights,
            read_registry=read_registry, intake_only=intake_only, **remote,
        )
        return
    await run_corpus_validator(
        jobs=jobs, wallet=wallet, netuid=netuid, signer_client=signer_client,
        http_host=http_host, http_port=http_port, set_weights=set_weights,
        read_registry=read_registry, intake_only=intake_only, **remote,
    )


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


@app.command("mine-episodes")
def mine_episodes(
    episode_envs: str = typer.Option(
        ..., "--episode-envs", help="Comma-separated signed-episode environments to mine"),
    validator_url: str = typer.Option(..., "--validator-url", help="The RL validator's URL"),
    validator_hotkey: str = typer.Option(
        ..., "--validator-hotkey",
        help="The RL validator's ss58 hotkey: precommits and session opens are signed for it"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    wallet_path: str = typer.Option(os.getenv("BT_WALLET_PATH", ""), help="Optional wallet base path"),
    generate_port: int = typer.Option(8012, "--generate-port", help="Loopback port of the generate endpoint"),
    checkpoint_dir: str = typer.Option(
        "", "--checkpoint-dir",
        help="Where announced checkpoints are downloaded (huggingface_hub's cache by default)"),
    max_live: int = typer.Option(
        0, "--max-live", min=0, help="Live sessions per group (0 = every seed of the pool at once)"),
    groups_in_flight: int = typer.Option(
        2, "--groups-in-flight", min=1, max=2,
        help="Episode groups mined at once (at most 2: the validator's per-operator cap)"),
    harness_env: list[str] = typer.Option(
        [], "--harness-env", help="KEY=VALUE passed to the episode harness (repeatable)"),
    gpu_memory_utilization: float = typer.Option(
        0.6, "--gpu-memory-utilization",
        help="vLLM's share of the GPU; the HF proof model takes the rest of the same GPU"),
    max_num_seqs: int = typer.Option(16, "--max-num-seqs", help="vLLM's concurrent sequences"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    """Mine signed-episode groups on the RL validator: per window, precommit a task, play its public seed
    pool against a local forced-draw vLLM (sandbox tool calls on the validator's machines), and submit
    one proved group. Needs the reliquary[sandbox-miner] extra and the order's pinned env package. The
    legacy `mine` never mines these environments."""
    from reliquary.miner.episode_mining import (
        EpisodeMinerConfig,
        parse_episode_environments,
        parse_harness_env,
    )

    setup_logging(log_level)
    try:
        config = EpisodeMinerConfig(
            environments=parse_episode_environments(episode_envs), validator_url=validator_url.rstrip("/"),
            validator_hotkey=validator_hotkey, generate_port=generate_port,
            checkpoint_dir=checkpoint_dir or None, max_live=max_live or None, groups_in_flight=groups_in_flight,
            harness_env=parse_harness_env(harness_env) or None,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    import bittensor as bt

    from reliquary.miner import episode_mining

    wallet_kwargs = {"name": wallet_name, "hotkey": hotkey}
    if wallet_path:
        wallet_kwargs["path"] = wallet_path
    wallet = bt.Wallet(**wallet_kwargs)
    try:
        asyncio.run(episode_mining.run_episode_miner(config=config, wallet=wallet))
    except ValueError as exc:   # a refusal to start: an env, its package or the engine caps
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=4) from exc


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
    # Before the route serves, so its first write is not the one that pays
    # for sealing a v1 seen set.
    from reliquary.validator.corpus_service import migrate_ledgers_at_startup, recover_pending_records

    seen_index = await migrate_ledgers_at_startup(store, job)
    await recover_pending_records(store, None, job)

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
        seen_index=seen_index,
        job=job,
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
                TASK_IDS,
            )
            from reliquary.infrastructure.task_registry_store import read_registry
            from reliquary.validator.task_config import (
                TaskConfigError,
                legacy_registry_fallback,
                legacy_task_config,
                resolve_corpus_task_configs,
                resolve_task_config,
            )

            if len(TASK_IDS) > 1:
                # Several ids: one corpus validator, one loaded model, one job
                # per id. Anything else among them refuses, like any other
                # undeclared task, before the GPU is touched.
                try:
                    registry_entries, _ = await read_task_registry_with_retry(
                        read_registry
                    )
                    corpus_configs = resolve_corpus_task_configs(
                        registry_entries,
                        TASK_IDS,
                        profile_id=PROTOCOL_PROFILE_ID,
                        generation_contract=PROTOCOL_GENERATION_CONTRACT,
                    )
                except TaskConfigError as exc:
                    logger.critical(
                        "%s; declare them with `reliquary jobs create` before "
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
                try:
                    await _run_corpus(
                        jobs=[(c.entry, c.emission_cap) for c in corpus_configs],
                        wallet=wallet, netuid=netuid, signer_client=signer_client,
                        http_host=http_host, http_port=http_port,
                        set_weights=set_weights,
                    )
                except (RuntimeError, ValueError) as exc:
                    logger.critical("%s; fix the declaration before starting this validator", exc)
                    raise typer.Exit(code=4) from exc
                return

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
                try:
                    await _run_corpus(
                        jobs=[(task_config.entry, task_config.emission_cap)],
                        wallet=wallet, netuid=netuid, signer_client=signer_client,
                        http_host=http_host, http_port=http_port,
                        set_weights=set_weights,
                    )
                except (RuntimeError, ValueError) as exc:
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
                service_contract=task_config.service_contract,
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
