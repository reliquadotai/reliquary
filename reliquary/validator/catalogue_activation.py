"""The activation checks of an env added to a running run. Pure: the validator calls ``activation_problem`` in a
worker thread, first when it sees the request (it builds the env then, never at the window boundary) and again
at the boundary, and refuses the request with the first problem found:

1. the package is on the owner's allowlist (``episode-packages.allow`` in the service folder: one
   ``name==version`` per line), so a third-party env never runs in the validator process unreviewed;
2. its distribution name is plain (letters and digits joined by single ``-`` or ``_``, no ``.``), so its import
   module (the name with ``-`` -> ``_``, lowercased) is a plain identifier, and pip's normalised name and the
   module name agree; checked before anything is imported;
3. every per-env budget it declares is at most the order's: ``max_episode_tokens``, ``max_turns``,
   ``max_tokens_per_turn`` and each session budget (the values the trainer and the sandbox fleet were sized
   for); checked before anything is imported;
4. the operator's attestation is a conformance GATE report of exactly this env (the format the sandbox's
   conformance ``--report`` writes): gate and passed true, no problem, real boxes, a full sample, the same env
   package identity, taskset entry, split and options;
5. the env is installed on this validator: built through the bridge, its identity, image lock, manifest digest
   and row count are the entry's, and its import module is its distribution's and no active env's nor any order
   env's (two distributions shipping one module overwrite each other in one venv);
6. installing it moved no active bridged env: each one's package on disk still has its pinned identity (the
   operator installs with ``pip install --no-deps`` and runs ``pip check``; this catches what slipped through);
7. at least ``MIN_ACTIVATION_MACHINES`` fresh active machines serve it at that identity, with its options and its
   image lock, with at least ``min(images, ACTIVATION_MIN_IMAGES)`` locked images present, and at least
   ``min(rows, ACTIVATION_MIN_SERVABLE_TASKS)`` tasks of its RL split are servable (their image is held by a
   serving machine; at most ``ACTIVATION_SCAN_MAX`` tasks are resolved to count them, once per manifest and set
   of images served: ``memo``).

A shortfall in 7 is final only on a complete fresh fleet view (``Fleet.view_complete``, the precommit route's
rule): while the directory is stale, no machine was heard, or a machine whose last report serves the env is
stale or drained, the answer is a ``NotYet`` problem (a ``str``): the request waits, it is not refused."""
from __future__ import annotations

import keyword
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CONFORMANCE_GATE_SAMPLE = 32
MIN_ACTIVATION_MACHINES = 1
ACTIVATION_MIN_IMAGES = 32
ACTIVATION_MIN_SERVABLE_TASKS = 256
ACTIVATION_SCAN_MAX = 20_000
ADDED_PACKAGES_FILE = "episode-packages.allow"
MAX_ALLOW_FILE_BYTES = 64 * 1024
EPISODE_LIMIT_FIELDS = ("max_episode_tokens", "max_turns", "max_tokens_per_turn")
_PLAIN_DISTRIBUTION = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\Z")
_VERIFIERS_TASKSETS = "verifiers.v1.tasksets."


class NotYet(str):
    """A problem the fleet's current view cannot settle (stale or incomplete): the request waits for a complete
    fresh view, it is never refused for it."""


def allowed_packages(folder: str | Path) -> frozenset[str]:
    """The owner-reviewed ``name==version`` packages of ``folder/episode-packages.allow`` (empty when absent,
    unsafe or too large: then nothing can be added)."""
    path = Path(folder) / ADDED_PACKAGES_FILE
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ALLOW_FILE_BYTES:
            return frozenset()
        if os.name == "posix" and path.stat().st_mode & 0o022:
            return frozenset()                                  # writable by others: not the owner's
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return frozenset()
    return frozenset(line.strip() for line in lines if line.strip() and not line.strip().startswith("#"))


def package_problem(added: Mapping[str, Any], allowed: frozenset[str]) -> str | None:
    package = added["catalogue"]["package"]
    if package not in allowed:
        return f"{package} is not on the owner's allowlist ({ADDED_PACKAGES_FILE} in the service folder)"
    return None


def _module_of(env_package: str) -> str:
    """The import module of a pinned package (its distribution name with ``-`` -> ``_``, lowercased)."""
    return str(env_package).split("==", 1)[0].replace("-", "_").lower()


def _normal_module(env_package: str) -> str:
    """The module a package's distribution claims, as pip compares names (PEP 503, then ``-`` -> ``_``): equal for
    two spellings pip would install over each other."""
    from reliquary.protocol.service_catalogue import distribution_of

    return distribution_of(env_package).replace("-", "_")


def name_problem(added: Mapping[str, Any]) -> str | None:
    """The added package's distribution name must be plain, so its import module is a plain identifier that pip's
    normalised name (PEP 503) maps one to one: no ``.`` (a dotted module), no run of separators."""
    for what, package in (("catalogue.package", added["catalogue"]["package"]),
                          ("episode.env_package", added["environment"]["episode"]["env_package"])):
        name = str(package).split("==", 1)[0]
        module = _module_of(package)
        if (not _PLAIN_DISTRIBUTION.fullmatch(name) or not module.isidentifier() or not module.isascii()
                or keyword.iskeyword(module)):
            return (f"its {what} distribution name {name!r} is not plain (letters and digits joined by single "
                    f"'-' or '_', no '.'): its import module must be a plain identifier")
    return None


def budget_problem(added: Mapping[str, Any], order: Any) -> str | None:
    """Every per-env budget of the added env at most the order's largest (``order``: the order's contract; a view
    is bounded by its order's envs, never by an env added earlier)."""
    from reliquary.protocol.service_contract import EPISODE_BUDGET_FIELDS

    episode = added["environment"]["episode"]
    bounds = [entry["episode"] for entry in order.order_environments.values()
              if isinstance(entry, Mapping) and "episode" in entry]
    if not bounds:
        return "the order has no episode environment to bound an added one"
    for field in EPISODE_LIMIT_FIELDS:
        limit, value = max(int(bound[field]) for bound in bounds), episode.get(field)
        if type(value) is not int or value > limit:
            return f"episode.{field} {value!r} is above the order's {limit} (the trainer was sized for the order)"
    budgets = episode.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != set(EPISODE_BUDGET_FIELDS):
        extra = sorted(set(budgets or ()) ^ set(EPISODE_BUDGET_FIELDS)) if isinstance(budgets, Mapping) else []
        return f"episode.budgets must be exactly {list(EPISODE_BUDGET_FIELDS)} (differs on {extra})"
    for field in EPISODE_BUDGET_FIELDS:
        limit, value = max(int(bound["budgets"][field]) for bound in bounds), budgets[field]
        if type(value) is not int or value > limit:
            return (f"episode.budgets.{field} {value!r} is above the order's {limit} (the sandbox fleet was sized "
                    f"for the order)")
    return None


def _options_digest(options: Any) -> str | None:
    from reliquary_sandbox_verifiers.loading import options_sha256

    if not isinstance(options, Mapping):
        return None
    try:
        return options_sha256(dict(options))
    except (TypeError, ValueError):
        return None


def attestation_problem(attestation: Any, added: Mapping[str, Any]) -> str | None:
    if not isinstance(attestation, Mapping):
        return "the attestation is not a JSON object"
    episode, catalogue = added["environment"]["episode"], added["catalogue"]
    sample = attestation.get("sample_size")
    digest = _options_digest(catalogue["options"])
    checks = (
        (attestation.get("gate") is True, "its conformance run is not a gate"),
        (attestation.get("passed") is True, "its conformance run did not pass"),
        (attestation.get("problems") == [], "its conformance run reports problems"),
        (attestation.get("fake") is False, "its conformance run used fake boxes"),
        (type(sample) is int and sample >= CONFORMANCE_GATE_SAMPLE,
         f"its sample is below {CONFORMANCE_GATE_SAMPLE} tasks"),
        (attestation.get("env_package") == episode["env_package"], "it attests another env package"),
        (attestation.get("entry") == f"verifiers:{catalogue['package']}", "it attests another taskset"),
        (attestation.get("split") == catalogue["rl_split"], "it attests another split"),
        (digest is not None and _options_digest(attestation.get("options")) == digest, "it attests other options"),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


def install_problem(source: Any, added: Mapping[str, Any]) -> str | None:
    env, catalogue = added["environment"], added["catalogue"]
    if source.identity != env["episode"]["env_package"]:
        return f"{source.identity} is installed, the entry names {env['episode']['env_package']}"
    if source.lock_sha256 != catalogue["image_lock_sha256"]:
        return "the installed image lock differs from the entry's"
    if source.manifest_sha256() != env["version"]:
        return "the installed manifest differs from the entry's version"
    if len(source) != env["dataset"]["rows"]:
        return f"the installed RL split has {len(source)} rows, the entry {env['dataset']['rows']}"
    return None


def _imported_module(source: Any) -> str | None:
    """The taskset module a bridged source imported (verifiers' own namespace stripped), or None (unknown)."""
    module = getattr(source, "module_name", None)
    if module is None:
        return None
    return str(module).removeprefix(_VERIFIERS_TASKSETS).lower()


def module_problem(source: Any, added: Mapping[str, Any], active_sources: Mapping[str, Any],
                   order: Any = None) -> str | None:
    """The added package's import module must be its distribution name with ``-`` -> ``_`` and no active env's
    (by the module it imported, or the one its distribution names) nor any order env's."""
    expected = _module_of(added["catalogue"]["package"])
    module = _imported_module(source)
    if module is not None and module != expected:
        return f"its import module {module} is not {expected} (the module is the distribution name)"
    for name, other in sorted(active_sources.items()):
        identity = getattr(other, "identity", None)
        theirs = {_imported_module(other), None if identity is None else _normal_module(identity)}
        if expected in theirs:
            return f"its import module {expected} is the active environment {name}'s (a new version is a new module)"
    environments = getattr(order, "order_environments", None) or {}
    for name, entry in sorted(environments.items()):
        episode = entry.get("episode") if isinstance(entry, Mapping) else None
        if isinstance(episode, Mapping) and _normal_module(episode.get("env_package", "")) == expected:
            return f"its import module {expected} is the order environment {name}'s (a new version is a new module)"
    return None


def identities_problem(sources: Mapping[str, Any]) -> str | None:
    """The active bridged envs whose installed package no longer has their pinned identity."""
    moved = []
    for name, source in sorted(sources.items()):
        try:
            now = source.installed_identity()
        except Exception as exc:
            now = f"unreadable ({type(exc).__name__})"
        if now != source.identity:
            moved.append(f"{name}: {source.identity} is pinned, {now} is installed")
        if len(moved) >= 8:
            break
    return None if not moved else "an active environment's install moved: " + "; ".join(moved)


def _scan(source: Any, images: frozenset[str], needed: int) -> tuple[int, int]:
    """(servable tasks found, tasks whose image could not be read), stopping at ``needed``."""
    servable = unreadable = 0
    for index in range(min(len(source), ACTIVATION_SCAN_MAX)):
        try:
            image = source.task_image(index)
        except Exception:
            unreadable += 1
            continue
        if image in images:
            servable += 1
            if servable >= needed:
                break
    return servable, unreadable


def machines_problem(fleet: Any, added: Mapping[str, Any], *, source: Any, now: float | None = None,
                     memo: dict | None = None) -> str | None:
    """The fleet's part of the activation; a ``NotYet`` problem while the fleet's view is incomplete."""
    episode = added["environment"]["episode"]
    env, package = episode["sandbox_env"], episode["env_package"]
    complete = fleet.view_complete(env, package, env_options_sha256=source.options_sha256, now=now)

    def answer(problem: str) -> str:
        if complete:
            return problem
        return NotYet(f"{problem} (not yet: the fleet's view is stale or incomplete; checked again later)")

    min_present = min(len(source.lock_images), ACTIVATION_MIN_IMAGES)
    serving = fleet.serving(env, package, env_options_sha256=source.options_sha256, lock_sha256=source.lock_sha256,
                            min_present=min_present, now=now)
    if len(serving) < MIN_ACTIVATION_MACHINES:
        reported = ", ".join(fleet.reported_packages(env)) or "nothing"
        return answer(f"{len(serving)} machine(s) serve {package} for {env} with its options and at least "
                      f"{min_present} locked images (need {MIN_ACTIVATION_MACHINES}); machines report {reported}")
    images = fleet.images_served(env, package, env_options_sha256=source.options_sha256, now=now)
    needed = min(len(source), ACTIVATION_MIN_SERVABLE_TASKS)
    key = (source.manifest_sha256(), images)
    if memo is not None and memo.get("key") == key:
        servable, unreadable = memo["scan"]             # the same images served: the last scan's count
    else:
        servable, unreadable = _scan(source, images, needed)
        if memo is not None:
            memo.update(key=key, scan=(servable, unreadable))
    if servable >= needed:
        return None
    detail = f"; the image of {unreadable} task(s) could not be read" if unreadable else ""
    return answer(f"only {servable} task(s) of its RL split are servable by the machines serving it (need "
                  f"{needed}){detail}: pre-pull more of its images")


def activation_problem(name: str, added: Mapping[str, Any], attestation: Any, *, build: Callable[[Any], Any],
                       fleet: Any, allowed: frozenset[str], active_sources: Mapping[str, Any], order: Any,
                       memo: dict | None = None) -> tuple[str | None, Any]:
    """(the first problem, None), or (None, the built environment). A ``NotYet`` problem is not a refusal."""
    from reliquary.environment.bridged_episode import BridgedEnvConfig

    for check in (lambda: package_problem(added, allowed), lambda: name_problem(added),
                  lambda: budget_problem(added, order)):
        problem = check()
        if problem is not None:                     # refused before anything of the package is imported
            return problem, None
    problem = attestation_problem(attestation, added)
    if problem is not None:
        return f"the attestation: {problem}", None
    try:
        environment = build(BridgedEnvConfig.from_catalogue(name, added["catalogue"]))
    except Exception as exc:          # an env that cannot be built here is not installed, whatever the cause
        return f"not installed on this validator ({type(exc).__name__}: {exc})", None
    source = environment.source
    for check in (lambda: install_problem(source, added),
                  lambda: module_problem(source, added, active_sources, order),
                  lambda: identities_problem(active_sources),
                  lambda: machines_problem(fleet, added, source=source, memo=memo)):
        problem = check()
        if problem is not None:
            return problem, None
    return None, environment


__all__ = ["ACTIVATION_MIN_IMAGES", "ACTIVATION_MIN_SERVABLE_TASKS", "ACTIVATION_SCAN_MAX", "ADDED_PACKAGES_FILE",
           "CONFORMANCE_GATE_SAMPLE", "EPISODE_LIMIT_FIELDS", "MIN_ACTIVATION_MACHINES", "NotYet",
           "activation_problem", "allowed_packages", "attestation_problem", "budget_problem", "identities_problem",
           "install_problem", "machines_problem", "module_problem", "name_problem", "package_problem"]
