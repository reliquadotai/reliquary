"""The activation checks of an env added to a running run. Pure: the validator calls ``activation_problem`` in a
worker thread, first when it sees the request (it builds the env then, never at the window boundary) and again
at the boundary, and refuses the request with the first problem found.

Before anything of the package is imported (``before_import_problem``; the validator's prestage runs it too):

1. the package is on the owner's allowlist (``episode-packages.allow`` in the service folder: one
   ``name==version`` per line, or the full identity ``name==version+g<hash>.vfb<N>``, which then pins the code
   and not only the version), so a third-party env never runs in the validator process unreviewed;
2. its distribution name is plain (letters and digits joined by single ``-`` or ``_``, no ``.``), so its import
   module (the name with ``-`` -> ``_``, lowercased) is a plain identifier that pip's normalised name maps one to
   one, and that module is no standard-library module;
3. every per-env budget it declares is at most the order's: ``max_episode_tokens``, ``max_turns``,
   ``max_tokens_per_turn`` and each session budget (the values the trainer and the sandbox fleet were sized for);
4. the installed distributions (name, version, RECORD digest, read from their metadata only) are the ones this
   validator booted with plus, at most, the added distribution, whose only top-level name is its module, a name
   no other distribution claims: an install moved no dependency of an active env and shadows nothing.

Then:

5. the operator's attestation is a conformance GATE report of exactly this env (the format the sandbox's
   conformance ``--report`` writes): gate and passed true, no problem, real boxes, a full sample, the same env
   package identity, taskset entry, split and options;
6. the env is installed on this validator: built through the bridge, its identity, image lock, manifest digest
   and row count are the entry's; its import module is exactly its distribution's and no active env's nor any
   order env's; and that module was not loaded earlier in this process at another identity (new files under a
   module already in memory need a validator restart);
7. installing it moved no active bridged env: each one's package on disk still has its pinned identity;
8. at least ``MIN_ACTIVATION_MACHINES`` fresh active machines could host its sessions (the open path's own
   filter, its image aside: identity, options, image lock, tools, caps against its budgets, token validity), with
   at least ``min(images, ACTIVATION_MIN_IMAGES)`` locked images present, and at least
   ``min(rows, ACTIVATION_MIN_SERVABLE_TASKS)`` tasks of its RL split are servable (their image is held by such a
   machine under its lock). Each task's image is read once per manifest (``memo``, at most
   ``ACTIVATION_SCAN_MAX`` of them) and counted against the images served now.

A problem the validator cannot settle now is a ``NotYet`` (a ``str``): the request waits, it is not refused. That
is a shortfall in 8 on an incomplete or stale fleet view (``Fleet.view_complete``, the precommit route's rule),
and an I/O error or timeout while building the env or reading an active env's identity."""
from __future__ import annotations

import hashlib
import keyword
import os
import re
import sys
from collections.abc import Callable, Mapping, MutableMapping
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
_LISTED = 8                                 # differences named in one message
_TRANSIENT = (OSError, TimeoutError)        # an I/O error or a timeout: not a verdict on the env


class NotYet(str):
    """A problem the validator cannot settle now (a stale or incomplete fleet view, an I/O error): the request
    waits and is checked again, it is never refused for it."""


def allowed_packages(folder: str | Path) -> frozenset[str]:
    """The owner-reviewed lines of ``folder/episode-packages.allow`` (empty when absent, unsafe or too large:
    then nothing can be added)."""
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
    """The package must be allowed. A line naming a full identity of this ``name==version`` pins it: then only
    that identity (``episode.env_package``) is allowed."""
    package = added["catalogue"]["package"]
    identity = added["environment"]["episode"]["env_package"]
    pinned = {line for line in allowed if line.startswith(package + "+")}
    if pinned:
        if identity not in pinned:
            return (f"{identity} is not the identity the owner's allowlist pins for {package} "
                    f"({ADDED_PACKAGES_FILE} in the service folder)")
        return None
    if package not in allowed:
        return f"{package} is not on the owner's allowlist ({ADDED_PACKAGES_FILE} in the service folder)"
    return None


def _module_of(env_package: str) -> str:
    """The import module of a pinned package (its distribution name with ``-`` -> ``_``, lowercased)."""
    return str(env_package).split("==", 1)[0].replace("-", "_").lower()


def _normal(name: str) -> str:
    """A distribution name as pip compares it (PEP 503)."""
    from reliquary.protocol.service_catalogue import distribution_of

    return distribution_of(name)


def _normal_module(env_package: str) -> str:
    """The module a package's distribution claims, as pip compares names (PEP 503, then ``-`` -> ``_``): equal for
    two spellings pip would install over each other."""
    return _normal(env_package).replace("-", "_")


def name_problem(added: Mapping[str, Any]) -> str | None:
    """The added package's distribution name must be plain, so its import module is a plain identifier that pip's
    normalised name (PEP 503) maps one to one: no ``.`` (a dotted module), no run of separators; and the module
    must not be a standard-library module (it would shadow it, or never be imported)."""
    for what, package in (("catalogue.package", added["catalogue"]["package"]),
                          ("episode.env_package", added["environment"]["episode"]["env_package"])):
        name = str(package).split("==", 1)[0]
        module = _module_of(package)
        if (not _PLAIN_DISTRIBUTION.fullmatch(name) or not module.isidentifier() or not module.isascii()
                or keyword.iskeyword(module)):
            return (f"its {what} distribution name {name!r} is not plain (letters and digits joined by single "
                    f"'-' or '_', no '.'): its import module must be a plain identifier")
        if module in sys.stdlib_module_names:
            return f"its import module {module} is a standard-library module's name: rename the distribution"
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


# --- installed distributions, from their metadata only ------------------------------------------------------------
def _top_level(files: Any) -> frozenset[str]:
    """The importable top-level names a distribution's RECORD installs (``.pth`` files kept by their file name:
    they run at interpreter start, so they are never a module's name)."""
    names = set()
    for path in files or ():
        parts = getattr(path, "parts", None) or Path(str(path)).parts
        if not parts or parts[0] in ("..", "__pycache__") or parts[0].endswith((".dist-info", ".data")):
            continue
        first = parts[0]
        if len(parts) == 1:
            first = first if first.endswith(".pth") else first.split(".", 1)[0]
        names.add(first)
    return frozenset(names)


def installed_distributions(distributions: Callable[[], Any] | None = None) -> dict[str, frozenset]:
    """``{PEP 503 name: {(version, RECORD sha256, top-level names)}}`` of every installed distribution, read from
    their metadata only (nothing imported). The validator takes it at boot; activation compares it again."""
    if distributions is None:
        from importlib.metadata import distributions
    found: dict[str, set] = {}
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
            record = dist.read_text("RECORD") or ""
            files = dist.files
        except Exception as exc:                 # an unreadable distribution is still listed: it differs
            name, record, files = getattr(dist, "name", None) or "?", f"unreadable {type(exc).__name__}", ()
        if not name:
            continue
        digest = hashlib.sha256(record.encode("utf-8")).hexdigest()
        found.setdefault(_normal(name), set()).add((str(dist.version), digest, _top_level(files)))
    return {name: frozenset(entries) for name, entries in found.items()}


def install_set_problem(added: Mapping[str, Any], boot: Mapping[str, frozenset],
                        current: Mapping[str, frozenset]) -> str | None:
    """The installed distributions must be the boot's plus, at most, the added one (installed once, at the entry's
    version, its only top-level name its module, a name no other distribution claims)."""
    package = added["catalogue"]["package"]
    name, version = _normal(package), package.split("==", 1)[1]
    moved = sorted(other for other in set(boot) | set(current)
                   if other != name and boot.get(other) != current.get(other))
    if moved:
        listed = ", ".join(f"{other} ({'new' if other not in boot else 'gone' if other not in current else 'changed'})"
                           for other in moved[:_LISTED])
        return (f"the installed distributions moved since this validator started: {listed} (install an added env "
                f"with pip install --no-deps, nothing else)")
    if name in boot and boot[name] != current.get(name):
        return f"{name} changed on disk since this validator started (restart the validator)"
    entries = current.get(name) or frozenset()
    if len(entries) != 1:
        return f"{name} is installed {len(entries)} times, need exactly once"
    installed_version, _digest, top = next(iter(entries))
    if installed_version != version:
        return f"{name} {installed_version} is installed, the entry names {version}"
    expected = _module_of(package)
    if top != {expected}:
        return f"{name} installs the top-level names {sorted(top)}, need exactly {expected!r}"
    claims = sorted(other for other, others in current.items() if other != name
                    and any(expected in entry[2] for entry in others))
    if claims:
        return f"the import module {expected} is also installed by {', '.join(claims[:_LISTED])}"
    return None


def before_import_problem(added: Mapping[str, Any], allowed: frozenset[str], order: Any, *,
                          boot: Mapping[str, frozenset],
                          current: Mapping[str, frozenset] | None = None) -> str | None:
    """Every check that runs before anything of the package is imported: the allowlist, the name, the budgets and
    the installed distributions (``boot``: ``installed_distributions()`` at the validator's start; ``current``:
    now, read when not given)."""
    for check in (lambda: package_problem(added, allowed), lambda: name_problem(added),
                  lambda: budget_problem(added, order)):
        problem = check()
        if problem is not None:
            return problem
    if current is None:
        try:
            current = installed_distributions()
        except _TRANSIENT as exc:
            return NotYet(f"the installed distributions cannot be read now ({type(exc).__name__}: {exc})")
    return install_set_problem(added, boot, current)


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
    """The taskset module an ACTIVE bridged source imported (verifiers' own namespace stripped), or None."""
    module = getattr(source, "module_name", None)
    if module is None:
        return None
    return str(module).removeprefix(_VERIFIERS_TASKSETS).lower()


def module_problem(source: Any, added: Mapping[str, Any], active_sources: Mapping[str, Any],
                   order: Any = None) -> str | None:
    """The added package's import module must be exactly its distribution name with ``-`` -> ``_`` (its own
    top-level module, never one under another package's namespace) and no active env's (by the module it imported,
    or the one its distribution names) nor any order env's."""
    expected = _module_of(added["catalogue"]["package"])
    module = getattr(source, "module_name", None)
    if module is not None and str(module) != expected:
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
    """The active bridged envs whose installed package no longer has their pinned identity (a ``NotYet`` when an
    identity only cannot be read now)."""
    moved, unreadable = [], []
    for name, source in sorted(sources.items()):
        try:
            now = source.installed_identity()
        except _TRANSIENT as exc:
            unreadable.append(f"{name} ({type(exc).__name__})")
            continue
        except Exception as exc:
            now = f"unreadable ({type(exc).__name__})"
        if now != source.identity:
            moved.append(f"{name}: {source.identity} is pinned, {now} is installed")
        if len(moved) >= _LISTED:
            break
    if moved:
        return "an active environment's install moved: " + "; ".join(moved)
    if unreadable:
        return NotYet("an active environment's identity cannot be read now: " + ", ".join(unreadable[:_LISTED]))
    return None


def _servable(source: Any, images: frozenset[str], needed: int, memo: dict | None) -> tuple[int, int]:
    """(servable tasks, tasks whose image could not be read), stopping at ``needed``. Each task's image is read
    once per manifest (``memo["tasks"] = (manifest, images by index)``, one atomic entry) and counted against
    ``images`` on each call."""
    manifest = source.manifest_sha256()
    entry = None if memo is None else memo.get("tasks")
    known = list(entry[1]) if entry is not None and entry[0] == manifest else []
    limit = min(len(source), ACTIVATION_SCAN_MAX)
    servable = sum(1 for image in known if image is not None and image in images)
    while servable < needed and len(known) < limit:
        try:
            image = source.task_image(len(known))
        except Exception:
            image = None
        known.append(image)
        if image is not None and image in images:
            servable += 1
    if memo is not None:
        memo["tasks"] = (manifest, tuple(known))
    return servable, sum(1 for image in known if image is None)


def machines_problem(fleet: Any, added: Mapping[str, Any], *, source: Any, validity_s: int,
                     now: float | None = None, memo: dict | None = None) -> str | None:
    """The fleet's part of the activation; a ``NotYet`` problem while the fleet's view is incomplete.
    ``validity_s``: the token validity the issuer would give its sessions (machines' caps must allow it)."""
    episode = added["environment"]["episode"]
    env, package, budgets = episode["sandbox_env"], episode["env_package"], dict(episode["budgets"])
    host = dict(env_options_sha256=source.options_sha256, lock_sha256=source.lock_sha256, budgets=budgets,
                validity_s=int(validity_s), now=now)
    complete = fleet.view_complete(env, package, **host)

    def answer(problem: str) -> str:
        if complete:
            return problem
        return NotYet(f"{problem} (not yet: the fleet's view is stale or incomplete; checked again later)")

    min_present = min(len(source.lock_images), ACTIVATION_MIN_IMAGES)
    serving = fleet.serving(env, package, min_present=min_present, **host)
    if len(serving) < MIN_ACTIVATION_MACHINES:
        reported = ", ".join(fleet.reported_packages(env)) or "nothing"
        return answer(f"{len(serving)} machine(s) could host {package} for {env} (its options, image lock, tools, "
                      f"caps for its budgets and a {int(validity_s)} s token) with at least {min_present} locked "
                      f"images (need {MIN_ACTIVATION_MACHINES}); machines report {reported}")
    images = fleet.images_served(env, package, **host)
    needed = min(len(source), ACTIVATION_MIN_SERVABLE_TASKS)
    servable, unreadable = _servable(source, images, needed, memo)
    if servable >= needed:
        return None
    detail = f"; the image of {unreadable} task(s) could not be read" if unreadable else ""
    return answer(f"only {servable} task(s) of its RL split are servable by the machines that could host it (need "
                  f"{needed}){detail}: pre-pull more of its images")


def activation_problem(name: str, added: Mapping[str, Any], attestation: Any, *, build: Callable[[Any], Any],
                       fleet: Any, allowed: frozenset[str], active_sources: Mapping[str, Any], order: Any,
                       boot: Mapping[str, frozenset], validity_s: int,
                       current: Mapping[str, frozenset] | None = None, memo: dict | None = None,
                       loaded: MutableMapping[str, str] | None = None) -> tuple[str | None, Any]:
    """(the first problem, None), or (None, the built environment). A ``NotYet`` problem is not a refusal.
    ``boot``: ``installed_distributions()`` at the validator's start; ``validity_s``: the issuer's token validity
    for the env's sessions; ``loaded``: the record of modules built in this process (default: the process's)."""
    from reliquary.environment.bridged_episode import (
        BridgedEnvConfig,
        loaded_identity_problem,
    )

    problem = before_import_problem(added, allowed, order, boot=boot, current=current)
    if problem is not None:                         # refused before anything of the package is imported
        return problem, None
    problem = attestation_problem(attestation, added)
    if problem is not None:
        return f"the attestation: {problem}", None
    try:
        environment = build(BridgedEnvConfig.from_catalogue(name, added["catalogue"]))
    except _TRANSIENT as exc:         # a download or a disk read that failed now: tried again later
        return NotYet(f"cannot be built now ({type(exc).__name__}: {exc})"), None
    except Exception as exc:          # an env that cannot be built here is not installed, whatever the cause
        return f"not installed on this validator ({type(exc).__name__}: {exc})", None
    source = environment.source
    for check in (lambda: install_problem(source, added),
                  lambda: module_problem(source, added, active_sources, order),
                  lambda: loaded_identity_problem(getattr(source, "module_name", None), source.identity, loaded),
                  lambda: identities_problem(active_sources),
                  lambda: machines_problem(fleet, added, source=source, validity_s=validity_s, memo=memo)):
        problem = check()
        if problem is not None:
            return problem, None
    return None, environment


__all__ = ["ACTIVATION_MIN_IMAGES", "ACTIVATION_MIN_SERVABLE_TASKS", "ACTIVATION_SCAN_MAX", "ADDED_PACKAGES_FILE",
           "CONFORMANCE_GATE_SAMPLE", "EPISODE_LIMIT_FIELDS", "MIN_ACTIVATION_MACHINES", "NotYet",
           "activation_problem", "allowed_packages", "attestation_problem", "before_import_problem", "budget_problem",
           "identities_problem", "install_problem", "install_set_problem", "installed_distributions",
           "machines_problem", "module_problem", "name_problem", "package_problem"]
