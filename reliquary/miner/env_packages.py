"""The env packages and harness a miner needs for agentic orders, at the commits this build pins.

Eval orders on SWE-bench Verified, SWE-bench Pro, DeepSWE, Terminal-Bench 2.1 and AutomationBench refuse a
miner whose env package is not installed from git at the order's commit (the validator reads the
commit from pip's ``direct_url.json``). ``reliquary[sandbox-miner]`` lists them, but pip does not
reinstall a git dependency whose commit moved while its version stayed the same, so an update
alone can leave an old commit in place. ``reliquary miner-envs --sync`` (run by the miner at start when
reliquary-sandbox is installed, i.e. it took the extra) reinstalls exactly the ones missing or stale.

The pins here and the ``miner-envs`` extra in pyproject.toml are the same list
(tests/unit/test_miner_env_packages.py).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from reliquary.sandbox import COMPATIBLE_SANDBOX_COMMITS, SANDBOX_COMMIT
from reliquary.shared.dist_commit import dist_commit

logger = logging.getLogger(__name__)

ENVIRONMENTS_REPO = "https://github.com/reliquadotai/reliquary-environments.git"
SANDBOX_REPO = "https://github.com/reliquadotai/reliquary-sandbox.git"
# The env packages are built for the gateways' Python (their Requires-Python is >=3.12,<3.13).
ENV_PYTHON = (3, 12)
SYNCED_ENV_VAR = "RELIQUARY_MINER_ENVS_CHECKED"


@dataclass(frozen=True)
class PinnedPackage:
    distribution: str
    repo: str
    commit: str
    subdirectory: str | None = None
    extras: str = ""
    serves: str = ""
    python: tuple[int, int] | None = None
    compatible: frozenset[str] = frozenset()

    def requirement(self, *, marker: bool = False) -> str:
        spec = f"{self.distribution}{self.extras} @ git+{self.repo}@{self.commit}"
        if self.subdirectory:
            spec += f"#subdirectory={self.subdirectory}"
        if marker and self.python:
            spec += f" ; python_version == '{self.python[0]}.{self.python[1]}'"
        return spec

    def accepts(self, commit: str | None) -> bool:
        return commit == self.commit or commit in self.compatible


SANDBOX = PinnedPackage("reliquary-sandbox", SANDBOX_REPO, SANDBOX_COMMIT, extras="[verifiers]",
                        serves="every signed-sandbox order",
                        compatible=COMPATIBLE_SANDBOX_COMMITS)
HARNESS = PinnedPackage("reliquary-harness", SANDBOX_REPO, SANDBOX_COMMIT, "harness", "[qwen]",
                        serves="every agentic order")


def _env(distribution: str, commit: str, subdirectory: str, serves: str) -> PinnedPackage:
    return PinnedPackage(distribution, ENVIRONMENTS_REPO, commit, f"environments/{subdirectory}",
                         serves=serves, python=ENV_PYTHON)


ENV_PACKAGES = (
    _env("reliquary-swe", "15ad131b473f59f46b62d363bfacf430f4931f39", "code/reliquary_swe",
         "SWE-bench Verified orders"),
    _env("reliquary-swebench-pro", "d2cf12b1c144515397fae634f5e848f7df55c180",
         "code/reliquary_swebench_pro", "SWE-bench Pro orders"),
    _env("reliquary-deepswe", "faaefd675e6f6583e6e1ce3f3a3eca109fe8e766", "code/reliquary_deepswe",
         "DeepSWE orders"),
    _env("reliquary-terminal-bench", "a7718445c156a36eec2835c43d5fb6c86c045a5e",
         "code/reliquary_terminal_bench", "Terminal-Bench 2.1 orders"),
    _env("reliquary-automationbench", "b404a0bb93b289a44ae242c001148f21982812a7",
         "tool_use/reliquary_automationbench", "AutomationBench orders"),
)
PACKAGES = (SANDBOX, HARNESS, *ENV_PACKAGES)


@dataclass(frozen=True)
class PackageStatus:
    package: PinnedPackage
    installed: str | None
    """The installed commit, or None (missing, or not a git install)."""
    problem: str | None
    editable: bool = False

    @property
    def ok(self) -> bool:
        return self.problem is None


def _direct_url(distribution: str) -> dict | None:
    try:
        raw = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(raw) if raw else {}


def _commit(distribution: str) -> str | None:
    return dist_commit(distribution)


def _status(package: PinnedPackage) -> PackageStatus:
    if package.python and sys.version_info[:2] != package.python:
        need = f"{package.python[0]}.{package.python[1]}"
        return PackageStatus(package, None, f"needs Python {need} (this is "
                                            f"{sys.version_info[0]}.{sys.version_info[1]})")
    info = _direct_url(package.distribution)
    if info is None:
        return PackageStatus(package, None, "not installed")
    editable = bool((info.get("dir_info") or {}).get("editable"))
    installed = _commit(package.distribution)
    if installed is None:
        return PackageStatus(package, None, "installed, but not from git at a known commit "
                                            "(a wheel, or a dirty checkout)", editable)
    if not package.accepts(installed):
        return PackageStatus(package, installed, f"at {installed[:12]}, needs {package.commit[:12]}",
                             editable)
    return PackageStatus(package, installed, None, editable)


_DIGEST_PROBE = ("from reliquary.environment.agentic_swe import harness_refusal\n"
                 "print(harness_refusal() or '', end='')")


def harness_digest_refusal() -> str | None:
    """`agentic_swe.harness_refusal`, read in a child process: this one must not import a harness it
    may be about to replace (it would keep the old one in memory)."""
    try:
        probe = subprocess.run([sys.executable, "-c", _DIGEST_PROBE], capture_output=True, text=True,
                               timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"the harness check could not run ({exc})"
    if probe.returncode != 0:
        last = (probe.stderr.strip().splitlines() or ["no output"])[-1]
        return f"reliquary-harness does not load this build's specs ({last})"
    return probe.stdout.strip() or None


def env_status() -> list[PackageStatus]:
    statuses = [_status(package) for package in PACKAGES]
    digest = harness_digest_refusal()
    if digest:
        statuses = [PackageStatus(s.package, s.installed, s.problem or digest, s.editable)
                    if s.package is HARNESS else s for s in statuses]
    return statuses


def report_lines(statuses: list[PackageStatus]) -> list[str]:
    lines = []
    for status in statuses:
        package = status.package
        where = package.serves
        if status.ok:
            lines.append(f"ok       {package.distribution} @ {status.installed[:12]} ({where})")
        else:
            label = "MISSING " if status.installed is None else "STALE   "
            lines.append(f"{label} {package.distribution}: {status.problem} ({where} refused)")
    return lines


def _installer() -> list[str] | None:
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip"]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip"]
    return None


def sync_commands(statuses: list[PackageStatus]) -> list[list[str]]:
    """The installer commands that bring the stale or missing packages to their pins: the stale ones
    are uninstalled first (pip keeps a git install whose version did not change), then every one is
    installed with the reliquary-sandbox verifiers bridge, so their ``verifiers==0.3.1`` resolves to
    the bridge's git verifiers (tag v0.3.1) and not to PyPI's. Editable installs and packages this
    Python cannot install are left alone."""
    todo = [s for s in statuses if not s.ok and not s.editable and not _wrong_python(s)]
    if not todo:
        return []
    installer = _installer() or [sys.executable, "-m", "pip"]
    python = [] if installer[0] == sys.executable else ["--python", sys.executable]
    commands = []
    stale = [s.package.distribution for s in todo if _direct_url(s.package.distribution) is not None]
    if stale:
        confirm = [] if python else ["-y"]  # uv pip uninstall asks nothing (and has no -y)
        commands.append([*installer, "uninstall", *python, *confirm, *stale])
    requirements = [s.package.requirement() for s in todo]
    if SANDBOX.requirement() not in requirements:
        requirements.insert(0, SANDBOX.requirement())
    commands.append([*installer, "install", *python, *requirements])
    return commands


def sync(statuses: list[PackageStatus] | None = None) -> list[PackageStatus]:
    """Install what is missing or stale, then check again."""
    statuses = env_status() if statuses is None else statuses
    commands = sync_commands(statuses)
    if commands and _installer() is None:
        logger.warning("miner envs: neither pip nor uv is available here; run: %s",
                       " && ".join(" ".join(c) for c in commands))
        return statuses
    for command in commands:
        logger.info("miner envs: %s", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            logger.warning("miner envs: `%s` failed (exit %s); see its output above",
                           " ".join(command[:4]), result.returncode)
            break
    importlib.invalidate_caches()
    return env_status()


def startup_check(*, sync_envs: bool) -> None:
    """The miner's start: log which order families this install can serve, after installing what is
    missing or stale when ``sync_envs``. Never fatal: single-turn orders need none of these. Done
    once per miner, not again in the order follower's children."""
    if os.environ.get(SYNCED_ENV_VAR):
        return
    os.environ[SYNCED_ENV_VAR] = "1"
    try:
        statuses = env_status()
        # Only an install that took the miner extra (reliquary-sandbox is there) is synced: a single-turn
        # miner without it is told what agentic orders need, not given them at every start.
        opted_in = _direct_url(SANDBOX.distribution) is not None
        if sync_envs and opted_in and not all(s.ok for s in statuses) and sync_commands(statuses):
            statuses = sync(statuses)
        log_status(statuses)
    except Exception:
        logger.warning("miner envs: the check failed", exc_info=True)


def _wrong_python(status: PackageStatus) -> bool:
    return status.package.python is not None and sys.version_info[:2] != status.package.python


def hint(statuses: list[PackageStatus]) -> str | None:
    """What to do about the packages that are not ok, or None."""
    bad = [s for s in statuses if not s.ok]
    if not bad:
        return None
    if any(s.package is SANDBOX and s.installed is None and s.problem == "not installed" for s in bad):
        return ('agentic orders need the miner extra: pip install -e ".[sandbox-miner]" && '
                "reliquary miner-envs --sync")
    if any(_wrong_python(s) for s in bad):
        need = ".".join(map(str, ENV_PYTHON))
        return (f"the agentic env packages are built for Python {need}: mine from a Python {need} venv "
                f"(python{need} -m venv .venv), then update as usual")
    if any(s.editable for s in bad):
        return "an editable checkout is not at its pin: check it out at the commit above, or reinstall from git"
    return "run `reliquary miner-envs --sync` to install them at their pins"


def log_status(statuses: list[PackageStatus]) -> None:
    lines = report_lines(statuses)
    if all(s.ok for s in statuses):
        logger.info("miner envs: every agentic order family is served\n  %s", "\n  ".join(lines))
        return
    logger.warning("miner envs: some agentic orders will be refused; %s\n  %s", hint(statuses),
                   "\n  ".join(lines))
