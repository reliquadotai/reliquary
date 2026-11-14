"""The env packages a miner installs for agentic orders: one list, and the start check over it."""

import logging
import sys
import tomllib
from pathlib import Path

import pytest

from reliquary.miner import env_packages as ep
from reliquary.sandbox import SANDBOX_COMMIT

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
DEPENDENCIES = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]


def test_the_extra_is_the_pinned_list():
    assert EXTRAS["miner-envs"] == [p.requirement(marker=True) for p in ep.ENV_PACKAGES]


def test_the_miner_extra_pulls_the_env_packages():
    assert "reliquary[miner-envs]" in EXTRAS["sandbox-miner"]
    assert EXTRAS["sandbox-miner"][0] == ep.SANDBOX.requirement()


def test_the_harness_pin_is_the_base_dependency():
    assert ep.HARNESS.requirement() in DEPENDENCIES
    assert ep.HARNESS.commit == SANDBOX_COMMIT


@pytest.mark.parametrize("package", ep.ENV_PACKAGES, ids=lambda p: p.distribution)
def test_every_env_package_is_a_full_commit_from_git(package):
    assert len(package.commit) == 40 and int(package.commit, 16) >= 0
    assert package.requirement().startswith(f"{package.distribution} @ git+https://")


def _fake_install(monkeypatch, installed: dict[str, dict | None], python=(3, 12)):
    """``installed``: distribution -> its direct_url.json (None: not installed)."""
    monkeypatch.setattr(ep, "_direct_url", lambda name: installed.get(name))
    monkeypatch.setattr(ep, "_commit",
                        lambda name: ((installed.get(name) or {}).get("vcs_info") or {}).get("commit_id"))
    monkeypatch.setattr(ep, "harness_digest_refusal", lambda: None)
    monkeypatch.setattr(sys, "version_info", (*python, 0, "final", 0))


def _git(commit):
    return {"url": "https://example", "vcs_info": {"vcs": "git", "commit_id": commit}}


def _all_pinned():
    return {p.distribution: _git(p.commit) for p in ep.PACKAGES}


def test_everything_at_its_pin_is_ok(monkeypatch):
    _fake_install(monkeypatch, _all_pinned())
    statuses = ep.env_status()
    assert all(s.ok for s in statuses)
    assert ep.sync_commands(statuses) == []
    assert ep.hint(statuses) is None
    assert all(line.startswith("ok") for line in ep.report_lines(statuses))


def test_missing_and_stale_packages_are_named_and_reinstalled(monkeypatch):
    installed = _all_pinned()
    installed["reliquary-deepswe"] = None
    installed["reliquary-terminal-bench"] = _git("0" * 40)
    _fake_install(monkeypatch, installed)
    statuses = {s.package.distribution: s for s in ep.env_status()}
    assert statuses["reliquary-deepswe"].problem == "not installed"
    assert "000000000000" in statuses["reliquary-terminal-bench"].problem
    lines = ep.report_lines(list(statuses.values()))
    assert any("DeepSWE orders refused" in line for line in lines)
    uninstall, install = ep.sync_commands(list(statuses.values()))
    assert uninstall[-2:] == ["-y", "reliquary-terminal-bench"]
    # The verifiers bridge comes with them, so their verifiers==0.3.1 is its git verifiers.
    assert ep.SANDBOX.requirement() in install
    by_name = {p.distribution: p for p in ep.PACKAGES}
    assert {r for r in install if r.startswith("reliquary-")} == {
        ep.SANDBOX.requirement(), by_name["reliquary-deepswe"].requirement(),
        by_name["reliquary-terminal-bench"].requirement()}


def test_a_compatible_sandbox_release_is_accepted(monkeypatch):
    installed = _all_pinned()
    other = next(iter(ep.SANDBOX.compatible - {SANDBOX_COMMIT}))
    installed["reliquary-sandbox"] = _git(other)
    _fake_install(monkeypatch, installed)
    assert all(s.ok for s in ep.env_status())


def test_an_editable_checkout_is_reported_not_touched(monkeypatch):
    installed = _all_pinned()
    installed["reliquary-swe"] = {"url": "file:///src", "dir_info": {"editable": True}}
    _fake_install(monkeypatch, installed)
    statuses = ep.env_status()
    assert not all(s.ok for s in statuses)
    assert ep.sync_commands(statuses) == []


def test_another_python_names_the_one_needed(monkeypatch):
    installed = _all_pinned()
    _fake_install(monkeypatch, installed, python=(3, 11))
    statuses = {s.package.distribution: s for s in ep.env_status()}
    assert statuses["reliquary-deepswe"].problem == "needs Python 3.12 (this is 3.11)"
    assert statuses["reliquary-harness"].ok
    assert ep.sync_commands(list(statuses.values())) == []
    assert "Python 3.12 venv" in ep.hint(list(statuses.values()))


def test_a_wrong_harness_digest_is_reported(monkeypatch):
    _fake_install(monkeypatch, _all_pinned())
    monkeypatch.setattr(ep, "harness_digest_refusal", lambda: "digest x, this build needs y")
    (harness,) = [s for s in ep.env_status() if s.package is ep.HARNESS]
    assert harness.problem == "digest x, this build needs y"


def test_the_start_check_runs_once_and_never_raises(monkeypatch, caplog):
    monkeypatch.delenv(ep.SYNCED_ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(ep, "env_status", lambda: calls.append(1) or (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr(ep.logger, "disabled", False)
    with caplog.at_level(logging.WARNING, logger=ep.logger.name):
        ep.startup_check(sync_envs=True)
        ep.startup_check(sync_envs=True)
    assert calls == [1]
    assert "the check failed" in caplog.text


def test_the_start_check_syncs_only_an_install_that_took_the_miner_extra(monkeypatch):
    monkeypatch.delenv(ep.SYNCED_ENV_VAR, raising=False)
    installed = _all_pinned()
    installed["reliquary-deepswe"] = None
    synced = []
    monkeypatch.setattr(ep, "sync", lambda statuses: synced.append(1) or statuses)
    monkeypatch.setattr(ep, "log_status", lambda statuses: None)
    _fake_install(monkeypatch, installed)
    ep.startup_check(sync_envs=True)
    assert synced == [1]

    monkeypatch.delenv(ep.SYNCED_ENV_VAR, raising=False)
    installed["reliquary-sandbox"] = None
    ep.startup_check(sync_envs=True)
    assert synced == [1]
    assert "sandbox-miner" in ep.hint(ep.env_status())
