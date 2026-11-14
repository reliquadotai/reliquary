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
    monkeypatch.setattr(ep, "_mcp_version", lambda: "1.30.0")
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
    monkeypatch.setattr(ep, "_installer", lambda: [sys.executable, "-m", "pip"])
    install, reinstall = ep.sync_commands(list(statuses.values()))
    assert "--override" not in install                 # pip has no overrides
    by_name = {p.distribution: p for p in ep.PACKAGES}
    # The missing one with its dependencies; the stale one in place, never uninstalled first.
    assert install[-2:] == ["install", by_name["reliquary-deepswe"].requirement()]
    assert reinstall[-4:] == ["install", "--force-reinstall", "--no-deps",
                              by_name["reliquary-terminal-bench"].requirement()]
    assert not any("uninstall" in command for command in (install, reinstall))
    # The sandbox is at its pin: it is not in either command.
    assert ep.SANDBOX.requirement() not in install + reinstall


def test_a_stale_or_missing_sandbox_is_synced_but_not_an_editable_one(monkeypatch):
    installed = _all_pinned()
    installed["reliquary-sandbox"] = None
    _fake_install(monkeypatch, installed)
    (install,) = ep.sync_commands(ep.env_status())
    assert install[-1] == ep.SANDBOX.requirement()

    installed["reliquary-sandbox"] = _git("0" * 40)
    (reinstall,) = ep.sync_commands(ep.env_status())
    assert reinstall[-3:] == ["--force-reinstall", "--no-deps", ep.SANDBOX.requirement()]

    installed["reliquary-sandbox"] = {"url": "file:///src/sandbox", "dir_info": {"editable": True}}
    installed["reliquary-deepswe"] = None
    (install,) = ep.sync_commands(ep.env_status())
    assert ep.SANDBOX.requirement() not in install
    assert install[-1] == {p.distribution: p for p in ep.PACKAGES}["reliquary-deepswe"].requirement()


def test_the_check_and_sync_hold_the_env_lock(monkeypatch, tmp_path):
    import fcntl

    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.delenv(ep.SYNCED_ENV_VAR, raising=False)
    held = []

    def probe():
        with open(ep.lock_path(), "a") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(True)
            else:
                held.append(False)
                fcntl.flock(other, fcntl.LOCK_UN)
        return []

    monkeypatch.setattr(ep, "env_status", probe)
    monkeypatch.setattr(ep, "_direct_url", lambda name: None)
    monkeypatch.setattr(ep, "log_status", lambda statuses: None)
    ep.startup_check(sync_envs=True)
    ep.sync()
    assert len(held) >= 2 and all(held)  # every check, at start and in the sync, under the lock
    probe()
    assert held[-1] is False  # released afterwards


def test_no_writable_prefix_runs_without_a_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "missing"))
    with ep.envs_lock():
        pass


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
    monkeypatch.setattr(ep, "_sync_locked", lambda statuses: synced.append(1) or statuses)
    monkeypatch.setattr(ep, "log_status", lambda statuses: None)
    _fake_install(monkeypatch, installed)
    ep.startup_check(sync_envs=True)
    assert synced == [1]

    monkeypatch.delenv(ep.SYNCED_ENV_VAR, raising=False)
    installed["reliquary-sandbox"] = None
    ep.startup_check(sync_envs=True)
    assert synced == [1]
    assert "sandbox-miner" in ep.hint(ep.env_status())


def test_mcp_2_refuses_the_bridge_and_a_sync_reinstalls_it(monkeypatch):
    """vLLM 0.30.0 asks mcp>=2; verifiers 0.3.1 imports mcp 1's FastMCP, so no env package loads."""
    _fake_install(monkeypatch, _all_pinned())
    monkeypatch.setattr(ep, "_mcp_version", lambda: "2.3.0")
    statuses = ep.env_status()
    (bridge,) = [s for s in statuses if s.package is ep.SANDBOX]
    assert "mcp 2.3.0" in bridge.problem and "miner-overrides.txt" in bridge.problem
    assert [s for s in statuses if not s.ok] == [bridge]
    monkeypatch.setattr(ep, "_installer", lambda: [sys.executable, "-m", "pip"])
    reinstall, mcp = ep.sync_commands(statuses)
    assert reinstall[-1] == ep.SANDBOX.requirement() and "--no-deps" in reinstall
    assert mcp[-2:] == ["install", ep.MCP_OVERRIDE]          # the no-deps reinstall alone keeps mcp 2


def test_the_override_file_is_the_bound_the_check_enforces():
    lines = [line.strip() for line in (ROOT / "miner-overrides.txt").read_text().splitlines()]
    assert [line for line in lines if line and not line.startswith("#")] == [ep.MCP_OVERRIDE]


@pytest.mark.parametrize("doc", ["docs/mining-orders.md", "docs/mining.md"])
def test_the_documented_install_applies_the_override(doc):
    from reliquary.miner.vllm_generation import SUPPORTED_VLLM_SERIES

    text = (ROOT / doc).read_text()
    assert ('uv pip install -e ".[sandbox-miner]" vllm==0.30.0 --override miner-overrides.txt'
            in text)
    assert "0.30.0".startswith(SUPPORTED_VLLM_SERIES)


def test_a_sync_prefers_uv_and_applies_the_mcp_override(monkeypatch, tmp_path):
    installed = _all_pinned()
    installed["reliquary-deepswe"] = None
    installed["reliquary-terminal-bench"] = _git("0" * 40)
    _fake_install(monkeypatch, installed)
    monkeypatch.setattr(ep, "_mcp_version", lambda: "2.3.0")
    monkeypatch.setattr(ep.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    assert ep._installer() == ["/usr/bin/uv", "pip"]
    install, reinstall, mcp = ep.sync_commands(ep.env_status())
    assert [c[:3] for c in (install, reinstall, mcp)] == [["/usr/bin/uv", "pip", "install"]] * 3
    for command in (install, mcp):                     # what resolves dependencies keeps verifiers' mcp bound
        override = command[command.index("--override") + 1]
        assert Path(override).read_text().strip() == ep.MCP_OVERRIDE
        assert command[command.index("--python") + 1] == sys.executable
    assert "--override" not in reinstall and "--no-deps" in reinstall
