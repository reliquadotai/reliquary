"""The active profile may come from a contract file instead of the compiled
catalogue. With no file named, resolution must be exactly what it was."""

import json

import pytest

from reliquary.protocol.profiles import (
    DEFAULT_PROFILE_ID,
    PROFILES,
    TASK_CONTRACT_ENV_VAR,
    resolve_protocol_profile,
)


def test_without_the_variable_resolution_is_unchanged(monkeypatch):
    monkeypatch.delenv(TASK_CONTRACT_ENV_VAR, raising=False)
    monkeypatch.delenv("RELIQUARY_PROTOCOL_PROFILE", raising=False)
    assert resolve_protocol_profile() is PROFILES[DEFAULT_PROFILE_ID]


def test_an_explicit_profile_id_still_wins_over_the_catalogue(monkeypatch):
    monkeypatch.delenv(TASK_CONTRACT_ENV_VAR, raising=False)
    chosen = sorted(PROFILES)[0]
    assert resolve_protocol_profile(chosen) is PROFILES[chosen]


def test_a_contract_file_replaces_the_catalogue(tmp_path, monkeypatch):
    source = PROFILES[sorted(PROFILES)[0]]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(source.to_generation_contract()))
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, str(path))

    resolved = resolve_protocol_profile()
    assert resolved == source
    # Rebuilt, not fetched from the catalogue: a task's contract is the
    # authority, and the catalogue is only a template.
    assert resolved is not source


def test_a_missing_contract_file_is_a_loud_failure(tmp_path, monkeypatch):
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, str(tmp_path / "absent.json"))
    with pytest.raises(ValueError) as caught:
        resolve_protocol_profile()
    assert "absent.json" in str(caught.value)


def test_an_unreadable_contract_file_is_a_loud_failure(tmp_path, monkeypatch):
    path = tmp_path / "contract.json"
    path.write_text("{not json")
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, str(path))
    with pytest.raises(ValueError):
        resolve_protocol_profile()


def test_an_explicit_profile_id_overrides_the_contract_file(tmp_path, monkeypatch):
    # The CLI passes an explicit id when it seeds a template; that must not be
    # hijacked by whatever contract this process happens to run.
    source = PROFILES[sorted(PROFILES)[0]]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(source.to_generation_contract()))
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, str(path))
    other = sorted(PROFILES)[-1]
    assert resolve_protocol_profile(other) is PROFILES[other]
