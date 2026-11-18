"""``RELIQUARY_RL_BASE_MODEL_REVISION``: the order's pinned revision reaches the base-model download."""
import sys
from types import SimpleNamespace

from reliquary.constants import DEFAULT_BASE_MODEL, DEFAULT_BASE_MODEL_REVISION
from reliquary.trainer import async_cli

ENV = {"RELIQUARY_RL_TRAINER_URL": "http://c", "RELIQUARY_RL_TRAINER_TOKEN": "t",
       "RELIQUARY_RL_VALIDATOR_HOTKEY": "5Hot"}


def _download(monkeypatch, settings):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(snapshot_download=lambda repo, **kw: calls.append((repo, kw)) or "/snap"))
    return async_cli._download_base_model(settings), calls


def test_settings_carry_the_revision():
    assert async_cli.async_trainer_settings(ENV)["base_model_revision"] is None
    env = {**ENV, "RELIQUARY_RL_BASE_MODEL_REVISION": " abc123 "}
    assert async_cli.async_trainer_settings(env)["base_model_revision"] == "abc123"


def test_download_receives_the_pinned_revision(monkeypatch):
    path, calls = _download(monkeypatch, {"base_model": "org/m", "base_model_revision": "abc123"})
    assert path == "/snap" and calls == [("org/m", {"revision": "abc123"})]


def test_revision_overrides_the_default_model_pin(monkeypatch):
    _, calls = _download(monkeypatch, {"base_model": None, "base_model_revision": "abc123"})
    assert calls == [(DEFAULT_BASE_MODEL, {"revision": "abc123"})]


def test_unchanged_without_the_variable(monkeypatch):
    _, calls = _download(monkeypatch, {"base_model": "org/m", "base_model_revision": None})
    assert calls == [("org/m", {})]
    _, calls = _download(monkeypatch, {"base_model": None, "base_model_revision": None})
    assert calls == [(DEFAULT_BASE_MODEL, {"revision": DEFAULT_BASE_MODEL_REVISION})]
