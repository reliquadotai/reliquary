"""Detached-mode serial-beat gating and flag-off parity."""

from types import SimpleNamespace

from reliquary import constants as C
from reliquary.validator.service import ValidationService


class _StoreStub:
    def __init__(self, manifest=None):
        self._manifest = manifest

    def current_manifest(self):
        return self._manifest


def _stub(intake=None, trained_since=0, publish_every=16):
    return SimpleNamespace(
        _checkpoint_intake=intake,
        _trained_windows_since_publish=trained_since,
        _publish_every=publish_every,
        _adaptive_publication_pending=False,
        _checkpoint_n=100,
        _checkpoint_store=_StoreStub(manifest=object()),
    )


def test_flag_off_parity_cadence(monkeypatch):
    monkeypatch.setattr(C, "DETACHED_TRAINER", False)
    monkeypatch.delenv("RELIQUARY_DISABLE_TRAIN", raising=False)
    assert ValidationService._publication_due_next_half(
        _stub(trained_since=15, publish_every=16)
    ) is True
    assert ValidationService._publication_due_next_half(
        _stub(trained_since=3, publish_every=16)
    ) is False


def test_detached_due_only_when_staged(monkeypatch):
    monkeypatch.setattr(C, "DETACHED_TRAINER", True)

    class _Intake:
        staged_ready = False

    # Cadence counters are irrelevant in detached mode.
    stub = _stub(intake=_Intake(), trained_since=15)
    assert ValidationService._publication_due_next_half(stub) is False
    _Intake.staged_ready = True
    assert ValidationService._publication_due_next_half(stub) is True


def test_detached_no_intake_yet(monkeypatch):
    monkeypatch.setattr(C, "DETACHED_TRAINER", True)
    assert ValidationService._publication_due_next_half(
        _stub(intake=None, trained_since=15)
    ) is False
