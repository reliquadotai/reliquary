"""Shared fixtures for ``tests/unit``.

The envelope-signature gate (introduced 2026-05 to close the
hotkey-spoof DoS on ``/submit``) defaults to ENABLED in production. The
vast majority of legacy unit tests pre-date the gate and exercise other
validator behaviour with unsigned synthetic requests — they have no
need to construct sr25519 signatures on every call. To keep them
unchanged we disable the gate at module-import time for this directory
via this autouse session-scoped fixture.

Tests that specifically exercise the gate (``test_envelope_signature.py``)
opt back into enforcement using the ``enforce_envelope`` fixture
defined here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_envelope_enforcement_by_default():
    """Disable envelope-signature enforcement for the whole unit test
    session unless an individual test overrides it.

    Patches the validator-server module's module-level constant — that
    binding is what the ``/submit`` handler reads. Patching the
    ``reliquary.constants`` value alone would not flip the in-handler
    behaviour because the validator imports the name eagerly at import
    time.
    """
    import reliquary.validator.server as server_mod
    original = getattr(server_mod, "ENFORCE_ENVELOPE_SIGNATURE", True)
    server_mod.ENFORCE_ENVELOPE_SIGNATURE = False
    try:
        yield
    finally:
        server_mod.ENFORCE_ENVELOPE_SIGNATURE = original


@pytest.fixture
def enforce_envelope(monkeypatch):
    """Opt back into envelope-signature enforcement for one test.

    Use this in tests that explicitly verify the
    ``BAD_ENVELOPE_SIGNATURE`` path or the rate-limit-counter
    no-touch-on-bad-sig invariant. The autouse fixture above turned
    enforcement OFF for the session; this fixture flips it back ON for
    the duration of the requesting test only.
    """
    import reliquary.validator.server as server_mod
    monkeypatch.setattr(server_mod, "ENFORCE_ENVELOPE_SIGNATURE", True)
    return None


@pytest.fixture(autouse=True)
def _whole_split_rl_prompt_ranges(request, monkeypatch):
    """Offer RL every row of every environment unless a test asks otherwise.

    Fake environments here carry real names (``reliquary_code_v1``) with a few
    rows, and the real table refuses a split of another length than the one
    its slice was cut against. Tests of the table itself request
    ``real_rl_prompt_ranges``.
    """
    if "real_rl_prompt_ranges" in request.fixturenames:
        return
    import reliquary.protocol.rl_prompt_ranges as ranges
    monkeypatch.setattr(ranges, "RL_PROMPT_RANGES", {})


@pytest.fixture
def real_rl_prompt_ranges():
    """Keep the production RL prompt ranges for this test."""
    return None
