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
