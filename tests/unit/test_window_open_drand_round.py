"""``window_open_drand_round`` observability must match the window wall clock."""

from types import SimpleNamespace

import pytest

from reliquary.infrastructure.chain import compute_current_drand_round
from reliquary.validator.batcher import GrpoWindowBatcher

GENESIS = 1_700_000_000
PERIOD = 3
CHAIN = {"genesis_time": GENESIS, "period": PERIOD}


class _FakeEnv:
    name = "openmathinstruct"

    def __len__(self):
        return 1000

    def get_problem(self, idx):
        return {"prompt": "p", "ground_truth": "1", "id": "x"}


def _batcher(wall_clock, chain=CHAIN):
    return GrpoWindowBatcher(
        window_start=100,
        env=_FakeEnv(),
        model=SimpleNamespace(),
        completion_text_fn=lambda r: "",
        verify_commitment_proofs_fn=lambda *a, **k: True,
        verify_signature_fn=lambda c, h: True,
        wall_clock_fn=lambda: wall_clock,
        drand_chain_info=chain,
        drand_round_check_enabled=False,
    )


def test_window_open_round_is_populated_on_open():
    """The anchor is None until the window opens, then holds the drand round
    matching the open wall-clock."""
    opened_at = GENESIS + 3000 * PERIOD
    expected = compute_current_drand_round(opened_at, GENESIS, PERIOD)
    b = _batcher(opened_at)
    assert b.window_open_drand_round is None

    b.mark_window_opened()

    # Asserted against the protocol function, not a hand-computed number: drand
    # rounds are 1-indexed (round 1 publishes at genesis).
    assert b.window_open_drand_round == expected


def test_window_open_round_tracks_wall_clock():
    """A window opening one round later anchors one round later — elapsed is
    measured from a real moving reference, not a constant."""
    a = _batcher(GENESIS + 3000 * PERIOD)
    c = _batcher(GENESIS + 3001 * PERIOD)
    a.mark_window_opened()
    c.mark_window_opened()
    assert c.window_open_drand_round - a.window_open_drand_round == 1


def test_window_open_round_degrades_to_none_on_drand_failure(monkeypatch):
    """Any drand hiccup leaves the optional observability anchor unset."""
    b = _batcher(GENESIS + 3000 * PERIOD, chain=None)

    def _boom():
        raise RuntimeError("drand unavailable")

    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain", _boom
    )
    b.mark_window_opened()
    assert b.window_open_drand_round is None
