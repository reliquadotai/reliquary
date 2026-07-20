"""Tests for deterministic beacon round selection.

Miner and validator must agree on the same drand round for a given window.
The round is derived from the window_start block number, not fetched as 'latest'.
"""

class TestDeterministicBeaconRound:
    def test_compute_window_randomness_includes_round(self):
        """Window randomness must bind the drand round number to prevent
        a miner from choosing a favorable round."""
        from reliquary.infrastructure.chain import compute_window_randomness

        block_hash = "aa" * 32
        drand_rand = "bb" * 32

        r1 = compute_window_randomness(block_hash, drand_rand, drand_round=100)
        r2 = compute_window_randomness(block_hash, drand_rand, drand_round=101)
        r_no_round = compute_window_randomness(block_hash, drand_rand, drand_round=None)

        assert r1 != r2, "Different rounds must produce different randomness"
        assert r1 != r_no_round, "Providing a round must change the result"

    def test_compute_window_randomness_drand_only(self):
        """v2.3+: block_hash may be None — drand-only seed derivation."""
        from reliquary.infrastructure.chain import compute_window_randomness

        drand_rand = "bb" * 32
        r_no_block = compute_window_randomness(None, drand_rand, drand_round=42)
        r_with_block = compute_window_randomness("aa" * 32, drand_rand, drand_round=42)

        # Both forms valid, but produce different seeds.
        assert isinstance(r_no_block, str) and len(r_no_block) == 64
        assert r_no_block != r_with_block

        # Drand-only is still round-bound.
        r_other_round = compute_window_randomness(None, drand_rand, drand_round=43)
        assert r_no_block != r_other_round

    def test_compute_window_randomness_requires_some_source(self):
        """At least one of block_hash or drand_randomness must be provided."""
        import pytest

        from reliquary.infrastructure.chain import compute_window_randomness

        with pytest.raises(ValueError):
            compute_window_randomness(None, None, drand_round=42)


class TestCurrentDrandRound:
    def test_current_round_at_timestamp(self):
        """Round in progress at t is `1 + (t - genesis) // period`."""
        from reliquary.infrastructure.chain import compute_current_drand_round

        # genesis=1000, period=3. At t=1000 → round 1. At t=1003 → round 2.
        assert compute_current_drand_round(1000, 1000, 3) == 1
        assert compute_current_drand_round(1002.99, 1000, 3) == 1
        assert compute_current_drand_round(1003, 1000, 3) == 2
        assert compute_current_drand_round(1006, 1000, 3) == 3

    def test_current_round_before_genesis_clamps_to_1(self):
        from reliquary.infrastructure.chain import compute_current_drand_round
        assert compute_current_drand_round(900, 1000, 3) == 1

    def test_current_round_accepts_float_timestamp(self):
        """Validators use time.time() — must accept floats."""
        from reliquary.infrastructure.chain import compute_current_drand_round
        # Subsecond difference must not advance the round.
        r1 = compute_current_drand_round(1001.0, 1000, 3)
        r2 = compute_current_drand_round(1001.999, 1000, 3)
        assert r1 == r2 == 1


class TestNextDrandBoundary:
    def test_at_boundary_returns_zero(self):
        """At t = genesis_time + N*period exactly, no wait."""
        from reliquary.infrastructure.chain import seconds_until_next_drand_boundary
        assert seconds_until_next_drand_boundary(1000, 1000, 3) == 0.0
        assert seconds_until_next_drand_boundary(1003, 1000, 3) == 0.0
        assert seconds_until_next_drand_boundary(1006, 1000, 3) == 0.0

    def test_mid_round_returns_remaining(self):
        from reliquary.infrastructure.chain import seconds_until_next_drand_boundary
        # 1s into a 3s round → 2s remaining.
        assert seconds_until_next_drand_boundary(1001, 1000, 3) == 2
        # 2s into a 3s round → 1s remaining.
        assert seconds_until_next_drand_boundary(1002, 1000, 3) == 1
        # 0.5s into a 3s round → 2.5s remaining.
        assert abs(seconds_until_next_drand_boundary(1000.5, 1000, 3) - 2.5) < 1e-9

    def test_before_genesis_waits_for_genesis(self):
        from reliquary.infrastructure.chain import seconds_until_next_drand_boundary
        assert seconds_until_next_drand_boundary(950, 1000, 3) == 50.0
