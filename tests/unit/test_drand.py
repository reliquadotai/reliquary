from reliquary.infrastructure.drand import get_mock_beacon, get_beacon


class TestMockBeacon:
    def test_has_required_fields(self):
        b = get_mock_beacon()
        assert "round" in b
        assert "randomness" in b
        assert isinstance(b["randomness"], str)

    def test_incrementing_rounds(self):
        a = get_mock_beacon()
        b = get_mock_beacon()
        assert b["round"] > a["round"]

    def test_randomness_is_hex(self):
        b = get_mock_beacon()
        bytes.fromhex(b["randomness"])  # Should not raise

    def test_randomness_is_64_chars(self):
        b = get_mock_beacon()
        assert len(b["randomness"]) == 64


class TestGetBeacon:
    def test_mock_mode(self):
        b = get_beacon(use_drand=False)
        assert "randomness" in b
        assert b["source"] == "mock"

    def test_mock_has_all_fields(self):
        b = get_beacon(use_drand=False)
        assert "round" in b
        assert "randomness" in b
        assert "source" in b


def test_get_drand_beacon_does_not_call_verify_synchronously(monkeypatch):
    """The bittensor_drand cross-check is moved to a background task on
    the validator side (Task 5). The fetch path must not block on it
    anymore. Miner callers don't need cross-check (a bad relay just
    makes their commitments fail GRAIL on validator-side).
    """
    from unittest.mock import MagicMock
    from reliquary.infrastructure import drand as D

    verify_mock = MagicMock(return_value=True)
    monkeypatch.setattr(D, "verify_beacon_signature", verify_mock)

    # Stub _http_get_json to return a synthetic valid beacon.
    payload = {
        "round": 12345,
        "signature": "ab" * 48,
    }
    monkeypatch.setattr(D, "_http_get_json", lambda paths: payload)
    monkeypatch.setattr(D, "_DRAND_CHAIN_HASH", "test_hash")
    monkeypatch.setattr(D, "_DRAND_PERIOD", 3)

    result = D.get_drand_beacon(round_id=12345)
    assert result["round"] == 12345
    assert result["signature"] == "ab" * 48
    # randomness was derived locally via SHA256(sig).
    import hashlib
    assert result["randomness"] == hashlib.sha256(bytes.fromhex("ab" * 48)).hexdigest()
    # The expensive cross-check MUST NOT have been called on the hot path.
    assert verify_mock.call_count == 0, (
        "get_drand_beacon called verify_beacon_signature synchronously — "
        "moves it back onto the critical path (cost: ~700-1000ms p50)."
    )
