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


def test_get_drand_beacon_verifies_locally_before_returning(monkeypatch):
    """Every fetched beacon is checked locally before it reaches a caller."""
    from unittest.mock import MagicMock
    from reliquary.infrastructure import drand as D

    verify_mock = MagicMock(return_value=True)
    monkeypatch.setattr(D, "verify_round_signature", verify_mock)

    # Stub _http_get_json to return a synthetic valid beacon.
    payload = {
        "round": 12345,
        "signature": "ab" * 48,
    }
    monkeypatch.setattr(D, "_http_get_json", lambda paths: payload)

    result = D.get_drand_beacon(round_id=12345)
    assert result["round"] == 12345
    assert result["signature"] == "ab" * 48
    # randomness was derived locally via SHA256(sig).
    import hashlib
    assert result["randomness"] == hashlib.sha256(bytes.fromhex("ab" * 48)).hexdigest()
    verify_mock.assert_called_once_with(12345, "ab" * 48)
