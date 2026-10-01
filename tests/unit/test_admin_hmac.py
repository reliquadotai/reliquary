"""R4: every admin request is signed, fresh and its nonce used once."""

from __future__ import annotations

import hashlib
import hmac
import secrets

import pytest

from reliquary.admin.auth import MAX_SKEW_SECONDS, HmacVerifier, sign_request

SECRET = b"s3cret"


def _signed(now, method="POST", path="/admin/v1/jobs", body=b"{}", nonce=None):
    timestamp = str(int(now))
    nonce = nonce or secrets.token_hex(16)
    return timestamp, nonce, sign_request(SECRET, timestamp, nonce, method, path, body)


def test_the_signature_is_the_hmac_of_the_spec_message():
    body = b'{"a":1}'
    nonce = "0123456789abcdef"
    expected = hmac.new(
        SECRET,
        b"1700000000\n0123456789abcdef\nPOST\n/admin/v1/jobs\n"
        + hashlib.sha256(body).hexdigest().encode(),
        hashlib.sha256,
    ).hexdigest()
    assert sign_request(SECRET, "1700000000", nonce, "post", "/admin/v1/jobs", body) == expected


def test_a_fresh_signed_request_passes():
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0)
    timestamp, nonce, signature = _signed(1000)
    assert verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature, b"{}") is None


def test_two_identical_bodies_in_one_second_pass_under_two_nonces():
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0)
    for _ in range(2):
        timestamp, nonce, signature = _signed(1000)
        assert verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature,
                                b"{}") is None


def test_a_reused_nonce_is_refused_even_with_a_fresh_signature():
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0)
    timestamp, nonce, signature = _signed(1000, nonce="ab" * 8)
    assert verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature, b"{}") is None
    assert verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature,
                            b"{}") == "replayed"
    timestamp, nonce, signature = _signed(1001, path="/admin/v1/executors", nonce="AB" * 8)
    assert verifier.refusal("POST", "/admin/v1/executors", timestamp, nonce, signature,
                            b"{}") == "replayed"


@pytest.mark.parametrize("skew", [MAX_SKEW_SECONDS + 1, -(MAX_SKEW_SECONDS + 1)])
def test_a_stale_or_future_timestamp_is_refused(skew):
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0 + skew)
    timestamp, nonce, signature = _signed(1000)
    assert verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature,
                            b"{}") == "stale_timestamp"


def test_a_signature_over_other_bytes_path_method_or_nonce_is_refused():
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0)
    timestamp, nonce, signature = _signed(1000)
    for method, path, body, used in (("POST", "/admin/v1/jobs", b'{"x":1}', nonce),
                                     ("POST", "/admin/v1/executors", b"{}", nonce),
                                     ("DELETE", "/admin/v1/jobs", b"{}", nonce),
                                     ("POST", "/admin/v1/jobs", b"{}", "f" * 32)):
        assert verifier.refusal(method, path, timestamp, used, signature, body) == "bad_signature"


def test_a_missing_or_malformed_header_is_refused():
    verifier = HmacVerifier(SECRET, clock=lambda: 1000.0)
    nonce = "a" * 16
    assert verifier.refusal("GET", "/x", None, nonce, "ab", b"") == "missing_signature"
    assert verifier.refusal("GET", "/x", "1000", None, "ab", b"") == "missing_signature"
    assert verifier.refusal("GET", "/x", "1000", nonce, None, b"") == "missing_signature"
    assert verifier.refusal("GET", "/x", "soon", nonce, "ab", b"") == "stale_timestamp"
    for bad in ("a" * 15, "a" * 65, "z" * 16):
        assert verifier.refusal("GET", "/x", "1000", bad, "ab", b"") == "bad_nonce"


def test_the_nonce_cache_forgets_nonces_older_than_the_window():
    now = [1000.0]
    verifier = HmacVerifier(SECRET, clock=lambda: now[0])
    for i in range(50):
        timestamp, nonce, signature = _signed(1000, path=f"/x/{i}")
        assert verifier.refusal("POST", f"/x/{i}", timestamp, nonce, signature, b"{}") is None
    now[0] += 2 * MAX_SKEW_SECONDS + 1
    timestamp, nonce, signature = _signed(now[0])
    verifier.refusal("POST", "/admin/v1/jobs", timestamp, nonce, signature, b"{}")
    assert len(verifier._seen) == 1


def test_an_empty_secret_is_refused():
    with pytest.raises(ValueError):
        HmacVerifier(b"")
