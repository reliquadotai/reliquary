"""Bounded recovery from a stale HTTP keep-alive connection."""
import httpx
import pytest

from reliquary.validator.remote_proof import RemoteProofPool
from reliquary.validator.proof_worker import ProofWorkerUnavailable


def test_health_transport_retry_is_bounded():
    calls = []
    failures = 1

    def respond(request):
        calls.append(request)
        if len(calls) <= failures:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response")
        return httpx.Response(200, content=b"healthy")

    pool = object.__new__(RemoteProofPool)
    with httpx.Client(transport=httpx.MockTransport(respond), base_url="https://worker") as client:
        pool._client = client
        assert pool._request("GET", "/v1/health", timeout=1) == b"healthy"
        assert len(calls) == 2
        calls.clear()
        failures = 10
        with pytest.raises(ProofWorkerUnavailable):
            pool._request("GET", "/v1/health", timeout=1)
        assert len(calls) == 2
        calls.clear()
        with pytest.raises(ProofWorkerUnavailable):
            pool._request("POST", "/v1/adopt", timeout=1)
        assert len(calls) == 1
