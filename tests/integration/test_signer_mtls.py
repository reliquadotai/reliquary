"""Real TLS handshake and checkpoint-signature round trip for signer-01."""

from __future__ import annotations

import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import bittensor as bt
import httpx
import pytest
import uvicorn

from reliquary.signer.backend import ChainResult
from reliquary.signer.client import RemoteSignerClient
from reliquary.signer.journal import SignerJournal
from reliquary.signer.service import SignerPolicy, create_signer_app


class KeypairBackend:
    def __init__(self) -> None:
        mnemonic = bt.Keypair.generate_mnemonic()
        self.keypair = bt.Keypair.create_from_mnemonic(mnemonic)
        self.hotkey_address = self.keypair.ss58_address

    def sign_checkpoint(self, payload: bytes) -> bytes:
        return bytes(self.keypair.sign(payload))

    async def set_weights(self, **_kwargs) -> ChainResult:
        return ChainResult(accepted=True, message="test-only")

    async def serve_axon(self, **_kwargs) -> ChainResult:
        return ChainResult(accepted=True, message="test-only")


@pytest.mark.skipif(not Path("/usr/bin/openssl").exists(), reason="openssl required")
def test_mtls_required_and_remote_checkpoint_signature_verified(tmp_path):
    repo_dir = Path(__file__).resolve().parents[2]
    pki_dir = tmp_path / "pki"
    subprocess.run(
        [
            str(repo_dir / "scripts" / "generate_signer_pki.sh"),
            str(pki_dir),
            "127.0.0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    backend = KeypairBackend()
    journal = SignerJournal(str(tmp_path / "signer.sqlite3"))
    app = create_signer_app(
        policy=SignerPolicy(
            network="finney",
            netuid=81,
            repo_id="aivolutionedge/reliquary-sn",
        ),
        backend=backend,
        journal=journal,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        app,
        log_level="error",
        access_log=False,
        server_header=False,
        ssl_certfile=str(pki_dir / "signer" / "server.crt"),
        ssl_keyfile=str(pki_dir / "signer" / "server.key"),
        ssl_ca_certs=str(pki_dir / "signer" / "ca.crt"),
        ssl_cert_reqs=ssl.CERT_REQUIRED,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    try:
        unauthenticated_context = ssl.create_default_context(
            cafile=str(pki_dir / "signer" / "ca.crt")
        )
        with pytest.raises(httpx.HTTPError):
            httpx.get(
                f"https://127.0.0.1:{port}/readyz",
                verify=unauthenticated_context,
                timeout=2.0,
                trust_env=False,
            )

        client = RemoteSignerClient(
            base_url=f"https://127.0.0.1:{port}",
            ca_path=str(pki_dir / "signer-client" / "ca.crt"),
            cert_path=str(pki_dir / "signer-client" / "client.crt"),
            key_path=str(pki_dir / "signer-client" / "client.key"),
            expected_hotkey=backend.hotkey_address,
            network="finney",
            netuid=81,
            repo_id="aivolutionedge/reliquary-sn",
        )
        health = client.assert_ready()
        assert health.signer_hotkey == backend.hotkey_address
        signature = client.sign_checkpoint(
            checkpoint_n=1,
            repo_id="aivolutionedge/reliquary-sn",
            revision="a" * 40,
        )
        assert backend.keypair.verify(
            data=f"1|{'a' * 40}".encode(), signature=signature
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        journal.close()
    assert not thread.is_alive()
