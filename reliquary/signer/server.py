"""Production entry point for signer-01."""

from __future__ import annotations

import logging
import json
import os
import ssl
from pathlib import Path


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> None:
    logging.basicConfig(
        level=getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        ),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    from reliquary.signer.backend import BittensorSignerBackend
    from reliquary.signer.journal import SignerJournal
    from reliquary.signer.service import SignerPolicy, create_signer_app

    network = _required("BT_NETWORK")
    netuid = int(_required("BT_NETUID"))
    expected_hotkey = _required("RELIQUARY_SIGNER_EXPECTED_HOTKEY")
    policy = SignerPolicy(
        network=network,
        netuid=netuid,
        repo_id=_required("RELIQUARY_SIGNER_REPO_ID"),
        axon_ip=os.environ.get("RELIQUARY_SIGNER_AXON_IP", "").strip() or None,
        axon_port=(
            int(os.environ["RELIQUARY_SIGNER_AXON_PORT"])
            if os.environ.get("RELIQUARY_SIGNER_AXON_PORT", "").strip()
            else None
        ),
    )
    if (policy.axon_ip is None) != (policy.axon_port is None):
        raise RuntimeError("axon IP and port must be configured together")
    backend = BittensorSignerBackend(
        wallet_name=_required("BT_WALLET_NAME"),
        hotkey_name=_required("BT_HOTKEY"),
        wallet_path=_required("BT_WALLET_PATH"),
        network=network,
        expected_hotkey=expected_hotkey,
    )
    journal = SignerJournal(_required("RELIQUARY_SIGNER_JOURNAL"))
    app = create_signer_app(policy=policy, backend=backend, journal=journal)

    bind_ip = _required("RELIQUARY_SIGNER_HOST")
    port = int(os.environ.get("RELIQUARY_SIGNER_PORT", "8444"))
    cert = _required("RELIQUARY_SIGNER_TLS_CERT")
    key = _required("RELIQUARY_SIGNER_TLS_KEY")
    client_ca = _required("RELIQUARY_SIGNER_CLIENT_CA")

    health_file = Path(
        os.environ.get(
            "RELIQUARY_SIGNER_HEALTH_FILE",
            "/tmp/reliquary-signer-health.json",
        )
    )
    health_tmp = health_file.with_suffix(".tmp")
    health_tmp.write_text(
        json.dumps(
            {
                "status": "ready",
                "protocol_version": 1,
                "signer_hotkey": backend.hotkey_address,
                "network": network,
                "netuid": netuid,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.chmod(health_tmp, 0o600)
    os.replace(health_tmp, health_file)

    import uvicorn

    uvicorn.run(
        app,
        host=bind_ip,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=False,
        server_header=False,
        ssl_certfile=cert,
        ssl_keyfile=key,
        ssl_ca_certs=client_ca,
        ssl_cert_reqs=ssl.CERT_REQUIRED,
    )


if __name__ == "__main__":
    main()
