"""Trusted control-plane client for the semantic signer API."""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass

import httpx

from reliquary.signer.protocol import (
    SIGNER_PROTOCOL_VERSION,
    CheckpointSignRequest,
    CheckpointSignResponse,
    ServeAxonRequest,
    ServeAxonResponse,
    SetWeightsRequest,
    SetWeightsResponse,
    SignerHealth,
    axon_operation_id,
    checkpoint_operation_id,
    checkpoint_payload,
    weights_operation_id,
)


@dataclass(frozen=True)
class _PublicHotkey:
    ss58_address: str

    def sign(self, _data: bytes) -> bytes:
        raise RuntimeError("private signing is available only through signer-01")


@dataclass(frozen=True)
class PublicWalletIdentity:
    hotkey: _PublicHotkey

    @classmethod
    def from_address(cls, address: str) -> "PublicWalletIdentity":
        return cls(hotkey=_PublicHotkey(ss58_address=address))


class RemoteSignerClient:
    def __init__(
        self,
        *,
        base_url: str,
        ca_path: str,
        cert_path: str,
        key_path: str,
        expected_hotkey: str,
        network: str,
        netuid: int,
        repo_id: str,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ValueError("remote signer requires an https:// URL")
        self.base_url = base_url.rstrip("/")
        self.ca_path = ca_path
        self.cert = (cert_path, key_path)
        self.expected_hotkey = expected_hotkey
        self.network = network
        self.netuid = int(netuid)
        self.repo_id = repo_id
        self.public_wallet = PublicWalletIdentity.from_address(expected_hotkey)
        self._tls_context = ssl.create_default_context(cafile=ca_path)
        self._tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._tls_context.load_cert_chain(certfile=cert_path, keyfile=key_path)

    @classmethod
    def from_environment(
        cls, *, network: str, netuid: int, repo_id: str
    ) -> "RemoteSignerClient":
        required = {
            "base_url": "RELIQUARY_SIGNER_URL",
            "ca_path": "RELIQUARY_SIGNER_CA",
            "cert_path": "RELIQUARY_SIGNER_CERT",
            "key_path": "RELIQUARY_SIGNER_KEY",
            "expected_hotkey": "RELIQUARY_SIGNER_EXPECTED_HOTKEY",
        }
        values: dict[str, str] = {}
        for field, environment_name in required.items():
            value = os.environ.get(environment_name, "").strip()
            if not value:
                raise RuntimeError(
                    f"{environment_name} is required for remote signer mode"
                )
            values[field] = value
        return cls(network=network, netuid=netuid, repo_id=repo_id, **values)

    def _check_hotkey(self, actual: str) -> None:
        if actual != self.expected_hotkey:
            raise RuntimeError("signer response came from an unexpected hotkey")

    def assert_ready(self) -> SignerHealth:
        with httpx.Client(
            verify=self._tls_context,
            timeout=10.0,
            trust_env=False,
        ) as client:
            response = client.get(f"{self.base_url}/readyz")
            response.raise_for_status()
        health = SignerHealth.model_validate(response.json())
        self._check_hotkey(health.signer_hotkey)
        if (
            health.protocol_version != SIGNER_PROTOCOL_VERSION
            or health.network != self.network
            or health.netuid != self.netuid
            or health.repo_id != self.repo_id
        ):
            raise RuntimeError(
                "signer readiness contract does not match control configuration"
            )
        return health

    def sign_checkpoint(
        self, *, checkpoint_n: int, repo_id: str, revision: str
    ) -> bytes:
        if repo_id != self.repo_id:
            raise ValueError("checkpoint repository differs from signer policy")
        operation = checkpoint_operation_id(
            netuid=self.netuid,
            checkpoint_n=checkpoint_n,
            repo_id=repo_id,
            revision=revision,
        )
        request = CheckpointSignRequest(
            operation_id=operation,
            netuid=self.netuid,
            checkpoint_n=checkpoint_n,
            repo_id=repo_id,
            revision=revision,
        )
        with httpx.Client(
            verify=self._tls_context,
            timeout=15.0,
            trust_env=False,
        ) as client:
            response = client.post(
                f"{self.base_url}/v1/checkpoints/sign",
                json=request.model_dump(),
            )
            response.raise_for_status()
        result = CheckpointSignResponse.model_validate(response.json())
        self._check_hotkey(result.signer_hotkey)
        if result.operation_id != operation:
            raise RuntimeError("signer checkpoint response is not bound to the request")
        signature = bytes.fromhex(result.signature_hex)
        import bittensor as bt

        verifier = bt.Keypair(ss58_address=self.expected_hotkey)
        if not verifier.verify(
            data=checkpoint_payload(checkpoint_n, revision),
            signature=signature,
        ):
            raise RuntimeError("remote checkpoint signature verification failed")
        return signature

    async def set_weights(
        self,
        *,
        epoch_id: int,
        netuid: int,
        uids: list[int],
        weights: list[float],
    ) -> bool:
        if int(netuid) != self.netuid:
            raise ValueError("weight netuid differs from signer policy")
        ordered = sorted(zip(uids, weights), key=lambda item: int(item[0]))
        ordered_uids = [int(uid) for uid, _ in ordered]
        ordered_weights = [float(weight) for _, weight in ordered]
        operation = weights_operation_id(
            netuid=netuid,
            epoch_id=epoch_id,
            uids=ordered_uids,
            weights=ordered_weights,
        )
        request = SetWeightsRequest(
            operation_id=operation,
            netuid=netuid,
            epoch_id=epoch_id,
            uids=ordered_uids,
            weights=ordered_weights,
        )
        timeout = httpx.Timeout(250.0, connect=10.0)
        async with httpx.AsyncClient(
            verify=self._tls_context,
            timeout=timeout,
            trust_env=False,
        ) as client:
            response = await client.post(
                f"{self.base_url}/v1/weights/set", json=request.model_dump()
            )
            response.raise_for_status()
        result = SetWeightsResponse.model_validate(response.json())
        self._check_hotkey(result.signer_hotkey)
        if result.operation_id != operation:
            raise RuntimeError("signer weight response is not bound to the request")
        return result.accepted

    async def serve_axon(self, *, netuid: int, ip: str, port: int) -> bool:
        if int(netuid) != self.netuid:
            raise ValueError("axon netuid differs from signer policy")
        operation = axon_operation_id(netuid=netuid, ip=ip, port=port)
        request = ServeAxonRequest(
            operation_id=operation,
            netuid=netuid,
            ip=ip,
            port=port,
        )
        timeout = httpx.Timeout(250.0, connect=10.0)
        async with httpx.AsyncClient(
            verify=self._tls_context,
            timeout=timeout,
            trust_env=False,
        ) as client:
            response = await client.post(
                f"{self.base_url}/v1/axon/serve", json=request.model_dump()
            )
            response.raise_for_status()
        result = ServeAxonResponse.model_validate(response.json())
        self._check_hotkey(result.signer_hotkey)
        if result.operation_id != operation:
            raise RuntimeError("signer axon response is not bound to the request")
        return result.accepted
