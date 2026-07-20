"""Cryptographic signature functions for GRAIL protocol."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bittensor as bt
else:
    try:
        import bittensor as bt
    except ImportError:
        bt = None  # type: ignore

from reliquary.protocol.tokens import hash_tokens
from reliquary.constants import GRAIL_PROOF_VERSION

logger = logging.getLogger(__name__)

COMMIT_DOMAIN = b"grail-commit-v1"

# Domain separation tag for the per-request envelope signature. Distinct
# from ``COMMIT_DOMAIN`` so a per-rollout commit signature can never be
# replayed as an envelope signature (or vice versa). Bumping the v1 suffix
# is reserved for breaking changes to the envelope field set; rollout
# clients are expected to construct the envelope with the same byte layout
# the validator expects.
ENVELOPE_DOMAIN = b"reliquary-envelope-v1"
PRECOMMIT_DOMAIN = b"reliquary-upload-precommit-v2"


def hash_commitments(commitments: list[dict]) -> bytes:
    """Return SHA-256 over a canonical JSON encoding of proof commitments."""
    try:
        payload = json.dumps(commitments, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).digest()
    except Exception as e:
        logger.warning("Failed to hash commitments: %s", e)
        return hashlib.sha256(b"").digest()


def build_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    commitments: list[dict],
) -> bytes:
    """Build domain-separated commit binding to be signed.

    Format: SHA256(COMMIT_DOMAIN || len(x)||x for each x in
    [tokens_hash, rand_bytes, model_name_bytes, layer_index_be, commitments_hash]).
    """

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    rand_clean = randomness_hex.strip().replace("0x", "").replace("0X", "")
    if len(rand_clean) % 2 != 0:
        rand_clean = "0" + rand_clean
    rand_bytes = bytes.fromhex(rand_clean)

    tokens_h = hash_tokens(tokens)
    commitments_h = hash_commitments(commitments)
    model_b = (model_name or "").encode("utf-8")
    layer_b = int(layer_index).to_bytes(4, "big", signed=True)

    h = hashlib.sha256()
    h.update(COMMIT_DOMAIN)
    for part in (tokens_h, rand_bytes, model_b, layer_b, commitments_h):
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def sign_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    commitments: list[dict],
    wallet: bt.Wallet,  # type: ignore[misc]
) -> bytes:
    """Sign the commit-binding message with wallet hotkey."""
    if bt is None:
        raise ImportError("bittensor is required for sign_commit_binding")

    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")

    msg = build_commit_binding(tokens, randomness_hex, model_name, layer_index, commitments)
    return wallet.hotkey.sign(msg)  # type: ignore[union-attr]


def verify_commit_signature(commit: dict, wallet_address: str) -> bool:
    """Verify commit signature binding tokens, randomness, model, layer, and proofs."""
    if bt is None:
        raise ImportError("bittensor is required for verify_commit_signature")

    try:
        sig = bytes.fromhex(commit["signature"])
        proof_version = commit.get("proof_version")

        if not proof_version or proof_version != GRAIL_PROOF_VERSION:
            logger.debug("Invalid proof version: %s", proof_version)
            return False

        tokens = commit["tokens"]
        commitments = commit["commitments"]
        beacon = commit.get("beacon", {})
        randomness = beacon["randomness"]
        model_info = commit.get("model", {})
        model_name = model_info.get("name", "")
        layer_index = int(model_info.get("layer_index"))

        msg = build_commit_binding(tokens, randomness, model_name, layer_index, commitments)

        keypair = bt.Keypair(ss58_address=wallet_address)
        return keypair.verify(data=msg, signature=sig)  # type: ignore[union-attr,return-value]
    except Exception as e:
        logger.debug("Signature verification failed: %s", e)
        return False


def build_envelope_binding(
    *,
    miner_hotkey: str,
    window_start: int,
    prompt_idx: int,
    merkle_root: str,
    checkpoint_hash: str,
    drand_round: int,
    randomness: str,
    nonce: str,
) -> bytes:
    """Build the canonical message bytes signed by the miner over the
    ``BatchSubmissionRequest`` envelope.

    Domain-separated under ``ENVELOPE_DOMAIN``. Every field that the
    validator routes on must be bound here so the signature attests to
    the COMPLETE intent of the submission:

      * ``miner_hotkey``  — who claims to be sending (the verified signer
        must equal this exact ss58)
      * ``window_start``  — the batcher window the submission targets,
        bounding the signature's validity to one window
      * ``prompt_idx``    — which env prompt this batch is for
      * ``merkle_root``   — the per-batch GRAIL merkle root
      * ``checkpoint_hash`` — the model revision the miner ran
      * ``drand_round``   — the drand quicknet round the miner attached
      * ``randomness``    — the validator-published window randomness the
        miner's GRAIL sketches are derived against (binds the signature
        to a specific validator's window so it can't be replayed cross-
        chain or against a forked validator)
      * ``nonce``         — caller-chosen freshness token; the validator
        does not (currently) dedupe on it but it prevents an attacker
        from precomputing a signature without knowing the miner's
        intended payload

    Layout: ``SHA256(ENVELOPE_DOMAIN || len(x)||x for each x in
    [hotkey_bytes, window_be, prompt_be, merkle_bytes, ckpt_bytes,
     round_be, rand_bytes, nonce_bytes])``.

    Same length-prefix-then-bytes pattern as ``build_commit_binding`` so
    field boundaries are unambiguous and no extension attack is possible.
    """

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    hotkey_b = miner_hotkey.encode("utf-8")
    # Use 8-byte big-endian for the integer fields — comfortable headroom
    # over any expected window/prompt/round magnitude.
    window_b = int(window_start).to_bytes(8, "big", signed=False)
    prompt_b = int(prompt_idx).to_bytes(8, "big", signed=False)
    round_b = int(drand_round).to_bytes(8, "big", signed=False)

    # Hex-string fields: accept with or without ``0x`` prefix. Empty
    # string is permitted for ``checkpoint_hash`` (bootstrap sentinel)
    # and ``randomness`` (pre-OPEN windows in tests).
    def _hex_bytes(s: str) -> bytes:
        clean = (s or "").strip().replace("0x", "").replace("0X", "")
        if len(clean) % 2 != 0:
            clean = "0" + clean
        if not clean:
            return b""
        return bytes.fromhex(clean)

    merkle_b = _hex_bytes(merkle_root)
    ckpt_b = (checkpoint_hash or "").encode("utf-8")  # opaque string, not always hex
    rand_b = _hex_bytes(randomness)
    nonce_b = (nonce or "").encode("utf-8")

    h = hashlib.sha256()
    h.update(ENVELOPE_DOMAIN)
    for part in (hotkey_b, window_b, prompt_b, merkle_b, ckpt_b, round_b, rand_b, nonce_b):
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def sign_envelope(
    *,
    wallet,
    miner_hotkey: str,
    window_start: int,
    prompt_idx: int,
    merkle_root: str,
    checkpoint_hash: str,
    drand_round: int,
    randomness: str,
    nonce: str,
) -> bytes:
    """Sign the canonical envelope binding with the miner's hotkey keypair.

    Caller is expected to set ``miner_hotkey`` to ``wallet.hotkey.ss58_address``;
    the binding intentionally includes the hotkey so a stolen signature
    can't be reattributed to a different signer.
    """
    if bt is None:
        raise ImportError("bittensor is required for sign_envelope")
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")

    msg = build_envelope_binding(
        miner_hotkey=miner_hotkey,
        window_start=window_start,
        prompt_idx=prompt_idx,
        merkle_root=merkle_root,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        randomness=randomness,
        nonce=nonce,
    )
    return wallet.hotkey.sign(msg)  # type: ignore[union-attr]


def verify_envelope_signature(
    *,
    miner_hotkey: str,
    window_start: int,
    prompt_idx: int,
    merkle_root: str,
    checkpoint_hash: str,
    drand_round: int,
    randomness: str,
    nonce: str,
    envelope_signature: str,
) -> bool:
    """Verify ``envelope_signature`` is a valid sr25519 sig of the canonical
    binding under the ``miner_hotkey`` public key.

    Returns ``False`` on any failure (bad hex, key parse error, sig
    mismatch, missing bittensor). Callers wrap this in a fast-reject
    that records ``BAD_ENVELOPE_SIGNATURE`` and never touches the
    per-hotkey rate-limit counter — the whole point of the binding.
    """
    if bt is None:
        # In test/CI envs without bittensor we cannot verify. Fail-closed.
        logger.debug("verify_envelope_signature: bittensor unavailable")
        return False
    if not envelope_signature:
        return False
    try:
        sig_bytes = bytes.fromhex(envelope_signature)
    except ValueError:
        logger.debug("verify_envelope_signature: signature not valid hex")
        return False
    try:
        msg = build_envelope_binding(
            miner_hotkey=miner_hotkey,
            window_start=window_start,
            prompt_idx=prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=checkpoint_hash,
            drand_round=drand_round,
            randomness=randomness,
            nonce=nonce,
        )
        keypair = bt.Keypair(ss58_address=miner_hotkey)  # type: ignore[union-attr]
        return bool(keypair.verify(data=msg, signature=sig_bytes))
    except Exception as e:
        logger.debug("envelope signature verify failed: %s", e)
        return False


def build_precommit_binding(
    *,
    miner_hotkey: str,
    window_start: int,
    prompt_idx: int,
    merkle_root: str,
    checkpoint_hash: str,
    environment: str,
    payload_bytes: int,
    payload_sha256: str,
    drand_round: int,
    randomness: str,
    protocol_version: int,
    nonce: str,
) -> bytes:
    """Build the domain-separated upload-precommit message.

    This repeats the envelope's routing identity and additionally binds the
    environment, exact serialized byte count and digest, and protocol version.
    The validator can therefore reserve a short reveal grace without trusting
    an unsigned header or any miner-controlled arrival timestamp, and the miner
    cannot replace a committed body with a same-sized reveal.
    """

    def _lp(value: bytes) -> bytes:
        return len(value).to_bytes(4, "big") + value

    def _hex_bytes(value: str) -> bytes:
        clean = (value or "").strip().replace("0x", "").replace("0X", "")
        if len(clean) % 2:
            clean = "0" + clean
        return bytes.fromhex(clean) if clean else b""

    fields = (
        miner_hotkey.encode("utf-8"),
        int(window_start).to_bytes(8, "big", signed=False),
        int(prompt_idx).to_bytes(8, "big", signed=False),
        _hex_bytes(merkle_root),
        (checkpoint_hash or "").encode("utf-8"),
        environment.encode("utf-8"),
        int(payload_bytes).to_bytes(8, "big", signed=False),
        _hex_bytes(payload_sha256),
        int(drand_round).to_bytes(8, "big", signed=False),
        _hex_bytes(randomness),
        int(protocol_version).to_bytes(8, "big", signed=False),
        nonce.encode("utf-8"),
    )
    h = hashlib.sha256()
    h.update(PRECOMMIT_DOMAIN)
    for field in fields:
        h.update(_lp(field))
    return h.digest()


def sign_precommit(*, wallet, **binding_fields) -> bytes:
    """Sign a canonical upload precommit with the miner hotkey."""
    if bt is None:
        raise ImportError("bittensor is required for sign_precommit")
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")
    return wallet.hotkey.sign(build_precommit_binding(**binding_fields))


def verify_precommit_signature(
    *, precommit_signature: str, **binding_fields
) -> bool:
    """Verify a signed upload precommit, failing closed on malformed input."""
    if bt is None or not precommit_signature:
        return False
    try:
        signature = bytes.fromhex(precommit_signature)
        message = build_precommit_binding(**binding_fields)
        keypair = bt.Keypair(  # type: ignore[union-attr]
            ss58_address=binding_fields["miner_hotkey"]
        )
        return bool(keypair.verify(data=message, signature=signature))
    except Exception as exc:
        logger.debug("precommit signature verify failed: %s", exc)
        return False
