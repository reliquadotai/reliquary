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
from reliquary.constants import GRAIL_EPISODE_PROOF_VERSION, GRAIL_PROOF_VERSION

logger = logging.getLogger(__name__)

COMMIT_DOMAIN = b"grail-commit-v1"
EPISODE_COMMIT_DOMAIN = b"grail-commit-v8-episode"

# Domain separation tag for the per-request envelope signature. Distinct
# from ``COMMIT_DOMAIN`` so a per-rollout commit signature can never be
# replayed as an envelope signature (or vice versa). Bumping the v1 suffix
# is reserved for breaking changes to the envelope field set; rollout
# clients are expected to construct the envelope with the same byte layout
# the validator expects.
ENVELOPE_DOMAIN = b"reliquary-envelope-v1"
ENVELOPE_DOMAIN_V3 = b"reliquary-envelope-v3"
PRECOMMIT_DOMAIN = b"reliquary-upload-precommit-v2"
PRECOMMIT_DOMAIN_V3 = b"reliquary-upload-precommit-v3"


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


def build_episode_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    commitments: list[dict],
    episode: dict,
) -> bytes:
    """Bind the v7 proof material and the complete Episode v1 trace metadata."""

    base = build_commit_binding(
        tokens,
        randomness_hex,
        model_name,
        layer_index,
        commitments,
    )
    episode_bytes = json.dumps(
        episode,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    h = hashlib.sha256()
    h.update(EPISODE_COMMIT_DOMAIN)
    for part in (base, episode_bytes):
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def sign_episode_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    commitments: list[dict],
    episode: dict,
    wallet: bt.Wallet,  # type: ignore[misc]
) -> bytes:
    if bt is None:
        raise ImportError("bittensor is required for sign_episode_commit_binding")
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")
    return wallet.hotkey.sign(  # type: ignore[union-attr]
        build_episode_commit_binding(
            tokens,
            randomness_hex,
            model_name,
            layer_index,
            commitments,
            episode,
        )
    )


def verify_commit_signature(commit: dict, wallet_address: str) -> bool:
    """Verify commit signature binding tokens, randomness, model, layer, and proofs."""
    if bt is None:
        raise ImportError("bittensor is required for verify_commit_signature")

    try:
        sig = bytes.fromhex(commit["signature"])
        proof_version = commit.get("proof_version")

        if proof_version not in (
            GRAIL_PROOF_VERSION,
            GRAIL_EPISODE_PROOF_VERSION,
        ):
            logger.debug("Invalid proof version: %s", proof_version)
            return False

        tokens = commit["tokens"]
        commitments = commit["commitments"]
        beacon = commit.get("beacon", {})
        randomness = beacon["randomness"]
        model_info = commit.get("model", {})
        model_name = model_info.get("name", "")
        layer_index = int(model_info.get("layer_index"))

        if proof_version == GRAIL_EPISODE_PROOF_VERSION:
            episode = (commit.get("rollout") or {}).get("episode")
            if not isinstance(episode, dict):
                logger.debug("Episode v8 commit missing episode metadata")
                return False
            msg = build_episode_commit_binding(
                tokens,
                randomness,
                model_name,
                layer_index,
                commitments,
                episode,
            )
        else:
            msg = build_commit_binding(
                tokens,
                randomness,
                model_name,
                layer_index,
                commitments,
            )

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
    protocol_version: int = 0,
    generation_profile_id: str = "",
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

    profile_b = generation_profile_id.encode("utf-8")
    protocol_b = int(protocol_version).to_bytes(8, "big", signed=False)
    parts = (
        hotkey_b,
        window_b,
        prompt_b,
        merkle_b,
        ckpt_b,
        round_b,
        rand_b,
        nonce_b,
    )
    h = hashlib.sha256()
    if profile_b:
        h.update(ENVELOPE_DOMAIN_V3)
        parts = (*parts, protocol_b, profile_b)
    else:
        # Exact legacy binding: existing v2 signatures remain valid.
        h.update(ENVELOPE_DOMAIN)
    for part in parts:
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
    protocol_version: int = 0,
    generation_profile_id: str = "",
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
        protocol_version=protocol_version,
        generation_profile_id=generation_profile_id,
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
    protocol_version: int = 0,
    generation_profile_id: str = "",
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
            protocol_version=protocol_version,
            generation_profile_id=generation_profile_id,
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
    generation_profile_id: str = "",
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
    if generation_profile_id:
        h.update(PRECOMMIT_DOMAIN_V3)
        fields = (*fields, generation_profile_id.encode("utf-8"))
    else:
        # Exact legacy binding: existing v2 precommit signatures remain valid.
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


CORPUS_DOMAIN = b"reliquary/corpus-submission/v1"
# A trajectory is signed under its own domain, so its binding can never be
# replayed as a single-turn submission's (or the other way round).
CORPUS_TRAJECTORY_DOMAIN = b"reliquary/corpus-trajectory/v1"
# A trajectory that carries a sandbox transcript: its own domain, plus the transcript's
# digest as one more part. A trajectory without one binds exactly as before.
CORPUS_SIGNED_TRAJECTORY_DOMAIN = b"reliquary/corpus-trajectory-signed/v1"


def transcript_digest(transcript) -> bytes:
    return hashlib.sha256(json.dumps(transcript, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).digest()


def _trajectory_parts(trajectory) -> list[bytes]:
    """sha256 of the tokens, of the span list, of each span's proofs, of the
    final diff, and the stop label (spec §6)."""
    tokens = b"".join(int(t).to_bytes(4, "big", signed=False) for t in trajectory["tokens"])
    spans = b"".join(int(turn["start"]).to_bytes(8, "big", signed=False)
                     + int(turn["end"]).to_bytes(8, "big", signed=False)
                     for turn in trajectory["turns"])
    parts = [hashlib.sha256(tokens).digest(), hashlib.sha256(spans).digest()]
    for turn in trajectory["turns"]:
        proofs = b"".join(len(p.encode("ascii")).to_bytes(4, "big") + p.encode("ascii")
                          for p in turn.get("proofs") or [])
        parts.append(hashlib.sha256(proofs).digest())
    parts.append(hashlib.sha256(str(trajectory["final_diff"]).encode("utf-8")).digest())
    parts.append(str(trajectory["stop"]).encode("utf-8"))
    return parts


def _corpus_fields(request) -> dict:
    return request.model_dump() if hasattr(request, "model_dump") else dict(request)


def build_corpus_binding(request) -> bytes:
    """Digest of every field a corpus submission is admitted and paid on.

    The same length-prefixed layout as the envelope binding. Completions are
    bound in order, each as the digests of its tokens, text and proofs, so the
    binding stays small however long the completions are.
    """
    body = _corpus_fields(request)

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    def _sha(b: bytes) -> bytes:
        return hashlib.sha256(b).digest()

    parts = [
        str(body["job_id"]).encode("utf-8"),
        str(body["miner_hotkey"]).encode("utf-8"),
        int(body["cursor"]).to_bytes(8, "big", signed=False),
        int(body["prompt_index"]).to_bytes(8, "big", signed=False),
        str(body["checkpoint_sha256"]).encode("utf-8"),
        _sha(str(body["rendered_prompt"]).encode("utf-8")),
    ]
    trajectory = body.get("trajectory")
    if trajectory is not None:
        transcript = trajectory.get("transcript")
        domain = CORPUS_TRAJECTORY_DOMAIN if transcript is None else CORPUS_SIGNED_TRAJECTORY_DOMAIN
        parts += _trajectory_parts(trajectory)
        if transcript is not None:
            parts.append(transcript_digest(transcript))
    else:
        domain = CORPUS_DOMAIN
        for completion in body["completions"]:
            tokens = b"".join(int(t).to_bytes(4, "big", signed=False) for t in completion["tokens"])
            proofs = b"".join(
                _len_bytes(p.encode("ascii")) + p.encode("ascii")
                for p in completion.get("proofs") or []
            )
            parts += [_sha(tokens), _sha(str(completion["text"]).encode("utf-8")), _sha(proofs)]
    h = hashlib.sha256()
    h.update(domain)
    for part in parts:
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def corpus_submission_id(request) -> str:
    """The submission's storage key: its signed binding, so a resend is the same object."""
    return build_corpus_binding(request).hex()


def sign_corpus_submission(wallet, request) -> str:
    if bt is None:
        raise ImportError("bittensor is required for sign_corpus_submission")
    return wallet.hotkey.sign(build_corpus_binding(request)).hex()  # type: ignore[union-attr]


def verify_corpus_signature(request) -> bool:
    """False on any failure; fail-closed without bittensor, like the envelope."""
    if bt is None:
        logger.debug("verify_corpus_signature: bittensor unavailable")
        return False
    body = _corpus_fields(request)
    try:
        sig_bytes = bytes.fromhex(str(body.get("signature") or ""))
    except ValueError:
        return False
    if not sig_bytes:
        return False
    try:
        keypair = bt.Keypair(ss58_address=str(body["miner_hotkey"]))  # type: ignore[union-attr]
        return bool(keypair.verify(data=build_corpus_binding(request), signature=sig_bytes))
    except Exception as e:
        logger.debug("corpus signature verify failed: %s", e)
        return False


# Its own domain, so a skip signature never verifies as a submission's (or the
# other way round) even over the same job, hotkey, cursor and index.
CORPUS_SKIP_DOMAIN = b"reliquary/corpus-skip/v2"


def build_corpus_skip_binding(request) -> bytes:
    """Digest of a skip: the job, the hotkey, and the walk steps it gives up."""
    body = _corpus_fields(request)
    parts = [
        str(body["job_id"]).encode("utf-8"),
        str(body["miner_hotkey"]).encode("utf-8"),
        int(body["cursor"]).to_bytes(8, "big", signed=False),
        int(body["prompt_index"]).to_bytes(8, "big", signed=False),
        int(body["to_cursor"]).to_bytes(8, "big", signed=False),
    ]
    h = hashlib.sha256()
    h.update(CORPUS_SKIP_DOMAIN)
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def sign_corpus_skip(wallet, request) -> str:
    if bt is None:
        raise ImportError("bittensor is required for sign_corpus_skip")
    return wallet.hotkey.sign(build_corpus_skip_binding(request)).hex()  # type: ignore[union-attr]


def verify_corpus_skip_signature(request) -> bool:
    """False on any failure; fail-closed without bittensor, like a submission."""
    if bt is None:
        logger.debug("verify_corpus_skip_signature: bittensor unavailable")
        return False
    body = _corpus_fields(request)
    try:
        sig_bytes = bytes.fromhex(str(body.get("signature") or ""))
    except ValueError:
        return False
    if not sig_bytes:
        return False
    try:
        keypair = bt.Keypair(ss58_address=str(body["miner_hotkey"]))  # type: ignore[union-attr]
        return bool(keypair.verify(data=build_corpus_skip_binding(request), signature=sig_bytes))
    except Exception as e:
        logger.debug("corpus skip signature verify failed: %s", e)
        return False


# Sandbox session requests (plan 3), each under its own domain: an open never verifies
# as a close, a submission or a skip.
SANDBOX_OPEN_DOMAIN = b"reliquary/sandbox-session-open/v1"
SANDBOX_CLOSE_DOMAIN = b"reliquary/sandbox-session-close/v1"


def _bound(domain: bytes, parts: list[bytes]) -> bytes:
    h = hashlib.sha256()
    h.update(domain)
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def _engagement_bytes(engagement) -> bytes:
    present = {key: value for key, value in dict(engagement).items() if value is not None}
    return json.dumps(present, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def build_sandbox_open_binding(request) -> bytes:
    body = _corpus_fields(request)
    return _bound(SANDBOX_OPEN_DOMAIN, [
        str(body["miner_hotkey"]).encode("utf-8"), str(body["request_id"]).encode("utf-8"),
        int(body["at"]).to_bytes(8, "big", signed=False),
        hashlib.sha256(_engagement_bytes(body["engagement"])).digest()])


def build_sandbox_close_binding(request) -> bytes:
    body = _corpus_fields(request)
    transcript = body.get("transcript")
    return _bound(SANDBOX_CLOSE_DOMAIN, [
        str(body["miner_hotkey"]).encode("utf-8"), str(body["request_id"]).encode("utf-8"),
        int(body["at"]).to_bytes(8, "big", signed=False), str(body["session_id"]).encode("utf-8"),
        str(body["reason"]).encode("utf-8"),
        b"" if transcript is None else transcript_digest(transcript)])


def verify_hotkey_signature(hotkey: str, binding: bytes, signature_hex: str) -> bool:
    """False on any failure; fail-closed without bittensor."""
    if bt is None:
        return False
    try:
        signature = bytes.fromhex(signature_hex or "")
    except ValueError:
        return False
    if not signature:
        return False
    try:
        keypair = bt.Keypair(ss58_address=hotkey)  # type: ignore[union-attr]
        return bool(keypair.verify(data=binding, signature=signature))
    except Exception as e:
        logger.debug("hotkey signature verify failed: %s", type(e).__name__)
        return False


def verify_sandbox_open_signature(request) -> bool:
    body = _corpus_fields(request)
    return verify_hotkey_signature(str(body["miner_hotkey"]), build_sandbox_open_binding(request),
                                   str(body.get("signature") or ""))


def verify_sandbox_close_signature(request) -> bool:
    body = _corpus_fields(request)
    return verify_hotkey_signature(str(body["miner_hotkey"]), build_sandbox_close_binding(request),
                                   str(body.get("signature") or ""))
