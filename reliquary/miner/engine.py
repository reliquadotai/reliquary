"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load.

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update is
    needed (remote ≤ local, or remote has no repo/revision yet), returns
    inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_repo_id is None or state.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path."""
    import asyncio
    from huggingface_hub import snapshot_download

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json", "tokenizer*"],
    )


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    The reference miner uses uniform-random selection with rejection
    sampling against the cooldown set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    rng = rng or _random
    n = len(env)
    if len(cooldown_prompts) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches what
    the validator sees at receipt (modulo the 1-round tolerance). Uses
    chain params cached at process start; one drand period of skew is
    tolerated by the validator.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.1: poll state, pull checkpoint on n-change, submit when OPEN.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation
        (asyncio.CancelledError) or if env becomes fully cooldown'd.
        """
        import httpx
        import random

        from reliquary.constants import M_ROLLOUTS, POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # v2.3: randomness is fetched per-window from /state instead of
        # recomputed locally. The validator aligns window OPEN to a drand
        # boundary and binds randomness to the round publishing at that
        # boundary — a value that didn't exist a few seconds earlier, so
        # nothing to pre-fetch. The miner just reads what /state reports.
        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except SubmissionError:
                    # /state may return 503 between windows; wait briefly.
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pull new checkpoint if needed (works at any state).
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue

                # v2.3: trust the validator's per-window randomness rather
                # than recomputing locally. Empty string means the validator
                # hasn't yet finished _set_window_randomness — wait briefly.
                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(0.1)
                    continue

                # Pick prompt, generate, submit.
                cooldown_set = set(state.cooldown_prompts)
                try:
                    prompt_idx = pick_prompt_idx(self.env, cooldown_set, rng=rng)
                except RuntimeError:
                    logger.info("env fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue

                get_public_problem = getattr(self.env, "get_public_problem", None)
                problem = (
                    get_public_problem(prompt_idx)
                    if callable(get_public_problem)
                    else self.env.get_problem(prompt_idx)
                )
                generations = self._generate_m_rollouts(problem, randomness)
                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d/%d for prompt %d; skipping",
                        len(generations), M_ROLLOUTS, prompt_idx,
                    )
                    continue

                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness)
                    for gen in generations
                ]
                merkle_root = _compute_merkle_root(rollout_submissions)

                # v2.3 design A': fetch the drand round just before the POST.
                # The attached round determines the submission's chronological
                # slot at seal time. Miss this and the validator rejects with
                # STALE_ROUND or FUTURE_ROUND.
                current_round = _current_drand_round_at_send()
                # Envelope signature (introduced 2026-05 to close the
                # /submit hotkey-spoof DoS). Sign the canonical binding
                # over every field the validator routes on, including
                # the validator-published window randomness — that ties
                # the signature to this exact validator's view of the
                # window so a captured signature cannot be replayed
                # against a different validator instance or a future
                # window. The nonce is per-submission fresh randomness.
                import os as _os
                _nonce = _os.urandom(16).hex()
                _envelope_sig = sign_envelope(
                    wallet=self.wallet,
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    window_start=state.window_n,
                    prompt_idx=prompt_idx,
                    merkle_root=merkle_root,
                    checkpoint_hash=local_hash,
                    drand_round=current_round,
                    randomness=state.randomness or "",
                    nonce=_nonce,
                ).hex()
                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=state.window_n,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                    checkpoint_hash=local_hash,
                    drand_round=current_round,
                    nonce=_nonce,
                    envelope_signature=_envelope_sig,
                )
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    logger.info(
                        "submitted window=%d prompt=%d accepted=%s reason=%s",
                        state.window_n, prompt_idx, resp.accepted,
                        resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
                    )
                    results.append(resp)
                except SubmissionError as exc:
                    logger.error("submit failed: %s", exc)

        return results

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        Both attributes are ``AutoModelForCausalLM`` instances despite the
        historical ``vllm_model`` naming — vllm_model is the fast-generation
        copy on ``self.vllm_gpu``, hf_model is the GRAIL-proof copy on
        ``self.proof_gpu``.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # 1. Reload hf_model (for GRAIL proofs) on the proof GPU.
        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 2. Reload vllm_model on the generation GPU.
        try:
            new_gen = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload vllm_model from %s; miner generation is "
                "BROKEN until the next successful pull. hf_model was swapped "
                "so GRAIL proofs will be inconsistent.",
                local_path,
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO in one batched call.

        One .generate() with batch shape (M_ROLLOUTS, prompt_len) is ~5-7×
        faster on GPU than M_ROLLOUTS serial calls — the matmul tiling
        utilizes far more of the GPU's compute. Each row samples
        independently (do_sample=True), so GRPO-group semantics are
        preserved. Each output row is truncated at its first post-prompt
        EOS so trailing batch-padding (which HF pads with pad_token_id =
        eos_token_id) is not carried downstream — otherwise the validator's
        GRAIL forward pass would see extra EOS tokens the miner didn't
        "generate" in the usual sense.
        """
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos = self.tokenizer.eos_token_id
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            try:
                first_eos = gen.index(eos)
                gen = gen[: first_eos + 1]
            except ValueError:
                pass
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission with validator-authoritative reward."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=0.0,
            commit=commit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from the drand beacon (v2.3+: drand-only).

        Matches the validator's ``service._derive_randomness``: block_hash is
        no longer mixed in, so the miner does not need a substrate roundtrip
        for the GRAIL seed. The legacy ``use_drand=False`` path remains for
        offline tests and uses block_hash as a single-source seed.
        """
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"]
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
        return chain.compute_window_randomness(block_hash)

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # HF forward pass on proof GPU
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # fp32 log_softmax to match the validator and reduce tail-token drift.
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Sign
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }
