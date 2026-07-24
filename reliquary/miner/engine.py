"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    FORCED_SEED_PROTOCOL_VERSION,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    PROMPT_RANGE_SIZE,
)
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.infrastructure import chain
from reliquary.protocol.submission import RolloutSubmission

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


def _initial_runtime_bound_nonce(runtime_fingerprint) -> str:
    """Build a schema-valid placeholder before the submitter signs its attempt.

    ``submit_batch_v2`` replaces this with a fresh signed nonce immediately
    before each precommit. The request model still enforces the runtime binding
    at construction time, so the in-memory placeholder must obey that contract.
    """
    if runtime_fingerprint is None:
        return ""
    from reliquary.shared.runtime_fingerprint import bind_runtime_profile_nonce

    return bind_runtime_profile_nonce(
        os.urandom(16).hex(), runtime_fingerprint.profile_hash,
    )


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
    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    prompt_range: tuple[int, int] | None = None,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    When ``prompt_range`` is given, sampling is confined to ``[lo, hi)`` —
    the per-window slice the validator enforces. The reference miner uses
    uniform-random selection with rejection sampling against the cooldown
    set; more sophisticated strategies are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random
    n = len(env)
    lo, hi = (0, n) if prompt_range is None else prompt_range
    lo = max(0, lo)
    hi = min(n, hi)
    span = hi - lo
    if span <= 0:
        raise RuntimeError("no eligible prompt — empty range")
    cd_in_span = sum(1 for c in cooldown_prompts if lo <= c < hi)
    if cd_in_span < span / 2:
        for _ in range(max_attempts):
            idx = lo + rng.randrange(span)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(lo, hi) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — range fully in cooldown")
    return rng.choice(eligible)


def pick_env_and_prompt(
    envs: dict,
    mix: list[tuple[str, int]],
    cooldown_per_env: dict[str, set[int]],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    randomness: str | None = None,
) -> tuple[str, int]:
    """Sample env per `mix` weights, then a prompt within that env.

    When ``randomness`` is given, each env's prompt is drawn only from that
    window's slice (``window_prompt_range``), matching the validator. Falls
    through to the next env (re-sampling with the chosen env masked) if the
    chosen env's slice is fully in cooldown.
    """
    rng = rng or _random
    names = [n for n, _ in mix]
    weights = [w for _, w in mix]
    if not names:
        raise RuntimeError("pick_env_and_prompt: empty mix")

    available = list(names)
    while available:
        avail_weights = [weights[names.index(n)] for n in available]
        env_name = rng.choices(available, weights=avail_weights)[0]
        env = envs[env_name]
        prompt_range = None
        if randomness:
            env_label = getattr(env, "name", env_name)
            prompt_range = window_prompt_range(
                randomness, env_label, len(env), PROMPT_RANGE_SIZE,
            )
        try:
            idx = pick_prompt_idx(
                env, cooldown_per_env.get(env_name, set()),
                rng=rng, max_attempts=max_attempts, prompt_range=prompt_range,
            )
            return env_name, idx
        except RuntimeError:
            available.remove(env_name)

    raise RuntimeError("pick_env_and_prompt: all envs fully in cooldown")


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
    the validator sees at precommit receipt. Production is zero-tolerance;
    ``submit_batch_v2`` avoids signing inside the final second before a round
    boundary so normal scheduling latency cannot stale an honest precommit.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


def _bft_assemble_rollouts(
    *, model, phase1_tensor, prompt_tokens, think_close_ids, force_ids,
    eos_ids, answer_budget, randomness, hotkey, prompt_idx, checkpoint_hash,
    gen_kwargs=None,
):
    """Budget-Forced Termination assembly.

    Rows that hit EOS are kept as-is (truncated at first EOS). Rows that emitted
    ``</think>`` but did not hit EOS are naturally closed and continue sampling
    the answer for ``answer_budget`` tokens. Rows that did not close thinking are
    *forced*: ``force_ids`` are appended and the same phase-2 generation samples
    the boxed answer. Returns one rollout dict per row with ``forced`` and, for
    forced rows, ``force_span`` = (start, end) of the injected ids within
    ``tokens`` (so the validator carve-out and trainer mask can locate them).

    Phase-2 answer tokens are drawn from the same protocol forced-seed stream as
    phase-1, resuming at each row's own completion offset (its primed length past
    the prompt). The injected ``force_ids`` are not sampled and the validator
    excludes that span from the seed-consistency check.
    """
    import torch

    from reliquary.miner.forced_seed_sampler import (
        ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets,
    )
    from reliquary.shared.modeling import first_eos_index, has_think_close

    plen = len(prompt_tokens)
    n = int(phase1_tensor.shape[0])
    close_set = {int(t) for t in think_close_ids}
    force_ids = [int(t) for t in force_ids]

    out: list = [None] * n
    unfinished_idx: list[int] = []
    unfinished_primed: list[list[int]] = []
    unfinished_force_spans: list[tuple[int, int] | None] = []
    for i in range(n):
        seq = phase1_tensor[i].tolist()
        gen = seq[plen:]
        fe = first_eos_index(gen, eos_ids)
        if fe is not None:
            # Finished on EOS: trim padding/trailing garbage and keep as-is.
            gen = gen[: fe + 1]
            out[i] = {"tokens": prompt_tokens + gen,
                      "prompt_length": plen, "forced": False}
        elif has_think_close(gen, close_set):
            # Naturally closed thinking but did not EOS within phase-1. Continue
            # into the answer phase without injecting FORCE and without a carve.
            unfinished_idx.append(i)
            unfinished_primed.append(seq)
            unfinished_force_spans.append(None)
        else:
            force_start = len(seq)
            primed = seq + force_ids
            unfinished_idx.append(i)
            unfinished_primed.append(primed)
            unfinished_force_spans.append((force_start, force_start + len(force_ids)))

    if unfinished_primed:
        width = max(len(p) for p in unfinished_primed)
        pad = min(eos_ids) if eos_ids else 0
        rows = [[pad] * (width - len(p)) + p for p in unfinished_primed]
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in unfinished_primed]
        device = getattr(model, "device", "cpu")
        proc = ForcedSeedLogitsProcessor(
            randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_indices=list(unfinished_idx),
            base_offsets=phase2_base_offsets(
                [len(p) for p in unfinished_primed], plen,
            ),
            start_len=width,
        )
        ans = model.generate(
            torch.tensor(rows, device=device),
            attention_mask=torch.tensor(mask, device=device),
            max_new_tokens=answer_budget,
            **forced_seed_generate_kwargs(gen_kwargs or {}, proc),
        )
        for k, i in enumerate(unfinished_idx):
            primed = unfinished_primed[k]
            tail = ans[k].tolist()[width:]
            fe = first_eos_index(tail, eos_ids)
            tail = tail[: fe + 1] if fe is not None else tail
            forced_span = unfinished_force_spans[k]
            rollout = {"tokens": primed + tail, "prompt_length": plen,
                       "forced": forced_span is not None}
            if forced_span is not None:
                rollout["force_span"] = forced_span
            out[i] = rollout
    return out


def _rollout_metadata(generation: dict, token_logprobs: list) -> dict:
    """Per-rollout metadata embedded in the GRAIL commit. Carries the BFT
    ``forced`` flag and ``force_span`` so the validator carve-out and trainer
    mask can locate the injected span."""
    prompt_length = int(generation["prompt_length"])
    all_tokens = generation["tokens"]
    force_span = generation.get("force_span")
    return {
        "prompt_length": prompt_length,
        "completion_length": len(all_tokens) - prompt_length,
        "success": True,
        "total_reward": 0.0,
        "advantage": 0.0,
        "token_logprobs": token_logprobs,
        "forced": bool(generation.get("forced", False)),
        "force_span": list(force_span) if force_span else None,
    }


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment | None" = None,
        *,
        envs: "dict[str, Environment] | None" = None,
        mix: "list[tuple[str, int]] | None" = None,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override

        if envs is not None and mix is not None:
            self.envs = envs
            self.mix = mix
            self.env = next(iter(envs.values()))  # legacy fallback
        else:
            assert env is not None, "must pass either env or envs+mix"
            self.envs = {env.name: env}
            self.mix = [(env.name, 1)]
            self.env = env
        self._cooldown_per_env: dict[str, set[int]] = {n: set() for n in self.envs}

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
            get_runtime_contract_v1, get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, RuntimeFingerprint, WindowState,
        )
        from reliquary.shared.runtime_fingerprint import (
            collect_runtime_fingerprint,
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
            runtime_fingerprint = None
            try:
                contract = await get_runtime_contract_v1(url, client=client)
                runtime_fingerprint = RuntimeFingerprint.model_validate(
                    collect_runtime_fingerprint(
                        generation_model=self.vllm_model,
                        proof_model=self.hf_model,
                    )
                )
                logger.info(
                    "validator runtime telemetry enabled version=%d "
                    "validator_profile=%s",
                    contract.telemetry_version,
                    contract.validator_profile.profile_hash,
                )
            except (SubmissionError, ValueError):
                # Older validators may omit the capability or expose an older
                # telemetry schema. Omitting the optional request field keeps
                # mining wire-compatible in both cases.
                runtime_fingerprint = None
                logger.info("validator runtime telemetry unavailable")
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

                # Per-env cooldown: /state's flat ``cooldown_prompts`` covers
                # only the validator's first env, but ``prompt_idx`` is per-env,
                # so query each env for its own set. Fall back to the base set
                # on a fetch error rather than stall the loop.
                for env_name in self._cooldown_per_env:
                    try:
                        env_state = await get_window_state_v2(
                            url, env=env_name, client=client,
                        )
                        self._cooldown_per_env[env_name] = set(
                            env_state.cooldown_prompts
                        )
                    except Exception:
                        self._cooldown_per_env[env_name] = set(
                            state.cooldown_prompts
                        )
                try:
                    env_name, prompt_idx = pick_env_and_prompt(
                        self.envs, self.mix, self._cooldown_per_env, rng=rng,
                        randomness=randomness,
                    )
                except RuntimeError:
                    logger.info("all envs fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue

                env = self.envs[env_name]
                problem = env.get_problem(prompt_idx)
                generations = self._generate_m_rollouts(
                    problem, randomness, env_name=env_name,
                    prompt_idx=prompt_idx, checkpoint_hash=local_hash,
                )
                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d/%d for prompt %d; skipping",
                        len(generations), M_ROLLOUTS, prompt_idx,
                    )
                    continue

                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness, env=env)
                    for gen in generations
                ]
                merkle_root = _compute_merkle_root(rollout_submissions)

                _runtime_fingerprint = runtime_fingerprint
                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=state.window_n,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                    checkpoint_hash=local_hash,
                    runtime_fingerprint=_runtime_fingerprint,
                    nonce=_initial_runtime_bound_nonce(_runtime_fingerprint),
                    # Rollouts are drawn from the forced-seed stream; advertise it.
                    protocol_version=FORCED_SEED_PROTOCOL_VERSION,
                )
                try:
                    resp = await submit_batch_v2(
                        url,
                        request,
                        client=client,
                        wallet=self.wallet,
                        randomness=state.randomness or "",
                        drand_round_fn=_current_drand_round_at_send,
                    )
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

        vllm_model is the fast-generation copy on ``self.vllm_gpu``;
        hf_model is the GRAIL-proof copy on ``self.proof_gpu``. The shared
        loader picks CausalLM for legacy text checkpoints and conditional
        text-only loading for Qwen3.5.
        """
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.shared.modeling import load_text_generation_model

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # 1. Reload hf_model (for GRAIL proofs) on the proof GPU.
        try:
            new_hf = load_text_generation_model(
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
            new_gen = load_text_generation_model(
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

    def _generate_m_rollouts(
        self, problem, randomness, *, env_name: str | None = None,
        prompt_idx: int, checkpoint_hash: str,
    ) -> list[dict]:
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

        from reliquary.constants import (
            BFT_ANSWER_BUDGET,
            BFT_ENABLED,
            BFT_FORCE_ANSWER,
            BFT_THINKING_BUDGET,
        )
        from reliquary.miner.forced_seed_sampler import (
            ForcedSeedLogitsProcessor, forced_seed_generate_kwargs,
        )
        from reliquary.protocol.tokens import encode_prompt
        from reliquary.shared.modeling import (
            first_eos_index,
            force_close_token_ids,
            resolve_eos_token_ids,
            think_close_token_ids,
        )

        hotkey = self.wallet.hotkey.ss58_address
        prompt_tokens = encode_prompt(self.tokenizer, problem["prompt"])
        prompt_length = len(prompt_tokens)
        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = min(eos_ids)
        active_env_name = env_name
        if active_env_name is None:
            active_env_name = getattr(getattr(self, "env", None), "name", None)
        bft_applicable = BFT_ENABLED and (
            active_env_name is None or active_env_name == "openmathinstruct"
        )

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            attention_mask = torch.ones_like(input_tensor)
            # Force phase-1 sampling onto the protocol seed stream: the
            # processor applies the T_PROTO/top_k/top_p warp itself and picks
            # the inverse-CDF token, so HF's own warpers are stripped and
            # do_sample is off (see forced_seed_generate_kwargs). Row r is
            # rollout index r, resuming at completion offset 0.
            base_kwargs = {
                "max_new_tokens": (
                    min(self.max_new_tokens, BFT_THINKING_BUDGET)
                    if bft_applicable else self.max_new_tokens
                ),
                "pad_token_id": pad_token_id,
                "attention_mask": attention_mask,
            }
            if eos_ids:
                base_kwargs["eos_token_id"] = sorted(eos_ids)
            phase1_proc = ForcedSeedLogitsProcessor(
                randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_indices=list(range(M_ROLLOUTS)),
                base_offsets=[0] * M_ROLLOUTS, start_len=prompt_length,
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                **forced_seed_generate_kwargs(base_kwargs, phase1_proc),
            )

            if bft_applicable and BFT_FORCE_ANSWER:
                # Phase-2 answer generation continues on the same forced stream
                # (identity threaded so it resumes at each row's offset). Skipped
                # under clean-cap (BFT_FORCE_ANSWER=False): a rollout that did not
                # close </think> within the phase-1 budget is left truncated and
                # grades as bad_termination instead of being force-answered.
                phase2_kwargs = {"pad_token_id": pad_token_id}
                if eos_ids:
                    phase2_kwargs["eos_token_id"] = sorted(eos_ids)
                return _bft_assemble_rollouts(
                    model=self.vllm_model,
                    phase1_tensor=outputs,
                    prompt_tokens=prompt_tokens,
                    think_close_ids=set(think_close_token_ids(self.tokenizer)),
                    force_ids=force_close_token_ids(self.tokenizer),
                    eos_ids=eos_ids,
                    answer_budget=BFT_ANSWER_BUDGET,
                    randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                    gen_kwargs=phase2_kwargs,
                )
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            first_eos = first_eos_index(gen, eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
                "forced": False,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness, *, env=None):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        active_env = env if env is not None else self.env
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        if getattr(active_env, "validator_authoritative_reward", False):
            reward = 0.0
        else:
            reward = active_env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
            env_name=active_env.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
            "rollout": _rollout_metadata(generation, token_logprobs),
        }
