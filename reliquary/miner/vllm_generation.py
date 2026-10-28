"""Generate the protocol's forced draw on vLLM instead of transformers.

The draw is public: token t of rollout r is the profile's pick for ``u_at`` (the
inverse CDF over the warped policy, or Gumbel-max over the raw logits; see
``forced_sampling``), and the validator teacher-forces the same pick from its own
forward. Any engine may produce it, so long as it reads the same logits — which
is why this processor computes warp + pick itself and hands vLLM a one-hot,
with the request sampled greedily so it can select nothing else.

Measured on an H200 (20-09, Teutonic-I, code prompts): sequences generated here
and verified by the transformers validator agree on 0.964-0.980 of stochastic
positions per group, worst rollout 0.941, no group or rollout under its floor
and no terminal pick refused — level with the batched transformers path it
replaces (0.970-0.974). Throughput at 256 concurrent rollouts is 6,050 tok/s
against 507 for a 16-wide transformers batch, because a transformers decode step
costs ~31 ms whatever the batch, so the win is concurrency, not the engine.

The engine draws with ``RunnerForcedSeedVLLMProcessor``: the position of each
pick is read from the model runner's input batch at sampling time, keyed by
request id. ``ForcedSeedVLLMProcessor`` counted its own calls per slot, and
vLLM's recompute preemption re-adds a request with its output so far, which
restarted that counter: measured 12-15 % of honest rollouts drawn with the wrong
u from there on (H100) and 30 % on an RTX 5090. The runner-keyed draw held at 0
bad positions over millions, with hundreds of preemptions, on 3090/A6000/5090
(vLLM 0.30.0, async scheduling on; offpolicy-lab tracecheck, 14-10).
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from typing import Any

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import host_to_device, pick_rows, u_at
from reliquary.shared.modeling import first_eos_index

logger = logging.getLogger(__name__)

try:  # vLLM is an optional miner dependency; the class must stay importable.
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except ImportError:  # pragma: no cover - exercised on hosts without vLLM
    BatchUpdate = Any  # type: ignore[assignment,misc]
    LogitsProcessor = object  # type: ignore[assignment,misc]

FORCED_SEED_KEY = "forced_seed"

# The runner internals the forced draw reads were checked on these vLLM series
# only (see ``require_runner_forced_draw``); a new series must be re-measured.
SUPPORTED_VLLM_SERIES = ("0.30",)


class ForcedSeedVLLMProcessor(LogitsProcessor):  # type: ignore[misc,valid-type]
    """One-hot the forced pick for every running request, in one batched pass.

    vLLM addresses requests by their slot in the persistent batch, and reuses a
    slot as soon as a request leaves it, so the per-slot state follows the
    engine's own removed/added/moved order rather than a request id.

    The position is a per-slot call counter. That is exact when each request is
    sampled once per token and never re-added, which holds for the in-process
    fakes the tests drive, but not on an engine: a preempted request comes back
    with its counter at ``base_offset``. Engines use
    ``RunnerForcedSeedVLLMProcessor``.
    """

    def __init__(self, vllm_config: Any = None, device: Any = None,
                 is_pin_memory: bool = False) -> None:
        self._slots: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        """The pick is not the argmax: this processor decides the token."""
        return False

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if batch_update is None:
            return
        for index in getattr(batch_update, "removed", ()):
            self._slots.pop(_slot_index(index), None)
        for added in getattr(batch_update, "added", ()):
            index, params = added[0], added[1]
            slot = _slot_index(index)
            self._slots.pop(slot, None)
            forced = dict(getattr(params, "extra_args", None) or {}).get(FORCED_SEED_KEY)
            if forced:
                self._slots[slot] = _forced_state(forced)
        for moved in getattr(batch_update, "moved", ()):
            source, destination = _slot_index(moved[0]), _slot_index(moved[1])
            swap = "SWAP" in str(moved[2]).upper() if len(moved) > 2 else False
            from_state = self._slots.pop(source, None)
            to_state = self._slots.pop(destination, None)
            if from_state is not None:
                self._slots[destination] = from_state
            if swap and to_state is not None:
                self._slots[source] = to_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        slots = [slot for slot in sorted(self._slots) if slot < logits.shape[0]]
        if not slots:
            return logits
        forced = _one_hot_picks(
            logits, slots,
            [_uniform(self._slots[slot], self._slots[slot]["offset"]) for slot in slots],
            kinds=[self._slots[slot].get("draw") for slot in slots],
        )
        for slot in slots:
            self._slots[slot]["offset"] += 1
        return forced


def _forced_state(forced: dict[str, Any]) -> dict[str, Any]:
    """A request's place in the draw, checked once: a public pool must be the
    request's own and its seed inside the pool."""
    state = {
        "randomness": str(forced["randomness"]),
        "prompt_idx": int(forced["prompt_idx"]),
        "checkpoint_hash": str(forced["checkpoint_hash"]),
        "rollout_index": int(forced["rollout_index"]),
        "offset": int(forced.get("base_offset", 0)),
    }
    # An eval job's draw (``corpus.eval_draw``) names its own stream domain and draw, whatever the profile's.
    if forced.get("domain") is not None:
        state["domain"] = str(forced["domain"])
    if forced.get("draw") is not None:
        from reliquary.environment.forced_sampling import check_draw

        check_draw(str(forced["draw"]))     # a request-named draw samples at T=1 over the full vocabulary
        state["draw"] = str(forced["draw"])
    if "public_pool" in forced:
        from reliquary.protocol.seed_pool import SeedPool
        pool = SeedPool.from_dict(forced["public_pool"])
        seed_index = forced.get("seed_index")
        pool.uniform(seed_index, 0)  # refuses a seed outside the pool
        if pool.prompt_idx != state["prompt_idx"] or pool.checkpoint_hash != state["checkpoint_hash"]:
            raise ValueError("vLLM request context differs from public pool")
        state.update(seed_pool=pool, seed_index=seed_index)
    elif "seed_index" in forced:
        raise ValueError("seed_index requires public pool")
    return state


def _uniform(state: dict[str, Any], position: int) -> float:
    if "seed_pool" in state:
        return state["seed_pool"].uniform(state["seed_index"], position)
    return u_at(state["randomness"], state["prompt_idx"], state["checkpoint_hash"],
                state["rollout_index"], position, domain=state.get("domain"))


def _row_picks(logits: torch.Tensor, draws: list[float], kinds: list[str | None] | None) -> torch.Tensor:
    """The forced pick of each row: the profile's draw, or the one its request names (T=1, full vocabulary)."""
    if kinds is None or all(kind is None for kind in kinds):
        return pick_rows(logits, draws, draw=default_draw(), t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    picks = torch.empty(len(draws), dtype=torch.int64, device=logits.device)
    for kind in set(kinds):
        at = [i for i, k in enumerate(kinds) if k == kind]
        index = host_to_device(at, torch.int64, logits.device)
        picks[index] = (_row_picks(logits[index], [draws[i] for i in at], None) if kind is None
                        else pick_rows(logits[index], [draws[i] for i in at], draw=kind))
    return picks


def _one_hot_picks(logits: torch.Tensor, rows: list[int], draws: list[float], *,
                   keep_other_rows: bool = False, kinds: list[str | None] | None = None) -> torch.Tensor:
    """Each forced row becomes -inf but for the profile's forced pick (the shared
    ``forced_sampling.pick_rows``), or the draw its request names (``kinds``). The other
    rows are kept as they are, or blanked as the slot-counter processor always did."""
    index = host_to_device(rows, torch.int64, logits.device)
    picks = _row_picks(logits[index], draws, kinds)
    # index_fill_ takes its value as a kernel argument; ``forced[index] = v`` copies v to the GPU
    # from pageable memory, which blocks the host on the stream every decode step.
    if keep_other_rows:
        forced = logits.clone(memory_format=torch.contiguous_format)
        forced.index_fill_(0, index, float("-inf"))
    else:
        forced = torch.full_like(logits, float("-inf"), memory_format=torch.contiguous_format)
    forced.view(-1).index_fill_(0, index * forced.shape[-1] + picks, 0.0)
    return forced


def default_draw() -> str:
    """The active profile's forced draw (``sampling.draw``), read at call time. The
    engine process resolves the same profile as the miner that started it."""
    from reliquary import constants

    return constants.FORCED_DRAW


def default_lm_head_fp32() -> bool:
    from reliquary import constants

    return bool(constants.LM_HEAD_FP32)


# ``sampling.lm_head_fp32``: vLLM >= 0.30 runs the LM head in fp32 with this override,
# torch.mm(bf16, bf16, out_dtype=fp32), the validator's fp32(h) @ fp32(W).
FP32_HEAD_OVERRIDES = {"head_dtype": "float32"}


def _slot_index(value: Any) -> int:
    return int(value[0] if isinstance(value, (tuple, list)) else value)


def forced_seed_extra_args(
    *, randomness: str, prompt_idx: int, checkpoint_hash: str,
    rollout_index: int, base_offset: int = 0, seed_pool=None,
    seed_index: int | None = None, domain: str | None = None, draw: str | None = None,
) -> dict[str, Any]:
    """What one request tells the processor about its place in the draw.

    With a public seed pool the draw is the one of ``seed_index`` and does not
    depend on ``rollout_index``.
    """
    result = {
        FORCED_SEED_KEY: {
            "randomness": randomness,
            "prompt_idx": int(prompt_idx),
            "checkpoint_hash": checkpoint_hash,
            "rollout_index": int(rollout_index),
            "base_offset": int(base_offset),
            # An eval job's own stream and draw (``corpus.eval_draw``); absent, the profile's.
            **({"domain": str(domain)} if domain is not None else {}),
            **({"draw": str(draw)} if draw is not None else {}),
        }
    }
    if seed_pool is not None:
        seed_pool.uniform(seed_index, 0)  # refuses a seed outside the pool
        result[FORCED_SEED_KEY].update(public_pool=seed_pool.to_dict(), seed_index=seed_index)
    elif seed_index is not None:
        raise ValueError("seed_index requires public pool")
    return result


# ---------------------------------------------------------------- runner-keyed draw
#
# vLLM 0.30's V1 GPUModelRunner samples in ``_sample(logits, spec_decode_metadata)``
# with logits rows in ``input_batch.req_ids`` order (non-speculative). At that
# moment ``input_batch.num_tokens_no_spec[row]`` counts the request's prompt plus
# every token sampled for it so far, the async-scheduling placeholder of the step
# still in flight included, and is reset to prompt + output when a preempted
# request is re-added. So, per row,
#
#     position = num_tokens_no_spec[row] - num_prompt_tokens[row] + base_offset
#
# is the completion position the pick lands at, whatever happened to the slot, and
# a sample vLLM throws away (a partial chunked-prefill step) does not move it. The
# wrapped ``_sample`` publishes those rows for the step; the processor, which vLLM
# calls inside the sampler after the raw logprobs are taken, one-hots them.

_STEP = threading.local()  # .rows: [(row, request id, forced args, position)] of the step being sampled
_PATCH_MARK = "_reliquary_forced_draw"


def _runner_class() -> Any:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # local import: optional dependency

    return GPUModelRunner


def _step_rows(runner: Any, n_rows: int) -> list[tuple[int, str, dict[str, Any], int]]:
    batch = runner.input_batch
    try:
        req_ids = batch.req_ids
        token_counts = batch.num_tokens_no_spec
        prompt_counts = batch.num_prompt_tokens
    except AttributeError as exc:
        raise RuntimeError(
            f"forced draw: the vLLM input batch has no {exc.name}; "
            f"the runner internals differ from vLLM {SUPPORTED_VLLM_SERIES}"
        ) from exc
    rows = []
    for row in range(min(n_rows, len(req_ids))):
        req_id = req_ids[row]
        state = runner.requests.get(req_id) if req_id is not None else None
        params = getattr(state, "sampling_params", None)
        forced = dict(getattr(params, "extra_args", None) or {}).get(FORCED_SEED_KEY)
        if not forced:
            continue
        position = int(token_counts[row]) - int(prompt_counts[row])
        if position < 0:
            raise RuntimeError(f"forced draw: request {req_id} at completion position {position}")
        rows.append((row, req_id, forced, position))
    return rows


def install_runner_forced_draw(runner_cls: Any = None) -> None:
    """Wrap ``runner_cls._sample`` (vLLM's GPUModelRunner by default) so each step
    publishes its forced rows. Idempotent, and in the process that runs the model:
    the processor installs it from its constructor, which vLLM calls there. It
    touches neither ``execute_model`` nor ``_model_forward``, so it composes with
    ``vllm_hidden_capture.capture_hidden_states``."""
    if runner_cls is None:
        runner_cls = _runner_class()
    if getattr(runner_cls._sample, _PATCH_MARK, False):
        return
    original = runner_cls._sample

    def _sample(self, logits, spec_decode_metadata):
        rows = _step_rows(self, logits.shape[0]) if logits is not None else []
        if rows and spec_decode_metadata is not None:
            # The rejection sampler ignores custom logits processors.
            raise RuntimeError("forced draw does not support speculative decoding")
        previous = getattr(_STEP, "rows", None)
        _STEP.rows = rows
        try:
            return original(self, logits, spec_decode_metadata)
        finally:
            _STEP.rows = previous

    setattr(_sample, _PATCH_MARK, True)
    runner_cls._sample = _sample


def require_runner_forced_draw() -> None:
    """Refuse, before any weights load, a vLLM whose runner the draw was not
    checked on: a silent mismatch would draw honest rollouts off the stream."""
    import vllm  # local import: optional dependency

    version = str(getattr(vllm, "__version__", ""))
    if ".".join(version.split(".")[:2]) not in SUPPORTED_VLLM_SERIES:
        raise RuntimeError(
            f"forced draw: vLLM {version or '?'} is not one of {SUPPORTED_VLLM_SERIES}; "
            "its runner internals must be re-checked (tracecheck) before mining on it"
        )
    sample = getattr(_runner_class(), "_sample", None)
    params = list(inspect.signature(sample).parameters) if sample is not None else []
    if params[:3] != ["self", "logits", "spec_decode_metadata"]:
        raise RuntimeError(
            f"forced draw: GPUModelRunner._sample{tuple(params[1:])} is not "
            "(logits, spec_decode_metadata)"
        )


class RunnerForcedSeedVLLMProcessor(ForcedSeedVLLMProcessor):
    """The forced pick at the position the model runner reports (see above).

    The slot bookkeeping is kept for two things only: refusing a malformed draw
    when the request is added, and noticing a forced request sampled outside the
    patched ``_sample``, which would otherwise fall to a greedy argmax.
    """

    def __init__(self, vllm_config: Any = None, device: Any = None,
                 is_pin_memory: bool = False) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise RuntimeError("forced draw does not support speculative decoding")
        install_runner_forced_draw()
        self._states: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        rows = getattr(_STEP, "rows", None)
        if rows is None:
            if any(slot < logits.shape[0] for slot in self._slots):
                raise RuntimeError(
                    "forced requests sampled outside the patched vLLM runner: "
                    "the draw position is unknown"
                )
            return logits
        if not rows:
            return logits
        states = {}
        for _, req_id, forced, _ in rows:
            cached = self._states.get(req_id)
            states[req_id] = (cached if cached is not None and cached[0] is forced
                              else (forced, _forced_state(forced)))
        self._states = states  # requests not sampled this step are parsed again if they return
        draws = [_uniform(states[req_id][1], states[req_id][1]["offset"] + position)
                 for _, req_id, _, position in rows]
        return _one_hot_picks(logits, [row for row, *_ in rows], draws, keep_other_rows=True,
                              kinds=[states[req_id][1].get("draw") for _, req_id, _, _ in rows])


class VLLMRolloutGenerator:
    """A group of rollouts for one prompt, generated concurrently on vLLM.

    The engine is built once and reused; a checkpoint change rebuilds it, since
    vLLM holds its weights for the process's life.
    """

    def __init__(
        self, model_path: str, *, revision: str | None = None,
        max_model_len: int | None = None, gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256, enforce_eager: bool = False,
        engine: Any | None = None, sampling_params_class: Any | None = None,
        lm_head_fp32: bool | None = None, quantization: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self._settings = {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
        }
        if quantization:
            # vLLM's own weight format (e.g. "fp8" on Ada/Hopper): the miner's choice, the draw is unchanged.
            self._settings["quantization"] = quantization
        if default_lm_head_fp32() if lm_head_fp32 is None else lm_head_fp32:
            self._settings["hf_overrides"] = dict(FP32_HEAD_OVERRIDES)
        self._sampling_params_class = sampling_params_class
        self._pending: dict[str, tuple[str, int, set[int]]] = {}    # request id -> (group key, rollout, stop ids)
        self._groups: dict[str, list[list[int] | None]] = {}        # group key -> its completions so far
        self._llm = engine if engine is not None else self._build(model_path, revision)

    def _build(self, model_path: str, revision: str | None) -> Any:
        # The forced draw hooks the V1 runner; the V2 runner would bypass it.
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        # mine-async's forked engine core died with "Cannot re-initialize CUDA in forked subprocess" (A100,
        # 2026-10-16, 4 of 4 starts); spawned, it runs.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if os.environ["VLLM_USE_V2_MODEL_RUNNER"].strip().lower() not in ("0", "false"):
            raise RuntimeError("forced draw is hooked on the V1 runner: set VLLM_USE_V2_MODEL_RUNNER=0")
        require_runner_forced_draw()
        from vllm import LLM  # local import: optional dependency

        from reliquary.miner.engine_reaper import child_pids, reap_engine_children_on_exit

        before = set(child_pids())
        engine = LLM(
            model=model_path, revision=revision, dtype="bfloat16",
            logits_processors=[RunnerForcedSeedVLLMProcessor], **self._settings,
        )
        # The spawned engine core holds the GPU: it must not outlive a miner killed with SIGKILL.
        reap_engine_children_on_exit(before)
        logger.info(
            "vLLM generation backend ready (%s%s)", model_path,
            f"@{revision}" if revision else "",
        )
        return engine

    def reload(self, model_path: str, revision: str | None = None) -> None:
        """Point the generator at new weights.

        vLLM holds its weights for the engine's life, so this rebuilds it. That
        costs a model load per checkpoint, which is why the miner pulls a
        checkpoint between windows rather than during one.
        """
        del self._llm
        self.model_path, self.revision = model_path, revision
        self._llm = self._build(model_path, revision)

    def generate(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int], seed_pool=None, seeds=None,
    ) -> list[list[int]]:
        """Completion token ids per rollout, truncated at their first stop token.

        vLLM keeps the token that stopped a request (measured 20-09), which the
        protocol needs: the terminal pick is checked on it. The truncation is the
        transformers path's, so a stop token appearing mid-batch cannot carry
        padding downstream either way.
        """
        if seed_pool is not None:
            # Rollout ``index`` is drawn from pool seed ``seeds[index]``.
            selection = seed_pool.selection(seeds)
            seed_pool.validate_selection(selection, rollout_count=rollouts)
            seeds = selection.seeds
        elif seeds is not None:
            raise ValueError("seeds require public pool")
        extra = [
            forced_seed_extra_args(
                randomness=randomness, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash, rollout_index=index,
                seed_pool=seed_pool,
                seed_index=seeds[index] if seed_pool is not None else None,
            )
            for index in range(rollouts)
        ]
        return self.generate_batch([(list(prompt_tokens), extra, int(max_new_tokens))], eos_ids=eos_ids)[0]

    def generate_batch(
        self, groups: list[tuple[list[int], list[dict[str, Any]], int]], *, eos_ids: list[int],
    ) -> list[list[list[int]]]:
        """Many groups in ONE engine call (the async miner's batch of leases): ``groups`` is
        ``[(prompt tokens, per-rollout extra_args, max new tokens)]``; the answer is, per group, its
        rollouts' completions (``generate``'s truncation).

        The draw stays exact per request: each request carries its own ``extra_args``
        (its lease's randomness, prompt, checkpoint tag and rollout index), which the
        runner-keyed processor reads by request id at its runner-reported position,
        whatever else shares the batch. vLLM answers in request order.
        """
        sampling_params = self._sampling_params_class
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params

        stop_ids = sorted(int(token) for token in eos_ids)
        requests, sampling, sizes = [], [], []
        for prompt_tokens, extra_args, max_new_tokens in groups:
            sizes.append(len(extra_args))
            for extra in extra_args:
                requests.append({"prompt_token_ids": list(prompt_tokens)})
                sampling.append(sampling_params(
                    temperature=0.0, max_tokens=int(max_new_tokens), detokenize=False,
                    stop_token_ids=stop_ids or None, extra_args=extra,
                ))
        outputs = self._llm.generate(requests, sampling, use_tqdm=False) if requests else []
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM answered {len(outputs)} of {len(requests)} requests")
        completions: list[list[int]] = []
        for output in outputs:
            completion = list(output.outputs[0].token_ids)
            end = first_eos_index(completion, set(stop_ids))
            if end is not None:
                completion = completion[: end + 1]
            completions.append(completion)
        answer, start = [], 0
        for size in sizes:
            answer.append(completions[start:start + size])
            start += size
        return answer

    # --- continuous batching: groups join the running engine, each returns when its last rollout ends -------
    def add_group(self, key: str, prompt_tokens: list[int], extra_args: list[dict[str, Any]],
                  max_new_tokens: int, *, eos_ids: list[int]) -> None:
        """Queue one group's requests in the running engine (``step_groups`` drives it). ``key`` names the group
        in what ``step_groups`` returns; the draw is per request, as in ``generate_batch``."""
        sampling_params = self._sampling_params_class
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params
        from vllm.sampling_params import RequestOutputKind

        pending, groups = self._pending, self._groups
        if key in groups:
            raise ValueError(f"group {key} is already running")
        stop_ids = sorted(int(token) for token in eos_ids)
        groups[key] = [None] * len(extra_args)
        for index, extra in enumerate(extra_args):
            params = sampling_params(temperature=0.0, max_tokens=int(max_new_tokens), detokenize=False,
                                     stop_token_ids=stop_ids or None, extra_args=extra)
            params.output_kind = RequestOutputKind.FINAL_ONLY
            request_id = f"rl-{key}-{index}"
            self._llm.llm_engine.add_request(request_id, {"prompt_token_ids": list(prompt_tokens)}, params)
            pending[request_id] = (key, index, set(stop_ids))

    def step_groups(self, budget_s: float) -> list[tuple[str, list[list[int]]]]:
        """Step the engine for about ``budget_s`` seconds, or until a group completes; the completed groups as
        ``(key, completions)`` (``generate``'s truncation)."""
        pending, groups = self._pending, self._groups
        engine, done = self._llm.llm_engine, []
        deadline = time.monotonic() + float(budget_s)
        while pending and engine.has_unfinished_requests():
            for output in engine.step():
                entry = pending.pop(output.request_id, None) if output.finished else None
                if entry is None:
                    continue
                key, index, stops = entry
                completion = list(output.outputs[0].token_ids)
                end = first_eos_index(completion, stops)
                groups[key][index] = completion if end is None else completion[: end + 1]
                if all(c is not None for c in groups[key]):
                    done.append((key, groups.pop(key)))
            if done or time.monotonic() >= deadline:
                break
        return done

    def has_running_groups(self) -> bool:
        return bool(self._pending)

    def abort_groups(self) -> None:
        """Drop every running group (after a failed step)."""
        if self._pending:
            self._llm.llm_engine.abort_request(list(self._pending))
        self._pending.clear()
        self._groups.clear()
